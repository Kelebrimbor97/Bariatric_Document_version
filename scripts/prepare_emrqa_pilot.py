#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: reportlab\n"
        "Install it with:\n"
        "  pip install reportlab"
    ) from exc


QUESTION_ALIASES = ("question", "query", "q")
ANSWER_ALIASES = (
    "required_answer_terms",
    "answer_terms",
    "answer",
    "answers",
    "answer_text",
    "gold_answer",
)
TEXT_ALIASES = (
    "note_text",
    "context",
    "evidence",
    "passage",
    "document",
    "text",
    "source_text",
)
PATIENT_ID_ALIASES = ("patient_id", "subject_id", "hadm_id", "note_id", "id")
DOCUMENT_TYPE_ALIASES = ("document_type", "note_type", "category")
FORBIDDEN_ALIASES = ("forbidden_answer_terms", "forbidden_terms")
ANY_TERMS_ALIASES = ("required_any_terms", "answer_any_terms")

DOC_TYPE_PATHS = {
    "clinic_note": "Clinical Documents/Outpatient Core/clinic_note_{idx:04d}.pdf",
    "progress_note": "Clinical Documents/Outpatient Core/progress_note_{idx:04d}.pdf",
    "history_and_physical": "Clinical Documents/Outpatient Core/history_and_physical_{idx:04d}.pdf",
    "discharge_summary": "Clinical Documents/Inpatient Core/discharge_summary_{idx:04d}.pdf",
    "lab_report": "Laboratory Documents/lab_results_{idx:04d}.pdf",
    "radiology": "Radiology/radiology_report_{idx:04d}.pdf",
    "medication_list": "Clinical Documents/Outpatient Core/medication_list_{idx:04d}.pdf",
    "nutrition_note": "Clinical Documents/Outpatient Core/nutrition_note_{idx:04d}.pdf",
    "operative_report": "Perioperative Documents/operative_report_{idx:04d}.pdf",
}


@dataclass
class PilotExample:
    patient_id: str
    question: str
    note_text: str
    required_answer_terms: list[str]
    required_any_terms: list[list[str]]
    forbidden_answer_terms: list[str]
    document_type: str
    source_id: str


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def safe_id(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # Some datasets use a top-level {"data": [...]} wrapper.
            if isinstance(obj.get("data"), list):
                return [x for x in obj["data"] if isinstance(x, dict)]
            return [obj]
    except json.JSONDecodeError:
        pass

    records: list[dict[str, Any]] = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL in {path} at line {line_num}: {exc}") from exc
        if isinstance(obj, dict):
            records.append(obj)
    return records


def first_present(record: dict[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        if alias in record and record[alias] not in (None, ""):
            return record[alias]
    return None


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [coerce_text(item) for item in value]
        return "\n\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "evidence", "context", "note_text", "value", "answer"):
            if key in value:
                return coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = normalize_spaces(value)
        return [text] if text else []
    if isinstance(value, dict):
        text = coerce_text(value)
        return [normalize_spaces(text)] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(coerce_string_list(item))
        # Preserve order while deduplicating.
        seen = set()
        deduped = []
        for item in out:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped
    text = normalize_spaces(str(value))
    return [text] if text else []


def coerce_any_terms(value: Any) -> list[list[str]]:
    if not value:
        return []
    if not isinstance(value, list):
        return []
    groups: list[list[str]] = []
    for group in value:
        terms = coerce_string_list(group)
        if terms:
            groups.append(terms)
    return groups


def expand_records(records: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Expand common QA shapes into flat records.

    Supports already-flat JSONL plus simple SQuAD-like records with paragraphs/qas.
    This is intentionally permissive for a tiny pilot, not a full emrQA parser.
    """
    for record in records:
        # Top-level SQuAD-like item: {title, paragraphs: [{context, qas: [...]}]}
        paragraphs = record.get("paragraphs")
        if isinstance(paragraphs, list):
            for p_idx, paragraph in enumerate(paragraphs, start=1):
                if not isinstance(paragraph, dict):
                    continue
                context = first_present(paragraph, TEXT_ALIASES) or first_present(record, TEXT_ALIASES)
                qas = paragraph.get("qas")
                if isinstance(qas, list):
                    for qa_idx, qa in enumerate(qas, start=1):
                        if not isinstance(qa, dict):
                            continue
                        merged = dict(record)
                        merged.update(paragraph)
                        merged.update(qa)
                        merged["context"] = context
                        merged.setdefault("id", f"{record.get('title', 'record')}_{p_idx}_{qa_idx}")
                        yield merged
                else:
                    yield paragraph
            continue

        # Paragraph-like item: {context, qas: [...]}
        qas = record.get("qas")
        if isinstance(qas, list):
            context = first_present(record, TEXT_ALIASES)
            for qa_idx, qa in enumerate(qas, start=1):
                if not isinstance(qa, dict):
                    continue
                merged = dict(record)
                merged.update(qa)
                merged["context"] = context
                merged.setdefault("id", f"{record.get('id', 'record')}_{qa_idx}")
                yield merged
            continue

        yield record


def make_answer_terms(record: dict[str, Any], max_terms: int) -> list[str]:
    explicit = first_present(record, ("required_answer_terms", "answer_terms"))
    if explicit is not None:
        return coerce_string_list(explicit)[:max_terms]

    answers = first_present(record, ANSWER_ALIASES)
    return coerce_string_list(answers)[:max_terms]


def make_examples(
    raw_records: list[dict[str, Any]],
    default_document_type: str,
    max_examples: int | None,
    max_answer_terms: int,
) -> list[PilotExample]:
    examples: list[PilotExample] = []

    for idx, record in enumerate(expand_records(raw_records), start=1):
        question = coerce_text(first_present(record, QUESTION_ALIASES))
        note_text = coerce_text(first_present(record, TEXT_ALIASES))
        required_answer_terms = make_answer_terms(record, max_answer_terms=max_answer_terms)

        if not question or not note_text or not required_answer_terms:
            continue

        raw_patient_id = first_present(record, PATIENT_ID_ALIASES)
        patient_id = safe_id(raw_patient_id, fallback=f"EMRQA{idx:05d}")
        source_id = safe_id(first_present(record, ("id", "source_id", "note_id")), fallback=f"example_{idx:05d}")
        document_type = coerce_text(first_present(record, DOCUMENT_TYPE_ALIASES)) or default_document_type
        document_type = document_type.strip() or default_document_type

        examples.append(
            PilotExample(
                patient_id=patient_id,
                question=question,
                note_text=note_text,
                required_answer_terms=required_answer_terms,
                required_any_terms=coerce_any_terms(first_present(record, ANY_TERMS_ALIASES)),
                forbidden_answer_terms=coerce_string_list(first_present(record, FORBIDDEN_ALIASES)),
                document_type=document_type,
                source_id=source_id,
            )
        )

        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples


def doc_relative_path(document_type: str, idx: int) -> Path:
    template = DOC_TYPE_PATHS.get(
        document_type,
        "Clinical Documents/Outpatient Core/clinic_note_{idx:04d}.pdf",
    )
    return Path(template.format(idx=idx))


def write_pdf(path: Path, title: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(path), pagesize=letter)
    _, height = letter

    x = 72
    y = height - 72
    line_height = 13
    max_chars = 92

    full_text = f"{title}\n\n{text}".strip()
    for raw_line in full_text.splitlines():
        raw_line = raw_line.rstrip()
        if not raw_line:
            y -= line_height
            continue

        for line in textwrap.wrap(raw_line, width=max_chars) or [""]:
            if y < 72:
                c.showPage()
                y = height - 72
            c.drawString(x, y, line)
            y -= line_height

    c.save()


def write_outputs(
    examples: list[PilotExample],
    out_root: Path,
    questions_out: Path,
    expected_out: Path,
) -> None:
    questions_out.parent.mkdir(parents=True, exist_ok=True)
    expected_out.parent.mkdir(parents=True, exist_ok=True)

    per_patient_counts: dict[str, int] = {}
    with questions_out.open("w", encoding="utf-8") as fq, expected_out.open("w", encoding="utf-8") as fe:
        for example in examples:
            per_patient_counts[example.patient_id] = per_patient_counts.get(example.patient_id, 0) + 1
            doc_idx = per_patient_counts[example.patient_id]

            patient_folder = out_root / f"EMRQA Pilot - {example.patient_id}"
            pdf_path = patient_folder / doc_relative_path(example.document_type, doc_idx)
            write_pdf(
                pdf_path,
                title=f"Clinical Note Source: {example.source_id}",
                text=example.note_text,
            )

            fq.write(
                json.dumps(
                    {
                        "patient_id": example.patient_id,
                        "question": example.question,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fe.write(
                json.dumps(
                    {
                        "patient_id": example.patient_id,
                        "question": example.question,
                        "required_answer_terms": example.required_answer_terms,
                        "required_any_terms": example.required_any_terms,
                        "required_source_document_types": [example.document_type],
                        "forbidden_answer_terms": example.forbidden_answer_terms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a tiny local emrQA-style pilot corpus for the existing PDF RAG pipeline. "
            "This script does not download data; point it at a small JSON/JSONL sample."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Local JSON/JSONL pilot input file.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("Data/public_testbed/emrqa_pilot/Test Patients"),
        help="Output root to use as PATIENTS_ROOT.",
    )
    parser.add_argument(
        "--questions-out",
        type=Path,
        default=Path("eval/emrqa_pilot_questions.jsonl"),
        help="Where to write pilot questions JSONL.",
    )
    parser.add_argument(
        "--expected-out",
        type=Path,
        default=Path("eval/emrqa_pilot_expected_checks.jsonl"),
        help="Where to write expected checks JSONL.",
    )
    parser.add_argument(
        "--document-type",
        default="clinic_note",
        help="Default document_type to assign when the input does not specify one.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum examples to convert.")
    parser.add_argument("--max-answer-terms", type=int, default=3, help="Maximum answer terms per question.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing pilot outputs.")
    args = parser.parse_args()

    if args.out_root.exists():
        if not args.force:
            raise SystemExit(
                f"Output root already exists: {args.out_root}\n"
                "Use --force if you want to overwrite/regenerate the pilot corpus."
            )
        shutil.rmtree(args.out_root)

    if args.force:
        for path in (args.questions_out, args.expected_out):
            if path.exists():
                path.unlink()

    raw_records = load_json_or_jsonl(args.input)
    examples = make_examples(
        raw_records,
        default_document_type=args.document_type,
        max_examples=args.limit,
        max_answer_terms=args.max_answer_terms,
    )

    if not examples:
        raise SystemExit(
            "No usable examples found. Each example needs a question, note/context/evidence text, "
            "and answer/required_answer_terms."
        )

    write_outputs(
        examples=examples,
        out_root=args.out_root,
        questions_out=args.questions_out,
        expected_out=args.expected_out,
    )

    print(f"Loaded raw records: {len(raw_records)}")
    print(f"Wrote pilot examples: {len(examples)}")
    print(f"Wrote PDF corpus under: {args.out_root}")
    print(f"Wrote questions: {args.questions_out}")
    print(f"Wrote expected checks: {args.expected_out}")
    print()
    print("Example build env:")
    print(f"  PATIENTS_ROOT='{args.out_root}' COLLECTION_NAME=ehr_chunks_emrqa_pilot ./run_build.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
