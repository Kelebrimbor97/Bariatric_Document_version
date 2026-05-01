#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import chunk_text_with_sections


ATOMIC_EVIDENCE_KINDS = {
    "admission_summary",
    "diagnosis_list",
    "procedure_list",
    "medication_list",
}


@dataclass
class AdmissionCase:
    subject_id: str
    hadm_id: str
    note_id: str
    discharge_text: str
    admission: dict[str, str] | None = None
    diagnoses: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    radiology_reports: list[dict[str, str]] = field(default_factory=list)

    @property
    def patient_id(self) -> str:
        return f"MIMICIV_{self.subject_id}_{self.hadm_id}"

    @property
    def patient_folder_name(self) -> str:
        return f"MIMIC IV Pilot - {self.patient_id}"


def find_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        raise SystemExit(f"Could not find {filename!r} under {root}")
    if len(matches) > 1:
        print(f"[WARN] Multiple {filename!r} files found; using {matches[0]}")
    return matches[0]


def read_csv_gz(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        yield from csv.DictReader(f)


def compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = compact(value)
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def choose_discharge_cases(discharge_path: Path, limit_admissions: int, min_text_chars: int) -> list[AdmissionCase]:
    cases: list[AdmissionCase] = []
    seen_hadm: set[str] = set()

    for row in read_csv_gz(discharge_path):
        subject_id = compact(row.get("subject_id"))
        hadm_id = compact(row.get("hadm_id"))
        note_id = compact(row.get("note_id")) or f"discharge_{subject_id}_{hadm_id}"
        text = str(row.get("text") or "").strip()
        if not subject_id or not hadm_id or not text:
            continue
        if hadm_id in seen_hadm or len(text) < min_text_chars:
            continue

        seen_hadm.add(hadm_id)
        cases.append(AdmissionCase(subject_id=subject_id, hadm_id=hadm_id, note_id=note_id, discharge_text=text))
        if len(cases) >= limit_admissions:
            break

    return cases


def load_admissions(admissions_path: Path, hadm_ids: set[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv_gz(admissions_path):
        hadm_id = compact(row.get("hadm_id"))
        if hadm_id in hadm_ids:
            out[hadm_id] = row
            if len(out) >= len(hadm_ids):
                break
    return out


def load_icd_dictionary(path: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for row in read_csv_gz(path):
        code = compact(row.get("icd_code"))
        version = compact(row.get("icd_version"))
        title = compact(row.get("long_title"))
        if code and version and title:
            out[(code, version)] = title
    return out


def load_icd_items(
    path: Path,
    dictionary: dict[tuple[str, str], str],
    hadm_ids: set[str],
    max_per_admission: int,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {hadm_id: [] for hadm_id in hadm_ids}
    for row in read_csv_gz(path):
        hadm_id = compact(row.get("hadm_id"))
        if hadm_id not in hadm_ids:
            continue
        code = compact(row.get("icd_code"))
        version = compact(row.get("icd_version"))
        title = dictionary.get((code, version)) or code
        if title and len(out[hadm_id]) < max_per_admission:
            out[hadm_id].append(title)
    return {k: unique_preserve_order(v) for k, v in out.items()}


def load_medications(path: Path, hadm_ids: set[str], max_per_admission: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {hadm_id: [] for hadm_id in hadm_ids}
    for row in read_csv_gz(path):
        hadm_id = compact(row.get("hadm_id"))
        if hadm_id not in hadm_ids:
            continue
        drug = compact(row.get("drug"))
        if drug and len(out[hadm_id]) < max_per_admission:
            out[hadm_id].append(drug)
    return {k: unique_preserve_order(v) for k, v in out.items()}


def load_radiology_reports(
    path: Path,
    hadm_ids: set[str],
    max_per_admission: int,
    min_text_chars: int,
) -> dict[str, list[dict[str, str]]]:
    """Load a small number of MIMIC-IV-Note radiology reports per admission.

    Radiology is intentionally opt-in for the pilot so the existing direct MIMIC
    baseline remains reproducible unless --include-radiology is used.
    """
    out: dict[str, list[dict[str, str]]] = {hadm_id: [] for hadm_id in hadm_ids}
    for row in read_csv_gz(path):
        hadm_id = compact(row.get("hadm_id"))
        if hadm_id not in hadm_ids:
            continue
        if len(out[hadm_id]) >= max_per_admission:
            continue

        text = str(row.get("text") or "").strip()
        if len(text) < min_text_chars:
            continue

        out[hadm_id].append(
            {
                "note_id": compact(row.get("note_id")) or f"radiology_{hadm_id}_{len(out[hadm_id]) + 1}",
                "subject_id": compact(row.get("subject_id")),
                "hadm_id": hadm_id,
                "note_type": compact(row.get("note_type")),
                "note_seq": compact(row.get("note_seq")),
                "charttime": compact(row.get("charttime")),
                "storetime": compact(row.get("storetime")),
                "text": text,
            }
        )
    return out


def document_meta(
    case: AdmissionCase,
    relative_path: str,
    file_name: str,
    document_type: str,
    evidence_kind: str,
    source_table: str,
) -> dict[str, Any]:
    return {
        "patient_id": case.patient_id,
        "actual_patient_id": case.patient_id,
        "patient_folder_name": case.patient_folder_name,
        "pdf_path": relative_path,
        "relative_path": relative_path,
        "path_parts": str(Path(relative_path).parent).split("/"),
        "file_name": file_name,
        "path_tags": {
            "document_families": [],
            "care_contexts": [],
            "note_type_candidates": [document_type],
        },
        "document_type": document_type,
        "evidence_kind": evidence_kind,
        "source_table": source_table,
    }


def chunk_direct_text(text: str, evidence_kind: str) -> tuple[str, list[dict[str, Any]]]:
    """Chunk generated direct-text documents.

    Compact structured list documents are kept atomic so the full list is visible
    when that source is retrieved. Long discharge summaries and radiology reports
    use the normal section-aware chunker.
    """
    if evidence_kind in ATOMIC_EVIDENCE_KINDS:
        return "atomic", [
            {
                "chunk_text": text,
                "section_title": None,
                "section_chunk_index": 0,
            }
        ]
    return "section_aware", chunk_text_with_sections(text)


def add_document(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    case: AdmissionCase,
    relative_path: str,
    document_type: str,
    evidence_kind: str,
    source_table: str,
    text: str,
) -> None:
    text = text.strip()
    if not text:
        return

    file_name = Path(relative_path).name
    meta = document_meta(case, relative_path, file_name, document_type, evidence_kind, source_table)
    chunking_strategy, page_chunks = chunk_direct_text(text, evidence_kind)
    documents.append(
        {
            **meta,
            "n_pages": 1,
            "raw_text": text,
            "source_format": "direct_text",
            "chunking_strategy": chunking_strategy,
        }
    )

    stem = Path(file_name).stem
    for idx, ch in enumerate(page_chunks):
        section_title = ch.get("section_title")
        section_key = section_title or chunking_strategy
        chunks.append(
            {
                **meta,
                "page_num": 1,
                "section_title": section_title,
                "section_chunk_index": ch.get("section_chunk_index"),
                "chunking_strategy": chunking_strategy,
                "chunk_id": f"{case.patient_id}::{stem}::p1::s{section_key}::c{idx}",
                "chunk_text": ch["chunk_text"],
            }
        )


def admission_summary_text(case: AdmissionCase) -> str:
    adm = case.admission or {}
    lines = [
        "Admission Summary",
        f"Subject ID: {case.subject_id}",
        f"Hospital admission ID: {case.hadm_id}",
        f"Admission time: {compact(adm.get('admittime')) or 'not available'}",
        f"Discharge time: {compact(adm.get('dischtime')) or 'not available'}",
        f"Admission type: {compact(adm.get('admission_type')) or 'not available'}",
        f"Admission location: {compact(adm.get('admission_location')) or 'not available'}",
        f"Discharge location: {compact(adm.get('discharge_location')) or 'not available'}",
        f"Race: {compact(adm.get('race')) or 'not available'}",
        f"Hospital expire flag: {compact(adm.get('hospital_expire_flag')) or 'not available'}",
    ]
    return "\n".join(lines)


def list_text(title: str, hadm_id: str, items: list[str], empty_message: str) -> str:
    lines = [title, f"Hospital admission ID: {hadm_id}"]
    if items:
        lines.extend(f"{idx}. {item}" for idx, item in enumerate(items, start=1))
    else:
        lines.append(empty_message)
    return "\n".join(lines)


def safe_file_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in compact(value))
    return token or "unknown"


def radiology_report_text(case: AdmissionCase, report: dict[str, str]) -> str:
    lines = [
        "Radiology Report",
        f"Subject ID: {case.subject_id}",
        f"Hospital admission ID: {case.hadm_id}",
        f"Radiology note ID: {compact(report.get('note_id')) or 'not available'}",
        f"Note type: {compact(report.get('note_type')) or 'not available'}",
        f"Note sequence: {compact(report.get('note_seq')) or 'not available'}",
        f"Chart time: {compact(report.get('charttime')) or 'not available'}",
        f"Store time: {compact(report.get('storetime')) or 'not available'}",
        "",
        str(report.get("text") or "").strip(),
    ]
    return "\n".join(lines)


def add_case_documents(documents: list[dict[str, Any]], chunks: list[dict[str, Any]], case: AdmissionCase) -> None:
    add_document(
        documents,
        chunks,
        case,
        relative_path="Clinical Documents/Inpatient Core/discharge_summary.txt",
        document_type="discharge_summary",
        evidence_kind="discharge_summary",
        source_table="discharge",
        text=case.discharge_text,
    )
    add_document(
        documents,
        chunks,
        case,
        relative_path="Clinical Documents/Inpatient Core/admission_summary.txt",
        document_type="clinic_note",
        evidence_kind="admission_summary",
        source_table="admissions",
        text=admission_summary_text(case),
    )
    add_document(
        documents,
        chunks,
        case,
        relative_path="Structured Documents/diagnosis_list.txt",
        document_type="clinic_note",
        evidence_kind="diagnosis_list",
        source_table="diagnoses_icd",
        text=list_text(
            "Coded Diagnoses",
            case.hadm_id,
            case.diagnoses,
            "No ICD-coded diagnoses were loaded for this pilot admission.",
        ),
    )
    add_document(
        documents,
        chunks,
        case,
        relative_path="Structured Documents/procedure_list.txt",
        document_type="operative_report",
        evidence_kind="procedure_list",
        source_table="procedures_icd",
        text=list_text(
            "Coded Procedures",
            case.hadm_id,
            case.procedures,
            "No ICD-coded procedures were loaded for this pilot admission.",
        ),
    )
    add_document(
        documents,
        chunks,
        case,
        relative_path="Structured Documents/medication_list.txt",
        document_type="medication_list",
        evidence_kind="medication_list",
        source_table="prescriptions",
        text=list_text(
            "Medication List",
            case.hadm_id,
            case.medications,
            "No medications were loaded from prescriptions for this pilot admission.",
        ),
    )

    for idx, report in enumerate(case.radiology_reports, start=1):
        note_id = safe_file_token(report.get("note_id") or f"radiology_{idx}")
        add_document(
            documents,
            chunks,
            case,
            relative_path=f"Clinical Documents/Radiology/radiology_{idx:02d}_{note_id}.txt",
            document_type="radiology",
            evidence_kind="radiology_report",
            source_table="radiology",
            text=radiology_report_text(case, report),
        )


def add_question(
    questions: list[dict[str, str]],
    expected: list[dict[str, Any]],
    case: AdmissionCase,
    question: str,
    required_terms: list[str],
    required_doc_types: list[str],
    required_evidence_kinds: list[str],
) -> None:
    required_terms = [compact(term) for term in required_terms if compact(term)]
    if not required_terms:
        return

    questions.append({"patient_id": case.patient_id, "question": question})
    expected.append(
        {
            "patient_id": case.patient_id,
            "question": question,
            "required_answer_terms": required_terms,
            "required_any_terms": [],
            "required_source_document_types": required_doc_types,
            "required_evidence_kinds": required_evidence_kinds,
            "forbidden_answer_terms": [],
        }
    )


def split_radiology_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None

    header_re = re.compile(r"^\s*([A-Z][A-Z /_-]{2,40})\s*:?\s*(.*)$")
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = header_re.match(line)
        if match:
            header = compact(match.group(1)).lower()
            rest = compact(match.group(2))
            current = header
            sections.setdefault(current, [])
            if rest:
                sections[current].append(rest)
            continue
        if current:
            sections[current].append(line)

    return {key: "\n".join(value).strip() for key, value in sections.items() if "\n".join(value).strip()}


def select_radiology_expected_terms(report: dict[str, str], max_terms: int) -> list[str]:
    """Pick short answer-check terms from impression/findings text.

    This is only for deterministic pilot checks. It intentionally favors exact
    phrases visible in the report over a rigid radiology schema.
    """
    text = str(report.get("text") or "")
    sections = split_radiology_sections(text)
    preferred_text = ""
    for section_name in ("impression", "conclusion", "findings"):
        if sections.get(section_name):
            preferred_text = sections[section_name]
            break
    if not preferred_text:
        preferred_text = text

    fragments: list[str] = []
    for fragment in re.split(r"[.;\n]+", preferred_text):
        fragment = compact(re.sub(r"^\s*\d+\.?\s*", "", fragment))
        if not fragment:
            continue
        lower = fragment.lower()
        if lower in {"impression", "findings", "conclusion", "exam", "history", "indication", "comparison", "technique"}:
            continue
        if len(fragment) < 4 or len(fragment) > 100:
            continue
        if not re.search(r"[A-Za-z]{4}", fragment):
            continue
        fragments.append(fragment)

    return unique_preserve_order(fragments)[:max_terms]


def make_questions_for_case(case: AdmissionCase, max_terms_per_question: int) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    questions: list[dict[str, str]] = []
    expected: list[dict[str, Any]] = []
    adm = case.admission or {}

    discharge_location = compact(adm.get("discharge_location"))
    if discharge_location:
        add_question(
            questions,
            expected,
            case,
            "What discharge location or disposition is documented?",
            [discharge_location],
            ["clinic_note"],
            ["admission_summary"],
        )

    admission_type = compact(adm.get("admission_type"))
    if admission_type:
        add_question(
            questions,
            expected,
            case,
            "What admission type is documented?",
            [admission_type],
            ["clinic_note"],
            ["admission_summary"],
        )

    if case.diagnoses:
        add_question(
            questions,
            expected,
            case,
            "What ICD-coded diagnoses are documented for this admission?",
            case.diagnoses[:max_terms_per_question],
            ["clinic_note"],
            ["diagnosis_list"],
        )

    if case.medications:
        add_question(
            questions,
            expected,
            case,
            "What medications are documented for this admission?",
            case.medications[:max_terms_per_question],
            ["medication_list"],
            ["medication_list"],
        )

    if case.procedures:
        add_question(
            questions,
            expected,
            case,
            "What ICD-coded procedures are documented for this admission?",
            case.procedures[:max_terms_per_question],
            ["operative_report"],
            ["procedure_list"],
        )

    if case.radiology_reports:
        radiology_terms = select_radiology_expected_terms(case.radiology_reports[0], max_terms_per_question)
        if radiology_terms:
            add_question(
                questions,
                expected,
                case,
                "What radiology findings or impression are documented for this admission?",
                radiology_terms,
                ["radiology"],
                ["radiology_report"],
            )

    return questions, expected


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small MIMIC-IV + MIMIC-IV-Note pilot corpus directly as documents.jsonl/chunks.jsonl. "
            "This avoids CSV/text -> PDF -> PDF extraction and isolates checkpoints via --processed-dir."
        )
    )
    parser.add_argument("--mimiciv-root", type=Path, default=Path("/media/nishad/Desk SSD/Datasets/mimic/physionet.org/files/mimiciv"))
    parser.add_argument("--mimic-note-root", type=Path, default=Path("/media/nishad/Desk SSD/Datasets/mimic/physionet.org/files/mimic-iv-note"))
    parser.add_argument("--processed-dir", type=Path, default=Path("Data/processed_mimic_iv_pilot"))
    parser.add_argument("--questions-out", type=Path, default=Path("eval/mimic_iv_pilot_questions.jsonl"))
    parser.add_argument("--expected-out", type=Path, default=Path("eval/mimic_iv_pilot_expected_checks.jsonl"))
    parser.add_argument("--limit-admissions", type=int, default=10)
    parser.add_argument("--max-questions", type=int, default=25)
    parser.add_argument("--max-diagnoses", type=int, default=5)
    parser.add_argument("--max-procedures", type=int, default=5)
    parser.add_argument("--max-medications", type=int, default=8)
    parser.add_argument("--max-terms-per-question", type=int, default=2)
    parser.add_argument("--min-discharge-text-chars", type=int, default=1000)
    parser.add_argument("--include-radiology", action="store_true", help="Also ingest MIMIC-IV-Note radiology.csv.gz reports for selected admissions.")
    parser.add_argument("--max-radiology-reports", type=int, default=2, help="Maximum radiology reports to add per selected admission when --include-radiology is used.")
    parser.add_argument("--min-radiology-text-chars", type=int, default=80, help="Minimum radiology report text length to include.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.processed_dir.exists():
        if not args.force:
            raise SystemExit(f"Processed directory already exists: {args.processed_dir}\nUse --force to overwrite/regenerate it.")
        shutil.rmtree(args.processed_dir)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        for path in (args.questions_out, args.expected_out):
            if path.exists():
                path.unlink()

    discharge_path = find_file(args.mimic_note_root, "discharge.csv.gz")
    radiology_path = find_file(args.mimic_note_root, "radiology.csv.gz") if args.include_radiology else None
    admissions_path = find_file(args.mimiciv_root, "admissions.csv.gz")
    diagnoses_path = find_file(args.mimiciv_root, "diagnoses_icd.csv.gz")
    d_diagnoses_path = find_file(args.mimiciv_root, "d_icd_diagnoses.csv.gz")
    procedures_path = find_file(args.mimiciv_root, "procedures_icd.csv.gz")
    d_procedures_path = find_file(args.mimiciv_root, "d_icd_procedures.csv.gz")
    prescriptions_path = find_file(args.mimiciv_root, "prescriptions.csv.gz")

    print(f"Using discharge notes: {discharge_path}")
    if radiology_path:
        print(f"Using radiology notes: {radiology_path}")
    print(f"Using MIMIC-IV admissions: {admissions_path}")
    print(f"Writing processed corpus to: {args.processed_dir}")

    cases = choose_discharge_cases(discharge_path, args.limit_admissions, args.min_discharge_text_chars)
    if not cases:
        raise SystemExit("No discharge-summary cases found for the requested pilot settings.")

    hadm_ids = {case.hadm_id for case in cases}
    admissions = load_admissions(admissions_path, hadm_ids)
    diagnoses = load_icd_items(diagnoses_path, load_icd_dictionary(d_diagnoses_path), hadm_ids, args.max_diagnoses)
    procedures = load_icd_items(procedures_path, load_icd_dictionary(d_procedures_path), hadm_ids, args.max_procedures)
    medications = load_medications(prescriptions_path, hadm_ids, args.max_medications)
    radiology_reports = (
        load_radiology_reports(
            radiology_path,
            hadm_ids,
            args.max_radiology_reports,
            args.min_radiology_text_chars,
        )
        if radiology_path
        else {hadm_id: [] for hadm_id in hadm_ids}
    )

    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    all_questions: list[dict[str, str]] = []
    all_expected: list[dict[str, Any]] = []

    for case in cases:
        case.admission = admissions.get(case.hadm_id)
        case.diagnoses = diagnoses.get(case.hadm_id, [])
        case.procedures = procedures.get(case.hadm_id, [])
        case.medications = medications.get(case.hadm_id, [])
        case.radiology_reports = radiology_reports.get(case.hadm_id, [])
        add_case_documents(documents, chunks, case)

        questions, expected = make_questions_for_case(case, args.max_terms_per_question)
        for q, exp in zip(questions, expected):
            if len(all_questions) >= args.max_questions:
                break
            all_questions.append(q)
            all_expected.append(exp)
        if len(all_questions) >= args.max_questions:
            break

    n_docs = write_jsonl(args.processed_dir / "documents.jsonl", documents)
    n_chunks = write_jsonl(args.processed_dir / "chunks.jsonl", chunks)
    n_questions = write_jsonl(args.questions_out, all_questions)
    n_expected = write_jsonl(args.expected_out, all_expected)

    print(f"Wrote admissions: {len(cases)}")
    print(f"Wrote documents: {n_docs} -> {args.processed_dir / 'documents.jsonl'}")
    print(f"Wrote chunks: {n_chunks} -> {args.processed_dir / 'chunks.jsonl'}")
    print(f"Wrote questions: {n_questions} -> {args.questions_out}")
    print(f"Wrote expected checks: {n_expected} -> {args.expected_out}")
    print()
    print("Index with:")
    print(f"  PROCESSED_DIR='{args.processed_dir}' COLLECTION_NAME=ehr_chunks_mimic_iv_pilot python scripts/index_qdrant_medcpt.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
