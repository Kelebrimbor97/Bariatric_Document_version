#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import textwrap
from dataclasses import dataclass, field
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

    @property
    def patient_id(self) -> str:
        return f"MIMICIV_{self.subject_id}_{self.hadm_id}"


def find_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        raise SystemExit(f"Could not find {filename!r} under {root}")
    if len(matches) > 1:
        print(f"[WARN] Multiple {filename!r} files found; using {matches[0]}")
    return matches[0]


def read_csv_gz(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        yield from reader


def compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
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
        if hadm_id in seen_hadm:
            continue
        if len(text) < min_text_chars:
            continue

        seen_hadm.add(hadm_id)
        cases.append(
            AdmissionCase(
                subject_id=subject_id,
                hadm_id=hadm_id,
                note_id=note_id,
                discharge_text=text,
            )
        )

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


def load_diagnoses(
    diagnoses_path: Path,
    dictionary: dict[tuple[str, str], str],
    hadm_ids: set[str],
    max_per_admission: int,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {hadm_id: [] for hadm_id in hadm_ids}
    for row in read_csv_gz(diagnoses_path):
        hadm_id = compact(row.get("hadm_id"))
        if hadm_id not in hadm_ids:
            continue
        code = compact(row.get("icd_code"))
        version = compact(row.get("icd_version"))
        title = dictionary.get((code, version)) or code
        if title and len(out[hadm_id]) < max_per_admission:
            out[hadm_id].append(title)
    return {k: unique_preserve_order(v) for k, v in out.items()}


def load_procedures(
    procedures_path: Path,
    dictionary: dict[tuple[str, str], str],
    hadm_ids: set[str],
    max_per_admission: int,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {hadm_id: [] for hadm_id in hadm_ids}
    for row in read_csv_gz(procedures_path):
        hadm_id = compact(row.get("hadm_id"))
        if hadm_id not in hadm_ids:
            continue
        code = compact(row.get("icd_code"))
        version = compact(row.get("icd_version"))
        title = dictionary.get((code, version)) or code
        if title and len(out[hadm_id]) < max_per_admission:
            out[hadm_id].append(title)
    return {k: unique_preserve_order(v) for k, v in out.items()}


def load_medications(
    prescriptions_path: Path,
    hadm_ids: set[str],
    max_per_admission: int,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {hadm_id: [] for hadm_id in hadm_ids}
    for row in read_csv_gz(prescriptions_path):
        hadm_id = compact(row.get("hadm_id"))
        if hadm_id not in hadm_ids:
            continue
        drug = compact(row.get("drug"))
        if drug and len(out[hadm_id]) < max_per_admission:
            out[hadm_id].append(drug)
    return {k: unique_preserve_order(v) for k, v in out.items()}


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


def case_root(out_root: Path, case: AdmissionCase) -> Path:
    # build_ehr_corpus.py treats text after " - " as the queryable actual_patient_id.
    return out_root / f"MIMIC IV Pilot - {case.patient_id}"


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


def diagnoses_text(case: AdmissionCase) -> str:
    lines = ["Coded Diagnoses", f"Hospital admission ID: {case.hadm_id}"]
    if case.diagnoses:
        lines.extend(f"{idx}. {title}" for idx, title in enumerate(case.diagnoses, start=1))
    else:
        lines.append("No ICD-coded diagnoses were loaded for this pilot admission.")
    return "\n".join(lines)


def procedures_text(case: AdmissionCase) -> str:
    lines = ["Coded Procedures", f"Hospital admission ID: {case.hadm_id}"]
    if case.procedures:
        lines.extend(f"{idx}. {title}" for idx, title in enumerate(case.procedures, start=1))
    else:
        lines.append("No ICD-coded procedures were loaded for this pilot admission.")
    return "\n".join(lines)


def medications_text(case: AdmissionCase) -> str:
    lines = ["Medication List", f"Hospital admission ID: {case.hadm_id}"]
    if case.medications:
        lines.extend(f"{idx}. {drug}" for idx, drug in enumerate(case.medications, start=1))
    else:
        lines.append("No medications were loaded from prescriptions for this pilot admission.")
    return "\n".join(lines)


def write_case_documents(out_root: Path, case: AdmissionCase) -> None:
    root = case_root(out_root, case)

    write_pdf(
        root / "Clinical Documents/Inpatient Core/discharge_summary.pdf",
        title=f"Discharge Summary Note {case.note_id}",
        text=case.discharge_text,
    )
    write_pdf(
        root / "Clinical Documents/Inpatient Core/clinic_note_admission_summary.pdf",
        title="Admission Metadata Clinic Note",
        text=admission_summary_text(case),
    )
    write_pdf(
        root / "Clinical Documents/Inpatient Core/clinic_note_diagnoses.pdf",
        title="Diagnoses Clinic Note",
        text=diagnoses_text(case),
    )
    write_pdf(
        root / "Perioperative Documents/operative_report_procedures.pdf",
        title="Procedure Coding Report",
        text=procedures_text(case),
    )
    write_pdf(
        root / "Clinical Documents/Inpatient Core/medication_list.pdf",
        title="Medication List",
        text=medications_text(case),
    )


def add_question(
    questions: list[dict[str, str]],
    expected: list[dict[str, Any]],
    case: AdmissionCase,
    question: str,
    required_terms: list[str],
    required_doc_types: list[str],
    required_any_terms: list[list[str]] | None = None,
    forbidden_terms: list[str] | None = None,
) -> None:
    required_terms = [compact(term) for term in required_terms if compact(term)]
    if not required_terms and not required_any_terms:
        return

    questions.append({"patient_id": case.patient_id, "question": question})
    expected.append(
        {
            "patient_id": case.patient_id,
            "question": question,
            "required_answer_terms": required_terms,
            "required_any_terms": required_any_terms or [],
            "required_source_document_types": required_doc_types,
            "forbidden_answer_terms": forbidden_terms or [],
        }
    )


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
        )

    if case.diagnoses:
        add_question(
            questions,
            expected,
            case,
            "What ICD-coded diagnoses are documented for this admission?",
            case.diagnoses[:max_terms_per_question],
            ["clinic_note"],
        )

    if case.medications:
        add_question(
            questions,
            expected,
            case,
            "What medications are documented for this admission?",
            case.medications[:max_terms_per_question],
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
        )

    return questions, expected


def write_questions_and_expected(
    cases: list[AdmissionCase],
    questions_out: Path,
    expected_out: Path,
    max_questions: int,
    max_terms_per_question: int,
) -> tuple[int, int]:
    questions_out.parent.mkdir(parents=True, exist_ok=True)
    expected_out.parent.mkdir(parents=True, exist_ok=True)

    all_questions: list[dict[str, str]] = []
    all_expected: list[dict[str, Any]] = []

    for case in cases:
        questions, expected = make_questions_for_case(case, max_terms_per_question=max_terms_per_question)
        for q, exp in zip(questions, expected):
            if len(all_questions) >= max_questions:
                break
            all_questions.append(q)
            all_expected.append(exp)
        if len(all_questions) >= max_questions:
            break

    with questions_out.open("w", encoding="utf-8") as fq:
        for item in all_questions:
            fq.write(json.dumps(item, ensure_ascii=False) + "\n")

    with expected_out.open("w", encoding="utf-8") as fe:
        for item in all_expected:
            fe.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(all_questions), len(all_expected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a small MIMIC-IV + MIMIC-IV-Note PDF pilot corpus for the existing EHR RAG pipeline. "
            "The first version intentionally uses discharge notes plus lightweight structured companion PDFs."
        )
    )
    parser.add_argument(
        "--mimiciv-root",
        type=Path,
        default=Path("/media/nishad/Desk SSD/Datasets/mimic/physionet.org/files/mimiciv"),
        help="Root containing the MIMIC-IV version directory, e.g. .../mimiciv.",
    )
    parser.add_argument(
        "--mimic-note-root",
        type=Path,
        default=Path("/media/nishad/Desk SSD/Datasets/mimic/physionet.org/files/mimic-iv-note"),
        help="Root containing the MIMIC-IV-Note version directory, e.g. .../mimic-iv-note.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("Data/public_testbed/mimic_iv_pilot/Test Patients"),
        help="Output root to use as PATIENTS_ROOT.",
    )
    parser.add_argument(
        "--questions-out",
        type=Path,
        default=Path("eval/mimic_iv_pilot_questions.jsonl"),
        help="Where to write pilot questions JSONL.",
    )
    parser.add_argument(
        "--expected-out",
        type=Path,
        default=Path("eval/mimic_iv_pilot_expected_checks.jsonl"),
        help="Where to write pilot expected checks JSONL.",
    )
    parser.add_argument("--limit-admissions", type=int, default=10, help="Number of admissions to convert.")
    parser.add_argument("--max-questions", type=int, default=25, help="Maximum total questions to write.")
    parser.add_argument("--max-diagnoses", type=int, default=5, help="Maximum diagnoses loaded per admission.")
    parser.add_argument("--max-procedures", type=int, default=5, help="Maximum procedures loaded per admission.")
    parser.add_argument("--max-medications", type=int, default=8, help="Maximum medications loaded per admission.")
    parser.add_argument("--max-terms-per-question", type=int, default=2, help="Maximum required answer terms per generated question.")
    parser.add_argument("--min-discharge-text-chars", type=int, default=1000, help="Minimum discharge note length.")
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

    discharge_path = find_file(args.mimic_note_root, "discharge.csv.gz")
    admissions_path = find_file(args.mimiciv_root, "admissions.csv.gz")
    diagnoses_path = find_file(args.mimiciv_root, "diagnoses_icd.csv.gz")
    d_diagnoses_path = find_file(args.mimiciv_root, "d_icd_diagnoses.csv.gz")
    procedures_path = find_file(args.mimiciv_root, "procedures_icd.csv.gz")
    d_procedures_path = find_file(args.mimiciv_root, "d_icd_procedures.csv.gz")
    prescriptions_path = find_file(args.mimiciv_root, "prescriptions.csv.gz")

    print(f"Using discharge notes: {discharge_path}")
    print(f"Using MIMIC-IV admissions: {admissions_path}")

    cases = choose_discharge_cases(
        discharge_path,
        limit_admissions=args.limit_admissions,
        min_text_chars=args.min_discharge_text_chars,
    )
    if not cases:
        raise SystemExit("No discharge-summary cases found for the requested pilot settings.")

    hadm_ids = {case.hadm_id for case in cases}
    admissions = load_admissions(admissions_path, hadm_ids)
    diag_dict = load_icd_dictionary(d_diagnoses_path)
    proc_dict = load_icd_dictionary(d_procedures_path)
    diagnoses = load_diagnoses(diagnoses_path, diag_dict, hadm_ids, max_per_admission=args.max_diagnoses)
    procedures = load_procedures(procedures_path, proc_dict, hadm_ids, max_per_admission=args.max_procedures)
    medications = load_medications(prescriptions_path, hadm_ids, max_per_admission=args.max_medications)

    for case in cases:
        case.admission = admissions.get(case.hadm_id)
        case.diagnoses = diagnoses.get(case.hadm_id, [])
        case.procedures = procedures.get(case.hadm_id, [])
        case.medications = medications.get(case.hadm_id, [])
        write_case_documents(args.out_root, case)

    n_questions, n_expected = write_questions_and_expected(
        cases,
        questions_out=args.questions_out,
        expected_out=args.expected_out,
        max_questions=args.max_questions,
        max_terms_per_question=args.max_terms_per_question,
    )

    print(f"Wrote admissions: {len(cases)}")
    print(f"Wrote questions: {n_questions} -> {args.questions_out}")
    print(f"Wrote expected checks: {n_expected} -> {args.expected_out}")
    print(f"Wrote PDF corpus under: {args.out_root}")
    print()
    print("Example build env:")
    print(f"  PATIENTS_ROOT='{args.out_root}' COLLECTION_NAME=ehr_chunks_mimic_iv_pilot ./run_build.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
