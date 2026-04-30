#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_text(x: Any) -> str:
    return str(x or "").lower()


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # First try a normal JSON object/array, e.g. curl output.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL, e.g. scripts/test_ehr_retrieval_api.py --out output.
    records = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL in {path} at line {line_num}: {exc}") from exc

        if isinstance(obj, dict):
            records.append(obj)

    return records


def unwrap_result(record: dict[str, Any]) -> dict[str, Any]:
    # test_ehr_retrieval_api.py writes:
    # {"patient_id": ..., "question": ..., "result": {...}}
    if isinstance(record.get("result"), dict):
        merged = dict(record["result"])
        merged.setdefault("patient_id", record.get("patient_id"))
        merged.setdefault("question", record.get("question"))
        return merged

    # curl /ask output has answer/sources directly, but not always question.
    return record


def load_expected(path: Path) -> list[dict[str, Any]]:
    expected = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            expected.append(json.loads(line))
    return expected


def result_key(record: dict[str, Any]) -> tuple[str | None, str | None]:
    return record.get("patient_id"), record.get("question")


def check_record(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    failures = []

    answer = normalize_text(actual.get("answer"))
    structured = actual.get("structured_answer")
    sources = actual.get("sources") or []

    for term in expected.get("required_answer_terms", []):
        if normalize_text(term) not in answer:
            failures.append(f"missing required answer term: {term!r}")

    for group in expected.get("required_any_terms", []):
        if not any(normalize_text(term) in answer for term in group):
            failures.append(f"missing at least one term from group: {group!r}")

    for term in expected.get("forbidden_answer_terms", []):
        if normalize_text(term) in answer:
            failures.append(f"forbidden answer term present: {term!r}")

    source_doc_types = {
        source.get("document_type")
        for source in sources
        if isinstance(source, dict)
    }
    for doc_type in expected.get("required_source_document_types", []):
        if doc_type not in source_doc_types:
            failures.append(f"missing required source document_type: {doc_type!r}")

    if not isinstance(structured, dict):
        failures.append("structured_answer missing or not an object")
    else:
        findings = structured.get("findings")
        if not isinstance(findings, list) or not findings:
            failures.append("structured_answer.findings missing or empty")

        for idx, finding in enumerate(findings or [], start=1):
            if not isinstance(finding, dict):
                failures.append(f"finding {idx} is not an object")
                continue

            status = finding.get("status")
            evidence = finding.get("evidence")

            if status not in {"found", "not_found", "uncertain", "inferred_from_evidence"}:
                failures.append(f"finding {idx} has invalid status: {status!r}")

            if status in {"found", "inferred_from_evidence"}:
                if not isinstance(evidence, list) or not evidence:
                    failures.append(f"finding {idx} status={status} but evidence is empty")

            if status == "not_found":
                if evidence not in ([], None):
                    failures.append(f"finding {idx} status=not_found but evidence is non-empty")

        missing_information = structured.get("missing_information")
        if missing_information is not None and not isinstance(missing_information, list):
            failures.append("structured_answer.missing_information is not a list")

        uncertainty = structured.get("uncertainty")
        if uncertainty is not None and not isinstance(uncertainty, list):
            failures.append("structured_answer.uncertainty is not a list")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check synthetic bariatric smoke-test outputs against simple expected terms."
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        type=Path,
        help="One or more JSON/JSONL result files from synthetic smoke tests.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path("eval/synthetic_bariatric_expected_checks.jsonl"),
        help="Expected checks JSONL.",
    )
    parser.add_argument(
        "--show-passing",
        action="store_true",
        help="Print passing records too.",
    )
    args = parser.parse_args()

    expected_records = load_expected(args.expected)

    actual_records: list[dict[str, Any]] = []
    for path in args.result_files:
        for rec in load_json_or_jsonl(path):
            actual_records.append(unwrap_result(rec))

    actual_by_key = {
        result_key(rec): rec
        for rec in actual_records
    }

    total = 0
    passed = 0
    failed = 0

    for exp in expected_records:
        total += 1
        key = (exp.get("patient_id"), exp.get("question"))
        actual = actual_by_key.get(key)

        if actual is None:
            failed += 1
            print("=" * 100)
            print("FAILED")
            print("patient_id:", exp.get("patient_id"))
            print("question:", exp.get("question"))
            print("reason: matching result not found")
            continue

        failures = check_record(exp, actual)

        if failures:
            failed += 1
            print("=" * 100)
            print("FAILED")
            print("patient_id:", exp.get("patient_id"))
            print("question:", exp.get("question"))
            print("answer:", actual.get("answer"))
            print("failures:")
            for failure in failures:
                print(" -", failure)
        else:
            passed += 1
            if args.show_passing:
                print("=" * 100)
                print("PASSED")
                print("patient_id:", exp.get("patient_id"))
                print("question:", exp.get("question"))
                print("answer:", actual.get("answer"))

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("records:", total)
    print("passed:", passed)
    print("failed:", failed)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())