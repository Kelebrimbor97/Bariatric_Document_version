from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_STATUSES = {
    "found",
    "not_found",
    "uncertain",
    "inferred_from_evidence",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                records.append(
                    {
                        "_load_error": f"JSON decode error on line {line_no}: {e}",
                        "_line_no": line_no,
                    }
                )
    return records


def check_record(record: dict[str, Any], idx: int) -> list[str]:
    errors: list[str] = []

    if "_load_error" in record:
        return [record["_load_error"]]

    result = record.get("result")
    if not isinstance(result, dict):
        return ["missing or invalid result object"]

    sources = result.get("sources") or []
    if not isinstance(sources, list):
        errors.append("sources is not a list")
        sources = []

    max_evidence = len(sources)
    structured_answer = result.get("structured_answer")

    if structured_answer is None:
        errors.append("structured_answer is null")
        return errors

    if not isinstance(structured_answer, dict):
        errors.append("structured_answer is not an object")
        return errors

    concise_answer = structured_answer.get("concise_answer")
    if not isinstance(concise_answer, str) or not concise_answer.strip():
        errors.append("missing or empty concise_answer")

    findings = structured_answer.get("findings")
    if not isinstance(findings, list):
        errors.append("findings is missing or not a list")
        return errors

    if not findings:
        errors.append("findings list is empty")

    for finding_idx, finding in enumerate(findings, start=1):
        if not isinstance(finding, dict):
            errors.append(f"finding {finding_idx} is not an object")
            continue

        field = finding.get("field")
        if not isinstance(field, str) or not field.strip():
            errors.append(f"finding {finding_idx} missing field")

        status = finding.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"finding {finding_idx} invalid status: {status!r}")

        evidence = finding.get("evidence")
        if evidence is None:
            evidence = []
        if not isinstance(evidence, list):
            errors.append(f"finding {finding_idx} evidence is not a list")
            continue

        for ev in evidence:
            if not isinstance(ev, int):
                errors.append(f"finding {finding_idx} evidence value is not int: {ev!r}")
                continue
            if ev < 1 or ev > max_evidence:
                errors.append(
                    f"finding {finding_idx} evidence index {ev} out of range 1..{max_evidence}"
                )

        if status == "found" and not evidence:
            errors.append(f"finding {finding_idx} status=found but evidence is empty")

        if status == "not_found" and evidence:
            errors.append(f"finding {finding_idx} status=not_found but evidence is non-empty")

    for key in ["missing_information", "uncertainty"]:
        value = structured_answer.get(key)
        if value is not None and not isinstance(value, list):
            errors.append(f"{key} is not a list")

    return errors


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result") or {}
    structured_answer = result.get("structured_answer") or {}
    findings = structured_answer.get("findings") or []
    sources = result.get("sources") or []

    status_counts = Counter()
    for finding in findings:
        if isinstance(finding, dict):
            status_counts[str(finding.get("status"))] += 1

    source_type_counts = Counter(str(s.get("document_type")) for s in sources if isinstance(s, dict))

    return {
        "n_sources": len(sources),
        "n_findings": len(findings),
        "status_counts": dict(status_counts),
        "source_type_counts": dict(source_type_counts),
        "has_structured_answer": result.get("structured_answer") is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate structured EHR RAG smoke-test JSONL output."
    )
    parser.add_argument("path", help="Path to JSONL output from scripts/test_ehr_retrieval_api.py --structured")
    parser.add_argument("--show-passing", action="store_true", help="Print summaries for passing records too")
    args = parser.parse_args()

    path = Path(args.path)
    records = load_jsonl(path)

    total = len(records)
    failed = 0
    all_statuses = Counter()
    all_source_types = Counter()

    for idx, record in enumerate(records, start=1):
        question = record.get("question", f"record {idx}")
        errors = check_record(record, idx)
        summary = summarize_record(record)

        all_statuses.update(summary["status_counts"])
        all_source_types.update(summary["source_type_counts"])

        if errors:
            failed += 1
            print("\n" + "=" * 88)
            print(f"FAIL record {idx}: {question}")
            print("=" * 88)
            for err in errors:
                print(f"- {err}")
            print("summary:", json.dumps(summary, indent=2, ensure_ascii=False))
        elif args.show_passing:
            print("\n" + "=" * 88)
            print(f"PASS record {idx}: {question}")
            print("=" * 88)
            print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "#" * 88)
    print("STRUCTURED SMOKE CHECK SUMMARY")
    print("#" * 88)
    print(f"records: {total}")
    print(f"passed:  {total - failed}")
    print(f"failed:  {failed}")
    print("finding status counts:", dict(all_statuses))
    print("source document_type counts:", dict(all_source_types))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
