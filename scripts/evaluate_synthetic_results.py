#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

VALID_STATUSES = {"found", "not_found", "uncertain", "inferred_from_evidence"}
INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")


def normalize_text(value: Any) -> str:
    return str(value or "").lower()


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
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


def unwrap_result(record: dict[str, Any]) -> dict[str, Any]:
    # scripts/test_ehr_retrieval_api.py writes:
    # {"patient_id": ..., "question": ..., "structured": true, "result": {...}}
    if isinstance(record.get("result"), dict):
        merged = dict(record["result"])
        merged.setdefault("patient_id", record.get("patient_id"))
        merged.setdefault("question", record.get("question"))
        merged.setdefault("structured", record.get("structured"))
        return merged
    return record


def result_key(record: dict[str, Any]) -> tuple[str | None, str | None]:
    return record.get("patient_id"), record.get("question")


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def flatten_answer_text(actual: dict[str, Any]) -> str:
    parts = [str(actual.get("answer") or "")]
    structured = actual.get("structured_answer")
    if isinstance(structured, dict):
        # Include structured fields so a valid concise structured answer is not
        # unfairly penalized if the free-text answer is terse.
        parts.append(json.dumps(structured, ensure_ascii=False))
    return normalize_text("\n".join(parts))


def validate_structured_answer(structured: Any) -> tuple[bool, list[str], Counter[str]]:
    failures: list[str] = []
    status_counts: Counter[str] = Counter()

    if not isinstance(structured, dict):
        return False, ["structured_answer missing or not an object"], status_counts

    concise_answer = structured.get("concise_answer")
    if not isinstance(concise_answer, str) or not concise_answer.strip():
        failures.append("structured_answer.concise_answer missing or empty")

    findings = structured.get("findings")
    if not isinstance(findings, list) or not findings:
        failures.append("structured_answer.findings missing or empty")
        findings = []

    for idx, finding in enumerate(findings, start=1):
        if not isinstance(finding, dict):
            failures.append(f"finding {idx} is not an object")
            continue

        status = finding.get("status")
        status_counts[str(status)] += 1
        if status not in VALID_STATUSES:
            failures.append(f"finding {idx} has invalid status: {status!r}")

        evidence = finding.get("evidence")
        if evidence is not None and not isinstance(evidence, list):
            failures.append(f"finding {idx}.evidence is not a list")

        # Keep this intentionally lighter than check_structured_smoke_results.py:
        # not_found findings may cite evidence explaining absence in messy notes.
        if status in {"found", "inferred_from_evidence"}:
            if not isinstance(evidence, list) or not evidence:
                failures.append(f"finding {idx} status={status} but evidence is empty")

    missing_information = structured.get("missing_information")
    if missing_information is not None and not isinstance(missing_information, list):
        failures.append("structured_answer.missing_information is not a list")

    uncertainty = structured.get("uncertainty")
    if uncertainty is not None and not isinstance(uncertainty, list):
        failures.append("structured_answer.uncertainty is not a list")

    return not failures, failures, status_counts


def evidence_indices_from_structured(structured: Any) -> list[Any]:
    if not isinstance(structured, dict):
        return []
    findings = structured.get("findings")
    if not isinstance(findings, list):
        return []

    indices: list[Any] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        evidence = finding.get("evidence")
        if isinstance(evidence, list):
            indices.extend(evidence)
    return indices


def validate_evidence_citations(actual: dict[str, Any]) -> tuple[bool, list[str], int]:
    sources = actual.get("sources") or []
    source_count = len(sources) if isinstance(sources, list) else 0
    failures: list[str] = []
    checked = 0

    for raw_idx in evidence_indices_from_structured(actual.get("structured_answer")):
        checked += 1
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            failures.append(f"non-integer structured evidence index: {raw_idx!r}")
            continue
        if idx < 1 or idx > source_count:
            failures.append(f"structured evidence index out of range: {idx}; sources={source_count}")

    answer = str(actual.get("answer") or "")
    for match in INLINE_CITATION_RE.finditer(answer):
        checked += 1
        idx = int(match.group(1))
        if idx < 1 or idx > source_count:
            failures.append(f"inline citation out of range: [{idx}]; sources={source_count}")

    return not failures, failures, checked


def required_doc_type_rank(actual: dict[str, Any], required_doc_types: list[str]) -> int | None:
    sources = actual.get("sources") or []
    if not isinstance(sources, list) or not required_doc_types:
        return None

    required = set(required_doc_types)
    for rank, source in enumerate(sources, start=1):
        if isinstance(source, dict) and source.get("document_type") in required:
            return rank
    return None


def evaluate_record(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    answer_text = flatten_answer_text(actual)
    sources = actual.get("sources") or []
    if not isinstance(sources, list):
        sources = []

    required_terms = expected.get("required_answer_terms") or []
    required_any_groups = expected.get("required_any_terms") or []
    forbidden_terms = expected.get("forbidden_answer_terms") or []
    required_doc_types = expected.get("required_source_document_types") or []

    matched_required_terms = [term for term in required_terms if normalize_text(term) in answer_text]
    missing_required_terms = [term for term in required_terms if normalize_text(term) not in answer_text]

    matched_any_groups = []
    missing_any_groups = []
    for group in required_any_groups:
        matched_terms = [term for term in group if normalize_text(term) in answer_text]
        if matched_terms:
            matched_any_groups.append({"group": group, "matched_terms": matched_terms})
        else:
            missing_any_groups.append(group)

    forbidden_matches = [term for term in forbidden_terms if normalize_text(term) in answer_text]

    source_doc_types = [
        source.get("document_type")
        for source in sources
        if isinstance(source, dict)
    ]
    source_doc_type_set = set(source_doc_types)
    matched_required_doc_types = [doc_type for doc_type in required_doc_types if doc_type in source_doc_type_set]
    missing_required_doc_types = [doc_type for doc_type in required_doc_types if doc_type not in source_doc_type_set]

    structured_valid, structured_failures, status_counts = validate_structured_answer(
        actual.get("structured_answer")
    )
    citations_valid, citation_failures, citation_refs_checked = validate_evidence_citations(actual)

    first_required_rank = required_doc_type_rank(actual, required_doc_types)
    mrr_score = 0.0 if first_required_rank is None else 1.0 / first_required_rank
    top1_doc_type = source_doc_types[0] if source_doc_types else None
    top1_correct = bool(required_doc_types and top1_doc_type in set(required_doc_types))

    failures = []
    for term in missing_required_terms:
        failures.append(f"missing required answer term: {term!r}")
    for group in missing_any_groups:
        failures.append(f"missing at least one term from group: {group!r}")
    for term in forbidden_matches:
        failures.append(f"forbidden answer term present: {term!r}")
    for doc_type in missing_required_doc_types:
        failures.append(f"missing required source document_type: {doc_type!r}")
    failures.extend(structured_failures)
    failures.extend(citation_failures)

    return {
        "patient_id": expected.get("patient_id"),
        "question": expected.get("question"),
        "passed": not failures,
        "failures": failures,
        "required_terms": {
            "matched": matched_required_terms,
            "missing": missing_required_terms,
            "total": len(required_terms),
        },
        "required_any_groups": {
            "matched": matched_any_groups,
            "missing": missing_any_groups,
            "total": len(required_any_groups),
        },
        "forbidden_terms": {
            "matched": forbidden_matches,
            "total_configured": len(forbidden_terms),
        },
        "required_source_document_types": {
            "matched": matched_required_doc_types,
            "missing": missing_required_doc_types,
            "total": len(required_doc_types),
        },
        "structured_answer_valid": structured_valid,
        "structured_answer_failures": structured_failures,
        "evidence_citations_valid": citations_valid,
        "evidence_citation_failures": citation_failures,
        "evidence_citation_refs_checked": citation_refs_checked,
        "top1_document_type": top1_doc_type,
        "top1_document_type_correct": top1_correct,
        "first_required_document_type_rank": first_required_rank,
        "required_document_type_mrr_score": mrr_score,
        "status_counts": dict(status_counts),
        "source_count": len(sources),
    }


def build_summary(
    questions: list[dict[str, Any]],
    expected_records: list[dict[str, Any]],
    actual_records: list[dict[str, Any]],
    per_record: list[dict[str, Any]],
    missing_result_keys: list[tuple[str | None, str | None]],
    extra_result_keys: list[tuple[str | None, str | None]],
) -> dict[str, Any]:
    evaluated = len(per_record)
    passed = sum(1 for rec in per_record if rec["passed"])
    failed = evaluated - passed

    required_terms_total = sum(rec["required_terms"]["total"] for rec in per_record)
    required_terms_matched = sum(len(rec["required_terms"]["matched"]) for rec in per_record)
    any_groups_total = sum(rec["required_any_groups"]["total"] for rec in per_record)
    any_groups_matched = sum(len(rec["required_any_groups"]["matched"]) for rec in per_record)

    required_doc_total = sum(rec["required_source_document_types"]["total"] for rec in per_record)
    required_doc_matched = sum(len(rec["required_source_document_types"]["matched"]) for rec in per_record)

    records_with_forbidden = sum(1 for rec in per_record if rec["forbidden_terms"]["matched"])
    forbidden_total = sum(len(rec["forbidden_terms"]["matched"]) for rec in per_record)

    structured_valid = sum(1 for rec in per_record if rec["structured_answer_valid"])
    citation_valid = sum(1 for rec in per_record if rec["evidence_citations_valid"])
    citation_refs_checked = sum(rec["evidence_citation_refs_checked"] for rec in per_record)

    top1_den = sum(1 for rec in per_record if rec["required_source_document_types"]["total"] > 0)
    top1_correct = sum(1 for rec in per_record if rec["top1_document_type_correct"])
    mrr_scores = [
        rec["required_document_type_mrr_score"]
        for rec in per_record
        if rec["required_source_document_types"]["total"] > 0
    ]

    retrieval_source_counts: Counter[str] = Counter()
    source_doc_type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    actual_by_key = {result_key(rec): rec for rec in actual_records}
    for expected in expected_records:
        actual = actual_by_key.get(result_key(expected))
        if actual is None:
            continue
        sources = actual.get("sources") or []
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict):
                continue
            retrieval_source_counts[str(source.get("retrieval_source") or "missing")] += 1
            source_doc_type_counts[str(source.get("document_type") or "missing")] += 1

    for rec in per_record:
        status_counts.update(rec["status_counts"])

    failures = [
        {
            "patient_id": rec["patient_id"],
            "question": rec["question"],
            "failures": rec["failures"],
        }
        for rec in per_record
        if not rec["passed"]
    ]
    for patient_id, question in missing_result_keys:
        failures.append(
            {
                "patient_id": patient_id,
                "question": question,
                "failures": ["matching result not found"],
            }
        )

    return {
        "input_counts": {
            "questions": len(questions),
            "expected_records": len(expected_records),
            "actual_records": len(actual_records),
            "missing_results": len(missing_result_keys),
            "extra_results": len(extra_result_keys),
        },
        "records": evaluated + len(missing_result_keys),
        "evaluated_records": evaluated,
        "passed": passed,
        "failed": failed + len(missing_result_keys),
        "evidence_grounded_task_success_rate": safe_div(passed, evaluated + len(missing_result_keys)),
        "answer_term_coverage": {
            "required_terms_matched": required_terms_matched,
            "required_terms_total": required_terms_total,
            "coverage": safe_div(required_terms_matched, required_terms_total),
            "required_any_groups_matched": any_groups_matched,
            "required_any_groups_total": any_groups_total,
            "any_group_coverage": safe_div(any_groups_matched, any_groups_total),
        },
        "forbidden_term_violations": {
            "records_with_violations": records_with_forbidden,
            "total_violations": forbidden_total,
        },
        "required_source_document_type_recall": {
            "matched": required_doc_matched,
            "total": required_doc_total,
            "recall": safe_div(required_doc_matched, required_doc_total),
        },
        "structured_answer_validity": {
            "valid": structured_valid,
            "invalid": evaluated - structured_valid,
            "rate": safe_div(structured_valid, evaluated),
        },
        "evidence_citation_validity": {
            "valid": citation_valid,
            "invalid": evaluated - citation_valid,
            "rate": safe_div(citation_valid, evaluated),
            "references_checked": citation_refs_checked,
        },
        "retrieval_source_distribution": dict(sorted(retrieval_source_counts.items())),
        "source_document_type_distribution": dict(sorted(source_doc_type_counts.items())),
        "structured_status_distribution": dict(sorted(status_counts.items())),
        "top1_source_document_type_accuracy": {
            "correct": top1_correct,
            "total": top1_den,
            "accuracy": safe_div(top1_correct, top1_den),
        },
        "required_document_type_mrr": {
            "mean": safe_div(sum(mrr_scores), len(mrr_scores)),
            "total": len(mrr_scores),
        },
        "missing_result_keys": [
            {"patient_id": patient_id, "question": question}
            for patient_id, question in missing_result_keys
        ],
        "extra_result_keys": [
            {"patient_id": patient_id, "question": question}
            for patient_id, question in extra_result_keys
        ],
        "failures": failures,
        "per_record": per_record,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute metrics for synthetic bariatric /ask outputs."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("eval/synthetic_bariatric_smoke_questions.jsonl"),
        help="Synthetic smoke question JSONL file.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path("eval/synthetic_bariatric_expected_checks.jsonl"),
        help="Expected checks JSONL file.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSON/JSONL result files from scripts/test_ehr_retrieval_api.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path for metrics JSON.",
    )
    parser.add_argument(
        "--fail-on-failed",
        action="store_true",
        help="Exit non-zero if any expected record fails or is missing.",
    )
    args = parser.parse_args()

    questions = load_json_or_jsonl(args.questions) if args.questions.exists() else []
    expected_records = load_json_or_jsonl(args.expected)

    actual_records: list[dict[str, Any]] = []
    for result_file in args.results:
        for record in load_json_or_jsonl(result_file):
            actual_records.append(unwrap_result(record))

    actual_by_key = {result_key(rec): rec for rec in actual_records}
    expected_keys = [result_key(rec) for rec in expected_records]
    expected_key_set = set(expected_keys)
    actual_key_set = set(actual_by_key)

    missing_result_keys = [key for key in expected_keys if key not in actual_by_key]
    extra_result_keys = sorted(actual_key_set - expected_key_set, key=lambda x: (str(x[0]), str(x[1])))

    per_record = []
    for expected in expected_records:
        actual = actual_by_key.get(result_key(expected))
        if actual is None:
            continue
        per_record.append(evaluate_record(expected, actual))

    summary = build_summary(
        questions=questions,
        expected_records=expected_records,
        actual_records=actual_records,
        per_record=per_record,
        missing_result_keys=missing_result_keys,
        extra_result_keys=extra_result_keys,
    )

    summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary_json + "\n", encoding="utf-8")

    print(summary_json)

    if args.fail_on_failed and summary["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
