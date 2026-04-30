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


def load_chunks_by_id(path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    chunks: dict[str, dict[str, Any]] = {}
    for record in load_json_or_jsonl(path):
        chunk_id = record.get("chunk_id")
        if chunk_id:
            chunks[str(chunk_id)] = record
    return chunks


def unwrap_result(record: dict[str, Any]) -> dict[str, Any]:
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
        if isinstance(finding, dict) and isinstance(finding.get("evidence"), list):
            indices.extend(finding["evidence"])
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

    for match in INLINE_CITATION_RE.finditer(str(actual.get("answer") or "")):
        checked += 1
        idx = int(match.group(1))
        if idx < 1 or idx > source_count:
            failures.append(f"inline citation out of range: [{idx}]; sources={source_count}")
    return not failures, failures, checked


def first_rank_for_field(actual: dict[str, Any], field_name: str, required_values: list[str]) -> int | None:
    sources = actual.get("sources") or []
    if not isinstance(sources, list) or not required_values:
        return None
    required = set(required_values)
    for rank, source in enumerate(sources, start=1):
        if isinstance(source, dict) and source.get(field_name) in required:
            return rank
    return None


def matched_values_from_sources(actual: dict[str, Any], field_name: str, required_values: list[str]) -> list[str]:
    sources = actual.get("sources") or []
    source_values = {
        source.get(field_name)
        for source in sources
        if isinstance(source, dict)
    }
    return [value for value in required_values if value in source_values]


def retrieved_evidence_term_debug(
    actual: dict[str, Any],
    missing_terms: list[str],
    chunks_by_id: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Check whether answer terms missed by the model were visible in retrieved source text.

    This is diagnostic only. It does not affect pass/fail scoring.
    """
    if chunks_by_id is None:
        return None

    sources = actual.get("sources") or []
    if not isinstance(sources, list):
        sources = []

    retrieved_chunks: list[dict[str, Any]] = []
    unresolved_chunk_ids: list[str] = []

    for rank, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        chunk_id = source.get("chunk_id")
        if not chunk_id:
            continue
        chunk = chunks_by_id.get(str(chunk_id))
        if not chunk:
            unresolved_chunk_ids.append(str(chunk_id))
            continue
        retrieved_chunks.append(
            {
                "rank": rank,
                "chunk_id": str(chunk_id),
                "document_type": source.get("document_type") or chunk.get("document_type"),
                "evidence_kind": source.get("evidence_kind") or chunk.get("evidence_kind"),
                "source_table": source.get("source_table") or chunk.get("source_table"),
                "text": str(chunk.get("chunk_text") or ""),
            }
        )

    present_terms: list[str] = []
    absent_terms: list[str] = []
    first_source_rank: dict[str, int] = {}
    source_hits: dict[str, list[dict[str, Any]]] = {}

    for term in missing_terms:
        needle = normalize_text(term)
        hits = []
        for chunk in retrieved_chunks:
            if needle and needle in normalize_text(chunk["text"]):
                hits.append(
                    {
                        "rank": chunk["rank"],
                        "chunk_id": chunk["chunk_id"],
                        "document_type": chunk.get("document_type"),
                        "evidence_kind": chunk.get("evidence_kind"),
                        "source_table": chunk.get("source_table"),
                    }
                )
        if hits:
            present_terms.append(term)
            first_source_rank[term] = hits[0]["rank"]
            source_hits[term] = hits
        else:
            absent_terms.append(term)

    return {
        "enabled": True,
        "retrieved_sources": len(sources),
        "retrieved_chunks_resolved": len(retrieved_chunks),
        "retrieved_chunks_unresolved": len(unresolved_chunk_ids),
        "unresolved_chunk_ids": unresolved_chunk_ids[:20],
        "missing_terms_present_in_retrieved_evidence": present_terms,
        "missing_terms_absent_from_retrieved_evidence": absent_terms,
        "missing_term_first_source_rank": first_source_rank,
        "missing_term_source_hits": source_hits,
    }


def evaluate_record(
    expected: dict[str, Any],
    actual: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    answer_text = flatten_answer_text(actual)
    sources = actual.get("sources") or []
    if not isinstance(sources, list):
        sources = []

    required_terms = expected.get("required_answer_terms") or []
    required_any_groups = expected.get("required_any_terms") or []
    forbidden_terms = expected.get("forbidden_answer_terms") or []
    required_doc_types = expected.get("required_source_document_types") or []
    required_evidence_kinds = expected.get("required_evidence_kinds") or []

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

    matched_required_doc_types = matched_values_from_sources(actual, "document_type", required_doc_types)
    missing_required_doc_types = [d for d in required_doc_types if d not in matched_required_doc_types]
    matched_required_evidence_kinds = matched_values_from_sources(actual, "evidence_kind", required_evidence_kinds)
    missing_required_evidence_kinds = [k for k in required_evidence_kinds if k not in matched_required_evidence_kinds]

    structured_valid, structured_failures, status_counts = validate_structured_answer(actual.get("structured_answer"))
    citations_valid, citation_failures, citation_refs_checked = validate_evidence_citations(actual)

    first_doc_rank = first_rank_for_field(actual, "document_type", required_doc_types)
    doc_mrr_score = 0.0 if first_doc_rank is None else 1.0 / first_doc_rank
    first_kind_rank = first_rank_for_field(actual, "evidence_kind", required_evidence_kinds)
    kind_mrr_score = 0.0 if first_kind_rank is None else 1.0 / first_kind_rank

    top1 = sources[0] if sources else {}
    top1_doc_type = top1.get("document_type") if isinstance(top1, dict) else None
    top1_evidence_kind = top1.get("evidence_kind") if isinstance(top1, dict) else None
    top1_doc_correct = bool(required_doc_types and top1_doc_type in set(required_doc_types))
    top1_kind_correct = bool(required_evidence_kinds and top1_evidence_kind in set(required_evidence_kinds))

    failures = []
    failures.extend(f"missing required answer term: {term!r}" for term in missing_required_terms)
    failures.extend(f"missing at least one term from group: {group!r}" for group in missing_any_groups)
    failures.extend(f"forbidden answer term present: {term!r}" for term in forbidden_matches)
    failures.extend(f"missing required source document_type: {doc_type!r}" for doc_type in missing_required_doc_types)
    failures.extend(f"missing required evidence_kind: {kind!r}" for kind in missing_required_evidence_kinds)
    failures.extend(structured_failures)
    failures.extend(citation_failures)

    evidence_debug = retrieved_evidence_term_debug(actual, missing_required_terms, chunks_by_id)

    record = {
        "patient_id": expected.get("patient_id"),
        "question": expected.get("question"),
        "passed": not failures,
        "failures": failures,
        "required_terms": {"matched": matched_required_terms, "missing": missing_required_terms, "total": len(required_terms)},
        "required_any_groups": {"matched": matched_any_groups, "missing": missing_any_groups, "total": len(required_any_groups)},
        "forbidden_terms": {"matched": forbidden_matches, "total_configured": len(forbidden_terms)},
        "required_source_document_types": {"matched": matched_required_doc_types, "missing": missing_required_doc_types, "total": len(required_doc_types)},
        "required_evidence_kinds": {"matched": matched_required_evidence_kinds, "missing": missing_required_evidence_kinds, "total": len(required_evidence_kinds)},
        "structured_answer_valid": structured_valid,
        "structured_answer_failures": structured_failures,
        "evidence_citations_valid": citations_valid,
        "evidence_citation_failures": citation_failures,
        "evidence_citation_refs_checked": citation_refs_checked,
        "top1_document_type": top1_doc_type,
        "top1_document_type_correct": top1_doc_correct,
        "first_required_document_type_rank": first_doc_rank,
        "required_document_type_mrr_score": doc_mrr_score,
        "top1_evidence_kind": top1_evidence_kind,
        "top1_evidence_kind_correct": top1_kind_correct,
        "first_required_evidence_kind_rank": first_kind_rank,
        "required_evidence_kind_mrr_score": kind_mrr_score,
        "status_counts": dict(status_counts),
        "source_count": len(sources),
    }
    if evidence_debug is not None:
        record["answer_term_evidence_debug"] = evidence_debug
    return record


def build_summary(
    questions: list[dict[str, Any]],
    expected_records: list[dict[str, Any]],
    actual_records: list[dict[str, Any]],
    per_record: list[dict[str, Any]],
    missing_result_keys: list[tuple[str | None, str | None]],
    extra_result_keys: list[tuple[str | None, str | None]],
    chunks_debug_enabled: bool = False,
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
    required_kind_total = sum(rec["required_evidence_kinds"]["total"] for rec in per_record)
    required_kind_matched = sum(len(rec["required_evidence_kinds"]["matched"]) for rec in per_record)
    records_with_forbidden = sum(1 for rec in per_record if rec["forbidden_terms"]["matched"])
    forbidden_total = sum(len(rec["forbidden_terms"]["matched"]) for rec in per_record)
    structured_valid = sum(1 for rec in per_record if rec["structured_answer_valid"])
    citation_valid = sum(1 for rec in per_record if rec["evidence_citations_valid"])
    citation_refs_checked = sum(rec["evidence_citation_refs_checked"] for rec in per_record)

    top1_doc_den = sum(1 for rec in per_record if rec["required_source_document_types"]["total"] > 0)
    top1_doc_correct = sum(1 for rec in per_record if rec["top1_document_type_correct"])
    doc_mrr_scores = [rec["required_document_type_mrr_score"] for rec in per_record if rec["required_source_document_types"]["total"] > 0]
    top1_kind_den = sum(1 for rec in per_record if rec["required_evidence_kinds"]["total"] > 0)
    top1_kind_correct = sum(1 for rec in per_record if rec["top1_evidence_kind_correct"])
    kind_mrr_scores = [rec["required_evidence_kind_mrr_score"] for rec in per_record if rec["required_evidence_kinds"]["total"] > 0]

    retrieval_source_counts: Counter[str] = Counter()
    source_doc_type_counts: Counter[str] = Counter()
    evidence_kind_counts: Counter[str] = Counter()
    source_table_counts: Counter[str] = Counter()
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
            evidence_kind_counts[str(source.get("evidence_kind") or "missing")] += 1
            source_table_counts[str(source.get("source_table") or "missing")] += 1
    for rec in per_record:
        status_counts.update(rec["status_counts"])

    failures = [
        {"patient_id": rec["patient_id"], "question": rec["question"], "failures": rec["failures"]}
        for rec in per_record
        if not rec["passed"]
    ]
    for patient_id, question in missing_result_keys:
        failures.append({"patient_id": patient_id, "question": question, "failures": ["matching result not found"]})

    summary = {
        "input_counts": {"questions": len(questions), "expected_records": len(expected_records), "actual_records": len(actual_records), "missing_results": len(missing_result_keys), "extra_results": len(extra_result_keys)},
        "records": evaluated + len(missing_result_keys),
        "evaluated_records": evaluated,
        "passed": passed,
        "failed": failed + len(missing_result_keys),
        "evidence_grounded_task_success_rate": safe_div(passed, evaluated + len(missing_result_keys)),
        "answer_term_coverage": {"required_terms_matched": required_terms_matched, "required_terms_total": required_terms_total, "coverage": safe_div(required_terms_matched, required_terms_total), "required_any_groups_matched": any_groups_matched, "required_any_groups_total": any_groups_total, "any_group_coverage": safe_div(any_groups_matched, any_groups_total)},
        "forbidden_term_violations": {"records_with_violations": records_with_forbidden, "total_violations": forbidden_total},
        "required_source_document_type_recall": {"matched": required_doc_matched, "total": required_doc_total, "recall": safe_div(required_doc_matched, required_doc_total)},
        "required_evidence_kind_recall": {"matched": required_kind_matched, "total": required_kind_total, "recall": safe_div(required_kind_matched, required_kind_total)},
        "structured_answer_validity": {"valid": structured_valid, "invalid": evaluated - structured_valid, "rate": safe_div(structured_valid, evaluated)},
        "evidence_citation_validity": {"valid": citation_valid, "invalid": evaluated - citation_valid, "rate": safe_div(citation_valid, evaluated), "references_checked": citation_refs_checked},
        "retrieval_source_distribution": dict(sorted(retrieval_source_counts.items())),
        "source_document_type_distribution": dict(sorted(source_doc_type_counts.items())),
        "evidence_kind_distribution": dict(sorted(evidence_kind_counts.items())),
        "source_table_distribution": dict(sorted(source_table_counts.items())),
        "structured_status_distribution": dict(sorted(status_counts.items())),
        "top1_source_document_type_accuracy": {"correct": top1_doc_correct, "total": top1_doc_den, "accuracy": safe_div(top1_doc_correct, top1_doc_den)},
        "required_document_type_mrr": {"mean": safe_div(sum(doc_mrr_scores), len(doc_mrr_scores)), "total": len(doc_mrr_scores)},
        "top1_evidence_kind_accuracy": {"correct": top1_kind_correct, "total": top1_kind_den, "accuracy": safe_div(top1_kind_correct, top1_kind_den)},
        "required_evidence_kind_mrr": {"mean": safe_div(sum(kind_mrr_scores), len(kind_mrr_scores)), "total": len(kind_mrr_scores)},
        "missing_result_keys": [{"patient_id": patient_id, "question": question} for patient_id, question in missing_result_keys],
        "extra_result_keys": [{"patient_id": patient_id, "question": question} for patient_id, question in extra_result_keys],
        "failures": failures,
        "per_record": per_record,
    }

    if chunks_debug_enabled:
        debug_records = [rec.get("answer_term_evidence_debug") for rec in per_record if rec.get("answer_term_evidence_debug")]
        missing_terms_total = sum(len(rec["required_terms"]["missing"]) for rec in per_record)
        present = sum(len(debug.get("missing_terms_present_in_retrieved_evidence", [])) for debug in debug_records)
        absent = sum(len(debug.get("missing_terms_absent_from_retrieved_evidence", [])) for debug in debug_records)
        unresolved_chunks = sum(debug.get("retrieved_chunks_unresolved", 0) for debug in debug_records)
        summary["missing_answer_term_evidence_visibility"] = {
            "enabled": True,
            "missing_answer_terms_total": missing_terms_total,
            "present_in_retrieved_evidence": present,
            "absent_from_retrieved_evidence": absent,
            "present_rate": safe_div(present, missing_terms_total),
            "absent_rate": safe_div(absent, missing_terms_total),
            "unresolved_retrieved_chunks": unresolved_chunks,
        }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute metrics for EHR RAG /ask outputs.")
    parser.add_argument("--questions", type=Path, default=Path("eval/synthetic_bariatric_smoke_questions.jsonl"))
    parser.add_argument("--expected", type=Path, default=Path("eval/synthetic_bariatric_expected_checks.jsonl"))
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Optional chunks.jsonl file. When provided, missing answer terms are checked against retrieved chunk text by chunk_id.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fail-on-failed", action="store_true")
    args = parser.parse_args()

    questions = load_json_or_jsonl(args.questions) if args.questions.exists() else []
    expected_records = load_json_or_jsonl(args.expected)
    chunks_by_id = load_chunks_by_id(args.chunks)

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
        if actual is not None:
            per_record.append(evaluate_record(expected, actual, chunks_by_id=chunks_by_id))

    summary = build_summary(
        questions,
        expected_records,
        actual_records,
        per_record,
        missing_result_keys,
        extra_result_keys,
        chunks_debug_enabled=chunks_by_id is not None,
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
