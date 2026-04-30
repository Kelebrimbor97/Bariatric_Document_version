#!/usr/bin/env python3
import argparse
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path


def find_sources(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("sources"), list):
            return obj["sources"]
        for v in obj.values():
            found = find_sources(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_sources(item)
            if found is not None:
                return found
    return None


def find_first_string(obj, keys):
    if isinstance(obj, dict):
        for key in keys:
            if isinstance(obj.get(key), str):
                return obj[key]
        for v in obj.values():
            found = find_first_string(v, keys)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_first_string(item, keys)
            if found:
                return found
    return None


def load_chunks_by_id(chunks_path):
    chunks_by_id = {}
    if not chunks_path or not chunks_path.exists():
        return chunks_by_id

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = rec.get("chunk_id")
            if chunk_id:
                chunks_by_id[chunk_id] = rec

    return chunks_by_id


def looks_boilerplate_like(text):
    text_l = (text or "").lower()
    boilerplate_terms = [
        "completed action list",
        "printed by",
        "printed on",
        "result status",
        "result title",
        "performed by",
        "verified by",
        "encounter info",
    ]
    return any(term in text_l for term in boilerplate_terms)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect dense/keyword/both retrieval sources from EHR RAG smoke results."
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Path to smoke result JSONL, e.g. Data/processed/ehr_retrieval_hybrid_sources_smoke_results.jsonl",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("Data/processed/chunks.jsonl"),
        help="Path to chunks.jsonl so chunk_text can be recovered.",
    )
    parser.add_argument(
        "--source",
        choices=["keyword", "dense", "both", "missing", "all"],
        default="keyword",
        help="Which retrieval_source to print in detail.",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print chunk text for matching sources.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2500,
        help="Max chunk text chars to print when --show-text is used.",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(f"Missing results file: {args.results}")

    chunks_by_id = load_chunks_by_id(args.chunks)

    retrieval_source_counts = Counter()
    doc_type_counts = Counter()
    section_counts = Counter()
    scores_by_source = defaultdict(list)
    printed = 0

    with args.results.open("r", encoding="utf-8") as f:
        for record_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue

            result = json.loads(line)
            question = find_first_string(result, ["question", "query", "input"]) or ""
            answer = find_first_string(result, ["answer"]) or ""
            sources = find_sources(result) or []

            for rank, source in enumerate(sources, start=1):
                retrieval_source = source.get("retrieval_source") or "missing"
                retrieval_source_counts[retrieval_source] += 1

                doc_type = source.get("document_type") or "missing"
                section = source.get("section_title") or "None"
                doc_type_counts[doc_type] += 1
                section_counts[section] += 1

                score = source.get("rerank_score")
                if isinstance(score, (int, float)):
                    scores_by_source[retrieval_source].append(float(score))

                if args.source != "all" and retrieval_source != args.source:
                    continue

                chunk_id = source.get("chunk_id")
                chunk = chunks_by_id.get(chunk_id, {})
                chunk_text = chunk.get("chunk_text") or source.get("chunk_text") or ""

                printed += 1

                print("=" * 120)
                print(f"record: {record_idx}")
                print(f"rank: {rank}")
                print(f"retrieval_source: {retrieval_source}")
                print(f"rerank_score: {score}")
                print(f"document_type: {doc_type}")
                print(f"section_title: {section}")
                print(f"relative_path: {source.get('relative_path')}")
                print(f"page_num: {source.get('page_num')}")
                print(f"chunk_id: {chunk_id}")
                print(f"boilerplate_like: {looks_boilerplate_like(chunk_text)}")
                print()
                print("question:")
                print(textwrap.fill(question, width=110))
                print()
                print("answer excerpt:")
                print(textwrap.shorten(answer.replace("\n", " "), width=900, placeholder=" ..."))

                if args.show_text:
                    print()
                    print("chunk text:")
                    print("-" * 120)
                    print(chunk_text[: args.max_chars])
                    if len(chunk_text) > args.max_chars:
                        print("\n...[truncated]")

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    print("\nretrieval_source counts:")
    for key, value in retrieval_source_counts.most_common():
        print(f"  {key}: {value}")

    print("\ndocument_type counts:")
    for key, value in doc_type_counts.most_common():
        print(f"  {key}: {value}")

    print("\nrerank score ranges by retrieval_source:")
    for key, values in sorted(scores_by_source.items()):
        if not values:
            continue
        print(
            f"  {key}: "
            f"n={len(values)}, "
            f"min={min(values):.4f}, "
            f"median={sorted(values)[len(values)//2]:.4f}, "
            f"max={max(values):.4f}"
        )

    print(f"\nprinted matching sources: {printed}")


if __name__ == "__main__":
    main()