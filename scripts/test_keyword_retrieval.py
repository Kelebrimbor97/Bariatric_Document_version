from __future__ import annotations

import argparse
import json

from src.keyword_retrieval import get_keyword_retriever


DEFAULT_QUERIES = [
    "B12 ferritin iron thiamine vitamin D calcium PTH",
    "multivitamin calcium citrate vitamin D B12 iron thiamine",
    "Roux-en-Y gastric bypass sleeve gastrectomy bariatric procedure",
    "follow up monitoring nutrition dietitian supplements",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test local BM25 keyword retrieval over chunks.jsonl.")
    parser.add_argument("--patient-id", default=None, help="Optional patient identifier")
    parser.add_argument("--query", default=None, help="Single query to test")
    parser.add_argument("--limit", type=int, default=8, help="Hits per query")
    parser.add_argument(
        "--document-types",
        default=None,
        help="Optional comma-separated document_type filter, e.g. lab_report,nutrition_note",
    )
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    document_types = None
    if args.document_types:
        document_types = [x.strip() for x in args.document_types.split(",") if x.strip()]

    retriever = get_keyword_retriever()
    print(f"Loaded keyword records: {len(retriever.records)}")

    for query in queries:
        print("\n" + "#" * 88)
        print("QUERY:", query)
        print("#" * 88)
        hits = retriever.search(
            query=query,
            patient_id=args.patient_id,
            document_types=document_types,
            limit=args.limit,
        )
        print(f"hits: {len(hits)}")
        for i, hit in enumerate(hits, start=1):
            r = hit.record
            snippet = (r.get("chunk_text") or "").replace("\n", " ")[:320]
            print("\n" + "-" * 88)
            print(f"[{i}] bm25={hit.score:.4f}")
            print(
                json.dumps(
                    {
                        "patient_id": r.get("patient_id"),
                        "actual_patient_id": r.get("actual_patient_id"),
                        "document_type": r.get("document_type"),
                        "section_title": r.get("section_title"),
                        "relative_path": r.get("relative_path"),
                        "page_num": r.get("page_num"),
                        "chunk_id": r.get("chunk_id"),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            print("snippet:", snippet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
