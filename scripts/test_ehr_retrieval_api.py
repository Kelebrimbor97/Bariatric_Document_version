from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests


DEFAULT_QUESTIONS = [
    "What bariatric procedure or surgical history is documented?",
    "What vitamin or micronutrient supplementation is documented?",
    "What relevant labs are available for anemia or micronutrient deficiency?",
    "What follow-up or monitoring plan is documented?",
]


def post_question(
    base_url: str,
    question: str,
    patient_id: str | None = None,
    structured: bool = False,
    timeout: int = 600,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "structured": structured,
    }
    if patient_id:
        payload["patient_id"] = patient_id

    response = requests.post(
        f"{base_url.rstrip('/')}/ask",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def print_result(result: dict[str, Any], show_answer: bool = True) -> None:
    print("\n" + "=" * 88)
    print("ANSWER")
    print("=" * 88)
    if show_answer:
        print(result.get("answer", ""))
    else:
        print("<answer hidden; use --show-answer to print it>")

    structured_answer = result.get("structured_answer")
    if structured_answer is not None:
        print("\n" + "=" * 88)
        print("STRUCTURED ANSWER")
        print("=" * 88)
        print(json.dumps(structured_answer, indent=2, ensure_ascii=False))

    print("\n" + "=" * 88)
    print("RETRIEVAL PLAN")
    print("=" * 88)
    print(json.dumps(result.get("retrieval_plan"), indent=2, ensure_ascii=False))

    print("\n" + "=" * 88)
    print("SOURCES")
    print("=" * 88)
    for i, source in enumerate(result.get("sources") or [], start=1):
        print(
            f"[{i}] type={source.get('document_type')} "
            f"section={source.get('section_title')} "
            f"score={source.get('rerank_score')} "
            f"page={source.get('page_num')}"
        )
        print(f"    path={source.get('relative_path')}")
        print(f"    chunk_id={source.get('chunk_id')}")


def load_questions(args: argparse.Namespace) -> list[str]:
    if args.question:
        return [args.question]

    if args.questions_file:
        questions_path = Path(args.questions_file)
        questions = []
        with questions_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                    q = obj.get("question")
                    if q:
                        questions.append(str(q))
                except json.JSONDecodeError:
                    questions.append(line)
        return questions

    return DEFAULT_QUESTIONS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the EHR RAG /ask API and print retrieval diagnostics."
    )
    parser.add_argument("--base-url", default="http://localhost:8090", help="EHR RAG API base URL")
    parser.add_argument("--patient-id", default=None, help="Optional patient identifier")
    parser.add_argument("--question", default=None, help="Single question to ask")
    parser.add_argument(
        "--questions-file",
        default=None,
        help="Optional text or JSONL file. Each line is either a plain question or {'question': ...}.",
    )
    parser.add_argument("--out", default=None, help="Optional JSONL output path")
    parser.add_argument("--show-answer", action="store_true", help="Print full answer text")
    parser.add_argument("--structured", action="store_true", help="Request structured JSON answer mode")
    args = parser.parse_args()

    questions = load_questions(args)
    if not questions:
        print("No questions provided.", file=sys.stderr)
        return 2

    out_f = None
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    try:
        for idx, question in enumerate(questions, start=1):
            print("\n" + "#" * 88)
            print(f"QUESTION {idx}/{len(questions)}")
            print("#" * 88)
            print(question)

            result = post_question(
                args.base_url,
                question,
                patient_id=args.patient_id,
                structured=args.structured,
            )
            result_record = {
                "patient_id": args.patient_id,
                "question": question,
                "structured": args.structured,
                "result": result,
            }

            if out_f:
                out_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                out_f.flush()

            print_result(result, show_answer=args.show_answer)
    finally:
        if out_f:
            out_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
