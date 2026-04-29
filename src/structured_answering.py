from __future__ import annotations

import json
import re
from typing import Any


STRUCTURED_ANSWER_SCHEMA = {
    "concise_answer": "short evidence-grounded answer in plain English",
    "findings": [
        {
            "field": "clinical field or question-specific item",
            "status": "found | not_found | uncertain | inferred_from_evidence",
            "value": "answer value, or null if absent",
            "evidence": [1, 2],
            "rationale": "brief explanation tied to retrieved evidence",
        }
    ],
    "missing_information": ["important items not found in retrieved evidence"],
    "uncertainty": ["limitations or ambiguities in retrieved evidence"],
}


def build_structured_answer_prompt(
    question: str,
    evidence_blocks: list[str],
    retrieval_plan: dict[str, Any],
) -> str:
    schema_json = json.dumps(STRUCTURED_ANSWER_SCHEMA, indent=2)
    evidence_text = "\n".join(evidence_blocks)

    return f"""
You are answering a local EHR/chart question using retrieved evidence only.

Return ONLY valid JSON. Do not include markdown. Do not include commentary outside JSON.

Required JSON schema shape:
{schema_json}

Rules:
- Use only the evidence below.
- Cite evidence using bracket numbers from the evidence blocks, e.g. [1], [2].
- For each finding, set status to exactly one of: found, not_found, uncertain, inferred_from_evidence.
- Use found only when the value is explicitly supported by evidence.
- Use inferred_from_evidence only for cautious inferences directly supported by evidence.
- Use not_found when relevant information is absent from the retrieved evidence.
- Use uncertain when evidence is conflicting, vague, or incomplete.
- Do not invent diagnoses, procedures, dates, medications, labs, recommendations, or source details.
- If a requested item is missing, put it in missing_information.
- Keep rationale concise and evidence-tied.

Retrieval plan:
{json.dumps(retrieval_plan, ensure_ascii=False)}

Question:
{question}

Evidence:
{evidence_text}
""".strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None
