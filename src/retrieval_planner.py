from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass
class RetrievalPlan:
    primary_query: str
    subqueries: list[str]
    target_document_types: list[str]
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


BARIATRIC_TERMS = re.compile(
    r"\b(bariatric|gastric bypass|roux|rygb|sleeve|gastrectomy|weight loss surgery|post.?op|postoperative)\b",
    re.IGNORECASE,
)
LAB_TERMS = re.compile(
    r"\b(lab|labs|laboratory|b12|folate|ferritin|iron|thiamine|vitamin|calcium|pth|albumin|hemoglobin|hgb|anemia)\b",
    re.IGNORECASE,
)
MED_TERMS = re.compile(
    r"\b(medication|medications|meds|supplement|supplements|multivitamin|calcium citrate|vitamin d|b12|iron|thiamine)\b",
    re.IGNORECASE,
)
SURGERY_TERMS = re.compile(
    r"\b(surgery|procedure|operative|operation|bypass|sleeve|gastrectomy|roux|rygb)\b",
    re.IGNORECASE,
)
FOLLOWUP_TERMS = re.compile(
    r"\b(follow.?up|monitoring|surveillance|plan|recommendation|instructions|discharge)\b",
    re.IGNORECASE,
)
NUTRITION_TERMS = re.compile(
    r"\b(nutrition|diet|dietitian|dietary|protein|intake|malnutrition|deficiency)\b",
    re.IGNORECASE,
)


def _append_unique(items: list[str], additions: list[str]) -> list[str]:
    seen = set(items)
    for item in additions:
        if item not in seen:
            items.append(item)
            seen.add(item)
    return items


def build_retrieval_plan(question: str) -> RetrievalPlan:
    """
    Deterministic, low-risk retrieval planner inspired by CLI-RAG.

    It does not call an LLM. It only expands the user question into a small set
    of clinically targeted retrieval subqueries and preferred document types.
    """
    q = question.strip()
    subqueries = [q]
    doc_types: list[str] = []
    reasons: list[str] = []

    if BARIATRIC_TERMS.search(q):
        reasons.append("bariatric concepts detected")
        _append_unique(
            doc_types,
            [
                "operative_report",
                "discharge_summary",
                "nutrition_note",
                "lab_report",
                "medication_list",
                "clinic_note",
                "patient_instructions",
            ],
        )
        _append_unique(
            subqueries,
            [
                "bariatric procedure type Roux-en-Y gastric bypass sleeve gastrectomy operative history",
                "post bariatric discharge instructions vitamins supplements follow up monitoring",
                "post bariatric nutrition dietitian protein intake supplement adherence",
            ],
        )

    if SURGERY_TERMS.search(q):
        reasons.append("surgery/procedure concepts detected")
        _append_unique(doc_types, ["operative_report", "discharge_summary", "history_and_physical"])
        _append_unique(
            subqueries,
            [
                "operative report procedure performed surgical history",
                "past surgical history bariatric procedure",
            ],
        )

    if LAB_TERMS.search(q):
        reasons.append("lab/micronutrient concepts detected")
        _append_unique(doc_types, ["lab_report", "discharge_summary", "clinic_note", "nutrition_note"])
        _append_unique(
            subqueries,
            [
                "B12 folate ferritin iron thiamine vitamin D calcium PTH albumin hemoglobin labs",
                "micronutrient deficiency anemia vitamin deficiency laboratory results",
            ],
        )

    if MED_TERMS.search(q):
        reasons.append("medication/supplement concepts detected")
        _append_unique(doc_types, ["medication_list", "discharge_summary", "clinic_note", "patient_instructions"])
        _append_unique(
            subqueries,
            [
                "multivitamin calcium citrate vitamin D B12 iron thiamine supplement medication list",
                "discharge medications supplements bariatric vitamins",
            ],
        )

    if NUTRITION_TERMS.search(q):
        reasons.append("nutrition concepts detected")
        _append_unique(doc_types, ["nutrition_note", "clinic_note", "discharge_summary"])
        _append_unique(
            subqueries,
            [
                "nutrition assessment dietitian note protein intake diet tolerance",
                "nutrition intervention supplement adherence deficiency risk",
            ],
        )

    if FOLLOWUP_TERMS.search(q):
        reasons.append("follow-up/plan concepts detected")
        _append_unique(doc_types, ["discharge_summary", "patient_instructions", "clinic_note", "nutrition_note"])
        _append_unique(
            subqueries,
            [
                "follow up plan monitoring recommendations discharge instructions",
                "postoperative monitoring laboratory surveillance supplementation follow up",
            ],
        )

    if not doc_types:
        doc_types = [
            "discharge_summary",
            "clinic_note",
            "history_and_physical",
            "progress_note",
            "unknown",
        ]
        reasons.append("general clinical question fallback")

    return RetrievalPlan(
        primary_query=q,
        subqueries=subqueries[:8],
        target_document_types=doc_types[:8],
        rationale="; ".join(reasons),
    )
