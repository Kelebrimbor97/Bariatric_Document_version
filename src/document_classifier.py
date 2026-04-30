from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.path_parser import FILENAME_HINTS


@dataclass
class DocumentClassification:
    document_type: str
    document_type_source: str
    document_type_confidence: float
    document_type_signals: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def _norm(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def classify_from_explicit(document_type: str | None) -> DocumentClassification | None:
    if not document_type:
        return None
    document_type = _norm(document_type)
    if not document_type or document_type == "unknown":
        return None
    return DocumentClassification(
        document_type=document_type,
        document_type_source="explicit_metadata",
        document_type_confidence=1.0,
        document_type_signals=[f"explicit document_type={document_type}"],
    )


def classify_from_filename(file_name: str) -> DocumentClassification | None:
    haystack = _norm(Path(file_name).stem)
    if not haystack:
        return None

    for pattern, doc_type in FILENAME_HINTS:
        if re.search(pattern, haystack):
            return DocumentClassification(
                document_type=doc_type,
                document_type_source="filename",
                document_type_confidence=0.9,
                document_type_signals=[f"filename matched /{pattern}/"],
            )
    return None


def _has_any(text: str, phrases: list[str]) -> list[str]:
    return [phrase for phrase in phrases if phrase in text]


def _regex_hits(text: str, patterns: list[str]) -> list[str]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            hits.append(pattern)
    return hits


def classify_from_content(text: str) -> DocumentClassification | None:
    raw_text = text or ""
    text_norm = _norm(raw_text[:50000])
    if not text_norm:
        return None

    candidates: list[DocumentClassification] = []

    pathology_hits = _has_any(
        text_norm,
        [
            "final diagnosis",
            "gross description",
            "microscopic description",
            "specimen submitted",
            "pathologic diagnosis",
            "surgical pathology",
        ],
    )
    if len(pathology_hits) >= 2 or "surgical pathology" in pathology_hits:
        candidates.append(
            DocumentClassification(
                "pathology",
                "content",
                0.92,
                [f"content phrase: {h}" for h in pathology_hits[:5]],
            )
        )

    radiology_hits = _has_any(
        text_norm,
        ["impression", "findings", "comparison", "indication", "exam", "technique"],
    )
    radiology_modality_hits = _regex_hits(
        text_norm,
        [r"\bct\b", r"\bmri\b", r"\bx[- ]?ray\b", r"\bultrasound\b", r"\bus\b", r"\bportable chest\b", r"\bcontrast\b"],
    )
    if len(radiology_hits) >= 3 and radiology_modality_hits:
        candidates.append(
            DocumentClassification(
                "radiology",
                "content",
                0.88,
                [*(f"content phrase: {h}" for h in radiology_hits[:5]), *(f"modality pattern: {h}" for h in radiology_modality_hits[:3])],
            )
        )

    discharge_hits = _has_any(
        text_norm,
        [
            "discharge diagnosis",
            "discharge diagnoses",
            "discharge medications",
            "hospital course",
            "admission date",
            "discharge date",
            "followup instructions",
            "follow-up instructions",
        ],
    )
    if len(discharge_hits) >= 3:
        candidates.append(
            DocumentClassification(
                "discharge_summary",
                "content",
                0.9,
                [f"content phrase: {h}" for h in discharge_hits[:6]],
            )
        )

    operative_hits = _has_any(
        text_norm,
        [
            "preoperative diagnosis",
            "postoperative diagnosis",
            "procedure performed",
            "operation performed",
            "estimated blood loss",
            "anesthesia",
            "surgeon",
        ],
    )
    if len(operative_hits) >= 3:
        candidates.append(
            DocumentClassification(
                "operative_report",
                "content",
                0.9,
                [f"content phrase: {h}" for h in operative_hits[:6]],
            )
        )

    lab_phrase_hits = _has_any(
        text_norm,
        [
            "reference range",
            "ref range",
            "specimen",
            "collection time",
            "result value",
            "abnormal flag",
            "component",
            "units",
        ],
    )
    lab_value_patterns = _regex_hits(
        raw_text[:50000],
        [
            r"\b\d+(?:\.\d+)?\s*(mg/dl|mmol/l|g/dl|k/u?l|iu/l|u/l|pg/ml|ng/ml|%)\b",
            r"\b(wbc|hgb|hemoglobin|platelet|sodium|potassium|creatinine|glucose|albumin|bilirubin)\b",
        ],
    )
    if len(lab_phrase_hits) >= 2 or (lab_phrase_hits and lab_value_patterns):
        candidates.append(
            DocumentClassification(
                "lab_report",
                "content",
                0.84,
                [*(f"content phrase: {h}" for h in lab_phrase_hits[:5]), *(f"lab pattern: {h}" for h in lab_value_patterns[:3])],
            )
        )

    medication_hits = _regex_hits(
        raw_text[:50000],
        [
            r"\b(medication list|current medications|discharge medications|prescriptions?)\b",
            r"\b(tablet|capsule|injection|nebulizer|neb|oral|iv|subcutaneous|mg|mcg)\b",
            r"\b(daily|bid|tid|qid|qhs|prn|every \d+ hours?)\b",
        ],
    )
    if len(medication_hits) >= 2 and "discharge_summary" not in {c.document_type for c in candidates}:
        candidates.append(
            DocumentClassification(
                "medication_list",
                "content",
                0.78,
                [f"medication pattern: {h}" for h in medication_hits[:5]],
            )
        )

    nutrition_hits = _has_any(
        text_norm,
        ["nutrition", "dietitian", "dietary", "protein", "calories", "bariatric", "vitamin", "supplement", "fluid intake"],
    )
    if len(nutrition_hits) >= 3:
        candidates.append(
            DocumentClassification(
                "nutrition_note",
                "content",
                0.82,
                [f"content phrase: {h}" for h in nutrition_hits[:6]],
            )
        )

    h_and_p_hits = _has_any(
        text_norm,
        ["history of present illness", "past medical history", "chief complaint", "physical exam", "assessment and plan"],
    )
    if len(h_and_p_hits) >= 3:
        candidates.append(
            DocumentClassification(
                "history_and_physical",
                "content",
                0.82,
                [f"content phrase: {h}" for h in h_and_p_hits[:5]],
            )
        )

    progress_hits = _has_any(
        text_norm,
        ["interval history", "subjective", "objective", "assessment", "plan", "progress note"]
    )
    if len(progress_hits) >= 4 and "discharge_summary" not in {c.document_type for c in candidates}:
        candidates.append(
            DocumentClassification(
                "progress_note",
                "content",
                0.74,
                [f"content phrase: {h}" for h in progress_hits[:5]],
            )
        )

    if not candidates:
        return None

    return max(candidates, key=lambda c: c.document_type_confidence)


def classify_from_path_hint(path_document_type: str | None, path_tags: dict[str, Any] | None = None) -> DocumentClassification | None:
    if not path_document_type or path_document_type == "unknown":
        return None
    signals = [f"path parser suggested document_type={path_document_type}"]
    if path_tags:
        candidates = path_tags.get("note_type_candidates") or []
        families = path_tags.get("document_families") or []
        if candidates:
            signals.append(f"path note_type_candidates={candidates}")
        if families:
            signals.append(f"path document_families={families}")
    return DocumentClassification(
        document_type=path_document_type,
        document_type_source="path_hint",
        document_type_confidence=0.55,
        document_type_signals=signals,
    )


def classify_document(
    *,
    file_name: str,
    text: str,
    path_document_type: str | None = None,
    path_tags: dict[str, Any] | None = None,
    explicit_document_type: str | None = None,
    use_path_hints: bool = False,
) -> DocumentClassification:
    """Classify document type using path only as an optional low-confidence fallback.

    Priority:
      explicit source metadata > filename hint > content inference > optional path hint > unknown
    """
    for classifier in (
        lambda: classify_from_explicit(explicit_document_type),
        lambda: classify_from_filename(file_name),
        lambda: classify_from_content(text),
    ):
        result = classifier()
        if result is not None:
            return result

    if use_path_hints:
        path_result = classify_from_path_hint(path_document_type, path_tags)
        if path_result is not None:
            return path_result

    return DocumentClassification(
        document_type="unknown",
        document_type_source="unknown",
        document_type_confidence=0.0,
        document_type_signals=[],
    )
