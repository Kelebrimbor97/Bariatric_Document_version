from pathlib import Path
import re

DOCUMENT_FAMILY_TERMS = {
    "clinical documents": "clinical_documents",
    "radiology": "radiology",
    "pathology reports": "pathology",
    "laboratory documents": "laboratory",
    "lab": "laboratory",
    "lab view": "laboratory",
    "vital signs": "vitals",
    "patient information": "patient_info",
    "diagnostic studies": "diagnostic_studies",
    "documents": "documents",
}

CARE_CONTEXT_TERMS = {
    "emergency department": "ed",
    "inpatient core": "inpatient",
    "outpatient core": "outpatient",
    "perioperative documents": "perioperative",
    "behavioral health": "behavioral_health",
    "outpatient confidential documents": "outpatient_confidential",
    "inpatient confidential documents": "inpatient_confidential",
}

NOTE_TYPE_HINTS = {
    "discharge summary": "discharge_summary",
    "operative report": "operative_report",
    "history & physical": "history_and_physical",
    "progress notes": "progress_note",
    "psychology clinic note": "psychology_note",
    "psychiatry clinic note": "psychiatry_note",
    "well child visit": "well_child_visit",
    "ed rn triage": "ed_triage",
}

def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def infer_path_tags(parts: list[str]) -> dict:
    families, contexts, note_types = [], [], []

    for p in parts:
        n = normalize(p)
        if n in DOCUMENT_FAMILY_TERMS:
            families.append(DOCUMENT_FAMILY_TERMS[n])
        if n in CARE_CONTEXT_TERMS:
            contexts.append(CARE_CONTEXT_TERMS[n])
        if n in NOTE_TYPE_HINTS:
            note_types.append(NOTE_TYPE_HINTS[n])

    return {
        "document_families": sorted(set(families)),
        "care_contexts": sorted(set(contexts)),
        "note_type_candidates": sorted(set(note_types)),
    }

def parse_pdf_path(pdf_path: Path, patient_root: Path) -> dict:
    rel = pdf_path.relative_to(patient_root)
    parent_parts = list(rel.parts[:-1])

    return {
        "relative_path": str(rel),
        "path_parts": parent_parts,
        "file_name": pdf_path.name,
        "path_tags": infer_path_tags(parent_parts),
    }