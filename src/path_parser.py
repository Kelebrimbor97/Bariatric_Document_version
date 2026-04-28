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
    "op report": "operative_report",
    "history & physical": "history_and_physical",
    "history and physical": "history_and_physical",
    "h&p": "history_and_physical",
    "progress notes": "progress_note",
    "progress note": "progress_note",
    "psychology clinic note": "psychology_note",
    "psychiatry clinic note": "psychiatry_note",
    "well child visit": "well_child_visit",
    "ed rn triage": "ed_triage",
    "nutrition": "nutrition_note",
    "dietitian": "nutrition_note",
    "dietary": "nutrition_note",
    "medication": "medication_list",
    "medications": "medication_list",
    "lab results": "lab_report",
    "laboratory": "lab_report",
    "pathology": "pathology",
    "radiology": "radiology",
}

# Broad buckets used by the retrieval planner. These are intentionally practical
# rather than exhaustive so they work on private PDF folder names, Synthea exports,
# and MIMIC-style note metadata.
DOCUMENT_TYPE_PRIORITY = [
    "operative_report",
    "discharge_summary",
    "nutrition_note",
    "lab_report",
    "medication_list",
    "clinic_note",
    "history_and_physical",
    "progress_note",
    "radiology",
    "pathology",
    "patient_instructions",
    "unknown",
]

FILENAME_HINTS = [
    (r"discharge", "discharge_summary"),
    (r"operative|\bop\s*report\b|surgery", "operative_report"),
    (r"nutrition|dietitian|dietary", "nutrition_note"),
    (r"lab|laboratory|result", "lab_report"),
    (r"medication|meds|rx", "medication_list"),
    (r"clinic|office|follow.?up|visit", "clinic_note"),
    (r"history|physical|\bhp\b|h&p", "history_and_physical"),
    (r"progress", "progress_note"),
    (r"instruction|education|patient.?information", "patient_instructions"),
    (r"radiology|xray|ct|mri|ultrasound", "radiology"),
    (r"pathology", "pathology"),
]


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
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

        for phrase, note_type in NOTE_TYPE_HINTS.items():
            if phrase in n:
                note_types.append(note_type)

    return {
        "document_families": sorted(set(families)),
        "care_contexts": sorted(set(contexts)),
        "note_type_candidates": sorted(set(note_types)),
    }


def infer_document_type(parts: list[str], file_name: str) -> str:
    haystack = normalize(" ".join(parts + [file_name]))

    for pattern, doc_type in FILENAME_HINTS:
        if re.search(pattern, haystack):
            return doc_type

    tags = infer_path_tags(parts)
    candidates = set(tags.get("note_type_candidates") or [])
    for doc_type in DOCUMENT_TYPE_PRIORITY:
        if doc_type in candidates:
            return doc_type

    families = set(tags.get("document_families") or [])
    if "radiology" in families:
        return "radiology"
    if "pathology" in families:
        return "pathology"
    if "laboratory" in families:
        return "lab_report"

    return "unknown"


def parse_pdf_path(pdf_path: Path, patient_root: Path) -> dict:
    rel = pdf_path.relative_to(patient_root)
    parent_parts = list(rel.parts[:-1])
    path_tags = infer_path_tags(parent_parts)
    document_type = infer_document_type(parent_parts, pdf_path.name)

    return {
        "relative_path": str(rel),
        "path_parts": parent_parts,
        "file_name": pdf_path.name,
        "path_tags": path_tags,
        "document_type": document_type,
    }
