from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
from tqdm import tqdm

from src.config import PATIENTS_ROOT, PROCESSED_DIR, USE_PATH_HINTS_FOR_DOCUMENT_TYPE
from src.path_parser import parse_pdf_path
from src.pdf_extract import extract_pdf_text
from src.chunking import chunk_text_with_sections
from src.document_classifier import classify_document


ERRORS_FILE = PROCESSED_DIR / "build_ehr_corpus.errors.jsonl"
CHECKPOINT_FILE = PROCESSED_DIR / "build_ehr_corpus.ckpt"


def find_patient_dirs(root: Path):
    for p in root.iterdir():
        if p.is_dir():
            yield p


def make_patient_uid(patient_folder_name: str) -> str:
    digest = hashlib.sha256(patient_folder_name.encode("utf-8")).hexdigest()[:16]
    return f"patient_{digest}"


def extract_actual_patient_id(patient_folder_name: str) -> str:
    if " - " in patient_folder_name:
        return patient_folder_name.split(" - ", 1)[1].strip()
    return patient_folder_name.strip()


def load_processed_pdfs() -> set[str]:
    if not CHECKPOINT_FILE.exists():
        return set()

    seen = set()
    with CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(line)
    return seen


def append_checkpoint(pdf_path: Path) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        f.write(str(pdf_path) + "\n")


def log_error(pdf_path: Path, error: Exception) -> None:
    ERRORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ERRORS_FILE.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "pdf_path": str(pdf_path),
                    "error": str(error),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    documents_out = PROCESSED_DIR / "documents.jsonl"
    chunks_out = PROCESSED_DIR / "chunks.jsonl"

    processed_pdfs = load_processed_pdfs()
    patient_dirs = list(find_patient_dirs(PATIENTS_ROOT))

    print(f"Patients found: {len(patient_dirs)}")
    print(f"Already processed PDFs in checkpoint: {len(processed_pdfs)}")
    print(f"Use path hints for document_type fallback: {USE_PATH_HINTS_FOR_DOCUMENT_TYPE}")

    with documents_out.open("a", encoding="utf-8") as f_doc, \
         chunks_out.open("a", encoding="utf-8") as f_chunk:

        for patient_dir in tqdm(patient_dirs, desc="Patients"):
            patient_folder_name = patient_dir.name
            patient_uid = make_patient_uid(patient_folder_name)
            actual_patient_id = extract_actual_patient_id(patient_folder_name)

            for pdf_path in patient_dir.rglob("*.pdf"):
                if pdf_path.name.startswith("._") or pdf_path.name.startswith("."):
                    continue
                pdf_path_str = str(pdf_path)

                if pdf_path_str in processed_pdfs:
                    continue

                try:
                    path_meta = parse_pdf_path(pdf_path, patient_dir)
                    path_document_type = path_meta.pop("document_type", "unknown")
                    pages = extract_pdf_text(pdf_path)
                    full_text = "\n\n".join(
                        p["text"] for p in pages if p["text"].strip()
                    )
                    classification = classify_document(
                        file_name=pdf_path.name,
                        text=full_text,
                        path_document_type=path_document_type,
                        path_tags=path_meta.get("path_tags"),
                        use_path_hints=USE_PATH_HINTS_FOR_DOCUMENT_TYPE,
                    )
                    classification_meta = classification.to_metadata()

                    doc_record = {
                        "patient_id": patient_uid,
                        "actual_patient_id": actual_patient_id,
                        "patient_folder_name": patient_folder_name,
                        "pdf_path": pdf_path_str,
                        **path_meta,
                        **classification_meta,
                        "path_document_type_hint": path_document_type,
                        "n_pages": len(pages),
                        "raw_text": full_text,
                    }
                    f_doc.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

                    for page in pages:
                        page_chunks = chunk_text_with_sections(page["text"])
                        for idx, ch in enumerate(page_chunks):
                            section_title = ch.get("section_title")
                            section_key = section_title or "none"
                            chunk_record = {
                                "patient_id": patient_uid,
                                "actual_patient_id": actual_patient_id,
                                "patient_folder_name": patient_folder_name,
                                "pdf_path": pdf_path_str,
                                **path_meta,
                                **classification_meta,
                                "path_document_type_hint": path_document_type,
                                "page_num": page["page_num"],
                                "section_title": section_title,
                                "section_chunk_index": ch.get("section_chunk_index"),
                                "chunk_id": f"{patient_uid}::{pdf_path.stem}::p{page['page_num']}::s{section_key}::c{idx}",
                                "chunk_text": ch["chunk_text"],
                            }
                            f_chunk.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

                    f_doc.flush()
                    f_chunk.flush()

                    append_checkpoint(pdf_path)
                    processed_pdfs.add(pdf_path_str)

                except Exception as e:
                    import traceback
                    print(f"[WARN] Skipping bad PDF: {pdf_path}")
                    print(f"       Reason: {e}")
                    traceback.print_exc()
                    log_error(pdf_path, e)
                    append_checkpoint(pdf_path)
                    processed_pdfs.add(pdf_path_str)
                    continue

    print(f"Wrote/appended: {documents_out}")
    print(f"Wrote/appended: {chunks_out}")
    print(f"Errors logged to: {ERRORS_FILE}")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
