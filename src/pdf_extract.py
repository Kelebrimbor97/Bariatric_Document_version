from pathlib import Path
from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """
    Extract text page-by-page from a PDF.

    Returns:
        [
            {"page_num": 1, "text": "..."},
            ...
        ]

    Behavior:
    - If the whole PDF cannot be opened, raises RuntimeError.
    - If an individual page fails, that page is skipped and replaced with empty text.
    """
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Could not open PDF {pdf_path}: {e}") from e

    pages = []

    for page_num in range(len(reader.pages)):
        try:
            page_obj = reader.pages[page_num]
            text = page_obj.extract_text() or ""
        except Exception as e:
            text = ""
            # Keep the page entry so downstream numbering stays stable
            # but do not crash the whole file on one bad page.
            print(f"[WARN] Failed to extract page {page_num + 1} from {pdf_path}: {e}")

        pages.append(
            {
                "page_num": page_num + 1,
                "text": text.strip(),
            }
        )

    return pages