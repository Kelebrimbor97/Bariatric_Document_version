from pathlib import Path
import json
from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages).strip()
    except Exception as e:
        return f"[ERROR extracting text: {e}]"


def insert_into_tree(tree: dict, relative_parts: tuple, value: str):
    """
    Insert value into nested dict using path parts.
    Example:
      ('a', 'b', 'file.pdf') -> tree['a']['b']['file.pdf'] = value
    """
    current = tree
    for part in relative_parts[:-1]:
        current = current.setdefault(part, {})
    current[relative_parts[-1]] = value


def build_pdf_json_structure(root_folder: str) -> dict:
    root = Path(root_folder)
    result = {}

    for pdf_path in root.rglob("*.pdf"):
        rel_path = pdf_path.relative_to(root)
        pdf_text = extract_pdf_text(pdf_path)
        insert_into_tree(result, rel_path.parts, pdf_text)

    return result


if __name__ == "__main__":
    root_folder = "/home/nishad/Bariatric/Data/MBS LLM/Test Patients/Test 4 - 101150422"
    output_json = "/home/nishad/Bariatric/Data/MBS LLM/Test Patients/Test 4 - 101150422.json"

    data = build_pdf_json_structure(root_folder)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON to {output_json}")