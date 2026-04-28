import re

SECTION_HEADER_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9 /&()\-]{2,}:|[A-Z][A-Za-z0-9 /&()\-]{2,}:)\s*$"
)

COMMON_SECTION_HEADERS = {
    "chief complaint",
    "history of present illness",
    "hpi",
    "past medical history",
    "past surgical history",
    "hospital course",
    "discharge diagnosis",
    "discharge diagnoses",
    "discharge medications",
    "medications",
    "allergies",
    "assessment",
    "assessment and plan",
    "plan",
    "nutrition assessment",
    "nutrition intervention",
    "diet",
    "laboratory data",
    "labs",
    "results",
    "follow up",
    "follow-up",
    "instructions",
    "operative findings",
    "procedure",
    "impression",
    "findings",
    "recommendation",
    "recommendations",
}


def split_into_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if len(paras) <= 1:
        # Some extracted PDFs have single-newline paragraph breaks only.
        paras = [p.strip() for p in re.split(r"\n", text or "") if p.strip()]
    return paras


def looks_like_section_header(line: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (line or "").strip())
    if not cleaned or len(cleaned) > 90:
        return False

    normalized = cleaned.rstrip(":").strip().lower()
    if normalized in COMMON_SECTION_HEADERS:
        return True

    if SECTION_HEADER_RE.match(cleaned):
        return True

    # Numbered headings such as "1. Hospital Course".
    if re.match(r"^\d+(?:\.\d+)*[.)]?\s+[A-Za-z][A-Za-z0-9 /&()\-]{2,}:?$", cleaned):
        return True

    return False


def split_into_sections(text: str) -> list[dict]:
    """Return [{'section_title': str | None, 'text': str}] preserving rough headings."""
    lines = (text or "").splitlines()
    sections: list[dict] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if looks_like_section_header(line):
            if current_lines:
                sections.append(
                    {
                        "section_title": current_title,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
            current_title = line.rstrip(":").strip()
            current_lines = []
        else:
            current_lines.append(raw_line)

    if current_lines:
        sections.append(
            {
                "section_title": current_title,
                "text": "\n".join(current_lines).strip(),
            }
        )

    if not sections and text.strip():
        sections.append({"section_title": None, "text": text.strip()})

    return [s for s in sections if s.get("text", "").strip()]


def chunk_text(text: str, max_chars: int = 1800) -> list[str]:
    paras = split_into_paragraphs(text)
    chunks = []
    current = []

    current_len = 0
    for para in paras:
        if current_len + len(para) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_text_with_sections(text: str, max_chars: int = 1800) -> list[dict]:
    """
    Section-aware chunking for clinical notes/PDF-extracted text.

    Returns records with section metadata:
      {'section_title': str | None, 'chunk_text': str, 'section_chunk_index': int}
    """
    out: list[dict] = []

    for section in split_into_sections(text):
        title = section.get("section_title")
        section_text = section.get("text") or ""
        for idx, chunk in enumerate(chunk_text(section_text, max_chars=max_chars)):
            if title:
                display_text = f"Section: {title}\n\n{chunk}"
            else:
                display_text = chunk
            out.append(
                {
                    "section_title": title,
                    "section_chunk_index": idx,
                    "chunk_text": display_text,
                }
            )

    return out
