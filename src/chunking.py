import re

def split_into_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras

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