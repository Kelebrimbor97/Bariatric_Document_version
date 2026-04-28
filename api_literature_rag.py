from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import json
import re
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

import src.config as cfg


NCBI_EUTILS_BASE_URL = getattr(
    cfg,
    "NCBI_EUTILS_BASE_URL",
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
)
NCBI_EMAIL = getattr(cfg, "NCBI_EMAIL", "your_email@example.com")
NCBI_TOOL = getattr(cfg, "NCBI_TOOL", "local_literature_rag")
NCBI_API_KEY = getattr(cfg, "NCBI_API_KEY", None)

VLLM_BASE_URL = cfg.VLLM_BASE_URL
VLLM_MODEL_NAME = cfg.VLLM_MODEL_NAME
ENCODER_API_URL = cfg.ENCODER_API_URL

LITERATURE_CACHE_DIR = getattr(
    cfg,
    "LITERATURE_CACHE_DIR",
    PROJECT_ROOT / "Data" / "literature_cache",
)

PROPOSALS_FILE = LITERATURE_CACHE_DIR / "query_proposals.jsonl"
PMC_ARTICLE_CACHE_DIR = LITERATURE_CACHE_DIR / "pmc_articles"

LITERATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PMC_ARTICLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Literature Approval RAG API",
    version="0.1.0",
    description=(
        "Two-step user-approved PMC full-text grounding API. "
        "First proposes a sanitized literature query, then executes only after approval."
    ),
)


class ProposeQueryRequest(BaseModel):
    user_question: str = Field(..., description="User's original question.")
    chart_context: Optional[str] = Field(
        None,
        description="Optional locally generated chart context. Must not be sent externally.",
    )


class ProposeQueryResponse(BaseModel):
    query_id: str
    proposed_query: str
    phi_risk: str
    removed_terms: list[str]
    will_call_external_api: bool
    requires_user_approval: bool
    approval_phrase: str
    message_to_user: str


class ExecuteQueryRequest(BaseModel):
    query_id: str
    approval_phrase: str
    retmax: int = Field(10, description="Number of PMC articles to retrieve before chunk reranking.")
    final_k: int = Field(8, description="Number of article chunks to send to the LLM.")


class LiteratureSource(BaseModel):
    pmcid: str | None = None
    pmid: str | None = None
    doi: str | None = None
    title: str | None = None
    journal: str | None = None
    year: str | None = None
    section_title: str | None = None
    url: str | None = None
    score: float | None = None


class ExecuteQueryResponse(BaseModel):
    query_id: str
    proposed_query: str
    answer: str
    sources: list[LiteratureSource]

class ApproveAndExecuteRequest(BaseModel):
    approval_phrase: str = Field(
        ...,
        description=(
            "The exact approval phrase provided by the user, e.g. "
            "'APPROVE litq_abc123'. Must be copied exactly from the user message."
        ),
    )
    retmax: int = Field(
        10,
        description="Number of PMC articles to retrieve before chunk reranking.",
    )
    final_k: int = Field(
        5,
        description="Number of article chunks to send to the LLM.",
    )

def parse_query_id_from_approval_phrase(approval_phrase: str) -> str:
    m = re.search(r"\bAPPROVE\s+(litq_[A-Za-z0-9]+)\b", approval_phrase.strip())
    if not m:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid approval phrase. Expected exact format like: "
                "APPROVE litq_abc123"
            ),
        )
    return m.group(1)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def get_llm_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")


def ncbi_params(extra: dict) -> dict:
    params = {
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    params.update(extra)
    return params


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def clean_query_syntax(query: str) -> str:
    """
    Generic syntax cleanup only.
    Does not add biomedical concepts or rewrite medical meaning.
    """
    query = query or ""

    # Remove leading/trailing punctuation artifacts left by PHI scrubbing
    query = query.strip()
    query = re.sub(r"^[\s,;:.\-–—]+", "", query)
    query = re.sub(r"[\s,;:.\-–—]+$", "", query)

    # Remove awkward punctuation spacing
    query = re.sub(r"\s+([,;:.])", r"\1", query)
    query = re.sub(r"([,;:.]){2,}", r"\1", query)

    # Remove empty comma artifacts like ", ," or " , "
    query = re.sub(r"\s*,\s*,\s*", ", ", query)
    query = re.sub(r"^\s*,\s*", "", query)

    # Collapse whitespace
    query = re.sub(r"\s+", " ", query).strip()

    return query

def clean_literature_query_text(query: str) -> str:
    query = query or ""

    bad_phrases = [
        "what does the literature say about",
        "what does pubmed say about",
        "what does pmc say about",
        "for patient",
        "tell me about",
        "summarize",
        "evidence for",
        "external literature query about",
        "literature query about",
        "research on",
    ]

    q = query.lower()
    for phrase in bad_phrases:
        q = q.replace(phrase, " ")

    q = re.sub(r"\bpatient\b", " ", q)
    q = re.sub(r"\bthe literature\b", " ", q)
    q = re.sub(r"\bwhat\b|\bdoes\b|\bsay\b|\babout\b", " ", q)
    q = re.sub(r"[?]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    return q


def extract_json_object(text: str) -> dict:
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}

    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def rule_based_phi_scrub(text: str) -> tuple[str, list[str], list[str]]:
    """
    Conservative scrubber for things that should not be sent to external literature APIs.

    Returns:
        cleaned_text, removed_terms, risk_flags
    """

    removed = []
    risk_flags = []
    cleaned = text or ""

    patterns = [
        ("hashed_patient_id", r"\bpatient_[0-9a-fA-F]{8,}\b"),
        ("mrn_like_label", r"\b(MRN|medical record number|patient id|patient_id)\s*[:#]?\s*[A-Za-z0-9_-]+\b"),
        ("folder_test_id", r"\bTest\s*\d+[A-Za-z]?\s*-\s*\d+\b"),
        ("long_numeric_identifier", r"\b\d{7,12}\b"),
        ("dob_label", r"\b(DOB|date of birth)\s*[:#]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
        ("full_date", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    ]

    for label, pat in patterns:
        for m in re.finditer(pat, cleaned, flags=re.IGNORECASE):
            removed.append(m.group(0))
            risk_flags.append(label)
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)


    # Clean punctuation artifacts left after removing identifiers
    cleaned = re.sub(r"\bfor patient\s*[,;:.-]*\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpatient\s*[,;:.-]*\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,;:.])", r"\1", cleaned)
    cleaned = re.sub(r"^[\s,;:.\-–—]+", "", cleaned)
    cleaned = re.sub(r"[\s,;:.\-–—]+$", "", cleaned)
    cleaned = normalize_space(cleaned)
    return cleaned, sorted(set(removed)), sorted(set(risk_flags))


def llm_make_sanitized_literature_query(user_question: str, chart_context: str | None) -> dict:
    """
    Local-only LLM sanitizer. This does not call external APIs.
    """

    scrubbed_question, removed_q, flags_q = rule_based_phi_scrub(user_question)
    scrubbed_context, removed_c, flags_c = rule_based_phi_scrub(chart_context or "")

    prompt = f"""
/no_think

You are creating a de-identified biomedical search query for PMC/PubMed full-text retrieval.

Task:
Convert the user's request and optional chart context into a concise biomedical literature search query.

Rules:
- Return ONLY valid JSON.
- Do not include markdown.
- Do not answer the medical question.
- Do not include patient names, MRNs, patient IDs, folder names, exact dates, institutions, provider names, or note excerpts.
- Do not include rare identifying details.
- Remove conversational framing such as asking what the literature says, asking for a summary, or referring to a specific patient.
- Keep only general biomedical concepts, diagnoses, procedures, medications, outcomes, population terms, and follow-up/monitoring concepts that are directly implied by the input.
- Use standard biomedical terminology when possible.
- You may expand shorthand or informal clinical wording into standard biomedical terms, but only when the expansion is clearly implied by the input.
- Do not add unrelated concepts.
- The query should be suitable for PMC/PubMed search.
- If the input is too patient-specific to safely generalize, set phi_risk to "high".

Return schema:
{{
  "proposed_query": string,
  "phi_risk": "low" | "medium" | "high",
  "reason": string
}}

User question after rule-based PHI scrub:
{scrubbed_question}

Optional chart context after rule-based PHI scrub:
{scrubbed_context[:3000]}
"""

    try:
        llm = get_llm_client()
        resp = llm.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        data = extract_json_object(resp.choices[0].message.content or "")
    except Exception as e:
        data = {
            "proposed_query": scrubbed_question,
            "phi_risk": "medium",
            "reason": f"local LLM sanitizer failed; used rule-based scrub only: {e}",
        }

    proposed_query = clean_query_syntax(
        normalize_space(str(data.get("proposed_query") or scrubbed_question))
    )
    proposed_query = clean_literature_query_text(proposed_query)
    phi_risk = str(data.get("phi_risk") or "medium").lower()
    if phi_risk not in {"low", "medium", "high"}:
        phi_risk = "medium"

    removed_terms = sorted(set(removed_q + removed_c))
    risk_flags = sorted(set(flags_q + flags_c))

    if risk_flags and phi_risk == "low":
        phi_risk = "medium"

    # Final safety pass on the proposed query too
    final_query, removed_final, flags_final = rule_based_phi_scrub(proposed_query)
    removed_terms = sorted(set(removed_terms + removed_final))
    risk_flags = sorted(set(risk_flags + flags_final))
    proposed_query = clean_literature_query_text(final_query)

    if risk_flags and phi_risk == "low":
        phi_risk = "medium"

    if not proposed_query:
        phi_risk = "high"

    return {
        "proposed_query": proposed_query,
        "phi_risk": phi_risk,
        "removed_terms": removed_terms,
        "risk_flags": risk_flags,
        "reason": str(data.get("reason", ""))[:500],
    }


def append_proposal(record: dict) -> None:
    with PROPOSALS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_proposal(query_id: str) -> dict | None:
    if not PROPOSALS_FILE.exists():
        return None

    found = None
    with PROPOSALS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("query_id") == query_id:
                found = obj
    return found


def pmc_search_ids(query: str, retmax: int) -> list[str]:
    url = f"{NCBI_EUTILS_BASE_URL}/esearch.fcgi"
    params = ncbi_params(
        {
            "db": "pmc",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        }
    )

    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


def article_cache_path(pmc_numeric_id: str) -> Path:
    return PMC_ARTICLE_CACHE_DIR / f"pmc_{pmc_numeric_id}.json"


def get_text(node) -> str:
    if node is None:
        return ""
    return normalize_space(" ".join("".join(node.itertext()).split()))


def parse_pmc_xml(xml_text: str, pmc_numeric_id: str) -> dict:
    root = ET.fromstring(xml_text)

    article = root.find(".//article")
    if article is None:
        article = root

    title = get_text(article.find(".//front/article-meta/title-group/article-title"))
    journal = get_text(article.find(".//front/journal-meta/journal-title-group/journal-title"))

    pmcid = None
    pmid = None
    doi = None

    for id_node in article.findall(".//front/article-meta/article-id"):
        id_type = id_node.attrib.get("pub-id-type")
        val = get_text(id_node)
        if id_type == "pmc":
            pmcid = val
        elif id_type == "pmid":
            pmid = val
        elif id_type == "doi":
            doi = val

    if pmcid and not pmcid.upper().startswith("PMC"):
        pmcid_display = f"PMC{pmcid}"
    elif pmcid:
        pmcid_display = pmcid
    else:
        pmcid_display = f"PMC{pmc_numeric_id}"

    year = get_text(article.find(".//front/article-meta/pub-date/year"))

    abstract = get_text(article.find(".//front/article-meta/abstract"))

    sections = []

    if abstract:
        sections.append(
            {
                "section_title": "Abstract",
                "text": abstract,
            }
        )

    body = article.find(".//body")
    if body is not None:
        for sec in body.findall(".//sec"):
            section_title = get_text(sec.find("./title")) or "Body"
            paragraphs = []
            for p in sec.findall("./p"):
                txt = get_text(p)
                if txt:
                    paragraphs.append(txt)
            combined = "\n\n".join(paragraphs).strip()
            if len(combined) >= 150:
                sections.append(
                    {
                        "section_title": section_title,
                        "text": combined,
                    }
                )

    return {
        "pmc_numeric_id": pmc_numeric_id,
        "pmcid": pmcid_display,
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "journal": journal,
        "year": year,
        "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid_display}/",
        "sections": sections,
    }


def fetch_pmc_article(pmc_numeric_id: str) -> dict | None:
    cache_path = article_cache_path(pmc_numeric_id)
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    url = f"{NCBI_EUTILS_BASE_URL}/efetch.fcgi"
    params = ncbi_params(
        {
            "db": "pmc",
            "id": pmc_numeric_id,
            "retmode": "xml",
        }
    )

    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()

    try:
        parsed = parse_pmc_xml(r.text, pmc_numeric_id)
    except Exception:
        return None

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    return parsed


def chunk_text(text: str, max_chars: int = 2200) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current and current_len + len(para) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def embed_query(text: str) -> list[float]:
    r = requests.post(
        f"{ENCODER_API_URL}/embed/query",
        json={"texts": [text]},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["vectors"][0]


def embed_articles(texts: list[str]) -> list[list[float]]:
    r = requests.post(
        f"{ENCODER_API_URL}/embed/article",
        json={"texts": texts},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["vectors"]


def dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def build_candidate_chunks(articles: list[dict], max_chunks_total: int = 120) -> list[dict]:
    candidates = []

    for article in articles:
        for sec in article.get("sections", []):
            section_title = sec.get("section_title") or "Section"
            for idx, chunk in enumerate(chunk_text(sec.get("text") or "")):
                if len(chunk.strip()) < 120:
                    continue
                candidates.append(
                    {
                        "pmc_numeric_id": article.get("pmc_numeric_id"),
                        "pmcid": article.get("pmcid"),
                        "pmid": article.get("pmid"),
                        "doi": article.get("doi"),
                        "title": article.get("title"),
                        "journal": article.get("journal"),
                        "year": article.get("year"),
                        "url": article.get("url"),
                        "section_title": section_title,
                        "chunk_index": idx,
                        "text": chunk,
                    }
                )

    return candidates[:max_chunks_total]


def rerank_chunks(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    qvec = embed_query(query)
    texts = [
        f"{c.get('title') or ''}\nSection: {c.get('section_title')}\n\n{c.get('text') or ''}"
        for c in candidates
    ]
    avecs = embed_articles(texts)

    scored = []
    for c, avec in zip(candidates, avecs):
        item = dict(c)
        item["score"] = dot(qvec, avec)
        scored.append(item)

    scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return scored


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/literature/propose_query",
    response_model=ProposeQueryResponse,
    operation_id="propose_literature_query",
    summary="Propose a de-identified literature search query for user approval",
    description=(
        "Creates a sanitized literature query but does not call external APIs. "
        "Returns an approval phrase that the user must copy exactly before execution."
    ),
)
def propose_query(req: ProposeQueryRequest):
    proposal = llm_make_sanitized_literature_query(
        user_question=req.user_question,
        chart_context=req.chart_context,
    )

    query_id = f"litq_{uuid.uuid4().hex[:12]}"
    approval_phrase = f"APPROVE {query_id}"

    will_call_external_api = proposal["phi_risk"] != "high" and bool(proposal["proposed_query"])
    requires_user_approval = True

    record = {
        "query_id": query_id,
        "created_at": now_iso(),
        "user_question_hash": stable_hash(req.user_question),
        "proposed_query": proposal["proposed_query"],
        "phi_risk": proposal["phi_risk"],
        "removed_terms": proposal["removed_terms"],
        "risk_flags": proposal["risk_flags"],
        "reason": proposal["reason"],
        "approval_phrase": approval_phrase,
        "will_call_external_api": will_call_external_api,
    }
    append_proposal(record)

    if not will_call_external_api:
        msg = (
            "I could not create a sufficiently safe de-identified external literature query. "
            f"Risk: {proposal['phi_risk']}. Please rephrase as a general biomedical literature question."
        )
    else:
        msg = (
            "I can search PMC full-text articles using this de-identified query:\n\n"
            f"{proposal['proposed_query']}\n\n"
            f"PHI risk: {proposal['phi_risk']}.\n"
            f"Removed terms: {proposal['removed_terms'] or 'none'}.\n\n"
            f"To approve the external PMC search, reply exactly:\n{approval_phrase}"
        )

    return ProposeQueryResponse(
        query_id=query_id,
        proposed_query=proposal["proposed_query"],
        phi_risk=proposal["phi_risk"],
        removed_terms=proposal["removed_terms"],
        will_call_external_api=will_call_external_api,
        requires_user_approval=requires_user_approval,
        approval_phrase=approval_phrase,
        message_to_user=msg,
    )


@app.post(
    "/literature/execute_query",
    response_model=ExecuteQueryResponse,
    operation_id="execute_approved_literature_query",
    summary="Execute a previously approved literature query",
    description=(
        "Use this only after the user has provided the exact approval phrase. "
        "Requires both query_id and approval_phrase."
    ),
)
def execute_query(req: ExecuteQueryRequest):
    proposal = load_proposal(req.query_id)

    if not proposal:
        raise HTTPException(status_code=404, detail="Unknown query_id.")

    if not proposal.get("will_call_external_api"):
        raise HTTPException(
            status_code=403,
            detail="This query was not approved for external API execution.",
        )

    if proposal.get("phi_risk") == "high":
        raise HTTPException(
            status_code=403,
            detail="PHI risk too high for external query.",
        )

    expected_phrase = proposal.get("approval_phrase")
    if req.approval_phrase.strip() != expected_phrase:
        raise HTTPException(
            status_code=403,
            detail=f"Approval phrase mismatch. Expected: {expected_phrase}",
        )

    query = proposal["proposed_query"]

    pmc_ids = pmc_search_ids(query, retmax=req.retmax)
    time.sleep(0.12)

    articles = []
    for pmc_id in pmc_ids:
        try:
            article = fetch_pmc_article(pmc_id)
            if article and article.get("sections"):
                articles.append(article)
        except Exception:
            continue
        time.sleep(0.12)

    candidates = build_candidate_chunks(articles)
    reranked = rerank_chunks(query, candidates)
    top = reranked[: req.final_k]

    if not top:
        return ExecuteQueryResponse(
            query_id=req.query_id,
            proposed_query=query,
            answer=(
                "NO_USABLE_LITERATURE_EVIDENCE_RETRIEVED. "
                "No usable PMC full-text article chunks were retrieved for the approved query. "
                "Do not answer this biomedical literature question from memory. "
                "Ask the user whether they want to propose and approve a broader de-identified literature query."
            ),
            sources=[],
        )

    evidence_blocks = []
    sources = []

    for i, c in enumerate(top, start=1):
        evidence_blocks.append(
            f"[{i}] PMCID={c.get('pmcid')} PMID={c.get('pmid')} score={c.get('score', 0.0):.4f}\n"
            f"Title: {c.get('title')}\n"
            f"Journal: {c.get('journal')} Year: {c.get('year')}\n"
            f"Section: {c.get('section_title')}\n"
            f"URL: {c.get('url')}\n"
            f"Text: {c.get('text')}"
        )
        sources.append(
            LiteratureSource(
                pmcid=c.get("pmcid"),
                pmid=c.get("pmid"),
                doi=c.get("doi"),
                title=c.get("title"),
                journal=c.get("journal"),
                year=c.get("year"),
                section_title=c.get("section_title"),
                url=c.get("url"),
                score=c.get("score"),
            )
        )

    prompt = f"""
You are answering a biomedical question using retrieved PMC full-text article evidence.

Rules:
- Use only the article evidence below.
- Cite claims using bracketed evidence numbers like [1], [2].
- Do not invent papers, PMIDs, PMCIDs, study results, or citations.
- Distinguish strong evidence from limited, indirect, or mixed evidence.
- Be cautious and clinically precise.

Original approved literature query:
{query}

Evidence:
{chr(10).join(evidence_blocks)}
"""

    llm = get_llm_client()
    resp = llm.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    return ExecuteQueryResponse(
        query_id=req.query_id,
        proposed_query=query,
        answer=resp.choices[0].message.content,
        sources=sources,
    )

@app.post(
    "/literature/approve_and_execute",
    response_model=ExecuteQueryResponse,
    operation_id="approve_and_execute_literature_query",
    summary="Execute a literature query after the user provides the exact approval phrase",
    description=(
        "Use this only after the user explicitly replies with the exact approval phrase. "
        "The approval phrase should look like 'APPROVE litq_abc123'. "
        "This endpoint parses the query_id from the approval phrase and executes only "
        "the previously proposed, approved de-identified literature query."
    ),
)
def approve_and_execute(req: ApproveAndExecuteRequest):
    query_id = parse_query_id_from_approval_phrase(req.approval_phrase)

    return execute_query(
        ExecuteQueryRequest(
            query_id=query_id,
            approval_phrase=req.approval_phrase,
            retmax=req.retmax,
            final_k=req.final_k,
        )
    )