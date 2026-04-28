#!/usr/bin/env python3
'''

cd ~/Bariatric
export PYTHONPATH=$PWD/scripts:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_MODEL_NAME="medgemma-27b-it"

python scripts/gradio_bariatric_rag.py
'''
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import chromadb
from chromadb.config import Settings

from openai import OpenAI

# If this import fails, run:
#   export PYTHONPATH=$PWD/scripts:$PYTHONPATH
from scripts.utils.big_chungus import BiomedCLIPTextEmbedder


MRN_KEY = os.environ.get("MRN_KEY", "MRN")
RECORD_ID_KEY = os.environ.get("RECORD_ID_KEY", "record_id")

DEFAULT_COLLECTION = os.environ.get("CHROMA_COLLECTION", "notes_chunks_mrn")
DEFAULT_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "Data/chroma_dbs/chroma_notes_biomedclip")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "/llm_weights/medgemma-27b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")


@dataclass
class AppState:
    collection_name: str
    persist_dir: str
    device: str
    ckpt_dir: str
    model_name: str


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_chroma_collection(persist_dir: str, collection_name: str):
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(collection_name)


def make_vllm_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


def is_mrn_message(text: str) -> Tuple[bool, str, str]:
    """
    Returns (is_mrn, mrn, remainder_question).
    Accepts:
      - mrn:12345
      - mrn=12345
      - just digits (len>=5)
    Also supports: "mrn:12345 what meds is he on?"
    """
    t = (text or "").strip()

    m = re.match(r"(?i)^\s*mrn\s*[:=]\s*([0-9]+)\s*(.*)$", t)
    if m:
        mrn = m.group(1).strip()
        rest = (m.group(2) or "").strip()
        return True, mrn, rest

    if t.isdigit() and len(t) >= 5:
        return True, t, ""

    return False, "", ""


def format_retrieved_notes(items: List[Dict[str, Any]], max_chars_per_note: int) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        md = it.get("meta") or {}
        mrn = md.get(MRN_KEY, "")
        rec = md.get(RECORD_ID_KEY, md.get("record_id", ""))
        date = md.get("note_date", "") or md.get("date", "")
        ntype = md.get("note_type", "") or md.get("type", "")

        header = f"[{i}] MRN={mrn} record={rec}"
        if date:
            header += f" date={date}"
        if ntype:
            header += f" type={ntype}"

        text = (it.get("text") or "")[:max_chars_per_note]

        lines.append(header)
        lines.append(text)
        lines.append("---")
    return "\n".join(lines)


def build_prompt(user_question: str, context: str) -> List[Dict[str, str]]:
    system = (
        "You are a clinical assistant. Use ONLY the provided retrieved notes as evidence.\n"
        "If the notes do not contain the answer, say you cannot find it in the provided notes.\n"
        "When you use information, cite note numbers like [1], [2]."
    )
    user = f"""USER QUESTION:
{user_question}

RETRIEVED NOTES:
{context}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---- BioMedCLIP embedder singleton ----
_EMBEDDER: Optional[BiomedCLIPTextEmbedder] = None


def get_embedder(device: str, ckpt_dir: str, model_name: str) -> BiomedCLIPTextEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = BiomedCLIPTextEmbedder(
            device=device,
            ckpt_dir=ckpt_dir,
            model_name=model_name,
        )
    return _EMBEDDER


def embed_query_biomedclip(text: str, device: str, ckpt_dir: str, model_name: str) -> List[float]:
    emb = get_embedder(device, ckpt_dir, model_name).embed_texts([text])[0]
    return emb.tolist() if hasattr(emb, "tolist") else list(emb)


def retrieve_by_mrn(collection, mrn: str, limit_docs: int) -> List[Dict[str, Any]]:
    res = collection.get(
        where={MRN_KEY: mrn},
        include=["documents", "metadatas"],
    )
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    items = [{"text": d, "meta": md} for d, md in zip(docs, metas)]
    # stable-ish ordering
    items.sort(key=lambda x: (str((x["meta"] or {}).get(RECORD_ID_KEY, "")), int((x["meta"] or {}).get("chunk_id", 0))))
    return items[:limit_docs]


def retrieve_topk(collection, query: str, top_k: int, state: AppState) -> List[Dict[str, Any]]:
    q_emb = embed_query_biomedclip(query, state.device, state.ckpt_dir, state.model_name)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out = []
    for d, md, dist in zip(docs, metas, dists):
        out.append({"text": d, "meta": md, "distance": dist})
    return out


def call_llm(messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    client = make_vllm_client()
    resp = client.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
    )
    return resp.choices[0].message.content


def answer(message: str, mode: str, top_k: int, mrn_limit_chunks: int, max_chars_per_note: int,
           temperature: float, max_tokens: int, state: AppState) -> str:
    collection = make_chroma_collection(state.persist_dir, state.collection_name)

    msg = (message or "").strip()
    if not msg:
        return "Type a question, or `mrn:12345`."

    auto_is_mrn, mrn, remainder_q = is_mrn_message(msg)

    if mode == "Auto":
        use_mrn = auto_is_mrn
    elif mode == "MRN":
        use_mrn = True
        if not auto_is_mrn:
            digits = re.sub(r"\D+", "", msg)
            mrn = digits
            remainder_q = ""
    else:
        use_mrn = False

    if use_mrn:
        if not mrn:
            return "MRN mode selected, but I couldn't parse an MRN. Use `mrn:12345`."
        items = retrieve_by_mrn(collection, mrn=mrn, limit_docs=mrn_limit_chunks)
        if not items:
            return f"No chunks found for MRN={mrn} (metadata key '{MRN_KEY}')."

        user_q = remainder_q if remainder_q else f"Summarize everything clinically relevant for MRN={mrn}."
        context = format_retrieved_notes(items, max_chars_per_note=max_chars_per_note)
        return call_llm(build_prompt(user_q, context), temperature=temperature, max_tokens=max_tokens)

    items = retrieve_topk(collection, query=msg, top_k=top_k, state=state)
    if not items:
        return "No relevant chunks retrieved."

    context = format_retrieved_notes(items, max_chars_per_note=max_chars_per_note)
    return call_llm(build_prompt(msg, context), temperature=temperature, max_tokens=max_tokens)


def main():
    root = get_project_root()

    state = AppState(
        collection_name=DEFAULT_COLLECTION,
        persist_dir=str(root / DEFAULT_PERSIST_DIR),
        device=os.environ.get("EMBED_DEVICE", "cuda:0"),
        ckpt_dir=os.environ.get(
            "BIOMEDCLIP_CKPT_DIR",
            str(
                Path.home()
                / "LLM_Weights"
                / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            ),
        ),
        model_name=os.environ.get("BIOMEDCLIP_MODEL_NAME", "biomedclip_local"),
    )

    with gr.Blocks(title="Bariatric Notes RAG") as demo:
        gr.Markdown(
            """
# Bariatric Notes RAG
- **MRN mode:** type `mrn:12345` (optionally add a question after it).
- **Query mode:** type a natural-language question; retrieval uses top-k semantic search (BioMedCLIP).
"""
        )

        with gr.Row():
            mode = gr.Dropdown(choices=["Auto", "MRN", "Query"], value="Auto", label="Mode")
            top_k = gr.Slider(1, 20, value=5, step=1, label="Top-k (Query mode)")
            mrn_limit = gr.Slider(50, 2000, value=400, step=50, label="MRN chunk limit (MRN mode)")

        with gr.Row():
            max_chars = gr.Slider(200, 4000, value=1800, step=100, label="Max chars per retrieved chunk")
            temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="LLM temperature")
            max_tokens = gr.Slider(128, 2048, value=700, step=64, label="Max output tokens")

        # ✅ Critical fix: tell Gradio we are using tuple history
        chatbot = gr.Chatbot(height=520)

        msg = gr.Textbox(placeholder="Ask a question or type mrn:12345 ...", label="Message")
        clear = gr.Button("Clear")

        def respond(message, history):
            history = history or []
            history.append({"role": "user", "content": message})

            bot_message = answer(
                message=message,
                mode=mode.value,
                top_k=int(top_k.value),
                mrn_limit_chunks=int(mrn_limit.value),
                max_chars_per_note=int(max_chars.value),
                temperature=float(temperature.value),
                max_tokens=int(max_tokens.value),
                state=state,
            )

            history.append({"role": "assistant", "content": bot_message})
            return history


        msg.submit(respond, [msg, chatbot], chatbot)

        clear.click(lambda: [], None, chatbot)

        gr.Markdown(
            f"""
**Chroma:** `{state.persist_dir}` / `{state.collection_name}`  
**Embeddings device:** `{state.device}`  
**vLLM base URL:** `{VLLM_BASE_URL}`  
**vLLM model name:** `{VLLM_MODEL_NAME}`  
"""
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
