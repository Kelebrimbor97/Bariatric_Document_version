#!/usr/bin/env python3
from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import torch
import chromadb
from chromadb.config import Settings

import json
from transformers import AutoTokenizer
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


# ---------------------------
# Chunking (token-aware, for 256 context length)
# ---------------------------
@dataclass
class ChunkConfig:
    # Target chunk size should be comfortably below 256 because:
    # - tokenizer adds special tokens
    # - we may add overlap
    chunk_tokens: int = 200
    overlap_tokens: int = 40
    min_chunk_tokens: int = 60


def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras if paras else [text.strip()]


def chunk_text_for_context_length(
    text: str,
    hf_tokenizer,
    cfg: ChunkConfig,
) -> List[str]:
    """
    Chunk so each piece stays <= cfg.chunk_tokens (counted using the HF tokenizer),
    then apply token-overlap between chunks.
    """
    text = normalize_text(text)
    paras = split_paragraphs(text)

    # Precompute tokens per paragraph
    para_toks: List[Tuple[str, int]] = []
    for p in paras:
        n = len(hf_tokenizer.encode(p, add_special_tokens=False))
        para_toks.append((p, n))

    chunks: List[str] = []
    curr: List[str] = []
    curr_n = 0

    def flush():
        nonlocal curr, curr_n
        if not curr:
            return
        c = "\n\n".join(curr).strip()
        if c:
            chunks.append(c)
        curr = []
        curr_n = 0

    for p, n in para_toks:
        # If paragraph alone is too big, split by sentences
        if n > cfg.chunk_tokens:
            flush()
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf: List[str] = []
            buf_n = 0
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                s_n = len(hf_tokenizer.encode(s, add_special_tokens=False))
                if buf and (buf_n + s_n > cfg.chunk_tokens):
                    chunks.append(" ".join(buf).strip())
                    buf = []
                    buf_n = 0
                buf.append(s)
                buf_n += s_n
            if buf:
                chunks.append(" ".join(buf).strip())
            continue

        # Greedy pack paragraphs
        if curr and (curr_n + n > cfg.chunk_tokens):
            flush()
        curr.append(p)
        curr_n += n

    flush()

    # Apply overlap in token space
    if cfg.overlap_tokens > 0 and len(chunks) > 1:
        out: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = out[-1]
            prev_ids = hf_tokenizer.encode(prev, add_special_tokens=False)
            take = min(cfg.overlap_tokens, len(prev_ids))
            overlap_txt = hf_tokenizer.decode(prev_ids[-take:], skip_special_tokens=True)
            merged = (overlap_txt.strip() + "\n\n" + chunks[i]).strip()
            out.append(merged)
        chunks = out

    # Drop very small chunks unless only one
    if len(chunks) > 1:
        filtered = []
        for c in chunks:
            n = len(hf_tokenizer.encode(c, add_special_tokens=False))
            if n >= cfg.min_chunk_tokens:
                filtered.append(c)
        if filtered:
            chunks = filtered

    return chunks


# ---------------------------
# BiomedCLIP embedder (OpenCLIP)
# ---------------------------
class BiomedCLIPTextEmbedder:
    """
    Local-load BiomedCLIP using:
      - checkpoints/open_clip_config.json
      - checkpoints/open_clip_pytorch_model.bin

    This matches the sample pattern:
      1) read open_clip_config.json
      2) register model_cfg into open_clip.factory._MODEL_CONFIGS under a custom model_name
      3) create_model_and_transforms(model_name=..., pretrained=/path/to/bin, image_* preprocess args)
      4) get_tokenizer(model_name)
    """
    def __init__(
        self,
        device: Optional[str] = None,
        ckpt_dir: str | None = None,
        model_name: str = "biomedclip_local",
    ):
        default_weights_dir = Path(
            os.getenv(
                "LLM_WEIGHTS_DIR",
                str(Path(__file__).resolve().parents[3] / "LLM_Weights"),
            )
        )
        if ckpt_dir is None:
            ckpt_dir = os.getenv(
                "BIOMEDCLIP_CKPT_DIR",
                str(default_weights_dir / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
            )

        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        ckpt_dir = Path(ckpt_dir)
        config_path = ckpt_dir / "open_clip_config.json"
        weights_path = ckpt_dir / "open_clip_pytorch_model.bin"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights: {weights_path}")

        # ---- Load config JSON (sample style) ----
        with config_path.open("r") as f:
            config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

        # ---- Register this local model config under a name (sample style) ----
        if (
            not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS
            and config is not None
        ):
            _MODEL_CONFIGS[model_name] = model_cfg

        # ---- Create tokenizer + model from local weights ----
        self.tokenizer = get_tokenizer(model_name)

        self.model, _, _preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=str(weights_path),
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )

        self.model.to(self.device)
        self.model.eval()

        # For chunk sizing / token counting (keep your local HF tokenizer path)
        hf_tokenizer_path = os.getenv(
            "BIOMEDBERT_TOKENIZER_PATH",
            str(default_weights_dir / "BiomedNLP-BiomedBERT-base-uncased-abstract"),
        )
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            hf_tokenizer_path
        )

        self.context_length = 256

    @torch.no_grad()
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        use_fp16: bool = True
    ) -> List[List[float]]:
        all_vecs: List[List[float]] = []
        autocast_ok = (self.device.type == "cuda" and use_fp16)

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            toks = self.tokenizer(batch, context_length=self.context_length).to(self.device)

            if autocast_ok:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self.model.encode_text(toks)
            else:
                feats = self.model.encode_text(toks)

            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_vecs.extend(feats.detach().cpu().float().tolist())

        return all_vecs


# ---------------------------
# Chroma indexer (ID-scoped chunks)
# ---------------------------
class NotesIndexer:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "notes_chunks",
        device: str | None = None,
        ckpt_dir: str | None = None,
        model_name: str = "biomedclip_local",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(collection_name)

        # Pass local-loading args to the embedder
        self.embedder = BiomedCLIPTextEmbedder(
            device=device,
            ckpt_dir=ckpt_dir,
            model_name=model_name,
        )
        
    def has_record(self, record_id: str) -> bool:
        res = self.collection.get(
            where={"record_id": record_id},
            limit=1,
            include=[]
        )
        return bool(res["ids"])


    def upsert_record(
        self,
        record_id: str,
        notes_text: str,
        cfg: ChunkConfig = ChunkConfig(),
        extra_meta: Optional[Dict[str, object]] = None,
    ) -> int:
        chunks = chunk_text_for_context_length(notes_text, self.embedder.hf_tokenizer, cfg)

        ids = [f"{record_id}::c{i:05d}" for i in range(len(chunks))]

        base = {"record_id": record_id}
        if extra_meta:
            # avoid accidental override unless you want it
            for k, v in extra_meta.items():
                if v is not None and v != "":
                    base[k] = v

        metas = [{**base, "chunk_id": i} for i in range(len(chunks))]

        embs = self.embedder.embed_texts(chunks)

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metas,
            embeddings=embs,
        )
        return len(chunks)


    def query_record(self, record_id: str, question: str, top_k: int = 6) -> List[str]:
        q_emb = self.embedder.embed_texts([question])[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={"record_id": record_id},
            include=["documents", "metadatas", "distances"],
        )
        return res["documents"][0] if res.get("documents") else []


# ---------------------------
# Minimal demo
# ---------------------------
if __name__ == "__main__":
    """
    Install hint (from model README commit):
      pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib :contentReference[oaicite:10]{index=10}
    You also need chromadb + torch.
    """

    idx = NotesIndexer(persist_dir="./Data/chroma_dbs/chroma_notes_biomedclip")

    record_id = "123"
    notes = "PAST MEDICAL HISTORY...\n\nASSESSMENT...\n\nPLAN...\n\n(very long note...)"
    n = idx.upsert_record(record_id, notes)
    print(f"Indexed record {record_id} into {n} chunks")

    question = "What is the assessment and plan?"
    top_chunks = idx.query_record(record_id, question, top_k=6)
    print("\n--- Retrieved chunks ---\n")
    for j, c in enumerate(top_chunks, 1):
        print(f"[{j}] {c[:400]}...\n")
