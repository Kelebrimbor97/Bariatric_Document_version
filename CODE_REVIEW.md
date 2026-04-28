# Code Review (2026-04-28)

## Scope reviewed
- API entrypoints: `api_encoder.py`, `api_ehr_rag.py`, `api_literature_rag.py`
- Core modules: `src/*`
- Operational scripts: `scripts/*`
- Runtime/deploy files: `docker-compose.yml`, `start_services.sh`, `run_build.sh`, `restart_services.sh`, `stop_services.sh`, `README.md`

## Architecture assessment

### What is good
1. **Clear pipeline separation**: ingest/chunk (`scripts/build_ehr_corpus.py`) -> index (`scripts/index_qdrant_medcpt.py`) -> retrieve/rerank/answer (`src/ehr_rag_service.py`, `api_ehr_rag.py`).
2. **Reasonable modular boundaries**: extraction (`src/pdf_extract.py`), chunking (`src/chunking.py`), vector store helper (`src/qdrant_store.py`), and path semantics (`src/path_parser.py`) are isolated and reusable.
3. **Operational resilience**: both build/index scripts use checkpoints to support resume behavior.
4. **Service split is sensible**: encoder service is isolated (`api_encoder.py`) and called over HTTP by retrieval (`src/encoder_client.py`), reducing duplicate model memory load.

### Structural issues and recommended changes

#### 1) Configuration is hard-coded and environment-specific (High)
- `src/config.py` hardcodes absolute host paths (`/home/nishad/...`) and localhost endpoints.
- This blocks portability to other hosts/containers and makes testing difficult.

**Recommended change**
- Introduce a settings layer (e.g., pydantic-settings or `os.getenv`) with sensible defaults.
- Keep required paths/env vars in `.env.example` and document them.
- Add validation at startup (missing model dirs, missing data root).

#### 2) Retrieval service has dead/duplicated paths (Medium)
- `src/ehr_rag_service.py` defines `get_query_encoder()` using `MedCPTEncoder` but actual query encoding uses `embed_query_texts()` via HTTP.
- Similar reranking logic is duplicated in `scripts/ask_ehr_rag.py`.

**Recommended change**
- Remove or use `get_query_encoder()` consistently.
- Extract shared retrieval/rerank code into one module consumed by API and CLI.

#### 3) API model defaults use mutable list literal (Medium)
- `api_ehr_rag.py` uses `sources: list[SourceItem] = []`.

**Recommended change**
- Use `Field(default_factory=list)` to avoid shared mutable defaults.

#### 4) Error handling and observability are inconsistent (Medium)
- Build script prints traceback inline and appends errors but lacks structured logging.
- API modules use bare runtime exceptions without explicit HTTP mapping in some paths.

**Recommended change**
- Add module-level logger configuration and consistent error envelopes.
- Convert expected external failures (encoder unavailable, qdrant unavailable, llm unavailable) into clear 5xx/503 responses.

#### 5) Data model could be formalized (Low/Medium)
- JSONL payload schemas are implicit and spread across scripts.

**Recommended change**
- Add typed models (Pydantic/dataclass) for `document` and `chunk` records.
- Validate required fields before indexing.

#### 6) Device placement is rigid (Low/Medium)
- Multiple modules pin to `cuda:1` directly.

**Recommended change**
- Make device configurable via env (`EMBED_DEVICE`, `RERANK_DEVICE`) with fallback order.

## Candidate scripts likely unrelated or legacy

These look out-of-scope for the current MedCPT + Qdrant EHR workflow and likely represent older experiments:

1. **`scripts/bariatric_biomedclip_chromadb.py`**
   - Uses ChromaDB + BioMedCLIP path, not the Qdrant + MedCPT stack.
   - Imports `Document_version.scripts.utils.big_chungus`, suggesting an older package layout assumption.

2. **`scripts/gradio_bariatric_rag.py`**
   - Gradio app over ChromaDB/BioMedCLIP with default model `medgemma-27b-it`; diverges from main README architecture.

3. **`scripts/utils/big_chungus.py`**
   - Utility module only used by the two Chroma/BioMedCLIP scripts above.

4. **`scripts/data_wrangling/jsonl_former.py`**
   - Hardcoded absolute paths outside this repo tree and single-patient export behavior.

5. **`notebooks/*.ipynb`**
   - Exploratory artifacts; not part of operational build/index/query flow.

## Suggested cleanup plan
1. Move legacy/experimental files to `archive/legacy_chroma_biomedclip/`.
2. Keep a short migration note in README explaining active vs legacy paths.
3. Add CI checks:
   - `python -m compileall`
   - formatter/lint (`ruff`, `black --check`)
   - minimal import smoke test for APIs.
4. Add a `scripts/` README table with status tags: `active`, `ops`, `legacy`, `experimental`.
