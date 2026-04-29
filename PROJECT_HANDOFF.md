# Bariatric Document RAG Project — Handoff Notes

_Last updated in chat: 2026-04-29_

This file is intended to be pasted or uploaded into a new ChatGPT chat so the next chat can continue without needing the oversized previous thread.

---

## 1. Repository and branch

**GitHub repo**

```text
https://github.com/Kelebrimbor97/Bariatric_Document_version
```

**Working branch**

```text
feature/public-ehr-rag-testbed
```

As of the last verification in this chat, this branch was:

```text
22 commits ahead of main
0 commits behind main
```

**Important instruction for future chats**

When working on this repo, ask before touching the repo unless the user explicitly says to connect to the repo/Codex/GitHub. The user prefers “baby steps”: if the code change is tiny, give exact edit instructions so the user can do it locally; if larger, make a small isolated repo change and verify it.

---

## 2. Big-picture goal

The project is a local clinical-document RAG system for bariatric/EHR-style chart review.

The system should eventually support:

1. Local patient-document ingestion from messy PDF folders.
2. Section/document-type-aware chunking.
3. Dense retrieval using MedCPT/Qdrant.
4. Reranking using MedCPT cross-encoder.
5. Keyword/BM25 retrieval for exact clinical terms.
6. Structured evidence-backed answers with `found`, `not_found`, `uncertain`, and `inferred_from_evidence`.
7. A future public/synthetic testbed before relying on private bariatric data.

The immediate working goal in this branch has been to implement CLI-RAG/EHR-RAG-inspired retrieval improvements without fully reproducing either paper.

---

## 3. Papers / concepts being adapted

The relevant papers are:

```text
EHR-RAG - https://arxiv.org/abs/2601.21340
CLI-RAG - https://arxiv.org/abs/2507.06715
```

Important clarification from the chat:

- **EHR-RAG** here refers to arXiv `2601.21340`, not an older PubMed EHR summarization RAG paper.
- **CLI-RAG** here refers to arXiv `2507.06715`.

### Adapted from CLI-RAG

Useful parts for this project:

- clinical document/note-type awareness
- hierarchical/section-aware chunking
- global/local-ish retrieval
- task-specific subqueries
- temporal/metadata-aware prompting later

Not trying to fully reproduce:

- SOAP/progress-note generation benchmark
- exact MIMIC-III experimental setup
- 15-note-type CLI-RAG benchmark

### Adapted from EHR-RAG

Useful parts for this project:

- long-horizon clinical evidence retrieval
- temporal/event-aware retrieval later
- factual/counterfactual reasoning later
- adaptive iterative retrieval later

Not trying to fully reproduce:

- full structured-EHR prediction pipeline
- Macro-F1 prediction benchmark
- complete event-sequence architecture

### Engineering stack

The actual implementation uses:

- FastAPI APIs
- Qdrant vector database
- MedCPT article/query encoders
- MedCPT cross-encoder reranker
- vLLM/Qwen as local LLM
- OpenWebUI as UI/tool caller
- optional Literature-RAG with approval workflow

---

## 4. Existing environment assumptions

From prior project context:

```text
Repo root:
  /home/nishad/Bariatric/Bariatric_Document_version

Conda env:
  ehr_rag

OpenWebUI:
  http://localhost:8080

vLLM/Qwen:
  http://localhost:8000/v1

Qdrant:
  http://localhost:6333

EHR RAG API:
  http://localhost:8090

Encoder API:
  http://localhost:8092

Literature Approval API:
  http://localhost:8093
```

Model weights are typically under:

```text
/home/nishad/LLM_Weights
```

The branch defaults the Qdrant collection to:

```text
ehr_chunks_test
```

unless overridden by:

```bash
export COLLECTION_NAME=...
```

---

## 5. Current implemented features on `feature/public-ehr-rag-testbed`

### 5.1 Clinical document taxonomy

File:

```text
src/path_parser.py
```

Now infers `document_type` from folder names/path/file names.

Common document types:

```text
operative_report
discharge_summary
nutrition_note
lab_report
medication_list
clinic_note
history_and_physical
progress_note
radiology
pathology
patient_instructions
unknown
```

This is meant to support CLI-RAG-style “do not treat all documents equally.”

---

### 5.2 Section-aware chunking

File:

```text
src/chunking.py
```

Added:

```python
looks_like_section_header(...)
split_into_sections(...)
chunk_text_with_sections(...)
```

The section detector is intentionally conservative. It now rejects known false positives that appeared during testing:

```text
The patient needs the following limitations
Time of addendum
Completed Action List
1000 mL Initial Volume
```

Useful detected section examples after clean rebuild:

```text
RADREPORT
Procedure History
FINDINGS
IMPRESSION
LEFT ADNEXA
```

Important conclusion from the chat:

> Do not obsess over sections. These notes are extremely unstructured. Sections are a bonus, not the core success criterion.

---

### 5.3 Corpus builder writes document/section metadata

File:

```text
scripts/build_ehr_corpus.py
```

The corpus builder now uses:

```python
chunk_text_with_sections(page["text"])
```

and writes:

```text
document_type
section_title
section_chunk_index
```

into chunk records.

---

### 5.4 Qdrant indexer preserves metadata

File:

```text
scripts/index_qdrant_medcpt.py
```

Qdrant payloads now include:

```text
document_type
section_title
section_chunk_index
```

---

### 5.5 Configurable Qdrant collection name

File:

```text
src/config.py
```

Changed to:

```python
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ehr_chunks")
```

Shell scripts on this feature branch default to:

```bash
COLLECTION_NAME="${COLLECTION_NAME:-ehr_chunks_test}"
```

This prevents accidentally polluting the original/default `ehr_chunks` collection while testing.

---

### 5.6 Shell scripts use test collection by default

Files:

```text
run_build.sh
start_services.sh
```

`run_build.sh` now exports:

```bash
export COLLECTION_NAME="${COLLECTION_NAME:-ehr_chunks_test}"
echo "Qdrant collection: $COLLECTION_NAME"
```

`start_services.sh` now:

- defines `COLLECTION_NAME="${COLLECTION_NAME:-ehr_chunks_test}"`
- passes it into the EHR RAG API
- passes it into the Literature API as well
- prints the active Qdrant collection at the end

You can override when needed:

```bash
COLLECTION_NAME=ehr_chunks ./start_services.sh
COLLECTION_NAME=ehr_chunks ./run_build.sh
```

---

### 5.7 Retrieval planner

File:

```text
src/retrieval_planner.py
```

Added deterministic, non-LLM planner:

```python
build_retrieval_plan(question)
```

It conditionally detects intents:

```text
bariatric
surgery/procedure
labs/micronutrients
medications/supplements
nutrition
follow-up/plan
```

It returns:

```text
primary_query
subqueries
target_document_types
rationale
```

The planner is deliberately conservative and deterministic.

---

### 5.8 Planned retrieval in EHR service

File:

```text
src/ehr_rag_service.py
```

Retrieval now does:

```text
build retrieval plan
→ run targeted dense searches by subquery × document_type
→ run broad fallback searches
→ deduplicate by chunk_id
→ rerank all candidates with MedCPT cross-encoder
→ send top evidence to LLM
```

Important: broad fallback remains so older indexes or missed document types still work.

Returned source metadata now includes:

```text
relative_path
page_num
chunk_id
document_type
section_title
rerank_score
```

---

### 5.9 API exposes retrieval diagnostics

File:

```text
api_ehr_rag.py
```

The `/ask` response now includes:

```text
answer
sources
retrieval_plan
structured_answer
```

`structured_answer` is only populated if `structured=true`.

---

### 5.10 Structured answer mode

Files:

```text
src/structured_answering.py
src/ehr_rag_service.py
api_ehr_rag.py
```

`/ask` now accepts:

```json
{
  "patient_id": "021494762",
  "question": "What vitamin or micronutrient supplementation is documented?",
  "structured": true
}
```

Structured response shape:

```json
{
  "concise_answer": "...",
  "findings": [
    {
      "field": "...",
      "status": "found | not_found | uncertain | inferred_from_evidence",
      "value": "...",
      "evidence": [1, 2],
      "rationale": "..."
    }
  ],
  "missing_information": [],
  "uncertainty": []
}
```

Important conclusion from the chat:

- The structured mode works.
- It returns parseable JSON.
- The checker flags some schema strictness issues, especially `status=not_found` with non-empty evidence.
- This was tabled for now because notes are extremely unstructured and flexibility matters.
- Treat the checker as a diagnostic tool, not a gate.

---

### 5.11 Structured smoke-test script

File:

```text
scripts/test_ehr_retrieval_api.py
```

Supports:

```bash
--structured
--show-answer
--questions-file
--out
```

Example:

```bash
python scripts/test_ehr_retrieval_api.py   --patient-id 021494762   --questions-file eval/ehr_retrieval_smoke_questions.jsonl   --out Data/processed/ehr_retrieval_structured_smoke_results.jsonl   --structured   --show-answer
```

---

### 5.12 Structured smoke-result checker

File:

```text
scripts/check_structured_smoke_results.py
```

Checks for:

```text
structured_answer is null
missing concise_answer
missing/empty findings
invalid statuses
evidence indices outside source range
status=found with no evidence
status=not_found with evidence
missing_information / uncertainty type issues
```

Current known issue:

```text
status=not_found but evidence is non-empty
```

Decision:

```text
Do not block on this. Keep flexibility for unstructured notes.
```

---

### 5.13 Keyword/BM25 retrieval module

File:

```text
src/keyword_retrieval.py
```

Standalone local BM25 retriever over:

```text
Data/processed/chunks.jsonl
```

Uses:

```python
rank_bm25.BM25Okapi
```

Indexes fields:

```text
chunk_text
document_type
section_title
relative_path
file_name
```

Supports:

```text
patient filter
document_type filter
BM25 score
```

Purpose:

> Improve recall for exact clinical terms that dense retrieval may miss.

Examples:

```text
B12
ferritin
iron
thiamine
vitamin D
calcium
PTH
RYGB
sleeve
```

Important clarification:

- The BM25 module itself is query-driven.
- The test script has hardcoded smoke-test queries.
- Main retrieval planner has conditional subquery expansion.
- BM25 is **not yet wired into `/ask`**.

---

### 5.14 Keyword retrieval smoke-test script

File:

```text
scripts/test_keyword_retrieval.py
```

User fixed import path issue by adding:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

Tested successfully:

```bash
python scripts/test_keyword_retrieval.py   --patient-id 021494762   --limit 8
```

Result:

```text
hits=8 for all default queries
```

This validates standalone BM25 loading/search.

---

## 6. Important test commands

### Pull current branch

```bash
git checkout feature/public-ehr-rag-testbed
git pull origin feature/public-ehr-rag-testbed
```

### Clean rebuild into test collection

Use a fresh collection name when validating chunk/index changes:

```bash
COLLECTION_NAME=ehr_chunks_test_v3 ./run_build.sh
COLLECTION_NAME=ehr_chunks_test_v3 ./start_services.sh
```

If processed corpus is stale, back up these before rebuilding:

```bash
mkdir -p Data/processed/backups

ts=$(date +%Y%m%d_%H%M%S)

for f in   Data/processed/documents.jsonl   Data/processed/chunks.jsonl   Data/processed/build_ehr_corpus.ckpt   Data/processed/index_qdrant_medcpt.ckpt
do
  if [ -f "$f" ]; then
    mv "$f" "Data/processed/backups/$(basename "$f").$ts"
  fi
done
```

Then rebuild:

```bash
COLLECTION_NAME=ehr_chunks_test_v3 ./run_build.sh
COLLECTION_NAME=ehr_chunks_test_v3 ./start_services.sh
```

### EHR retrieval smoke test

```bash
python scripts/test_ehr_retrieval_api.py   --patient-id 021494762   --questions-file eval/ehr_retrieval_smoke_questions.jsonl   --out Data/processed/ehr_retrieval_smoke_results_metadata_index_v3.jsonl   --show-answer
```

### Structured smoke test

```bash
python scripts/test_ehr_retrieval_api.py   --patient-id 021494762   --questions-file eval/ehr_retrieval_smoke_questions.jsonl   --out Data/processed/ehr_retrieval_structured_smoke_results.jsonl   --structured   --show-answer
```

### Structured smoke checker

```bash
python scripts/check_structured_smoke_results.py   Data/processed/ehr_retrieval_structured_smoke_results.jsonl   --show-passing
```

### Keyword retrieval smoke test

```bash
python scripts/test_keyword_retrieval.py   --patient-id 021494762   --limit 8
```

Specific keyword query:

```bash
python scripts/test_keyword_retrieval.py   --patient-id 021494762   --query "B12 ferritin iron thiamine vitamin D calcium PTH"   --limit 10
```

With document-type filter:

```bash
python scripts/test_keyword_retrieval.py   --patient-id 021494762   --query "multivitamin calcium citrate vitamin D B12 iron thiamine"   --document-types nutrition_note,clinic_note,discharge_summary,medication_list   --limit 10
```

---

## 7. Latest validation results

### Clean metadata validation

After truly clean rebuild into `ehr_chunks_test_v3`, metadata looked good:

```text
sources: 48

document_type counts:
nutrition_note: 20
operative_report: 18
unknown: 5
clinic_note: 4
radiology: 1

section_title counts:
None: 37
RADREPORT: 4
Procedure History: 3
FINDINGS: 2
LEFT ADNEXA: 1
IMPRESSION: 1
```

Interpretation:

```text
document_type metadata works
section_title metadata works enough
false section headers were reduced
None is acceptable because many notes are unstructured
```

### Structured checker result

Structured mode works, but checker showed:

```text
records: 6
passed: 3
failed: 3
```

Main issue:

```text
status=not_found but evidence is non-empty
```

Decision:

```text
Do not block on this. Keep flexibility for unstructured notes.
```

### Keyword/BM25 test

Standalone keyword retrieval tested successfully:

```text
hits=8 for all default queries
```

---

## 8. Lessons learned / debugging notes

### GPU OOM during rebuild/index

A CUDA OOM happened because GPU 0 was full, mostly by vLLM/Qwen.

Fix that worked:

```text
Stop services temporarily before rebuild/index.
```

Example:

```bash
docker stop ehr_vllm_qwen || true

pkill -f "uvicorn api_ehr_rag:app" || true
pkill -f "uvicorn api_literature_rag:app" || true
pkill -f "uvicorn api_encoder:app" || true
```

Keep/start Qdrant:

```bash
docker start ehr_qdrant || true
```

Then rebuild/index. Restart services after.

### Missing test collection caused 500

A 500 happened because API was pointed at `ehr_chunks_test`, but that collection was missing. Rebuilding into the collection fixed it.

Check collections:

```bash
curl -s http://localhost:6333/collections | python -m json.tool
curl -s http://localhost:6333/collections/ehr_chunks_test | python -m json.tool
```

### Stale chunks.jsonl problem

At one point, bad section headers persisted even after code changes. Cause was likely stale `chunks.jsonl` and checkpoint reuse.

Fix:

```text
Back up documents.jsonl/chunks.jsonl and checkpoints, then rebuild fresh.
```

---

## 9. Current development philosophy

The user prefers:

```text
baby steps
small commits
verify after each change
if change is tiny, provide direct edit instructions
if change is bigger, make the repo change and verify
avoid over-engineering
avoid rigid assumptions because notes are messy
```

Important design preference:

> Flexibility over rigid schemas. These clinical notes are extremely unstructured.

Do not spend too much time on perfect section detection or overly strict structured-answer semantics right now.

---

## 10. Recommended next step

The next logical implementation step is to wire BM25 keyword retrieval into the main EHR retrieval path.

Current desired flow:

```text
retrieval planner
→ dense planned retrieval
→ BM25 keyword retrieval using planner subqueries
→ deduplicate by chunk_id
→ rerank combined candidates with MedCPT cross-encoder
→ answer / structured answer
```

Conservative integration plan:

1. Modify `src/ehr_rag_service.py`.
2. Import `get_keyword_retriever`.
3. In `collect_planned_hits(...)`, after dense planned retrieval and before/after broad fallback, run keyword retrieval over the same planner subqueries.
4. Convert BM25 records into a small hit-like object compatible with existing reranking pipeline, or adapt the code to operate on payload dicts instead of Qdrant hit objects.
5. Deduplicate by `chunk_id`.
6. Preserve dense broad fallback.
7. Add source metadata field later like `retrieval_source: dense | keyword | both` if useful.

Start with standalone wiring and test with the existing smoke tests.

---

## 11. Do not forget

### Current branch

```text
feature/public-ehr-rag-testbed
```

### Main is not yet changed

All work is on the feature branch. Do not merge yet.

### Important patient used in tests

```text
021494762
```

### Useful smoke question file

```text
eval/ehr_retrieval_smoke_questions.jsonl
```

Contains six bariatric/EHR test questions.

### Default Qdrant collection for branch

```text
ehr_chunks_test
```

Override explicitly if needed:

```bash
COLLECTION_NAME=ehr_chunks_test_v3 ./start_services.sh
```

---

## 12. Suggested prompt for a new chat

Paste this at the top of a new chat:

```text
We are working on the GitHub repo:
https://github.com/Kelebrimbor97/Bariatric_Document_version

Use branch:
feature/public-ehr-rag-testbed

This is a local bariatric/EHR document RAG project using FastAPI, Qdrant, MedCPT embeddings/reranker, vLLM/Qwen, and OpenWebUI.

Important: ask me before referring to or modifying the repo unless I explicitly ask you to connect to it. Use baby steps. If a change is tiny, give me the exact edit; if larger, make a small isolated repo change and verify it.

Current branch already has:
- document_type taxonomy in src/path_parser.py
- section-aware chunking in src/chunking.py
- build/index scripts writing document_type/section_title into chunks and Qdrant
- configurable COLLECTION_NAME, with shell scripts defaulting to ehr_chunks_test
- deterministic retrieval planner in src/retrieval_planner.py
- planned dense retrieval and broad fallback in src/ehr_rag_service.py
- /ask response exposing retrieval_plan and rich source metadata
- optional structured=true answer mode
- structured smoke test and checker scripts
- standalone BM25 keyword retriever in src/keyword_retrieval.py
- standalone keyword smoke test script

Validated:
- clean metadata collection worked
- section false positives were reduced
- structured=true returned parseable JSON
- checker flags not_found+evidence but we tabled this because notes are messy
- standalone BM25 keyword retrieval returned hits=8 for default queries

Next planned task:
wire BM25 keyword retrieval into the main /ask retrieval path conservatively:
dense planned retrieval + BM25 candidates + deduplicate + MedCPT rerank.
```

---

## 13. Files changed/added on feature branch

Verified file list from branch diff against `main`:

```text
api_ehr_rag.py
eval/ehr_retrieval_smoke_questions.jsonl
run_build.sh
scripts/build_ehr_corpus.py
scripts/check_structured_smoke_results.py
scripts/index_qdrant_medcpt.py
scripts/test_ehr_retrieval_api.py
scripts/test_keyword_retrieval.py
src/chunking.py
src/config.py
src/ehr_rag_service.py
src/keyword_retrieval.py
src/path_parser.py
src/retrieval_planner.py
src/structured_answering.py
start_services.sh
```

---

## 14. Current open questions

1. How aggressively should BM25 be integrated into main retrieval?
2. Should we add `retrieval_source` metadata: `dense`, `keyword`, or `both`?
3. Should structured answer validation be relaxed rather than normalized?
4. When should public/synthetic datasets be added?
5. Should we build a curated bariatric guideline/literature index locally?
6. Should the branch be split before PR/merge, since it is now fairly large?

---

## 15. Recommended immediate next action

Continue from:

```text
Wire BM25 keyword retrieval into collect_planned_hits(...) in src/ehr_rag_service.py.
```

Keep it conservative and test with:

```bash
COLLECTION_NAME=ehr_chunks_test_v3 ./start_services.sh

python scripts/test_ehr_retrieval_api.py   --patient-id 021494762   --questions-file eval/ehr_retrieval_smoke_questions.jsonl   --out Data/processed/ehr_retrieval_hybrid_smoke_results.jsonl   --structured   --show-answer
```

Then compare source document types and answers against the prior dense-only structured smoke results.
