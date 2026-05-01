# Bariatric Document RAG Project — Handoff Notes

_Last updated in chat: 2026-04-30 — content-first PDF classification, clean build support, MIMIC-IV v4 atomic structured chunks, and current synthetic/MIMIC metrics baseline_

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
36 commits ahead of main
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


```text
Synthetic bariatric PDF classifier benchmark:
  records: 12
  passed: 12
  failed: 0

MIMIC-IV direct pilot v4:
  records: 25
  passed: 25
  failed: 0
```

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

### 5.1 Content-first document classification for PDFs

Files:

```text
src/path_parser.py
src/document_classifier.py
scripts/build_ehr_corpus.py
```

The project no longer treats folder path as the default source of truth for document_type.

Current PDF document-type decision order:

```text
explicit source metadata
→ filename hint
→ content inference
→ optional path hint
→ unknown
```

path_parser.py still extracts path metadata and path hints, but document_classifier.py now makes the final document-type decision.

PDF chunk/document metadata now includes:

```text
document_type
document_type_source
document_type_confidence
document_type_signals
path_document_type_hint
```

Path fallback is disabled by default:

```bash
USE_PATH_HINTS_FOR_DOCUMENT_TYPE=false
```

It can be enabled as a low-confidence fallback:

```bash
USE_PATH_HINTS_FOR_DOCUMENT_TYPE=true
```

Supported deterministic content/filename classes currently include:

```text
radiology
pathology
lab_report
operative_report
discharge_summary
medication_list
nutrition_note
history_and_physical
progress_note
clinic_note
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

### 5.6 Clean build support

Files:

```text
scripts/build_ehr_corpus.py
run_build.sh
```

PDF builds now support a clean build option to prevent stale checkpoint/output reuse.

Use:

```bash
CLEAN_BUILD=1 ./run_build.sh
```

This removes generated files inside PROCESSED_DIR before rebuilding:

```text
documents.jsonl
chunks.jsonl
build_ehr_corpus.ckpt
index_qdrant_medcpt.ckpt
build_ehr_corpus.errors.jsonl
```

It does not remove source PDFs, eval files, logs, or Qdrant collections.


### 5.7 Shell scripts use test collection by default

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

### 5.8 Retrieval planner

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

### 5.9 Planned retrieval in EHR service

File:

```text
src/ehr_rag_service.py
```

Retrieval now does:

```text
build retrieval plan
→ run targeted dense searches by subquery × document_type
→ run BM25 keyword searches using planner primary query + subqueries
→ deduplicate by chunk_id
→ run broad dense fallback searches
→ annotate retrieval_source as dense | keyword | both
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
retrieval_source
```

---

### 5.10 API exposes retrieval diagnostics

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
Each source now includes retrieval diagnostics including:

```
retrieval_source: dense | keyword | both
```

`structured_answer` is only populated if `structured=true`.

---

### 5.11 Structured answer mode

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

### 5.12 Structured smoke-test script

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

### 5.13 Structured smoke-result checker

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

### 5.14 Keyword/BM25 retrieval module

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
- The standalone test script has hardcoded smoke-test queries.
- Main retrieval uses planner primary query + subqueries for BM25.
- BM25 is now wired into `/ask` conservatively.
- BM25 candidates are deduplicated by `chunk_id` with dense candidates before MedCPT reranking.

---

### 5.15 Keyword retrieval smoke-test script

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

### 5.16 Retrieval source inspection helper

File:

```text
scripts/inspect_retrieval_sources.py
```

Added a helper script to inspect final /ask sources by retrieval_source.
This was used to inspect the single keyword-only source from the hybrid smoke test. The source was:

```text
record: 5
source rank: 7
question: Is there evidence of B12, iron, ferritin, thiamine, vitamin D, calcium, or PTH monitoring?
document_type: clinic_note
section_title: None
rerank_score: -11.4140
```

The chunk was mostly boilerplate/admin text from a nutrition clinic note and did not contain the requested micronutrient/lab terms.

Conclusion:

The single keyword-only survivor was noisy, but harmless.
BM25 is functioning as conservative recall support.
Do not tune BM25 yet based on this one noisy hit.

---

### 5.17 Synthetic/public-safe bariatric testbed

Files:
```text
scripts/create_synthetic_bariatric_testbed.py
scripts/check_synthetic_smoke_results.py
eval/synthetic_bariatric_smoke_questions.jsonl
eval/synthetic_bariatric_expected_checks.jsonl
```

Added a small no-PHI synthetic PDF testbed with three patients:

```text
SYN001:
  sleeve gastrectomy
  nutrition follow-up
  medication list
  micronutrient lab report with B12, iron, ferritin, vitamin D, calcium, PTH
  thiamine explicitly not measured/ordered

SYN002:
  preoperative Roux-en-Y evaluation
  no prior bariatric surgery documented
  basic labs only
  micronutrient monitoring explicitly absent

SYN003:
  postoperative Roux-en-Y dehydration/nausea
  discharge medications include thiamine 100 mg daily for 14 days
  bariatric multivitamin daily
  continue calcium citrate and vitamin D
```

The generator writes text-based PDFs under:

```text
Data/public_testbed/synthetic_bariatric_pdf/Test Patients
```
and writes smoke questions to:

```text
eval/synthetic_bariatric_smoke_questions.jsonl
```

The deterministic checker validates expected answer terms, required source document types, forbidden answer terms, and basic structured-answer shape.

Current synthetic PDF classifier benchmark:

```text
records: 12
passed: 12
failed: 0
answer_term_coverage: 1.0
required_source_document_type_recall: 1.0
structured_answer_validity: 1.0
evidence_citation_validity: 1.0
```

Important design decision:

```text
Do not add agentic verifier/retry logic yet.
A verifier was considered, but it added complexity without clear benefit.
Prefer deterministic synthetic regression checks for now.
```

### 5.18 Structured answer prompt tightening

File:

```text
src/structured_answering.py
```

The structured prompt was tightened to reduce over-answering on broad questions.

New behavior validated on `SYN003`:

Question:

```text
What thiamine or vitamin supplementation is documented?
```

Final desired answer behavior:

```text
The documentation includes thiamine 100 mg daily for 14 days,
a bariatric multivitamin daily, and continued calcium citrate and vitamin D supplementation.
```

The structured answer should not add unrelated missing items such as B12, folate, or iron unless explicitly asked.

Prompt rules added:

```text
Only create findings for items directly requested by the question or clearly necessary to answer it.
Do not add extra not_found findings for unrelated labs, vitamins, diagnoses, or medications.
Only put items in missing_information if the user explicitly requested them and they are absent from the evidence.
For broad "what is documented" questions, do not list unrelated absent items in missing_information.
```


### 5.19 Direct MIMIC-IV pilot corpus builder

File:

```text
scripts/build_mimic_iv_pilot_corpus.py
```

This direct builder avoids the old inefficient path:

```text
CSV/text → PDF → PDF extraction → chunks
```

and instead writes:

```text
MIMIC CSV/text → documents.jsonl + chunks.jsonl → Qdrant
```

Current source tables:

```text
MIMIC-IV-Note discharge.csv.gz
MIMIC-IV admissions.csv.gz
MIMIC-IV diagnoses_icd.csv.gz
MIMIC-IV d_icd_diagnoses.csv.gz
MIMIC-IV procedures_icd.csv.gz
MIMIC-IV d_icd_procedures.csv.gz
MIMIC-IV prescriptions.csv.gz
```

Current evidence kinds:

```text
admission_summary
diagnosis_list
procedure_list
medication_list
discharge_summary
```

Important v4 fix:

Compact generated structured documents are now kept atomic:

```text
admission_summary
diagnosis_list
procedure_list
medication_list
```

Long discharge summaries still use section-aware chunking.

Reason:

The v3 debug evaluator showed that missing expected answer terms were absent from retrieved evidence text even though the correct evidence_kind was present. Keeping compact structured lists atomic fixed the issue.

## 6. Important test commands

### Pull current branch

```bash
git checkout feature/public-ehr-rag-testbed
git pull origin feature/public-ehr-rag-testbed
```

### Clean synthetic PDF rebuild into test collection

Use `CLEAN_BUILD=1` to avoid stale JSONL/checkpoint reuse:

```bash
PROCESSED_DIR="$PWD/Data/processed_classifier_synth_test" \
PATIENTS_ROOT="$PWD/Data/public_testbed/synthetic_bariatric_pdf/Test Patients" \
COLLECTION_NAME=ehr_chunks_classifier_synth_test \
CLEAN_BUILD=1 \
./run_build.sh
```

Restart APIs against the same processed directory and collection:

PROCESSED_DIR="$PWD/Data/processed_classifier_synth_test" \
COLLECTION_NAME=ehr_chunks_classifier_synth_test \
./restart_services.sh
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

### Synthetic bariatric PDF classifier benchmark

Generate or refresh the synthetic PDFs if needed:

```bash
python scripts/create_synthetic_bariatric_testbed.py --force
```

Build/index:

```bash
PROCESSED_DIR="$PWD/Data/processed_classifier_synth_test" \
PATIENTS_ROOT="$PWD/Data/public_testbed/synthetic_bariatric_pdf/Test Patients" \
COLLECTION_NAME=ehr_chunks_classifier_synth_test \
CLEAN_BUILD=1 \
./run_build.sh
```

Restart APIs:

```bash
PROCESSED_DIR="$PWD/Data/processed_classifier_synth_test" \
COLLECTION_NAME=ehr_chunks_classifier_synth_test \
./restart_services.sh
```

Run all synthetic smoke questions:

```bash
python scripts/test_ehr_retrieval_api.py \
  --questions-file eval/synthetic_bariatric_smoke_questions.jsonl \
  --structured \
  --show-answer \
  --out Data/processed_classifier_synth_test/synth_results_classifier_test.jsonl
```

Evaluate:

```bash
python scripts/evaluate_synthetic_results.py \
  --questions eval/synthetic_bariatric_smoke_questions.jsonl \
  --expected eval/synthetic_bariatric_expected_checks.jsonl \
  --results Data/processed_classifier_synth_test/synth_results_classifier_test.jsonl \
  --out Data/processed_classifier_synth_test/synth_metrics_classifier_test.json
```
Expected result:

```text
records: 12
passed: 12
failed: 0
```
---

## 7. Latest validation results

### Synthetic bariatric PDF classifier benchmark

Validated after content-first document classification and clean build support.

```text
records: 12
passed: 12
failed: 0
evidence_grounded_task_success_rate: 1.0
answer_term_coverage: 1.0
required_source_document_type_recall: 1.0
structured_answer_validity: 1.0
evidence_citation_validity: 1.0
```

Validated document types:

clinic_note
discharge_summary
lab_report
medication_list
nutrition_note
operative_report
radiology

Validated question types:

```text
bariatric procedure / surgical history
planned bariatric procedure
prior bariatric surgery absence
micronutrient laboratory monitoring
vitamin / micronutrient supplementation
thiamine measurement/order absence
postoperative complication / follow-up issue
postoperative lab-result absence
postoperative leak imaging absence
```

### MIMIC-IV direct pilot v4 benchmark

Build the direct MIMIC-IV pilot corpus:

```bash
python scripts/build_mimic_iv_pilot_corpus.py \
  --limit-admissions 10 \
  --max-questions 25 \
  --processed-dir Data/processed_mimic_iv_pilot_v4 \
  --force
```

Index:

```bash
PROCESSED_DIR="$PWD/Data/processed_mimic_iv_pilot_v4" \
COLLECTION_NAME=ehr_chunks_mimic_iv_pilot_v4 \
python scripts/index_qdrant_medcpt.py
```

Restart APIs:

```bash
PROCESSED_DIR="$PWD/Data/processed_mimic_iv_pilot_v4" \
COLLECTION_NAME=ehr_chunks_mimic_iv_pilot_v4 \
./restart_services.sh
```

Run questions:

```bash
python scripts/test_ehr_retrieval_api.py \
  --questions-file eval/mimic_iv_pilot_questions.jsonl \
  --structured \
  --show-answer \
  --out Data/processed_mimic_iv_pilot_v4/mimic_iv_pilot_results_v4.jsonl
```

Validated after atomic structured chunks.

```text
records: 25
passed: 25
failed: 0
evidence_grounded_task_success_rate: 1.0
answer_term_coverage: 1.0
required_source_document_type_recall: 1.0
required_evidence_kind_recall: 1.0
structured_answer_validity: 1.0
evidence_citation_validity: 1.0
```

Validated question types:

```text
admission type
discharge location / disposition
ICD-coded diagnoses
ICD-coded procedures
medications documented for admission
```

Important interpretation:

```text
The previous MIMIC failure was not primarily an LLM-copying problem.
The --chunks debug join showed missing terms were absent from retrieved chunk text.
Atomic structured-list chunks fixed the evidence visibility problem.
```

Remaining note:

```text
Top-1 evidence_kind accuracy is still moderate, but end-to-end task success is now 25/25.
Do not tune ranking unless larger evaluations show ranking-driven failures.
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

### Restarting local APIs after code changes

For normal code changes, use:

```bash
COLLECTION_NAME=ehr_chunks_test_v3 ./restart_services.sh
```

This restarts local FastAPI/uvicorn services only:

```text
api_encoder:app
api_ehr_rag:app
api_literature_rag:app
```

It intentionally does not touch Docker services such as Qdrant, vLLM/Qwen, or OpenWebUI.

To stop only local APIs:

```bash
./stop_services.sh
```

Important: pass `COLLECTION_NAME=ehr_chunks_test_v3` explicitly when testing against the clean validation collection.


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

Current stable baseline:

```text
Synthetic bariatric PDF classifier benchmark: 12/12
MIMIC-IV direct pilot v4: 25/25
```

Do not prioritize:

```text
OpenWebUI polish
agentic verifier/retry workflows
BM25 tuning by gut feel
ranking tuning before larger evidence
```

Recommended next direction:

```text
Expand public-data coverage carefully.
```

Concrete next baby step:

```text
Add direct MIMIC radiology and/or lab-event coverage.
```

Likely next extensions:


1. Add radiology.csv.gz from MIMIC-IV-Note:
  - document_type = radiology
  - evidence_kind = radiology_report
  - source_table = radiology
2. Add labevents.csv.gz + d_labitems.csv.gz from MIMIC-IV:
  - document_type = lab_report
  - evidence_kind = lab_result_table
  - source_table = labevents
3. Generate small deterministic expected checks for these new evidence kinds.
4. Re-run metrics before changing retrieval/ranking.

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

First, read PROJECT_HANDOFF.md from the repo and use it as the source of truth for current project state.

This is a local bariatric/EHR document RAG project using FastAPI, Qdrant, MedCPT embeddings/reranker, vLLM/Qwen, and OpenWebUI, but we are NOT focusing on OpenWebUI polish right now.

Important working rules:
- Ask me before referring to or modifying the repo unless I explicitly ask you to connect to it.
- Use baby steps.
- If a change is tiny, give me the exact edit and let me make/push it locally.
- If a change is larger, make one small isolated repo change, then verify it.
- Do not over-engineer rigid schemas because the notes/PDFs are extremely unstructured.
- Prioritize flexibility and testability over perfect schemas.

Current branch already has:
- content-first PDF document classifier in src/document_classifier.py
- path_parser.py retained for path metadata and optional low-confidence path hints
- USE_PATH_HINTS_FOR_DOCUMENT_TYPE=false by default
- document_type taxonomy and classification diagnostics
- section-aware chunking in src/chunking.py
- clean build support via CLEAN_BUILD=1 ./run_build.sh
- configurable PROCESSED_DIR and COLLECTION_NAME
- deterministic retrieval planner in src/retrieval_planner.py
- planned dense retrieval, BM25 keyword retrieval, broad fallback, and MedCPT reranking in src/ehr_rag_service.py
- retrieval_source diagnostics exposed in /ask sources as dense | keyword | both
- evidence_kind/source_table support for direct MIMIC structured data
- optional structured=true answer mode
- synthetic bariatric PDF testbed
- metrics evaluator with optional --chunks evidence visibility diagnostics
- direct MIMIC-IV pilot corpus builder in scripts/build_mimic_iv_pilot_corpus.py
- atomic chunks for compact MIMIC structured sources: admission_summary, diagnosis_list, procedure_list, medication_list

Validated:
- synthetic bariatric PDF classifier benchmark passes 12/12
- MIMIC-IV direct pilot v4 passes 25/25
- structured answer validity is 1.0 on both current benchmarks
- citation validity is 1.0 on both current benchmarks
- MIMIC v3 failure was diagnosed with --chunks debug as evidence visibility/chunking, not LLM copying
- atomic structured chunks fixed MIMIC pilot from 14/25 to 25/25

Current intended direction:
Expand public-data coverage carefully, likely MIMIC radiology and lab events next. Do not tune BM25/ranking by gut feel.
```

---

## 13. Files changed/added on feature branch

Verified file list from branch diff against `main`:

```text
api_ehr_rag.py
eval/ehr_retrieval_smoke_questions.jsonl
eval/synthetic_bariatric_expected_checks.jsonl
eval/synthetic_bariatric_smoke_questions.jsonl
run_build.sh
scripts/build_ehr_corpus.py
scripts/check_structured_smoke_results.py
scripts/check_synthetic_smoke_results.py
scripts/create_synthetic_bariatric_testbed.py
scripts/index_qdrant_medcpt.py
scripts/inspect_retrieval_sources.py
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

1. Should BM25 retrieval be tuned further, or left conservative for now?
2. Is the single keyword-only source from the latest smoke test clinically useful?
3. Should structured answer validation be relaxed rather than normalized?
4. When should public/synthetic datasets be added?
5. Should we build a curated bariatric guideline/literature index locally?
6. Should the branch be split before PR/merge, since it is now fairly large?

---

## 15. Recommended immediate next action

The keyword-only source has already been inspected and was noisy but harmless. The BM25 `/ask` integration should be left alone for now.

The synthetic deterministic checker now covers all 7 synthetic smoke questions and passes 7/7.

Current committed synthetic smoke questions:

```text
eval/synthetic_bariatric_smoke_questions.jsonl
```

Current committed expected-check file:

```text
eval/synthetic_bariatric_expected_checks.jsonl
```

The 7-question synthetic baseline covers:

```text
SYN001:
  procedure history
  micronutrient monitoring
  supplementation

SYN002:
  planned bariatric procedure
  absent micronutrient monitoring

SYN003:
  postoperative complication / follow-up issue
  thiamine and vitamin supplementation
```

Avoid overfitting to exact wording. These checks should remain lightweight regression probes, not rigid clinical grading.

Recommended next direction:

```text
Metrics-based evaluation.
```

Concrete next baby step:

```text
Add a metrics evaluator for synthetic results that computes:
- number of evaluated records
- pass/fail count
- expected answer term coverage
- forbidden term violations
- required source document_type recall
- structured_answer validity
- evidence citation validity
- retrieval_source distribution
- top-1 source document_type
- MRR-like score for required document_type ranking
```

Do not prioritize OpenWebUI prompt/tool polish yet. Do not add agentic verifier/retry workflows yet. Use metrics first to decide whether retrieval, reranking, prompts, or BM25 need changes.