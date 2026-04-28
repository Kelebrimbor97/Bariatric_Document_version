# Document_version EHR + Literature RAG Workflow

This repository runs a local clinical-document RAG stack with optional literature grounding in OpenWebUI.

The current workflow uses:

- **Docker Compose** for long-running infrastructure:
  - Qdrant vector database
  - vLLM / Qwen LLM server
- **Conda (`ehr_rag`)** for local Python API services:
  - MedCPT encoder API
  - EHR RAG API
  - Literature approval RAG API
- **OpenWebUI** separately on port `8080`
- **Host disk** for code, data, vector storage, logs, and model weights

---

## 1. Directory assumptions

Project root:

```bash
/home/nishad/Bariatric/Document_version
```

Model weights:

```bash
/home/nishad/LLM_Weights
```

OpenWebUI:

```text
http://localhost:8080
```

Expected project layout:

```text
Document_version/
├── docker-compose.yml
├── start_services.sh
├── run_build.sh
├── SYSTEM_PROMPT.md
├── api_encoder.py
├── api_ehr_rag.py
├── api_literature_rag.py
├── scripts/
│   ├── build_ehr_corpus.py
│   └── index_qdrant_medcpt.py
├── src/
│   ├── config.py
│   ├── ehr_rag_service.py
│   ├── encoder_client.py
│   └── ...
├── Data/
│   ├── MBS LLM/
│   │   └── Test Patients/
│   ├── processed/
│   ├── qdrant_storage/
│   └── literature_cache/
└── logs/
```

---

## 2. Services and ports

| Service | Purpose | Port |
|---|---|---:|
| OpenWebUI | Chat UI | `8080` |
| vLLM / Qwen | Local LLM backend | `8000` |
| Qdrant | Vector database | `6333` |
| EHR RAG API | EHR retrieval tool server | `8090` |
| MedCPT Encoder API | Shared encoder service | `8092` |
| Literature Approval API | Two-step PMC/PubMed approval workflow | `8093` |

---

## 3. Normal startup workflow

From a terminal:

```bash
cd /home/nishad/Bariatric/Document_version
./start_services.sh
```

This script should:

1. Start Docker Compose services:
   - `qdrant`
   - `vllm_qwen`
2. Activate the conda environment:
   - `ehr_rag`
3. Start local uvicorn services:
   - `api_encoder.py` on `8092`
   - `api_ehr_rag.py` on `8090`
   - `api_literature_rag.py` on `8093`

---

## 4. Docker Compose services

Docker Compose is responsible for the persistent infrastructure services.

Start manually:

```bash
cd /home/nishad/Bariatric/Document_version
docker compose up -d qdrant vllm_qwen
```

Stop manually:

```bash
docker compose down
```

Check status:

```bash
docker compose ps
```

Follow logs:

```bash
docker logs -f ehr_qdrant
docker logs -f ehr_vllm_qwen
```

---

## 5. Verify all services

After running `./start_services.sh`, verify:

```bash
curl http://localhost:6333
curl http://localhost:8000/v1/models
curl http://localhost:8092/health
curl http://localhost:8090/health
curl http://localhost:8093/health
```

Expected health responses from local APIs:

```json
{"status":"ok"}
```

---

## 6. Logs

The local uvicorn API logs are written to:

```text
logs/api_encoder.log
logs/api_ehr_rag.log
logs/api_literature_rag.log
```

Watch logs:

```bash
tail -f logs/api_encoder.log
tail -f logs/api_ehr_rag.log
tail -f logs/api_literature_rag.log
```

Docker logs:

```bash
docker logs -f ehr_qdrant
docker logs -f ehr_vllm_qwen
```

---

## 7. Building and indexing EHR data

Use `run_build.sh` only when you need to rebuild the EHR corpus or regenerate vector embeddings.

Run:

```bash
cd /home/nishad/Bariatric/Document_version
./run_build.sh
```

This script:

1. activates the `ehr_rag` conda environment
2. runs `scripts/build_ehr_corpus.py`
3. runs `scripts/index_qdrant_medcpt.py`
4. writes logs to:
   - `logs/build_ehr_corpus.out`
   - `logs/index_qdrant_medcpt.out`

Use this when you changed:

- source PDFs
- chunking logic
- metadata fields
- Qdrant payload schema
- embedding/indexing logic
- patient data root

Do **not** run this for normal daily startup.

---

## 8. EHR API tests

List patients:

```bash
curl http://localhost:8090/patients
```

Ask a patient-specific question:

```bash
curl -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"021494762","question":"What discharge summary is present?"}'
```

Ask a corpus-wide question:

```bash
curl -X POST http://localhost:8090/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Across the corpus, which patients appear to have surgical documentation?"}'
```

Supported patient identifiers:

- actual patient ID, for example `021494762`
- full patient folder name, for example `Test 1 - 021494762`
- hashed internal ID, for example `patient_d5b00deea01a2606`

---

## 9. Encoder API tests

Health check:

```bash
curl http://localhost:8092/health
```

Query embedding:

```bash
curl -X POST http://localhost:8092/embed/query \
  -H "Content-Type: application/json" \
  -d '{"texts":["What discharge summary is present?"]}'
```

Article/document embedding:

```bash
curl -X POST http://localhost:8092/embed/article \
  -H "Content-Type: application/json" \
  -d '{"texts":["Discharge summary document text here."]}'
```

The encoder API is an internal service and does **not** need to be added to OpenWebUI.

---

## 10. Literature approval API tests

The literature tool uses a two-step approval workflow.

### Step 1: propose a de-identified query

```bash
curl -X POST http://localhost:8093/literature/propose_query \
  -H "Content-Type: application/json" \
  -d '{"user_question":"For patient 021494762, what does the literature say about post-bariatric vitamin monitoring?"}'
```

This returns:

- `query_id`
- `proposed_query`
- `phi_risk`
- `removed_terms`
- `approval_phrase`
- `message_to_user`

The proposal step does **not** call external literature APIs.

### Step 2: execute only after approval

Use the exact returned approval phrase:

```bash
curl -X POST http://localhost:8093/literature/execute_query \
  -H "Content-Type: application/json" \
  -d '{"query_id":"litq_REPLACE_ME","approval_phrase":"APPROVE litq_REPLACE_ME","retmax":5,"final_k":5}'
```

The execution step may call external NCBI/PMC services using only the sanitized literature query.

---

## 11. OpenWebUI integration

OpenWebUI runs separately at:

```text
http://localhost:8080
```

Add the following OpenAPI tool servers in OpenWebUI:

```text
http://host.docker.internal:8090
http://host.docker.internal:8093
```

If `host.docker.internal` does not resolve on Linux, use the host machine IP instead:

```text
http://<your-host-ip>:8090
http://<your-host-ip>:8093
```

Do **not** add the encoder API to OpenWebUI. It is only used internally by the EHR and literature APIs.

---

## 12. System prompt

Maintain the OpenWebUI system prompt in:

```text
SYSTEM_PROMPT.md
```

Paste its contents into the relevant OpenWebUI model/system prompt configuration.

The prompt should instruct the model to:

- use the EHR tool for patient-chart questions
- use the literature proposal tool before any external literature search
- wait for the user’s exact approval phrase before executing literature search
- avoid sending patient identifiers, exact dates, note excerpts, institutions, names, or other PHI to external literature APIs
- answer normally when no tool is needed

---

## 13. Typical daily workflow

```bash
cd /home/nishad/Bariatric/Document_version
./start_services.sh
```

Then open:

```text
http://localhost:8080
```

In OpenWebUI:

1. choose the normal Qwen/vLLM model
2. enable the EHR tool
3. enable the literature approval tool
4. use `SYSTEM_PROMPT.md` as the system prompt

---

## 14. Example OpenWebUI questions

General, no tool needed:

```text
What is overfitting in machine learning?
```

EHR patient-specific:

```text
What discharge summary is present for patient 021494762?
```

EHR corpus-wide:

```text
Across the corpus, which patients appear to have surgical documentation?
```

Literature proposal:

```text
What does the literature say about post-bariatric vitamin monitoring?
```

Mixed EHR + literature:

```text
For patient 021494762, summarize the chart findings relevant to post-bariatric nutrition, then propose a de-identified literature query for external PMC grounding.
```

---

## 15. Shutdown

Stop local API services:

```bash
pkill -f "uvicorn api_encoder:app" || true
pkill -f "uvicorn api_ehr_rag:app" || true
pkill -f "uvicorn api_literature_rag:app" || true
```

Stop Docker services:

```bash
cd /home/nishad/Bariatric/Document_version
docker compose down
```

---

## 16. Troubleshooting

### Qdrant not reachable

```bash
docker compose ps
docker logs -f ehr_qdrant
```

### vLLM not reachable

```bash
docker logs -f ehr_vllm_qwen
curl http://localhost:8000/v1/models
```

### EHR API failing

```bash
tail -f logs/api_ehr_rag.log
curl http://localhost:8090/health
```

### Encoder API failing

```bash
tail -f logs/api_encoder.log
curl http://localhost:8092/health
```

### Literature API failing

```bash
tail -f logs/api_literature_rag.log
curl http://localhost:8093/health
```

### OpenWebUI cannot reach tools

Try host IP instead of `host.docker.internal`:

```text
http://<your-host-ip>:8090
http://<your-host-ip>:8093
```

### Conda environment missing packages

```bash
conda activate ehr_rag
pip install -r requirements.txt
```

---

## 17. Notes on privacy

- The EHR API is local.
- The encoder API is local.
- The literature proposal step is local.
- The literature execution step may call external NCBI/PMC services.
- External literature calls should only use sanitized, de-identified biomedical queries.
- Do not send patient IDs, names, dates of birth, MRNs, exact dates, institutions, provider names, or note excerpts to external literature APIs.
