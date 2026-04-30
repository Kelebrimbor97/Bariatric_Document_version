# Metrics Strategy for the Bariatric/EHR Document RAG Project

## 1. Metrics used by the reference papers

The reference papers are useful, but neither matches this project perfectly. **EHR-RAG** is mainly a structured EHR prediction paper, while **CLI-RAG** is mainly a clinical note generation paper. This project is closer to **evidence-grounded clinical document QA**: retrieve messy patient PDFs/notes, answer a question, expose sources, and optionally emit structured JSON.

### EHR-RAG

EHR-RAG evaluates on EHRSHOT long-horizon structured-EHR prediction tasks, such as long length of stay, 30-day readmission, acute myocardial infarction, and anemia. It reports:

| Metric | What it measures | Notes |
|---|---|---|
| Accuracy | Overall classification correctness | Useful but can hide poor minority-class behavior. |
| Macro-F1 | Average F1 across classes | Important because clinical prediction tasks are often imbalanced. |
| Per-class F1 | F1 for each outcome class | Helps show whether the model only performs well on the majority class. |
| Ablation drops | Accuracy/Macro-F1 after removing components | Shows whether individual system components help. |

**Takeaway for this project:** Macro-F1 is excellent when the output is a discrete label, such as “readmitted: yes/no” or “anemia severity class.” It is not enough for the current free-text document QA task.

---

### CLI-RAG

CLI-RAG evaluates generated SOAP-style progress notes from MIMIC-III visits. It reports a broader generation-oriented set:

| Metric | What it measures | Notes |
|---|---|---|
| BLEU | N-gram precision overlap | Low scores are expected for paraphrased clinical text. |
| ROUGE-1/2/L | N-gram and longest-subsequence overlap | Common for summarization, but weak for factual grounding. |
| Semantic similarity | Embedding cosine similarity between generated and gold notes | Better for paraphrases than BLEU/ROUGE. |
| SOAP sections present | Structural completeness | Checks whether Subjective, Objective, Assessment, Plan are present. |
| Length ratio | Generated length vs. gold length | Detects verbosity or under-answering. |
| Temporal consistency/coherence | Similarity between adjacent notes in a patient trajectory | Useful for longitudinal note generation. |
| LLM preference voting | Blinded model preference over clinician-authored notes | Quality proxy, but evaluator-dependent. |

**Takeaway for this project:** CLI-RAG’s retrieval ideas are very relevant, but its headline metrics are for note generation, not for patient-specific QA. We should borrow structural validity, semantic/factual checks, and source-document analysis, but not use BLEU/ROUGE as the primary metric.

---

## 2. Metrics usually reported for projects like this

For EHR/document RAG systems, metrics usually fall into five buckets.

### A. Retrieval metrics

These measure whether the right evidence was retrieved.

| Metric | Use |
|---|---|
| Recall@k | Did the system retrieve the needed evidence somewhere in the top k? |
| Hit@k / Top-k accuracy | Did at least one relevant document appear in the top k? |
| MRR@k | How high was the first relevant source ranked? |
| nDCG@k | Ranking quality, especially if evidence relevance is graded. |
| Precision@k | How much of the retrieved context is actually useful? |

For this project, **Recall@k and MRR@k matter more than Precision@k initially**, because missing the right clinical evidence is worse than retrieving one extra noisy chunk that the reranker/LLM can ignore.

---

### B. Answer correctness metrics

These measure whether the final answer is correct.

| Metric | Use |
|---|---|
| Exact Match | Best for short extractive QA, but too strict for this project’s answers. |
| Token F1 | Common in QA datasets with reference answers. |
| Expected fact / concept coverage | Best for the current synthetic checks. |
| Forbidden fact rate / hallucination rate | Catches unsafe additions. |
| Abstention correctness | Checks whether the model says “not found” when evidence is absent. |

For this project, **expected fact coverage plus forbidden-term violations** is better than Exact Match, because answers can be correct in many phrasings.

---

### C. Grounding / citation metrics

These are especially important for clinical RAG.

| Metric | Use |
|---|---|
| Citation validity | Are cited source indices valid? |
| Citation support | Does the cited source actually support the claim? |
| Unsupported claim rate | What fraction of generated claims lack evidence? |
| Evidence sufficiency | Were enough sources retrieved to justify the answer? |

For this project, this bucket should be central. In clinical chart QA, an answer that sounds good but is not source-grounded is worse than a short answer that admits uncertainty.

---

### D. Structured-output metrics

These measure whether the output is machine-readable and clinically usable.

| Metric | Use |
|---|---|
| JSON parse rate | Does `structured_answer` parse? |
| Schema validity | Are expected fields present? |
| Status validity | Are statuses from the allowed set? |
| Evidence-index validity | Do findings cite real source indices? |
| Over-answering rate | Does it invent extra findings not asked for? |

This is very relevant to the current `structured=true` mode. Given the messy PDFs, this should be a diagnostic metric, not a strict blocker yet.

---

### E. Clinical summarization / note-generation metrics

These matter when testing on summarization datasets.

| Metric | Use |
|---|---|
| ROUGE-1/2/L | Standard summarization overlap. |
| BLEU / METEOR | Lexical generation quality. |
| BERTScore / embedding similarity | Semantic similarity. |
| BLEURT / COMET | Learned text quality metrics. |
| Medical concept F1 / UMLS concept overlap | Clinical fact preservation. |

For the current project, these are not the best primary metrics because the main task is not full clinical note generation. Later, if evaluating summarization or note-generation datasets, add them.

---

## 3. Best metric strategy for this project

Do not choose a single metric like ROUGE, BLEU, or Macro-F1 as the headline metric. The best fit is a **multi-layer evaluation**, with one headline score and several diagnostic metrics.

### Recommended headline metric

Use:

```text
Evidence-Grounded Task Success Rate
```

A record passes only if:

```text
1. Required answer facts are present.
2. Required “any-of” answer groups are satisfied.
3. Forbidden/hallucinated facts are absent.
4. Required source document_type, and later required source/chunk, appears in retrieved sources.
5. Structured answer is valid if structured=true.
6. Evidence citations are valid.
```

This directly matches the project task: **answer the patient-specific question correctly using retrieved chart evidence**. It works on synthetic/public-safe data now and can later work on curated real/private or public datasets.

---

## 4. Recommended metric suite for the evaluator

For `scripts/evaluate_synthetic_results.py`, report these in priority order.

### Tier 1: End-to-end safety/correctness

| Metric | Primary? | Why |
|---|---:|---|
| Evidence-Grounded Task Success Rate | Yes | Best single headline score. |
| Expected answer term/fact coverage | Yes | Measures completeness. |
| Forbidden term / hallucination violation rate | Yes | Measures unsafe additions. |
| Abstention correctness | Soon | Needed for “not documented” questions. |

For the current synthetic expected-check format, term coverage is fine. Later, evolve from “terms” to “facts,” for example:

```json
{
  "required_facts": [
    {"concept": "procedure", "value": "sleeve gastrectomy"},
    {"concept": "vitamin_d", "value": "documented"}
  ],
  "forbidden_facts": [
    {"concept": "procedure", "value": "Roux-en-Y"}
  ]
}
```

Not yet. Baby step first.

---

### Tier 2: Retrieval quality

| Metric | Primary? | Why |
|---|---:|---|
| Required source document_type recall@k | Yes, currently | Works with current labels. |
| Top-1 source document_type accuracy | Diagnostic | Tells whether the best-ranked source is the right type. |
| MRR for required document_type | Diagnostic | Measures how high the first useful source appears. |
| Retrieval_source distribution | Diagnostic | Tracks dense vs. keyword vs. both. |
| Evidence recall@k by chunk/document ID | Later primary | Better once gold evidence IDs exist. |
| nDCG@k | Later | Best when graded relevance labels exist. |

Right now, document_type-level recall/MRR is a good proxy because the synthetic expected file knows source document types, not exact evidence chunks. Later, when testing on datasets with gold evidence spans or curated synthetic gold chunks, Recall@k and nDCG@k over exact evidence IDs should replace document_type recall as the stronger retrieval metric.

---

### Tier 3: Grounding and citation quality

| Metric | Primary? | Why |
|---|---:|---|
| Citation index validity | Yes | Cheap and deterministic. |
| Claim-to-source support rate | Later primary | Best hallucination metric, but requires claim extraction or a judge. |
| Unsupported claim rate | Later primary | Important for clinical trust. |
| Source sufficiency | Later | Whether evidence is enough to justify the answer. |

For now, citation validity can be deterministic. Later, add an optional LLM-judge or rule-based claim-support evaluator, but keep it secondary until validated.

---

### Tier 4: Structured output quality

| Metric | Primary? | Why |
|---|---:|---|
| Structured answer parse rate | Yes | Ensures usable structured output. |
| Schema validity | Yes | Ensures expected fields exist. |
| Evidence field validity | Yes | Ensures evidence references are valid. |
| Status distribution | Diagnostic | Helps understand answer behavior. |
| Over-answering rate | Important later | Detects unrelated extra findings. |

Do not make this too rigid yet. The project design philosophy is correct: messy clinical notes need flexibility.

---

### Tier 5: Dataset-specific metrics for future public benchmarks

When testing on other datasets:

| Dataset/task type | Add these metrics |
|---|---|
| EHRSHOT-style prediction | Accuracy, Macro-F1, per-class F1. |
| emrQA / extractive clinical QA | Exact Match, token F1, evidence Recall@k/MRR. |
| MIMIC-IV-Note summarization | ROUGE, BLEU, METEOR, BERTScore, maybe COMET. |
| ACI-Bench / note generation | ROUGE, BERTScore, BLEURT, medical concept F1 / UMLS overlap. |
| Retrieval-only datasets | nDCG@10, Recall@k, MRR@k, Precision@k. |
| Bariatric chart QA | Evidence-Grounded Task Success, fact coverage, hallucination rate, evidence recall/MRR, citation support. |

---

## 5. Final recommendation

Use this as the core scoreboard:

```text
Primary:
  Evidence-Grounded Task Success Rate

Core submetrics:
  Answer fact coverage
  Forbidden/hallucination violation rate
  Required evidence recall@k
  MRR@k for required evidence
  Citation validity
  Structured-answer validity

Diagnostics:
  Top-1 source document_type accuracy
  retrieval_source distribution: dense / keyword / both
  document_type distribution
  status distribution: found / not_found / uncertain / inferred_from_evidence
  answer length
```

For the immediate baby step, implement only what the current files support:

```text
records
passed / failed
required_answer_terms coverage
required_any_terms coverage
forbidden_answer_terms violations
required_source_document_type recall
structured_answer validity
evidence citation validity
retrieval_source distribution
top-1 source document_type accuracy
MRR-like required document_type ranking
```

Then the next evolution should be to add a gold-evidence file with exact expected source IDs/chunk IDs. Once that exists, the best retrieval metrics become:

```text
Recall@5 / Recall@10
MRR@10
nDCG@10
```

And the best end-to-end clinical metric remains:

```text
Evidence-Grounded Task Success Rate
```

This gives the project something publishable-looking, comparable across future datasets, and still practical for the current synthetic bariatric testbed.
