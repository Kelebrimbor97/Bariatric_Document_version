import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "qwen-ehr")

COLLECTION_NAME = "ehr_chunks"

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"
QDRANT_STORAGE_DIR = PROJECT_ROOT / "Data" / "qdrant_storage"

DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "Data")))
PATIENTS_ROOT = Path(
    os.getenv(
        "PATIENTS_ROOT",
        str(DATA_DIR / "MBS LLM" / "Test Patients"),
    )
)

LLM_WEIGHTS_DIR = Path(
    os.getenv("LLM_WEIGHTS_DIR", str(PROJECT_ROOT.parent / "LLM_Weights"))
)
MEDCPT_QUERY_MODEL = os.getenv(
    "MEDCPT_QUERY_MODEL",
    str(LLM_WEIGHTS_DIR / "MedCPT-Query-Encoder"),
)
MEDCPT_ARTICLE_MODEL = os.getenv(
    "MEDCPT_ARTICLE_MODEL",
    str(LLM_WEIGHTS_DIR / "MedCPT-Article-Encoder"),
)
MEDCPT_RERANK_MODEL = os.getenv(
    "MEDCPT_RERANK_MODEL",
    str(LLM_WEIGHTS_DIR / "MedCPT-Cross-Encoder"),
)

HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", str(LLM_WEIGHTS_DIR)))

ENCODER_API_URL = os.getenv("ENCODER_API_URL", "http://localhost:8092")

# PMC API calling

LITERATURE_API_PORT = int(os.getenv("LITERATURE_API_PORT", "8093"))

NCBI_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "local@example.com")
NCBI_TOOL = os.getenv("NCBI_TOOL", "local_literature_rag")
NCBI_API_KEY = os.getenv("NCBI_API_KEY") or None  # optional

LITERATURE_CACHE_DIR = PROJECT_ROOT / "Data" / "literature_cache"
