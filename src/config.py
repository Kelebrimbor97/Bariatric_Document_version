from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QDRANT_URL = "http://localhost:6333"
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL_NAME = "qwen-ehr"

COLLECTION_NAME = "ehr_chunks"

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"
QDRANT_STORAGE_DIR = PROJECT_ROOT / "Data" / "qdrant_storage"

# Update this to your real patient-root folder
PATIENTS_ROOT = Path("/home/nishad/Bariatric/Document_version/Data/MBS LLM/Test Patients")

MEDCPT_QUERY_MODEL = "/home/nishad/LLM_Weights/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_MODEL = "/home/nishad/LLM_Weights/MedCPT-Article-Encoder"
MEDCPT_RERANK_MODEL = "/home/nishad/LLM_Weights/MedCPT-Cross-Encoder"

HF_CACHE_DIR = Path("/home/nishad/LLM_Weights")

ENCODER_API_URL = "http://localhost:8092"

# PMC API calling

LITERATURE_API_PORT = 8093

NCBI_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_EMAIL = "your_email@example.com"
NCBI_TOOL = "local_literature_rag"
NCBI_API_KEY = None  # optional

LITERATURE_CACHE_DIR = PROJECT_ROOT / "Data" / "literature_cache"