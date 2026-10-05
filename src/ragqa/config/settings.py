import os
from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen-max")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = DASHSCOPE_API_KEY  # 向后兼容旧引用

# Embedding
EMBED_MODE = os.getenv("EMBED_MODE", "api")  # "api" or "local"
EMBED_API_MODEL = os.getenv("EMBED_API_MODEL", "text-embedding-async-v2")
EMBED_LOCAL_MODEL = os.getenv("EMBED_LOCAL_MODEL", "models/bge-m3")

# Rerank
RERANK_MODE = os.getenv("RERANK_MODE", "api")  # "api" or "local"
RERANK_API_MODEL = os.getenv("RERANK_API_MODEL", "gte-rerank-v2")
RERANK_LOCAL_MODEL = os.getenv("RERANK_LOCAL_MODEL", "models/bge-reranker-base")

# FAISS
FAISS_PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "faiss_db")
DATA_DIR = os.getenv("DATA_DIR", "data")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
PAPERS_DIR = os.getenv("PAPERS_DIR", "data/papers")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
