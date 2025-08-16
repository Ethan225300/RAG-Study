import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY

# 切分与检索参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

# （FAISS 本地目录仍保留，如需回滚可继续用）
VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-demo-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")      # 'aws' 或 'gcp'
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_DIM = 1536  # 对应 text-embedding-3-small

