OLLAMA_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pdf_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K = 3

# PostgreSQL
POSTGRES_URL = "postgresql+psycopg://raguser:ragpassword@localhost:5432/ragdb"

# Redis
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_TTL = 3600  # время жизни кэша в секундах (1 час)

# FastAPI
API_HOST = "0.0.0.0"
API_PORT = 8000
