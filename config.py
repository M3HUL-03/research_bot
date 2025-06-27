import os

class Config:
    MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
    COLLECTION_NAME = "doc_chat"
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 400
    DENSE_DIM = 1024
    MISTRAL_MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH", r"D:\new_chatbot\mistral-7b-v0.1.Q4_K_M.gguf")


