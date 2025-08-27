from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()

# 로컬 모델 경로 사용
model_path = "/models"  # 컨테이너 내부 경로
model = SentenceTransformer(model_path)

class EmbeddingRequest(BaseModel):
    input: str
    model: str = "nvidia-colembed"

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    embedding = model.encode([request.input])
    
    return {
        "object": "list",
        "data": [{
            "object": "embedding", 
            "embedding": embedding[0].tolist(),
            "index": 0
        }],
        "model": request.model
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
