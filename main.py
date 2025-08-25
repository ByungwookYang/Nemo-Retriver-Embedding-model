import os
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import torch
import io
import base64
from typing import List, Optional
import numpy as np
from transformers import AutoModel, AutoTokenizer

app = FastAPI()

# 환경변수에서 모델 경로 가져오기
MODEL_PATH = os.getenv("MODEL_PATH", "/models/models--nvidia--llama-nemoretriever-colembed-3b-v1")

# 모델 로딩 (컨테이너 시작시 1번만)
model = AutoModel.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

@app.post("/embeddings")
async def create_embeddings(
    texts: Optional[List[str]] = None,
    images: Optional[List[UploadFile]] = None
):
    """텍스트와/또는 이미지를 임베딩 벡터로 변환"""
    
    results = []
    
    # 텍스트 처리
    if texts:
        for text in texts:
            embedding = model.encode([text])
            results.append({
                "type": "text",
                "content": text,
                "embedding": embedding.tolist()
            })
    
    # 이미지 처리
    if images:
        for img_file in images:
            image_data = await img_file.read()
            image = Image.open(io.BytesIO(image_data))
            
            embedding = model.encode([image])
            results.append({
                "type": "image", 
                "filename": img_file.filename,
                "embedding": embedding.tolist()
            })
    
    return {"embeddings": results}

@app.post("/similarity")
async def calculate_similarity(
    query_text: Optional[str] = None,
    query_image: Optional[UploadFile] = None,
    candidate_embeddings: List[List[float]] = []
):
    """쿼리와 후보들 간의 유사도 계산"""
    
    # 쿼리 임베딩 생성
    query_embedding = None
    
    if query_text:
        query_embedding = model.encode([query_text])
    elif query_image:
        image_data = await query_image.read()
        image = Image.open(io.BytesIO(image_data))
        query_embedding = model.encode([image])
    
    if query_embedding is None:
        return {"error": "쿼리 텍스트 또는 이미지 필요"}
    
    # 유사도 계산
    similarities = []
    for i, candidate in enumerate(candidate_embeddings):
        candidate_tensor = torch.tensor([candidate])
        similarity = torch.cosine_similarity(query_embedding, candidate_tensor)
        similarities.append({
            "index": i,
            "similarity": float(similarity)
        })
    
    # 유사도 순으로 정렬
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {"similarities": similarities}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": f"loaded from {MODEL_PATH}"}
