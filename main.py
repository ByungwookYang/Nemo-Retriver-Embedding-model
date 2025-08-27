from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import torch
import uvicorn
import numpy as np
from typing import List
import json

app = FastAPI()

# 모델 경로를 Python path에 추가
model_path = "/models"
sys.path.append(model_path)

print(f"Loading model from: {model_path}")
print(f"Files in model directory: {os.listdir(model_path)}")

# 설정 파일 확인
with open(os.path.join(model_path, "config.json"), "r") as f:
    config = json.load(f)
    print(f"Model config: {config}")

# GPU 체크
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델과 토크나이저 전역 변수로 선언
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    try:
        # 커스텀 모델링 파일들 import
        from transformers import AutoConfig, AutoTokenizer
        
        # trust_remote_code=True로 커스텀 코드 허용
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 모델 로드 시도 1: AutoModel
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True, config=config)
            print("✅ AutoModel로 로드 성공")
        except Exception as e:
            print(f"AutoModel 실패: {e}")
            
            # 모델 로드 시도 2: 직접 클래스 import
            try:
                from modeling_llama_nemoretrievercolembed import LlamaNemoRetrieverColembedModel
                model = LlamaNemoRetrieverColembedModel.from_pretrained(model_path)
                print("✅ 커스텀 클래스로 로드 성공")
            except Exception as e2:
                print(f"커스텀 클래스도 실패: {e2}")
                raise Exception("모든 모델 로딩 방법 실패")
        
        model = model.to(device)
        model.eval()
        print("✅ 모델 로딩 완료")
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        raise e

class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str = "nvidia-colembed"
    encoding_format: str = "float"

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """텍스트를 임베딩 벡터로 변환"""
    try:
        # 토크나이저로 인코딩
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 모델 추론
        with torch.no_grad():
            outputs = model(**inputs)
            
            # 다양한 출력 형태 처리
            if hasattr(outputs, 'last_hidden_state'):
                # 일반적인 경우: last_hidden_state 평균
                embeddings = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, 'pooler_output'):
                # pooler_output 있는 경우
                embeddings = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                # 직접 tensor 반환하는 경우
                embeddings = outputs
            else:
                # 다른 형태인 경우 첫 번째 출력 사용
                embeddings = outputs[0].mean(dim=1)
        
        # CPU로 이동하고 리스트로 변환
        embeddings = embeddings.cpu().numpy().tolist()
        return embeddings
        
    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    load_model()

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    try:
        # 입력을 리스트로 변환
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        # 임베딩 생성
        embeddings = get_embeddings(texts)
        
        # OpenAI API 형식으로 응답
        data = []
        for i, embedding in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
        
        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        }
    
    except Exception as e:
        print(f"❌ API 에러: {e}")
        return {"error": str(e)}, 500

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/")
async def root():
    return {"message": "NVIDIA ColEmbed Embedding API Server"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8182)
