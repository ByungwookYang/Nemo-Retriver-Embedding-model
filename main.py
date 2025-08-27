import os
import sys
import torch
from flask import Flask, request, jsonify
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    
    model_path = "/app/models"
    
    # 모델 경로를 Python path에 추가
    sys.path.insert(0, model_path)
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # 커스텀 모델 클래스 직접 임포트
        from modeling_llama_nemoretrievercolembed import llama_NemoRetrieverColEmbed, llama_NemoRetrieverColEmbedConfig
        from transformers import AutoTokenizer
        
        # Config 로드
        config = llama_NemoRetrieverColEmbedConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # 모델 직접 로드
        model = llama_NemoRetrieverColEmbed.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # 일단 CPU로
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False
        )
        
        logger.info("✅ Model loaded successfully!")
        
    except ImportError as e:
        logger.error(f"Failed to import custom model class: {e}")
        
        # Fallback: AutoModel with trust_remote_code
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # transformers 최신 버전 필요할 수 있음
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                local_files_only=True,
                revision=None,
                force_download=False,
                ignore_mismatched_sizes=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            logger.info("✅ Model loaded with AutoModel fallback!")
            
        except Exception as e2:
            logger.error(f"All loading methods failed: {e2}")
            raise

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # ColBERT 스타일이므로 last_hidden_state 사용
        if hasattr(outputs, 'last_hidden_state'):
            embedding = outputs.last_hidden_state.mean(dim=1)
        else:
            # 다른 출력 형태일 수 있음
            embedding = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.mean(dim=1)
            
        return embedding[0].cpu().float().numpy().tolist()

@app.route('/v1/embeddings', methods=['POST'])
def create_embeddings():
    data = request.json
    input_text = data['input']
    
    if isinstance(input_text, str):
        texts = [input_text]
    else:
        texts = input_text
    
    embeddings_list = []
    for text in texts:
        embedding = generate_embedding(text)
        embeddings_list.append(embedding)
    
    response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": emb,
                "index": i
            } for i, emb in enumerate(embeddings_list)
        ],
        "model": "nvidia-colembed-3b",
        "usage": {
            "prompt_tokens": sum(len(tokenizer.encode(t)) for t in texts),
            "total_tokens": sum(len(tokenizer.encode(t)) for t in texts)
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8201)
