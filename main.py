import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    
    # 볼륨 마운트된 경로
    model_path = "/app/models"  # 경로 수정
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # 파일 목록 확인
    files = os.listdir(model_path)
    logger.info(f"Files in model directory: {files}")
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # 더 안전한 로딩
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # 일단 CPU로 시작
            local_files_only=True,
            attn_implementation="eager",
            use_safetensors=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False  # slow tokenizer 사용
        )
        
        logger.info("✅ Model loaded!")
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        # config.json 내용 확인
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
                logger.info(f"Config contents: {config}")
        raise

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
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
