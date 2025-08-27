import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer, AutoConfig
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
    
    # config.json 내용 확인 및 출력
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            logger.info(f"Config contents: {config}")
            
            # model_type이 없으면 추가
            if 'model_type' not in config:
                logger.info("Adding model_type to config...")
                config['model_type'] = 'llama'  # 기본값으로 llama 설정
                
                # 임시로 config 수정
                with open(config_path, 'w') as fw:
                    json.dump(config, fw, indent=2)
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # 더 직접적인 방법으로 로딩
        from transformers.models.auto.modeling_auto import MODEL_FOR_PRETRAINING_MAPPING
        
        # 커스텀 모델 클래스 직접 import
        import sys
        sys.path.append(model_path)
        
        # 모델 로딩 시도
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            local_files_only=True,
            attn_implementation="eager",
            use_safetensors=True,
            revision=None,
            force_download=False
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False
        )
        
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        
        # 마지막 수단: 모델명으로 직접 로딩 시도
        try:
            logger.info("Trying to load with model name instead of path...")
            model = AutoModel.from_pretrained(
                "nvidia/llama-nemoretriever-colembed-3b-v1",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                cache_dir=model_path,
                local_files_only=True,
                attn_implementation="eager"
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                "nvidia/llama-nemoretriever-colembed-3b-v1",
                trust_remote_code=True,
                cache_dir=model_path,
                local_files_only=True
            )
            
            logger.info("✅ Model loaded with model name!")
            
        except Exception as e2:
            logger.error(f"Both methods failed: {e2}")
            raise e

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
