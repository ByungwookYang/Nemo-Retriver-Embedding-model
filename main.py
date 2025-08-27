import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

# 전역 변수
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    
    model_path = "/app/model"
    logger.info(f"Loading model from: {model_path}")
    
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        attn_implementation="eager"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    logger.info("✅ Model loaded!")

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
