FROM python:3.9-slim

WORKDIR /app

# 필요한 패키지 설치
RUN pip install transformers torch torchvision pillow fastapi uvicorn python-multipart sentence-transformers numpy

# API 서버 파일 복사
COPY main.py .

# 포트 노출
EXPOSE 8182

# 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8182"]
