FROM python:3.9-slim

WORKDIR /app

# requirements 복사 및 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# API 서버 파일 복사
COPY main.py .

# 포트 노출
EXPOSE 8182

# 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8182"]
