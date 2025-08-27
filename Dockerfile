FROM python:3.9-slim

WORKDIR /app
RUN pip install fastapi uvicorn sentence-transformers torch
COPY main.py .
EXPOSE 8182
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
