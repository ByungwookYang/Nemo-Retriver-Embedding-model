FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://nexus.koreacb.com:8443/repository/kcb-pypi-std/simple \
    --trusted-host nexus.koreacb.com \
    -r requirements.txt

COPY main.py .
EXPOSE 8182
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8182"]
