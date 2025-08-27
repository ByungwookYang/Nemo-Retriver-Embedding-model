FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
--index-url https://nexus.koreacb.com:8443/repository/kcb-pypi-std/simple \
--trusted-host nexus.koreacb.com \
-r requirements.txt

COPY main.py .
RUN mkdir -p /app/model

EXPOSE 8201

CMD ["python3", "main.py"]
