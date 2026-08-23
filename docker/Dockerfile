FROM python:3.11-slim

WORKDIR /app

# 상위 폴더(DanAI)에 있는 requirements.txt 및 소스 코드 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]