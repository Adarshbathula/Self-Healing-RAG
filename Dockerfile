FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p documents faiss_index

EXPOSE 7860

ENV FAISS_INDEX_PATH=faiss_index
ENV DOCUMENTS_DIR=documents

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
