# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only build of torch FIRST, from PyTorch's own index.
# sentence-transformers depends on torch, and without this, pip pulls in
# the default GPU/CUDA build -- which drags in several gigabytes of
# NVIDIA driver packages that are completely unused here (this container
# has no GPU access anyway). This cuts the download size drastically and
# saves a lot of build time.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build-time data preparation: load the CSV into DuckDB and ingest the
# PDFs into ChromaDB during the build, so the image is immediately
# usable the moment a container starts from it.
RUN python -m src.sql_tool.db_loader
RUN python -m scripts.ingest_pdfs

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]