FROM python:3.11-slim

WORKDIR /app

# System dependencies jo vector store ya native extensions ke liye chahiye
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv inside container
RUN pip install uv

# Docker layer caching use karne ke liye pehle lockfiles copy karein
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Baqi saara project source code copy karein
COPY . .

# Gradio default interface port expose karein
EXPOSE 7860

# Container starting command inside uv virtual environment context
CMD ["uv", "run", "--frozen", "python", "app.py"]