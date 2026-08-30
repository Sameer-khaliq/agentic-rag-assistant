FROM python:3.12-slim

WORKDIR /app

# System dependencies for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv using official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency definition and lockfiles first for optimal layer caching
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-cache

# Copy the rest of the application files
COPY . .

# Expose Gradio default interface port
EXPOSE 7860

# Run Gradio application inside uv environment
CMD ["uv", "run", "--frozen", "python", "app.py"]