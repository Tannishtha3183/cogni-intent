FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration
COPY pyproject.toml ./

# Install python dependencies using pip and pyproject.toml
RUN pip install --no-cache-dir .[dev]

# Copy the rest of the application
COPY . .

# Create data directory for ChromaDB
RUN mkdir -p data/chroma_db

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "main_simulation.py"]
