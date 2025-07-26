FROM python:3.11-slim

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        && rm -rf /var/lib/apt/lists/*

# Copy the project into the container.  We copy the entire repository
# under /app and then work inside the rag-industrial-eval-guardrails
# subdirectory where the pyproject.toml resides.
WORKDIR /app
COPY . /app

# Change into the project directory and install dependencies.  Using
# `pip install .[dev]` here installs both the runtime and development
# dependencies; remove the `[dev]` suffix in production to minimise
# image size.
WORKDIR /app/rag-industrial-eval-guardrails
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .[dev]

EXPOSE 8000

# Default command: run uvicorn within the project directory.  We
# reference `app.main` because the working directory adds this
# directory to sys.path.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]