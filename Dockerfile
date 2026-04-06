FROM python:3.13-slim

# Copy uv from the official astral-sh Docker image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy dependency files first to cache this layer
COPY pyproject.toml uv.lock ./

# Install dependencies securely without development dependencies
RUN uv sync --frozen --no-dev

# Copy the rest of the application source code
COPY . .

# Set PYTHONPATH to point to the src directory so python can find the modules
ENV PYTHONPATH="src"

# Expose the port the FastAPI app will run on
EXPOSE 8000

# Start the uvicorn server via uv run
CMD ["uv", "run", "uvicorn", "vocalog_ai_api.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
