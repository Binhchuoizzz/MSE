FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    mlflow==2.17.2 \
    psycopg2-binary==2.9.10 \
    boto3==1.37.14 \
    gunicorn==21.2.0

# Create working directory
WORKDIR /mlflow

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - will be overridden by docker-compose
# CMD ["mlflow", "server"]