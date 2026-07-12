# Dockerfile for PlumeTrace Backend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY app/ ./app/
COPY mock_sensors.py .
COPY passwords.txt .
COPY amqtt.yaml .

EXPOSE 8000
