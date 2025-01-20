# Use the official Python image from the DockerHub
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies, including Nginx
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && curl -fsSL https://nginx.org/packages/mainline/debian/ | tee /etc/apt/sources.list.d/nginx.list \
    && apt-get update \
    && apt-get install -y nginx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code
COPY . /app/

# Expose the FastAPI and Nginx ports
EXPOSE 8000
EXPOSE 80

# Copy Nginx configuration file
COPY nginx.conf /etc/nginx/nginx.conf

# Command to start both Uvicorn and Nginx
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & nginx -g 'daemon off;'"]
