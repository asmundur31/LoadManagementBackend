# Use the official Python image from the DockerHub
FROM python:3.13-slim AS app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code
COPY . /app/

# Set the environment variable to your app's entry point
ENV PYTHONPATH=/app

# Stage 2: Add Nginx for reverse proxy
FROM nginx:alpine

# Copy Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start both Nginx and the app
CMD ["nginx", "-g", "daemon off;"]