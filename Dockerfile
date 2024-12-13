# Use the official Python image from the DockerHub
FROM python:3.11-slim

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

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
