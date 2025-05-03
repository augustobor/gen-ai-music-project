# Use an official Python image as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install Python dependencies
RUN pip install --no-cache-dir --default-timeout=10000 --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --default-timeout=10000 -r requirements.txt

COPY requirements2.txt .
RUN pip install --no-cache-dir --default-timeout=10000 -r requirements2.txt

# Copy the rest of the application code into the container
COPY . .

ENV PYTHONPATH=/app

# Expose the port the Flask app will run on
EXPOSE 8000

# Set the default command to run the Flask server
CMD ["python","-m","src.main"]