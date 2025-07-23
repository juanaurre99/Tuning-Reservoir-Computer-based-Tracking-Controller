# Use a slim official Python base image with >= 3.11 (needed for type hints like `type | type`)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for cache efficiency
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the source code
COPY . .

# Default command (you can override this in `docker run`)
# Default command: initialize then run experiment
CMD ["bash", "-c", "python run_experiment.py"]
