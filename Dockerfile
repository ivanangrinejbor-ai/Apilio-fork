# syntax=docker/dockerfile:1
FROM python:3.12-trixie

# Expose the required port
EXPOSE 6969

# Set up working directory
WORKDIR /app

# Install system dependencies, clean up cache to keep image size small
RUN apt update && \
    apt install -y -qq ffmpeg && \
    apt install -y -qq libportaudio2 && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy application files into the container
COPY . .

# Create a virtual environment in the app directory and install dependencies
# torch/torchaudio are installed separately below (CUDA build), so exclude them from requirements.txt
RUN python3 -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir python-ffmpeg && \
    pip install --no-cache-dir -r <(grep -vE '^torch' requirements.txt) && \
    pip install --no-cache-dir torch==2.7.1 torchvision torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Define volumes for persistent storage
VOLUME ["/app/logs/"]

# Set environment variables if necessary
ENV PATH="/app/.venv/bin:$PATH"

# Run the app (authentication is enabled when APP_USERNAME/APP_PASSWORD are set)
ENTRYPOINT ["sh", "/app/entrypoint.sh"]
