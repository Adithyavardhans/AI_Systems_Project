FROM python:3.10-slim

# Work directory inside container
WORKDIR /app

# Install system dependencies (optional but safe for numpy, pillow, opencv)
RUN apt-get update && apt-get install -y \
    python3-dev gcc libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements from src folder
COPY ../src/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ../src /app

# Copy Prometheus config
COPY ../monitoring/prometheus.yml /etc/prometheus/prometheus.yml

# Expose ports
EXPOSE 7860
EXPOSE 8000

# Run the Gradio app
CMD ["python", "app.py"]
