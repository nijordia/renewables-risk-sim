# Renewables Risk Simulator
# Analyzes how high renewable penetration drives electricity price volatility
# in Spain's grid using REData API data

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY utils.py .
COPY data_fetch.py .
COPY analysis.py .
COPY main.py .

# Create output directories
RUN mkdir -p /data /outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - run full pipeline for 2024
ENTRYPOINT ["python", "main.py"]
CMD ["all", "--start-date", "2024-01-01", "--end-date", "2025-12-31"]
