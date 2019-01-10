FROM tensorflow/tensorflow:1.12.0-py3

# Install dependencies
COPY requirements.txt /tmp/

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    cd /tmp/ && pip install -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
