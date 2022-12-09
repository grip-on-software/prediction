ARG TENSORFLOW_VERSION=1.13.2-py3
FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}

# Install dependencies
COPY requirements.txt /tmp/

RUN rm /etc/apt/source.list.d/cuda.list && apt-get update && \
	apt-get install -y --no-install-recommends git g++ libffi-dev libssl-dev && \
    cd /tmp/ && sed -i 's/^tensorflow/#tensorflow/' requirements.txt && \
    pip install -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
