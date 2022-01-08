###1
FROM python:3.8-slim as builder
WORKDIR /app

COPY src/api /app/api

COPY models /app/models

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc

COPY requirements.txt /app

RUN pip install --no-cache-dir --user -r /app/requirements.txt


####2
FROM  nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local/lib/python3.8/site-packages /usr/local/lib/python3.8/dist-packages

CMD ["uvicorn", "api.api:app --reload"]

EXPOSE 8000
