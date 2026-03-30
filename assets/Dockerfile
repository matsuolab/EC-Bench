FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive

# ubuntu22.04にpython3.11をいれるにはppaを追加しないとダメ

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common curl gnupg git git-lfs sudo ffmpeg \
        build-essential cmake && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    # ---- ここで Python 3.11 系をまとめて取得 ----
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils python3.11-dev && \
    # pip を追加
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 - && \
    ln -s /usr/bin/python3.11 /usr/local/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

#パッケージのインストール
RUN pip3 install --no-cache-dir -U pip && \
    pip install numpy==1.26.4 torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 accelerate==0.30.1 bitsandbytes==0.43.1 moviepy==1.0.3 

RUN pip install git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d

RUN pip install timm ftfy regex einops fvcore eva-decord==0.6.1 iopath matplotlib types-regex cartopy \
                ctranslate2==4.4.0 faster_whisper==1.0.3 neo4j hnswlib xxhash nano-vectordb \
                transformers==4.37.1 tiktoken openai tenacity ollama

