FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install -U pip && pip install -r /tmp/requirements.txt --no-cache-dir

COPY . /src

WORKDIR /src
