FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONPATH=/workspace/ORTrack

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements_docker.txt /workspace/requirements_docker.txt
RUN pip install --no-cache-dir -r /workspace/requirements_docker.txt

COPY code/ORTrack /workspace/ORTrack
COPY model/ORTrack_ep0008.pth.tar /workspace/model/ORTrack_ep0008.pth.tar
COPY run_inference.sh /workspace/run_inference.sh
COPY run_evaluation.sh /workspace/run_evaluation.sh

RUN chmod +x /workspace/run_inference.sh /workspace/run_evaluation.sh

CMD ["/workspace/run_inference.sh"]
