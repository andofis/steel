# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm AS source
WORKDIR /context
COPY . .
RUN mkdir -p /source/inference_service && \
    if [ -d inference_service ]; then cd inference_service; fi && \
    cp *.py requirements.txt /source/inference_service/

FROM python:3.11-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_CACHE_PATH=/tmp/steel-cuda-cache \
    PIPE_INFERENCE_MODEL_DIR=/models
WORKDIR /service
COPY --from=source /source/inference_service/requirements.txt /service/inference_service/requirements.txt
RUN python -m pip install --no-cache-dir -r inference_service/requirements.txt
COPY --from=source /source/inference_service/ /service/inference_service/
COPY --from=geometry tile_detection.py /service/tube_unwrap/tile_detection.py
USER 65532:65532
EXPOSE 8766
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8766/health', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "inference_service.app:app", "--host", "0.0.0.0", "--port", "8766", "--workers", "1", "--limit-concurrency", "16"]

FROM base AS gpu
USER root
RUN python -m pip install --no-cache-dir 'onnxruntime-gpu[cuda,cudnn]==1.23.2'
ENV PIPE_INFERENCE_BACKENDS=cpu,gpu
USER 65532:65532

FROM base AS tensorrt-dependencies
USER root
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --prefix /opt/tensorrt \
      tensorrt-rtx==1.7.1.116 cuda-bindings==13.0.3

FROM gpu AS tensorrt
COPY --from=tensorrt-dependencies /opt/tensorrt/ /usr/local/
ENV PIPE_INFERENCE_BACKENDS=tensorrt,gpu,cpu

FROM tensorrt AS release
COPY --from=weights --chown=65532:65532 \
    yolo_world_seg_m_640_fp32.onnx yolo_world_seg_m_640_b1-8_fp16.engine /models/

FROM base AS cpu
USER root
RUN python -m pip install --no-cache-dir onnxruntime==1.23.2
ENV PIPE_INFERENCE_BACKENDS=cpu
USER 65532:65532
