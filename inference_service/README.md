# Local panorama inference

The capture application uses `http://127.0.0.1:8766` by default. Set `PIPE_INFERENCE_URL` on the capture application to use another address.

## Docker Desktop

Use the Linux engine. CUDA and TensorRT-RTX require NVIDIA GPU access in Docker Desktop's WSL 2 backend.

Run from the repository root in PowerShell. The Dockerfile allows only the service sources and requirements in the primary build context. The named context supplies the shared cylindrical geometry module.

```powershell
$models = (Resolve-Path '.\models\pipe_hole_yolo_world_seg_m_v1\ONNX_TensorRT_FP16').Path
docker build --target cpu --build-context geometry=./tube_unwrap -f inference_service/Dockerfile -t steel-inference:cpu .
docker run --rm -p 127.0.0.1:8766:8766 --mount "type=bind,source=$models,target=/models,readonly" steel-inference:cpu
```

To build from the `inference_service` directory, use this equivalent command. Capture `$models` from the repository root before changing directories.

```powershell
docker build --target cpu --build-context geometry=../tube_unwrap -t steel-inference:cpu .
```

Use target `gpu` for FP32 ONNX Runtime with CPU and CUDA. Use target `tensorrt` for all three backends. For example, from the repository root run the TensorRT-RTX engine by itself.

```powershell
docker build --target tensorrt --build-context geometry=./tube_unwrap -f inference_service/Dockerfile -t steel-inference:tensorrt .
docker run --rm --gpus all -p 127.0.0.1:8766:8766 -e PIPE_INFERENCE_BACKENDS=tensorrt --mount "type=bind,source=$models,target=/models,readonly" steel-inference:tensorrt
```

The bundled engine requires TensorRT-RTX 1.7.1.116 for CUDA 13 and a compatible RTX GPU and driver. An incompatible engine or missing runtime is reported by the backend's readiness entry. [TensorRT-RTX compatibility](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/engine-compatibility.html)

### Self-contained release image

From the repository root, build the `release` target to include the FP32 ONNX model and the TensorRT-RTX FP16 engine. The running container reads both from `/models`.

The default startup order is TensorRT-RTX, ONNX CUDA, then CPU. This lets TensorRT-RTX reserve its buffers before ONNX CUDA warms its memory allocator.

```powershell
$models = (Resolve-Path '.\models\pipe_hole_yolo_world_seg_m_v1\ONNX_TensorRT_FP16').Path
docker buildx build --load --platform linux/amd64 --target release --build-context geometry=./tube_unwrap --build-context "weights=$models" -f inference_service/Dockerfile -t steel-inference:world-seg-m-v1 inference_service
docker run --rm --gpus all -p 127.0.0.1:8766:8766 steel-inference:world-seg-m-v1
```

## Native Windows

Use a Python 3.11 virtual environment. From the repository root install the HTTP service and the backend dependencies you intend to enable.

```powershell
python -m pip install -r inference_service/requirements.txt
python -m pip install onnxruntime==1.23.2
python -m uvicorn inference_service.app:app --host 127.0.0.1 --port 8766
```

For CUDA, install `onnxruntime-gpu[cuda,cudnn]==1.23.2` in place of the CPU distribution. The GPU distribution supplies both providers. Enable it with `$env:PIPE_INFERENCE_BACKENDS = 'cpu,gpu'`. The service requires the CUDA execution provider for `gpu`, disables session fallback and TF32, and uses ONNX Runtime's standard CPU placement for shape operations. [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

For TensorRT-RTX, install `tensorrt-rtx==1.7.1.116` and `cuda-bindings==13.0.3`. The existing Python 3.11 training setup can reuse its `datasets/_training_work/yolo_world_seg/rtx_runtime` bindings via `PYTHONPATH`. Enable the engine with `$env:PIPE_INFERENCE_BACKENDS = 'tensorrt'` and use the same Uvicorn command.

The model directory defaults to `models/pipe_hole_yolo_world_seg_m_v1/ONNX_TensorRT_FP16`. Set `PIPE_INFERENCE_MODEL_DIR` to use a different directory with the same artifact names.

## HTTP contract

```powershell
curl.exe -sS http://127.0.0.1:8766/health
curl.exe -sS -H 'Content-Type: image/png' --data-binary '@D:\path\to\display.png' 'http://127.0.0.1:8766/v1/segment?backend=cpu&cut_x=0'
```

- `GET /health` returns HTTP 200 when at least one configured backend passed model initialization and inference probes with the production batch sizes, or 503 when all are unavailable. Inspect `backends.<name>.available` for the backend you intend to use. A failed or disabled backend includes a `detail` field.
- `POST /v1/segment` accepts the original display PNG, with an explicit backend and the integer `cut_x` used to publish that display. The service restores the calibrated origin before inference and returns coordinates in the display image. Use `cut_x=0` for a display published at the calibrated origin.
- The result has `width`, `height`, `backend`, `model_id`, `elapsed_ms`, and `holes`. Each hole includes `score`, `center`, `contours`, and `area`. A seam-crossing hole has a single center and area with contours on both sides. Coordinates are fractional display pixels and area is foreground mask area in display square pixels.
- The supplied FP32 ONNX and TensorRT-RTX artifacts report the same checkpoint SHA-256 as `model_id`. Other artifacts report their own file SHA-256.
- The service accepts PNGs up to 32 MiB encoded and 32 megapixels decoded, with each dimension between 2 and 16,384 pixels. A single segmentation request runs at a time. A busy or unavailable backend yields HTTP 503, and errors contain a JSON `detail`.

To run the service tests, install `httpx==0.28.1` alongside the service requirements and run `python -m unittest discover -s tests -p test_inference_service.py -v` from the repository root.
