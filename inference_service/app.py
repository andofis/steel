"""HTTP entry point for local panorama inference."""

import asyncio
import logging
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse
from starlette.routing import Route

from .backends import BACKENDS, load_backend
from .pipeline import MaskBudgetExceeded, segment

LOG = logging.getLogger(__name__)
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models" / "pipe_hole_yolo_world_seg_m_v1" / "ONNX_TensorRT_FP16"
)
PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 32 * 1024 * 1024
MAX_DIMENSION = 16384


async def read_png(request):
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type != "image/png":
        raise HTTPException(415, "Content-Type must be image/png")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        if not re.fullmatch(r"[0-9]{1,20}", content_length):
            raise HTTPException(400, "Invalid Content-Length")
        if int(content_length) > MAX_BODY_BYTES:
            raise HTTPException(413, "PNG body exceeds the upload limit")

    payload = bytearray()
    try:
        async with asyncio.timeout(30):
            async for chunk in request.stream():
                if len(payload) + len(chunk) > MAX_BODY_BYTES:
                    raise HTTPException(413, "PNG body exceeds the upload limit")
                payload.extend(chunk)
    except TimeoutError as exc:
        raise HTTPException(408, "PNG upload timed out") from exc
    except ClientDisconnect as exc:
        raise HTTPException(400, "PNG upload interrupted") from exc
    return payload


def infer(payload, backend, cut_x, runner):
    start = perf_counter()
    if len(payload) < 33 or not payload.startswith(PNG_HEADER):
        raise HTTPException(400, "Body must be a valid PNG with an IHDR header")
    width, height = struct.unpack_from(">II", payload, 16)
    if max(width, height) > MAX_DIMENSION or width * height > MAX_PIXELS:
        raise HTTPException(413, "PNG dimensions exceed the decode limit")
    if min(width, height) < 2:
        raise HTTPException(422, "PNG must be at least 2 pixels wide and high")
    if not 0 <= cut_x < width:
        raise HTTPException(422, f"cut_x must be between 0 and {width - 1}")
    try:
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error as exc:
        raise HTTPException(400, "PNG could not be decoded") from exc
    if image is None or image.shape != (height, width):
        raise HTTPException(400, "PNG could not be decoded")
    try:
        holes = segment(image, cut_x, runner)
    except MaskBudgetExceeded as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        LOG.exception("%s inference failed", backend)
        raise HTTPException(500, f"{backend} inference failed: {exc}") from exc
    result = {
        "width": width,
        "height": height,
        "backend": backend,
        "elapsed_ms": round((perf_counter() - start) * 1000, 2),
        "holes": holes,
    }
    if getattr(runner, "model_id", None):
        result["model_id"] = runner.model_id
    return result


def create_app(enabled=None, model_dir=None, *, load_runner=load_backend):
    if enabled is None:
        enabled = tuple(part.strip() for part in os.getenv("PIPE_INFERENCE_BACKENDS", "cpu").split(","))
    if not enabled or set(enabled) - set(BACKENDS) or len(set(enabled)) != len(enabled):
        raise ValueError("PIPE_INFERENCE_BACKENDS must list one or more of cpu,gpu,tensorrt without duplicates")
    if model_dir is None:
        model_dir = Path(os.getenv("PIPE_INFERENCE_MODEL_DIR", str(DEFAULT_MODEL_DIR)))

    @asynccontextmanager
    async def lifespan(app):
        cv2.setNumThreads(2)
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipe-inference")
        runners = {}
        statuses = {name: {"available": False, "detail": "Backend is disabled"} for name in BACKENDS}
        app.state.runners = runners
        app.state.statuses = statuses
        app.state.executor = executor
        app.state.inflight = asyncio.Lock()

        def close_runners():
            for name, runner in runners.items():
                if hasattr(runner, "close"):
                    try:
                        runner.close()
                    except Exception:
                        LOG.exception("Could not close %s backend", name)

        try:
            for name in enabled:
                try:
                    runner = await loop.run_in_executor(executor, load_runner, name, model_dir)
                except Exception as exc:
                    statuses[name]["detail"] = str(exc) or type(exc).__name__
                    LOG.warning("%s backend unavailable: %s", name, statuses[name]["detail"])
                else:
                    runners[name] = runner
                    statuses[name] = {"available": True}
                    if getattr(runner, "model_id", None):
                        statuses[name]["model_id"] = runner.model_id
                    LOG.info("%s backend passed its startup probe", name)
            yield
        finally:
            try:
                await loop.run_in_executor(executor, close_runners)
            finally:
                executor.shutdown()

    async def health(request):
        state = request.app.state
        ready = bool(state.runners)
        return JSONResponse(
            {"status": "ok" if ready else "unavailable", "backends": state.statuses},
            status_code=200 if ready else 503,
        )

    async def segment_route(request):
        params = request.query_params
        backends = params.getlist("backend")
        if len(backends) != 1 or backends[0] not in BACKENDS:
            raise HTTPException(422, "Specify one backend: cpu, gpu, or tensorrt")
        backend = backends[0]
        cuts = params.getlist("cut_x")
        if len(cuts) != 1 or not re.fullmatch(r"[+-]?[0-9]{1,10}", cuts[0]):
            raise HTTPException(422, "Specify one integer cut_x")

        state = request.app.state
        if backend not in state.runners:
            raise HTTPException(503, f"{backend} unavailable: {state.statuses[backend]['detail']}")
        if state.inflight.locked():
            raise HTTPException(503, "Inference service is busy", headers={"Retry-After": "1"})
        async with state.inflight:
            payload = await read_png(request)
            response = await asyncio.get_running_loop().run_in_executor(
                state.executor, infer, payload, backend, int(cuts[0]), state.runners[backend]
            )
            return JSONResponse(response)

    async def http_error(_request, exc):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    return Starlette(
        routes=[Route("/health", health), Route("/v1/segment", segment_route, methods=["POST"])],
        lifespan=lifespan,
        exception_handlers={HTTPException: http_error},
    )


app = create_app()
