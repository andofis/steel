"""ONNX Runtime and TensorRT-RTX adapters for the exported model."""

import ctypes
from contextlib import ExitStack, contextmanager
from hashlib import file_digest

import numpy as np

from .pipeline import TILE, WINDOWS

BACKENDS = ("cpu", "gpu", "tensorrt")
INPUT = "images"
OUTPUTS = ("boxes", "scores", "mask_coefficients", "mask_prototypes")
ONNX_NAME = "yolo_world_seg_m_640_fp32.onnx"
ENGINE_NAME = "yolo_world_seg_m_640_b1-8_fp16.engine"
RTX_VERSION = "1.7.1.116"
CHECKPOINT_ID = "sha256:2b70861ba4fe39076f15f2454bcb202441a51fcb237bad517adb59486115139c"
CHECKPOINT_ARTIFACTS = {
    "873793dddd5fe32809cbc5d089d475944839f05ca0abf612d1dfed119b20f395",
    "fd3cf779f35aec8af5c528453dbfa13ac241a8a6c60e282cbe2e58ae0ed8755f",
}


class OrtRunner:
    def __init__(self, path, gpu):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install onnxruntime-gpu for gpu or onnxruntime for cpu") from exc

        provider = "CUDAExecutionProvider" if gpu else "CPUExecutionProvider"
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"{provider} is unavailable in this ONNX Runtime installation")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        providers = [provider]
        if gpu:
            ort.preload_dlls()
            providers = [(provider, {
                "use_tf32": "0",
                "cudnn_conv_use_max_workspace": "0",
                "arena_extend_strategy": "kSameAsRequested",
            })]
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.session.disable_fallback()
        if self.session.get_providers()[0] != provider:
            raise RuntimeError(f"{provider} failed to initialize; check the CUDA 12 and cuDNN 9 libraries")
        self.batch_size = 8 if gpu else 1

    def __call__(self, inputs):
        return self.session.run(list(OUTPUTS), {INPUT: inputs})


class RtxRunner:
    batch_size = 8

    def __init__(self, path):
        try:
            import tensorrt_rtx as trt
            from cuda.bindings import driver
        except ImportError as exc:
            raise RuntimeError("Install TensorRT-RTX CUDA 13 and cuda-bindings for tensorrt") from exc
        if trt.__version__ != RTX_VERSION:
            raise RuntimeError(f"Engine requires TensorRT-RTX {RTX_VERSION}; found {trt.__version__}")

        self.driver = driver
        self.resources = ExitStack()
        self.cuda(driver.cuInit, 0)
        self.device = self.cuda(driver.cuDeviceGet, 0)[0]
        self.cuda_context = self.cuda(driver.cuDevicePrimaryCtxRetain, self.device)[0]
        self.runtime = self.engine = self.context = None
        try:
            with self.current_device():
                self.logger = trt.Logger(trt.Logger.WARNING)
                self.runtime = trt.Runtime(self.logger)
                self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
                if self.engine is None:
                    raise RuntimeError("Could not deserialize the TensorRT-RTX engine for this GPU and runtime")
                self.context = self.engine.create_execution_context()
                if self.context is None:
                    raise RuntimeError("Could not create the TensorRT-RTX execution context")
                self.stream = self.cuda(driver.cuStreamCreate, 1)[0]  # CU_STREAM_NON_BLOCKING
                self.resources.callback(self.cuda, driver.cuStreamDestroy, self.stream)
                self.set_shape((self.batch_size, 3, TILE, TILE))
                self.buffers = {}
                for name in (INPUT, *OUTPUTS):
                    dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                    if dtype not in (np.float32, np.float16):
                        raise RuntimeError(f"Unexpected TensorRT-RTX data type for {name}: {dtype}")
                    size = self.tensor_shape(name)
                    self.buffers[name] = self.allocate(name, dtype, size)
        except BaseException:
            self.close()
            raise

    def cuda(self, operation, *args):
        code, *result = operation(*args)
        if code != self.driver.CUresult.CUDA_SUCCESS:
            _, name = self.driver.cuGetErrorName(code)
            raise RuntimeError(f"{operation.__name__} failed: {name.decode() if name else code}")
        return result

    @contextmanager
    def current_device(self):
        self.cuda(self.driver.cuCtxPushCurrent, self.cuda_context)
        try:
            yield
        finally:
            self.cuda(self.driver.cuCtxPopCurrent)

    def set_shape(self, shape):
        if not self.context.set_input_shape(INPUT, shape):
            raise RuntimeError(f"TensorRT-RTX rejected the input shape {shape}")

    def tensor_shape(self, name):
        shape = tuple(self.context.get_tensor_shape(name))
        if not shape or min(shape) < 1:
            raise RuntimeError(f"Unresolved TensorRT-RTX shape for {name}: {shape}")
        return shape

    def allocate(self, name, dtype, shape):
        size = int(np.prod(shape)) * dtype.itemsize
        device = self.cuda(self.driver.cuMemAlloc, size)[0]
        self.resources.callback(self.cuda, self.driver.cuMemFree, device)
        host = self.cuda(self.driver.cuMemAllocHost, size)[0]
        self.resources.callback(self.cuda, self.driver.cuMemFreeHost, host)
        memory = (ctypes.c_ubyte * size).from_address(int(host))
        array = np.frombuffer(memory, dtype=dtype)
        if not self.context.set_tensor_address(name, int(device)):
            raise RuntimeError(f"TensorRT-RTX rejected the device address for {name}")
        return device, host, array

    def __call__(self, inputs):
        cuda = self.driver
        with self.current_device():
            self.set_shape(inputs.shape)
            device, host, array = self.buffers[INPUT]
            np.copyto(array[:inputs.size].reshape(inputs.shape), inputs)
            self.cuda(cuda.cuMemcpyHtoDAsync, device, host, inputs.size * array.itemsize, self.stream)
            try:
                if not self.context.execute_async_v3(int(self.stream)):
                    raise RuntimeError("TensorRT-RTX failed to execute the model")
                results = []
                for name in OUTPUTS:
                    device, host, array = self.buffers[name]
                    shape = self.tensor_shape(name)
                    view = array[:int(np.prod(shape))].reshape(shape)
                    self.cuda(cuda.cuMemcpyDtoHAsync, host, device, view.nbytes, self.stream)
                    results.append(view)
            finally:
                self.cuda(cuda.cuStreamSynchronize, self.stream)
            return results

    def close(self):
        try:
            with self.current_device():
                self.context = self.engine = self.runtime = None
                self.resources.close()
        finally:
            self.cuda(self.driver.cuDevicePrimaryCtxRelease, self.device)


def load_backend(backend, model_dir):
    path = model_dir / (ENGINE_NAME if backend == "tensorrt" else ONNX_NAME)
    if not path.is_file():
        raise RuntimeError(f"Model file {path.name} is missing from {model_dir}")

    with path.open("rb") as handle:
        artifact_id = file_digest(handle, "sha256").hexdigest()
    runner = RtxRunner(path) if backend == "tensorrt" else OrtRunner(path, gpu=backend == "gpu")
    runner.model_id = CHECKPOINT_ID if artifact_id in CHECKPOINT_ARTIFACTS else f"sha256:{artifact_id}"
    try:
        output_shapes = ((8400, 4), (8400,), (8400, 32), (32, 160, 160))
        batches = {runner.batch_size, len(WINDOWS) % runner.batch_size} - {0}
        for batch in sorted(batches):
            outputs = runner(np.zeros((batch, 3, TILE, TILE), dtype=np.float32))
            if tuple(output.shape for output in outputs) != tuple((batch, *shape) for shape in output_shapes):
                raise RuntimeError(f"Unexpected {backend} output shapes for the segmentation model")
    except BaseException:
        if isinstance(runner, RtxRunner):
            runner.close()
        raise
    return runner
