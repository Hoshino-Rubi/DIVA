"""ONNX execution provider policy. Explicit CUDA must never benchmark CPU by mistake."""
from pathlib import Path

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"


def create_session(model: Path, options, device: str):
    import onnxruntime as ort

    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Unknown ONNX device: {device}")
    providers = [CPU]
    if device != "cpu" and CUDA in ort.get_available_providers():
        # The CUDA provider on Windows may share cuDNN DLLs with installed PyTorch.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        providers = [CUDA, CPU]
    elif device == "cuda":
        raise RuntimeError("CUDAExecutionProvider is unavailable")
    try:
        session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    except RuntimeError:
        if device != "auto" or providers == [CPU]:
            raise
        session = ort.InferenceSession(str(model), sess_options=options, providers=[CPU])
    if device == "cuda" and session.get_providers()[0] != CUDA:
        raise RuntimeError("CUDAExecutionProvider failed to initialize")
    return session
