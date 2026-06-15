import logging
import os
import sysconfig
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CUDA_DLL_HANDLES: list[Any] = []
_CUDA_DLL_PATHS_PREPARED = False


def prepare_onnx_cuda_runtime() -> list[Path]:
    """
    Make NVIDIA CUDA/cuDNN wheel DLLs visible before ONNX Runtime sessions load.
    """
    global _CUDA_DLL_PATHS_PREPARED

    site_packages = Path(sysconfig.get_paths()["purelib"])
    nvidia_root = site_packages / "nvidia"

    dll_dirs = [
        nvidia_root / "cudnn" / "bin",
        nvidia_root / "cublas" / "bin",
        nvidia_root / "cuda_runtime" / "bin",
        nvidia_root / "cufft" / "bin",
        nvidia_root / "curand" / "bin",
        nvidia_root / "cuda_nvrtc" / "bin",
        nvidia_root / "nvjitlink" / "bin",
    ]

    existing_dirs = [path for path in dll_dirs if path.exists()]

    if not _CUDA_DLL_PATHS_PREPARED:
        for path in existing_dirs:
            try:
                handle = os.add_dll_directory(str(path))
                _CUDA_DLL_HANDLES.append(handle)
            except Exception:
                logger.debug("Could not add CUDA DLL directory: %s", path, exc_info=True)

        current_path = os.environ.get("PATH", "")
        prepend = ";".join(str(path) for path in existing_dirs)
        if prepend:
            os.environ["PATH"] = prepend + ";" + current_path

        _CUDA_DLL_PATHS_PREPARED = True

    return existing_dirs


def select_onnx_providers(device: str) -> list[str]:
    import onnxruntime as ort

    requested = (device or "cpu").strip().lower()

    if requested == "cpu":
        return ["CPUExecutionProvider"]

    if requested in {"gpu", "cuda"}:
        prepare_onnx_cuda_runtime()

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")

        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "GPU requested, but CUDAExecutionProvider is not available. "
                "Install/configure onnxruntime-gpu + CUDA, or use --device cpu."
            )

        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    raise ValueError(f"Unsupported device={device!r}. Expected: cpu, gpu.")


def verify_onnx_session_provider(
    session: Any,
    providers: list[str],
    model_name: str,
) -> None:
    requested_cuda = "CUDAExecutionProvider" in providers
    active_providers = set(session.get_providers())

    if requested_cuda and "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            f"CUDA requested but {model_name} session providers are "
            f"{session.get_providers()}"
        )


def create_onnx_session(
    model_path: str | Path,
    *,
    providers: list[str],
    model_name: str,
) -> Any:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
    )
    verify_onnx_session_provider(
        session=session,
        providers=providers,
        model_name=model_name,
    )
    return session
