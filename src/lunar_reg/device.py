import logging
import torch

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages GPU/CPU device selection with automatic fallback."""

    def __init__(self, preferred: str = "auto"):
        self.preferred = preferred
        self.device_str = self._resolve(preferred)
        self.device = torch.device(self.device_str)

    def _resolve(self, preferred: str) -> str:
        """
        Resolution order:
        1. If preferred="cuda" and CUDA available -> cuda
        2. If preferred="auto" and CUDA available -> cuda
        3. Otherwise -> cpu
        """
        cuda_available = torch.cuda.is_available()
        if preferred == "cuda":
            if cuda_available:
                return "cuda"
            else:
                logger.warning("CUDA preferred but not available. Falling back to CPU.")
                return "cpu"
        elif preferred == "auto":
            if cuda_available:
                return "cuda"
            else:
                return "cpu"
        else:
            return "cpu"

    def get_torch_device(self) -> torch.device:
        return self.device

    def is_gpu_available(self) -> bool:
        return self.device_str == "cuda"
