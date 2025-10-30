"""GPU utilities for TAMP improvisational policies."""

from typing import Any, Union

import numpy as np
import torch


def get_device(device_name: str = "cuda") -> torch.device:
    """Get PyTorch device."""
    if device_name.startswith("cuda"):
        if torch.cuda.is_available():
            # Handle specific GPU index if provided (e.g., "cuda:1")
            if ":" in device_name:
                index = int(device_name.split(":")[1])
                if index < torch.cuda.device_count():
                    return torch.device(device_name)
                return torch.device("cuda:0")
            return torch.device("cuda")
    return torch.device("cpu")


def set_torch_seed(seed: int) -> None:
    """Set all PyTorch random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # For deterministic behavior on GPU (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_torch(
    data: Any,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert data to torch tensor on specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    if isinstance(data, (list, float, int)):
        data = np.array(data)
    return torch.tensor(data, device=device, dtype=dtype)


def to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


class DeviceContext:
    """Context for managing device placement of data."""

    def __init__(self, device_name: str = "cuda"):
        self.device = get_device(device_name)

    def __call__(self, data: Any) -> Any:
        """Convert data to torch tensor on device."""
        return to_torch(data, self.device)

    def numpy(self, data: Any) -> np.ndarray:
        """Convert data back to numpy."""
        return to_numpy(data)


def get_gpu_memory_info() -> Union[str, list[dict[str, Any]]]:
    """Get memory information for all GPUs."""
    if not torch.cuda.is_available():
        return "CUDA not available"

    memory_info = []
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        properties = torch.cuda.get_device_properties(i)
        total = properties.total_memory / 1e9
        name = properties.name

        memory_info.append(
            {
                "device_index": i,
                "name": name,
                "total_memory": total,
                "allocated_memory": allocated,
                "reserved_memory": reserved,
                "free_memory": total - allocated,
            }
        )

    return memory_info
