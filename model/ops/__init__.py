import os
import torch


def detect_torch_cuda_arch_list():

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 CUDA 设备")
    caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}

    archs = sorted({f"{maj}.{min}" for maj, min in caps})
    arch_list = ";".join(archs)

    key = "TORCH_CUDA_ARCH_LIST"

    if os.environ.get(key) is None:
        os.environ[key] = arch_list
        print(f"{key} = {arch_list}")


detect_torch_cuda_arch_list()
