import os

import torch


def detect_torch_cuda_arch_list():

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 CUDA 设备")
    caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}  # 返回 (major, minor)
    # 转为字符串，并去重、排序
    archs = sorted({f"{maj}.{min}" for maj, min in caps})
    arch_list = ";".join(archs)

    key = "TORCH_CUDA_ARCH_LIST"
    current = os.environ.get(key)
    if current is None:
        os.environ[key] = arch_list
        print(f"{key} = {arch_list}")


# 调用一次即可
detect_torch_cuda_arch_list()
