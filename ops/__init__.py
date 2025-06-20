import os
import torch


def detect_torch_cuda_arch_list():
    """
    扫描所有可见 GPU，返回形如 ["7.0","7.5","8.0"] 的列表，并设定环境变量。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 CUDA 设备")
    caps = {
        torch.cuda.get_device_capability(i)  # 返回 (major, minor)
        for i in range(torch.cuda.device_count())
    }
    # 转为字符串，并去重、排序
    archs = sorted({f"{maj}.{min}" for maj, min in caps})
    arch_list = ";".join(archs)
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    print(f'自动设置 TORCH_CUDA_ARCH_LIST="{arch_list}"')


# 调用一次即可
detect_torch_cuda_arch_list()
