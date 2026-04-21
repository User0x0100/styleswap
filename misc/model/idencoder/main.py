from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .iresnet import iresnet50, iresnet100
from .simswap_backend import IRBlock, ResNet

SIZE = 112
ARCFACE_DST_PTS = torch.tensor([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=torch.float32)


def get_dst_pts(dst_size: int = 112):
    if dst_size % 112 == 0:
        ratio = float(dst_size) / 112.0
        diff_x = 0
    else:
        ratio = float(dst_size) / 128.0
        diff_x = 8.0 * ratio
    dst = ARCFACE_DST_PTS * ratio
    dst[:, 0] += diff_x
    return dst


from enum import Enum


class PROVIDER(Enum):
    ARC_OFFICIAL_R50 = 1
    ARC_OFFICIAL_R100 = 2
    ARC_SIMSWAP = 3
    BLENDFACE = 4


class Model(nn.Module):
    def __init__(self, provider: PROVIDER = PROVIDER.BLENDFACE):
        super().__init__()

        match provider:
            case PROVIDER.ARC_OFFICIAL_R50:

                weight = Path(__file__).with_name("glint360k_r50_fp16.pth")
                # glint360k_r50_fp16.pth from https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
                self.backbone = iresnet50()

            case PROVIDER.ARC_OFFICIAL_R100:
                weight = Path(__file__).with_name("glint360k_cosface_r100_fp16_0.1.pth")
                self.backbone = iresnet100()

            case PROVIDER.ARC_SIMSWAP:

                weight = Path(__file__).with_name("simswap.pth")
                self.backbone = ResNet(block=IRBlock, layers=[3, 4, 23, 3], use_se=True)
            case PROVIDER.BLENDFACE:

                weight = Path(__file__).with_name("blendface.pth")
                self.backbone = iresnet100()

            case _:
                raise TypeError

        ckpt = torch.load(weight, weights_only=True, map_location=torch.device("cpu"))
        self.backbone.load_state_dict(ckpt)

        self.eval()
        self.requires_grad_(False)

    def print_named_modules(self):
        for name, module in self.backbone.named_modules():
            print(f"{name:40s} -> {module.__class__.__name__}")

    def forward(self, x):
        x = F.interpolate(x, SIZE, mode="bicubic", align_corners=False)
        id = self.backbone(x)
        return F.normalize(id, p=2, dim=1)


if __name__ == "__main__":
    import time
    from torch.amp import autocast

    device = "cuda"

    arcface = Model(provider=PROVIDER.ARC_OFFICIAL_R100).to(device)

    input = torch.randn(24, 3, 512, 512, dtype=torch.float32, device=device)

    # Warmup
    for i in range(5):
        arcface(input)

    # FP32 inference
    with torch.no_grad():
        out_fp32 = arcface(input)

    # FP16 inference
    with torch.no_grad(), autocast(device_type=device):
        out_fp16 = arcface(input)

    # Compare precision
    diff = (out_fp32 - out_fp16.float()).abs().mean().item()
    max_diff = (out_fp32 - out_fp16.float()).abs().max().item()
    print(f"FP32 vs FP16 平均绝对误差: {diff:.9f}")
    print(f"FP32 vs FP16 最大绝对误差: {max_diff:.9f}")

    print("执行性能测试...")
    times = []
    for i in range(10):
        start_time = time.time()
        arcface(input)
        times.append((time.time() - start_time) * 1000)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    fps = 1000 / avg_time

    print(f"平均处理时间: {avg_time:.2f} ms")
    print(f"最快处理时间: {min_time:.2f} ms")
    print(f"处理帧率: {fps:.1f} FPS")
