from pathlib import Path
import torch
from .bisenet import BiSeNet
from torch.nn import functional as F


class FaceParsing:
    """
    Args:
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: True.
    """

    def __init__(
        self,
        compile_model: bool = True,
        range_norm: bool = True,
        device: torch.device = torch.device("cuda", 0),
    ):

        self.device = device
        self.range_norm = range_norm
        self.net = BiSeNet(n_classes=19).to(device=self.device)

        model_path = Path(__file__).with_name("79999_iter.pth")
        self.net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.net.eval()
        if compile_model:
            self.net = torch.compile(
                self.net,
                fullgraph=True,
                dynamic=False,
                options={
                    "triton.cudagraphs": True,
                    "max_autotune": True,
                    "epilogue_fusion": True,
                },
            )

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.parts_idx = torch.tensor([1, 2, 3, 4, 5, 6, 10, 11, 12, 13], device=self.device)

    @torch.no_grad()
    def process(self, image: torch.Tensor):
        H, W = image.size()[2:]
        image = F.interpolate(image, [512, 512], mode="area")
        if self.range_norm:
            image.sub_(self.mean).div_(self.std)
        out = self.net(image, h_w=[H, W])
        parsing = out.argmax(dim=1)
        mask = torch.isin(parsing, self.parts_idx).to(dtype=torch.float, device=image.device).unsqueeze(1)
        return mask


if __name__ == "__main__":
    import time
    import random
    from torchvision.io import read_image
    from torchvision import utils

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 24
    print(f"使用设备: {device}")

    parser = FaceParsing(device=device, range_norm=True)

    images = [read_image(f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0,69999):05d}.png") for _ in range(batch_size)]

    # 0~1
    # images = [img.float().div(255.0) for img in images]

    # -1~1
    images = [img.float() / 127.5 - 1.0 for img in images]
    images = torch.stack(images, dim=0).to(device=device)

    # 性能测试
    print("执行性能测试...")
    times = []
    for i in range(100):
        start_time = time.time()
        mask = parser.process(images)
        times.append((time.time() - start_time) * 1000)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    fps = 1000 / avg_time

    print(f"平均处理时间: {avg_time:.2f} ms")
    print(f"最快处理时间: {min_time:.2f} ms")
    print(f"处理帧率: {fps:.1f} FPS")

    print("保存对比图像...")

    # mask = mask.repeat(1, 3, 1, 1)
    mask = (mask.repeat(1, 3, 1, 1) - 0.5) / 0.5

    grid = torch.cat((mask, images), dim=0)
    utils.save_image(
        grid,
        "mask.png",
        nrow=batch_size,
        normalize=True,
        value_range=(-1, 1),
    )
