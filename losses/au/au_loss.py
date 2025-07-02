from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from .model.MEFL import MEFARG


class OpenGraphAU(nn.Module):
    """
    Args:
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: True.
    """

    def __init__(self, range_norm: bool = True):
        super().__init__()

        self.range_norm = range_norm
        self.register_buffer(
            "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )
        self.net = MEFARG()
        state_dict = torch.load(
            Path(__file__).with_name(
                "OpenGprahAU-ResNet50_second_stage.pth"
            ),  # https://github.com/lingjivoo/OpenGraphAU/tree/main
            map_location=torch.device("cpu"),
        )["state_dict"]

        self.net.load_state_dict(
            {k.removeprefix("module."): v for k, v in state_dict.items()}, strict=False
        )

    def forward(self, x):
        x = self.transform(x)
        if self.range_norm:
            x = (x + 1) / 2
        x = (x - self.mean) / self.std
        return self.net(x)


class AULoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.open_graph_au = OpenGraphAU()
        self.open_graph_au.requires_grad_(False)
        self.open_graph_au.eval()
        self.weight = weight

    def forward(self, fake, real):
        au_fake = self.open_graph_au(fake)
        au_real = self.open_graph_au(real)
        return F.l1_loss(au_fake, au_real) * self.weight


if __name__ == "__main__":
    from PIL import Image
    import random

    device = torch.device("cuda", 0)
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),  # 或自定义尺寸
            transforms.ToTensor(),  # 转为 [0,1] 的 FloatTensor, shape: [C,H,W]
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # 变为 [-1,1]
        ]
    )
    x = Image.open(
        f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0,69999):05d}.png"
    ).convert("RGB")
    y = Image.open(
        f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0,69999):05d}.png"
    ).convert("RGB")

    x = transform(x).unsqueeze_(0).to(device)
    y = transform(y).unsqueeze_(0).to(device)

    au_loss = AULoss(weight=50.0).to(device=device)
    loss = au_loss(x, y)

    print(loss)
