import os
import torch
from .bisenet import BiSeNet
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch import nn


class FaceParsing(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = BiSeNet(n_classes=19)
        self.net.load_state_dict(
            torch.load(
                os.path.join(
                    os.path.dirname(__file__), "79999_iter.pth"
                ),  # https://github.com/zllrunning/face-parsing.PyTorch
                weights_only=True,
            )
        )
        self.net.requires_grad_(False)
        self.net.eval()

        self.register_buffer(
            "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "parts_idx", torch.Tensor([1, 2, 3, 4, 5, 6, 10, 11, 12, 13])
        )

        self.to_tensor = transforms.ToTensor()

    def process(self, image: Image.Image):
        image = image.resize((512, 512), Image.Resampling.BOX)
        image = self.to_tensor(image).unsqueeze_(0).to(self.mean.device)
        image = (image - self.mean) / self.std

        with torch.no_grad():
            out = self.net(image)
            parsing = out.argmax(dim=1)

            masks = torch.isin(parsing, self.parts_idx)  # 检查像素是否属于指定类别
            masks = masks.float() * 255  # 转换为浮点数并缩放

            # 将结果移至 CPU 并转换为图像
            masks = masks.cpu().numpy().astype(np.uint8)[0]
            return Image.fromarray(masks)


if __name__ == "__main__":
    from PIL import Image
    import random

    image_path = (
        f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0,69999):05d}.png"
    )
    image = Image.open(image_path)

    p = FaceParsing().to("cuda:0")
    mask = p.process(image)

    import time

    start_time = time.time()
    mask = p.process(image)
    mask.save("1.png")
    print(f"Processing time: {(time.time() - start_time)*1000:.4f} ms")
