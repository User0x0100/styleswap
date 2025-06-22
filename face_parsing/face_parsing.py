import os
import torch
from .bisenet import BiSeNet
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class FaceParsing:
    def __init__(self, device):
        self.device = device

        self.net = BiSeNet(n_classes=19)
        self.net.load_state_dict(
            torch.load(
                os.path.join(
                    os.path.dirname(__file__), "79999_iter.pth"
                ),  # https://github.com/zllrunning/face-parsing.PyTorch
                weights_only=True,
            )
        )
        self.net.to(self.device)
        self.net.eval()
        self.net.requires_grad_(False)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        parts_idx = np.array([1, 2, 3, 4, 5, 6, 10, 11, 12, 13])
        self.parts_idx_tensor = torch.tensor(parts_idx, device=self.device)

    def process(self, image: Image.Image):
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        image = self.transform(image)
        image.unsqueeze_(0)
        image = image.to(self.device)
        with torch.no_grad():
            out = self.net(image)
            parsing = out.argmax(dim=1)

        masks = torch.isin(parsing, self.parts_idx_tensor)  # 检查像素是否属于指定类别
        masks = masks.float() * 255  # 转换为浮点数并缩放

        # 将结果移至 CPU 并转换为图像
        masks = masks.cpu().numpy().astype(np.uint8)[0]
        return Image.fromarray(masks)


if __name__ == "__main__":
    size = 512
    image_path = "/home/liaohaixun/styleswap_data/dateset/ffhq_1024/15555.png"
    p = FaceParsing("cuda:0")
    mask = p.process(image_path, size)

    import time

    start_time = time.time()
    mask = p.process(image_path, size)
    mask.save("1.png")
    print(f"Processing time: {(time.time() - start_time)*1000:.4f} ms")
