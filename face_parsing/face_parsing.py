import os
from pathlib import Path
import torch
from .bisenet import BiSeNet
import cv2
import numpy as np
from PIL import Image


class FaceParsing:
    def __init__(self, device: torch.device = torch.device("cuda", 0)):

        self.device = device
        self.net = BiSeNet(n_classes=19).to(device=self.device)

        model_path = Path(__file__).with_name("79999_iter.pth")
        self.net.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        self.net.eval()

        self.input_buffer = torch.empty(1, 3, 512, 512, device=self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(
            1, 3, 1, 1
        )
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(
            1, 3, 1, 1
        )
        self.parts_idx = torch.tensor(
            [1, 2, 3, 4, 5, 6, 10, 11, 12, 13], device=self.device
        )

    @torch.no_grad()
    def process(self, image):

        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image

        # OpenCV缩放 + 直接转换
        img_resized = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # 拷贝到缓冲区并标准化
        self.input_buffer.copy_(img_tensor.unsqueeze(0).to(device=self.device))
        self.input_buffer.sub_(self.mean).div_(self.std)

        out = self.net(self.input_buffer)
        parsing = out.argmax(dim=1)

        # 生成掩码
        mask = (
            torch.isin(parsing, self.parts_idx).cpu().numpy().astype(np.uint8)[0] * 255
        )
        return Image.fromarray(mask)


if __name__ == "__main__":
    import time
    import random

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 初始化模型
    parser = FaceParsing(device=device)

    # 加载测试图像
    image_path = (
        f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0,69999):05d}.png"
    )
    image = Image.open(image_path)
    print(f"加载图像: {image_path}")

    # 单次测试
    print("执行单次测试...")
    start_time = time.time()
    mask = parser.process(image)
    single_time = (time.time() - start_time) * 1000
    mask.save("face_mask_result.png")
    print(f"单次处理时间: {single_time:.2f} ms")

    # 性能测试
    print("执行性能测试...")
    times = []
    for i in range(100):
        start_time = time.time()
        mask = parser.process(image)
        times.append((time.time() - start_time) * 1000)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    fps = 1000 / avg_time

    print(f"平均处理时间: {avg_time:.2f} ms")
    print(f"最快处理时间: {min_time:.2f} ms")
    print(f"处理帧率: {fps:.1f} FPS")

    # 保存对比图像
    print("保存对比图像...")

    # 创建对比图
    original = np.array(image.resize((512, 512)))
    mask_array = np.array(mask)

    # 创建彩色掩码
    colored_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    colored_mask[mask_array > 0] = [255, 0, 0]  # 红色标记面部区域

    # 混合图像
    overlay = cv2.addWeighted(original, 0.7, colored_mask, 0.3, 0)

    # 横向拼接保存
    comparison = np.hstack(
        [original, mask_array[..., np.newaxis].repeat(3, axis=2), overlay]
    )
    Image.fromarray(comparison).save("face_parsing_comparison.png")

    print("结果已保存:")
    print("- face_mask_result.png: 面部掩码")
    print("- face_parsing_comparison.png: 对比图 (原图|掩码|叠加)")
