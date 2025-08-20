from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F

MODEL_INPUT_SIZE = 256


def convert_68_to_5(pts68: torch.Tensor) -> torch.Tensor:

    left_eye = pts68[:, 36:42, :].mean(dim=1)  # 左眼中心
    right_eye = pts68[:, 42:48, :].mean(dim=1)  # 右眼中心
    nose_tip = pts68[:, 30, :]  # 鼻尖
    left_mouth = pts68[:, 48, :]  # 左嘴角
    right_mouth = pts68[:, 54, :]  # 右嘴角

    pts5 = torch.stack([left_eye, right_eye, nose_tip, left_mouth, right_mouth], dim=1)
    return pts5


def draw_landmarks_on_image(img: torch.Tensor, coords: torch.Tensor, radius: int = 2, color=(1.0, 0.0, 0.0)) -> torch.Tensor:

    B, C, H, W = img.shape
    _, K, _ = coords.shape

    # Normalize color
    color = torch.tensor(color, device=img.device).view(3, 1, 1)

    # Clone input image
    img_out = img.clone()

    # Generate a circular kernel
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1, device=img.device), torch.arange(-radius, radius + 1, device=img.device), indexing="ij")
    mask = (x**2 + y**2 <= radius**2).float()  # [r, r]

    for b in range(B):
        for k in range(K):
            cx, cy = coords[b, k]
            cx, cy = int(round(cx.item())), int(round(cy.item()))

            if 0 <= cx < W and 0 <= cy < H:
                x1 = max(cx - radius, 0)
                y1 = max(cy - radius, 0)
                x2 = min(cx + radius + 1, W)
                y2 = min(cy + radius + 1, H)

                mask_x1 = radius - (cx - x1)
                mask_y1 = radius - (cy - y1)
                mask_x2 = mask_x1 + (x2 - x1)
                mask_y2 = mask_y1 + (y2 - y1)

                for c in range(3):
                    img_out[b, c, y1:y2, x1:x2] = (1 - mask[mask_y1:mask_y2, mask_x1:mask_x2]) * img_out[b, c, y1:y2, x1:x2] + mask[mask_y1:mask_y2, mask_x1:mask_x2] * color[c]

    return img_out.clamp(0, 1)


def draw_landmarks_on_image_idxs(img: torch.Tensor, coords: torch.Tensor, color=(1.0, 0.0, 0.0), font_size: int = 10) -> torch.Tensor:

    B, C, H, W = img.shape
    color_int = tuple(int(c * 255) for c in color)

    imgs_out = []
    img_cpu = img.detach().cpu()
    coords_cpu = coords.detach().cpu()

    for b in range(B):
        pil_img = TF.to_pil_image(img_cpu[b])
        draw = ImageDraw.Draw(pil_img)

        font = ImageFont.load_default(font_size)

        for k, (x, y) in enumerate(coords_cpu[b]):
            x = int(round(x.item()))
            y = int(round(y.item()))
            if 0 <= x < W and 0 <= y < H:
                draw.text((x, y), str(k), fill=color_int, font=font)

        # 转回 tensor
        img_tensor = TF.to_tensor(pil_img)  # [3, H, W], float32 [0,1]
        imgs_out.append(img_tensor)

    img_out = torch.stack(imgs_out, dim=0)  # [B, 3, H, W]
    return img_out


def get_pts68(heatmap: torch.Tensor, pts_dst_size: int) -> torch.Tensor:

    B, K, H, W = heatmap.shape
    idx = heatmap.view(B, K, -1).argmax(dim=-1)  # [B, K], no maxval

    preds = torch.zeros((B, K, 2), device=heatmap.device, dtype=torch.float32)
    preds[..., 0] = (idx % W).float()
    preds[..., 1] = (idx // W).float()

    # Optional sub-pixel refinement
    px = preds[..., 0].long().clamp(1, W - 2)
    py = preds[..., 1].long().clamp(1, H - 2)

    dx = heatmap[torch.arange(B).unsqueeze(1), torch.arange(K), py, px + 1] - heatmap[torch.arange(B).unsqueeze(1), torch.arange(K), py, px - 1]
    dy = heatmap[torch.arange(B).unsqueeze(1), torch.arange(K), py + 1, px] - heatmap[torch.arange(B).unsqueeze(1), torch.arange(K), py - 1, px]

    offset = torch.stack([dx.sign(), dy.sign()], dim=-1) * 0.25
    preds += offset
    preds -= 0.5

    # Map to input image resolution
    scale = pts_dst_size / H
    pts68 = preds * scale

    return pts68


class Model(nn.Module):
    """
    https://github.com/1adrianb/face-alignment
    Args:
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: True.
        swap_rb (bool): If True, swap r b channel.
            Default: False.
        pts_dst_size:
            if pts_dst_size == None:
                pts_dst_size = face.shape[2]
    """

    def __init__(self, range_norm: bool = True, swap_rb: bool = False, pts_dst_size: int | None = None):
        super().__init__()
        self.range_norm = range_norm
        self.swap_rb = swap_rb
        self.pts_dst_size = pts_dst_size
        self.net = torch.jit.load(Path(__file__).with_name("2DFAN4-cd938726ad.zip"))
        self.eval()
        self.requires_grad_(False)

    def forward(self, face: torch.Tensor):

        if self.pts_dst_size is None:
            size = face.shape[2]
        else:
            size = self.pts_dst_size

        face = F.interpolate(face, MODEL_INPUT_SIZE)

        if self.range_norm:
            face = (face + 1) / 2

        if self.swap_rb:
            face = face[:, [2, 1, 0], :, :]

        heatmap = self.net(face)

        pts68 = get_pts68(heatmap, size)
        return pts68


if __name__ == "__main__":

    import random
    import time

    import torchvision.utils as utils
    from torchvision.io import read_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2

    print(f"使用设备: {device}")

    net = Model().to(device=device)

    images = [read_image(f"/opt/share/deepfake/dataset_1/ffhq_1024/{random.randint(0,69999):05d}.png") for _ in range(batch_size)]

    images_raw = torch.stack(images).to(device=device).float().div_(255.0)

    images = [img.float() / 127.5 - 1.0 for img in images]
    images_rgb = torch.stack(images).to(device=device)
    images_bgr = images_rgb[:, [2, 1, 0], :, :]

    for _ in range(10):
        heatmap = net(images_bgr)

    print("执行性能测试...")
    times = []
    for i in range(5):
        start_time = time.time()
        pts68 = net(images_bgr, coords_dst_size=112)
        times.append((time.time() - start_time) * 1000)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    fps = 1000 / avg_time

    print(f"平均处理时间: {avg_time:.2f} ms")
    print(f"最快处理时间: {min_time:.2f} ms")
    print(f"处理帧率: {fps:.1f} FPS")

    pts5 = net.convert_68_to_5(pts68)

    vis_image_idxs = Model.draw_landmarks_on_image_idxs(F.interpolate(images_raw, size=112), pts5, color=(1.0, 0.0, 0.0), font_size=6)

    # utils.save_image(vis_image, "1adrianb_face-alignment.png")
    utils.save_image(vis_image_idxs, "1adrianb_face-alignment_idxs.png")
