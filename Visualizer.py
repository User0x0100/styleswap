import os
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader, RandomSampler
from torchvision import utils

from dataloader.data_loader import FaceDataset
from model.impl import FullGenerator
from losses import (GANLoss, L1Loss, PerceptualLoss, g_path_regularize,
                    r1_penalty)
from losses.arcface import ArcFace

# 全局配置
CONFIG = {
    "size": 512,
    "checkpoint": "log/train/ckpt/1185000_512.pth",
    "device": "cuda",
    "image_path": ["/home/liaohaixun/styleswap_data/dateset/ffhq_1024"],
    "output_dir": "loss_visualization_output",
    "loss_type": "vgg19",
    "layer_weights": {
        # "conv1_1": 1.0,
        # "relu1_1": 1.0,
        # "conv1_2": 1.0,
        # "relu1_2": 1.0,
        # "pool1": 1.0,
        # "conv2_1": 1.0,
        # "relu2_1": 1.0,
        # "conv2_2": 1.0,
        # "relu2_2": 1.0,
        # "pool2": 1.0,
        # "conv3_1": 1.0,
        # "relu3_1": 1.0,
        # "conv3_2": 1.0,
        # "relu3_2": 1.0,
        # "conv3_3": 1.0,
        # "relu3_3": 1.0,
        "conv3_4": 1.0,
        # "relu3_4": 1.0,
        # "pool3": 1.0,
        # "conv4_1": 1.0,
        # "relu4_1": 1.0,
        # "conv4_2": 1.0,
        # "relu4_2": 1.0,
        # "conv4_3": 1.0,
        # "relu4_3": 1.0,
        # "conv4_4": 1.0,
        # "relu4_4": 1.0,
        # "pool4": 1.0,
        # "conv5_1": 1.0,
        # "relu5_1": 1.0,
        # "conv5_2": 1.0,
        # "relu5_2": 1.0,
        # "conv5_3": 1.0,
        # "relu5_3": 1.0,
        # "conv5_4": 1.0,
        # "relu5_4": 1.0,
        # "pool5": 1.0,
    },
    "use_input_norm": True,
    "range_norm": True,
    "perceptual_weight": 10.0,
    "style_weight": 10.0,
    "criterion": "l1",
    "num_samples": 1,
    "visualization_target": "fake_img",  # 可选择 "target" 或 "fake_img"
    "again": True,  # 使用上次的数据
}


def load_generator(config: Dict) -> FullGenerator:
    """加载并初始化生成器模型"""
    try:
        generator = FullGenerator(config["size"])
        checkpoint = torch.load(
            config["checkpoint"], map_location=config["device"], weights_only=False
        )
        generator.load_state_dict(checkpoint["g_ema"])
        generator.to(config["device"])
        generator.train()
        return generator
    except FileNotFoundError as e:
        raise FileNotFoundError(f"无法找到检查点文件: {config['checkpoint']}") from e
    except Exception as e:
        raise RuntimeError(f"加载生成器模型失败: {str(e)}") from e


def setup_perceptual_loss(config: Dict) -> PerceptualLoss:
    """设置感知损失函数"""
    return PerceptualLoss(
        config["layer_weights"],
        config["loss_type"],
        config["use_input_norm"],
        config["range_norm"],
        config["perceptual_weight"],
        config["style_weight"],
        config["criterion"],
    ).to(config["device"])


def sample_data(loader: DataLoader):
    """从数据加载器中无限采样数据"""
    while True:
        for batch in loader:
            yield batch


def normalize_heatmap(heatmap: np.ndarray, method: str = "minmax") -> np.ndarray:
    """标准化热力图以便可视化"""
    if method == "minmax":
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    elif method == "percentile":
        p95 = np.percentile(heatmap, 95)
        heatmap = np.clip(heatmap / (p95 + 1e-8), 0, 1)
    return heatmap


def apply_colormap(heatmap: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """将热力图应用颜色映射"""
    if colormap == "jet":
        cmap = plt.cm.jet
    elif colormap == "hot":
        cmap = plt.cm.hot
    elif colormap == "viridis":
        cmap = plt.cm.viridis
    elif colormap == "custom":
        colors = ["#000080", "#0000FF", "#00FF00", "#FFFF00", "#FF0000"]
        cmap = LinearSegmentedColormap.from_list("custom", colors)
    else:
        cmap = plt.cm.jet
    colored_heatmap = cmap(heatmap)
    return (colored_heatmap[:, :, :3] * 255).astype(np.uint8)


def overlay_heatmap(
    image: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """将热力图叠加到原始图像上"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
    colored_heatmap = apply_colormap(heatmap, "jet")
    return cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)


def visualize_gradients(
    gradients: np.ndarray,
    image: torch.Tensor,
    loss_name: str,
    sample_idx: int,
    methods: List[str],
    output_dir: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """可视化梯度热力图"""
    results = {}
    for method in methods:
        if method == "l2_norm":
            heatmap = np.sqrt(np.sum(gradients**2, axis=1))
        elif method == "mean_abs":
            heatmap = np.mean(np.abs(gradients), axis=1)
        elif method == "max_abs":
            heatmap = np.max(np.abs(gradients), axis=1)
        elif method == "sum_abs":
            heatmap = np.sum(np.abs(gradients), axis=1)
        elif method == "l1":
            heatmap = np.sum(np.abs(gradients), axis=1)
        else:
            print(f"警告: 未知的方法 '{method}'，使用默认的 l2_norm")
            heatmap = np.sqrt(np.sum(gradients**2, axis=1))

        heatmap_single = heatmap[0]
        heatmap_norm = normalize_heatmap(heatmap_single, "percentile")
        heatmap_smooth = cv2.GaussianBlur(heatmap_norm, (15, 15), 0)

        results[method] = {
            "raw": heatmap_single,
            "normalized": heatmap_norm,
            "smooth": heatmap_smooth,
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"{loss_name} Gradient Visualization - {method.upper()}", fontsize=16
        )

        img_np = image[0].detach().cpu().numpy().transpose(1, 2, 0)
        img_np = (((img_np + 1.0) / 2.0) * 255).astype(np.uint8)

        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        im1 = axes[0, 1].imshow(heatmap_single, cmap="jet")
        axes[0, 1].set_title("Raw Heatmap")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(heatmap_norm, cmap="jet")
        axes[0, 2].set_title("Normalized Heatmap")
        axes[0, 2].axis("off")
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        im3 = axes[1, 0].imshow(heatmap_smooth, cmap="jet")
        axes[1, 0].set_title("Smooth Heatmap")
        axes[1, 0].axis("off")
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        overlay_norm = overlay_heatmap(img_np, heatmap_norm, alpha=0.5)
        axes[1, 1].imshow(overlay_norm)
        axes[1, 1].set_title("Overlay (Normalized)")
        axes[1, 1].axis("off")

        overlay_smooth = overlay_heatmap(img_np, heatmap_smooth, alpha=0.5)
        axes[1, 2].imshow(overlay_smooth)
        axes[1, 2].set_title("Overlay (Smooth)")
        axes[1, 2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(
            output_dir, f"{loss_name}_{method}_sample_{sample_idx}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"保存可视化结果: {save_path}")

    return results


def analyze_loss_regions(heatmap: np.ndarray, threshold: float = 0.7) -> List[Dict]:
    """分析损失的主要作用区域"""
    high_activation = heatmap > threshold
    num_labels, labels = cv2.connectedComponents(high_activation.astype(np.uint8))
    regions = []
    for i in range(1, num_labels):
        mask = labels == i
        area = np.sum(mask)
        if area > 100:
            y_coords, x_coords = np.where(mask)
            bbox = {
                "top": int(np.min(y_coords)),
                "bottom": int(np.max(y_coords)),
                "left": int(np.min(x_coords)),
                "right": int(np.max(x_coords)),
                "area": area,
                "center": (int(np.mean(x_coords)), int(np.mean(y_coords))),
            }
            regions.append(bbox)
    return regions


def main(config: Dict) -> None:
    """主函数，执行梯度可视化和分析"""
    os.makedirs(config["output_dir"], exist_ok=True)
    torch.set_float32_matmul_precision("high")

    # 初始化模型和数据
    generator = load_generator(config)
    arcface = ArcFace().to(config["device"])
    arcface.eval()

    optimizer = torch.optim.Adam(
        params=generator.parameters(),
        lr=0.002,
    )

    perceptual_loss = setup_perceptual_loss(config)
    dataset = FaceDataset(size=config["size"], image_dirs=config["image_path"])
    data_loader = DataLoader(dataset, batch_size=1, sampler=RandomSampler(dataset))
    data_iter = sample_data(data_loader)

    # 获取可视化目标设置
    vis_target = config.get("visualization_target", "target")
    print(f"当前可视化目标: {vis_target}")

    num_samples = 1 if config["again"] else config["num_samples"]

    for sample_idx in range(num_samples):
        print(f"\n处理样本 {sample_idx + 1}/{num_samples}")

        if config["again"]:
            t = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            target = Image.open(
                os.path.join(config["output_dir"], f"target_image_sample_0.png"),
            ).resize((config["size"], config["size"]), resample=Image.Resampling.BOX)
            source = Image.open(
                os.path.join(config["output_dir"], f"source_image_sample_0.png"),
            ).resize((config["size"], config["size"]), resample=Image.Resampling.BOX)

            target = t(target).unsqueeze(0)
            source = t(source).unsqueeze(0)

        else:

            target, source, _ = next(data_iter)

        target = target.to(config["device"])
        source = source.to(config["device"])
        # mask = mask.to(config["device"])

        # 根据配置选择需要计算梯度的张量
        if vis_target == "target":
            vis_tensor = target
        elif vis_target == "fake_img":
            vis_tensor = target  # 初始设置，后续会更新
        else:
            print(f"警告: 未知的可视化目标 '{vis_target}'，使用默认的 'target'")
            vis_tensor = target

        if vis_tensor.grad is not None:
            vis_tensor.grad.zero_()
        vis_tensor.requires_grad_(True)

        with torch.no_grad():
            z_id = arcface(source)

        fake_img, _, _ = generator(target, z_id.detach())

        # fake_z_id = arcface(fake_img)
        # norm = torch.linalg.vector_norm(fake_z_id, ord=2, dim=1)
        # print(
        #     torch.allclose(
        #         norm, torch.tensor([1.0], device=config["device"]), atol=1e-6
        #     )
        # )
        # id_loss = (1 - torch.cosine_similarity(fake_z_id, z_id.detach())).mean()

        # loss_vgg, _ = perceptual_loss(fake_img, target)

        # optimizer.zero_grad()

        # loss_vgg.backward(retain_graph=True)
        # grad_vgg = []
        # for p in generator.parameters():
        #     if p.grad is not None:
        #         grad_vgg.append(p.grad.detach().view(-1))
        # grad_vgg = torch.cat(grad_vgg)  # ∇θ L_vgg

        # optimizer.zero_grad()

        # id_loss.backward(retain_graph=True)
        # grad_id = []
        # for p in generator.parameters():
        #     if p.grad is not None:
        #         grad_id.append(p.grad.detach().view(-1))
        # grad_id = torch.cat(grad_id)

        # optimizer.zero_grad()

        # cos_sim = torch.dot(grad_vgg, grad_id) / (
        #     grad_vgg.norm() * grad_id.norm() + 1e-8
        # )

        # cos_sim = cos_sim.item()
        # angle = math.degrees(math.acos(max(min(cos_sim, 1.0), -1.0)))

        # print(f"梯度余弦相似度：{cos_sim:.4f}；对应梯度夹角：{angle:.1f}°")

        # exit()

        # 如果可视化目标是fake_img，更新vis_tensor
        if vis_target == "fake_img":
            fake_img.retain_grad()
            if fake_img.grad is not None:
                fake_img.grad.zero_()
            fake_img.requires_grad_(True)
            vis_tensor = fake_img

        # 计算感知损失梯度
        vis_tensor.grad = None
        # l_g_percep = F.l1_loss(target, fake_img, reduction="mean")

        l_g_percep, _ = perceptual_loss(fake_img, target)
        l_g_percep.backward(retain_graph=True)
        if vis_tensor.grad is not None:
            perceptual_gradients = vis_tensor.grad.data.cpu().numpy()
            print(f"感知损失梯度形状: {perceptual_gradients.shape}")
            print(
                f"感知损失梯度范围: [{perceptual_gradients.min():.6f}, {perceptual_gradients.max():.6f}]"
            )
            visualize_gradients(
                perceptual_gradients,
                vis_tensor,
                f"perceptual_loss_{vis_target}",
                sample_idx,
                # ["l2_norm", "mean_abs", "sum_abs", "l1"],
                ["l1"],
                config["output_dir"],  # 添加了l1方法
            )

        # 计算风格损失梯度
        vis_tensor.grad = None
        l_g_style, _ = perceptual_loss(fake_img, target)
        l_g_style.backward()
        if vis_tensor.grad is not None:
            style_gradients = vis_tensor.grad.data.cpu().numpy()
            print(f"风格损失梯度形状: {style_gradients.shape}")
            print(
                f"风格损失梯度范围: [{style_gradients.min():.6f}, {style_gradients.max():.6f}]"
            )
            visualize_gradients(
                style_gradients,
                vis_tensor,
                f"style_loss_{vis_target}",
                sample_idx,
                # ["l2_norm", "mean_abs", "sum_abs", "l1"],
                ["l1"],
                config["output_dir"],
            )

            # 分析高激活区域
            heatmap = np.sqrt(np.sum(style_gradients**2, axis=1))[0]
            heatmap_norm = normalize_heatmap(heatmap, "percentile")
            regions = analyze_loss_regions(heatmap_norm, threshold=0.7)
            print(f"发现 {len(regions)} 个高激活区域:")
            for i, region in enumerate(regions):
                print(
                    f"  区域 {i+1}: 面积={region['area']}, 中心={region['center']}, "
                    f"边界框=({region['left']},{region['top']})-({region['right']},{region['bottom']})"
                )

        # 保存样本图像
        if config["again"] == False:
            if sample_idx == 0:
                utils.save_image(
                    target,
                    os.path.join(
                        config["output_dir"], f"target_image_sample_{sample_idx}.png"
                    ),
                    normalize=True,
                    value_range=(-1, 1),
                )
                utils.save_image(
                    source,
                    os.path.join(
                        config["output_dir"], f"source_image_sample_{sample_idx}.png"
                    ),
                    normalize=True,
                    value_range=(-1, 1),
                )
        utils.save_image(
            fake_img,
            os.path.join(config["output_dir"], f"fake_image_sample_{sample_idx}.png"),
            normalize=True,
            value_range=(-1, 1),
        )

    print(f"\n可视化完成！结果保存在: {config['output_dir']}")
    print("生成的文件:")
    print("- 原始图像: target_image_*.png, source_image_*.png, fake_image_*.png")
    print(f"- 感知损失热力图: perceptual_loss_{vis_target}_*_sample_*.png")
    print(f"- 风格损失热力图: style_loss_{vis_target}_*_sample_*.png")
    print(f"- 支持的可视化方法: l2_norm, mean_abs, sum_abs, l1")


if __name__ == "__main__":
    main(CONFIG)

# 使用示例：
# 1. 可视化target图像的梯度（默认）:
#    CONFIG["visualization_target"] = "target"
#
# 2. 可视化fake_img图像的梯度:
#    CONFIG["visualization_target"] = "fake_img"

# 建议添加单元测试，例如：
# def test_normalize_heatmap():
#     heatmap = np.random.rand(512, 512)
#     norm_heatmap = normalize_heatmap(heatmap, "minmax")
#     assert norm_heatmap.min() >= 0 and norm_heatmap.max() <= 1
