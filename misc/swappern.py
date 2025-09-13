import json
import math
import multiprocessing as mp
import os
from itertools import batched
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.utils as utils
from torch import autocast, nn
from torch.nn import functional as F
from torchcodec.decoders import VideoDecoder
from torchvision.io import ImageReadMode, decode_image
from tqdm import tqdm

from model.impl import FullGenerator
from misc.model import FaceParsing

from .model import ArcFace
from .tools import FaceAlign, tensor2cv_8uc3_bgr
from .utils import DataFolder


class EmbeddingSpaceExplorer:
    """专门用于探索嵌入空间可解释性的工具"""

    def __init__(self, device, embedding_dim=512, swapper_instance=None):
        self.device = device
        self.embedding_dim = embedding_dim
        self.dimension_semantics = {}  # 存储发现的维度语义
        self.exploration_history = []  # 记录探索历史
        self.swapper = swapper_instance  # 保存swapper实例的引用

    # 1. 单维度系统性扫描
    def systematic_dimension_scan(
        self, base_emb: torch.Tensor, target_dim: int, value_range: Tuple[float, float] = (-3.0, 3.0), steps: int = 50
    ) -> Generator[Tuple[torch.Tensor, float], None, None]:
        """系统性地扫描单个维度的整个值域，观察变化"""
        min_val, max_val = value_range
        step_size = (max_val - min_val) / (steps - 1)

        for i in range(steps):
            current_value = min_val + i * step_size
            modified_emb = base_emb.clone()
            modified_emb[0, target_dim] = current_value

            self.exploration_history.append({"type": "systematic_scan", "dimension": target_dim, "value": current_value, "step": i})

            yield modified_emb, current_value

    # 2. 对比维度探索
    def comparative_dimension_exploration(self, base_emb: torch.Tensor, dim_list: List[int], amplitude: float = 2.0, steps: float = 0.1) -> Generator[Dict, None, None]:
        """同时探索多个维度，便于对比它们的效果"""
        t = 0.0
        while True:
            results = {}

            for dim_idx in dim_list:
                modified_emb = base_emb.clone()
                value = amplitude * math.sin(t + dim_idx * 0.5)
                modified_emb[0, dim_idx] = value

                results[f"dim_{dim_idx}"] = {"embedding": modified_emb, "value": value, "dimension": dim_idx}

            # 基准对照
            results["baseline"] = {"embedding": base_emb, "value": 0.0, "dimension": -1}

            yield results
            t += steps

    # 3. 渐进式维度激活
    def progressive_dimension_activation(
        self, base_emb: torch.Tensor, activation_strength: float = 1.0, hold_time: int = 30
    ) -> Generator[Tuple[torch.Tensor, int, str], None, None]:
        """逐个激活维度，每个维度保持一段时间以便观察"""
        for dim_idx in range(self.embedding_dim):
            for frame in range(hold_time):
                modified_emb = base_emb.clone()
                t = frame / hold_time * 2 * math.pi
                modified_emb[0, dim_idx] = activation_strength * math.sin(t)

                status = f"激活维度 {dim_idx}/{self.embedding_dim} (帧 {frame}/{hold_time})"
                yield modified_emb, dim_idx, status

    # 4. 语义方向发现
    def semantic_direction_discovery(self, base_emb: torch.Tensor, search_dims: List[int] = None, combination_size: int = 2) -> Generator[Dict, None, None]:
        """尝试发现有意义的语义方向（维度组合）"""
        if search_dims is None:
            search_dims = list(range(min(50, self.embedding_dim)))

        from itertools import combinations

        for dim_combo in combinations(search_dims, combination_size):
            configs = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

            for config in configs:
                modified_emb = base_emb.clone()
                for i, dim_idx in enumerate(dim_combo):
                    modified_emb[0, dim_idx] = config[i] * 1.5

                yield {"embedding": modified_emb, "dimensions": dim_combo, "config": config, "description": f"维度{dim_combo}, 配置{config}"}

    # 5. 语义标注和记录
    def annotate_dimension_semantics(self, dim_idx: int, semantic_label: str, confidence: float, notes: str = ""):
        """为发现的维度语义进行标注"""
        self.dimension_semantics[dim_idx] = {"label": semantic_label, "confidence": confidence, "notes": notes, "discovery_time": len(self.exploration_history)}

    def get_semantic_summary(self) -> Dict:
        """获取已发现的语义总结"""
        return {
            "discovered_semantics": self.dimension_semantics,
            "exploration_count": len(self.exploration_history),
            "coverage": len(self.dimension_semantics) / self.embedding_dim,
        }

    # 6. 交互式探索模式
    def interactive_exploration_mode(self, base_emb: torch.Tensor, target_image):
        """交互式探索模式，支持实时调整和记录"""
        current_dim = 0
        current_value = 0.0

        print("交互式探索模式启动!")
        print("按键说明：")
        print("  a/d: 切换维度")
        print("  w/s: 增加/减少当前维度值")
        print("  r: 重置当前维度")
        print("  space: 记录当前发现")
        print("  q: 退出")

        while True:
            # 应用当前修改
            modified_emb = base_emb.clone()
            modified_emb[0, current_dim] = current_value

            # 生成图像
            if self.swapper:
                fake_face = self.swapper.swap_single(target_image, modified_emb)
                fake_face_cpu = tensor2cv_8uc3_bgr(fake_face).squeeze(0)
                fake_face_cpu = np.ascontiguousarray(fake_face_cpu)

                # 显示信息
                info_text = f"维度: {current_dim}, 值: {current_value:.3f}"
                cv2.putText(fake_face_cpu, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Interactive Exploration", fake_face_cpu)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("a"):  # 上一个维度
                current_dim = max(0, current_dim - 1)
                current_value = 0.0
            elif key == ord("d"):  # 下一个维度
                current_dim = min(self.embedding_dim - 1, current_dim + 1)
                current_value = 0.0
            elif key == ord("w"):  # 增加值
                current_value = min(3.0, current_value + 0.1)
            elif key == ord("s"):  # 减少值
                current_value = max(-3.0, current_value - 0.1)
            elif key == ord("r"):  # 重置
                current_value = 0.0
            elif key == ord(" "):  # 记录发现
                semantic_label = input(f"\n为维度 {current_dim} 输入语义标签: ")
                if semantic_label:
                    self.annotate_dimension_semantics(current_dim, semantic_label, 0.8)
                    print(f"已记录: 维度{current_dim} -> {semantic_label}")
            elif key == ord("q"):  # 退出
                break

        cv2.destroyAllWindows()
        return self.get_semantic_summary()

    # 7. 批量维度探索
    def batch_dimension_exploration(
        self,
        base_emb: torch.Tensor,
        target_image,
        dim_range: Tuple[int, int] = (0, 50),
        value_range: Tuple[float, float] = (-2.0, 2.0),
        samples_per_dim: int = 5,
        save_results: bool = True,
    ):
        """批量探索指定范围的维度"""
        start_dim, end_dim = dim_range
        min_val, max_val = value_range

        results = {}

        for dim_idx in range(start_dim, min(end_dim, self.embedding_dim)):
            print(f"探索维度 {dim_idx}...")
            dim_results = []

            for i in range(samples_per_dim):
                value = min_val + (max_val - min_val) * i / (samples_per_dim - 1)
                modified_emb = base_emb.clone()
                modified_emb[0, dim_idx] = value

                if self.swapper:
                    fake_face = self.swapper.swap_single(target_image, modified_emb)

                    # 可以在这里添加特征提取和分析
                    dim_results.append(
                        {
                            "value": value,
                            "embedding": modified_emb.cpu(),
                            # 'features': extracted_features  # 如果需要的话
                        }
                    )

            results[dim_idx] = dim_results

        if save_results:
            with open(f"dimension_exploration_results.json", "w") as f:
                # 注意：torch.Tensor不能直接序列化为JSON
                serializable_results = {}
                for dim, dim_results in results.items():
                    serializable_results[dim] = [{"value": r["value"]} for r in dim_results]
                json.dump(serializable_results, f, indent=2)

        return results


class AdvancedInterpolation:
    """高级插值方法"""

    def __init__(self, device):
        self.device = device

    # 李萨如曲线插值（多维度）
    def lissajous_interpolation(
        self, base_emb: torch.Tensor, amplitude: float = 0.5, freq_x: float = 1.0, freq_y: float = 2.0, phase_shift: float = 0.0, steps: float = 0.05
    ) -> Generator[torch.Tensor, None, None]:
        """李萨如曲线插值，操作多个维度"""
        t = 0.0
        while True:
            x = amplitude * math.sin(freq_x * t)
            y = amplitude * math.sin(freq_y * t + phase_shift)

            modified_emb = base_emb.clone()
            if modified_emb.shape[1] >= 2:
                modified_emb[0, 0] += x
                modified_emb[0, 1] += y

            yield F.normalize(modified_emb, p=2, dim=1)
            t += steps

    # 单维度复杂插值（专为可解释性设计）
    def single_dim_complex_interpolation(self, base_emb: torch.Tensor, target_dim: int = 8, steps: float = 0.05, amplitude: float = 1.0) -> Generator[torch.Tensor, None, None]:
        """在单个维度上使用复杂函数，保持可解释性"""
        t = 0.0
        while True:
            modified_emb = base_emb.clone()
            # 复杂但单维度的变化
            value = amplitude * (math.sin(t) * math.cos(t * 0.3) + 0.3 * math.sin(t * 3.7) * math.exp(-abs(t) * 0.01))
            modified_emb[0, target_dim] = value
            yield modified_emb
            t += steps

    # 维度组合插值
    def dimension_combination_interpolation(
        self, base_emb: torch.Tensor, dim_groups: List[List[int]], amplitudes: List[float] = None, steps: float = 0.05
    ) -> Generator[torch.Tensor, None, None]:
        """对维度分组进行协调插值"""
        if amplitudes is None:
            amplitudes = [1.0] * len(dim_groups)

        t = 0.0
        while True:
            modified_emb = base_emb.clone()

            for group_idx, (dim_group, amplitude) in enumerate(zip(dim_groups, amplitudes)):
                # 为每组维度生成协调的值
                base_value = amplitude * math.sin(t + group_idx * math.pi / 4)

                for i, dim_idx in enumerate(dim_group):
                    # 组内维度略有不同但协调
                    variation = 0.1 * math.sin(t * 2 + i * 0.5)
                    modified_emb[0, dim_idx] = base_value + variation

            yield modified_emb
            t += steps


class Swapper(nn.Module):
    def __init__(self, ckpt: str, size: int = 512, isconcat: bool = False, align_dst: None | torch.Tensor = None):
        super().__init__()
        self.size = size
        self.swap_net = FullGenerator(size=size, isconcat=isconcat)
        checkpoint = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)

        if "g_ema" in checkpoint:
            self.swap_net.load_state_dict(checkpoint["g_ema"])
        else:
            print(f"There is no 'g_ema' in the checkpoint. Keys: {checkpoint.keys()}")
            raise KeyError

        self.swap_net.eval()
        self.swap_net.requires_grad_(False)

        self.aligner = FaceAlign(from_normalized=True)
        self.arcface = ArcFace.Model()
        self.arcface.eval()
        self.arcface.requires_grad_(False)

        if align_dst is None:
            align_dst = ArcFace.get_dst_pts(size)
        else:
            assert align_dst.shape == [5, 2]

        self.register_buffer("align_dst", align_dst, persistent=False)
        self.facepars = FaceParsing.Model(compile_model=False)

    def swap_single(self, aligned_faces: torch.Tensor, id: torch.Tensor) -> torch.Tensor:
        if id.dim() == 1:
            id = id.unsqueeze(0)

        B = aligned_faces.shape[0]
        id = id.expand(B, -1)

        fake_face, _, fake_mask = self.swap_net(aligned_faces, id)
        fake_face = fake_face * fake_mask + (1 - fake_mask) * aligned_faces
        return fake_face

    def get_fae_id(self, fp: str) -> torch.Tensor:
        device = self.align_dst.device

        image = decode_image(fp)
        image = image.to(device=device, dtype=torch.float).div_(255.0).mul_(2.0).sub_(1.0).unsqueeze_(0)
        align_face, _, _ = self.aligner.align(images=image, out_size=(ArcFace.SIZE,) * 2, dst_pts5_pixel=ArcFace.get_dst_pts().to(device=device))

        face_mask = self.facepars.process(align_face)
        align_face = align_face * face_mask
        id_emb = self.arcface(align_face)

        return id_emb

    def linear_interpolation(self, target: str, source: str | None = None, z_dim=8, steps=0.05):
        """原始的单维度插值方法"""
        device = self.align_dst.device

        def infinite_alphas(step_size=0.1):
            t = 0.0
            while True:
                yield math.sin(t)
                t += step_size

        if source is None:
            id_emb = torch.randn((1, 512), dtype=torch.float, device=device)
            id_emb = F.normalize(id_emb, p=2, dim=1)  # 修复这里的bug
        else:
            id_emb = self.get_fae_id(source)

        target_tensor = decode_image(target, mode=ImageReadMode.RGB).to(device=device, dtype=torch.float).unsqueeze_(0)
        target_tensor.div_(255.0).mul_(2.0).sub_(1.0)
        target_tensor, _, _ = self.aligner.align(target_tensor, (512,) * 2, self.align_dst)

        alphas = infinite_alphas(steps)

        while True:
            id_emb[0, z_dim] = next(alphas)
            print(f"{id_emb[0, z_dim].item():.4f}")

            fake_face = self.swap_single(target_tensor, id_emb)
            fake_face_cpu = tensor2cv_8uc3_bgr(fake_face).squeeze(0)
            cv2.imshow("swaped", fake_face_cpu)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def interpretable_exploration_demo(self, target: str, source: str | None = None, exploration_mode: str = "interactive"):
        """可解释性探索的完整流程"""
        device = self.align_dst.device
        explorer = EmbeddingSpaceExplorer(device, swapper_instance=self)

        # 获取基础嵌入
        if source is None:
            id_emb = torch.randn((1, 512), dtype=torch.float, device=device)
            id_emb = F.normalize(id_emb, p=2, dim=1)
        else:
            id_emb = self.get_fae_id(source)

        # 处理目标图像
        target_tensor = decode_image(target, mode=ImageReadMode.RGB).to(device=device, dtype=torch.float).unsqueeze_(0)
        target_tensor.div_(255.0).mul_(2.0).sub_(1.0)
        target_tensor, _, _ = self.aligner.align(target_tensor, (512,) * 2, self.align_dst)

        print("开始可解释性探索...")

        if exploration_mode == "systematic":
            # 系统性扫描
            print("1. 系统性维度扫描...")
            for dim in range(20):  # 扫描前20个维度
                print(f"扫描维度 {dim}")
                for modified_emb, value in explorer.systematic_dimension_scan(id_emb, dim, steps=30):
                    fake_face = self.swap_single(target_tensor, modified_emb)
                    fake_face_cpu = tensor2cv_8uc3_bgr(fake_face).squeeze(0)
                    fake_face_cpu = np.ascontiguousarray(fake_face_cpu)

                    info = f"Dim {dim}: {value:.2f}"
                    cv2.putText(fake_face_cpu, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Dimension Scan", fake_face_cpu)

                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break

        elif exploration_mode == "comparative":
            # 对比探索
            print("2. 对比维度探索...")
            dims_to_compare = [0, 1, 2, 3, 8, 16, 32, 64]

            for results in explorer.comparative_dimension_exploration(id_emb, dims_to_compare, amplitude=1.5):
                # 创建网格显示
                faces = []
                labels = []

                for key, result in results.items():
                    fake_face = self.swap_single(target_tensor, result["embedding"])
                    faces.append(fake_face)
                    labels.append(f"{key}: {result['value']:.2f}")

                # 这里可以添加网格显示的代码
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break

        elif exploration_mode == "interactive":
            # 交互式探索
            print("3. 启动交互式探索...")
            semantic_summary = explorer.interactive_exploration_mode(id_emb, target_tensor)
            print("发现的语义:", semantic_summary)
            return semantic_summary

        elif exploration_mode == "batch":
            # 批量探索
            print("4. 批量维度探索...")
            results = explorer.batch_dimension_exploration(id_emb, target_tensor, dim_range=(0, 100), samples_per_dim=10)
            print(f"批量探索完成，共探索了 {len(results)} 个维度")
            return results

    def advanced_single_dim_interpolation(self, target: str, source: str | None = None, target_dim: int = 8, **kwargs):
        """高级单维度插值，专为可解释性设计"""
        device = self.align_dst.device
        interpolator = AdvancedInterpolation(device)

        if source is None:
            id_emb = torch.randn((1, 512), dtype=torch.float, device=device)
            id_emb = F.normalize(id_emb, p=2, dim=1)
        else:
            id_emb = self.get_fae_id(source)

        target_tensor = decode_image(target, mode=ImageReadMode.RGB).to(device=device, dtype=torch.float).unsqueeze_(0)
        target_tensor.div_(255.0).mul_(2.0).sub_(1.0)
        target_tensor, _, _ = self.aligner.align(target_tensor, (512,) * 2, self.align_dst)

        # 使用单维度复杂插值
        alpha_generator = interpolator.single_dim_complex_interpolation(id_emb, target_dim=target_dim, steps=kwargs.get("steps", 0.05), amplitude=kwargs.get("amplitude", 1.0))

        for modified_emb in alpha_generator:
            fake_face = self.swap_single(target_tensor, modified_emb)
            fake_face_cpu = tensor2cv_8uc3_bgr(fake_face).squeeze(0)

            # 显示当前维度信息
            info = f"Dim {target_dim}: {modified_emb[0, target_dim].item():.3f}"
            cv2.putText(fake_face_cpu, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Single Dim Interpolation", fake_face_cpu)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # 保留其他原有方法...
    @autocast(device_type="cuda")
    def swap_folder(self, fp: str, source: str, wfp: str | None = None, sf_prefix: str = "", batch_size: int = 32):
        # 原有的swap_folder实现...
        pass

    @autocast(device_type="cuda")
    def swap_video(self, vfp: str, source: str, sfp: str | None = None, batch_size: int = 32):
        # 原有的swap_video实现...
        pass


if __name__ == "__main__":

    from loguru import logger

    from misc.utils import DataFolder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{device=}")
    dataset = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024")
    # dataset = DataFolder("IDAssets")
    logger.info(f"{dataset=}")

    ckpt = "log/train_4/ckpt/125418_512.pth"
    logger.info(f"{ckpt=}")

    # target = "/opt/share/deepfake/linshi/录屏素材/2023-04-14(李云帆 陈雨)/20230414-163616614陈雨(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-02-17(袁忠 曹属馨)/20230217-160724910曹属馨(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-06-20(邓叮玲 许馨文)/20230620-155807793邓叮玲(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-08-28(李小姐 邱小姐)/20230828-122302李小姐(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2025-07-03(蒋女士 林嘉惠 王钰 符媛媛 石星儿)/王钰(电脑录屏).mp4"
    target = "IDAssets/抖音-记录美好生活.mp4"

    # source = dataset.sample(rando=True)
    source = "IDAssets/jay_c.webp"
    logger.info(f"{source=}")

    swap = Swapper(ckpt).to(device=device)
    # swap.swap_folder("IDAssets", source)
    # swap.swap_video(target, source)
    # swap.linear_interpolation("IDAssets/Henry_Cavill.webp", source)
    # swap.advanced_linear_interpolation("IDAssets/Henry_Cavill.webp", source, "spherical")
    # swap.interpretable_exploration_demo("IDAssets/Henry_Cavill.webp", source)

    swap.interpretable_exploration_demo("IDAssets/Harry_Potter_character_poster.jpg", source, "interactive")
