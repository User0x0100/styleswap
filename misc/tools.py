from pathlib import Path
import torchvision.utils as utils
import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import custom_fwd
from torch import autocast
from .model import RetinaFace
from torchcodec.decoders import VideoDecoder
import cv2
import numpy as np
from .utils import Timer
from tqdm import tqdm
import multiprocessing as mp


@custom_fwd(device_type="cuda", cast_inputs=torch.float)
def compute_affine_transform(points_x: torch.Tensor, points_y: torch.Tensor) -> torch.Tensor:
    # points_x: [B, N, 2]
    # points_y: [B, N, 2]
    B, N, _ = points_x.shape

    # 构建设计矩阵X_aug: [B, N, 3]
    ones = torch.ones((B, N, 1), dtype=points_x.dtype, device=points_x.device)
    X_aug = torch.cat([points_x, ones], dim=2)  # [B, N, 3]

    # 求解仿射矩阵 A: [B, 3, 2] 使得 X_aug @ A ≈ points_y
    # 采用最小二乘解 A = (X^T X)^{-1} X^T Y
    # batched 矩阵计算
    Xt = X_aug.transpose(1, 2)  # [B, 3, N]
    XtX = torch.bmm(Xt, X_aug)  # [B, 3, 3]
    XtY = torch.bmm(Xt, points_y)  # [B, 3, 2]

    # 逆或伪逆
    A = torch.linalg.solve(XtX, XtY)  # [B, 3, 2]

    # 转换为 [B, 2, 3]
    affine = A.transpose(1, 2)  # [B, 2, 3]

    return affine


@custom_fwd(device_type="cuda", cast_inputs=torch.float)
@torch.jit.script
def compute_similarity_transform(points_x: torch.Tensor, points_y: torch.Tensor) -> torch.Tensor:
    """
    计算两组点之间的相似变换 (similarity transform)

    Args:
        points_x: [B, N, 2] 源点集
        points_y: [B, N, 2] 目标点集

    Returns:
        affine: [B, 2, 3] 仿射变换矩阵，形式为 [sR | t]
    """
    # 计算中心点
    p_c = points_x.mean(dim=1, keepdim=True)  # [B, 1, 2]
    q_c = points_y.mean(dim=1, keepdim=True)  # [B, 1, 2]
    p_prime = points_x - p_c  # [B, N, 2]
    q_prime = points_y - q_c  # [B, N, 2]

    # 协方差矩阵 H = p_prime^T @ q_prime
    H = torch.bmm(p_prime.transpose(1, 2), q_prime)  # [B, 2, 2]

    # SVD 分解
    U, S, Vh = torch.linalg.svd(H)  # U: [B,2,2], S: [B,2], Vh: [B,2,2]
    V = Vh.transpose(-2, -1)  # [B,2,2]

    # 校正反射：保证 det(R)=1
    det = torch.det(torch.bmm(V, U.transpose(-2, -1)))  # [B]
    sign = det.sign()  # [B]
    # 调整 V 的最后一列和奇异值 S 的最后一个元素
    V_corrected = V.clone()
    V_corrected[..., -1] *= sign.unsqueeze(-1)  # 最后一列乘以 sign
    S_corrected = S.clone()
    S_corrected[:, -1] *= sign  # 最后一个奇异值乘以 sign
    # 计算旋转矩阵（使用校正后的 V）
    R = torch.bmm(V_corrected, U.transpose(-2, -1))  # [B,2,2]

    # 缩放因子（使用校正后的奇异值！）
    sigma_p = p_prime.pow(2).sum(dim=[1, 2])  # [B]
    s = S_corrected.sum(dim=1) / sigma_p  # [B]

    # 平移 t = q_c - s * R @ p_c
    Rp_c = torch.bmm(R, p_c.transpose(1, 2)).squeeze(2)  # [B,2]
    t = q_c.squeeze(1) - s[:, None] * Rp_c  # [B,2]

    # 构建仿射矩阵 [B,2,3]
    linear_part = s[:, None, None] * R  # [B,2,2]
    translation_part = t.unsqueeze(-1)  # [B,2,1]
    affine = torch.cat([linear_part, translation_part], dim=2)  # [B,2,3]
    return affine


def pixel_to_normalized(points: torch.Tensor, size: tuple[int, int], align_corners: bool = True) -> torch.Tensor:
    """
    将像素坐标(points[...,0] ∈ [0, W-1], points[...,1] ∈ [0, H-1])
    转换为 PyTorch grid_sample 的归一化坐标 [-1, 1]。

    - image_size: (H,W)
    - align_corners=True  时:    (x/(W-1))*2 - 1
    - align_corners=False 时:  ((x+0.5)/W)*2 - 1
    """
    H, W = size
    offset = 0.0 if align_corners else 0.5
    scale_x = 2.0 / (W - (1 if align_corners else 0))
    scale_y = 2.0 / (H - (1 if align_corners else 0))

    pts = points.to(dtype=torch.float32)
    norm_x = (pts[..., 0] + offset) * scale_x - 1.0
    norm_y = (pts[..., 1] + offset) * scale_y - 1.0

    return torch.stack((norm_x, norm_y), dim=-1)


from typing import List, Tuple


def draw_normalized_points(img: torch.Tensor, points_norm: torch.Tensor, color: Tuple[float, ...] = (1.0, 0.0, 0.0), radius: int = 1) -> torch.Tensor:
    """
    在 CHW 图像上绘制已归一化到 [-1,1] 的点。

    Args:
        img:           Tensor [C, H, W]，数值范围任意（比如 [0,1] 或 [0,255]）。
        points_norm:   Tensor [N, 2]，每行是 (x, y)，x,y ∈ [-1,1]。
        color:         长度为 C 的元组，点的颜色，比如 RGB=(1,0,0)。
        radius:        整数，绘制方块的半径，radius=1 时为 3×3 方块。

    Returns:
        out_img:       Tensor [C, H, W]，在原图上叠加了标记点。
    """
    assert img.ndim == 3, "img 必须为 [C,H,W]"
    C, H, W = img.shape
    assert points_norm.ndim == 2 and points_norm.shape[1] == 2

    # 映射到像素坐标并四舍五入
    coords = (points_norm + 1) * 0.5  # ∈ [0,1]
    xs = torch.clamp((coords[:, 0] * (W - 1)).round().long(), 0, W - 1)
    ys = torch.clamp((coords[:, 1] * (H - 1)).round().long(), 0, H - 1)

    out = img.clone()
    # 画每个点：用一个 (2*radius+1)×(2*radius+1) 的方块
    fill = torch.tensor(color, device=img.device, dtype=img.dtype).view(C, 1, 1)
    for x, y in zip(xs, ys):
        x0 = max(int(x) - radius, 0)
        x1 = min(int(x) + radius, W - 1)
        y0 = max(int(y) - radius, 0)
        y1 = min(int(y) + radius, H - 1)
        out[:, y0 : y1 + 1, x0 : x1 + 1] = fill

    return out


def normalize_theta(affine_src2dst, in_hw, out_hw, align_corners=True):

    if affine_src2dst.dim() == 2:
        affine_src2dst = affine_src2dst.unsqueeze(0)

    B = affine_src2dst.shape[0]
    device = affine_src2dst.device
    dtype = affine_src2dst.dtype

    H_in, W_in = in_hw
    H_out, W_out = out_hw

    # 预计算缩放和偏移参数
    if align_corners:
        # norm = 2 * pix / (size - 1) - 1
        scale_in_x, scale_in_y = 2.0 / (W_in - 1), 2.0 / (H_in - 1)
        scale_out_x, scale_out_y = (W_out - 1) / 2.0, (H_out - 1) / 2.0
        offset_in_x, offset_in_y = -1.0, -1.0
        offset_out_x, offset_out_y = 0.5 * (W_out - 1), 0.5 * (H_out - 1)
    else:
        # norm = (2 * pix + 1) / size - 1
        scale_in_x, scale_in_y = 2.0 / W_in, 2.0 / H_in
        scale_out_x, scale_out_y = W_out / 2.0, H_out / 2.0
        offset_in_x, offset_in_y = (1.0 / W_in) - 1.0, (1.0 / H_in) - 1.0
        offset_out_x, offset_out_y = W_out / 2.0 - 0.5, H_out / 2.0 - 0.5

    # 获取 src->dst 变换的逆 (dst->src)
    # affine_src2dst: [[a, b, tx], [c, d, ty]]
    # 计算2x2部分的逆
    a, b = affine_src2dst[:, 0, 0], affine_src2dst[:, 0, 1]
    c, d = affine_src2dst[:, 1, 0], affine_src2dst[:, 1, 1]
    tx, ty = affine_src2dst[:, 0, 2], affine_src2dst[:, 1, 2]

    det = a * d - b * c
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    inv_tx = -(inv_a * tx + inv_b * ty)
    inv_ty = -(inv_c * tx + inv_d * ty)

    # 构建最终的theta矩阵
    # theta = scale_in * inv_affine * scale_out
    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)

    theta[:, 0, 0] = scale_in_x * (inv_a * scale_out_x)
    theta[:, 0, 1] = scale_in_x * (inv_b * scale_out_y)
    theta[:, 0, 2] = scale_in_x * (inv_a * offset_out_x + inv_b * offset_out_y + inv_tx) + offset_in_x

    theta[:, 1, 0] = scale_in_y * (inv_c * scale_out_x)
    theta[:, 1, 1] = scale_in_y * (inv_d * scale_out_y)
    theta[:, 1, 2] = scale_in_y * (inv_c * offset_out_x + inv_d * offset_out_y + inv_ty) + offset_in_y

    return theta


@custom_fwd(device_type="cuda", cast_inputs=torch.float)
def invert_normalized_theta(theta: torch.Tensor) -> torch.Tensor:
    """
    计算 normalized_theta 的逆变换

    Args:
        theta: [B, 2, 3] 归一化的仿射变换矩阵

    Returns:
        inv_theta: [B, 2, 3] 逆变换矩阵
    """
    # 提取2x2线性部分和平移部分
    A = theta[..., :2, :2]  # [B, 2, 2]
    t = theta[..., :2, 2]  # [B, 2]

    # 计算2x2矩阵的逆
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]  # [B]

    # 构建逆矩阵
    inv_A = torch.zeros_like(A)
    inv_A[..., 0, 0] = A[..., 1, 1] / det
    inv_A[..., 0, 1] = -A[..., 0, 1] / det
    inv_A[..., 1, 0] = -A[..., 1, 0] / det
    inv_A[..., 1, 1] = A[..., 0, 0] / det

    # 计算逆平移：-A^(-1) * t
    inv_t = -torch.matmul(inv_A, t.unsqueeze(-1)).squeeze(-1)  # [B, 2]

    # 组合成逆变换矩阵
    inv_theta = torch.cat([inv_A, inv_t.unsqueeze(-1)], dim=-1)  # [B, 2, 3]

    return inv_theta


def tensor2cv_8uc3_bgr(image: torch.Tensor, value_range=(-1, 1)) -> np.ndarray:
    """
    将形如 [C, H, W] 或 [B, C, H, W] 的图像 Tensor 转为 OpenCV 使用的 uint8 BGR 格式。

    参数:
    - image: torch.Tensor, 值域应在 value_range 范围内。
    - value_range: tuple, 指定输入值的范围，例如 (-1, 1), (0, 1), (0, 255)。

    返回:
    - np.ndarray, 形状为 [B, H, W, 3]，类型为 uint8，BGR 排列。
    """
    image = image.clone()

    # 自动补 batch 维度
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

    # 解包范围
    min_val, max_val = value_range
    assert max_val > min_val, "Invalid value_range"

    # 标准化到 0~1 再 *255
    image = (image - min_val) / (max_val - min_val)
    image = image.clamp(0, 1).mul(255)

    # RGB -> BGR
    image = image[:, [2, 1, 0], ...]

    # 转为 NHWC 格式并 uint8
    image = image.permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()

    return image


class FaceAlign(nn.Module):

    def __init__(self, from_normalized: bool = False, from_unit_range: bool = False, swap_rb_ch: bool = True):
        """
        Args:
            from_normalized: 如果为 True，则假设输入为 [-1, 1]，转换为 [0, 255]
            from_unit_range: 如果为 True，则假设输入为 [0, 1]，转换为 [0, 255]
        """
        super().__init__()
        self.register_buffer("device", torch.zeros(1), persistent=False)

        self.detector = RetinaFace.Model(from_normalized=from_normalized, from_unit_range=from_unit_range, swap_rb_ch=swap_rb_ch)
        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def align_with(images: list[torch.Tensor] | torch.Tensor, src_pts5_pixel: torch.Tensor, idxs: List[int], out_size: tuple[int, int], dst_pts5_pixel: torch.Tensor):

        mats = compute_similarity_transform(src_pts5_pixel, dst_pts5_pixel.unsqueeze(0).expand(src_pts5_pixel.shape[0], -1, -1))

        aligned_list = []
        norm_theata = []
        slic_n = 0
        for image, N in zip(images, idxs):

            if N == 0:
                continue

            image = image.unsqueeze(0).expand(N, -1, -1, -1)
            C = image.shape[1]

            mat = normalize_theta(mats[slic_n : slic_n + N], image.shape[-2:], out_size)

            grid = F.affine_grid(mat, (N, C, *out_size), align_corners=True)
            aligned = F.grid_sample(image, grid, align_corners=True, mode="bicubic")

            aligned_list.append(aligned)
            norm_theata.append(mat)
            slic_n += N

        if len(aligned_list) == 0:
            return torch.empty((0, 3, *out_size), device=images[0].device), torch.empty((0, 2, 3), device=images[0].device)

        aligned = torch.cat(aligned_list, dim=0)
        norm_theata = torch.cat(norm_theata, dim=0)

        return aligned, norm_theata

    @torch.no_grad()
    def align(self, images: list[torch.Tensor] | torch.Tensor, out_size: tuple[int, int], dst_pts5_pixel: torch.Tensor, conf_thresh: float = 0.85, min_box_size: int = 0):
        """
        对输入图片进行人脸对齐（仿射变换），输出对齐后的图片和仿射矩阵。

        参数:
            images (list[torch.Tensor] | torch.Tensor):
                输入图片，可以是单张图片（Tensor, 形状为[C, H, W]）或多张图片的列表（每张为Tensor，尺寸可不同）。
            out_size (tuple[int, int]):
                对齐后输出图片的尺寸 (height, width)。
            dst_pts5_pixel (torch.Tensor):
                目标五点关键点坐标，形状为[5, 2]，像素坐标。

        返回:
            aligned_list (List[torch.Tensor]):
                对齐后的人脸图片列表，每个元素形状为[N, C, H, W]，N为检测到的人脸数。
            norm_theta (List[torch.Tensor]):
                对应的仿射变换矩阵列表，每个元素形状为[N, 2, 3]。

        异常:
            TypeError:
                输入类型不是torch.Tensor或list[torch.Tensor]时抛出。

        说明:
            - 支持批量处理不同尺寸的图片。
            - 每张图片可能检测到多个人脸，返回每个人脸的对齐结果。
        """

        results, idxs = self.detector.detector(images, conf_thresh, min_box_size)  # results [N, 15]

        all_pts5 = RetinaFace.get_5pts(results)  # [N, 5, 2]
        aligned, norm_theata = self.align_with(images, all_pts5, idxs, out_size, dst_pts5_pixel)
        return aligned, norm_theata, idxs

    @autocast(device_type="cuda")
    def restore_faces_to_original(
        self, align_faces: torch.Tensor, aff_matrix: torch.Tensor, org_images: list[torch.Tensor] | torch.Tensor, idxs: list[int], face_masks: torch.Tensor | None = None
    ):
        inv_aff_mat = invert_normalized_theta(aff_matrix)

        if face_masks is None:
            device = align_faces.device
            dtype = align_faces.dtype
            T, _, h, w = align_faces.shape
            face_masks = torch.ones((T, 1, h, w), dtype=dtype, device=device)

        slic_n = 0
        outputs = []
        for N, org_image in zip(idxs, org_images):

            if N == 0:
                outputs.append(org_image)
                continue

            grid = F.affine_grid(inv_aff_mat[slic_n : slic_n + N], [N] + list(org_image.shape[1:]), align_corners=False)

            faces_on_canvas = F.grid_sample(align_faces[slic_n : slic_n + N], grid, padding_mode="zeros", align_corners=False, mode="bicubic")
            mask_on_canvas = F.grid_sample(face_masks[slic_n : slic_n + N], grid, padding_mode="zeros", align_corners=False, mode="bicubic")

            # 聚合多张人脸到单张
            sum_faces = (faces_on_canvas * mask_on_canvas).sum(dim=0, keepdim=True)  # [1, C, H, W]
            sum_weights = mask_on_canvas.sum(dim=0, keepdim=True).clamp_min(1e-6)  # [1, 1, H, W]
            composite = sum_faces / sum_weights  # [1, C, H, W]

            acc_mask = mask_on_canvas.max(dim=0, keepdim=True).values  # [1, 1, H, W]

            # 现在维度匹配：[1,C,H,W] * [1,1,H,W] + [1,C,H,W] * [1,1,H,W]
            restored = org_image * (1 - acc_mask) + composite * acc_mask  # [1, C, H, W]

            outputs.append(restored.squeeze(0))  # 移除批次维度
            slic_n += N

        return outputs

    @staticmethod
    def video_face_ext_save_worker(q: mp.Queue, wfp: str, sf_prefix: str, value_range):

        while True:
            idxs, slic, aligned = q.get()

            if idxs is None:
                break

            aligned = tensor2cv_8uc3_bgr(aligned, value_range)

            slic_align_n = 0
            for n, frame_idx in zip(idxs, range(*slic)):

                faces = aligned[slic_align_n : slic_align_n + n]

                for face_num, face in enumerate(faces):
                    save_path = Path(wfp) / f"{sf_prefix}{str(frame_idx)}_{str(face_num)}.png"
                    cv2.imwrite(save_path, face, [cv2.IMWRITE_PNG_COMPRESSION, 1])

                slic_align_n += n
        q.close()

    def video_face_ext(
        self,
        vfp: str,
        dst_pts_pixel: torch.Tensor,
        wfp: str | None = None,
        sf_prefix: str = "",
        out_size: tuple[int, int] = [512, 512],
        batch_size: int = 8,
        conf_thresh: float = 0.85,
        min_box_size: int = 256,
    ):
        fn = Path(vfp).stem
        folder = Path(vfp).parent
        if wfp is None:
            wfp = Path(folder) / f"{fn}_align_results/"

        Path(wfp).mkdir(exist_ok=True, parents=True)

        device = self.device.device
        self.detector.from_normalized = False
        self.detector.from_unit_range = False

        video_reader = VideoDecoder(vfp, device="cuda")
        num_frames = video_reader.metadata.num_frames_from_content

        mp.set_start_method("spawn", force=True)

        q = mp.Queue(maxsize=10)
        p = mp.Process(target=self.video_face_ext_save_worker, args=(q, wfp, sf_prefix, (0, 255)))
        p.start()

        pbar = tqdm(range(num_frames), unit="frame")
        pbar.set_description(fn)

        slices = [(i, min(i + batch_size, num_frames)) for i in range(0, num_frames, batch_size)]

        try:
            for slic in slices:
                pbar.n = slic[1]
                pbar.refresh()

                chunk = video_reader.get_frames_in_range(*slic).data.to(device=device, dtype=torch.float)

                aligned, _, idxs = self.align(list(chunk.unbind(dim=0)), out_size, dst_pts_pixel, conf_thresh, min_box_size)

                q.put((idxs, slic, aligned))

        except KeyboardInterrupt:
            q.put((None,) * 3)
            return

        q.put((None,) * 3)

    @staticmethod
    def align_folder_save_worker(q: mp.Queue, wfp: str, sf_prefix: str, value_range):
        while True:
            aligned, idxs, fn = q.get()

            if aligned is None:
                break

            aligned = tensor2cv_8uc3_bgr(aligned, value_range)

            slic_align_n = 0
            for n, name in zip(idxs, fn):

                faces = aligned[slic_align_n : slic_align_n + n]
                for idx, face in enumerate(faces):

                    save_path = Path(wfp) / f"{sf_prefix}{Path(name).stem}_{str(idx)}.png"
                    cv2.imwrite(save_path, face, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                slic_align_n += n

        q.close()

    def align_folder(
        self,
        fp: str,
        dst_pts_pixel: torch.Tensor,
        sfp: str | None = None,
        sf_prefix: str = "",
        batch_size: int = 32,
        out_size: tuple[int, int] = [512, 512],
        conf_thresh: float = 0.85,
        min_box_size: int = 256,
    ):

        device = self.device.device
        self.detector.from_normalized = False
        self.detector.from_unit_range = False

        if sfp is None:
            sfp = Path(fp) / "align_results/"
        Path(sfp).mkdir(exist_ok=True, parents=True)

        mp.set_start_method("spawn", force=True)
        q = mp.Queue(maxsize=10)
        p = mp.Process(target=self.align_folder_save_worker, args=(q, sfp, sf_prefix, (0, 255)))
        p.start()

        dataset = DataFolder(fp)

        pbar = tqdm(range(dataset.len), unit="frame")
        pbar.set_description(fp)
        try:
            for images, fns in dataset.iter_batch_with_tensor(batch_size, True, False):

                pbar.update(len(images))

                images = [image.to(device=device, dtype=torch.float) for image in images]

                aligned, _, idxs = self.align(images, out_size, dst_pts_pixel, conf_thresh, min_box_size)

                q.put((aligned, idxs, fns))
        except KeyboardInterrupt:
            q.put((None,) * 3)
            return

        q.put((None,) * 3)


if __name__ == "__main__":
    from misc.utils import DataFolder
    from .model import ArcFace
    from torch import autocast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 6
    dataset = DataFolder("IDAssets")
    # dataset = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024")
    aligner = FaceAlign().to(device=device)

    # images = [dataset.random_pick_tensor().float().div_(255).to(device=device) for _ in range(batch_size)]
    # images = torch.stack(images, dim=0)
    # images = F.interpolate(images, size=128)

    # out_size = 112
    # arc_dst = ArcFace.get_dst_pts(out_size).to(device=device)
    # with autocast(device_type="cuda"):
    #     for _ in range(5):
    #         aligned, M, idxs = aligner.align(images, (out_size, out_size), arc_dst)

    #     with Timer():
    #         aligned, M, idxs = aligner.align(images, (out_size, out_size), arc_dst)

    # restored = aligner.restore_faces_to_original(aligned, M, images, idxs)

    # for idx, image in enumerate(restored):
    #     utils.save_image(image, f"visi_aligned{str(idx)}.png", normalize=True, value_range=(0, 1))

    vfp = "/opt/share/deepfake/dataset_1/youtube/4k_Face_Close_Up_HDR_Video_Vivid_Colors_Ambient_Sound_-_Relaxing.webm"
    dst_pts_pixel = ArcFace.get_dst_pts(512).to(device=device)
    aligner.video_face_ext(vfp=vfp, dst_pts_pixel=dst_pts_pixel, out_size=(512,) * 2, batch_size=128, conf_thresh=0.95, min_box_size=384)

    # dst_pts_pixel = ArcFace.get_dst_pts(512).to(device=device)
    # aligner.align_folder(
    #     fp="/opt/share/deepfake/dataset_1/ffhq_1024",
    #     dst_pts_pixel=dst_pts_pixel,
    #     sfp="/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst",
    #     batch_size=64,
    #     out_size=(512,) * 2,
    #     min_box_size=384,
    # )
