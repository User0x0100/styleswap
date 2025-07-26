import itertools
from collections import OrderedDict
from math import ceil
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
import torchvision.ops as ops

from misc.utils import Timer

from .net import FPN as FPN
from .net import SSH as SSH
from .net import MobileNetV1 as MobileNetV1


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


CFG_MNET0_25G = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": True,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
}
CFG_R50 = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 24,
    "ngpu": 4,
    "epoch": 100,
    "decay1": 70,
    "decay2": 90,
    "image_size": 840,
    "pretrain": True,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256,
}


class Model(nn.Module):
    # https://github.com/biubug6/Pytorch_Retinaface
    def __init__(self, cfg=CFG_R50, from_normalized: bool = False, from_unit_range: bool = False, swap_rb_ch: bool = True):
        """
        Args:
            cfg:  Network related settings.
            from_normalized: 如果为 True，则假设输入为 [-1, 1]，转换为 [0, 255]
            from_unit_range: 如果为 True，则假设输入为 [0, 1]，转换为 [0, 255]
        """
        super().__init__()
        if from_normalized and from_unit_range:
            raise ValueError("from_normalized 与 from_unit_range 不能同时为 True。")
        self.from_normalized = from_normalized
        self.from_unit_range = from_unit_range
        self.swap_rb_ch = swap_rb_ch
        self.register_buffer("mean", torch.tensor([104.0, 117.0, 123.0], dtype=torch.float).view(1, 3, 1, 1), persistent=False)  # BGR

        backbone = None
        if cfg["name"] == "mobilenet0.25":
            pretrain_ckpt = Path(__file__).with_name("weights").joinpath("mobilenet0.25_Final.pth")
            backbone = MobileNetV1()
            if cfg["pretrain"]:
                checkpoint = Path(__file__).with_name("weights").joinpath("mobilenetV1X0.25_pretrain.tar")
                state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))["state_dict"]
                state_dict = {cleaned: v for k, v in state_dict.items() if (cleaned := k.removeprefix("module."))}  # 去掉 "module."
                backbone.load_state_dict(state_dict)
        elif cfg["name"] == "Resnet50":

            pretrain_ckpt = Path(__file__).with_name("weights").joinpath("Resnet50_Final.pth")
            backbone = models.resnet50(weights=None)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg["return_layers"])
        in_channels_stage2 = cfg["in_channel"]
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg["out_channel"]
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg["out_channel"])

        pretrain_state_dict = torch.load(pretrain_ckpt, map_location=torch.device("cpu"))
        pretrain_state_dict = {cleaned: v for k, v in pretrain_state_dict.items() if (cleaned := k.removeprefix("module."))}  # 去掉 "module."
        self.load_state_dict(pretrain_state_dict)
        self.eval()
        self.requires_grad_(False)

        self.cfg = cfg
        self._prior_cache = OrderedDict()

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def detector(self, images: list[torch.Tensor] | torch.Tensor, conf_thresh: float = 0.85, min_box_size: int = 0):

        if isinstance(images, torch.Tensor):
            images, scales = images.clone(), None
        elif isinstance(images, list) and all(isinstance(x, torch.Tensor) for x in images):
            images, scales = batch_resize_and_pad_varsize(images, CFG_R50["image_size"], CFG_R50["image_size"])

        else:
            raise TypeError("frame must be torch.Tensor or list[torch.Tensor]")

        if self.from_normalized:
            # images = (images + 1.0) * 127.5  # [-1, 1] → [0, 255]
            images.add_(1.0).mul_(127.5)
        elif self.from_unit_range:
            # images = images * 255.0  # [0, 1] → [0, 255]
            images.mul_(255.0)

        if self.swap_rb_ch:
            images = images[:, [2, 1, 0], :, :]

        images = images.sub_(self.mean)

        out = self.body(images)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        loc = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        conf = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        landms = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        H, W = images.shape[-2:]
        results, idxs = self.retinaface_postprocess(loc, conf, landms, (H, W), resize=scales, conf_thresh=conf_thresh, min_box_size=min_box_size)

        return results, idxs  # [N,15]

    def forward(self, inputs: torch.Tensor):
        if self.from_normalized:
            inputs = (inputs + 1.0) * 127.5  # [-1, 1] → [0, 255]
        elif self.from_unit_range:
            inputs = inputs * 255.0  # [0, 1] → [0, 255]

        if self.swap_rb_ch:
            inputs = inputs[:, [2, 1, 0], :, :]

        inputs = inputs - self.mean

        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        # loc, conf, landms = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        return (bbox_regressions, classifications, ldm_regressions)

    def generate_prior_boxes(
        self,
        h: int,
        w: int,
        device=torch.device("cpu"),
    ):
        key = (h, w)

        # 如果在缓存中，移动到末尾（最近使用）
        if key in self._prior_cache:
            self._prior_cache.move_to_end(key)
            return self._prior_cache[key]

        cfg = self.cfg
        min_sizes, steps, clip = cfg["min_sizes"], cfg["steps"], cfg["clip"]

        # 计算特征图尺寸 [height, width]
        feature_maps = [[ceil(h / step), ceil(w / step)] for step in steps]

        anchors = []

        for k, f in enumerate(feature_maps):
            min_sizes_k = min_sizes[k]
            for i, j in itertools.product(range(f[0]), range(f[1])):
                for min_size in min_sizes_k:
                    s_kx = min_size / w  # width方向
                    s_ky = min_size / h  # height方向

                    # 计算中心点
                    dense_cx = (j + 0.5) * steps[k] / w
                    dense_cy = (i + 0.5) * steps[k] / h

                    anchors.extend([dense_cx, dense_cy, s_kx, s_ky])

        # 转换为tensor
        output = torch.tensor(anchors, device=device).view(-1, 4)

        if clip:
            output.clamp_(max=1, min=0)

        # 添加到缓存
        self._prior_cache[key] = output
        self._prior_cache.move_to_end(key)

        # 如果超过缓存大小，丢弃最旧的
        if len(self._prior_cache) > 10:
            self._prior_cache.popitem(last=False)

        return output

    def retinaface_postprocess(
        self,
        loc: torch.Tensor,
        conf: torch.Tensor,
        landms: torch.Tensor,
        image_shape: Tuple[int, int],
        conf_thresh: float = 0.6,
        iou_thresh: float = 0.4,
        resize: torch.Tensor | None = None,
        min_box_size: int = 0,
    ):
        """
        Args:
            loc:     [B, N, 4] - 回归偏移
            conf:    [B, N, 2] - 分类 logits
            landms:  [B, N, 10] - 关键点偏移
            cfg:     dict - 包含 "variance" 项
            image_shape: Tuple[int, int] - 原始图像尺寸 (H, W)
            conf_thresh: float - 置信度阈值
            iou_thresh: float - NMS 阈值
            resize: [N, float] - 输入图像到原图的缩放因子（默认 1）

        Returns:
            List[Tensor] - 每个 batch 的检测结果 [K, 15]: x1,y1,x2,y2,score,x0,y0,...,x4,y4
        """

        cfg = self.cfg

        B = loc.shape[0]
        device = loc.device

        priors = self.generate_prior_boxes(*image_shape, device=device)

        if resize is None:
            resize = torch.ones((B, 1), dtype=torch.float, device=device)

        W, H = image_shape[1], image_shape[0]
        inv_resize = 1.0 / resize

        results: List[torch.Tensor] = []
        idxs: List[int] = []
        for b in range(B):
            conf_b = conf[b]  # [N, 2]
            scores = conf_b[:, 1]  # 人脸概率

            mask = scores > conf_thresh
            if mask.sum() == 0:
                # results.append(torch.empty((0, 15), device=device))
                idxs.append(0)
                continue

            boxes_b = decode_boxes(loc[b][mask], priors[mask], cfg["variance"])
            # boxes: [x1,y1,x2,y2]
            boxes_b[:, 0::2] *= W * inv_resize[b]  # X方向
            boxes_b[:, 1::2] *= H * inv_resize[b]  # Y方向

            widths = boxes_b[:, 2] - boxes_b[:, 0]
            heights = boxes_b[:, 3] - boxes_b[:, 1]
            size_mask = (widths >= min_box_size) & (heights >= min_box_size)

            if size_mask.sum() == 0:
                # results.append(torch.empty((0, 15), device=device))
                idxs.append(0)
                continue

            boxes_b = boxes_b[size_mask]
            scores_b = scores[mask][size_mask]
            landms_b = decode_landmarks(landms[b][mask][size_mask], priors[mask][size_mask], cfg["variance"])
            landms_b[:, 0::2] *= W * inv_resize[b]
            landms_b[:, 1::2] *= H * inv_resize[b]

            keep = ops.nms(boxes_b, scores_b, iou_thresh)

            boxes_b = boxes_b[keep]
            landms_b = landms_b[keep]
            scores_b = scores_b[keep].unsqueeze(1)

            out = torch.cat([boxes_b, scores_b, landms_b], dim=1)  # [K, 15]
            idxs.append(len(out))
            results.append(out)

        if len(results) == 0:
            return torch.empty(0, 15, device=device), idxs

        return torch.cat(results), idxs


def batch_resize_and_pad_varsize(imgs: List[torch.Tensor], out_h: int, out_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将一组不同大小的图片等比例缩放后填充到固定大小区域 (out_h, out_w)，左上角对齐。
    返回：处理结果 + 每张图的缩放比例。

    参数:
        imgs: List of Tensor，每个 [C, H_i, W_i]
        out_h: 输出高度
        out_w: 输出宽度

    返回:
        out: Tensor, [N, C, out_h, out_w]
        scales: Tensor, [N, S] 缩放比例
    """
    N = len(imgs)
    device = imgs[0].device
    dtype = imgs[0].dtype
    C = imgs[0].shape[0]

    # 输出张量初始化
    out = torch.zeros((N, C, out_h, out_w), device=device, dtype=dtype)
    scales = torch.zeros((N, 1), device=device, dtype=dtype)

    for i, img in enumerate(imgs):

        H_i, W_i = img.shape[-2:]
        scale = min(out_h / H_i, out_w / W_i)
        scales[i] = scale

        new_h = int(H_i * scale)
        new_w = int(W_i * scale)

        img_resized = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode="bicubic", align_corners=False).squeeze_(0)
        out[i, :, :new_h, :new_w] = img_resized

    return out, scales


@torch.jit.script
def decode_boxes(loc: torch.Tensor, priors: torch.Tensor, variances: List[float]) -> torch.Tensor:
    """
    Args:
        loc:     [N,4]  (dx, dy, dw, dh)
        priors:  [N,4]  (cx, cy, w, h)
        variances: [2]  [variance_xy, variance_wh]
    Returns:
        boxes: [N,4]  (x1, y1, x2, y2)
    """
    # 预分配输出
    boxes = loc.new_empty(loc.size(0), 4)

    # 中心点
    cx = priors[:, 0] + loc[:, 0] * variances[0] * priors[:, 2]
    cy = priors[:, 1] + loc[:, 1] * variances[0] * priors[:, 3]

    w = priors[:, 2] * torch.exp(loc[:, 2] * variances[1])
    h = priors[:, 3] * torch.exp(loc[:, 3] * variances[1])

    # 半宽高
    half_w = w * 0.5
    half_h = h * 0.5

    # 左上右下
    boxes[:, 0] = cx - half_w
    boxes[:, 1] = cy - half_h
    boxes[:, 2] = cx + half_w
    boxes[:, 3] = cy + half_h

    return boxes


@torch.jit.script
def decode_landmarks(landms: torch.Tensor, priors: torch.Tensor, variances: List[float]) -> torch.Tensor:
    """
    Args:
        landms:  [N,10]   (5 landmarks: x,y,...)
        priors:  [N,4]    (cx, cy, w, h)
        variances: [2]    (仅用到 variances[0])
    Returns:
        out: [N,10]
    """
    N = landms.size(0)
    out = landms.new_empty(N, 10)

    cx = priors[:, 0]
    cy = priors[:, 1]
    w = priors[:, 2] * variances[0]
    h = priors[:, 3] * variances[0]

    out[:, 0::2] = cx.unsqueeze(1) + landms[:, 0::2] * w.unsqueeze(1)
    out[:, 1::2] = cy.unsqueeze(1) + landms[:, 1::2] * h.unsqueeze(1)

    return out


def draw_boxes_batch(images: torch.Tensor, boxes_list: list, color=(0.0, 1.0, 0.0), thickness=3, point_size=7):
    """
    Args:
        images: Tensor[B, C, H, W], 范围 [0,1]
        boxes_list: List of length B, 每个元素 Tensor[N_i, 4]，格式 (x1,y1,x2,y2)，像素坐标
        color: tuple(float, float, float), RGB颜色
        thickness: int，线宽
    Returns:
        Tensor[B, C, H, W]，画框后的新图像
    """
    B, C, H, W = images.shape
    images = images.clone()
    color_tensor = torch.tensor(color, device=images.device).view(C, 1, 1)

    for b in range(B):
        boxes = boxes_list[b]

        if boxes.numel() == 0:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box[0:4].int()

            # 确保坐标不超出边界
            x1 = max(0, min(x1.item(), W - 1))
            x2 = max(0, min(x2.item(), W - 1))
            y1 = max(0, min(y1.item(), H - 1))
            y2 = max(0, min(y2.item(), H - 1))

            # 画上边框
            images[b, :, y1 : y1 + thickness, x1:x2] = color_tensor
            # 画下边框
            images[b, :, y2 - thickness : y2, x1:x2] = color_tensor
            # 画左边框
            images[b, :, y1:y2, x1 : x1 + thickness] = color_tensor
            # 画右边框
            images[b, :, y1:y2, x2 - thickness : x2] = color_tensor

            # 绘制关键点
            keypoints = box[5:15].view(5, 2)  # 提取5个关键点
            for kp in keypoints:
                kx = kp[0].int().item()  # x 坐标
                ky = kp[1].int().item()  # y 坐标
                kx = max(0, min(kx, W - 1))  # 限制在图像边界内
                ky = max(0, min(ky, H - 1))
                half_size = point_size // 2
                kx_start = max(0, kx - half_size)
                kx_end = min(W, kx + half_size + 1)
                ky_start = max(0, ky - half_size)
                ky_end = min(H, ky + half_size + 1)
                images[b, :, ky_start:ky_end, kx_start:kx_end] = color_tensor  # 填充正方形区域

    return images


def get_5pts(result: torch.Tensor):
    return result[..., 5:15].view(-1, 5, 2)  # [K, 5, 2]


if __name__ == "__main__":
    import torchvision.utils as utils

    from misc.utils import DataFolder, Timer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 6

    print(f"使用设备: {device}")

    model = Model().to(device=device)
    model = model.eval()
    model = model.requires_grad_(False)

    # model = torch.compile(
    #     model,
    #     fullgraph=True,
    #     dynamic=False,
    #     options={
    #         "triton.cudagraphs": True,
    #         "max_autotune": True,
    #         "epilogue_fusion": True,
    #     },
    # )

    # dataset = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024")
    dataset = DataFolder("IDAssets")
    images_org = [dataset.random_acquire_tensor().float().to(device=device) for _ in range(batch_size)]

    images, scales = batch_resize_and_pad_varsize(images_org, CFG_R50["image_size"], CFG_R50["image_size"])

    for _ in range(5):
        loc, conf, landms = model(images)
        result = model.retinaface_postprocess(loc, conf, landms, images.shape[-2:], conf_thresh=0.75, resize=scales)

    with Timer("model infer"):
        loc, conf, landms = model(images)
        result = model.retinaface_postprocess(loc, conf, landms, images.shape[-2:], conf_thresh=0.75, resize=scales)

    visi = draw_boxes_batch(images_org[1].unsqueeze(0), result[1].unsqueeze(0))

    utils.save_image(visi, "visi.png", normalize=True, value_range=(0, 255))
