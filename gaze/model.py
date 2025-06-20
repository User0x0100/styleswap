import os
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import numpy as np

import timm
import menpo.io as mio

# https://github.com/deepinsight/insightface/blob/master/reconstruction/gaze/README.md

latest_a = os.path.join(os.path.dirname(__file__), "latest_a.ckpt")
eyes3d = os.path.join(os.path.dirname(__file__), "eyes3d.pkl")


class GazeModel(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model(backbone, num_classes=481 * 2 * 3)

    def forward(self, x):
        return self.backbone(x)


class GazeHandler:
    def __init__(self, ckpt_path=latest_a, res_eyes_path=eyes3d, device="cuda:0"):
        R = 12.0
        eyes_mean = mio.import_pickle(res_eyes_path)
        idxs481 = eyes_mean["mask481"]["idxs"]
        self.tri481 = eyes_mean["mask481"]["trilist"]
        self.iris_idx_481 = eyes_mean["mask481"]["idxs_iris"]

        self.mean_l = eyes_mean["left_points"][idxs481][:, [0, 2, 1]]
        self.mean_r = eyes_mean["right_points"][idxs481][:, [0, 2, 1]]

        self.num_face = 1103
        self.num_eye = 481
        self.input_size = 160
        self.model = GazeModel.load_from_checkpoint(ckpt_path).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device

    def get(self, x):

        x = F.interpolate(x, [self.input_size, self.input_size], mode="area")
        y = self.model(x)
        y = y.view(y.shape[0], 962, 3)  # [batch, 962, 3]
        y[:, :, 0:2] += 1
        y[:, :, 0:2] *= self.input_size // 2
        y = y.view(y.shape[0], -1)  # 重新展平为 [batch, 2886]
        return y

    def _compute_gaze_loss(self, a, b, loss_type="cosine", reduction="mean"):

        iris_idx_481 = self.iris_idx_481
        batch_size = a.shape[0]

        # 重新整形为 [batch, 962, 3]
        a = a.view(batch_size, 962, 3)  # [batch, 962, 3]
        b = b.view(batch_size, 962, 3)  # [batch, 962, 3]

        num_eye_points = 481  # 每只眼睛的点数

        # 分离左右眼 (前481个点是左眼，后481个点是右眼)
        a_left = a[:, :num_eye_points, :]  # [batch, 481, 3]
        a_right = a[:, num_eye_points:, :]  # [batch, 481, 3]
        b_left = b[:, :num_eye_points, :]  # [batch, 481, 3]
        b_right = b[:, num_eye_points:, :]  # [batch, 481, 3]

        # 使用真实的虹膜点索引
        a_left_iris = a_left[:, iris_idx_481, :].mean(dim=1)  # [batch, 3]
        a_right_iris = a_right[:, iris_idx_481, :].mean(dim=1)  # [batch, 3]
        b_left_iris = b_left[:, iris_idx_481, :].mean(dim=1)  # [batch, 3]
        b_right_iris = b_right[:, iris_idx_481, :].mean(dim=1)  # [batch, 3]

        # 计算眼球中心点 (取前32个点的均值作为眼球中心)
        a_left_center = a_left[:, :32, :].mean(dim=1)  # [batch, 3]
        a_right_center = a_right[:, :32, :].mean(dim=1)  # [batch, 3]
        b_left_center = b_left[:, :32, :].mean(dim=1)  # [batch, 3]
        b_right_center = b_right[:, :32, :].mean(dim=1)  # [batch, 3]

        # 计算注视向量 (虹膜中心相对于眼球中心的方向)
        a_left_gaze = a_left_iris - a_left_center  # [batch, 3]
        a_right_gaze = a_right_iris - a_right_center  # [batch, 3]
        b_left_gaze = b_left_iris - b_left_center  # [batch, 3]
        b_right_gaze = b_right_iris - b_right_center  # [batch, 3]

        # 归一化注视向量，添加小的epsilon避免除零
        eps = 1e-8
        a_left_gaze = F.normalize(a_left_gaze + eps, p=2, dim=1)
        a_right_gaze = F.normalize(a_right_gaze + eps, p=2, dim=1)
        b_left_gaze = F.normalize(b_left_gaze + eps, p=2, dim=1)
        b_right_gaze = F.normalize(b_right_gaze + eps, p=2, dim=1)

        # 计算损失
        if loss_type == "cosine":
            # 余弦距离损失
            left_loss = 1 - F.cosine_similarity(a_left_gaze, b_left_gaze, dim=1)
            right_loss = 1 - F.cosine_similarity(a_right_gaze, b_right_gaze, dim=1)

        elif loss_type == "l2":
            # L2距离损失
            left_loss = torch.norm(a_left_gaze - b_left_gaze, p=2, dim=1)
            right_loss = torch.norm(a_right_gaze - b_right_gaze, p=2, dim=1)

        elif loss_type == "angular":
            # 角度损失
            def angles_from_vec(vec):
                x, y, z = -vec[:, 2], vec[:, 1], -vec[:, 0]
                theta = torch.atan2(y, x)
                phi = torch.atan2(torch.sqrt(x**2 + y**2), z) - np.pi / 2
                return torch.stack([phi, theta], dim=1)  # [theta_x, theta_y]

            try:
                a_left_angles = angles_from_vec(a_left_gaze)
                a_right_angles = angles_from_vec(a_right_gaze)
                b_left_angles = angles_from_vec(b_left_gaze)
                b_right_angles = angles_from_vec(b_right_gaze)

                left_loss = torch.norm(a_left_angles - b_left_angles, p=2, dim=1)
                right_loss = torch.norm(a_right_angles - b_right_angles, p=2, dim=1)
            except:
                # 如果角度计算失败，回退到余弦损失
                left_loss = 1 - F.cosine_similarity(a_left_gaze, b_left_gaze, dim=1)
                right_loss = 1 - F.cosine_similarity(a_right_gaze, b_right_gaze, dim=1)

        elif loss_type == "combined":
            # 组合损失：余弦 + L2
            left_cosine = 1 - F.cosine_similarity(a_left_gaze, b_left_gaze, dim=1)
            right_cosine = 1 - F.cosine_similarity(a_right_gaze, b_right_gaze, dim=1)
            left_l2 = torch.norm(a_left_gaze - b_left_gaze, p=2, dim=1)
            right_l2 = torch.norm(a_right_gaze - b_right_gaze, p=2, dim=1)

            left_loss = 0.5 * left_cosine + 0.5 * left_l2
            right_loss = 0.5 * right_cosine + 0.5 * right_l2

        # 合并左右眼损失 (平均权重)
        total_loss = 0.5 * left_loss + 0.5 * right_loss  # [batch]

        # 应用reduction
        if reduction == "mean":
            return total_loss.mean()
        elif reduction == "sum":
            return total_loss.sum()
        else:  # 'none'
            return total_loss

    def compute_gaze_loss(self, x, y, loss_type="cosine", reduction="mean"):
        """
        计算两个模型输出之间的眼球注视差异损失

        参数:
        - a, b: 模型输出 tensor, shape: [batch_size, 2886]
                2886 = 962 × 3 (962个3D点的展平坐标)
        - iris_idx_481: 虹膜点的索引列表，如果为None则使用估计值
        - loss_type: 'cosine', 'l2', 'angular', 'combined'
        - reduction: 'mean', 'sum', 'none'

        返回:
        - loss: 损失值
        """
        x = self.get(x)
        y = self.get(y)

        return self._compute_gaze_loss(x, y, loss_type=loss_type, reduction=reduction)

    def compute_gaze_loss_simple(self, a, b, reduction="mean"):
        """
        简化版本：直接对整个输出计算损失
        适用于不确定具体点分布的情况
        """
        if reduction == "mean":
            return F.mse_loss(a, b)
        elif reduction == "sum":
            return F.mse_loss(a, b, reduction="sum")
        else:
            return F.mse_loss(a, b, reduction="none").mean(dim=1)


# 测试代码
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    h = GazeHandler(device=device)
    input1 = torch.randn(
        8, 3, 512, 512, dtype=torch.float32, device=device, requires_grad=True
    )
    input2 = torch.randn(
        8, 3, 512, 512, dtype=torch.float32, device=device, requires_grad=True
    )

    print("Computing model outputs...")
    y = h.get(input1)
    z = h.get(input2)

    print(f"Output shapes: y={y.shape}, z={z.shape}")

    print("Computing losses...")
    loss_cosine = h.compute_gaze_loss(input1, input2)

    loss_simple = h.compute_gaze_loss_simple(input1, input2, reduction="mean")

    print(f"Cosine Loss: {loss_cosine.item():.6f}")

    print(f"Simple MSE Loss: {loss_simple.item():.6f}")

    print("Testing gradient flow...")
    loss_cosine.backward()
    print("Gradient test passed!")
