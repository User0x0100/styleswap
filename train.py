import os
import time
import itertools
from torchvision import utils
from pathlib import Path
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from losses.arcface import ArcFace
from model import FullGenerator, StyleGAN2Discriminator
from unetdisc import (
    U_Net_Discriminator,
    gradient_penalty,
    calc_pl_lengths,
    update_average,
)
from losses import GANLoss, L1Loss, PerceptualLoss, r1_penalty, g_path_regularize
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader.data_loader import FaceDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


DEVICE_TYPE = "cuda"

if DEVICE_TYPE == "cuda":
    assert torch.cuda.is_available()


class TrainSet:

    device: torch.device = torch.device(DEVICE_TYPE, 0)
    is_distributed: bool = False

    pretrain: None | str = None

    batch: int = 8
    size: int = 512
    style_dim: int = 512
    n_mlp: int = 8
    isconcat: bool = True

    save_samples_every: int = 250
    save_ckpt_every: int = 5000
    grad_accum_steps: int = 1

    r1_reg_every: int = 16
    r1_reg_weight: float = 10.0

    path_reg_every: int = 4
    path_reg_weight: float = 2.0

    d_lr: float = 0.0001
    g_lr: float = 0.0001

    base_path: str = "log/train_unet"
    ckpt_dir: str = "ckpt"
    sample_dir: str = "sample"
    tb_log_dir: str = "tb_log"

    id_loss_weight: float = 5.0

    pixel_rec_weight: float = 100.0

    gan_loss_weight: float = 1.0

    PerceptualLoss_loss_type: str = "vgg19"
    PerceptualLoss_layer_weights: dict[str, float] = {
        # "conv3_4": 1.0,
        # "relu3_4": 1.0,
        # "relu4_3": 1.0
        "conv4_3": 1.0,
    }
    PerceptualLoss_use_input_norm: bool = True
    PerceptualLoss_range_norm: bool = True
    PerceptualLoss_perceptual_weight: float = 100.0
    PerceptualLoss_style_weight: float = 100.0
    PerceptualLoss_criterion: str = "l1"

    mask_loss_weight: float = 1.0

    same_image_prob: float = 0.15

    # "/opt/share/deepfake/dataset_1/ffhq_1024",
    # "/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024",
    # "/opt/share/deepfake/dataset_1/vggface2_hq512",
    # "/opt/share/deepfake/dataset_1/closed_eyes"
    # "/opt/share/deepfake/dataset_1/large_angle"

    image_path = None

    target_dirs = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024", 0.25),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024", 0.25),
        ("/opt/share/deepfake/dataset_1/closed_eyes", 0.25),
        ("/opt/share/deepfake/dataset_1/large_angle", 0.25),
    ]
    source_dirs = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024", 0.5),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024", 0.5),
    ]


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(self, config: TrainSet):

        self.cfg = config
        self.iter = 1

        torch.cuda.set_device(self.cfg.device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        self.loss_dict = {}

        self.net_d = U_Net_Discriminator(image_size=self.cfg.size).to(self.cfg.device)

        self.net_g = FullGenerator(
            self.cfg.size,
            style_dim=self.cfg.style_dim,
            n_mlp=self.cfg.n_mlp,
            isconcat=self.cfg.isconcat,
        ).to(self.cfg.device)
        self.net_g_ema = FullGenerator(
            self.cfg.size,
            style_dim=self.cfg.style_dim,
            n_mlp=self.cfg.n_mlp,
            isconcat=self.cfg.isconcat,
        ).to(self.cfg.device)

        if self.cfg.pretrain is not None:
            print(f"Loading pretrain checkpoint from {self.cfg.pretrain}")

            checkpoint = torch.load(
                self.cfg.pretrain,
                map_location=self.cfg.device,
                weights_only=False,
            )

            self.iter = checkpoint.get("iter", 0)
            print(f"iter = {self.iter}")

            self.net_d.load_state_dict(checkpoint["d"])
            self.net_g.load_state_dict(checkpoint["g"])
            self.net_g_ema.load_state_dict(checkpoint["g_ema"])
            print("Pretrain checkpoint loaded successfully.")

        self.net_d.train()
        self.net_g.train()
        self.net_g_ema.eval()

        if self.cfg.is_distributed:

            self.net_d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_d)
            self.net_d = DDP(
                self.net_d,
                device_ids=[self.cfg.device.index],
                output_device=self.cfg.device.index,
                broadcast_buffers=False,
            )
            self.net_g = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_g)
            self.net_g = DDP(
                self.net_g,
                device_ids=[self.cfg.device.index],
                output_device=self.cfg.device.index,
                broadcast_buffers=False,
            )

        net_d_params = (
            self.net_d.module.parameters()
            if self.cfg.is_distributed
            else self.net_d.parameters()
        )
        net_g_params = (
            self.net_g.module.parameters()
            if self.cfg.is_distributed
            else self.net_g.parameters()
        )

        self.d_optimizer = torch.optim.Adam(
            params=net_d_params,
            lr=self.cfg.d_lr,
            betas=(0.0, 0.99),
        )

        self.g_optimizer = torch.optim.Adam(
            params=net_g_params,
            lr=self.cfg.g_lr,
            betas=(0.0, 0.99),
        )

        # if self.cfg.pretrain is not None:
        #     self.d_optimizer.load_state_dict(checkpoint["d_optim"])
        #     for param_group in self.d_optimizer.param_groups:
        #         param_group["lr"] = self.cfg.d_lr

        #     self.g_optimizer.load_state_dict(checkpoint["g_optim"])
        #     for param_group in self.g_optimizer.param_groups:
        #         param_group["lr"] = self.cfg.g_lr
        #     print("Optimizer state loaded from checkpoint.")

        print(f"d_optimizer_lr: {self.d_optimizer.param_groups[0]["lr"]}")
        print(f"g_optimizer_lr: {self.g_optimizer.param_groups[0]["lr"]}")

        self.ckpt_dir = os.path.join(self.cfg.base_path, self.cfg.ckpt_dir)
        self.sample_dir = os.path.join(self.cfg.base_path, self.cfg.sample_dir)
        self.tb_log_dir = os.path.join(self.cfg.base_path, self.cfg.tb_log_dir)

        Path(self.ckpt_dir).mkdir(exist_ok=True, parents=True)
        Path(self.sample_dir).mkdir(exist_ok=True, parents=True)
        Path(self.tb_log_dir).mkdir(exist_ok=True, parents=True)

        self.perceptualLoss = PerceptualLoss(
            self.cfg.PerceptualLoss_layer_weights,
            self.cfg.PerceptualLoss_loss_type,
            self.cfg.PerceptualLoss_use_input_norm,
            self.cfg.PerceptualLoss_range_norm,
            self.cfg.PerceptualLoss_perceptual_weight,
            self.cfg.PerceptualLoss_style_weight,
            self.cfg.PerceptualLoss_criterion,
        ).to(self.cfg.device)

        self.arcface = ArcFace().to(self.cfg.device)
        self.arcface.eval()
        self.arcface.requires_grad_(False)

        dataset = FaceDataset(
            size=self.cfg.size,
            target_dirs=self.cfg.target_dirs,
            source_dirs=self.cfg.source_dirs,
            same_image_prob=self.cfg.same_image_prob,
            device=self.cfg.device,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
            pin_memory_device=str(self.cfg.device),
            prefetch_factor=3,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        self.loader = sample_data(loader)

        if self.cfg.device.index == 0:
            self.writer = SummaryWriter(log_dir=self.tb_log_dir)

    def model_ema(self, decay=0.999):

        net_g_params = self.net_g.module if self.cfg.is_distributed else self.net_g
        net_g_params = dict(net_g_params.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(
                net_g_params[k].data, alpha=1 - decay
            )

    def save(self):

        cfg = self.cfg

        if cfg.is_distributed:
            d = self.net_d.module
            g = self.net_g.module
        else:
            d = self.net_d
            g = self.net_g

        save_dict = {
            "iter": self.iter,
            "d": d.state_dict(),
            "g": g.state_dict(),
            "g_ema": self.net_g_ema.state_dict(),
            "d_optim": self.d_optimizer.state_dict(),
            "g_optim": self.g_optimizer.state_dict(),
        }

        save_file = os.path.join(self.ckpt_dir, f"{str(self.iter)}_{self.cfg.size}.pth")

        torch.save(save_dict, save_file)

    def fetch_training_batch(self):
        cfg = self.cfg
        target, source, mask, same = next(self.loader)

        target = target.to(cfg.device)
        source = source.to(cfg.device)
        mask = mask.to(cfg.device)
        same = same.to(cfg.device)
        return target, source, mask, same

    def reduce_loss_dict(self):

        if not self.cfg.is_distributed:
            return

        world_size = dist.get_world_size()

        if world_size < 2:
            return

        with torch.no_grad():

            sorted_items = sorted(self.loss_dict.items())
            keys, losses = zip(*sorted_items) if sorted_items else ([], [])

            losses_tensor = torch.stack(losses, dim=0)
            dist.reduce(losses_tensor, op=dist.ReduceOp.SUM, dst=0)
            if self.cfg.device.index == 0:
                losses_tensor /= world_size
                self.loss_dict.update(zip(keys, losses_tensor))

    def train(self):

        cfg = self.cfg
        if cfg.device.index == 0:
            pbar = tqdm(
                itertools.count(start=self.iter), initial=self.iter, unit="iter"
            )
        else:
            pbar = itertools.count(start=self.iter)

        # pl_mean = 0

        # 主训练循环
        for self.iter in pbar:

            # =============== 训练判别器 ===============
            self.net_d.requires_grad_(True)
            self.net_g.requires_grad_(False)
            self.d_optimizer.zero_grad()
            for i in range(cfg.grad_accum_steps):

                target, source, mask, same = self.fetch_training_batch()
                is_r1_reg_every = self.iter % cfg.r1_reg_every == 0 and i == 0

                with torch.no_grad():
                    z_id = self.arcface(source)
                    fake_img, _, _ = self.net_g(target, z_id)

                fake_enc_out, fake_dec_out = self.net_d(fake_img)

                if is_r1_reg_every:
                    target.requires_grad_(True)

                real_enc_out, real_dec_out = self.net_d(target)

                enc_divergence = (
                    F.relu(1 + real_enc_out) + F.relu(1 - fake_enc_out)
                ).mean()
                dec_divergence = (
                    F.relu(1 + real_dec_out) + F.relu(1 - fake_dec_out)
                ).mean()
                divergence = enc_divergence + dec_divergence * cfg.gan_loss_weight

                self.loss_dict["D"] = divergence.detach().mean()
                self.loss_dict["fake_enc_out"] = fake_enc_out.detach().mean()
                self.loss_dict["fake_dec_out"] = fake_dec_out.detach().mean()
                self.loss_dict["real_enc_out"] = real_enc_out.detach().mean()
                self.loss_dict["real_dec_out"] = real_dec_out.detach().mean()

                if is_r1_reg_every:

                    gp = gradient_penalty(target, (real_enc_out,))
                    divergence += gp
                    self.loss_dict["gp"] = gp.detach().mean()

                divergence.backward()

            self.d_optimizer.step()

            # =============== 训练生成器 ===============
            self.net_d.requires_grad_(False)
            self.net_g.requires_grad_(True)
            self.g_optimizer.zero_grad()
            for i in range(cfg.grad_accum_steps):

                target, source, mask, same = self.fetch_training_batch()
                with torch.no_grad():
                    z_id = self.arcface(source)

                fake_img, _, fake_mask = self.net_g(target, z_id)
                fake_enc_output, fake_dec_output = self.net_d(fake_img)

                # ------------- 对抗损失 -------------
                g_loss = fake_enc_output.mean() + F.relu(1 + fake_dec_output).mean()
                self.loss_dict["G"] = g_loss.detach()

                # ------------- 身份损失 -------------
                fake_z_id = self.arcface(fake_img)
                id_loss = (1 - torch.cosine_similarity(fake_z_id, z_id)).mean()
                self.loss_dict["ID"] = id_loss.detach()
                g_loss += id_loss * cfg.id_loss_weight

                # ------------- 感知损失 -------------
                percep_loss, style_loss = self.perceptualLoss(fake_img, target)
                if percep_loss is not None:
                    g_loss += percep_loss
                    self.loss_dict["Percep"] = percep_loss.detach()
                if style_loss is not None:
                    g_loss += style_loss
                    self.loss_dict["Style"] = style_loss.detach()

                # ------------- 启发式误差识别细化(FaceShifter) -------------
                err_per_sample = torch.abs(fake_img - target).mean(dim=(1, 2, 3))
                numerator = torch.dot(err_per_sample, same)
                denominator = same.sum().clamp_min(1e-8)  # 防止除零
                rec_pix = numerator / denominator
                g_loss += rec_pix * cfg.pixel_rec_weight
                self.loss_dict["Pixel_Rec"] = rec_pix

                # ------------- 掩码损失 -------------
                mask_loss = F.binary_cross_entropy(fake_mask, mask).mean()
                self.loss_dict["Mask"] = mask_loss.detach()
                g_loss += mask_loss * cfg.mask_loss_weight

                g_loss = g_loss / cfg.grad_accum_steps

                # if self.iter % cfg.path_reg_every == 0 and i == 0:

                #     pl_lengths = calc_pl_lengths(latents, fake_img)
                #     avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())
                #     pl_mean = update_average(pl_mean, avg_pl_length)
                #     pl_loss = ((pl_lengths - pl_mean) ** 2).mean()
                #     g_loss += pl_loss
                #     self.loss_dict["pl_loss"] = pl_loss.detach().mean()
                #     self.loss_dict["pl_lengths"] = pl_lengths.detach().mean()

                # 反向传播
                g_loss.backward()

            self.g_optimizer.step()

            if cfg.device.index == 0:
                self.model_ema()

            if self.iter % 10 == 0:
                self.reduce_loss_dict()

            if cfg.device.index == 0:
                # 统计所有损失和分数
                values = {}
                for k, v in self.loss_dict.items():
                    # v 可能是张量或元组
                    if isinstance(v, tuple):
                        v = v[0]
                    if torch.is_tensor(v):
                        values[k] = v.mean().item()
                    else:
                        values[k] = float(v)

                # 记录到Tensorboard
                for k, v in values.items():
                    self.writer.add_scalar(f"Loss/{k}", v, self.iter)

                # 定期保存样本和日志
                if self.iter % cfg.save_samples_every == 0:

                    # 生成样本图像
                    with torch.no_grad():
                        fake, _, fake_mask = self.net_g_ema(target, z_id)
                        fake = fake * fake_mask + (1 - fake_mask) * target
                        fake_mask = (fake_mask.repeat(1, 3, 1, 1) - 0.5) / 0.5
                        mask = (mask.repeat(1, 3, 1, 1) - 0.5) / 0.5
                        grid = torch.cat((target, source, fake, fake_mask, mask), 0)
                        save_file = os.path.join(
                            self.sample_dir, f"{str(self.iter)}_{cfg.size}.png"
                        )

                        # 保存样本图像
                        utils.save_image(
                            grid,
                            save_file,
                            nrow=cfg.batch,
                            normalize=True,
                            value_range=(-1, 1),
                        )

                # 定期保存模型
                if self.iter % cfg.save_ckpt_every == 0:
                    self.save()


if __name__ == "__main__":

    cfg = TrainSet()
    cfg.pretrain = "log/train_unet/ckpt/350000_512.pth"

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    is_distributed = world_size > 1

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"分布式训练: world_size={world_size}, local_rank={local_rank}")
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size
        )
        cfg.device = torch.device(DEVICE_TYPE, local_rank)
        cfg.is_distributed = is_distributed
    else:
        print("单卡训练")
        cfg.device = torch.device(DEVICE_TYPE, 0)
        cfg.is_distributed = is_distributed

    trainer = Trainer(cfg)
    try:
        trainer.train()
    finally:
        if trainer.cfg.is_distributed:
            torch.distributed.barrier()
        if trainer.cfg.device.index == 0:
            trainer.save()
            trainer.writer.close()
            dist.destroy_process_group()

        torch.cuda.empty_cache()
