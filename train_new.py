import itertools
from pathlib import Path
from typing import Any

import cv2
import torch
from torch import optim
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from torch import Tensor
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader import datasetloader
from model import FullGenerator
from losses import IDLoss, L1Loss
from dfm.losses import PerceptualLoss, DLoss, GANLoss, MaskLoss
from misc.model import FaceParsing, IDEncoder
from dfm.model.discriminator import UNetDiscriminatorSN


EPS = 1e-6

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class Trainer:
    def __init__(
        self,
        src: list[tuple[str, float]],
        dst: list[tuple[str, float]],
        batch_size: int = 8,
        lr: float = 1e-4,
        bf16: bool = True,
        device: str = "cuda:0",
        weight: str | None = None,
        log_path: str = "log_dfm/train_styleswap",
        sample_save_every: int = 1000,
        weight_save_every: int = 10000,
        # 模型配置
        size: int = 128,
        style_dim: int = 512,
        n_mlp: int = 8,
        isconcat: bool = False,
        noise_blank_level: int = 9,
        # 损失配置
        id_encode_provider: IDEncoder.PROVIDER = IDEncoder.PROVIDER.ARC_OFFICIAL_R50,
        id_loss_weight: float = 10.0,
        wfm_weight: float = 100.0,
        rec_loss: float = 100.0,
        vgg19_loss_weight: dict[str, float] = {"conv3_4": 10.0},
        mask_loss_weight: float = 1.0,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.bf16 = bf16
        self.sample_save_every = sample_save_every
        self.weight_save_every = weight_save_every

        # ========================= Init Model =========================
        self.net_d = UNetDiscriminatorSN().to(self.device)

        if weight is not None:
            if Path(weight).exists() == False:
                raise FileNotFoundError(f"CheckPoint file: {weight} Not found")

            print(f"Loading weight from {weight}")
            weight: dict[str, Any] = torch.load(weight, map_location=torch.device("cpu"), weights_only=False)

            self.iter, self.size, self.style_dim, self.n_mlp, self.isconcat, self.noise_blank_level = (
                weight[k] for k in ("iter", "size", "style_dim", "n_mlp", "isconcat", "noise_blank_level")
            )

            print(
                f"Weight Info:\n"
                f"  {'iter':25}: {self.iter}\n"
                f"  {'size':25}: {self.size}\n"
                f"  {'style_dim':25}: {self.style_dim}\n"
                f"  {'n_mlp':25}: {self.n_mlp}\n"
                f"  {'isconcat':25}: {self.isconcat}\n"
                f"  {'noise_blank_level':25}: {self.noise_blank_level}"
            )

            self.net_g = FullGenerator(size=self.size, style_dim=self.style_dim, n_mlp=self.n_mlp, isconcat=self.isconcat).to(self.device)
            self.net_g_ema = FullGenerator(size=self.size, style_dim=self.style_dim, n_mlp=self.n_mlp, isconcat=self.isconcat).to(self.device)

            for module_name in ["net_g", "net_g_ema", "net_d"]:
                getattr(self, module_name).load_state_dict(weight[module_name])
        else:

            self.size = size
            self.style_dim = style_dim
            self.n_mlp = n_mlp
            self.isconcat = isconcat
            self.noise_blank_level = noise_blank_level

            self.iter = 0
            self.noise_blank_level = noise_blank_level
            self.net_g = FullGenerator(size=self.size, style_dim=self.style_dim, n_mlp=self.n_mlp, isconcat=self.isconcat).to(self.device)
            self.net_g_ema = FullGenerator(size=self.size, style_dim=self.style_dim, n_mlp=self.n_mlp, isconcat=self.isconcat).to(self.device)

        self.net_g.train()
        self.net_g_ema.eval()
        self.net_d.train()

        # ========================= Optim =========================
        self.g_optim = optim.Adam(self.net_g.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)
        self.d_optim = optim.Adam(self.net_d.parameters(), lr=lr, betas=(0.0, 0.99), fused=True)

        # ========================= LOSS =========================

        self.d_loss = DLoss(weight=1.0, reduction="mean").to(self.device)
        self.gan_loss = GANLoss(weight=1.0, reduction="none").to(self.device)

        self.id_loss = IDLoss(weight=id_loss_weight, id_encode_provider=id_encode_provider).to(self.device)
        self.wfm_loss = L1Loss(loss_weight=wfm_weight).to(self.device)

        self.rec_loss = L1Loss(loss_weight=rec_loss, reduction="none").to(self.device)
        self.vgg19_loss = PerceptualLoss(layer_weights=vgg19_loss_weight).to(self.device)

        self.mask_loss = MaskLoss(weight=mask_loss_weight).to(self.device)

        # ========================= LOG =========================
        base_log_path = Path(log_path)
        self.ckpt_dir = base_log_path.joinpath("ckpt")
        self.sample_dir = base_log_path.joinpath("sample")
        self.tb_log_dir = base_log_path.joinpath("tb_log")

        for p in [self.ckpt_dir, self.sample_dir, self.tb_log_dir]:
            p.mkdir(exist_ok=True, parents=True)

        self.log_writer = SummaryWriter(log_dir=self.tb_log_dir, max_queue=100)

        # ========================= Sample =========================
        pipe = datasetloader(
            batch_size=self.batch_size,
            num_threads=1,
            prefetch_queue_depth=25,
            py_num_workers=1,
            py_start_method="spawn",
            device_id=self.device.index,
            resize=self.size,
            src=src,
            dst=dst,
            same_image_prob=0.1,
        )

        self.dataset = DALIGenericIterator(pipelines=pipe, output_map=["src", "dst", "is_same"], auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)
        # ========================= Utils =========================
        self.face_segment = FaceParsing.Model(range_norm=True, occ=False)
        self.face_segment.to(device=self.device)
        self.face_segment.eval()
        self.face_segment.requires_grad_(False)

    @torch.no_grad()
    def log(self, k: str, v: Tensor, now: bool = False) -> None:
        if now or self.iter % 10 == 0:
            self.log_writer.add_scalar(f"Loss/{k}", v.detach().mean().item(), self.iter)

    @torch.no_grad()
    def fetch_sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        data: dict[str, Tensor] = self.dataset.next()[0]

        src, dst, is_same = data["src"], data["dst"], data["is_same"].squeeze_(-1)

        dst_mask: Tensor = self.face_segment(dst)  # [0.0 ~ 1.0]

        return src, dst, dst_mask, is_same

    def save_state_dict(self):
        state_dict = {
            "iter": self.iter,
            "size": self.size,
            "style_dim": self.style_dim,
            "n_mlp": self.n_mlp,
            "isconcat": self.isconcat,
            "noise_blank_level": self.noise_blank_level,
            "net_g": self.net_g.state_dict(),
            "net_g_ema": self.net_g_ema.state_dict(),
            "net_d": self.net_d.state_dict(),
        }
        try:
            ckpt_file = self.ckpt_dir / f"{self.iter}.pth"
            torch.save(state_dict, ckpt_file)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    @torch.no_grad()
    def ema_update(self, beta: float = 0.999):

        weight = 1.0 - beta
        for ema_param, param in zip(self.net_g_ema.parameters(), self.net_g.parameters()):
            ema_param.data.lerp_(param.data, weight)

    def mean_loss_with_mask(self, loss: Tensor, mask: Tensor) -> Tensor:
        loss *= mask
        return loss.sum() / (mask.sum() + EPS)

    def train(self):
        count = itertools.count(start=self.iter)
        pbar = tqdm(count, initial=self.iter, mininterval=1.0, bar_format="{n_fmt:7} | speed {rate_fmt:3} | train time {elapsed}")

        for self.iter in pbar:
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16):

                src, dst, dst_mask, is_same = self.fetch_sample()

                with torch.no_grad():
                    src_id_feats = self.id_loss.get_id_feats(src, None, realign=False).detach_()

                fake, fake_mask = self.net_g(dst, src_id_feats, noise_blank_level=self.noise_blank_level)

                # ========================= train d =========================
                self.d_optim.zero_grad()

                fake_score = self.net_d(fake.detach())
                real_score = self.net_d(dst.detach())

                d_loss: Tensor = self.d_loss(fake_score, real_score)
                self.log("d_loss", d_loss)

                d_loss.backward()
                self.d_optim.step()

                # ========================= train g =========================
                self.g_optim.zero_grad()

                loss: float | Tensor
                loss = 0.0

                # loss gan
                fake_score, fake_feats = self.net_d(fake, True)
                gan_loss = self.gan_loss(fake_score)
                gan_loss = self.mean_loss_with_mask(gan_loss, dst_mask.detach())
                self.log("gan_loss", gan_loss)
                loss += gan_loss

                # loss id
                fake_id_feats = self.id_loss.get_id_feats(fake, None, realign=False)
                id_loss = self.id_loss(fake_id_feats, src_id_feats)
                self.log("id_loss", id_loss)
                loss += id_loss

                # loss wfm
                dst_feats = self.net_d.get_feats(dst)
                wfm_loss = self.wfm_loss(fake_feats, dst_feats)
                self.log("wfm_loss", wfm_loss)
                loss += wfm_loss

                # loss vgg19
                vgg19_loss = self.vgg19_loss(fake, dst)
                self.log("vgg19_loss", vgg19_loss)
                loss += vgg19_loss

                # loss rec
                rec_loss: Tensor = self.rec_loss(fake, dst)  # BCHW
                rec_loss *= dst_mask.detach()
                rec_loss = rec_loss.mean(dim=[1, 2, 3])
                rec_loss = (rec_loss * is_same).sum() / is_same.sum().clamp_min(1.0)
                self.log("rec_loss", rec_loss)
                loss += rec_loss

                # loss mask
                mask_loss = self.mask_loss(fake_mask, dst_mask)
                self.log("mask_loss", mask_loss)
                loss += mask_loss

                loss.backward()
                self.g_optim.step()

            self.ema_update()

            if self.iter % self.weight_save_every == 0:
                self.save_state_dict()

            if self.iter % self.sample_save_every == 0:
                with torch.no_grad():
                    fake: Tensor
                    fake_mask: Tensor
                    fake, fake_mask = self.net_g_ema(dst, src_id_feats, noise_blank_level=self.noise_blank_level)

                    dst_mask.mul_(2.0).sub_(1.0).clamp_(-1.0, 1.0)
                    fake_mask.mul_(2.0).sub_(1.0).clamp_(-1.0, 1.0)

                    dst_mask = dst_mask.expand(-1, 3, -1, -1)
                    fake_mask = fake_mask.expand(-1, 3, -1, -1)

                    grid = torch.cat((src, dst, fake, dst_mask, fake_mask), dim=0)
                    grid.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
                    grid = make_grid(grid, nrow=self.batch_size)[[2, 1, 0], :, :]  # RGB -> BGR
                    grid = grid.permute(1, 2, 0)  # CHW -> HWC
                    grid_cpu = grid.to(device="cpu", dtype=torch.uint8).numpy()
                    cv2.imwrite(self.sample_dir / f"{self.iter}.png", grid_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])


if __name__ == "__main__":
    src = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
    ]

    dst = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024/realign_arcface_dst", 0.0),
        ("/opt/share/deepfake/dataset_1/RealOcc/image/realign_arcface_dst", 2.0),
    ]
    trainer = Trainer(src, dst, weight="log_dfm/train_styleswap/ckpt/105402.pth")

    try:
        trainer.train()
    finally:
        trainer.save_state_dict()
