import os
import itertools
from torchvision import utils
from pathlib import Path
import torch
from losses.arcface import ArcFace
from model import FullGenerator, StyleGAN2Discriminator
from losses import GANLoss, PerceptualLoss, r1_penalty, g_path_regularize
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from losses.au import AULoss
from torch.amp import autocast, GradScaler
from dataloader import swap_pipe
from face_parsing import FaceParsing
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

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

    path_reg_every: int = 32
    path_reg_weight: float = 0.5

    d_lr: float = 0.0001
    g_lr: float = 0.0001

    base_log_path: str = "log/train"
    ckpt_dir: str = "ckpt"
    sample_dir: str = "sample"
    tb_log_dir: str = "tb_log"

    id_loss_weight: float = 4.0

    pixel_rec_weight: float = 1.0

    gan_type: str = "wgan_softplus"
    gan_loss_weight: float = 1.0

    au_loss_weight: float = 1.0

    PerceptualLoss_loss_type: str = "vgg19"
    PerceptualLoss_layer_weights: dict[str, float] = {"conv3_4": 1.0, "conv4_3": 1.0}
    PerceptualLoss_use_input_norm: bool = True
    PerceptualLoss_range_norm: bool = True
    PerceptualLoss_perceptual_weight: float = 1.0
    PerceptualLoss_style_weight: float = 1.0
    PerceptualLoss_criterion: str = "l1"

    mask_loss_weight: float = 1.0

    same_image_prob: float = 0.1

    # "/opt/share/deepfake/dataset_1/ffhq_1024",
    # "/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024",
    # "/opt/share/deepfake/dataset_1/vggface2_hq512",
    # "/opt/share/deepfake/dataset_1/closed_eyes"
    # "/opt/share/deepfake/dataset_1/large_angle"

    image_path = None

    target_folders = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024", 0.333333),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024", 0.333333),
        ("/opt/share/deepfake/dataset_1/vggface2_hq512", 0.333333),
        # ("/opt/share/deepfake/dataset_1/closed_eyes", 0.25),
        # ("/opt/share/deepfake/dataset_1/large_angle", 0.25),
    ]
    source_folders = [
        ("/opt/share/deepfake/dataset_1/ffhq_1024", 0.333333),
        ("/opt/share/deepfake/dataset_1/CelebAHQ-1024x1024", 0.333333),
        ("/opt/share/deepfake/dataset_1/vggface2_hq512", 0.333333),
    ]


class Trainer:
    def __init__(self, cfg: TrainSet):

        self.cfg = cfg
        self.iter = 1

        torch.cuda.set_device(cfg.device)

        if cfg.pretrain is None:
            checkpoint = None
        else:
            print(f"Loading pretrain checkpoint from {cfg.pretrain}")
            checkpoint = torch.load(cfg.pretrain, map_location=cfg.device, weights_only=False)

        self.loss_dict = {}

        self.net_d = StyleGAN2Discriminator(out_size=cfg.size).to(cfg.device)
        self.net_g = FullGenerator(size=cfg.size, style_dim=cfg.style_dim, n_mlp=cfg.n_mlp, isconcat=cfg.isconcat).to(cfg.device)
        self.net_g_ema = FullGenerator(size=cfg.size, style_dim=cfg.style_dim, n_mlp=cfg.n_mlp, isconcat=cfg.isconcat).to(cfg.device)

        if checkpoint is not None:

            if "iter" in checkpoint:
                self.iter = checkpoint["iter"]
                print(f"iter = {self.iter}")

            if "d" in checkpoint:
                self.net_d.load_state_dict(checkpoint["d"])
                print("Loaded net_d state from checkpoint.")

            if "g" in checkpoint:
                self.net_g.load_state_dict(checkpoint["g"])
                print("Loaded net_g state from checkpoint.")

            if "g_ema" in checkpoint:
                self.net_g_ema.load_state_dict(checkpoint["g_ema"])
                print("Loaded net_g_ema state from checkpoint.")

        self.net_d.train()
        self.net_g.train()
        self.net_g_ema.eval()

        if cfg.is_distributed:
            self.net_d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_d)
            self.net_d = DDP(self.net_d, device_ids=[cfg.device.index], output_device=cfg.device.index, broadcast_buffers=False)
            self.net_g = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_g)
            self.net_g = DDP(self.net_g, device_ids=[cfg.device.index], output_device=cfg.device.index, broadcast_buffers=True)

        net_d_params = self.net_d.module.parameters() if cfg.is_distributed else self.net_d.parameters()
        net_g_params = self.net_g.module.parameters() if cfg.is_distributed else self.net_g.parameters()

        self.d_optimizer = torch.optim.Adam(params=net_d_params, lr=cfg.d_lr, betas=(0.0, 0.99))
        self.g_optimizer = torch.optim.Adam(params=net_g_params, lr=cfg.g_lr, betas=(0.0, 0.99))

        if checkpoint is not None:
            if "d_optim" in checkpoint:
                self.d_optimizer.load_state_dict(checkpoint["d_optim"])
                for param_group in self.d_optimizer.param_groups:
                    param_group["lr"] = cfg.d_lr
                print("Loaded d_optimizer state from checkpoint.")
                print(f"d_optimizer_lr: {self.d_optimizer.param_groups[0]["lr"]}")

            if "g_optim" in checkpoint:
                self.g_optimizer.load_state_dict(checkpoint["g_optim"])
                for param_group in self.g_optimizer.param_groups:
                    param_group["lr"] = cfg.g_lr
                print("Loaded g_optimizer state from checkpoint.")
                print(f"g_optimizer_lr: {self.g_optimizer.param_groups[0]["lr"]}")

        self.ckpt_dir = os.path.join(cfg.base_log_path, cfg.ckpt_dir)
        self.sample_dir = os.path.join(cfg.base_log_path, cfg.sample_dir)
        self.tb_log_dir = os.path.join(cfg.base_log_path, cfg.tb_log_dir)

        Path(self.ckpt_dir).mkdir(exist_ok=True, parents=True)
        Path(self.sample_dir).mkdir(exist_ok=True, parents=True)
        Path(self.tb_log_dir).mkdir(exist_ok=True, parents=True)

        self.gan_loss = GANLoss(cfg.gan_type, loss_weight=cfg.gan_loss_weight).to(cfg.device)

        self.perceptualLoss = PerceptualLoss(
            cfg.PerceptualLoss_layer_weights,
            cfg.PerceptualLoss_loss_type,
            cfg.PerceptualLoss_use_input_norm,
            cfg.PerceptualLoss_range_norm,
            cfg.PerceptualLoss_perceptual_weight,
            cfg.PerceptualLoss_style_weight,
            cfg.PerceptualLoss_criterion,
        ).to(cfg.device)

        self.au_loss = AULoss(weight=cfg.au_loss_weight).to(cfg.device)

        self.arcface = ArcFace().to(cfg.device)
        self.arcface.eval()
        self.arcface.requires_grad_(False)

        pipe = swap_pipe(
            batch_size=cfg.batch,
            num_threads=2,
            prefetch_queue_depth=4,
            py_num_workers=2,
            py_start_method="spawn",
            device_id=cfg.device.index,
            resize=cfg.size,
            target_folders=cfg.target_folders,
            source_folders=cfg.source_folders,
            same_image_prob=cfg.same_image_prob,
        )

        self.loader = DALIGenericIterator(
            pipelines=pipe, output_map=["tgt", "src", "is_same"], auto_reset=True, last_batch_policy=LastBatchPolicy.DROP
        )

        if cfg.device.index == 0:
            self.writer = SummaryWriter(log_dir=self.tb_log_dir)

        self.scaler_d = GradScaler()
        self.scaler_g = GradScaler()

        if checkpoint is not None:
            if "scaler_d" in checkpoint:
                self.scaler_d.load_state_dict(checkpoint["scaler_d"])
                print("Loaded scaler_d state from checkpoint.")

            if "scaler_g" in checkpoint:
                self.scaler_g.load_state_dict(checkpoint["scaler_g"])
                print("Loaded scaler_g state from checkpoint.")

        self.face_pras = FaceParsing(device=cfg.device)

    def model_ema(self, decay=0.999):

        net_g_params = self.net_g.module if cfg.is_distributed else self.net_g
        net_g_params = dict(net_g_params.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def save(self):

        if self.cfg.is_distributed:
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
            "scaler_d": self.scaler_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict(),
        }

        save_file = os.path.join(self.ckpt_dir, f"{str(self.iter)}_{self.cfg.size}.pth")

        torch.save(save_dict, save_file)

    def fetch_training_batch(self):

        data = self.loader.next()[0]

        target, source, same = data["tgt"], data["src"], data["is_same"]
        mask = self.face_pras.process(target)
        return target, source, mask, same.squeeze()

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
            pbar = tqdm(itertools.count(start=self.iter), initial=self.iter, unit="iter")
        else:
            pbar = itertools.count(start=self.iter)

        mean_path_length = 0
        torch.cuda.empty_cache()
        # 主训练循环
        for self.iter in pbar:

            # =============== 训练判别器 ===============
            self.net_d.requires_grad_(True)
            self.net_g.requires_grad_(False)
            self.d_optimizer.zero_grad()

            with autocast(device_type=DEVICE_TYPE, dtype=torch.float16):

                for accum_step in range(cfg.grad_accum_steps):

                    target, source, mask, same = self.fetch_training_batch()
                    is_r1_reg_every = (self.iter % cfg.r1_reg_every == 0) and (accum_step == cfg.grad_accum_steps - 1)

                    with torch.no_grad():
                        z_id = self.arcface(source)
                        fake_img, _, _ = self.net_g(target, z_id)

                    fake_pred = self.net_d(fake_img)

                    if is_r1_reg_every:
                        target.requires_grad_(True)

                    real_pred = self.net_d(target)

                    d_loss = self.gan_loss(real_pred, True, is_disc=True) + self.gan_loss(fake_pred, False, is_disc=True)

                    self.loss_dict["D"] = d_loss.detach()
                    self.loss_dict["real_score"] = real_pred.detach().mean()
                    self.loss_dict["fake_score"] = fake_pred.detach().mean()

                    d_loss = d_loss / cfg.grad_accum_steps

                    if is_r1_reg_every:
                        r1 = r1_penalty(real_pred, target)
                        r1 = cfg.r1_reg_weight / 2 * r1 * cfg.r1_reg_every
                        d_loss += r1
                        self.loss_dict["r1"] = r1.detach().mean()

                    self.scaler_d.scale(d_loss).backward()

            self.scaler_d.step(self.d_optimizer)
            self.scaler_d.update()

            # =============== 训练生成器 ===============
            self.net_d.requires_grad_(False)
            self.net_g.requires_grad_(True)
            self.g_optimizer.zero_grad()

            with autocast(device_type=DEVICE_TYPE, dtype=torch.float16):

                for accum_step in range(cfg.grad_accum_steps):

                    target, source, mask, same = self.fetch_training_batch()
                    with torch.no_grad():
                        z_id = self.arcface(source)

                    fake_img, latents, fake_mask = self.net_g(target, z_id)

                    # ------------- 对抗损失 -------------
                    fake_pred = self.net_d(fake_img)
                    g_loss = self.gan_loss(fake_pred, True, is_disc=False)
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

                    # ------------- FACS -------------
                    if self.iter > 500000:
                        au_loss = self.au_loss(fake_img, target)
                        g_loss += au_loss
                        self.loss_dict["AU"] = au_loss.detach()

                    # ------------- 启发式误差识别细化(FaceShifter) -------------
                    err_per_sample = torch.abs(fake_img - target).mean(dim=(1, 2, 3))
                    numerator = torch.dot(err_per_sample, same)
                    denominator = same.sum().clamp_min(1e-8)  # 防止除零
                    rec_pix = numerator / denominator
                    g_loss += rec_pix * cfg.pixel_rec_weight
                    self.loss_dict["Pixel_Rec"] = rec_pix

                    # ------------- 掩码损失 -------------
                    with autocast(device_type=DEVICE_TYPE, enabled=False):
                        mask_loss = F.binary_cross_entropy(fake_mask, mask).mean()
                        self.loss_dict["Mask"] = mask_loss.detach()
                        g_loss += mask_loss * cfg.mask_loss_weight

                    # -----------------------------------
                    g_loss = g_loss / cfg.grad_accum_steps

                    # 路径正则化
                    if (self.iter % cfg.path_reg_every == 0) and (accum_step == cfg.grad_accum_steps - 1):
                        path_loss, mean_path_length, path_lengths = g_path_regularize(fake_img, latents, mean_path_length)
                        weighted_path_loss = cfg.path_reg_weight * cfg.path_reg_every * path_loss
                        g_loss += weighted_path_loss
                        self.loss_dict["path_loss"] = path_loss
                        self.loss_dict["path_lengths"] = path_lengths.mean()
                        self.loss_dict["mean_path_length"] = mean_path_length.mean()

                    self.scaler_g.scale(g_loss).backward()

            self.scaler_g.step(self.g_optimizer)
            self.scaler_g.update()

            if cfg.device.index == 0:
                self.model_ema()

            if self.iter % 10 == 0:
                self.reduce_loss_dict()

            if self.iter % 1000 == 0:
                torch.cuda.empty_cache()

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
                        save_file = os.path.join(self.sample_dir, f"{str(self.iter)}_{cfg.size}.png")

                        # 保存样本图像
                        utils.save_image(grid, save_file, nrow=cfg.batch, normalize=True, value_range=(-1, 1))

                # 定期保存模型
                if self.iter % cfg.save_ckpt_every == 0:
                    self.save()


if __name__ == "__main__":

    cfg = TrainSet()
    cfg.batch = 20
    cfg.pretrain = "log/train/ckpt/238693_512.pth"

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"分布式训练: world_size={world_size}, local_rank={local_rank}")
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size)
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
