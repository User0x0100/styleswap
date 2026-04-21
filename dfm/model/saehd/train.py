import yaml
import random
import argparse
import itertools
from pathlib import Path

import torch
from torch import Tensor, autocast, optim, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms.v2.functional import gaussian_blur
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

import cv2
from tqdm import tqdm

from dfm.model import EPS
from dfm.model.saehd.arch import ARCH_SCHEMAS, DFUDT, LIAEUDT
from dfm.model.discriminator import UNetDiscriminatorSN
from dfm.dataloader import datasetloader
from dfm.losses import DLoss, GANLoss, MaskLoss, PixelLoss, StyleLossLabChroma, DSSIMLoss
from losses.id_loss import IDLoss
from misc.model import FaceParsing

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class TrainMask:
    def __init__(
        self,
        image: list[tuple[str, float]],
        weight: str,
        mask_loss_weight: float = 10.0,
        blur_mask: bool = True,
        arch_type: str = "LIAE",
        lr: float = 5e-5,
        batch_size: int = 8,
        device: str = "cuda:0",
        bf16: bool = True,
        compile_model: bool = True,
        log_path: str = "log_dfm/train_mask",
        sample_save_every: int = 1000,
        weight_save_every: int = 10000,
    ) -> None:

        match arch_type:
            case "LIAE" | "DF":
                self.arch_type = arch_type
            case _:
                raise TypeError(f"Unsupported model architecture: {arch_type}")

        self.arch_schema = ARCH_SCHEMAS[self.arch_type]
        self.blur_mask = blur_mask

        self.device = torch.device(device=device)
        self.batch_size = batch_size
        self.enable_bf16 = bf16
        self.compile_model = compile_model

        self.sample_save_every = sample_save_every
        self.weight_save_every = weight_save_every

        if Path(weight).exists() == False:
            raise FileNotFoundError(f"CheckPoint file: {weight} Not found")

        print(f"Loading weight from {weight}")
        weight: dict = torch.load(weight, map_location=torch.device("cpu"), weights_only=False)
        required_common = {"iter", "resolution", "e_dims", "ae_dims", "d_dims", "d_mask_dims", "discriminator"}
        required = required_common | set(self.arch_schema["modules"])
        missing = required - weight.keys()
        if missing:
            raise KeyError(f"Weight missing keys: {sorted(missing)}")

        self.iter = weight["iter"]
        resolution = weight["resolution"]
        e_dims = weight["e_dims"]
        ae_dims = weight["ae_dims"]
        d_dims = weight["d_dims"]
        d_mask_dims = weight["d_mask_dims"]
        self.discriminator_state_dict = weight["discriminator"]

        self.train_module: dict[str, nn.Module] = {}

        print(
            f"Weight Info:\n"
            f"  {'iter':25}: {self.iter}\n"
            f"  {'resolution':25}: {resolution}\n"
            f"  {'e_dims':25}: {e_dims}\n"
            f"  {'ae_dims':25}: {ae_dims}\n"
            f"  {'d_dims':25}: {d_dims}\n"
            f"  {'d_mask_dims':25}: {d_mask_dims}"
        )

        self.model: nn.Module | LIAEUDT | DFUDT = self.arch_schema["model_cls"](resolution=resolution, e_dims=e_dims, ae_dims=ae_dims, d_dims=d_dims, d_mask_dims=d_mask_dims)

        for module_name, module in self.model.named_children():
            print(f"Load {module_name:15} from weight")
            module.load_state_dict(weight[module_name])

        self.model.to(device=self.device)

        if self.enable_bf16:
            self.model = self.model.to(dtype=torch.bfloat16)

        self.model.eval()
        self.model.requires_grad_(False)

        for module_name in self.arch_schema["modules"]:
            self.train_module[module_name] = getattr(self.model, module_name)

        decoder = self.model.decoder if self.arch_type == "LIAE" else self.model.decoder_src

        mask_modules: list[nn.Module] = [decoder.upscalem0, decoder.upscalem1, decoder.upscalem2, decoder.upscalem3, decoder.upscalem4, decoder.out_convm]

        for m in mask_modules:
            m.train()
            m.requires_grad_(True)

        params = itertools.chain.from_iterable(m.parameters() for m in mask_modules)
        self.optim = optim.Adam(params, lr=lr)

        pipe = datasetloader(
            batch_size=batch_size,
            num_threads=4,
            prefetch_queue_depth=40,
            py_num_workers=2,
            py_start_method="spawn",
            device_id=self.device.index,
            resize=self.model.resolution,
            src=None,
            dst=image,
            src_random_warp=0.0,
            dst_random_warp=0.0,
            random_sampling=True,
        )

        self.dataset = DALIGenericIterator(pipelines=pipe, output_map=["image"], auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)

        self.face_segment = FaceParsing.Model(range_norm=False, occ=True)
        self.face_segment.to(device=self.device)
        self.face_segment.eval()
        self.face_segment.requires_grad_(False)

        base_log_path = Path(log_path)
        self.ckpt_dir = base_log_path.joinpath("ckpt")
        self.sample_dir = base_log_path.joinpath("sample")
        self.tb_log_dir = base_log_path.joinpath("tb_log")

        for p in [self.ckpt_dir, self.sample_dir, self.tb_log_dir]:
            p.mkdir(exist_ok=True, parents=True)

        self.log_writer = SummaryWriter(log_dir=self.tb_log_dir, max_queue=100)

        self.mask_loss = MaskLoss(weight=mask_loss_weight, reduction="mean").to(device=self.device)

        # if self.compile_model:
        #     for name in self.train_module:
        #         compile_module = torch.compile(self.train_module[name], mode="max-autotune")
        #         self.train_module[name] = compile_module

    def log(self, k: str, v: Tensor, now: bool = False) -> None:
        if now or self.iter % 10 == 0:
            self.log_writer.add_scalar(f"Loss/{k}", v.detach().mean().item(), self.iter)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        return cls(**cfg)

    @torch.no_grad()
    def fetch_sample(self) -> tuple[Tensor, Tensor]:

        data = self.dataset.next()[0]

        image = data["image"]
        mask = self.face_segment(image)

        return image, mask

    def save_state_dict(self):

        state_dict = {
            "iter": self.iter,
            "arch_type": self.arch_type,
            "resolution": self.model.resolution,
            "e_dims": self.model.e_dims,
            "ae_dims": self.model.ae_dims,
            "d_dims": self.model.d_dims,
            "d_mask_dims": self.model.d_mask_dims,
        }

        for name in self.arch_schema["modules"]:
            state_dict[name] = getattr(self.model, name).state_dict()

        state_dict["discriminator"] = self.discriminator_state_dict

        try:
            ckpt_file = self.ckpt_dir / f"{self.iter}.pth"
            torch.save(state_dict, ckpt_file)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def train(self):
        count = itertools.count(start=self.iter)
        pbar = tqdm(count, initial=self.iter, mininterval=1.0, bar_format="{n_fmt:7} | speed {rate_fmt:3} | train time {elapsed}")

        match self.arch_type:
            case "LIAE":
                encoder, _, inter_AB, decoder = (self.train_module[k] for k in ("encoder", "inter_B", "inter_AB", "decoder"))
            case "DF":
                encoder, inter, decoder, _ = (self.train_module[k] for k in ("encoder", "inter", "decoder_src", "decoder_dst"))

        gaussian_blur_sigma = self.model.resolution / 128
        gaussian_blur_kernel_size = max(3, self.model.resolution // 32)
        if gaussian_blur_kernel_size % 2 == 0:
            gaussian_blur_kernel_size += 1

        for self.iter in pbar:

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.enable_bf16):
                image, mask = self.fetch_sample()

                if self.blur_mask:
                    with torch.no_grad():
                        mask = gaussian_blur(inpt=mask, kernel_size=gaussian_blur_kernel_size, sigma=gaussian_blur_sigma).clamp(0.0, 0.5).mul(2.0)

                self.optim.zero_grad()

                with torch.no_grad():
                    image_code = encoder(image)
                    match self.arch_type:
                        case "LIAE":
                            image_inter = inter_AB(image_code)
                            image_inter = torch.cat((image_inter, image_inter), dim=1)
                        case "DF":
                            image_inter = inter(image_code)

                pred_image, pred_mask = decoder(image_inter)

                mask_loss: Tensor = self.mask_loss(pred_mask, mask)
                self.log("mask_loss", mask_loss)
                mask_loss.backward()
                self.optim.step()

                if self.iter % self.weight_save_every == 0:
                    self.save_state_dict()

                if self.iter % self.sample_save_every == 0:
                    mask = mask.expand(-1, 3, -1, -1)
                    pred_mask = pred_mask.expand(-1, 3, -1, -1)
                    grid = torch.cat((image, pred_image, mask, pred_mask), dim=0)
                    grid = make_grid(grid, nrow=self.batch_size)[[2, 1, 0], :, :]  # RGB -> BGR
                    grid_cpu = grid.mul_(255.0).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()  # CHW -> HWC
                    cv2.imwrite(self.sample_dir / f"{self.iter}.png", grid_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])


ANTI_MASK_WEIGHT = 0.0001


class Trainer:
    def __init__(
        self,
        src: list[tuple[str, float]] | None,
        dst: list[tuple[str, float]],
        set_iter: int | None = None,
        lr: float = 5e-5,
        lr_scheduler_t_max: int = 0,
        batch_size: int = 8,
        device: str = "cuda:0",
        bf16: bool = True,
        compile_model: bool = True,
        log_path: str = "log_dfm/train",
        sample_save_every: int = 1000,
        weight_save_every: int = 10000,
        pretrain: bool = False,
        dssim_loss_weight: float = 1.0,
        pixel_loss_weight: float = 10.0,
        mask_loss_weight: float = 10.0,
        weight: str | None = None,
        weight_load_selection: dict[str, bool] = {},
        seg_mask_with_occ: bool = True,
        masked_train: bool = True,
        blur_mask: bool = True,
        src_random_warp_prob: float = 0.3,
        dst_random_warp_prob: float = 0.3,
        gan_power: float = 0.0,
        gan_update_every: int = 5,
        face_style_power: float = 0.0,
        face_id: tuple[str | None, float] = (None, 0.0),
        arch_type: str = "LIAE",
        resolution: int = 320,
        e_dims: int = 64,
        ae_dims: int = 512,
        d_dims: int = 64,
        d_mask_dims: int = 22,
    ) -> None:

        print(
            f"Trainer Parameters:\n"
            f"  {'set_iter':25}: {set_iter}\n"
            f"  {'lr':25}: {lr}\n"
            f"  {'batch_size':25}: {batch_size}\n"
            f"  {'device':25}: {device}\n"
            f"  {'enable_bf16':25}: {bf16}\n"
            f"  {'compile_model':25}: {compile_model}\n"
            f"  {'log_path':25}: {log_path}\n"
            f"  {'sample_save_every':25}: {sample_save_every}\n"
            f"  {'weight_save_every':25}: {weight_save_every}\n"
            f"  {'pretrain':25}: {pretrain}\n"
            f"  {'dssim_loss_weight':25}: {dssim_loss_weight}\n"
            f"  {'pixel_loss_weight':25}: {pixel_loss_weight}\n"
            f"  {'mask_loss_weight':25}: {mask_loss_weight}\n"
            f"  {'weight':25}: {weight}\n"
            f"  {'weight_load_selection':25}: {weight_load_selection}\n"
            f"  {'masked_train':25}: {masked_train}\n"
            f"  {'blur_mask':25}: {blur_mask}\n"
            f"  {'src_random_warp_prob':25}: {src_random_warp_prob}\n"
            f"  {'dst_random_warp_prob':25}: {dst_random_warp_prob}\n"
            f"  {'gan_power':25}: {gan_power}\n"
            f"  {'face_style_power':25}: {face_style_power}\n"
            f"  {'face_id':25}: {face_id}\n"
            f"  {'arch_type':25}: {arch_type}\n"
            f"  {'resolution':25}: {resolution}\n"
            f"  {'e_dims':25}: {e_dims}\n"
            f"  {'ae_dims':25}: {ae_dims}\n"
            f"  {'d_dims':25}: {d_dims}\n"
            f"  {'d_mask_dims':25}: {d_mask_dims}"
        )

        if arch_type == "DF" and pretrain:
            raise KeyError("The 'DF' architecture does not have a pre-training mode.")

        match arch_type:
            case "LIAE" | "DF":
                self.arch_type = arch_type
            case _:
                raise TypeError(f"Unsupported model architecture: {arch_type}")

        self.arch_schema = ARCH_SCHEMAS[self.arch_type]

        self.device = torch.device(device=device)
        self.batch_size = batch_size
        self.enable_bf16 = bf16
        self.compile_model = compile_model

        self.sample_save_every = sample_save_every
        self.weight_save_every = weight_save_every

        self.masked_train = masked_train
        self.blur_mask = blur_mask
        self.pretrain = pretrain

        self.enable_src_random_warp = src_random_warp_prob > 0.0
        self.enable_face_style = face_style_power > 0.0
        self.enable_gan = gan_power > 0.0
        self.gan_update_every = gan_update_every
        self.enable_id_loss = face_id[0] is not None and face_id[1] > 0.0

        self.discriminator = UNetDiscriminatorSN()

        self.train_module: dict[str, nn.Module] = {}
        self.optimiser: dict[str, optim.Adam] = {}
        self.schedulers: dict[str, CosineAnnealingLR] = {}

        if self.pretrain:
            print("Pretrain mode: disable random_warp, face_style, gan, id_loss")
            self.enable_src_random_warp = False
            self.enable_face_style = False
            self.enable_gan = False
            self.enable_id_loss = False

        # ========================= Init Model =========================

        if weight is None:
            self.iter = 1
            self.model: nn.Module | LIAEUDT | DFUDT = self.arch_schema["model_cls"](resolution=resolution, e_dims=e_dims, ae_dims=ae_dims, d_dims=d_dims, d_mask_dims=d_mask_dims)
        else:
            if Path(weight).exists() == False:
                raise FileNotFoundError(f"CheckPoint file: {weight} Not found")

            print(f"Loading weight from {weight}")
            weight: dict = torch.load(weight, map_location=torch.device("cpu"), weights_only=False)
            required_common = {"iter", "resolution", "e_dims", "ae_dims", "d_dims", "d_mask_dims", "discriminator"}
            required = required_common | set(self.arch_schema["modules"])
            missing = required - weight.keys()
            if missing:
                raise KeyError(f"Weight missing keys: {sorted(missing)}")

            self.iter = weight["iter"]
            resolution = weight["resolution"]
            e_dims = weight["e_dims"]
            ae_dims = weight["ae_dims"]
            d_dims = weight["d_dims"]
            d_mask_dims = weight["d_mask_dims"]

            print(
                f"Weight Info:\n"
                f"  {'iter':25}: {self.iter}\n"
                f"  {'resolution':25}: {resolution}\n"
                f"  {'e_dims':25}: {e_dims}\n"
                f"  {'ae_dims':25}: {ae_dims}\n"
                f"  {'d_dims':25}: {d_dims}\n"
                f"  {'d_mask_dims':25}: {d_mask_dims}"
            )

            self.model = self.arch_schema["model_cls"](resolution=resolution, e_dims=e_dims, ae_dims=ae_dims, d_dims=d_dims, d_mask_dims=d_mask_dims)

            allowed_keys = set(dict(self.model.named_children())) | {"discriminator"}
            invalid = set(weight_load_selection) - allowed_keys
            if invalid:
                raise KeyError(f"Invalid weight_load_selection keys: {invalid}")

            for module_name, module in self.model.named_children():
                if not weight_load_selection.get(module_name, True):
                    print(f"Skip the loading of the weights of the {module_name} module")
                else:
                    print(f"Load {module_name:15} from weight")
                    module.load_state_dict(weight[module_name])

            if not weight_load_selection.get("discriminator", True):
                print(f"Skip the loading of the weights of the discriminator module")
            else:
                print(f"load {'discriminator':15} from weight")
                self.discriminator.load_state_dict(weight["discriminator"])

        self.model.to(device=self.device)
        self.model.train()

        self.discriminator.to(device=self.device)
        self.discriminator.train()

        if set_iter is not None:
            self.iter = set_iter

        if self.enable_src_random_warp == False and self.arch_type == "LIAE":
            self.model.inter_AB.requires_grad_(False)

        # ========================= Set Train Model and Optim =========================

        for module_name in self.arch_schema["modules"]:
            self.train_module[module_name] = getattr(self.model, module_name)

        self.train_module["encoder"] = self.model.encoder

        # -------------------------- optim --------------------------
        for name, module in self.train_module.items():
            self.optimiser[name] = optim.Adam(module.parameters(), lr=lr, fused=True)

        # -------------------------- schedulers --------------------------
        if lr_scheduler_t_max > 0:
            for module_name in self.arch_schema["modules"]:
                self.schedulers[module_name] = CosineAnnealingLR(self.optimiser[module_name], T_max=lr_scheduler_t_max, eta_min=lr * 0.1)

        # -------------------------- discriminator --------------------------
        if self.enable_gan:
            self.train_module["discriminator"] = self.discriminator
            self.optimiser["discriminator"] = optim.Adam(self.train_module["discriminator"].parameters(), lr=lr * 0.45, fused=True)

        # ========================= LOSS =========================

        self.adv_loss = DLoss(weight=1.0, reduction="mean").to(device=self.device) if self.enable_gan else None

        reduction = "none" if self.masked_train else "mean"

        self.gan_loss = GANLoss(weight=gan_power, reduction=reduction).to(device=self.device) if self.enable_gan else None
        self.pixel_loss = PixelLoss(weight=pixel_loss_weight, reduction=reduction).to(device=self.device)
        self.dssim_loss = DSSIMLoss(weight=dssim_loss_weight, reduction=reduction, window_size=7).to(device=self.device)

        self.mask_loss = MaskLoss(weight=mask_loss_weight, reduction="mean").to(device=self.device)
        self.style_loss = StyleLossLabChroma(weight=face_style_power).to(device=self.device) if self.enable_face_style else None

        # ========================= ID LOSS =========================
        if self.enable_id_loss:
            from torchvision.io import decode_image
            from torchvision.io.image import ImageReadMode
            from misc.tools import FaceAlign
            from misc.model import IDEncoder
            from misc.model.idencoder.main import PROVIDER

            face_path: str = face_id[0]
            id_loss_weight = face_id[1]

            if Path(face_path).exists() == False:
                raise FileNotFoundError(f"ID Photo File: {face_path} Not found")

            face = decode_image(face_path, ImageReadMode.RGB).to(device=self.device, dtype=torch.float).div_(127.5).sub_(1.0).unsqueeze(0)  # f32 1CHW [-1.0 ~ 1.0]
            align = FaceAlign(from_unit_range=True).to(self.device)
            aligned_face, _, _ = align.align(face, (self.model.resolution,) * 2, IDEncoder.get_dst_pts(self.model.resolution).to(self.device))

            self.id_loss = IDLoss(weight=id_loss_weight, id_encode_provider=PROVIDER.BLENDFACE).to(self.device)

            with torch.no_grad():
                tid = self.id_loss.get_id_feats(aligned_face, realign=False)[0:1].expand(batch_size, -1)
                self.target_id = tid.detach().clone()

        # ========================= Sample =========================
        pipe = datasetloader(
            batch_size=batch_size,
            num_threads=4,
            prefetch_queue_depth=25,
            py_num_workers=2,
            py_start_method="spawn",
            device_id=self.device.index,
            resize=self.model.resolution,
            src=src,
            dst=dst,
            src_random_warp=src_random_warp_prob,
            dst_random_warp=dst_random_warp_prob,
            random_sampling=True,
        )
        if self.pretrain:
            self.dataset = DALIGenericIterator(pipelines=pipe, output_map=["dst"], auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)
        else:
            self.dataset = DALIGenericIterator(pipelines=pipe, output_map=["src", "dst"], auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)

        # ========================= Utils =========================
        self.face_segment = FaceParsing.Model(range_norm=False, occ=seg_mask_with_occ)
        self.face_segment.to(device=self.device)
        self.face_segment.eval()
        self.face_segment.requires_grad_(False)

        # ========================= LOG =========================
        base_log_path = Path(log_path)
        self.ckpt_dir = base_log_path.joinpath("ckpt")
        self.sample_dir = base_log_path.joinpath("sample")
        self.tb_log_dir = base_log_path.joinpath("tb_log")

        for p in [self.ckpt_dir, self.sample_dir, self.tb_log_dir]:
            p.mkdir(exist_ok=True, parents=True)

        self.log_writer = SummaryWriter(log_dir=self.tb_log_dir, max_queue=100)

        # ========================= Compile Model =========================
        if self.compile_model:
            for name in self.train_module:
                mode = "max-autotune-no-cudagraphs" if name == "discriminator" else "max-autotune"
                compile_module = torch.compile(self.train_module[name], mode=mode)
                self.train_module[name] = compile_module

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        return cls(**cfg)

    @torch.no_grad()
    def fetch_sample(self) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:

        data = self.dataset.next()[0]

        dst = data["dst"]
        dst_mask = self.face_segment(dst)

        if self.pretrain:
            return None, None, dst, dst_mask
        else:
            src = data["src"]
            src_mask = self.face_segment(src)
            return src, src_mask, dst, dst_mask

    def save_state_dict(self):

        state_dict = {
            "iter": self.iter,
            "arch_type": self.arch_type,
            "resolution": self.model.resolution,
            "e_dims": self.model.e_dims,
            "ae_dims": self.model.ae_dims,
            "d_dims": self.model.d_dims,
            "d_mask_dims": self.model.d_mask_dims,
        }

        for name in self.arch_schema["modules"]:
            state_dict[name] = getattr(self.model, name).state_dict()

        state_dict["discriminator"] = self.discriminator.state_dict()

        try:
            ckpt_file = self.ckpt_dir / f"{self.iter}.pth"
            torch.save(state_dict, ckpt_file)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def log(self, k: str, v: Tensor, now: bool = False) -> None:

        if now or self.iter % 10 == 0:
            self.log_writer.add_scalar(f"Loss/{k}", v.detach().mean().item(), self.iter)

    def train_gan(self, fake: Tensor, real: Tensor):

        discriminator = self.train_module["discriminator"]
        d_optim = self.optimiser["discriminator"]
        d_optim.zero_grad()

        fake_score = discriminator(fake.detach())
        real_score = discriminator(real.detach())

        adv_loss: Tensor = self.adv_loss(fake_score, real_score)

        adv_loss.mean().backward()
        d_optim.step()
        self.log("adv_loss", adv_loss, now=True)

    def do_gan(self, pred: Tensor, mask: Tensor, tag: str) -> Tensor:

        fake_score = self.train_module["discriminator"](pred)
        gan_loss = self.gan_loss(fake_score)

        if self.masked_train:
            gan_loss = self.mean_loss_with_mask(gan_loss, mask)

        self.log(f"{tag}_gan_loss", gan_loss)
        return gan_loss

    def mean_loss_with_mask(self, loss: Tensor, mask: Tensor) -> Tensor:
        loss *= mask
        return loss.sum() / (mask.sum() + EPS)

    def base_loss(self, pred: Tensor, target: Tensor, pred_mask, target_mask: Tensor, lab: str) -> Tensor:

        dssim_loss = self.dssim_loss(pred, target)

        pixel_loss = self.pixel_loss(pred, target)

        mask_loss = self.mask_loss(pred_mask, target_mask)

        if self.masked_train:

            with torch.no_grad():
                anti_mask = (1.0 - target_mask) * ANTI_MASK_WEIGHT

            dssim_loss = self.mean_loss_with_mask(dssim_loss, target_mask)

            anti_rgb_loss = pixel_loss * anti_mask
            pixel_loss *= target_mask
            pixel_loss += anti_rgb_loss
            pixel_loss = pixel_loss.mean()

        self.log(f"{lab}_dssim_loss", dssim_loss)
        self.log(f"{lab}_pixel_loss", pixel_loss)
        self.log(f"{lab}_mask_loss", mask_loss)

        loss = dssim_loss + pixel_loss + mask_loss

        return loss

    def train(self):
        count = itertools.count(start=self.iter)
        pbar = tqdm(count, initial=self.iter, mininterval=1.0, bar_format="{n_fmt:7} | speed {rate_fmt:3} | train time {elapsed}")

        match self.arch_type:
            case "LIAE":
                encoder, inter_B, inter_AB, decoder = (self.train_module[k] for k in ("encoder", "inter_B", "inter_AB", "decoder"))
            case "DF":
                encoder, inter, decoder_src, decoder_dst = (self.train_module[k] for k in ("encoder", "inter", "decoder_src", "decoder_dst"))

        gaussian_blur_sigma = self.model.resolution / 128
        gaussian_blur_kernel_size = max(3, self.model.resolution // 32)
        if gaussian_blur_kernel_size % 2 == 0:
            gaussian_blur_kernel_size += 1

        for self.iter in pbar:

            src, src_mask, dst, dst_mask = self.fetch_sample()

            for name in self.arch_schema["modules"]:
                if name == "inter_AB" and not self.enable_src_random_warp:
                    continue
                self.optimiser[name].zero_grad()

            torch.compiler.cudagraph_mark_step_begin()
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.enable_bf16):

                match self.arch_type:
                    case "LIAE":
                        # forward src -> src
                        if not self.pretrain:
                            src_code = encoder(src)
                            src_inter_AB_code = inter_AB(src_code)
                            src_inter_AB_concat = torch.cat((src_inter_AB_code, src_inter_AB_code), dim=1)
                            pred_src, pred_src_mask = decoder(src_inter_AB_concat)

                        # forward dst -> dst
                        dst_code = encoder(dst)
                        dst_inter_B_code = inter_B(dst_code)

                        if self.pretrain:
                            dst_inter_AB_code = torch.rand_like(dst_inter_B_code)
                        else:
                            dst_inter_AB_code = inter_AB(dst_code)

                        dst_inter_B_inter_AB_concat = torch.cat((dst_inter_B_code, dst_inter_AB_code), dim=1)
                        pred_dst, pred_dst_mask = decoder(dst_inter_B_inter_AB_concat)

                        # forward src -> dst
                        if self.enable_face_style or self.enable_id_loss:
                            dst_inter_AB_inter_AB_concat = torch.cat((dst_inter_AB_code, dst_inter_AB_code), dim=1)
                            pred_src_dst_decoder_grad_only, pred_src_dst_mask_decoder_grad_only = decoder(dst_inter_AB_inter_AB_concat.detach())
                    case "DF":
                        # forward src -> src
                        src_code = encoder(src)
                        src_inter_code = inter(src_code)
                        pred_src, pred_src_mask = decoder_src(src_inter_code)

                        # forward dst -> dst
                        dst_code = encoder(dst)
                        dst_inter_code = inter(dst_code)
                        pred_dst, pred_dst_mask = decoder_dst(dst_inter_code)

                        # forward src -> dst
                        if self.enable_face_style or self.enable_id_loss:
                            pred_src_dst_decoder_grad_only, pred_src_dst_mask_decoder_grad_only = decoder_src(dst_inter_code.detach())

                if self.blur_mask:
                    with torch.no_grad():
                        if not self.pretrain:
                            src_mask = gaussian_blur(inpt=src_mask, kernel_size=gaussian_blur_kernel_size, sigma=gaussian_blur_sigma).clamp(0.0, 0.5).mul(2.0)

                        dst_mask = gaussian_blur(inpt=dst_mask, kernel_size=gaussian_blur_kernel_size, sigma=gaussian_blur_sigma).clamp(0.0, 0.5).mul(2.0)

                if self.masked_train:
                    kernel_size = random.randrange(3, 16, 2)
                    sigma = random.uniform(7.1, 29.9)
                    with torch.no_grad():
                        if not self.pretrain:
                            src_blur = gaussian_blur(inpt=src, kernel_size=kernel_size, sigma=sigma)
                            anti_src_mask = 1.0 - src_mask

                            src_blur.mul_(anti_src_mask)
                            src.mul_(src_mask).add_(src_blur)

                        dst_blur = gaussian_blur(inpt=dst, kernel_size=kernel_size, sigma=sigma)
                        anti_dst_mask = 1.0 - dst_mask

                        dst_blur.mul_(anti_dst_mask)
                        dst.mul_(dst_mask).add_(dst_blur)

                # ========================= Train Discriminator =========================
                if self.enable_gan and self.iter % self.gan_update_every == 0:
                    self.train_gan(pred_src, src)

                src_loss = 0.0
                dst_loss = 0.0
                src_dst_loss = 0.0

                # ========================= loss src -> src =========================
                if not self.pretrain:
                    src_loss += self.base_loss(pred_src, src, pred_src_mask, src_mask, "src")

                if self.enable_gan:
                    src_loss += self.do_gan(pred_src, src_mask.detach(), "src")

                # ========================= loss dst -> dst =========================
                dst_loss += self.base_loss(pred_dst, dst, pred_dst_mask, dst_mask, "dst")

                # ========================= loss src -> dst =========================
                if self.enable_face_style and self.iter % 10 == 0:
                    pred_src_dst_decoder_grad_only_masked = pred_src_dst_decoder_grad_only * pred_src_dst_mask_decoder_grad_only.detach()
                    with torch.no_grad():
                        dst_masked = dst.detach() * dst_mask.detach()

                    style_loss = self.style_loss(pred_src_dst_decoder_grad_only_masked, dst_masked)
                    self.log("style_loss", style_loss)
                    src_dst_loss += style_loss

                if self.enable_id_loss:
                    pred_src_dst_decoder_grad_only_norm = pred_src_dst_decoder_grad_only.mul(2.0).sub(1.0)
                    pred_id = self.id_loss.get_id_feats(face=pred_src_dst_decoder_grad_only_norm, face_mask=dst_mask, realign=False)
                    id_loss = self.id_loss(pred_id, self.target_id)
                    self.log("src_dst_id_loss", id_loss)
                    src_dst_loss += id_loss

                # ========================= loss summary =========================

                G_loss = src_loss + dst_loss + src_dst_loss
                G_loss.backward()

                for name in self.arch_schema["modules"]:
                    if name == "inter_AB" and not self.enable_src_random_warp:
                        continue
                    self.optimiser[name].step()

                [scheduler.step() for scheduler in self.schedulers.values()]

            if self.iter % self.weight_save_every == 0:
                self.save_state_dict()

            if self.iter % self.sample_save_every == 0:

                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.enable_bf16):
                    with torch.no_grad():
                        swaped, swap_mamk = self.model(dst)
                        swap_mamk = swap_mamk.expand(-1, 3, -1, -1)

                        dst_mask = dst_mask.expand(-1, 3, -1, -1)

                        pred_dst_mask = pred_dst_mask.expand(-1, 3, -1, -1)

                        if self.pretrain:
                            grid = torch.cat((dst, pred_dst, dst_mask, pred_dst_mask, swaped, swap_mamk), dim=0)
                        else:
                            src_mask = src_mask.expand(-1, 3, -1, -1)
                            pred_src_mask = pred_src_mask.expand(-1, 3, -1, -1)
                            grid = torch.cat((src, pred_src, src_mask, pred_src_mask, dst, pred_dst, dst_mask, pred_dst_mask, swaped, swap_mamk), dim=0)

                        grid = make_grid(grid, nrow=self.batch_size)[[2, 1, 0], :, :]  # RGB -> BGR
                        grid_cpu = grid.mul_(255.0).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()  # CHW -> HWC
                        cv2.imwrite(self.sample_dir / f"{self.iter}.png", grid_cpu, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def parse_args():
    parser = argparse.ArgumentParser(description="Trainer arguments")

    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--train-mask", action="store_true", help="Enable mask-only training (TrainMask)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    cfg = args.cfg

    if args.train_mask:
        trainer = TrainMask.from_yaml(cfg)
    else:
        trainer = Trainer.from_yaml(cfg)

    try:
        trainer.train()

    except KeyboardInterrupt:
        print("\n(Ctrl+C)")
        raise
    finally:
        trainer.log_writer.close()
        trainer.save_state_dict()
