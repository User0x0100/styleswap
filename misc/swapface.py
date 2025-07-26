import os
import math
from itertools import batched
from typing import Generator, List
import torch
from torch import autocast
from torch import nn
import torchvision.utils as utils
from torchvision.io import decode_image, ImageReadMode
from torchcodec.decoders import VideoDecoder
from face_swap_model import FullGenerator
from .utils import DataFolder
from .model import ArcFace
from .tools import FaceAlign, tensor2cv_8uc3_bgr
from misc.model import FaceParsing
from torch.nn import functional as F
from pathlib import Path
import cv2
import multiprocessing as mp

import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2
from typing import Dict, List, Tuple, Generator
import json

from tqdm import tqdm


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
            return KeyError
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

    @autocast(device_type="cuda")
    def swap_folder(self, fp: str, source: str | None = None, wfp: str | None = None, sf_prefix: str = "", batch_size: int = 32):

        device = self.align_dst.device
        data_folder = DataFolder(fp)

        if source is None:
            id_emb = torch.randn((1, 512), dtype=torch.float, device=device)
            id_emb = F.normalize(id_emb, p=2, dim=1)
        else:
            id_emb = self.get_fae_id(source)[0:1]

        if wfp is None:
            wfp = Path(fp).joinpath("swap_results/")

        Path(wfp).mkdir(exist_ok=True, parents=True)

        similarity_collect = []
        for images, fns in data_folder.iter_batch_with_tensor(batch_size, True):

            images = [image.float().to(device=device).div_(255.0).mul_(2.0).sub_(1.0) for image in images]
            aligned_faces, norm_affine_matrix, idxs = self.aligner.align(images, (self.size,) * 2, self.align_dst)

            swaped_face = self.swap_single(aligned_faces, id_emb[:1])

            for fake, name in zip(swaped_face, fns):
                fake_id = self.arcface(fake.unsqueeze(0))
                similarity = F.cosine_similarity(fake_id, id_emb.expand(fake_id.shape[0], -1))
                print(f"相似度: {similarity.mean():.4f} --- {name}")
                similarity_collect.append(similarity)

            restored = self.aligner.restore_faces_to_original(swaped_face, norm_affine_matrix, images, idxs)

            for swap, fn, org in zip(restored, fns, images):
                name = Path(fn).stem
                sfp = Path(wfp) / f"{sf_prefix}{name}_swaped.png"

                grid = torch.cat((swap, org), dim=2)
                grid = tensor2cv_8uc3_bgr(grid, (-1, 1))

                cv2.imwrite(sfp, grid.squeeze(0), [cv2.IMWRITE_PNG_COMPRESSION, 5])

        mean_tensor = torch.stack(similarity_collect, dim=0).mean(dim=0)
        print(f"平均相似度为: {mean_tensor.item():.4f}")

    @staticmethod
    def display_worker(q: mp.Queue, fps=25, value_range=(-1, 1)):

        delay = int(1000 / fps)

        while True:
            frames = q.get()
            if frames is None:
                break
            frames = tensor2cv_8uc3_bgr(frames, value_range)
            for frame in frames:
                cv2.imshow("swaped", frame)
                cv2.waitKey(delay)

        cv2.destroyAllWindows()
        q.close()

    @autocast(device_type="cuda")
    def swap_video(self, vfp: str, source: str, sfp: str | None = None, batch_size: int = 32):

        if sfp is None:
            fn = Path(vfp).stem
            folder = Path(vfp).parent
            sfp = Path(folder) / f"{fn}_swaped.mp4"

        mp.set_start_method("spawn", force=True)
        q = mp.Queue(maxsize=15)
        p = mp.Process(target=self.display_worker, args=(q,))
        p.start()

        id_emb = self.get_fae_id(source)[0:1]
        vr = VideoDecoder(vfp, device="cuda")
        num_frames = vr.metadata.num_frames_from_content
        device = self.align_dst.device

        pbar = tqdm(range(vr.metadata.num_frames_from_content), unit="frame")

        slices = [(i, min(i + batch_size, num_frames)) for i in range(0, num_frames, batch_size)]

        try:
            for slic in slices:
                pbar.n = slic[1]
                pbar.refresh()

                chunk = vr.get_frames_in_range(*slic).data.to(device=device, dtype=torch.float).div_(255.0).mul_(2.0).sub_(1.0)

                aligned, norm_affine_matrix, idxs = self.aligner.align(list(chunk.unbind(dim=0)), (self.size,) * 2, self.align_dst)

                swaped_face = self.swap_single(aligned, id_emb[:1])
                fake_id = self.arcface(swaped_face)
                similarity = F.cosine_similarity(fake_id, id_emb.expand(fake_id.shape[0], -1))

                pbar.set_description_str(f"相似度: {similarity.mean():.4f}")
                q.put(swaped_face)

        except KeyboardInterrupt:
            q.put(None)
            return

        q.put(None)


if __name__ == "__main__":
    from misc.utils import DataFolder
    from loguru import logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{device=}")
    dataset = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024")
    # dataset = DataFolder("IDAssets")
    logger.info(f"{dataset=}")

    ckpt = "log/train_4/ckpt/150000_512.pth"
    logger.info(f"{ckpt=}")

    # target = "/opt/share/deepfake/linshi/录屏素材/2023-04-14(李云帆 陈雨)/20230414-163616614陈雨(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-02-17(袁忠 曹属馨)/20230217-160724910曹属馨(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-06-20(邓叮玲 许馨文)/20230620-155807793邓叮玲(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-08-28(李小姐 邱小姐)/20230828-122302李小姐(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2025-07-03(蒋女士 林嘉惠 王钰 符媛媛 石星儿)/王钰(电脑录屏).mp4"
    target = "IDAssets/抖音-记录美好生活 (1).mp4"

    # source = dataset.sample(rando=True)
    source = "IDAssets/jay_c.webp"
    logger.info(f"{source=}")
    # /opt/share/deepfake/stable-diffusion/face_editor_test/高清数据/Twilight/twilight/全/1_A woman
    swap = Swapper(ckpt).to(device=device)
    swap.swap_folder("/opt/share/deepfake/stable-diffusion/face_editor_test/高清数据/Twilight/twilight/全/1_A woman", wfp="woman")
    # swap.swap_video(target, source)
