from typing import Any
from torchcodec.decoders import VideoDecoder

import multiprocessing as mp
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch import autocast, nn
from tqdm import tqdm

from dfm.model.saehd.arch import DFUDT, LIAEUDT
from misc.tools import FaceAlign
from misc.model import IDEncoder


class Swapper(nn.Module):

    def __init__(self, ckpt: str):
        super().__init__()

        print(f"Loading checkpoint from {ckpt}")
        ckpt: dict[str, Any] = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)

        for k in ["iter", "resolution", "e_dims", "ae_dims", "d_dims", "d_mask_dims"]:
            if k not in ckpt:
                print(f"ckpt: {ckpt} miss key: {k}")
                exit()

        resolution = ckpt["resolution"]
        e_dims = ckpt["e_dims"]
        ae_dims = ckpt["ae_dims"]
        d_dims = ckpt["d_dims"]
        d_mask_dims = ckpt["d_mask_dims"]
        print(
            f"Weight Info:\n"
            f"  {'resolution':25}: {resolution}\n"
            f"  {'e_dims':25}: {e_dims}\n"
            f"  {'ae_dims':25}: {ae_dims}\n"
            f"  {'d_dims':25}: {d_dims}\n"
            f"  {'d_mask_dims':25}: {d_mask_dims}"
        )

        if "inter" in ckpt:
            self.swap_model = DFUDT(resolution=resolution, e_dims=e_dims, ae_dims=ae_dims, d_dims=d_dims, d_mask_dims=d_mask_dims)
        else:
            self.swap_model = LIAEUDT(resolution=resolution, e_dims=e_dims, ae_dims=ae_dims, d_dims=d_dims, d_mask_dims=d_mask_dims)

        for name, m in self.swap_model.named_children():
            if name in ckpt:
                m.load_state_dict(ckpt[name])
            else:
                raise KeyError(f"The weight of the missing module {name}")

        self.aligner = FaceAlign(from_unit_range=True)

        align_dst = IDEncoder.get_dst_pts(self.swap_model.resolution)

        self.register_buffer("align_dst", align_dst, persistent=False)

        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def display_worker(q: mp.Queue, fps=25):

        delay = int(1000 / fps)

        while True:
            frames = q.get()
            if frames is None:
                break

            # BGR
            frames = frames[:, [2, 1, 0], :, :]
            frames = frames.mul_(255.0).permute(0, 2, 3, 1)
            frames = torch.clamp(frames, 0.0, 255.0).to(device="cpu", dtype=torch.uint8).numpy()

            for frame in frames:
                cv2.imshow("swaped", frame)
                cv2.waitKey(delay)

    # @autocast(device_type="cuda")
    def swap_video(self, vfp: str, batch_size: int = 32):

        device = self.align_dst.device

        mp.set_start_method("spawn", force=True)
        q = mp.Queue(maxsize=15)
        p = mp.Process(target=self.display_worker, args=(q,), daemon=True)
        p.start()

        vr = VideoDecoder(vfp, device="cuda")
        num_frames = vr.metadata.num_frames_from_content
        device = self.align_dst.device

        pbar = tqdm(range(vr.metadata.num_frames_from_content), unit="frame")

        slices = [(i, min(i + batch_size, num_frames)) for i in range(0, num_frames, batch_size)]

        try:
            for slic in slices:
                pbar.n = slic[1]
                pbar.refresh()

                chunk = vr.get_frames_in_range(*slic).data.to(device=device, dtype=torch.float).div_(255.0)

                aligned, norm_affine_matrix, idxs = self.aligner.align(list(chunk.unbind(dim=0)), (self.swap_model.resolution,) * 2, self.align_dst)
                swaped, mask = self.swap_model(aligned)

                swaped_masked = swaped * mask
                inv_mask = 1.0 - mask
                aligned_masked = aligned * inv_mask
                swaped = swaped_masked + aligned_masked

                # aligned = aligned[:, [2, 1, 0], :, :]
                # aligned = aligned.mul_(255.0).permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
                # for i in aligned:

                #     cv2.imshow("swaped", i)
                #     cv2.waitKey(1)
                concat = torch.cat((aligned, swaped), dim=3)
                q.put(concat)

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            exit()

        q.put(None)


if __name__ == "__main__":
    device = torch.device("cuda", 0)
    ckpt = "log_dfm/train_df/ckpt/3480000.pth"
    swap = Swapper(ckpt).to(device=device)
    # dst = "/opt/share/deepfake/linshi/录屏素材/2023-04-14(李云帆 陈雨)/20230414-163616614陈雨(电脑录屏).mp4"
    # dst = "/opt/share/deepfake/linshi/录屏素材/2023-02-17(袁忠 曹属馨)/20230217-160724910曹属馨(电脑录屏).mp4"
    dst = "/opt/share/deepfake/linshi/录屏素材/2023-06-20(邓叮玲 许馨文)/20230620-155807793邓叮玲(电脑录屏).mp4"
    # dst = "/opt/share/deepfake/linshi/录屏素材/2023-08-28(李小姐 邱小姐)/20230828-122302李小姐(电脑录屏).mp4"
    # dst = "/opt/share/deepfake/linshi/录屏素材/2025-07-03(蒋女士 林嘉惠 王钰 符媛媛 石星儿)/王钰(电脑录屏).mp4"
    # dst = "/opt/share/deepfake/dataset_1/oneman/1.mp4"
    swap.swap_video(dst)
