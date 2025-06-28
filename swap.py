import torch
from torch import tensor
import numpy as np
import torch.nn.functional as F
import random
from PIL import Image
from torchvision import transforms, utils
from torchvision.transforms import Normalize
import onnxruntime as ort
import cv2
from tqdm import tqdm

from model import FullGenerator
from losses.arcface import ArcFace

# InsightFace for detection and alignment
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop


class Swapper:
    def __init__(
        self,
        ckpt: str,
        device="cuda:0",
        size=512,
    ):
        self.device = device
        self.size = size

        # Load generator
        self.net_g = FullGenerator(size).to(self.device)
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=True)
        self.net_g.load_state_dict(ckpt_data["g_ema"])
        self.net_g.eval()

        # ArcFace for identity embedding
        self.arcface = ArcFace().to(self.device)
        self.arcface.eval()

        # InsightFace detector + aligner (suppress verbose logs)
        self.detector = FaceAnalysis(
            providers=["CUDAExecutionProvider"],
            allowed_modules=["detection"],
        )
        self.detector.prepare(ctx_id=0, det_thresh=0.2, det_size=(640, 640))

        # Preprocessing pipeline
        self.preproc = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),  # [0,1]
                Normalize([0.5] * 3, [0.5] * 3),  # [-1,1]
            ]
        )

    def _get_align_face_with_file(self, img_path: str):

        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

        return self._get_align_face(img)

    def _get_align_face(self, img):
        # Detect faces
        faces = self.detector.get(img)
        if len(faces) == 0:
            crop_img = Image.fromarray(img)
            tensor = self.preproc(crop_img).unsqueeze(0)
            return tensor

        lm = faces[0].kps
        crop_arr = norm_crop(img, lm, image_size=self.size)

        # Convert back to PIL for transforms
        crop_img = Image.fromarray(crop_arr)
        tensor = self.preproc(crop_img).unsqueeze(0)
        return tensor

    def swap(
        self,
        target_path: str,
        source_path: str,
        save_path: str,
    ):
        # Read and align images
        tgt = self._get_align_face_with_file(target_path).to(self.device)
        src = self._get_align_face_with_file(source_path).to(self.device)

        # Extract identity embedding
        with torch.no_grad():
            source_id = self.arcface(src)

        # Swap faces
        with torch.no_grad():
            fake, _, mask = self.net_g(tgt, source_id)
            fake_id = self.arcface(fake)

        print(f"相似度: {torch.cosine_similarity(source_id,fake_id).mean()}")

        # Composite output: only blend face area
        fake = torch.where(mask.bool(), fake, tgt)

        # Prepare mask for visualization: scale to [-1,1]
        mask_vis = (mask.repeat(1, 3, 1, 1) - 0.5) / 0.5

        # Concatenate: target, source, output, mask
        grid = torch.cat([tgt, src, fake, mask_vis], dim=0)

        utils.save_image(
            grid,
            save_path,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

    def swap_video(
        self,
        target_path: str,
        source_path: str,
        save_path: str,
    ):

        src = self._get_align_face_with_file(source_path).to(self.device)
        with torch.no_grad():
            source_id = self.arcface(src)

        cap = cv2.VideoCapture(target_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # writer = cv2.VideoWriter(save_path, fourcc, fps, (self.size, self.size))

        pbar = tqdm(range(total_frames), unit="frame")

        while cap.isOpened():
            ret, tgt = cap.read()
            if not ret:
                break

            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
            tgt = self._get_align_face(tgt).to(self.device)

            with torch.no_grad():
                fake, _, mask = self.net_g(tgt, source_id)

                fake_id = self.arcface(fake)
                similarity = torch.cosine_similarity(source_id, fake_id).mean()
                fake = fake * mask + (1 - mask) * tgt

            fake = fake.add(1.0).mul(127.5).clamp(0, 255).round().to(torch.uint8)
            fake = fake.permute(0, 2, 3, 1).contiguous()
            fake = fake.squeeze().cpu().numpy()
            fake = cv2.cvtColor(fake, cv2.COLOR_RGB2BGR)

            cv2.imshow("show_swap_res", fake)
            cv2.waitKey(1)

            # writer.write(fake)
            pbar.set_description(f"相似度: {similarity:.3f}")
            pbar.update()

        cap.release()
        # writer.release()


if __name__ == "__main__":
    # target = f"/opt/share/deepfake/stable-diffusion/face_editor_test/高清数据/dfl_big/{random.randint(0,3124):04d}/{random.randint(0,999):04d}.png"
    # source = f"/opt/share/deepfake/stable-diffusion/face_editor_test/高清数据/dfl_big/{random.randint(0,3124):04d}/{random.randint(0,999):04d}.png"

    # target = f"/home/liaohaixun/styleswap_data/dateset/ffhq_1024/{random.randint(0,69999):05d}.png"
    # source = f"/home/liaohaixun/styleswap_data/dateset/ffhq_1024/{random.randint(0,69999):05d}.png"
    # source = target

    # target = f"/home/liaohaixun/styleswap_data/dateset/vggface2_hq512/{random.randint(0,1160249):06d}.jpg"
    # source = f"/home/liaohaixun/styleswap_data/dateset/vggface2_hq512/{random.randint(0,1160249):06d}.jpg"

    target = "/opt/share/deepfake/linshi/录屏素材/2023-04-14(李云帆 陈雨)/20230414-163616614陈雨(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-02-17(袁忠 曹属馨)/20230217-160724910曹属馨(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-06-20(邓叮玲 许馨文)/20230620-155807793邓叮玲(电脑录屏).mp4"
    # target = "/opt/share/deepfake/linshi/录屏素材/2023-08-28(李小姐 邱小姐)/20230828-122302李小姐(电脑录屏).mp4"
    # source = f"/home/liaohaixun/styleswap_data/dateset/ffhq_1024/{random.randint(0,69999):05d}.png"

    # target = f"/home/liaohaixun/styleswap_data/dateset/ffhq_1024/25635.png"
    # source = f"/home/liaohaixun/styleswap_data/dateset/ffhq_1024/47125.png"
    swapper = Swapper("log/train_unet/ckpt/80000_512.pth")
    # swapper.swap(target, source, "result.png")
    swapper.swap_video(target, "IDAssets/Harry_Potter_character_poster.jpg", "result.mp4")
