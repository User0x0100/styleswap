from pathlib import Path
from torchvision.io import decode_image
from torchvision.io.image import ImageReadMode
from misc.tools import FaceAlign
from misc.model import IDEncoder
from misc.model.idencoder.main import PROVIDER
import torch
from misc.utils import ImageFolder
import torch.nn.functional as F


class FaceFilter:
    def __init__(self, target_face: str, dir: str, similarity_threshold: float = 0.3, device: str = "cuda:0"):

        if Path(target_face).exists() == False:
            raise FileNotFoundError(f"Target_face photo file: {target_face} not found")

        self.similarity_threshold = similarity_threshold

        self.dir = ImageFolder(dir)

        self.Eliminated_dir = Path(dir).joinpath("Eliminated")

        self.Eliminated_dir.mkdir(exist_ok=True, parents=True)

        self.device = torch.device(device)

        face = decode_image(target_face, ImageReadMode.RGB).to(device=self.device, dtype=torch.float).div_(255.0).unsqueeze(0)  # f32 1CHW [0.0 ~ 1.0]
        align = FaceAlign(from_unit_range=True).to(self.device)
        aligned_face, _, _ = align.align(face, (224,) * 2, IDEncoder.get_dst_pts(224).to(self.device))

        self.idencoder = IDEncoder.Model(PROVIDER.ARC_OFFICIAL_R100).to(self.device)

        self.target_id = self.idencoder(aligned_face)

    @torch.no_grad
    def do(self):

        for image in self.dir:
            face = decode_image(image, ImageReadMode.RGB).to(device=self.device, dtype=torch.float).div_(255.0).unsqueeze(0)  # f32 1CHW [0.0 ~ 1.0]
            face_id = self.idencoder(face)
            similarity = F.cosine_similarity(self.target_id, face_id).mean().item()

            if similarity < self.similarity_threshold:
                face.rename(self.Eliminated_dir / image.name)


if __name__ == "__main__":
    pass
