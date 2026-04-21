import torch
import torch.nn.functional as F
from torch import nn, Tensor

from misc.model import PTS682D, IDEncoder
from misc.model.idencoder.main import PROVIDER
from misc.tools import compute_similarity_transform, pixel_to_normalized


class IDLoss(nn.Module):
    def __init__(self, weight: float = 1.0, id_encode_provider: PROVIDER = PROVIDER.ARC_OFFICIAL_R50):
        super().__init__()
        self.weight = weight

        self.arcface = IDEncoder.Model(provider=id_encode_provider)
        self.pts682d = PTS682D.Model()

        # self.arcface = torch.compile(self.arcface, mode="max-autotune")
        # self.pts682d = torch.compile(self.pts682d, mode="max-autotune")

        dst_pts_norm = pixel_to_normalized(IDEncoder.ARCFACE_DST_PTS, [112, 112])

        self.register_buffer("dst_pts_norm", dst_pts_norm, persistent=False)

        self.eval()
        self.requires_grad_(False)

    def get_id_feats(self, face: Tensor, face_mask: Tensor | None = None, realign: bool = True) -> Tensor:
        if realign:
            with torch.no_grad():
                B, _, H, W = face.shape
                pts68 = self.pts682d(face)

                src = PTS682D.convert_68_to_5(pts68)
                src = pixel_to_normalized(src, [H, W])
                dst = self.dst_pts_norm.unsqueeze(0).expand(B, -1, -1)

                M = compute_similarity_transform(dst, src)

                face_grid = F.affine_grid(M, face.size(), align_corners=False)

                if face_mask is not None:
                    mask_grid = F.affine_grid(M, face_mask.size(), align_corners=False)
                    face_mask = F.grid_sample(face_mask, mask_grid, align_corners=False, mode="bicubic")

            face = F.grid_sample(face, face_grid, align_corners=False, mode="bicubic")

        if face_mask is not None:
            face_mask = face_mask.detach()
            face = face * face_mask + (1.0 - face_mask) * (-1.0)

        return self.arcface(face)

    def register_ifsr_hooks(self, block_names: list[str]):
        self._ifsr_feats: dict[str, Tensor] = {}
        self._ifsr_hooks = []

        module_dict = dict(self.arcface.backbone.named_modules())

        def make_hook(name):
            def hook(module, inputs, output):
                self._ifsr_feats[name] = output

            return hook

        for name in block_names:
            if name not in module_dict:
                raise KeyError(f"IFSR block not found in model: {name}")

            module = module_dict[name]

            h = module.register_forward_hook(make_hook(name))
            self._ifsr_hooks.append(h)

    def forward(self, fake_id, real_id):
        loss = (1.0 - F.cosine_similarity(fake_id, real_id)).mean()
        return loss * self.weight


if __name__ == "__main__":

    import torch
    import torch.nn.functional as F

    from misc.model import FaceParsing
    from misc.swapface import Swapper
    from misc.utils import ImageFolder, Timer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 26
    dataset = ImageFolder("/opt/share/deepfake/dataset_1/ffhq_1024")

    id_loss_sim = IDLoss(id_encode_provider=PROVIDER.ARC_SIMSWAP).to(device=device)
    id_loss_off = IDLoss(id_encode_provider=PROVIDER.ARC_OFFICIAL_R50).to(device=device)
    id_loss_bf = IDLoss(id_encode_provider=PROVIDER.BLENDFACE).to(device=device)

    facepars = FaceParsing.Model(compile_model=False).to(device=device)
    swapper = Swapper("log/train_6/ckpt/260000.pth").to(device=device)

    a = dataset.sample_tensor(rando=False, label=False).unsqueeze_(0).to(device=device, dtype=torch.float).div_(255.0).mul_(2.0).sub_(1.0)
    b = dataset.sample_tensor(rando=False, label=False).unsqueeze_(0).to(device=device, dtype=torch.float).div_(255.0).mul_(2.0).sub_(1.0)

    a = F.interpolate(a, size=512, mode="bicubic", align_corners=False)
    b = F.interpolate(b, size=512, mode="bicubic", align_corners=False)

    mask_a = facepars.forward(a)
    mask_b = facepars.forward(b)

    id_emb = id_loss_off.get_id_feats(a, mask_a, realign=True)

    fake = swapper.swap_single(b, id_emb)

    id_a = id_loss_sim.get_id_feats(a, mask_a, realign=True)
    id_fake = id_loss_sim.get_id_feats(fake, mask_b, realign=True)

    sim = F.cosine_similarity(id_a, id_fake, dim=1)
    print(f"simswap: {sim.item()}")

    id_a = id_loss_bf.get_id_feats(a, realign=True)
    id_fake = id_loss_bf.get_id_feats(fake, realign=True)

    sim = F.cosine_similarity(id_a, id_fake, dim=1)
    print(f"blendface: {sim.item()}")
