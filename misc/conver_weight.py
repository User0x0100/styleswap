import torch

from face_swap_model import MultiScaleDiscriminator

size = 512
device = "cuda"
ckpt = "1180000_512.pth"
ckpt = torch.load(
    ckpt,
    map_location=device,
    weights_only=False,
)

m = MultiScaleDiscriminator(out_size=size)

m.load_state_dict(ckpt["d"])
ckpt["d"] = m.discriminators[0].state_dict()


torch.save(ckpt, "sig_d.pth")
