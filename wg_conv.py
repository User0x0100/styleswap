import torch
from model import MultiScaleDiscriminator

checkpoint_old = torch.load(
    "ckpt/990000_512.pth",
    map_location="cpu",
    weights_only=False,
)
checkpoint_new = torch.load(
    "ckpt/1042069_512.pth",
    map_location="cpu",
    weights_only=False,
)

d = MultiScaleDiscriminator(out_size=512)
d.load_state_dict(checkpoint_new["d"])
d.load_sig_disc_wg(checkpoint_old["d"])

save_dict = {
    "iter": 990000,
    "d": d.state_dict(),
    "g": checkpoint_old["g"],
    "g_ema": checkpoint_old["g_ema"],
    "d_optim": checkpoint_old["d_optim"],
    "g_optim": checkpoint_old["d_optim"],
}
torch.save(save_dict, "multiscaldisc.pth")