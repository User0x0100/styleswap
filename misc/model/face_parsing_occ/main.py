from pathlib import Path
import segmentation_models_pytorch as smp
import torch
from torch.nn import functional as F
from misc.utils import ImageFolder
import torchvision.utils as utils

# https://github.com/face3d0725/FaceExtraction/blob/main/Dataset/utils.py
# ENCODER_WEIGHTS:
# "resnet18": {
#     "imagenet": {
#         "url": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
#         "input_space": "RGB",
#         "input_size": [3, 224, 224],
#         "input_range": [0, 1],
#         "mean": [0.485, 0.456, 0.406],
#         "std": [0.229, 0.224, 0.225],
#         "num_classes": 1000,
#     },


ENCODER = "resnet18"
ENCODER_WEIGHTS = "imagenet"
CLASSES = 1
ATTENTION = None
ACTIVATION = None
DEVICE = "cuda:0"


ENCODER_WEIGHTS = Path(__file__).with_name("resnet18-5c106cde.pth")
STATE_DICT = Path(__file__).with_name("epoch_16_best.ckpt")


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)

        y = self.model(x)

        # NCHW -> NHWC
        y = y.permute(0, 2, 3, 1)
        # y = (y > 0).to(dtype=torch.float)
        y = torch.where(y > 0, 1.0, 0.0)
        return y


if __name__ == "__main__":
    # model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, decoder_attention_type=ATTENTION, classes=CLASSES, activation=ACTIVATION)
    # model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, decoder_attention_type=ATTENTION, classes=CLASSES, activation=ACTIVATION)
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, decoder_attention_type=None, classes=1, activation=None)

    encoder_state_dict = torch.load(ENCODER_WEIGHTS, weights_only=False)
    model.encoder.load_state_dict(encoder_state_dict)

    state_dict = {cleaned: v for k, v in torch.load(STATE_DICT).items() if (cleaned := k.removeprefix("module."))}
    model.load_state_dict(state_dict)

    # model = model.to(DEVICE).eval()

    # dataset = DataFolder("/opt/share/deepfake/dataset_1/ffhq_1024").iter_batch_with_tensor(batch_size=1, rando=True)
    # image = torch.stack(next(dataset))
    # image = image.to(device=DEVICE, dtype=torch.float).div_(255.0)
    # input = F.interpolate(image, size=224, mode="bicubic")

    # save_folder = Path(__name__).with_name("visi")
    # save_folder.mkdir(parents=True, exist_ok=True)
    # output = model(input)
    # utils.save_image(output, save_folder / f"face_parseing_occ.png", nrow=16)

    model = Wrapper(model).eval()

    onnx_model_name = "face_parsing_occ.onnx"

    import onnx
    import onnxruntime
    import numpy as np

    onnx_model = torch.onnx.export(
        model,
        torch.randn(1, 224, 224, 3),
        onnx_model_name,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)

    sample = torch.randn(1, 224, 224, 3)

    ort_session = onnxruntime.InferenceSession(onnx_model_name, providers=["CPUExecutionProvider"])

    ort_inputs = {"input": sample.numpy()}
    ort_outputs = ort_session.run(output_names=None, input_feed=ort_inputs)
    with torch.inference_mode():
        torch_out = model(sample)

    np.testing.assert_allclose(torch_out.numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
    print("IR:", onnx_model.ir_version)
    print("Opset:", onnx_model.opset_import)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
