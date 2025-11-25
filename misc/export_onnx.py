import os
from typing import Tuple, Optional, Union
import onnx
import torch
import torch.nn as nn
import numpy as np
from misc.model.idencoder.main import PROVIDER
from losses import IDLoss
import onnxruntime as ort

TensorOrTuple = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


def print_onnx_summary(model, topk_initializers: int = 10, topk_ops: int = 20) -> None:
    import math
    from collections import Counter, defaultdict

    g = model.graph

    # === 基础信息 ===
    ir_version = getattr(model, "ir_version", None)
    opsets = {op.domain or "ai.onnx": op.version for op in model.opset_import}
    print("\n=== ONNX SUMMARY ===")
    print(f"IR version: {ir_version}")
    print("Opset imports:", ", ".join([f"'{d}': {v}" for d, v in sorted(opsets.items())]))

    # === dtype 映射与名称 ===
    # 参考 onnx.TensorProto.DataType
    _DT_NAME = {
        0: "UNDEFINED",
        1: "FLOAT32",  # 对应 TensorProto.FLOAT
        2: "UINT8",
        3: "INT8",
        4: "UINT16",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        8: "STRING",
        9: "BOOL",
        10: "FLOAT16",  # TensorProto.FLOAT16
        11: "DOUBLE",  # TensorProto.DOUBLE (float64)
        12: "UINT32",
        13: "UINT64",
        14: "COMPLEX64",
        15: "COMPLEX128",
        16: "BFLOAT16",
        17: "FLOAT8E4M3FN",
        18: "FLOAT8E4M3FNUZ",
        19: "FLOAT8E5M2",
        20: "FLOAT8E5M2FNUZ",
    }
    _DT_NBYTES = {  # 对 STRING/UNDEFINED/复数用近似或跳过
        1: 4,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 4,
        7: 8,
        9: 1,
        10: 2,
        11: 8,
        12: 4,
        13: 8,
        14: 8,
        15: 16,
        16: 2,
        17: 1,
        18: 1,
        19: 1,
        20: 1,
    }

    def _shape_dtype(v):
        t = v.type.tensor_type
        elem = t.elem_type
        shape = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                shape.append(d.dim_param)  # 符号维
            else:
                shape.append("?")
        return shape, elem

    # === 输入输出 ===
    init_names = {i.name for i in g.initializer} | {s.name for s in getattr(g, "sparse_initializer", [])}

    print("\nInputs:")
    for v in g.input:
        if v.name in init_names:
            continue
        shape, elem = _shape_dtype(v)
        print(f"  - {v.name:30s} shape={shape} dtype={_DT_NAME.get(elem, elem)}")

    print("\nOutputs:")
    for v in g.output:
        shape, elem = _shape_dtype(v)
        print(f"  - {v.name:30s} shape={shape} dtype={_DT_NAME.get(elem, elem)}")

    # === 节点统计 ===
    op_hist = Counter(n.op_type for n in g.node)
    total_nodes = sum(op_hist.values())
    print(f"\nNodes: total={total_nodes}")
    for op, cnt in op_hist.most_common(max(1, topk_ops)):
        print(f"  - {op:20s} x {cnt}")

    # 按域统计
    domain_hist = Counter((n.domain or "ai.onnx") for n in g.node)
    print("Domains:")
    for d, cnt in sorted(domain_hist.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {d:20s} x {cnt}")

    # === 初始器统计（dense + sparse）===
    def _tensor_nbytes(dims, elem_type, raw_len):
        if raw_len:  # external_data 无原地数据时 raw_len=0，需用 dims*elem_size 估算
            return raw_len
        n = 1
        for d in dims:
            n *= int(d) if isinstance(d, int) else 0
        item = _DT_NBYTES.get(elem_type)
        return n * item if item is not None else 0

    total_bytes = 0
    sizes = []
    dtype_hist = Counter()

    # dense initializers
    for init in g.initializer:
        sz = _tensor_nbytes(list(init.dims), init.data_type, len(init.raw_data))
        total_bytes += sz
        sizes.append((init.name, sz, list(init.dims), init.data_type))
        dtype_hist.update([_DT_NAME.get(init.data_type, init.data_type)])

    # sparse initializers
    # 只估算 values 部分
    for sinit in getattr(g, "sparse_initializer", []):
        vals = sinit.values
        sz = _tensor_nbytes(list(vals.dims), vals.data_type, len(vals.raw_data))
        total_bytes += sz
        sizes.append((sinit.name, sz, list(vals.dims), vals.data_type))
        dtype_hist.update([_DT_NAME.get(vals.data_type, vals.data_type)])

    sizes.sort(key=lambda x: x[1], reverse=True)
    mb = total_bytes / (1024**2)
    print(f"\nInitializers: count={len(g.initializer)} dense, {len(getattr(g, 'sparse_initializer', []))} sparse, total ~{mb:.2f} MiB")
    if dtype_hist:
        print("Initializer dtype distribution:")
        for k, v in dtype_hist.most_common():
            print(f"  - {k:12s}: {v}")

    print("Top tensors by size:")
    for name, sz, dims, dt in sizes[: max(1, topk_initializers)]:
        print(f"  - {name:30s} {sz/1024/1024:8.2f} MiB  shape={dims} dtype={_DT_NAME.get(dt, dt)}")

    # === external data ===
    uses_external = any(getattr(i, "external_data", None) for i in g.initializer) or any(getattr(s.values, "external_data", None) for s in getattr(g, "sparse_initializer", []))
    print(f"\nExternal data: {bool(uses_external)}")


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


class ONNXExporter:
    def __init__(self, device: Union[str, torch.device] = torch.device("cuda", 0)):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

    def export_model(
        self,
        model_module: nn.Module,
        sample_input: TensorOrTuple,
        onnx_path: str,
        opset_version: int | None = None,
    ) -> bool:
        _ensure_dir(onnx_path)
        model_module.eval()

        # import torch_tensorrt

        # model_trt = torch_tensorrt.compile(model_module, ir="dynamo", inputs=sample_input)

        # torch_tensorrt.save(module=model_trt, file_path="test.engine", output_format="aot_inductor")

        torch.onnx.export(
            model_module,
            sample_input,
            onnx_path,
            opset_version=opset_version,
            export_params=True,
            dynamo=True,
            fallback=False,
            optimize=True,
            verify=True,
            external_data=False,
            keep_initializers_as_inputs=False,
            verbose=True,
            # report=True,
            profile=True,
        )

        # torch_tensorrt.save(trt_gm,"test.engine",output_format=)

        # from onnxconverter_common.float16 import convert_float_to_float16
        # from onnxconverter_common import auto_mixed_precision, auto_convert_mixed_precision
        # import numpy as np

        onnx_model = onnx.load(onnx_path)
        # onnx_model = convert_float_to_float16(
        #     onnx_model,
        #     # min_positive_val=1e-4,
        #     keep_io_types=True,
        # )

        # onnx_model = auto_convert_mixed_precision(
        #     onnx_model,
        #     feed_dict={
        #         "inputs": np.random.rand(1, 3, 512, 512).astype(np.float32),
        #         "id_emb": np.random.rand(1, 512).astype(np.float32),
        #     },
        #     rtol=0.01,
        #     atol=0.001,
        #     keep_io_types=True,
        # )

        # onnx.save_model(onnx_model, onnx_path + ".fp16")

        onnx.checker.check_model(onnx_model, full_check=True)
        print(f"✅ 模型成功导出到: {onnx_path}")
        print_onnx_summary(onnx_model)
        return True

    @staticmethod
    def _basic_metrics(ref: np.ndarray, pred: np.ndarray, value_range: Optional[Tuple[float, float]] = None):
        a = ref.astype(np.float64, copy=False).ravel()
        b = pred.astype(np.float64, copy=False).ravel()
        d = b - a
        absd = np.abs(d)
        mae = float(absd.mean())
        mse = float((d * d).mean())
        max_abs = float(absd.max(initial=0.0))
        # 余弦相似度
        dot = float(np.dot(a, b))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        cos = dot / max(na * nb, 1e-12)
        # 可选 PSNR
        if value_range is not None:
            peak = max(float(value_range[1] - value_range[0]), 1e-12)
            psnr = 20.0 * np.log10(peak) - 10.0 * np.log10(max(mse, 1e-12))
        else:
            psnr = None
        return mae, mse, max_abs, cos, psnr

    def compare_precision(
        self,
        model_module: nn.Module,
        sample_input: TensorOrTuple,
        onnx_path: str,
        value_range: Optional[Tuple[float, float]] = None,  # 例如(-1,1)或(0,1)
    ) -> None:
        model_module.eval()
        with torch.inference_mode():
            tout = model_module(*sample_input) if isinstance(sample_input, tuple) else model_module(sample_input)
            if not isinstance(tout, tuple):
                tout = (tout,)
            torch_outputs = [t.detach().cpu().numpy() for t in tout]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 4
        # so.log_severity_level = 1
        sess = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": self.device.index}],
            sess_options=so,
        )
        feed = {sess.get_inputs()[i].name: x.cpu().numpy() for i, x in enumerate(sample_input if isinstance(sample_input, tuple) else (sample_input,))}
        onnx_outputs = sess.run(None, feed)
        if not isinstance(onnx_outputs, tuple):
            onnx_outputs = (*onnx_outputs,)

        print(f"\nPrecision Summary: {onnx_path}")
        for i, (a, b) in enumerate(zip(torch_outputs, onnx_outputs), 1):
            if a.shape != b.shape:
                print(f"Out{i}: shape mismatch {a.shape} vs {b.shape}")
            mae, mse, mx, cos, psnr = self._basic_metrics(a, b, value_range)
            if psnr is None:
                print(f"Out{i}: MAE={mae:.6g}  MSE={mse:.6g}  Max={mx:.6g}  Cos={cos:.6f}")
            else:
                print(f"Out{i}: MAE={mae:.6g}  MSE={mse:.6g}  Max={mx:.6g}  Cos={cos:.6f}  PSNR={psnr:.2f}dB")

    def export_swap(self, ckpt: str, sp: str, opset_version: int | None = None):
        import model
        from model import FullGenerator
        from train import TrainSet
        import torch.nn.functional as F

        model.mode.EXPORT_MODE = True

        ckpt_obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "train_info" not in ckpt_obj or "g_ema" not in ckpt_obj:
            raise KeyError("检查点文件格式错误")

        train_info = TrainSet(**ckpt_obj["train_info"])
        m = FullGenerator(
            size=train_info.size,
            style_dim=train_info.style_dim,
            n_mlp=train_info.n_mlp,
            isconcat=train_info.isconcat,
        ).to(self.device)
        m.load_state_dict(ckpt_obj["g_ema"], strict=True)
        m.eval()

        target = torch.rand(1, train_info.size, train_info.size, 3, device=self.device).uniform_(-1, 1)
        id_tensor = torch.rand(1, train_info.style_dim, device=self.device).uniform_(-1, 1)

        sample_inputs: Tuple[torch.Tensor, torch.Tensor] = (target, id_tensor)

        self.export_model(m, sample_inputs, sp, opset_version=opset_version)
        self.compare_precision(m, sample_inputs, sp)

        return train_info.id_encode_provider

    def export_idencode(self, sp: str, id_encode_provider: PROVIDER = PROVIDER.BLENDFACE, opset_version: int | None = None):
        import torch.nn.functional as F

        idloss = IDLoss(id_encode_provider=id_encode_provider).to(self.device)
        backbone = idloss.arcface.eval()

        inp = torch.randn(1, 3, 112, 112, device=self.device)

        self.export_model(backbone, inp, sp, opset_version=opset_version)
        self.compare_precision(backbone, inp, sp, value_range=(-1.0, 1.0))


if __name__ == "__main__":
    exp = ONNXExporter(device=torch.device("cuda:0"))

    swap_ckpt = "log/train_7_f32/ckpt/3620000.pth"
    swap_onnx_sp = "onnx_model/512.onnx"
    idencode_onnx_sp = "onnx_model/id_encode.onnx"

    id_encode_provider = exp.export_swap(swap_ckpt, swap_onnx_sp)
    # exp.export_idencode(idencode_onnx_sp, id_encode_provider=id_encode_provider)
