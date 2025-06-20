import os
import onnx
import torch
import numpy as np
import onnxruntime as ort
from onnx import checker
from onnx.checker import ValidationError
from losses.arcface import ArcFace
from model import FullGenerator


def export_onnx(model, input, onnx_file: str, opset_version: int = 20) -> None:

    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)
    model.eval()

    torch.onnx.export(
        model,
        input,
        onnx_file,
        opset_version=opset_version,
    )

    # 验证导出的模型
    try:
        model = onnx.load(onnx_file)
        checker.check_model(model, full_check=True)
        print(f"✅ ONNX模型验证成功，已保存到: {onnx_file}")

        # 打印模型输入输出信息
        print("\n模型信息:")
        print("输入:")
        for input in model.graph.input:
            print(
                f"  - {input.name}: {[d.dim_value if d.dim_value else 'dynamic' for d in input.type.tensor_type.shape.dim]}"
            )
        print("输出:")
        for output in model.graph.output:
            print(
                f"  - {output.name}: {[d.dim_value if d.dim_value else 'dynamic' for d in output.type.tensor_type.shape.dim]}"
            )
    except ValidationError as e:
        print("❌ ONNX模型验证失败：")
        print(e)


def compare_model_accuracy(
    pytorch_model, onnx_file, pytorch_device="cpu", onnx_device="cpu", num_tests=10
):
    """
    对比PyTorch模型与ONNX模型的推理精度差距

    Args:
        pytorch_model: PyTorch模型
        onnx_file: ONNX模型文件路径
        pytorch_device: PyTorch模型运行设备
        onnx_device: ONNX模型运行设备 ('cpu', 'cuda')
        num_tests: 测试次数
    """
    print(f"\n🔍 开始对比模型精度差距...")
    print(f"PyTorch设备: {pytorch_device}, ONNX设备: {onnx_device}")

    # 创建ONNX Runtime会话
    try:
        # 配置ONNX Runtime提供者
        providers = []
        if onnx_device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        ort_session = ort.InferenceSession(onnx_file, providers=providers)
        print(f"✅ ONNX Runtime会话创建成功，使用提供者: {ort_session.get_providers()}")
    except Exception as e:
        print(f"❌ 创建ONNX Runtime会话失败: {e}")
        return

    # 获取输入输出信息
    input_specs = []
    for inp in ort_session.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str) or dim is None:
                shape.append(1)  # 动态维度设为1
            else:
                shape.append(dim)
        input_specs.append({"name": inp.name, "shape": shape, "type": inp.type})

    output_specs = []
    for outp in ort_session.get_outputs():
        output_specs.append({"name": outp.name, "shape": outp.shape, "type": outp.type})

    print(f"模型输入: {len(input_specs)}个")
    for i, spec in enumerate(input_specs):
        print(f"  输入{i}: {spec['name']}, 形状: {spec['shape']}")

    print(f"模型输出: {len(output_specs)}个")
    for i, spec in enumerate(output_specs):
        print(f"  输出{i}: {spec['name']}, 形状: {spec['shape']}")

    # 存储每个输出的误差统计
    outputs_stats = {}
    for i, spec in enumerate(output_specs):
        outputs_stats[spec["name"]] = {
            "mae_errors": [],
            "mse_errors": [],
            "max_errors": [],
            "cosine_similarities": [],
        }

    pytorch_model.eval()

    with torch.no_grad():
        for test_idx in range(num_tests):
            # 根据ONNX模型信息生成测试输入
            test_inputs = []
            onnx_inputs = {}

            for spec in input_specs:
                # 生成随机输入数据
                if "float" in spec["type"]:
                    test_data = torch.randn(
                        spec["shape"], dtype=torch.float32, device=pytorch_device
                    )
                elif "int" in spec["type"]:
                    test_data = torch.randint(
                        0, 100, spec["shape"], dtype=torch.int64, device=pytorch_device
                    )
                else:
                    test_data = torch.randn(
                        spec["shape"], dtype=torch.float32, device=pytorch_device
                    )

                test_inputs.append(test_data)
                onnx_inputs[spec["name"]] = test_data.cpu().numpy()

            # PyTorch模型推理
            if len(test_inputs) == 1:
                pytorch_outputs = pytorch_model(test_inputs[0])
            else:
                pytorch_outputs = pytorch_model(*test_inputs)

            # 确保pytorch_outputs是列表或元组
            if not isinstance(pytorch_outputs, (list, tuple)):
                pytorch_outputs = [pytorch_outputs]

            # ONNX模型推理
            onnx_outputs = ort_session.run(None, onnx_inputs)

            # 对比每个输出
            for i, (pytorch_out, onnx_out) in enumerate(
                zip(pytorch_outputs, onnx_outputs)
            ):
                output_name = output_specs[i]["name"]

                # 转换为numpy
                if torch.is_tensor(pytorch_out):
                    pytorch_out_np = pytorch_out.cpu().numpy()
                else:
                    pytorch_out_np = pytorch_out

                # 确保形状匹配
                if pytorch_out_np.shape != onnx_out.shape:
                    print(
                        f"⚠️  输出{i}形状不匹配: PyTorch {pytorch_out_np.shape} vs ONNX {onnx_out.shape}"
                    )
                    continue

                # 计算误差指标
                mae = np.mean(np.abs(pytorch_out_np - onnx_out))
                mse = np.mean((pytorch_out_np - onnx_out) ** 2)
                max_error = np.max(np.abs(pytorch_out_np - onnx_out))

                # 余弦相似度
                pytorch_flat = pytorch_out_np.flatten()
                onnx_flat = onnx_out.flatten()
                if np.linalg.norm(pytorch_flat) > 0 and np.linalg.norm(onnx_flat) > 0:
                    cosine_sim = np.dot(pytorch_flat, onnx_flat) / (
                        np.linalg.norm(pytorch_flat) * np.linalg.norm(onnx_flat)
                    )
                else:
                    cosine_sim = 1.0 if np.allclose(pytorch_flat, onnx_flat) else 0.0

                # 存储结果
                outputs_stats[output_name]["mae_errors"].append(mae)
                outputs_stats[output_name]["mse_errors"].append(mse)
                outputs_stats[output_name]["max_errors"].append(max_error)
                outputs_stats[output_name]["cosine_similarities"].append(cosine_sim)

    # 打印每个输出的统计结果
    print(f"\n📊 精度对比结果 (基于{num_tests}次测试):")

    overall_quality_scores = []

    for output_name, stats in outputs_stats.items():
        print(f"\n🎯 输出 '{output_name}':")
        print(f"  平均绝对误差 (MAE):")
        print(f"    - 平均值: {np.mean(stats['mae_errors']):.8f}")
        print(f"    - 最大值: {np.max(stats['mae_errors']):.8f}")
        print(f"    - 最小值: {np.min(stats['mae_errors']):.8f}")

        print(f"  均方误差 (MSE):")
        print(f"    - 平均值: {np.mean(stats['mse_errors']):.8f}")
        print(f"    - 最大值: {np.max(stats['mse_errors']):.8f}")
        print(f"    - 最小值: {np.min(stats['mse_errors']):.8f}")

        print(f"  最大误差:")
        print(f"    - 平均值: {np.mean(stats['max_errors']):.8f}")
        print(f"    - 最大值: {np.max(stats['max_errors']):.8f}")
        print(f"    - 最小值: {np.min(stats['max_errors']):.8f}")

        print(f"  余弦相似度:")
        print(f"    - 平均值: {np.mean(stats['cosine_similarities']):.8f}")
        print(f"    - 最大值: {np.max(stats['cosine_similarities']):.8f}")
        print(f"    - 最小值: {np.min(stats['cosine_similarities']):.8f}")

        # 单个输出的质量评估
        avg_mae = np.mean(stats["mae_errors"])
        avg_cosine_sim = np.mean(stats["cosine_similarities"])

        if avg_mae < 1e-5 and avg_cosine_sim > 0.9999:
            quality = "优秀"
            score = 4
            print(f"  🎉 '{output_name}' 转换精度: {quality}")
        elif avg_mae < 1e-4 and avg_cosine_sim > 0.999:
            quality = "良好"
            score = 3
            print(f"  ✅ '{output_name}' 转换精度: {quality}")
        elif avg_mae < 1e-3 and avg_cosine_sim > 0.99:
            quality = "可接受"
            score = 2
            print(f"  ⚠️  '{output_name}' 转换精度: {quality}")
        else:
            quality = "需要检查"
            score = 1
            print(f"  ❌ '{output_name}' 转换精度: {quality}")

        overall_quality_scores.append(score)

    # 整体质量评估
    if overall_quality_scores:
        avg_score = np.mean(overall_quality_scores)
        min_score = min(overall_quality_scores)

        print(f"\n🏆 整体模型转换质量:")
        if min_score >= 4:
            print("🎉 优秀 - 所有输出精度都很高")
        elif min_score >= 3:
            print("✅ 良好 - 所有输出精度可接受")
        elif min_score >= 2:
            print("⚠️  可接受 - 部分输出精度需关注")
        else:
            print("❌ 需要检查 - 存在精度问题")

        print(f"平均质量分数: {avg_score:.2f}/4.0")


def export_arcface(
    onnx_file,
    device: str = "cpu",
    compare_accuracy: bool = True,
    pytorch_device: str = None,
    onnx_device: str = None,
    num_tests: int = 10,
):
    """
    导出ArcFace模型到ONNX格式并可选择性地对比精度

    Args:
        onnx_file: ONNX文件保存路径
        device: 导出时使用的设备类型
        compare_accuracy: 是否对比精度
        pytorch_device: PyTorch模型推理设备 (默认与device相同)
        onnx_device: ONNX模型推理设备 (默认为'cpu')
        num_tests: 精度测试次数
    """
    input = torch.randn(1, 3, 112, 112, dtype=torch.float32, device=device)
    arcface = ArcFace(fp16=False).to(device)

    # 导出ONNX模型
    export_onnx(arcface, input, onnx_file)

    # 对比精度
    if compare_accuracy:
        pytorch_device = pytorch_device or device
        onnx_device = onnx_device or "cpu"

        # 将模型移动到推理设备
        arcface_for_test = arcface.to(pytorch_device)

        compare_model_accuracy(
            arcface_for_test,
            onnx_file,
            pytorch_device=pytorch_device,
            onnx_device=onnx_device,
            num_tests=num_tests,
        )


def export_face_model(
    size,
    onnx_file,
    checkpoint,
    device: str = "cpu",
    compare_accuracy: bool = True,
    pytorch_device: str = None,
    onnx_device: str = None,
    num_tests: int = 10,
):
    """
    导出面部模型到ONNX格式并可选择性地对比精度

    Args:
        size: 模型尺寸
        onnx_file: ONNX文件保存路径
        checkpoint: 检查点文件路径
        device: 导出时使用的设备类型
        compare_accuracy: 是否对比精度
        pytorch_device: PyTorch模型推理设备 (默认与device相同)
        onnx_device: ONNX模型推理设备 (默认为'cpu')
        num_tests: 精度测试次数
    """
    model = FullGenerator(size=size).to(device)
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["g_ema"])

    id = torch.randn(1, 512, dtype=torch.float32, device=device)
    target = torch.randn(1, 3, 512, 512, dtype=torch.float32, device=device)
    test_input = (target, id)

    # 导出ONNX模型
    export_onnx(model, test_input, onnx_file)

    # 对比精度
    if compare_accuracy:
        pytorch_device = pytorch_device or device
        onnx_device = onnx_device or "cpu"

        # 将模型移动到推理设备
        model_for_test = model.to(pytorch_device)

        compare_model_accuracy(
            model_for_test,
            onnx_file,
            pytorch_device=pytorch_device,
            onnx_device=onnx_device,
            num_tests=num_tests,
        )


if __name__ == "__main__":

    size = 512
    checkpoint = "ckpt/45000_512.pth"
    face_onnx_file = "onnx_model/styleswap_512_ffhq.onnx"
    arcface_onnx_file = "onnx_model/glint360k_r50_fp16.onnx"

    # 导出ArcFace模型并对比精度
    export_arcface(
        arcface_onnx_file,
        device="cuda",
        compare_accuracy=True,
        pytorch_device="cuda",
        onnx_device="cuda",
        num_tests=200,
    )

    # 示例：在不同设备上进行对比
    # export_arcface(arcface_onnx_file, device="cuda", compare_accuracy=True,
    #               pytorch_device="cuda", onnx_device="cuda")

    # export_face_model(
    #     size,
    #     face_onnx_file,
    #     checkpoint,
    #     device="cpu",
    #     compare_accuracy=True,
    #     pytorch_device="cuda",
    #     onnx_device="cuda",
    #     num_tests=100,
    # )
