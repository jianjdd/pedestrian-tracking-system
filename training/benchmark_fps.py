"""
FPS 基准测试脚本
测试不同模块组合对推理速度的影响，用于消融实验中的速度对比。

测试模型:
  1. Baseline:           yolov8n (标准 YOLOv8)
  2. + P2:               yolov8n-p2 (增加高分辨率检测头)
  3. + P2 + BiFPN:       yolov8n-p2-bifpn (加权特征融合)
  4. + P2 + DCNv2:       yolov8n-p2-dcnv2 (可变形卷积)
  5. + P2 + BiFPN + DCN: yolov8n-p2-bifpn-dcnv2 (完整改进)

测试方式:
  使用随机张量模拟输入，GPU Warmup 后统计多轮推理的平均耗时和 FPS。
"""

import sys
import time
import torch
import numpy as np

# 将 ultralytics 加入路径（使用相对路径）
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ultralytics"))

from ultralytics import YOLO


def count_flops(model, input_tensor) -> float:
    """统计模型的 GFLOPs（基于 forward hook，覆盖 Conv2d / Linear / BN）。

    Args:
        model: nn.Module
        input_tensor: 示例输入 (1, 3, H, W)

    Returns:
        float: GFLOPs 值
    """
    flops = {}
    hooks = []

    def _conv2d_hook(m, inp, out):
        # FLOPs = 2 * k_h * k_w * C_in * C_out * H_out * W_out / groups
        if hasattr(m, "weight") and m.weight is not None:
            _, c_in, k_h, k_w = m.weight.shape
            c_out = m.out_channels
            groups = m.groups
            _, _, h_out, w_out = out.shape
            flops[id(m)] = 2 * k_h * k_w * (c_in // groups) * c_out * h_out * w_out

    def _linear_hook(m, inp, out):
        if hasattr(m, "weight") and m.weight is not None:
            flops[id(m)] = 2 * m.in_features * m.out_features

    def _bn_hook(m, inp, out):
        if out.ndim == 4:
            flops[id(m)] = 2 * out.shape[1] * out.shape[2] * out.shape[3]

    def _dcnv2_hook(m, inp, out):
        # DeformConv2d: same FLOPs formula as standard Conv2d
        # FLOPs = 2 * k_h * k_w * C_in * C_out * H_out * W_out / groups
        # Plus additional cost for offset/mask (minor, accounted by offset_conv + mask_conv hooks)
        if hasattr(m, "weight") and m.weight is not None:
            out_c, in_c_per_group, k_h, k_w = m.weight.shape
            groups = m.groups
            c_in = in_c_per_group * groups
            _, _, h_out, w_out = out.shape
            flops[id(m)] = 2 * k_h * k_w * c_in * out_c * h_out * w_out // groups

    try:
        import torchvision
        _has_torchvision = True
    except ImportError:
        _has_torchvision = False

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv2d_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            hooks.append(m.register_forward_hook(_bn_hook))
        elif _has_torchvision and isinstance(m, torchvision.ops.DeformConv2d):
            hooks.append(m.register_forward_hook(_dcnv2_hook))

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    total = sum(flops.values())
    return total / 1e9  # GFLOPs


def benchmark_model(model_yaml: str, imgsz: int = 640, warmup: int = 50, runs: int = 200, device: str = "cuda"):
    """
    对单个模型进行 FPS 基准测试。

    Args:
        model_yaml: 模型的 YAML 配置文件名 (如 'yolov8n.yaml')
        imgsz: 输入图像尺寸
        warmup: GPU 预热轮数
        runs: 正式测试轮数
        device: 推理设备 ('cuda' or 'cpu')
    """
    print(f"\n{'='*60}")
    print(f"  模型: {model_yaml}")
    print(f"  输入尺寸: {imgsz}x{imgsz} | 设备: {device}")
    print(f"{'='*60}")

    # ---------- 1. 加载模型 ----------
    try:
        model = YOLO(model_yaml, task="detect")
        torch_model = model.model.to(device).eval()
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return None

    # 统计参数量和计算量
    total_params = sum(p.numel() for p in torch_model.parameters())
    trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    print(f"  参数量: {total_params / 1e6:.2f}M (可训练: {trainable_params / 1e6:.2f}M)")

    # ---------- 2. 构造虚拟输入并计算 GFLOPs ----------
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=device)
    try:
        gflops = count_flops(torch_model, dummy_input)
        print(f"  GFLOPs: {gflops:.2f}G")
    except Exception as e:
        print(f"  GFLOPs: 计算失败 ({e})")
        gflops = None

    # ---------- 3. GPU Warmup ----------
    print(f"  🔥 GPU 预热中 ({warmup} 轮)...", end=" ", flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            _ = torch_model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()
    print("完成")

    # ---------- 4. 正式测试 ----------
    print(f"  ⏱️  正式测试中 ({runs} 轮)...", end=" ", flush=True)
    latencies = []

    with torch.no_grad():
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.synchronize()

            t_start = time.perf_counter()
            _ = torch_model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            latencies.append((t_end - t_start) * 1000)  # 转为毫秒

    print("完成")

    # ---------- 5. 统计结果 ----------
    latencies = np.array(latencies)
    avg_ms = latencies.mean()
    std_ms = latencies.std()
    fps = 1000.0 / avg_ms
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)

    print(f"\n  📊 测试结果:")
    print(f"     平均延迟:  {avg_ms:.2f} ms ± {std_ms:.2f} ms")
    print(f"     中位延迟:  {p50:.2f} ms (P50)")
    print(f"     尾部延迟:  {p95:.2f} ms (P95)")
    print(f"     平均 FPS:  {fps:.1f}")
    print(f"     参数量:    {total_params / 1e6:.2f}M")

    return {
        "model": model_yaml,
        "params_M": round(total_params / 1e6, 2),
        "gflops": round(gflops, 2) if gflops else None,
        "avg_ms": round(avg_ms, 2),
        "std_ms": round(std_ms, 2),
        "fps": round(fps, 1),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
    }


def benchmark_nms(num_boxes: int = 5000, runs: int = 100):
    """
    对比 Standard NMS 与 Soft-NMS 的处理速度。

    Args:
        num_boxes: 模拟的候选框数量
        runs: 测试轮数
    """
    from ultralytics.utils.nms import TorchNMS

    print(f"\n{'='*60}")
    print(f"  NMS 速度对比 (候选框数量: {num_boxes})")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟检测框和分数
    boxes = torch.rand(num_boxes, 4, device=device) * 640
    # 确保 x2 > x1, y2 > y1
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    scores = torch.rand(num_boxes, device=device)

    results = {}

    # --- Standard NMS ---
    print(f"\n  ⏱️  Standard NMS ({runs} 轮)...", end=" ", flush=True)
    latencies_nms = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = TorchNMS.nms(boxes.clone(), scores.clone(), iou_threshold=0.45)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies_nms.append((time.perf_counter() - t0) * 1000)
    avg_nms = np.mean(latencies_nms)
    print(f"完成 | 平均延迟: {avg_nms:.2f} ms")
    results["Standard NMS"] = round(avg_nms, 2)

    # --- Soft-NMS (Gaussian) ---
    print(f"  ⏱️  Soft-NMS Gaussian ({runs} 轮)...", end=" ", flush=True)
    latencies_soft = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = TorchNMS.soft_nms(boxes.clone(), scores.clone(), iou_threshold=0.45, sigma=0.5, method="gaussian")
        if device == "cuda":
            torch.cuda.synchronize()
        latencies_soft.append((time.perf_counter() - t0) * 1000)
    avg_soft = np.mean(latencies_soft)
    print(f"完成 | 平均延迟: {avg_soft:.2f} ms")
    results["Soft-NMS (Gaussian)"] = round(avg_soft, 2)

    # --- Soft-NMS (Linear) ---
    print(f"  ⏱️  Soft-NMS Linear ({runs} 轮)...", end=" ", flush=True)
    latencies_linear = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = TorchNMS.soft_nms(boxes.clone(), scores.clone(), iou_threshold=0.45, sigma=0.5, method="linear")
        if device == "cuda":
            torch.cuda.synchronize()
        latencies_linear.append((time.perf_counter() - t0) * 1000)
    avg_linear = np.mean(latencies_linear)
    print(f"完成 | 平均延迟: {avg_linear:.2f} ms")
    results["Soft-NMS (Linear)"] = round(avg_linear, 2)

    print(f"\n  📊 NMS 对比结果:")
    print(f"     {'方法':<25} {'平均延迟 (ms)':>15}")
    print(f"     {'-'*40}")
    for method, ms in results.items():
        print(f"     {method:<25} {ms:>12.2f} ms")

    return results


def print_summary_table(results: list):
    """打印汇总对比表格。"""
    print(f"\n\n{'='*90}")
    print(f"  📊 模型 FPS 汇总对比表")
    print(f"{'='*90}")
    print(f"  {'模型':<35} {'参数量':>8} {'GFLOPs':>9} {'延迟(ms)':>10} {'FPS':>8} {'P95(ms)':>10}")
    print(f"  {'-'*85}")

    baseline_fps = None
    for r in results:
        if r is None:
            continue
        if baseline_fps is None:
            baseline_fps = r["fps"]
            delta = ""
        else:
            change = ((r["fps"] - baseline_fps) / baseline_fps) * 100
            delta = f" ({change:+.1f}%)"

        gflops_str = f"{r['gflops']:.2f}G" if r.get("gflops") else "N/A"
        print(
            f"  {r['model']:<35} {r['params_M']:>6.2f}M "
            f"{gflops_str:>9} {r['avg_ms']:>8.2f}ms {r['fps']:>7.1f}{delta:<8} {r['p95_ms']:>8.2f}ms"
        )

    print(f"{'='*90}\n")


if __name__ == "__main__":
    # ========== 配置 ==========
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMGSZ = 640
    WARMUP = 10 if DEVICE == "cpu" else 50     # 预热轮数
    RUNS = 30 if DEVICE == "cpu" else 200       # 正式测试轮数

    print(f"🖥️  设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"🎮  GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾  显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"📐  输入尺寸: {IMGSZ}x{IMGSZ}")
    print(f"🔄  测试轮数: Warmup={WARMUP}, Runs={RUNS}")

    # ========== 消融实验对应的模型列表 ==========
    models = [
        "yolov8n.yaml",                  # Baseline
        "yolov8n-p2.yaml",               # + P2 检测头
        "yolov8n-p2-bifpn.yaml",         # + P2 + BiFPN
        "yolov8n-p2-dcnv2.yaml",         # + P2 + DCNv2
        "yolov8n-p2-bifpn-dcnv2.yaml",   # + P2 + BiFPN + DCNv2 (完整)
    ]

    # ========== 执行模型 FPS 测试 ==========
    all_results = []
    for model_yaml in models:
        result = benchmark_model(
            model_yaml,
            imgsz=IMGSZ,
            warmup=WARMUP,
            runs=RUNS,
            device=DEVICE,
        )
        all_results.append(result)

    # ========== 打印汇总表 ==========
    print_summary_table(all_results)

    # ========== NMS 速度对比 ==========
    nms_results = benchmark_nms(num_boxes=5000, runs=100)

    # ========== 保存结果到文件 ==========
    import os
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"设备: {DEVICE}\n")
        if DEVICE == "cuda":
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"输入尺寸: {IMGSZ}x{IMGSZ}\n")
        f.write(f"测试轮数: Warmup={WARMUP}, Runs={RUNS}\n\n")

        f.write(f"{'='*90}\n")
        f.write(f"  模型 FPS 汇总对比表\n")
        f.write(f"{'='*90}\n")
        f.write(f"  {'模型':<40} {'参数量':>8} {'GFLOPs':>9} {'延迟(ms)':>10} {'FPS':>8} {'P95(ms)':>10}\n")
        f.write(f"  {'-'*85}\n")
        baseline_fps2 = None
        for r in all_results:
            if r is None:
                continue
            if baseline_fps2 is None:
                baseline_fps2 = r['fps']
                delta = ""
            else:
                change = ((r['fps'] - baseline_fps2) / baseline_fps2) * 100
                delta = f" ({change:+.1f}%)"
            gflops_str = f"{r['gflops']:.2f}G" if r.get("gflops") else "N/A"
            f.write(
                f"  {r['model']:<40} {r['params_M']:>6.2f}M "
                f"{gflops_str:>9} {r['avg_ms']:>8.2f}ms {r['fps']:>7.1f}{delta:<8} {r['p95_ms']:>8.2f}ms\n"
            )
        f.write(f"{'='*90}\n\n")

        if nms_results:
            f.write(f"NMS 速度对比 (候选框数量: 5000)\n")
            f.write(f"{'-'*40}\n")
            for method, ms in nms_results.items():
                f.write(f"  {method:<25} {ms:>12.2f} ms\n")
            f.write(f"{'-'*40}\n")

    print(f"\n📄 结果已保存至: {output_path}")
    print("\n✅ 所有测试完成！")
