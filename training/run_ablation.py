import sys
import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ultralytics_path = PROJECT_ROOT / 'ultralytics'
if str(ultralytics_path) not in sys.path:
    sys.path.insert(0, str(ultralytics_path))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"导入 Ultralytics 失败，请检查路径: {e}")
    sys.exit(1)


def save_metrics_to_csv(metrics, save_dir, filename="metrics_summary.csv"):
    if not hasattr(metrics, 'results_dict'):
        print("⚠️ 无法获取 metrics.results_dict，跳过保存。")
        return
        
    results = metrics.results_dict
    csv_path = Path(save_dir) / filename
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for k, v in results.items():
            formatted_val = f"{v:.5f}" if isinstance(v, float) else v
            writer.writerow([k, formatted_val])
            
    print(f"📊 数值指标已成功提取并保存至: {csv_path}")


def run_double_val(model, data, device, name_prefix):
    print(f"\n--- 正在执行 Standard NMS 验证 ({name_prefix}) ---")
    metrics_std = model.val(data=data, soft_nms=False, name=f"{name_prefix}_StdNMS", device=device, save_json=True)
    if hasattr(metrics_std, 'save_dir'):
        save_metrics_to_csv(metrics_std, metrics_std.save_dir, "Standard_NMS_Metrics.csv")
    
    print(f"\n--- 正在执行 Soft-NMS 验证 ({name_prefix}) ---")
    metrics_soft = model.val(data=data, soft_nms=True, name=f"{name_prefix}_SoftNMS", device=device, save_json=True)
    if hasattr(metrics_soft, 'save_dir'):
        save_metrics_to_csv(metrics_soft, metrics_soft.save_dir, "Soft_NMS_Metrics.csv")


def run_standard_val(model, data, device, name_prefix):
    print(f"\n--- 正在执行 Standard NMS 验证 ({name_prefix}) ---")
    metrics_std = model.val(data=data, soft_nms=False, name=f"{name_prefix}_StdNMS", device=device, save_json=True)
    if hasattr(metrics_std, 'save_dir'):
        save_metrics_to_csv(metrics_std, metrics_std.save_dir, "Standard_NMS_Metrics.csv")


def _should_run(exp, target):
    """exp 可能是 int、'all' 或 list[int]，判断 target 是否应执行"""
    if exp == 'all':
        return True
    if isinstance(exp, list):
        return target in exp
    return exp == target


def run_experiment(args):
    data = args.data
    device = args.device
    epochs = args.epochs
    imgsz = args.imgsz
    exp = args.exp
    save_path = args.save_path

    print(f"\n🚀 准备执行实验 | 数据集: {data} | 设备: GPU {device} | Yolov8n 规模 | 从头训练\n")

    # ============================
    # Exp 0: Baseline
    # ============================
    if _should_run(exp, 0):
        print("="*50)
        print("▶ Exp 0: Baseline")
        print("="*50)
        model = YOLO("ultralytics/cfg/models/v8/yolov8n.yaml")
        model.train(data=data, epochs=epochs, imgsz=imgsz, name="exp0_baseline", device=device, patience=10, project=save_path)
        run_standard_val(model, data, device, "exp0_baseline_final")

    # ============================
    # Exp 1: Baseline + P2
    # ============================
    if _should_run(exp, 1):
        print("="*50)
        print("▶ Exp 1: Baseline + P2")
        print("="*50)
        model = YOLO("ultralytics/cfg/models/v8/yolov8n-p2.yaml")
        model.train(data=data, epochs=epochs, imgsz=imgsz, name="exp1_p2", device=device, patience=10, project=save_path)
        run_standard_val(model, data, device, "exp1_p2_final")

    # ============================
    # Exp 2: Baseline + P2 + DCNv2
    # ============================
    if _should_run(exp, 2):
        print("="*50)
        print("▶ Exp 2: Baseline + P2 + DCNv2")
        print("="*50)
        model = YOLO("ultralytics/cfg/models/v8/yolov8n-p2-dcnv2.yaml")
        model.train(data=data, epochs=epochs, imgsz=imgsz, name="exp2_p2_dcnv2", device=device, patience=10, project=save_path)
        run_standard_val(model, data, device, "exp2_p2_dcnv2_final")

    # ============================
    # Exp 3: Baseline + P2 + BiFPN
    # ============================
    if _should_run(exp, 3):
        print("="*50)
        print("▶ Exp 3: Baseline + P2 + BiFPN")
        print("="*50)
        model = YOLO("ultralytics/cfg/models/v8/yolov8n-p2-bifpn.yaml")
        model.train(data=data, epochs=epochs, imgsz=imgsz, name="exp3_p2_bifpn", device=device, patience=10, project=save_path)
        run_standard_val(model, data, device, "exp3_p2_bifpn_final")

    # ============================
    # Exp 4: 完整架构 (P2 + BiFPN + DCNv2)
    # ============================
    if _should_run(exp, 4):
        print("="*50)
        print("▶ Exp 4: 完整架构 (P2 + BiFPN + DCNv2)")
        print("="*50)
        model = YOLO("ultralytics/cfg/models/v8/yolov8n-p2-bifpn-dcnv2.yaml")
        model.train(data=data, epochs=epochs, imgsz=imgsz, name="exp4_full", device=device, patience=10, project=save_path)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🚀 密集行人检测消融实验启动脚本 (Linux)")
    parser.add_argument('--data', type=str, default="/home/featurize/data/CrowdHuman_YOLO/data.yaml", help="数据集 .yaml 文件的绝对路径")
    parser.add_argument('--exp', type=str, default='all', help="实验编号: 0-4 单个实验, 'all' 全部, 或逗号分隔如 '0,4'")
    parser.add_argument('--device', type=str, default="0", help="GPU 编号，例如 '0' 或 '0,1'")
    parser.add_argument('--epochs', type=int, default=100, help="训练轮数 (默认: 100)")
    parser.add_argument('--imgsz', type=int, default=640, help="输入图像尺寸 (默认: 640)")
    parser.add_argument('--save_path', type=str, default="/home/featurize/work/pedestrain_tracking/runs/all_first_end", help="训练结果保存根目录")
    
    args = parser.parse_args()

    if args.exp == 'all':
        pass
    elif args.exp.isdigit():
        args.exp = int(args.exp)
    elif ',' in args.exp:
        try:
            args.exp = [int(x.strip()) for x in args.exp.split(',')]
            for e in args.exp:
                if not 0 <= e <= 4:
                    raise ValueError
        except ValueError:
            print("❌ 错误: --exp 逗号分隔时每个值必须是 0-4 之间的数字")
            sys.exit(1)
    else:
        print("❌ 错误: --exp 必须是 0-4 的数字、'all'、或逗号分隔如 '0,4'")
        sys.exit(1)

    run_experiment(args)