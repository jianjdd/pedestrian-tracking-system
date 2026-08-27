"""
YOLOv8 密集行人检测 - 专用训练脚本

本脚本针对密集人群场景进行了专门优化:
  1. 大分辨率输入 (imgsz=1280) - 保留小目标特征
  2. 增强的数据增强策略 - Mosaic + Mixup 强化遮挡学习
  3. 优化的超参数 - 适合微调场景的学习率和正则化配置
  4. NMS IoU 阈值调高 - 减少密集框被"吞并"

使用方法:
  # 使用 CrowdHuman 训练
  python train_dense.py --data E:/datasets/CrowdHuman_yolo/data.yaml

  # 使用 MOT17 训练  
  python train_dense.py --data E:/datasets/MOT17_yolo/data.yaml

  # 指定模型规模和轮数
  python train_dense.py --data data.yaml --model yolov8s.pt --epochs 100

  # 使用默认配置模板（不传 --data 则使用内置的 data_crowdhuman.yaml）
  python train_dense.py
"""
import argparse
import os
import sys

# 将本地 ultralytics 源码仓库加入搜索路径
_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ultralytics')
_repo_root = os.path.normpath(_repo_root)
if os.path.isdir(os.path.join(_repo_root, 'ultralytics')):
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    if 'ultralytics' in sys.modules:
        del sys.modules['ultralytics']

from ultralytics import YOLO


def train(args):
    """执行训练"""
    # 1. 加载模型
    print(f"📦 加载模型: {args.model}")
    model = YOLO(args.model)

    # 2. 确定数据集配置
    data_path = args.data
    if not data_path:
        # 尝试使用同目录下的默认 yaml
        default_yaml = os.path.join(os.path.dirname(__file__), 'data_crowdhuman.yaml')
        if os.path.exists(default_yaml):
            data_path = default_yaml
        else:
            print("❌ 请通过 --data 指定 data.yaml 路径")
            return

    print(f"📂 数据集配置: {data_path}")
    # 3. 训练参数（针对密集行人场景优化 + 速度优化）
    train_params = {
        'data': data_path,
        'epochs': args.epochs,
        'imgsz': args.imgsz,        # 👈 建议运行命令时加上 --imgsz 960
        'batch': args.batch,
        'device': args.device,
        
        # —— 🏎️ 速度优化核心 ——
        'amp': True,                # 开启混合精度 (极速提升，节省显存)
        'cache': True,              # 将图片读进内存 (极大减少磁盘IO，前提是内存足够大)
        'workers': 8,               # 多线程拉取数据 (让GPU不空闲，默认4太少)
        
        # —— 学习率（微调用小学习率） ——
        'lr0': 0.001,           
        'lrf': 0.01,            
        'optimizer': 'AdamW',   
        'weight_decay': 0.0005,

        # —— 数据增强（强化遮挡和重叠场景学习） ——
        'mosaic': 1.0,          
        'mixup': 0.15,          
        'close_mosaic': 10,     # 👈 改成 10，让最后10轮飞速完成并收敛
        # ... 以下保持原来的图像增强参数 ...
        'degrees': 5.0,         
        'translate': 0.1,       
        'scale': 0.5,           
        'fliplr': 0.5,          
        'hsv_h': 0.015,         
        'hsv_s': 0.7,           
        'hsv_v': 0.4,           

        # —— NMS 和检测相关 ——
        'iou': 0.75,            
        'max_det': 300,         

        # —— 保存和日志 ——
        'project': os.path.join(os.path.dirname(__file__), '..', 'runs', 'train_dense'),
        'name': args.name,
        'save_period': 10,      
        'plots': True,          
        'verbose': True,
    }


    print("\n🚀 开始训练，关键参数:")
    print(f"   模型: {args.model}")
    print(f"   分辨率: {args.imgsz}")
    print(f"   批次: {args.batch}")
    print(f"   轮次: {args.epochs}")
    print(f"   NMS IoU: 0.75 (密集模式)")
    print(f"   max_det: 300")
    print()

    # 4. 开始训练
    results = model.train(**train_params)

    print("\n✅ 训练完成!")
    print(f"   最佳权重: {results.save_dir}/weights/best.pt")
    print(f"   将 best.pt 复制到项目 weights/ 目录即可在 Web 界面中使用")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 密集行人检测训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # CrowdHuman 训练（推荐第一步）
  python train_dense.py --data E:/datasets/CrowdHuman_yolo/data.yaml --model yolov8s.pt --epochs 80

  # MOT17 微调（在 CrowdHuman 训练后的权重上继续微调）
  python train_dense.py --data E:/datasets/MOT17_yolo/data.yaml --model runs/train_dense/crowdhuman/weights/best.pt --epochs 50

  # 小显存（6GB）配置
  python train_dense.py --data data.yaml --batch 4 --imgsz 960

  # 大显存（24GB）配置
  python train_dense.py --data data.yaml --model yolov8m.pt --batch 16 --imgsz 1280
"""
    )
    parser.add_argument('--data', type=str, default=None,
                        help='data.yaml 路径')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='预训练权重 (默认 yolov8s.pt)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='训练轮数 (默认 80)')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='输入分辨率 (默认 1280，密集场景建议 960-1280)')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小 (根据显存调整，默认 8)')
    parser.add_argument('--device', type=str, default='0',
                        help='设备 (0=GPU, cpu=CPU)')
    parser.add_argument('--name', type=str, default='dense_pedestrian',
                        help='实验名称')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载线程数')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
