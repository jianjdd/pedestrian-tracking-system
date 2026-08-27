import os
import sys

# 将工程目录加入搜索路径
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from app.core.algorithm.detector import VideoDetector

def main():
    print("--- TensorRT 预导出程序启动 ---")
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用。请确保已安装 GPU 版 PyTorch 和驱动。")
        return
    
    print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    
    detector = VideoDetector()
    
    # 默认权重路径
    model_path = os.path.join(repo_root, "weights", "yolov8n.pt")
    if not os.path.exists(model_path):
        print(f"错误: 找不到权重文件 {model_path}")
        return
    
    print(f"开始加载/转换模型 (大约需要 3-5 分钟): {model_path}")
    success = detector.load_model(model_path)
    
    if success:
        print("--- 转换成功！已经在 weights 目录下生成 .engine 文件 ---")
        # 验证加载的是不是 .engine
        if hasattr(detector.model, 'ckpt_path') and detector.model.ckpt_path.endswith('.engine'):
             print(f"当前加载模型: {detector.model.ckpt_path}")
        else:
             # 对于 engine，ultralytics 可能处理方式略有不同
             print("模型已成功加载。")
    else:
        print("--- 转换过程中出现错误，请检查日志 ---")

if __name__ == "__main__":
    main()
