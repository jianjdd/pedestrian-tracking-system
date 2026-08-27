# 训练与升级指南 (Training & Upgrade Guide)

本文档详细介绍了如何训练自定义的 YOLOv8 模型，以及如何将项目中的 DeepSORT 追踪器升级为更强大的 StrongSORT。

---

## 第一部分：YOLOv8 模型训练指南

本部分将指导您如何使用 Ultralytics 框架训练自己的 YOLOv8 模型。

### 1. 环境准备

确保您的环境中已安装 `ultralytics` 库。如果未安装，请运行：

```bash
pip install ultralytics
```

### 2. 数据集准备

YOLOv8 需要特定的数据集格式。您需要准备图片和对应的标签文件（txt格式）。

#### 2.1 目录结构
建议按以下结构组织您的数据集：

```text
datasets/
└── my_dataset/
    ├── train/
    │   ├── images/  # 训练集图片 (.jpg, .png)
    │   └── labels/  # 训练集标签 (.txt)
    ├── val/
    │   ├── images/  # 验证集图片
    │   └── labels/  # 验证集标签
    └── data.yaml    # 数据集配置文件
```

#### 2.2 标签格式
每个 `.txt` 文件对应一张图片，每一行代表一个目标：
```text
<class_id> <x_center> <y_center> <width> <height>
```
*   所有数值均为归一化坐标（0-1之间）。
*   `class_id` 从 0 开始。

#### 2.3 创建配置文件 (data.yaml)
在数据集根目录下创建 `data.yaml` 文件：

```yaml
path: ../datasets/my_dataset  # 数据集根目录 (绝对路径或相对路径)
train: train/images           # 训练集图片路径
val: val/images               # 验证集图片路径

# 类别数量
nc: 2

# 类别名称列表
names:
  0: person
  1: car
```

### 3. 开始训练

您可以使用命令行或 Python 脚本来启动训练。推荐使用命令行，简单快捷。

#### 命令行方式：

```bash
# 使用 yolov8n.pt 预训练模型开始训练
# data: 指定刚才创建的 data.yaml
# epochs: 训练轮数 (建议 100-300)
# imgsz: 图片大小 (默认 640)
# device: 使用 GPU (0) 或 CPU (cpu)

yolo detect train data=path/to/data.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0
```

#### Python 脚本方式：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')  # 加载预训练模型

# 训练模型
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    device=0,
    name='my_custom_model'
)
```

### 4. 模型导出与使用

训练完成后，权重文件通常保存在 `runs/detect/train/weights/` 目录下。
*   `best.pt`: 效果最好的模型权重。
*   `last.pt`: 最后一轮的模型权重。

**在本项目中使用：**
1.  将 `best.pt` 复制到本项目的 `models/` 目录下。
2.  运行程序，点击“加载模型”按钮，选择您的 `best.pt` 文件即可。

---

## 第二部分：升级 DeepSORT 为 StrongSORT

本项目默认使用的是轻量级的 DeepSORT 实现。如果您需要更强的抗遮挡能力和身份保持能力，可以按照以下步骤升级为 StrongSORT。

### 1. 安装依赖库

StrongSORT 的功能由开源库 `boxmot` 提供。请在终端运行：

```bash
pip install boxmot
```

### 2. 下载 ReID 权重文件

StrongSORT 依赖行人重识别（ReID）网络来提取特征。最常用的模型是 `osnet_x0_25_msmt17.pt`。

1.  **下载地址**：[OSNet 权重下载](https://github.com/mikel-brostrom/yolo_tracking/releases/download/v10.0/osnet_x0_25_msmt17.pt) (或者在 boxmot 官方仓库寻找链接)。
2.  **放置位置**：
    *   在项目根目录下创建一个 `weights` 文件夹。
    *   将下载好的 `osnet_x0_25_msmt17.pt` 放入 `weights/` 文件夹中。

### 3. 应用适配器代码

为了方便切换，我已经为您编写了一个适配器文件 `tracker_strongsort.py`。该文件已经创建在您的项目目录中。

它会自动封装 StrongSORT 的功能，使其接口与项目现有的代码完全兼容。

### 4. 启用 StrongSORT

您只需要修改 `detector.py` 中的**一行代码**即可启用。

**打开 `detector.py`，找到 `__init__` 方法：**

**修改前：**
```python
from tracker import DeepSORTTracker
# ...
class VideoDetector(QThread):
    def __init__(self):
        # ...
        self.tracker = DeepSORTTracker()  # <--- 修改这行
```

**修改后：**
```python
# 1. 在文件头部导入新的适配器
from tracker_strongsort import StrongSORTTrackerAdapter

# ...

class VideoDetector(QThread):
    def __init__(self):
        super().__init__()
        # ...
        # 2. 替换追踪器初始化
        # self.tracker = DeepSORTTracker() 
        self.tracker = StrongSORTTrackerAdapter() # <--- 使用新适配器
```

**还要记得修改 `process_frame` 或 `detect_and_track` 方法中的调用吗？**
**不需要！** `tracker_strongsort.py` 已经处理了所有的接口适配。
但有一点需要注意：StrongSORT 的 `update` 方法需要传入当前帧图像。

**修正 `detector.py` 的调用（必须步骤）：**

找到 `detector.py` 中的 `detect_and_track` 方法（约第 523 行）：

**修改前：**
```python
# 更新追踪器
tracks = self.tracker.update(detections)
```

**修改后：**
```python
# 更新追踪器 (传入 frame 参数，StrongSORT 需要它提取特征)
# DeepSORTTrackerAdapter 会忽略 frame 参数，所以这样写兼容两者
tracks = self.tracker.update(detections, frame=frame)
```

### 5. 常见问题

*   **报错 `No module named 'boxmot'`**：请检查是否成功运行了 `pip install boxmot`。
*   **报错找不到权重文件**：请确保 `osnet_x0_25_msmt17.pt` 位于项目根目录或 `weights/` 子目录下。
*   **速度变慢**：StrongSORT 比 DeepSORT 计算量更大，尤其是启用了 ReID 特征提取。如果显卡性能不足，FPS 可能会下降。可以尝试在 `StrongSORTTrackerAdapter` 中设置 `fp16=True` 来加速。
