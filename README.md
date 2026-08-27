# 🎯 行人追踪计数系统

基于 YOLOv8 + DeepSORT 的行人检测与追踪系统，支持实时视频流和离线视频分析，提供完整的 Web 可视化界面。

---

## ✨ 功能特点

- **实时检测与追踪**：YOLOv8 目标检测 + DeepSORT/BoTSORT 多目标追踪
- **双向计数**：支持 A→B 和 B→A 方向的人流统计
- **多类别支持**：可配置检测类别（行人、车辆等）
- **实时可视化**：MJPEG 视频流 + Canvas 绘制计数线
- **离线分析**：快速分析模式，全速处理历史视频
- **统计导出**：CSV 日志 + 折线图可视化
- **预设配置**：标准/拥挤/稀疏场景一键切换
- **前后端分离**：FastAPI 后端 + Vite 前端

---

## 🏗️ 技术栈

| 层级 | 技术 |
|------|------|
| **后端** | FastAPI + Uvicorn |
| **前端** | Vite + Vanilla JS + Chart.js |
| **检测** | YOLOv8 (Ultralytics) |
| **追踪** | DeepSORT / BoTSORT / StrongSORT |
| **数据库** | MySQL + SQLAlchemy 2.0 + Alembic |
| **图像处理** | OpenCV + NumPy |
| **可视化** | Matplotlib + Chart.js |

---

## 📁 项目结构

```
pedestrian-tracking-system/
├── app/                          # 后端 FastAPI 应用
│   ├── api/                      # API 路由
│   │   ├── detection.py          # 检测控制
│   │   ├── tracking.py           # 追踪控制
│   │   ├── video.py              # 视频源管理
│   │   ├── settings.py           # 参数配置
│   │   └── logs.py               # 日志管理
│   ├── core/                     # 核心模块
│   │   ├── algorithm/            # 算法实现
│   │   │   ├── detector.py       # 视频检测器
│   │   │   ├── tracker.py        # DeepSORT 追踪器
│   │   │   └── counter.py        # 过线计数器
│   │   └── config.py             # 配置管理
│   ├── db/                       # 数据库
│   │   ├── session.py            # 会话管理
│   │   └── models.py             # ORM 模型
│   ├── schemas/                  # Pydantic 数据模型
│   └── services/                 # 业务逻辑层
│       └── camera_manager.py     # 相机管理器
│
├── frontend/                     # 前端 Vite 应用
│   ├── src/
│   │   ├── api/                  # API 客户端
│   │   ├── state/                # 状态管理
│   │   ├── ui/                   # UI 组件
│   │   └── styles/               # 样式
│   └── index.html
│
├── data/                         # 数据目录
│   ├── uploads/                  # 上传文件（模型、视频）
│   └── logs/                     # 日志输出
│
├── migrations/                   # 数据库迁移（Alembic）
├── training/                     # 模型训练脚本
├── ultralytics/                  # YOLOv8 源码（git submodule，含自定义模块）
├── run.py                        # 统一启动脚本
├── requirements.txt              # Python 依赖
└── alembic.ini                   # Alembic 配置
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Node.js 16+
- MySQL 5.7+
- （可选）NVIDIA GPU + CUDA

### 1. 克隆项目

```bash
git clone --recurse-submodules https://github.com/jianjdd/pedestrian-tracking-system.git
cd pedestrian-tracking-system
```

> 如果已克隆但未初始化子模块，运行：`git submodule update --init --recursive`

### 2. 配置后端

```bash
# 复制环境变量文件
cp .env.example .env

# 编辑 .env，修改数据库连接
# DATABASE_URL=mysql+pymysql://root:password@localhost/pedestrian_tracking
```

### 3. 安装后端依赖

```bash
# 推荐使用 conda 或 venv
conda create -n tracker python=3.10
conda activate tracker

pip install -r requirements.txt
```

### 4. 配置前端

```bash
cd frontend
cp .env.example .env
npm install
cd ..
```

### 5. 初始化数据库

```bash
# 创建数据库
mysql -u root -p -e "CREATE DATABASE pedestrian_tracking CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 运行迁移
alembic upgrade head
```

### 6. 启动服务

```bash
# 一键启动前后端
python run.py
```

启动后访问：

| 服务 | 地址 |
|------|------|
| 前端页面 | http://127.0.0.1:5173 |
| 后端 API | http://127.0.0.1:5000 |
| API 文档 | http://127.0.0.1:5000/docs |

---

## 📖 使用指南

### 1. 加载模型

- 上传 `.pt` 模型文件，或选择本地已有模型
- 支持 YOLOv8 系列模型（n/s/m/l/x）

### 2. 选择视频源

- **摄像头**：自动检测可用摄像头（0, 1, 2...）
- **视频文件**：上传 MP4/AVI 等格式视频

### 3. 设置计数线

- 点击「绘制」按钮，在视频画面上拖拽绘制计数线
- 系统自动统计跨越该线的目标数量

### 4. 开始检测

- 点击「开始」启动实时检测
- 实时查看 A→B 和 B→A 方向的计数

### 5. 导出数据

- 点击「保存日志」导出 CSV 和折线图
- 日志保存在 `data/logs/` 目录

---

## ⚙️ 配置说明

### 后端配置 (.env)

```env
DATABASE_URL=mysql+pymysql://root:password@localhost/pedestrian_tracking
BACKEND_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

### 前端配置 (frontend/.env)

```env
VITE_API_BASE_URL=http://127.0.0.1:5000/api
```

### 检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| confidence | 0.6 | 检测置信度阈值 |
| max_age | 5 | 追踪器最大丢失帧数 |
| min_hits | 4 | 确认追踪的最小命中数 |
| iou_threshold | 0.3 | NMS IoU 阈值 |

---

## 🏃 运行模式

### 实时监测

- 支持摄像头和视频文件
- 实时 MJPEG 视频流
- 可暂停/恢复/停止
- 适用于在线监控场景

### 快速分析

- 仅支持视频文件
- 全速处理，无画面显示
- 适用于离线批量分析

---

## 📊 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/video/cameras` | 检测可用摄像头 |
| POST | `/api/video/source` | 设置视频源 |
| POST | `/api/video/model` | 加载模型 |
| GET | `/api/detection/stats` | 获取实时统计 |
| POST | `/api/detection/start` | 开始检测 |
| POST | `/api/detection/stop` | 停止检测 |
| POST | `/api/tracking/line` | 设置计数线 |
| GET | `/api/settings` | 获取配置 |
| POST | `/api/settings` | 更新配置 |
| GET | `/api/logs/list` | 列出日志 |
| POST | `/api/analysis/start` | 开始快速分析 |

完整文档：http://127.0.0.1:5000/docs

---

## 🧪 模型训练

项目包含完整的训练脚本，支持 CrowdHuman、MOT17、MOT20 等数据集：

```bash
cd training

# 训练密集行人检测模型
python train_dense.py

# 运行消融实验
python run_ablation.py

# 追踪评估
python track_botsort.py
python track_strongsort.py
```

---

## 🐛 常见问题

### 摄像头画面不动

- 关闭其他使用摄像头的程序
- 检查 Windows 隐私设置
- 尝试使用 `cv2.CAP_MSMF` 替代 `cv2.CAP_DSHOW`

### 摄像头全黑

- 检查摄像头是否被正确识别
- 增加预热帧数
- 尝试其他摄像头索引

### 数据库连接失败

```bash
# 检查 MySQL 服务
mysql -u root -p

# 确认数据库存在
SHOW DATABASES;
```

### 端口被占用

```bash
# 修改后端端口
# 编辑 run.py，修改 port=5000 为其他端口

# 修改前端端口
# 编辑 frontend/vite.config.js
```

---

## 📝 开发指南

### 添加新 API

1. 在 `app/api/` 创建路由文件
2. 在 `app/schemas/` 定义请求/响应模型
3. 在 `app/services/` 实现业务逻辑
4. 在 `app/api/api_router.py` 注册路由

### 添加新追踪器

1. 在 `app/core/algorithm/` 创建追踪器文件
2. 继承 `DeepSORTTracker` 基类
3. 在 `detector.py` 中注册新追踪器

---

## 📄 许可证

本项目基于 AGPL-3.0 许可证开源。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题或建议，请提交 Issue。
