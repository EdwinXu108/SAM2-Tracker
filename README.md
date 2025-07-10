# SAM2 视频跟踪项目

基于 Meta AI 的 SAM2 (Segment Anything Model 2) 实现的视频目标跟踪与分割工具。通过简单的初始标注即可实现复杂视频场景下的精准对象追踪。

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/EdwinXu108/SAM2-Tracker.git
cd SAM2-Tracker

# 创建虚拟环境
conda create -n sam2-env python=3.9 -y
conda activate sam2-env

# 安装依赖
pip install -e .
```

### 2. 下载模型

```bash
cd checkpoints/
bash download_ckpts.sh
cd ..
```

> 💡 如下载失败，请手动下载 `sam2.1_hiera_large.pt` 等模型权重至 `checkpoints/` 目录

### 3. 配置参数

编辑 `sam2_tracker.py` 中的配置：

```python
CONFIG = {
    "input_path": "your_video.mp4",           # 输入视频或帧目录
    "output_dir": "output/",                  # 输出目录
    "checkpoint_path": "checkpoints/sam2.1_hiera_large.pt",
    "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "max_frames": None,                       # 最大处理帧数
    "show_results": True,                     # 是否显示结果
}

ANNOTATIONS = [
    # 框标注: (帧索引, 对象ID, "box", [x1,y1,x2,y2], None)
    (0, 14, "box", [491, 241, 537, 357], None),
    
    # 点标注: (帧索引, 对象ID, "points", [[x,y],...], [label,...])
    # (5, 2, "points", [[100, 150], [110, 160]], [1, 0]),
]
```

### 4. 运行追踪

```bash
python sam2_tracker.py
```

## 📝 标注说明

### 框标注
- 格式：`(frame_idx, obj_id, "box", [x1, y1, x2, y2], None)`
- `[x1, y1]`：左上角坐标
- `[x2, y2]`：右下角坐标

### 点标注
- 格式：`(frame_idx, obj_id, "points", [[x1, y1], ...], [label1, ...])`
- `[[x1, y1], ...]`：点坐标列表
- `[label1, ...]`：标签列表（1: 前景点, 0: 背景点）

## 📁 项目结构

```
SAM2-Tracker/
├── checkpoints/        # 模型权重
├── configs/            # 配置文件
├── sam2_tracker.py     # 主执行脚本
├── setup.py            # 安装脚本
└── README.md           # 本文档
```

## 🎥 [Demo](https://drive.google.com/drive/folders/12YVVAoiqxdQou9oVAmQQ6Bn7H7yYkdII?usp=sharing)

