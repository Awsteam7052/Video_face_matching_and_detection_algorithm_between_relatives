# 亲人视频检测系统 (Family Video Detection System)

基于深度学习的人脸识别系统，用于从视频中检测和识别亲人。系统使用 RetinaFace 进行人脸检测，ArcFace 进行人脸特征提取，并通过 FAISS 进行高效向量检索。

## 功能特点

- 📹 **视频预处理**：基于场景切换的智能抽帧，大幅减少计算量
- 🎯 **人脸检测**：使用 RetinaFace (通过 insightface) 进行高精度人脸检测
- 🧬 **特征提取**：使用 ArcFace 提取 512 维特征向量
- 🔍 **高效检索**：基于 FAISS 的近似最近邻 (ANN) 检索
- 📊 **时序平滑**：滑动窗口投票机制，提高识别稳定性
- 📤 **结构化输出**：生成 JSON 格式的检测结果

## 系统架构

```
系统分为四个主要阶段：

1. 构建亲人特征底库 (Gallery Construction)
   - 照片预处理与人脸对齐
   - 特征向量提取 (512-dim, L2 normalized)
   - FAISS 索引构建

2. 视频流预处理 (Video Preprocessing)
   - 基于场景切换的智能抽帧
   - 无效帧过滤

3. 目标检测与特征提取 (Detection & Extraction)
   - 视频帧的人脸检测
   - 视频人脸特征向量化

4. 比对、投票与结果输出 (Matching & Annotation)
   - FAISS 近似最近邻检索
   - 时序上下文过滤
   - 生成结构化标记数据 (JSON)
```

## 安装依赖

### 环境要求

- Python 3.12+
- CUDA 12.4+ (可选，用于 GPU 加速)

### 安装步骤

```bash
# 创建虚拟环境 (推荐)
conda create --name LeadersIdentify python=3.12.12
conda activate LeadersIdentify

# 安装依赖
pip install -r requirements.txt.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 依赖说明

**基础依赖：**
- numpy, scipy, scipy - 数值计算
- opencv-python-headless - 图像处理
- Pillow, albumentations, scikit-image - 图像增强

**深度学习框架：**
- PyTorch (推荐 CUDA 12.4 版本)
- insightface - 人脸分析库 (包含 RetinaFace 和 ArcFace)

**向量检索：**
- faiss - Facebook AI 向量检索库

**工具库：**
- scenedetect - 视频场景检测
- matplotlib, tqdm - 可视化和进度条
- PyYAML, easydict - 配置管理

## 项目结构

```
API_Leader_20260319_v2/
├── model/
│   ├── family_photos/          # 亲人照片目录 (用于构建底库)
│   │   └── Family_*.png
│   ├── weights/                # 模型权重和索引
│   │   ├── all_features.npy    # 所有特征向量
│   │   ├── family_faiss_index.bin  # FAISS 索引
│   │   └── id_mapping.pkl      # ID 映射表
│   ├── __init__.py
│   ├── config.py               # 配置文件
│   ├── face_detector.py        # 人脸检测模块
│   ├── feature_extractor.py    # 特征提取模块
│   ├── gallery_builder.py      # 底库构建模块
│   ├── matcher.py              # 匹配模块
│   ├── video_preprocessor.py   # 视频预处理模块
│   └── video_probe.py          # 视频探测模块
├── main.py                     # 主程序入口
├── requirements.txt.txt        # 依赖列表
└── 实验设计步骤大纲.md         # 实验设计文档
```

## 使用方法

### 1. 构建亲人特征底库

首先需要准备亲人的照片，放在 `model/family_photos/` 目录下，然后运行：

```bash
python main.py --mode build
```

或使用交互式菜单：
```bash
python main.py
# 选择 1. 构建亲人特征底库
```

**照片要求：**
- 格式：PNG, JPG, JPEG
- 每张照片包含一张清晰的人脸
- 文件名将作为该亲人的 ID (例如：`Family_0123456789.png`)

### 2. 处理视频文件

将待检测的视频文件放入 `videos/` 目录，然后运行：

```bash
python main.py --mode process
```

或使用交互式菜单：
```bash
python main.py
# 选择 2. 处理视频文件夹
```

### 3. 一键构建底库并处理视频

```bash
python main.py --mode batch
```

或使用交互式菜单：
```bash
python main.py
# 选择 3. 两者都执行
```

### 4. 交互式菜单模式

```bash
python main.py
```

菜单选项：
1. 构建亲人特征底库
2. 处理视频文件夹
3. 两者都执行
4. 退出

### 命令行参数

```bash
python main.py --mode batch --videos-dir ./my_videos --output-dir ./my_output
```

参数说明：
- `--mode`: 运行模式
  - `interactive` - 交互式菜单模式 (默认)
  - `batch` - 批量处理模式
  - `build` - 仅构建底库
  - `process` - 仅处理视频
- `--videos-dir`: 视频文件夹路径 (默认: `./videos`)
- `--output-dir`: 输出文件夹路径 (默认: `./output`)

## 配置说明

编辑 `model/config.py` 文件进行配置：

```python
class Config:
    DETECTION_THRESHOLD = 0.5      # 检测置信度阈值
    FACE_SIZE_THRESHOLD = 40       # 最小人脸尺寸
    BATCH_SIZE = 16                # 批处理大小
    FEATURE_DIM = 512              # 特征维度
    
    FAMILY_PHOTOS_DIR = r'./model/family_photos'  # 亲人照片目录
    VIDEOS_DIR = r'./videos'       # 视频目录
    OUTPUT_DIR = r'./output'       # 输出目录
    
    INDEX_PATH = r'./model/weights/family_faiss_index.bin'      # FAISS 索引路径
    ID_MAPPING_PATH = r'./model/weights/id_mapping.pkl'         # ID 映射路径
    FEATURES_PATH = r'./model/weights/all_features.npy'         # 特征文件路径
```

## 输出结果

系统会为每个视频生成 JSON 格式的结果文件，位于 `output/{video_name}/{video_name}.json`。

### JSON 输出格式

```json
{
  "total_frames": 150,
  "matching_threshold": 0.45,
  "temporal_smoothing": {
    "enabled": true,
    "window_size": 3,
    "vote_threshold": 2
  },
  "frames": [
    {
      "frame_index": 0,
      "frame_path": "output/video1/frames/scene_0001_frame_000.jpg",
      "face_count": 2,
      "detected_faces": [
        {
          "bbox": [100, 150, 200, 250],
          "matched_id": "Family_0123456789",
          "similarity": 0.68,
          "det_score": 0.95,
          "is_match": true,
          "confidence_level": "high",
          "window_votes": 3,
          "avg_similarity": 0.72,
          "smoothed": true
        },
        {
          "bbox": [300, 200, 400, 300],
          "matched_id": "Unknown",
          "similarity": 0.32,
          "det_score": 0.88,
          "is_match": false
        }
      ]
    }
  ]
}
```

### 字段说明

- `matched_id`: 匹配的亲人 ID (或 "Unknown")
- `similarity`: 相似度得分 (0-1，越高越相似)
- `is_match`: 是否为有效匹配
- `confidence_level`: 置信度等级 (high/medium/low)
  - high: ≥ 0.6
  - medium: ≥ 0.5
  - low: < 0.5
- `window_votes`: 时序平滑投票数
- `avg_similarity`: 平滑后的平均相似度
- `smoothed`: 是否经过时序平滑

## 核心模块说明

### 1. FaceDetector ([face_detector.py](file:///d:/Tools/API_Leader_20260319_v2/model/face_detector.py))
- 基于 RetinaFace 的人脸检测
- 支持置信度过滤和尺寸过滤
- 提供批量处理能力

### 2. FeatureExtractor ([feature_extractor.py](file:///d:/Tools/API_Leader_20260319_v2/model/feature_extractor.py))
- 基于 ArcFace 的特征提取
- L2 归一化输出
- 支持指定人脸区域提取

### 3. FamilyGalleryBuilder ([gallery_builder.py](file:///d:/Tools/API_Leader_20260319_v2/model/gallery_builder.py))
- 构建亲人特征底库
- FAISS IndexFlatIP 索引
- 自动维护 ID 映射

### 4. VideoPreprocessor ([video_preprocessor.py](file:///d:/Tools/API_Leader_20260319_v2/model/video_preprocessor.py))
- 基于场景切换的智能抽帧
- 1 FPS 关键帧抽取
- 场景信息记录

### 5. VideoFaceProbe ([video_probe.py](file:///d:/Tools/API_Leader_20260319_v2/model/video_probe.py))
- 批量视频帧探测
- 检测 + 特征提取 + 匹配一体化
- 支持批量处理

### 6. FaceMatcher ([matcher.py](file:///d:/Tools/API_Leader_20260319_v2/model/matcher.py))
- FAISS 近似最近邻检索
- 时序平滑 (滑动窗口投票)
- 结果输出和统计

## 性能优化建议

1. **GPU 加速**：安装 CUDA 版本的 PyTorch 和 insightface 可大幅提升速度
2. **批量处理**：系统默认使用 16 张图片批量处理，可根据显存调整
3. **场景检测**：智能抽帧可将 10 分钟视频从 18000 帧压缩到约 600 帧
4. **底库优化**：亲人数量较多时，考虑使用 FAISS 的索引优化选项

## 注意事项

1. **首次运行**：系统会自动下载 insightface 的 buffalo_l 模型权重 (~100MB)
2. **照片质量**：使用清晰、正面、光线充足的照片效果最佳
3. **相似度阈值**：默认阈值 0.45 可根据实际效果调整
4. **视频删除**：匹配成功后，系统会自动删除原视频文件 (可配置)
5. **临时文件**：处理完成后会自动清理临时帧文件

## 常见问题

**Q: 如何调整识别阈值？**
A: 修改 `model/config.py` 中的 `DETECTION_THRESHOLD` 参数，或在 `FaceMatcher` 初始化时传入 `threshold` 参数。

**Q: 支持哪些视频格式？**
A: 支持 MP4, AVI, MOV, MKV, WMV 等常见格式。

**Q: 可以同时识别多个人吗？**
A: 可以，系统会检测视频中所有出现的人脸，并分别进行匹配。

**Q: 如何提高识别准确率？**
A: 
- 使用更高质量的亲人照片
- 调整相似度阈值
- 增加时序平滑的窗口大小
- 确保亲人照片和视频中的人脸质量相近

## 技术栈

- **人脸检测**: RetinaFace (insightface)
- **人脸识别**: ArcFace (insightface)
- **向量检索**: FAISS
- **视频处理**: OpenCV, scenedetect
- **深度学习**: PyTorch
- **语言**: Python 3.12+

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
