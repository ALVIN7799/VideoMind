# 🎥 本地视频处理系统使用指南

## 概述

本地视频处理系统提供了不依赖VideoDB的完整视频处理和索引功能，包括：

- ✅ **语音转文字** - 使用OpenAI Whisper模型
- ✅ **场景检测** - 使用PySceneDetect自动分割视频场景
- ✅ **帧提取** - 使用OpenCV提取关键帧
- ✅ **本地存储** - SQLite数据库管理视频索引
- ✅ **文本搜索** - 在转录文本中快速搜索
- ✅ **完全本地化** - 数据不离开本地环境

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基本依赖
pip install opencv-python whisper-openai scenedetect[opencv]

# 音频处理
pip install librosa soundfile

# 视频处理（需要FFmpeg）
pip install ffmpeg-python

# 确保安装FFmpeg
# Windows: choco install ffmpeg
# macOS: brew install ffmpeg  
# Ubuntu: sudo apt install ffmpeg
```

### 2. 基本使用

```python
from director.tools.local_video_processor import LocalVideoProcessor

# 初始化处理器
processor = LocalVideoProcessor(storage_path="./my_videos")

# 上传视频
result = processor.upload_video("path/to/video.mp4", "my_video_001")

# 转录视频
transcript = processor.transcribe_video("my_video_001")

# 检测场景
scenes = processor.detect_scenes("my_video_001")

# 搜索转录
results = processor.search_transcript("my_video_001", "关键词")
```

### 3. 使用Agent接口

```python
from director.agents.local_video_agent import LocalVideoAgent
from director.core.session import Session
from director.db.sqlite.db import SqliteDB

# 创建会话
db = SqliteDB(":memory:")
session = Session(db=db)

# 创建Agent
agent = LocalVideoAgent(session=session)

# 上传视频
response = agent.run(
    action="upload",
    video_path="/path/to/video.mp4",
    video_id="test_video"
)

# 转录视频
response = agent.run(
    action="transcribe", 
    video_id="test_video"
)

# 检测场景
response = agent.run(
    action="detect_scenes",
    video_id="test_video",
    threshold=25.0  # 调整敏感度
)

# 搜索转录
response = agent.run(
    action="search",
    video_id="test_video", 
    query="重要内容"
)
```

## 📁 文件结构

```
local_video_storage/
├── videos/           # 存储转换后的视频文件
├── frames/           # 存储提取的关键帧
├── transcripts/      # 存储完整转录JSON文件
├── scenes/           # 存储场景数据
└── video_index.db    # SQLite数据库
```

## 🔧 配置选项

### Whisper模型选择

```python
# 可用模型（按准确度和速度排序）
models = ["tiny", "base", "small", "medium", "large"]

# 在LocalVideoProcessor中修改
self.whisper_model = whisper.load_model("medium")  # 更高准确度
```

### 场景检测调优

```python
# 检测敏感度调整
processor.detect_scenes(video_id, threshold=30.0)  # 不太敏感
processor.detect_scenes(video_id, threshold=20.0)  # 更敏感
```

### 语言设置

```python
# 在transcribe_video中修改语言
result = self.whisper_model.transcribe(
    audio_path, 
    language="en"  # 英文
    # language="zh"  # 中文
    # language=None  # 自动检测
)
```

## 🎯 实际应用案例

### 案例1：教育视频分析

```python
# 处理在线课程视频
processor = LocalVideoProcessor("./course_videos")

# 批量处理多个视频
course_videos = ["lecture1.mp4", "lecture2.mp4", "lecture3.mp4"]

for i, video_path in enumerate(course_videos):
    video_id = f"lecture_{i+1}"
    
    # 上传和转录
    processor.upload_video(video_path, video_id)
    processor.transcribe_video(video_id)
    processor.detect_scenes(video_id)
    
    # 搜索特定主题
    results = processor.search_transcript(video_id, "机器学习")
    print(f"在第{i+1}讲中找到{len(results)}个相关片段")
```

### 案例2：会议录音分析

```python
# 处理会议录音
processor = LocalVideoProcessor("./meeting_records")

# 上传会议视频
processor.upload_video("meeting_2024_01_15.mp4", "meeting_20240115")

# 转录会议内容
transcript = processor.transcribe_video("meeting_20240115")

# 搜索关键决策点
decisions = processor.search_transcript("meeting_20240115", "决定")
action_items = processor.search_transcript("meeting_20240115", "行动项")

print("会议决策点：")
for item in decisions:
    print(f"  {item['start_time']:.1f}s: {item['text']}")
```

### 案例3：媒体内容索引

```python
# 处理媒体内容
processor = LocalVideoProcessor("./media_library")

def process_media_file(file_path):
    video_id = Path(file_path).stem
    
    # 完整处理流程
    upload_result = processor.upload_video(file_path, video_id)
    if upload_result["success"]:
        # 转录
        processor.transcribe_video(video_id)
        
        # 场景检测
        scenes = processor.detect_scenes(video_id, threshold=25.0)
        
        # 返回处理结果
        return {
            "video_id": video_id,
            "duration": upload_result["video_info"]["duration"],
            "scenes_count": scenes["total_scenes"] if scenes["success"] else 0
        }

# 批量处理
results = []
for video_file in Path("./raw_videos").glob("*.mp4"):
    result = process_media_file(str(video_file))
    results.append(result)

print(f"处理了{len(results)}个视频文件")
```

## 🔍 高级搜索功能

### 时间范围搜索

```python
def search_in_timerange(processor, video_id, query, start_time, end_time):
    """在指定时间范围内搜索"""
    all_results = processor.search_transcript(video_id, query)
    
    filtered_results = [
        r for r in all_results 
        if start_time <= r["start_time"] <= end_time
    ]
    
    return filtered_results

# 只在前5分钟搜索
results = search_in_timerange(processor, "video_001", "介绍", 0, 300)
```

### 批量搜索

```python
def search_multiple_videos(processor, video_ids, query):
    """在多个视频中搜索"""
    all_results = {}
    
    for video_id in video_ids:
        results = processor.search_transcript(video_id, query)
        if results:
            all_results[video_id] = results
    
    return all_results

# 在所有视频中搜索关键词
results = search_multiple_videos(
    processor, 
    ["video_001", "video_002", "video_003"], 
    "重要信息"
)
```

## 🛠️ 故障排除

### 常见问题

**1. FFmpeg未安装**
```bash
# 错误：FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
# 解决：安装FFmpeg
pip install ffmpeg-python
# 并确保系统PATH中有ffmpeg可执行文件
```

**2. Whisper模型下载失败**
```python
# 错误：URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
# 解决：手动下载模型
import whisper
whisper.load_model("base", download_root="./models")
```

**3. OpenCV安装问题**
```bash
# 错误：ImportError: No module named 'cv2'
# 解决：重新安装OpenCV
pip uninstall opencv-python
pip install opencv-python==4.10.0.84
```

**4. 内存不足**
```python
# 对于大视频文件，可以调整处理参数
processor = LocalVideoProcessor()
# 使用更小的Whisper模型
processor.whisper_model = whisper.load_model("tiny")
```

### 性能优化

**1. 选择合适的Whisper模型**
- `tiny`: 最快，准确度较低
- `base`: 平衡选择
- `small`: 较好准确度
- `medium`: 高准确度
- `large`: 最高准确度，最慢

**2. 场景检测优化**
```python
# 降低阈值增加敏感度（更多场景）
scenes = processor.detect_scenes(video_id, threshold=20.0)

# 提高阈值减少敏感度（更少场景）
scenes = processor.detect_scenes(video_id, threshold=35.0)
```

**3. 批处理优化**
```python
# 对于大量视频，考虑并行处理
from concurrent.futures import ThreadPoolExecutor

def process_video(video_path):
    # 处理单个视频的函数
    pass

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_video, path) for path in video_paths]
    results = [future.result() for future in futures]
```

## 📊 与VideoDB功能对比

| 功能 | VideoDB | 本地处理系统 |
|------|---------|-------------|
| 视频存储 | ✅ 云端 | ✅ 本地 |
| 语音转文字 | ✅ | ✅ (Whisper) |
| 场景检测 | ✅ | ✅ (PySceneDetect) |
| 语义搜索 | ✅ | ⚠️ (可扩展) |
| 视频流播放 | ✅ | ❌ |
| API访问 | ✅ | ✅ |
| 数据隐私 | ⚠️ 云端 | ✅ 完全本地 |
| 成本 | 💰 按使用付费 | 💰 一次性设置 |
| 扩展性 | ✅ 无限 | ⚠️ 硬件限制 |

## 🚀 未来扩展

可以考虑添加的功能：

1. **语义搜索** - 集成sentence-transformers
2. **人脸识别** - 使用face_recognition库
3. **物体检测** - 集成YOLO模型
4. **情感分析** - 分析语音情感
5. **自动标签** - AI生成视频标签
6. **视频压缩** - 自动优化存储空间

这个本地视频处理系统为你提供了完全的数据控制权，同时保持了强大的视频分析能力！ 