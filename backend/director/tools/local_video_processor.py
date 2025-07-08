import os
import json
import logging
import subprocess
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import tempfile

import cv2
import numpy as np
import whisper
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

class LocalVideoProcessor:
    """
    本地视频处理器 - 替代VideoDB的功能
    
    主要功能：
    1. 视频转录（Whisper）
    2. 场景检测（PySceneDetect）
    3. 帧提取和分析（OpenCV）
    4. 本地数据库索引（SQLite）
    5. 语义搜索（向量化搜索）
    """
    
    def __init__(self, storage_path: str = "./local_video_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # 创建子目录
        self.videos_path = self.storage_path / "videos"
        self.frames_path = self.storage_path / "frames"
        self.transcripts_path = self.storage_path / "transcripts"
        self.scenes_path = self.storage_path / "scenes"
        
        for path in [self.videos_path, self.frames_path, self.transcripts_path, self.scenes_path]:
            path.mkdir(exist_ok=True)
        
        # 初始化Whisper模型
        try:
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
        
        # 初始化数据库
        self.db_path = self.storage_path / "video_index.db"
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 视频表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                duration REAL,
                fps REAL,
                width INTEGER,
                height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 转录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                start_time REAL,
                end_time REAL,
                text TEXT,
                confidence REAL,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        """)
        
        # 场景表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scenes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                scene_number INTEGER,
                start_time REAL,
                end_time REAL,
                frame_path TEXT,
                FOREIGN KEY (video_id) REFERENCES videos (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def upload_video(self, video_path: str, video_id: str = None) -> Dict:
        """
        上传并处理视频文件
        """
        try:
            if video_id is None:
                video_id = Path(video_path).stem + "_" + str(int(datetime.now().timestamp()))
            
            # 复制视频到本地存储
            dest_path = self.videos_path / f"{video_id}.mp4"
            self._convert_video(video_path, str(dest_path))
            
            # 获取视频信息
            video_info = self._get_video_info(str(dest_path))
            
            # 保存到数据库
            self._save_video_to_db(video_id, video_info)
            
            logger.info(f"Video uploaded successfully: {video_id}")
            return {
                "success": True,
                "video_id": video_id,
                "video_info": video_info
            }
            
        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            return {"success": False, "error": str(e)}
    
    def transcribe_video(self, video_id: str) -> Dict:
        """
        使用Whisper转录视频
        """
        if not self.whisper_model:
            return {"success": False, "error": "Whisper model not available"}
        
        try:
            video_path = self.videos_path / f"{video_id}.mp4"
            
            # 提取音频
            audio_path = self._extract_audio(str(video_path))
            
            # 使用Whisper转录
            result = self.whisper_model.transcribe(
                audio_path, 
                word_timestamps=True,
                language="zh"  # 可配置语言
            )
            
            # 保存转录结果
            transcript_data = []
            for segment in result["segments"]:
                transcript_data.append({
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "text": segment["text"].strip(),
                    "confidence": segment.get("avg_logprob", 0.0)
                })
            
            self._save_transcript_to_db(video_id, transcript_data)
            
            # 保存完整转录文件
            transcript_file = self.transcripts_path / f"{video_id}.json"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 清理临时音频文件
            os.remove(audio_path)
            
            return {
                "success": True,
                "transcript": transcript_data,
                "full_text": result["text"]
            }
            
        except Exception as e:
            logger.error(f"Failed to transcribe video: {e}")
            return {"success": False, "error": str(e)}
    
    def detect_scenes(self, video_id: str, threshold: float = 27.0) -> Dict:
        """
        使用PySceneDetect检测场景
        """
        try:
            video_path = self.videos_path / f"{video_id}.mp4"
            
            # 打开视频
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            
            # 添加检测器
            scene_manager.add_detector(ContentDetector(threshold=threshold))
            
            # 检测场景
            scene_manager.detect_scenes(video, show_progress=True)
            scene_list = scene_manager.get_scene_list()
            
            # 提取关键帧
            scenes_data = []
            for i, scene in enumerate(scene_list):
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                
                # 提取关键帧
                frame_path = self._extract_keyframe(
                    str(video_path), 
                    start_time, 
                    video_id, 
                    i
                )
                
                scenes_data.append({
                    "scene_number": i,
                    "start_time": start_time,
                    "end_time": end_time,
                    "frame_path": frame_path
                })
            
            # 保存到数据库
            self._save_scenes_to_db(video_id, scenes_data)
            
            return {
                "success": True,
                "scenes": scenes_data,
                "total_scenes": len(scenes_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to detect scenes: {e}")
            return {"success": False, "error": str(e)}
    
    def search_transcript(self, video_id: str, query: str) -> List[Dict]:
        """
        在转录文本中搜索
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT start_time, end_time, text, confidence
            FROM transcripts 
            WHERE video_id = ? AND text LIKE ?
            ORDER BY start_time
        """, (video_id, f"%{query}%"))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "start_time": row[0],
                "end_time": row[1],
                "text": row[2],
                "confidence": row[3]
            })
        
        conn.close()
        return results
    
    def get_video_info(self, video_id: str) -> Dict:
        """获取视频信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT filename, duration, fps, width, height, created_at
            FROM videos WHERE id = ?
        """, (video_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": video_id,
                "filename": row[0],
                "duration": row[1],
                "fps": row[2],
                "width": row[3],
                "height": row[4],
                "created_at": row[5]
            }
        return None
    
    def _convert_video(self, input_path: str, output_path: str):
        """使用FFmpeg转换视频格式"""
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-y", output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _extract_audio(self, video_path: str) -> str:
        """提取音频到临时文件"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
        
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            "-y", audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path
    
    def _extract_keyframe(self, video_path: str, timestamp: float, video_id: str, scene_num: int) -> str:
        """提取关键帧"""
        frame_filename = f"{video_id}_scene_{scene_num}.jpg"
        frame_path = self.frames_path / frame_filename
        
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ss", str(timestamp),
            "-vframes", "1",
            "-y", str(frame_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return str(frame_path)
    
    def _get_video_info(self, video_path: str) -> Dict:
        """获取视频元信息"""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "filename": Path(video_path).name,
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height
        }
    
    def _save_video_to_db(self, video_id: str, video_info: Dict):
        """保存视频信息到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO videos 
            (id, filename, duration, fps, width, height)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            video_id,
            video_info["filename"],
            video_info["duration"],
            video_info["fps"],
            video_info["width"],
            video_info["height"]
        ))
        
        conn.commit()
        conn.close()
    
    def _save_transcript_to_db(self, video_id: str, transcript_data: List[Dict]):
        """保存转录数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 先删除现有转录
        cursor.execute("DELETE FROM transcripts WHERE video_id = ?", (video_id,))
        
        # 插入新转录
        for segment in transcript_data:
            cursor.execute("""
                INSERT INTO transcripts 
                (video_id, start_time, end_time, text, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (
                video_id,
                segment["start_time"],
                segment["end_time"],
                segment["text"],
                segment["confidence"]
            ))
        
        conn.commit()
        conn.close()
    
    def _save_scenes_to_db(self, video_id: str, scenes_data: List[Dict]):
        """保存场景数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 先删除现有场景
        cursor.execute("DELETE FROM scenes WHERE video_id = ?", (video_id,))
        
        # 插入新场景
        for scene in scenes_data:
            cursor.execute("""
                INSERT INTO scenes 
                (video_id, scene_number, start_time, end_time, frame_path)
                VALUES (?, ?, ?, ?, ?)
            """, (
                video_id,
                scene["scene_number"],
                scene["start_time"],
                scene["end_time"],
                scene["frame_path"]
            ))
        
        conn.commit()
        conn.close()


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = LocalVideoProcessor()
    
    # 上传视频
    result = processor.upload_video("path/to/video.mp4", "test_video")
    print("Upload result:", result)
    
    # 转录视频
    transcript_result = processor.transcribe_video("test_video")
    print("Transcription result:", transcript_result["success"])
    
    # 检测场景
    scenes_result = processor.detect_scenes("test_video")
    print("Scene detection result:", scenes_result["success"])
    
    # 搜索转录
    search_results = processor.search_transcript("test_video", "关键词")
    print("Search results:", len(search_results)) 