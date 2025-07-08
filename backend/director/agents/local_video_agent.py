import logging
from typing import Dict, List, Optional

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    MsgStatus,
    TextContent,
    VideoContent,
    VideoData,
    SearchResultsContent,
    SearchData,
    ShotData
)
from director.tools.local_video_processor import LocalVideoProcessor

logger = logging.getLogger(__name__)

LOCAL_VIDEO_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["upload", "transcribe", "detect_scenes", "search", "get_info"],
            "description": "Action to perform on video",
        },
        "video_path": {
            "type": "string",
            "description": "Path to video file (for upload action)",
        },
        "video_id": {
            "type": "string", 
            "description": "Video ID for processing",
        },
        "query": {
            "type": "string",
            "description": "Search query for transcript search",
        },
        "threshold": {
            "type": "number",
            "description": "Threshold for scene detection (default: 27.0)",
            "default": 27.0,
        },
    },
    "required": ["action"],
}


class LocalVideoAgent(BaseAgent):
    """
    本地视频处理Agent - 不依赖VideoDB的视频处理功能
    
    功能包括：
    1. 视频上传和存储
    2. 语音转文字（Whisper）
    3. 场景检测和分割
    4. 转录文本搜索
    5. 视频信息获取
    """
    
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "local_video"
        self.description = "Local video processing agent using Whisper, PySceneDetect, and OpenCV"
        self.parameters = LOCAL_VIDEO_AGENT_PARAMETERS
        
        # 初始化本地视频处理器
        self.processor = LocalVideoProcessor()
        
        super().__init__(session=session, **kwargs)

    def run(
        self,
        action: str,
        video_path: Optional[str] = None,
        video_id: Optional[str] = None,
        query: Optional[str] = None,
        threshold: float = 27.0,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        执行本地视频处理任务
        
        :param action: 要执行的操作类型
        :param video_path: 视频文件路径（上传时使用）
        :param video_id: 视频ID
        :param query: 搜索查询（搜索时使用）
        :param threshold: 场景检测阈值
        :return: AgentResponse
        """
        
        self.output_message.actions.append(f"Executing local video action: {action}")
        
        try:
            if action == "upload":
                return self._handle_upload(video_path, video_id)
            elif action == "transcribe":
                return self._handle_transcribe(video_id)
            elif action == "detect_scenes":
                return self._handle_detect_scenes(video_id, threshold)
            elif action == "search":
                return self._handle_search(video_id, query)
            elif action == "get_info":
                return self._handle_get_info(video_id)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Local video agent error: {e}")
            error_content = TextContent(
                agent_name=self.agent_name,
                status=MsgStatus.error,
                status_message=f"Error: {str(e)}",
                text=f"Failed to execute {action}: {str(e)}"
            )
            self.output_message.content.append(error_content)
            self.output_message.publish()
            
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Local video processing failed: {str(e)}",
                data={"error": str(e)}
            )

    def _handle_upload(self, video_path: str, video_id: Optional[str]) -> AgentResponse:
        """处理视频上传"""
        if not video_path:
            raise ValueError("Video path is required for upload")
            
        upload_content = TextContent(
            agent_name=self.agent_name,
            status=MsgStatus.progress,
            status_message="Uploading and processing video...",
        )
        self.output_message.content.append(upload_content)
        self.output_message.push_update()

        # 上传视频
        result = self.processor.upload_video(video_path, video_id)
        
        if result["success"]:
            video_info = result["video_info"]
            
            # 创建视频内容对象
            video_content = VideoContent(
                agent_name=self.agent_name,
                status=MsgStatus.success,
                status_message="Video uploaded successfully",
                video=VideoData(
                    id=result["video_id"],
                    name=video_info["filename"],
                    description=f"Local video - Duration: {video_info['duration']:.2f}s",
                    length=video_info["duration"]
                )
            )
            self.output_message.content.append(video_content)
            
            upload_content.status = MsgStatus.success
            upload_content.status_message = "Video uploaded successfully"
            upload_content.text = f"Video ID: {result['video_id']}\nDuration: {video_info['duration']:.2f} seconds"
            
            self.output_message.publish()
            
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                message="Video uploaded successfully",
                data=result
            )
        else:
            raise Exception(result["error"])

    def _handle_transcribe(self, video_id: str) -> AgentResponse:
        """处理视频转录"""
        if not video_id:
            raise ValueError("Video ID is required for transcription")
            
        transcribe_content = TextContent(
            agent_name=self.agent_name,
            status=MsgStatus.progress,
            status_message="Transcribing video using Whisper...",
        )
        self.output_message.content.append(transcribe_content)
        self.output_message.push_update()

        # 转录视频
        result = self.processor.transcribe_video(video_id)
        
        if result["success"]:
            transcript_text = result["full_text"]
            
            transcribe_content.status = MsgStatus.success
            transcribe_content.status_message = "Video transcribed successfully"
            transcribe_content.text = f"Transcript:\n\n{transcript_text}"
            
            self.output_message.publish()
            
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                message="Video transcribed successfully",
                data=result
            )
        else:
            raise Exception(result["error"])

    def _handle_detect_scenes(self, video_id: str, threshold: float) -> AgentResponse:
        """处理场景检测"""
        if not video_id:
            raise ValueError("Video ID is required for scene detection")
            
        scene_content = TextContent(
            agent_name=self.agent_name,
            status=MsgStatus.progress,
            status_message="Detecting scenes using PySceneDetect...",
        )
        self.output_message.content.append(scene_content)
        self.output_message.push_update()

        # 检测场景
        result = self.processor.detect_scenes(video_id, threshold)
        
        if result["success"]:
            scenes = result["scenes"]
            
            # 格式化场景信息
            scenes_text = f"Found {result['total_scenes']} scenes:\n\n"
            for scene in scenes:
                scenes_text += f"Scene {scene['scene_number'] + 1}: "
                scenes_text += f"{scene['start_time']:.2f}s - {scene['end_time']:.2f}s\n"
            
            scene_content.status = MsgStatus.success
            scene_content.status_message = f"Detected {result['total_scenes']} scenes"
            scene_content.text = scenes_text
            
            self.output_message.publish()
            
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                message=f"Detected {result['total_scenes']} scenes",
                data=result
            )
        else:
            raise Exception(result["error"])

    def _handle_search(self, video_id: str, query: str) -> AgentResponse:
        """处理转录搜索"""
        if not video_id:
            raise ValueError("Video ID is required for search")
        if not query:
            raise ValueError("Query is required for search")
            
        search_content = SearchResultsContent(
            agent_name=self.agent_name,
            status=MsgStatus.progress,
            status_message=f"Searching for '{query}' in transcript...",
        )
        self.output_message.content.append(search_content)
        self.output_message.push_update()

        # 搜索转录
        results = self.processor.search_transcript(video_id, query)
        
        if results:
            # 转换为SearchData格式
            search_data = SearchData(
                video_id=video_id,
                video_title=f"Local Video {video_id}",
                stream_url="",  # 本地视频没有流URL
                duration=0,  # 需要从数据库获取
                shots=[
                    ShotData(
                        search_score=result["confidence"],
                        start=result["start_time"],
                        end=result["end_time"],
                        text=result["text"]
                    )
                    for result in results
                ]
            )
            
            search_content.search_results = [search_data]
            search_content.status = MsgStatus.success
            search_content.status_message = f"Found {len(results)} matches"
            
            # 添加文本总结
            summary_content = TextContent(
                agent_name=self.agent_name,
                status=MsgStatus.success,
                status_message="Search results summary",
                text=f"Found {len(results)} matches for '{query}':\n\n" + 
                     "\n".join([f"• {r['start_time']:.1f}s: {r['text'][:100]}..." 
                               for r in results])
            )
            self.output_message.content.append(summary_content)
            
        else:
            search_content.status = MsgStatus.success
            search_content.status_message = "No matches found"
            search_content.search_results = []
            
        self.output_message.publish()
        
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Search completed, found {len(results)} matches",
            data={"results": results, "query": query}
        )

    def _handle_get_info(self, video_id: str) -> AgentResponse:
        """获取视频信息"""
        if not video_id:
            raise ValueError("Video ID is required to get info")
            
        info_content = TextContent(
            agent_name=self.agent_name,
            status=MsgStatus.progress,
            status_message="Getting video information...",
        )
        self.output_message.content.append(info_content)
        self.output_message.push_update()

        # 获取视频信息
        video_info = self.processor.get_video_info(video_id)
        
        if video_info:
            info_text = f"""Video Information:
• ID: {video_info['id']}
• Filename: {video_info['filename']}
• Duration: {video_info['duration']:.2f} seconds
• Resolution: {video_info['width']}x{video_info['height']}
• FPS: {video_info['fps']:.2f}
• Created: {video_info['created_at']}"""
            
            info_content.status = MsgStatus.success
            info_content.status_message = "Video information retrieved"
            info_content.text = info_text
            
            self.output_message.publish()
            
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                message="Video information retrieved successfully",
                data=video_info
            )
        else:
            raise Exception(f"Video not found: {video_id}") 