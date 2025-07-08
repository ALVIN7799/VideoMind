#!/usr/bin/env python3
"""
本地视频处理系统测试脚本

用于验证LocalVideoProcessor和LocalVideoAgent的功能
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 开始测试本地视频处理系统...")
    
    try:
        from director.tools.local_video_processor import LocalVideoProcessor
        print("✅ 成功导入LocalVideoProcessor")
    except ImportError as e:
        print(f"❌ 导入LocalVideoProcessor失败: {e}")
        return False
    
    try:
        from director.agents.local_video_agent import LocalVideoAgent
        print("✅ 成功导入LocalVideoAgent")
    except ImportError as e:
        print(f"❌ 导入LocalVideoAgent失败: {e}")
        return False
    
    # 测试依赖库
    dependencies = {
        "cv2": "OpenCV",
        "whisper": "Whisper",
        "scenedetect": "PySceneDetect",
        "numpy": "NumPy",
        "sqlite3": "SQLite",
    }
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name} 可用")
        except ImportError:
            print(f"❌ {name} 未安装")
            return False
    
    return True

def test_video_processor():
    """测试视频处理器基本功能"""
    print("\n🎥 测试LocalVideoProcessor...")
    
    try:
        from director.tools.local_video_processor import LocalVideoProcessor
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = LocalVideoProcessor(storage_path=temp_dir)
            print("✅ LocalVideoProcessor 初始化成功")
            
            # 检查目录结构
            expected_dirs = ["videos", "frames", "transcripts", "scenes"]
            for dir_name in expected_dirs:
                dir_path = Path(temp_dir) / dir_name
                if dir_path.exists():
                    print(f"✅ 目录 {dir_name} 创建成功")
                else:
                    print(f"❌ 目录 {dir_name} 创建失败")
                    return False
            
            # 检查数据库
            db_path = Path(temp_dir) / "video_index.db"
            if db_path.exists():
                print("✅ SQLite数据库创建成功")
            else:
                print("❌ SQLite数据库创建失败")
                return False
            
            return True
            
    except Exception as e:
        print(f"❌ LocalVideoProcessor 测试失败: {e}")
        return False

def test_agent_functionality():
    """测试Agent功能"""
    print("\n🤖 测试LocalVideoAgent...")
    
    try:
        from director.agents.local_video_agent import LocalVideoAgent
        from director.core.session import Session
        from director.db.sqlite.db import SqliteDB
        
        # 创建临时会话
        db = SqliteDB(":memory:")
        session = Session(db=db)
        
        # 创建Agent
        agent = LocalVideoAgent(session=session)
        print("✅ LocalVideoAgent 初始化成功")
        
        # 测试参数
        params = agent.parameters
        required_actions = ["upload", "transcribe", "detect_scenes", "search", "get_info"]
        
        if "action" in params["properties"]:
            available_actions = params["properties"]["action"]["enum"]
            missing_actions = set(required_actions) - set(available_actions)
            if not missing_actions:
                print("✅ 所有必需的action类型都可用")
            else:
                print(f"❌ 缺少action类型: {missing_actions}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ LocalVideoAgent 测试失败: {e}")
        return False

def create_test_video():
    """创建一个简单的测试视频"""
    print("\n📹 创建测试视频...")
    
    try:
        import cv2
        import numpy as np
        
        # 创建临时视频文件
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video.close()
        
        # 视频参数
        width, height = 640, 480
        fps = 30
        duration = 5  # 5秒
        total_frames = fps * duration
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
        
        # 生成帧
        for i in range(total_frames):
            # 创建彩色帧
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 改变颜色（创建场景变化）
            if i < total_frames // 3:
                color = (255, 0, 0)  # 蓝色
            elif i < 2 * total_frames // 3:
                color = (0, 255, 0)  # 绿色
            else:
                color = (0, 0, 255)  # 红色
            
            frame[:] = color
            
            # 添加文字
            cv2.putText(frame, f'Frame {i+1}', (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            out.write(frame)
        
        out.release()
        print(f"✅ 测试视频创建成功: {temp_video.name}")
        return temp_video.name
        
    except Exception as e:
        print(f"❌ 创建测试视频失败: {e}")
        return None

def test_full_workflow():
    """测试完整工作流程"""
    print("\n🔄 测试完整工作流程...")
    
    # 创建测试视频
    test_video_path = create_test_video()
    if not test_video_path:
        return False
    
    try:
        from director.tools.local_video_processor import LocalVideoProcessor
        
        # 创建临时处理器
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = LocalVideoProcessor(storage_path=temp_dir)
            
            # 1. 测试视频上传
            print("  📤 测试视频上传...")
            upload_result = processor.upload_video(test_video_path, "test_video")
            
            if upload_result["success"]:
                print("  ✅ 视频上传成功")
                
                # 2. 测试场景检测
                print("  🎬 测试场景检测...")
                scenes_result = processor.detect_scenes("test_video", threshold=25.0)
                
                if scenes_result["success"]:
                    print(f"  ✅ 场景检测成功，发现 {scenes_result['total_scenes']} 个场景")
                else:
                    print(f"  ❌ 场景检测失败: {scenes_result.get('error', 'Unknown error')}")
                
                # 3. 测试视频信息获取
                print("  ℹ️  测试视频信息获取...")
                video_info = processor.get_video_info("test_video")
                
                if video_info:
                    print(f"  ✅ 视频信息获取成功: {video_info['duration']:.2f}s")
                else:
                    print("  ❌ 视频信息获取失败")
                
                # 4. 测试转录（可选，因为测试视频没有音频）
                print("  🎙️  测试转录功能...")
                if processor.whisper_model:
                    transcript_result = processor.transcribe_video("test_video")
                    if transcript_result["success"]:
                        print("  ✅ 转录功能正常（无音频内容）")
                    else:
                        print(f"  ⚠️  转录遇到预期错误（测试视频无音频）")
                else:
                    print("  ⚠️  Whisper模型未加载，跳过转录测试")
                
                return True
            else:
                print(f"  ❌ 视频上传失败: {upload_result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        print(f"❌ 完整工作流程测试失败: {e}")
        return False
    
    finally:
        # 清理测试视频
        if test_video_path and os.path.exists(test_video_path):
            os.unlink(test_video_path)

def check_system_requirements():
    """检查系统要求"""
    print("\n🔧 检查系统要求...")
    
    # 检查FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg 可用")
        else:
            print("❌ FFmpeg 不可用")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg 未安装或不在PATH中")
        print("   请安装FFmpeg: https://ffmpeg.org/download.html")
        return False
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python版本 {python_version.major}.{python_version.minor} 符合要求")
    else:
        print(f"❌ Python版本 {python_version.major}.{python_version.minor} 过低，需要3.8+")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 本地视频处理系统测试")
    print("=" * 50)
    
    # 测试步骤
    tests = [
        ("系统要求检查", check_system_requirements),
        ("基本功能测试", test_basic_functionality),
        ("视频处理器测试", test_video_processor),
        ("Agent功能测试", test_agent_functionality),
        ("完整工作流程测试", test_full_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    # 总结
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！本地视频处理系统可以正常使用。")
        print("\n📖 使用指南:")
        print("   - 查看 backend/LOCAL_VIDEO_GUIDE.md 获取详细使用说明")
        print("   - 在 tools/ 目录中填入你的API密钥")
        print("   - 开始使用本地视频处理功能！")
    else:
        print("⚠️  部分测试失败，请检查相关依赖和配置。")
        print("\n🔧 常见解决方案:")
        print("   - 安装缺失的依赖: pip install -r requirements.txt")
        print("   - 确保FFmpeg已安装并在PATH中")
        print("   - 检查Python版本是否为3.8+")

if __name__ == "__main__":
    main() 