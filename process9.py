"""视频处理工具 - 添加边框和移动线条效果

包含以下功能:
1. 添加圆角黑色边框
2. 添加移动线条效果
"""

import os
import time
import shutil
from moviepy.editor import VideoFileClip
import numpy as np
import logging

# 从videoutils.py导入视频处理相关函数
from videoutils import (
    process_video_with_effects, clear_output_folder, 
    add_black_border, add_moving_line, save_video
)

# 从config.py导入配置类
from config import Config, VideoProcessError, InvalidParameterError, ProcessingError

# ===== 日志配置 =====
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # 设置输入和输出文件夹
        input_folder = "result2"
        output_folder = "result3"
        
        # 清空输出文件夹
        clear_output_folder(output_folder)
        print(f"已清空输出文件夹: {output_folder}")
        
        # 检查输入文件夹是否存在
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"输入文件夹 {input_folder} 不存在")
        
        # 获取所有视频文件
        video_files = []
        for file in os.listdir(input_folder):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                video_files.append(os.path.join(input_folder, file))
        
        if not video_files:
            print(f"在 {input_folder} 中未找到视频文件")
            return
        
        # 处理每个视频
        for video_file in video_files:
            output_file = os.path.join(output_folder, os.path.basename(video_file))
            print(f"开始处理视频: {video_file}")
            
            # 处理视频
            process_video_with_effects(video_file, output_file)
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
