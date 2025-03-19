import os
import random
import time
import shutil
import hashlib
import numpy as np
import sys
from moviepy.editor import VideoFileClip

from config import Config, InvalidParameterError
from videoutils import process_video, clear_output_folder

def main():
    try:
        # 清空输出文件夹
        output_folder = "result1"
        clear_output_folder(output_folder)
        print(f"已清空输出文件夹: {output_folder}")
        
        # 获取输入视频列表
        input_folder = "input"
        if not os.path.exists(input_folder):
            raise InvalidParameterError(f"输入文件夹 {input_folder} 不存在")
            
        video_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            raise InvalidParameterError(f"在 {input_folder} 中未找到视频文件")
        
        # 处理每个视频
        for video_file in video_files:
            try:
                input_path = os.path.join(input_folder, video_file)
                # 检查文件名长度，如果过长则重命名
                if len(video_file) > 100:  # 设置一个合理的长度阈值
                    file_ext = os.path.splitext(video_file)[1]
                    new_filename = f"renamed_{hashlib.md5(video_file.encode()).hexdigest()[:10]}{file_ext}"
                    new_input_path = os.path.join(input_folder, new_filename)
                    print(f"文件名过长，重命名为: {new_filename}")
                    # 复制文件而不是重命名，以保留原始文件
                    shutil.copy2(input_path, new_input_path)
                    input_path = new_input_path
                    video_file = new_filename
                
                # 获取原始文件名（不含扩展名）
                original_filename = os.path.splitext(video_file)[0]
                output_path = os.path.join(output_folder, f"{original_filename}.mp4")
                print(f"开始处理视频: {input_path}")
                process_video(input_path, output_path)
                
                # 如果是重命名的文件，处理完成后删除
                if input_path != os.path.join(input_folder, video_files[video_files.index(video_file)]):
                    os.remove(input_path)
            except Exception as e:
                print(f"处理视频 {video_file} 时出错: {str(e)}")
                # 继续处理下一个视频
                continue

    
            
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
