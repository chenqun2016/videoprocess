import os
import sys
from moviepy.editor import VideoFileClip

def cut_video(input_path, output_path, segment_duration=10):
    """
    将视频按照指定时长分割成多个片段
    
    参数:
        input_path: 输入视频文件路径
        output_path: 输出文件夹路径
        segment_duration: 每个片段的时长(秒)，默认为10秒
    """
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    video_ext = os.path.splitext(input_path)[1]
    
    try:
        # 加载视频
        video = VideoFileClip(input_path)
        
        # 获取视频总时长
        total_duration = video.duration
        
        # 计算需要分割的片段数
        num_segments = int(total_duration / segment_duration) + (1 if total_duration % segment_duration > 0 else 0)
        
        print(f"处理视频: {video_name}{video_ext}")
        print(f"视频时长: {total_duration:.2f}秒")
        print(f"将分割成 {num_segments} 个片段")
        
        # 分割视频
        for i in range(num_segments):
            # 计算当前片段的起始和结束时间
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            
            # 提取片段
            segment = video.subclip(start_time, end_time)
            
            # 生成输出文件名
            output_file = os.path.join(output_path, f"{video_name}_{i+1}{video_ext}")
            
            # 保存片段
            segment.write_videofile(output_file, codec="libx264", audio_codec="aac")
            
            print(f"已保存片段 {i+1}/{num_segments}: {os.path.basename(output_file)}")
        
        # 关闭视频
        video.close()
        
        print(f"视频 {video_name}{video_ext} 处理完成！")
        
    except Exception as e:
        print(f"处理视频 {video_name}{video_ext} 时出错: {str(e)}")
        return False
    
    return True

def process_videos_in_folder(input_folder, output_folder, segment_duration=10):
    """
    处理文件夹中的所有视频文件
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        segment_duration: 每个片段的时长(秒)，默认为10秒
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 清除输出文件夹中的所有视频文件
    print("清除输出文件夹中的视频文件...")
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            if os.path.splitext(file)[1].lower() in video_extensions:
                file_path = os.path.join(output_folder, file)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file}")
                except Exception as e:
                    print(f"删除文件 {file} 时出错: {str(e)}")
    
    # 获取输入文件夹中的所有文件
    files = os.listdir(input_folder)
    
    # 视频文件扩展名列表
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    # 过滤出视频文件
    video_files = [f for f in files if os.path.splitext(f)[1].lower() in video_extensions]
    
    if not video_files:
        print(f"在 {input_folder} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频文件
    success_count = 0
    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        if cut_video(input_path, output_folder, segment_duration):
            success_count += 1
    
    print(f"处理完成！成功处理 {success_count}/{len(video_files)} 个视频文件")

def main():
    # 设置输入和输出文件夹路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 修改输入文件夹为result2（位于项目根目录）
    input_folder = os.path.join(os.path.dirname(script_dir), "result2")
    output_folder = os.path.join(script_dir, "output")
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 {input_folder} 不存在")
        return
    
    # 处理视频
    process_videos_in_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
