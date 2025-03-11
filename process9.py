import os
import random
import time
import shutil
from moviepy.editor import VideoFileClip, vfx, concatenate_videoclips, ColorClip
import hashlib
import numpy as np
from moviepy.video.VideoClip import VideoClip
from moviepy.video.fx.all import colorx, rotate
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from scipy.ndimage import gaussian_filter
import math
import sys
import subprocess
from PIL import Image, ImageDraw

class BorderConfig:
    """边框配置"""
    # 边框宽度比例（相对于视频宽度）
    LEFT_BORDER_RATIO = 0.03    # 左边框宽度比例
    RIGHT_BORDER_RATIO = 0.03   # 右边框宽度比例
    TOP_BORDER_RATIO = 0.03     # 上边框宽度比例
    BOTTOM_BORDER_RATIO = 0.03  # 下边框宽度比例
    
    # 圆角半径比例（相对于视频宽度）
    CORNER_RADIUS_RATIO = 0.09  # 圆角半径比例
    
    # 边框颜色 (RGB)
    BORDER_COLOR = (0, 0, 0)    # 黑色
    
    # 边框透明度 (0-1)
    BORDER_OPACITY = 0.94        # 完全不透明
    ENABLE_BORDER = True         # 是否启用边框效果

class LineConfig:
    """移动线条配置"""
    # 线条基本属性
    LINE_WIDTH = 1              # 线条宽度（像素）
    LINE_OPACITY = 0.2          # 线条透明度
    LINE_COLOR = (0, 0, 0)  # 线条颜色（黑色）
    
    # 线条移动属性
    LINE_SPEED = 8              # 线条移动速度（1-10）
    ENABLE_LINES = True         # 是否启用线条效果

class VideoConfig:
    """视频编码配置"""
    CODEC = 'libx264'          # 视频编码器
    PRESET = 'medium'          # 编码速度预设
    CRF = "20"                 # 视频质量控制
    PROFILE = "high"           # 编码配置
    LEVEL = "4.0"              # 编码等级
    PIXEL_FORMAT = "yuv420p"   # 像素格式

class Config:
    """静态配置类"""
    # 子配置类
    VIDEO = VideoConfig
    BORDER = BorderConfig
    LINE = LineConfig

class VideoProcessError(Exception):
    """视频处理错误的基类"""
    pass

class InvalidParameterError(VideoProcessError):
    """参数无效错误"""
    pass

class ProcessingError(VideoProcessError):
    """处理过程中的错误"""
    pass

def clear_output_folder(folder_path):
    """清空输出文件夹"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def process_time(func):
    """装饰器：计算函数执行时间"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} 处理完成，耗时: {end_time - start_time:.2f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            print(f"{func.__name__} 处理失败: {str(e)}，耗时: {end_time - start_time:.2f}秒")
            raise
    return wrapper

def create_rounded_mask(width, height, radius):
    """创建带有圆角的蒙版
    
    Args:
        width (int): 蒙版宽度
        height (int): 蒙版高度
        radius (int): 圆角半径
        
    Returns:
        numpy.ndarray: 带有圆角的蒙版，0表示透明，255表示不透明
    """
    # 创建一个白色背景图像（255表示不透明）
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    
    # 计算内部矩形的尺寸
    left_border = int(width * Config.BORDER.LEFT_BORDER_RATIO)
    right_border = int(width * Config.BORDER.RIGHT_BORDER_RATIO)
    top_border = int(height * Config.BORDER.TOP_BORDER_RATIO)
    bottom_border = int(height * Config.BORDER.BOTTOM_BORDER_RATIO)
    
    # 绘制带有圆角的矩形（0表示透明）
    draw.rounded_rectangle(
        (left_border, top_border, width - right_border, height - bottom_border),
        radius=radius,
        fill=0  # 内部透明
    )
    
    # 将PIL图像转换为numpy数组
    mask_array = np.array(mask)
    
    return mask_array

@process_time
def add_black_border(video_clip):
    """添加带有圆角的黑色边框
    
    Args:
        video_clip (VideoClip): 输入视频片段
        
    Returns:
        VideoClip: 添加黑色边框后的视频片段
    """
    opacity1 = 0
    if Config.BORDER.ENABLE_BORDER:
        opacity1 = Config.BORDER.BORDER_OPACITY

    
    # 获取视频尺寸
    width, height = video_clip.size
    
    # 计算圆角半径
    corner_radius = int(width * Config.BORDER.CORNER_RADIUS_RATIO)
    
    # 创建圆角蒙版
    mask = create_rounded_mask(width, height, corner_radius)
    
    # 创建一个处理函数，将蒙版应用到每一帧
    def apply_mask(get_frame, t):
        # 获取原始帧
        frame = get_frame(t)
        
        # 创建黑色背景
        black_bg = np.zeros_like(frame)
        
        # 将蒙版扩展为3通道，并归一化到0-1范围
        mask_3d = np.stack([mask] * 3, axis=2) / 255.0
        
        # 应用蒙版和透明度：
        # 1. 在蒙版为0的地方显示原始视频
        # 2. 在蒙版为1的地方，根据透明度混合原始视频和黑色背景
        opacity = opacity1
        
        # 计算每个像素的最终颜色
        # 对于边框区域(mask=1)：原始视频 * (1-opacity) + 黑色背景 * opacity
        # 对于视频区域(mask=0)：完全显示原始视频
        masked_frame = frame * (1 - mask_3d * opacity) + black_bg * (mask_3d * opacity)
        
        return masked_frame.astype('uint8')
    
    # 创建新的视频片段
    masked_clip = VideoClip(lambda t: apply_mask(video_clip.get_frame, t))
    masked_clip = masked_clip.set_duration(video_clip.duration)
    masked_clip = masked_clip.set_fps(video_clip.fps)
    
    # 计算边框尺寸
    left_border = int(width * Config.BORDER.LEFT_BORDER_RATIO)
    right_border = int(width * Config.BORDER.RIGHT_BORDER_RATIO)
    top_border = int(height * Config.BORDER.TOP_BORDER_RATIO)
    bottom_border = int(height * Config.BORDER.BOTTOM_BORDER_RATIO)
    
    print(f"添加圆角黑色边框: 左={left_border}px, 右={right_border}px, 上={top_border}px, 下={bottom_border}px, 圆角半径={corner_radius}px, 透明度={opacity1}")
    
    return masked_clip

def validate_video_clip(video_clip):
    """验证视频片段的有效性
    
    Args:
        video_clip (VideoClip): 输入视频片段
        
    Raises:
        InvalidParameterError: 当视频片段无效时
    """
    if video_clip is None:
        raise InvalidParameterError("视频片段不能为空")
    if not hasattr(video_clip, 'fps') or video_clip.fps <= 0:
        raise InvalidParameterError("视频帧率无效")
    if not hasattr(video_clip, 'duration') or video_clip.duration <= 0:
        raise InvalidParameterError("视频长度无效")
    if not hasattr(video_clip, 'size') or any(s <= 0 for s in video_clip.size):
        raise InvalidParameterError("视频尺寸无效")

@process_time
def add_moving_line(video_clip):
    """添加横向移动的线条
    
    Args:
        video_clip (VideoClip): 输入视频片段
        
    Returns:
        VideoClip: 添加移动线条后的视频片段
    """
    # 如果未启用线条效果，直接返回原视频
    if not Config.LINE.ENABLE_LINES:
        return video_clip
    
    # 获取视频尺寸
    width, height = video_clip.size
    
    # 计算中间2/3区域的范围
    left_boundary = int(width * (1/6))  # 左边界（视频宽度的1/6处）
    right_boundary = int(width * (5/6))  # 右边界（视频宽度的5/6处）
    middle_width = right_boundary - left_boundary  # 中间区域的宽度
    
    # 设置一个固定的随机种子，避免每个周期都重置
    random.seed(42)
    
    # 用于存储每个周期的线条参数
    line_params = {}
    
    # 创建一个处理函数，在每一帧上绘制移动的线条
    def add_line(get_frame, t):
        # 获取原始帧
        frame = get_frame(t)
        
        # 计算线条移动的总时间（从中间区域一边移动到另一边）
        # 速度范围1-10，转换为实际速度（像素/秒）
        speed_factor = Config.LINE.LINE_SPEED  # 1-10的速度因子
        pixels_per_second = middle_width / (11 - speed_factor)  # 速度越大，移动时间越短
        
        # 计算一条线从中间区域一边移动到另一边需要的时间
        line_travel_time = middle_width / pixels_per_second
        
        # 计算当前时间点对应的线条周期
        cycle = int(t / line_travel_time)
        cycle_time = t % line_travel_time
        
        # 如果是新的周期，生成新的线条参数
        if cycle not in line_params:
            # 为每个周期生成一个随机的初始位置和方向
            cycle_random = random.Random(cycle)
            direction = cycle_random.choice([-1, 1])  # -1表示从右到左，1表示从左到右
            
            # 随机生成线条的初始位置（在中间2/3区域内）
            if direction == 1:  # 从左到右
                start_pos = cycle_random.randint(left_boundary, right_boundary - int(middle_width * 0.3))  # 在左侧70%范围内随机
                end_pos = right_boundary
            else:  # 从右到左
                start_pos = cycle_random.randint(left_boundary + int(middle_width * 0.3), right_boundary)  # 在右侧70%范围内随机
                end_pos = left_boundary
                
            # 保存这个周期的参数
            line_params[cycle] = {
                'direction': direction,
                'start_pos': start_pos,
                'end_pos': end_pos
            }
            
            # 清理旧的周期参数（保留最近的2个周期）
            old_cycles = [c for c in line_params.keys() if c < cycle - 1]
            for old_cycle in old_cycles:
                del line_params[old_cycle]
        
        # 获取当前周期的线条参数
        params = line_params[cycle]
        start_pos = params['start_pos']
        end_pos = params['end_pos']
        
        # 计算当前时间点的线条位置
        progress = cycle_time / line_travel_time
        current_pos = int(start_pos + (end_pos - start_pos) * progress)
        
        # 创建线条
        # 复制原始帧，以便在上面绘制线条
        result = frame.copy()
        
        # 绘制线条（垂直线）
        for y in range(height):
            # 只修改线条位置的像素
            if left_boundary <= current_pos <= right_boundary:  # 确保线条在中间2/3区域内
                # 混合线条颜色和原始像素颜色
                original_color = result[y, current_pos]
                line_color = np.array(Config.LINE.LINE_COLOR)
                opacity = Config.LINE.LINE_OPACITY
                
                # 应用透明度混合
                new_color = (1 - opacity) * original_color + opacity * line_color
                result[y, current_pos] = new_color.astype('uint8')
        
        return result
    
    # 创建新的视频片段
    line_clip = VideoClip(lambda t: add_line(video_clip.get_frame, t))
    line_clip = line_clip.set_duration(video_clip.duration)
    line_clip = line_clip.set_fps(video_clip.fps)
    
    print(f"添加移动线条: 宽度={Config.LINE.LINE_WIDTH}px, 透明度={Config.LINE.LINE_OPACITY}, 速度={Config.LINE.LINE_SPEED}/10, 移动区域=中间2/3")
    
    return line_clip

@process_time
def process_video(input_path, output_path):
    """处理视频的主函数
    
    Args:
        input_path (str): 输入视频路径
        output_path (str): 输出视频路径
        
    Raises:
        VideoProcessError: 当视频处理过程中出现错误时
    """
    try:
        # 加载视频
        video = VideoFileClip(input_path)
        validate_video_clip(video)
        print(f"原始视频长度: {video.duration:.2f}秒")
        
        # 添加黑色边框
        video = add_black_border(video)
        
        # 添加移动线条
        video = add_moving_line(video)
        
        # 保存视频
        save_video(video, output_path)
        
    except Exception as e:
        raise ProcessingError(f"处理视频 {os.path.basename(input_path)} 时出错: {str(e)}")
    finally:
        if 'video' in locals():
            video.close()

def save_video(video, output_path):
    """保存视频
    
    Args:
        video (VideoClip): 视频片段
        output_path (str): 输出路径
        
    Raises:
        ProcessingError: 当保存视频时出错
    """
    try:
        video.write_videofile(output_path,
                            codec=Config.VIDEO.CODEC,
                            preset=Config.VIDEO.PRESET,
                            ffmpeg_params=["-crf", Config.VIDEO.CRF,
                                         "-profile:v", Config.VIDEO.PROFILE,
                                         "-level", Config.VIDEO.LEVEL,
                                         "-pix_fmt", Config.VIDEO.PIXEL_FORMAT])
        print(f"视频已保存: {output_path}")
    except Exception as e:
        raise ProcessingError(f"保存视频时出错: {str(e)}")

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
            raise InvalidParameterError(f"输入文件夹 {input_folder} 不存在")
        
        # 获取输入视频列表
        video_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        if not video_files:
            raise InvalidParameterError(f"在 {input_folder} 中未找到视频文件")
        
        # 处理每个视频
        for video_file in video_files:
            try:
                input_path = os.path.join(input_folder, video_file)
                output_path = os.path.join(output_folder, video_file)
                print(f"开始处理视频: {input_path}")
                process_video(input_path, output_path)
            except Exception as e:
                print(f"处理视频 {video_file} 时出错: {str(e)}")
                # 继续处理下一个视频
                continue
        
        print("\n所有视频处理完成")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
