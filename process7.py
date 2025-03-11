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

class VideoConfig:
    """视频编码配置"""
    CODEC = 'libx264'          # 视频编码器。建议保持默认
    PRESET = 'medium'          # 编码速度预设。可选：ultrafast到veryslow
    CRF = "20"                # 视频质量控制。范围：0-51，值越小质量越好
    PROFILE = "high"          # 编码配置。可选：baseline, main, high
    LEVEL = "4.0"             # 编码等级。可选：3.0, 3.1, 4.0, 4.1, 4.2
    PIXEL_FORMAT = "yuv420p"  # 像素格式。建议保持默认

class FrameConfig:
    """帧处理配置"""
    # 删帧间隔
    DROP_INTERVAL_MIN: int = 28       # 最小丢帧间隔。建议范围：30-45
    DROP_INTERVAL_MAX: int = 33       # 最大丢帧间隔。建议范围：45-60
    
    # 正常删帧数量
    DROP_COUNT_MIN_NORMAL: int = 1    # 正常情况下每次最少丢弃帧数。建议范围：1-2
    DROP_COUNT_MAX_NORMAL: int = 2    # 正常情况下每次最多丢弃帧数。建议范围：1-2
    
    # 特殊删帧数量
    DROP_COUNT_MIN_SPECIAL: int = 4   # 特殊情况下每次最少丢弃帧数。建议范围：4-5
    DROP_COUNT_MAX_SPECIAL: int = 5   # 特殊情况下每次最多丢弃帧数。建议范围：4-5
    
    # 特殊删帧间隔
    DROP_SPECIAL_INTERVAL: int = 3    # 每隔多少次进行一次特殊删帧。例如：3表示每第3次删帧时使用特殊删帧数量
    
    
    FPS_ADJUST_MIN: float = -2.0      # 帧率调整最小值。建议范围：-3到0
    FPS_ADJUST_MAX: float = 2.0       # 帧率调整最大值。建议范围：0到3
    REMOVE_HEAD_FRAMES: int = 5       # 删除开头的帧数。建议范围：5-15
    REMOVE_TAIL_FRAMES: int = 5       # 删除结尾的帧数。建议范围：5-15

class EffectConfig:
    """视频效果配置"""
    ROTATE_ANGLE_MIN: float = -2.0   # 最小旋转角度。固定范围：-1到1度
    ROTATE_ANGLE_MAX: float = 2.0    # 最大旋转角度。固定范围：-1到1度
    SCALE: float = 1.05    # 视频放大比例

    SPEED_FACTOR: float = 1.04        # 视频加速倍数。建议范围：1.0-1.5，值越大速度越快
    COLOR_FACTOR_MIN: float = 0.95   # 颜色调整最小倍数。建议范围：0.9-1.0
    COLOR_FACTOR_MAX: float = 1.05   # 颜色调整最大倍数。建议范围：1.0-1.1

    BLUR_REGION_MIN: float = 0.05     # 模糊区域最小比例。建议范围：0.05-0.15
    BLUR_REGION_MAX: float = 0.1     # 模糊区域最大比例。建议范围：0.15-0.25
    BLUR_SIGMA: float = 0.3          # 高斯模糊强度。建议范围：0.5-2.0

class TransitionConfig:
    """过渡效果配置"""
    FADE_DURATION: float = 0.74      # 淡入淡出持续时间(秒)。建议范围：0.5-1.0
    COUNT_MIN: int = 1               # 最少插入过渡帧数。建议范围：1-2
    COUNT_MAX: int = 2               # 最多插入过渡帧数。建议范围：2-3
    DURATION: float = 0.1            # 中间过渡效果持续时间(秒)。建议范围：0.1-0.3
    
    # 删帧处过渡效果配置
    FRAME_DROP_TRANSITION_DURATION: float = 0.2  # 删帧处过渡效果持续时间(秒)。建议范围：0.05-0.15
    FRAME_DROP_FADE_TYPE: str = "crossfade"      # 删帧处过渡效果类型：crossfade, fade_in_out

class Config:
    """静态配置类"""
    # 功能开关
    ENABLE_TRANSITIONS: bool = False    # 是否启用过渡帧功能
    ENABLE_SCALE: bool = False         # 是否启用视频放大功能

    ENABLE_FRAME_DROP_HEAD_END: bool = True    # 是否启用删除首尾帧功能

    ENABLE_FRAME_DROP: bool = True    # 是否启用删帧功能
    ENABLE_FRAME_DROP_TRANSITIONS: bool = True  # 是否启用删帧处的过渡效果

    ENABLE_SPEED: bool = True         # 是否启用加速功能
    ENABLE_MUTE: bool = True          # 是否启用静音功能
    ENABLE_COLOR: bool = True         # 是否启用颜色调整功能
    ENABLE_MD5_MODIFY: bool = True    # 是否启用MD5修改功能
    ENABLE_MIRROR: bool = True        # 是否启用镜像翻转功能
    ENABLE_ROTATE: bool = True        # 是否启用旋转功能
    ENABLE_BLUR: bool = True          # 是否启用模糊功能
    ENABLE_FPS: bool = True           # 是否启用帧率调整功能
    

    # 子配置类
    VIDEO = VideoConfig
    FRAME = FrameConfig
    EFFECT = EffectConfig
    TRANSITION = TransitionConfig

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

@process_time
def modify_md5(video_clip):
    """修改视频MD5值"""
    duration = video_clip.duration
    blank_frame = video_clip.get_frame(0) * 0
    modified_clip = video_clip.set_make_frame(lambda t: blank_frame if t == duration else video_clip.get_frame(t))
    return modified_clip

@process_time
def mute_video(video_clip):
    """静音处理"""
    return video_clip.without_audio()

@process_time
def remove_head_tail_frames(video_clip):
    """删除视频开头和结尾的指定帧数"""
    fps = video_clip.fps
    start_time = Config.FRAME.REMOVE_HEAD_FRAMES / fps
    end_time = video_clip.duration - (Config.FRAME.REMOVE_TAIL_FRAMES / fps)
    print(f"删除开头 {Config.FRAME.REMOVE_HEAD_FRAMES} 帧和结尾 {Config.FRAME.REMOVE_TAIL_FRAMES} 帧")
    print(f"保留时间段: {start_time:.2f}s - {end_time:.2f}s")
    return video_clip.subclip(start_time, end_time)

@process_time
def drop_frames(video_clip):
    """随机间隔丢帧，支持按规律变化的删帧数量"""
    # 计算总帧数
    total_frames = int(video_clip.fps * video_clip.duration)
    print(f"视频总帧数: {total_frames}")
    
    # 创建帧索引数组
    frame_indices = np.arange(total_frames)
    
    # 生成丢帧位置
    current_pos = 0
    drop_positions = []
    drop_counts = []
    drop_count_sequence = []  # 记录每次删帧的数量，用于调试
    
    # 删帧计数器，用于确定是否使用特殊删帧数量
    drop_counter = 0
    
    while current_pos < total_frames:
        # 随机选择下一个丢帧位置
        interval = random.randint(Config.FRAME.DROP_INTERVAL_MIN, Config.FRAME.DROP_INTERVAL_MAX)
        current_pos += interval
        
        if current_pos >= total_frames:
            break
        
        # 增加删帧计数器
        drop_counter += 1
        
        # 根据删帧计数器决定使用正常删帧数量还是特殊删帧数量
        if drop_counter % Config.FRAME.DROP_SPECIAL_INTERVAL == 0:
            # 使用特殊删帧数量
            drop_count = random.randint(Config.FRAME.DROP_COUNT_MIN_SPECIAL, Config.FRAME.DROP_COUNT_MAX_SPECIAL)
            drop_count_sequence.append(f"特殊({drop_count})")
        else:
            # 使用正常删帧数量
            drop_count = random.randint(Config.FRAME.DROP_COUNT_MIN_NORMAL, Config.FRAME.DROP_COUNT_MAX_NORMAL)
            drop_count_sequence.append(f"正常({drop_count})")
        
        drop_positions.append(current_pos)
        drop_counts.append(drop_count)
    
    # 创建要丢弃的帧索引掩码
    drop_mask = np.zeros(total_frames, dtype=bool)
    
    # 记录每个删帧区间的前后帧
    transition_segments = []
    
    for pos, count in zip(drop_positions, drop_counts):
        end_pos = min(pos + count, total_frames)
        drop_mask[pos:end_pos] = True
        
        # 记录删帧前的最后一帧和删帧后的第一帧
        before_frame = max(0, pos - 1)
        after_frame = min(total_frames - 1, end_pos)
        
        # 只有当前后帧都不在其他删帧区间内时，才添加过渡段
        if not drop_mask[before_frame] and not drop_mask[after_frame]:
            transition_segments.append((before_frame, after_frame))
    
    # 获取要保留的帧索引
    keep_frames = frame_indices[~drop_mask]
    
    # 创建原始帧索引到新帧索引的映射
    # 这个映射告诉我们原始视频中的每一帧在新视频中对应的位置
    # -1表示该帧被删除
    original_to_new_index = np.full(total_frames, -1)
    for i, frame in enumerate(keep_frames):
        original_to_new_index[frame] = i
    
    # 创建新的时间轴
    new_times = np.arange(len(keep_frames)) / video_clip.fps
    
    # 如果启用了删帧处过渡效果
    if Config.ENABLE_FRAME_DROP_TRANSITIONS:
        transition_duration = Config.TRANSITION.FRAME_DROP_TRANSITION_DURATION
        half_duration = transition_duration / 2
        
        def make_frame(t):
            # 计算当前时间对应的帧索引
            frame_index = int(t * video_clip.fps)
            
            # 确保帧索引在有效范围内
            frame_index = min(frame_index, len(new_times) - 1)
            
            # 获取当前时间对应的原始帧
            original_frame = keep_frames[frame_index]
            
            # 检查是否在过渡区间内
            for before_frame, after_frame in transition_segments:
                # 只有当前后帧都被保留时，才应用过渡效果
                if before_frame in keep_frames and after_frame in keep_frames:
                    # 获取前后帧在新视频中的索引
                    before_index = original_to_new_index[before_frame]
                    after_index = original_to_new_index[after_frame]
                    
                    # 计算前后帧在新视频中的时间
                    before_time = new_times[before_index]
                    after_time = new_times[after_index]
                    
                    # 如果当前时间在过渡区间内
                    if before_time <= t <= after_time:
                        # 计算过渡进度
                        progress = (t - before_time) / (after_time - before_time)
                        
                        # 使用平滑的过渡函数
                        if Config.TRANSITION.FRAME_DROP_FADE_TYPE == "crossfade":
                            # 交叉淡入淡出
                            weight_before = 0.5 * (1 + math.cos(math.pi * progress))
                            weight_after = 1 - weight_before
                        else:  # fade_in_out
                            # 淡出淡入
                            if progress <= 0.5:
                                weight_before = math.cos(math.pi * progress)
                                weight_after = 0
                            else:
                                weight_before = 0
                                weight_after = math.sin(math.pi * (progress - 0.5))
                        
                        # 获取前后帧并应用权重
                        if weight_before > 0 and weight_after > 0:
                            frame_before = video_clip.get_frame(before_frame / video_clip.fps)
                            frame_after = video_clip.get_frame(after_frame / video_clip.fps)
                            return (frame_before * weight_before + frame_after * weight_after).astype('uint8')
                        elif weight_before > 0:
                            return video_clip.get_frame(before_frame / video_clip.fps)
                        else:
                            return video_clip.get_frame(after_frame / video_clip.fps)
            
            # 如果不在过渡区间内，直接返回对应的原始帧
            return video_clip.get_frame(original_frame / video_clip.fps)
    else:
        # 原始的帧获取逻辑
        def make_frame(t):
            # 计算当前时间对应的帧索引
            frame_index = int(t * video_clip.fps)
            
            # 确保帧索引在有效范围内
            frame_index = min(frame_index, len(keep_frames) - 1)
            
            # 获取对应的原始帧
            original_frame = keep_frames[frame_index]
            
            # 返回原始帧
            return video_clip.get_frame(original_frame / video_clip.fps)
    
    new_duration = len(keep_frames) / video_clip.fps
    new_clip = VideoClip(make_frame)
    new_clip = new_clip.set_duration(new_duration)
    new_clip = new_clip.set_fps(video_clip.fps)
    
    print(f"丢帧后保留帧数: {len(keep_frames)}")
    print(f"丢帧位置数量: {len(drop_positions)}")
    print(f"删帧序列: {', '.join(drop_count_sequence[:20])}{'...' if len(drop_count_sequence) > 20 else ''}")
    print(f"平均每个位置丢弃帧数: {sum(drop_counts) / len(drop_counts) if drop_counts else 0:.2f}")
    
    return new_clip

@process_time
def mirror_video(video_clip):
    """镜像翻转"""
    return video_clip.fx(vfx.mirror_x)

@process_time
def speed_up_video(video_clip):
    """视频加速"""
    print(f"加速倍率: {Config.EFFECT.SPEED_FACTOR}")
    return video_clip.speedx(Config.EFFECT.SPEED_FACTOR)

@process_time
def auto_crop_borders(video_clip):
    """自动检测和裁剪视频黑边
    
    Args:
        video_clip (VideoClip): 输入视频片段
        
    Returns:
        VideoClip: 裁剪后的视频片段
    """
    # 获取第一帧用于分析
    frame = video_clip.get_frame(0)
    h, w = frame.shape[:2]
    
    # 设置更严格的黑色阈值
    threshold = 3  # 像素阈值，小于这个值认为是黑色
    
    # 计算每行和每列的平均值
    row_means = np.mean(frame, axis=(1,2))
    col_means = np.mean(frame, axis=(0,2))
    
    # 找到非黑色区域的边界
    top = np.argmax(row_means > threshold)
    bottom = h - np.argmax(row_means[::-1] > threshold)
    left = np.argmax(col_means > threshold)
    right = w - np.argmax(col_means[::-1] > threshold)
    
    # 添加小边距
    margin = 2
    top = max(0, top - margin)
    bottom = min(h, bottom + margin)
    left = max(0, left - margin)
    right = min(w, right + margin)
    
    # 确保裁剪尺寸为偶数
    top = top - (top % 2)  # 向下取偶数
    bottom = bottom + (bottom % 2)  # 向上取偶数
    left = left - (left % 2)  # 向下取偶数
    right = right + (right % 2)  # 向上取偶数
    
    # 确保裁剪后的尺寸不会太小
    min_dim = min(h, w) // 3
    if (bottom - top) < min_dim or (right - left) < min_dim:
        print("检测到的有效区域太小，保持原始尺寸")
        return video_clip
    
    print(f"裁剪区域: 上={top}, 下={bottom}, 左={left}, 右={right}")
    print(f"原始尺寸: {w}x{h}")
    print(f"裁剪后尺寸: {right-left}x{bottom-top}")
    
    # 裁剪视频
    return video_clip.crop(x1=left, y1=top, x2=right, y2=bottom)

@process_time
def rotate_video(video_clip):
    """旋转视频并裁剪回原始尺寸
    
    Args:
        video_clip (VideoClip): 输入视频片段
        
    Returns:
        tuple: (处理后的视频片段, 旋转角度)
    """
    # 随机生成旋转角度
    angle = random.choice([Config.EFFECT.ROTATE_ANGLE_MIN, Config.EFFECT.ROTATE_ANGLE_MAX])

    print(f"旋转角度: {angle:.2f}度")
    
    # 获取原始尺寸
    original_width, original_height = video_clip.size
    print(f"原始尺寸: {original_width}x{original_height}")
    
    # 直接旋转视频
    rotated = video_clip.rotate(angle)
    
    print(f"旋转后尺寸: {rotated.w}x{rotated.h}")
    
    # 计算裁剪区域（居中裁剪）
    enlarged_width, enlarged_height = rotated.size
    
    # 计算裁剪的起始坐标（居中裁剪）
    x1 = (enlarged_width - original_width) // 2
    y1 = (enlarged_height - original_height) // 2
    x2 = x1 + original_width
    y2 = y1 + original_height
    
    print(f"裁剪区域: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    
    # 裁剪回原始尺寸
    cropped = rotated.crop(x1=x1, y1=y1, x2=x2, y2=y2)
    
    return cropped, angle

@process_time
def adjust_color(video_clip):
    """随机调整颜色"""
    factor = random.uniform(Config.EFFECT.COLOR_FACTOR_MIN, Config.EFFECT.COLOR_FACTOR_MAX)
    print(f"颜色调整因子: {factor:.2f}")
    return video_clip.fx(colorx, factor)

@process_time
def blur_region(video_clip):
    """随机区域模糊处理"""
    def blur_frame(get_frame, t):
        frame = get_frame(t)
        
        # 随机选择模糊区域
        h, w = frame.shape[:2]
        blur_w = int(w * random.uniform(Config.EFFECT.BLUR_REGION_MIN, Config.EFFECT.BLUR_REGION_MAX))
        blur_h = int(h * random.uniform(Config.EFFECT.BLUR_REGION_MIN, Config.EFFECT.BLUR_REGION_MAX))
        
        # 随机选择位置
        x = random.randint(0, w - blur_w)
        y = random.randint(0, h - blur_h)
        
        # 对选定区域进行高斯模糊
        region = frame[y:y+blur_h, x:x+blur_w]
        frame[y:y+blur_h, x:x+blur_w] = gaussian_filter(region, sigma=Config.EFFECT.BLUR_SIGMA)
        
        return frame
    
    new_clip = VideoClip(lambda t: blur_frame(video_clip.get_frame, t))
    new_clip = new_clip.set_duration(video_clip.duration)
    new_clip = new_clip.set_fps(video_clip.fps)
    
    print(f"添加随机区域模糊效果")
    return new_clip

@process_time
def adjust_fps(video_clip):
    """随机调整帧率"""
    original_fps = video_clip.fps
    fps_change = random.uniform(Config.FRAME.FPS_ADJUST_MIN, Config.FRAME.FPS_ADJUST_MAX)
    new_fps = original_fps + fps_change
    new_fps = max(new_fps, 1)  # 确保帧率不小于1
    print(f"帧率从 {original_fps:.2f} 调整为 {new_fps:.2f}")
    return video_clip.set_fps(new_fps)

@process_time
def scale_video(video_clip):
    """放大视频画面并裁剪回原始尺寸
    
    Args:
        video_clip (VideoClip): 输入视频片段
        
    Returns:
        VideoClip: 放大后的视频片段
    """
    scale_factor = Config.EFFECT.SCALE
    original_size = video_clip.size
    
    # 放大视频
    enlarged_clip = video_clip.resize(scale_factor)
    
    # 计算裁剪区域（居中裁剪）
    enlarged_width, enlarged_height = enlarged_clip.size
    original_width, original_height = original_size
    
    # 计算裁剪的起始坐标（居中裁剪）
    x1 = (enlarged_width - original_width) // 2
    y1 = (enlarged_height - original_height) // 2
    x2 = x1 + original_width
    y2 = y1 + original_height
    
    print(f"视频画面放大: 原始尺寸 {original_width}x{original_height}, 放大比例 {scale_factor}")
    print(f"裁剪区域: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    
    # 裁剪回原始尺寸
    return enlarged_clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)

@process_time
def add_transitions(video_clip):
    """添加淡入淡出效果，起始透明度为50%"""
    duration = video_clip.duration
    print(f"处理前视频长度: {duration:.2f}秒")
    
    # 创建一个黑色背景clip
    def make_bg_frame(t):
        return np.zeros((video_clip.size[1], video_clip.size[0], 3), dtype=np.uint8)
    
    bg = VideoClip(make_bg_frame)
    bg = bg.set_duration(duration)
    bg = bg.set_fps(video_clip.fps)
    
    # 创建帧处理函数
    def process_frame(get_frame, t):
        frame = get_frame(t)
        # 计算当前的不透明度
        if t < Config.TRANSITION.FADE_DURATION:
            # 淡入：从0.5到1.0
            opacity = 0.5 + (t / Config.TRANSITION.FADE_DURATION) * 0.5
        elif t > duration - Config.TRANSITION.FADE_DURATION:
            # 淡出：从1.0到0.5
            fade_progress = (duration - t) / Config.TRANSITION.FADE_DURATION
            opacity = 0.5 + fade_progress * 0.5
        else:
            opacity = 1.0
        
        # 应用不透明度
        return (frame * opacity).astype('uint8')
    
    # 创建新的视频clip
    video_with_transitions = VideoClip(
        lambda t: process_frame(video_clip.get_frame, t),
        duration=duration
    ).set_fps(video_clip.fps)
    
    # 合成到黑色背景
    final = CompositeVideoClip([bg, video_with_transitions])
    
    print(f"添加淡入淡出效果(各 {Config.TRANSITION.FADE_DURATION:.2f}秒)")
    print(f"处理后视频长度: {final.duration:.2f}秒")
    
    return final

def ensure_even_dimensions(width, height):
    """确保视频尺寸为2的倍数"""
    return width - (width % 2), height - (height % 2)

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

def validate_config():
    """验证配置参数的有效性
    
    Raises:
        InvalidParameterError: 当配置参数无效时
    """
    # 验证帧处理参数
    if Config.FRAME.REMOVE_HEAD_FRAMES < 0 or Config.FRAME.REMOVE_TAIL_FRAMES < 0:
        raise InvalidParameterError("删除帧数不能为负数")
    if Config.FRAME.DROP_INTERVAL_MIN > Config.FRAME.DROP_INTERVAL_MAX:
        raise InvalidParameterError("最小丢帧间隔不能大于最大丢帧间隔")
    
    # 验证正常删帧数量
    if Config.FRAME.DROP_COUNT_MIN_NORMAL > Config.FRAME.DROP_COUNT_MAX_NORMAL:
        raise InvalidParameterError("正常情况下最小丢帧数不能大于最大丢帧数")
    if Config.FRAME.DROP_COUNT_MIN_NORMAL < 0 or Config.FRAME.DROP_COUNT_MAX_NORMAL < 0:
        raise InvalidParameterError("正常情况下丢帧数不能为负数")
    
    # 验证特殊删帧数量
    if Config.FRAME.DROP_COUNT_MIN_SPECIAL > Config.FRAME.DROP_COUNT_MAX_SPECIAL:
        raise InvalidParameterError("特殊情况下最小丢帧数不能大于最大丢帧数")
    if Config.FRAME.DROP_COUNT_MIN_SPECIAL < 0 or Config.FRAME.DROP_COUNT_MAX_SPECIAL < 0:
        raise InvalidParameterError("特殊情况下丢帧数不能为负数")
    
    # 验证特殊删帧间隔
    if Config.FRAME.DROP_SPECIAL_INTERVAL <= 0:
        raise InvalidParameterError("特殊删帧间隔必须大于0")
    
    # 验证效果参数
    if Config.EFFECT.SPEED_FACTOR <= 0:
        raise InvalidParameterError("加速倍数必须大于0")
    if Config.EFFECT.ROTATE_ANGLE_MIN > Config.EFFECT.ROTATE_ANGLE_MAX:
        raise InvalidParameterError("最小旋转角度不能大于最大旋转角度")
    if Config.EFFECT.BLUR_SIGMA <= 0:
        raise InvalidParameterError("模糊强度必须大于0")
    if Config.EFFECT.SCALE <= 0:
        raise InvalidParameterError("放大比例必须大于0")
    
    # 验证过渡效果参数
    if Config.TRANSITION.FADE_DURATION <= 0:
        raise InvalidParameterError("淡入淡出持续时间必须大于0")
    if Config.TRANSITION.COUNT_MIN > Config.TRANSITION.COUNT_MAX:
        raise InvalidParameterError("最小过渡帧数不能大于最大过渡帧数")
    
    # 验证删帧过渡效果参数
    if Config.TRANSITION.FRAME_DROP_TRANSITION_DURATION <= 0:
        raise InvalidParameterError("删帧处过渡效果持续时间必须大于0")
    if Config.TRANSITION.FRAME_DROP_FADE_TYPE not in ["crossfade", "fade_in_out"]:
        raise InvalidParameterError("删帧处过渡效果类型必须是 'crossfade' 或 'fade_in_out'")

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
        # 验证配置参数
        validate_config()
        
        # 加载视频
        video = VideoFileClip(input_path)
        validate_video_clip(video)
        print(f"原始视频长度: {video.duration:.2f}秒")
        
        # 处理视频
        if Config.ENABLE_MD5_MODIFY:
            video = modify_md5(video)
        
        if Config.ENABLE_MUTE:
            video = mute_video(video)
            

        if Config.ENABLE_FRAME_DROP_HEAD_END:
            video = remove_head_tail_frames(video)
            print(f"删除头尾帧后长度: {video.duration:.2f}秒")
        
        if Config.ENABLE_FRAME_DROP:
            video = drop_frames(video)
            
        if Config.ENABLE_SPEED:
            video = speed_up_video(video)
            print(f"加速后长度: {video.duration:.2f}秒")
            
        if Config.ENABLE_MIRROR:
            video = mirror_video(video)
            
        if Config.ENABLE_COLOR:
            video = adjust_color(video)
            
        if Config.ENABLE_ROTATE:
            video, angle = rotate_video(video)
            
        if Config.ENABLE_BLUR:
            video = blur_region(video)
            
        if Config.ENABLE_FPS:
            video = adjust_fps(video)
            
        if Config.ENABLE_SCALE:
            video = scale_video(video)
            
        if Config.ENABLE_TRANSITIONS:
            video = add_transitions(video)
            print(f"添加过渡效果后长度: {video.duration:.2f}秒")
        
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
