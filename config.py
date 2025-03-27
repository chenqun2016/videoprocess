import os
import random
import time
import shutil
import hashlib
import numpy as np
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
    DROP_COUNT_MAX_NORMAL: int = 1    # 正常情况下每次最多丢弃帧数。建议范围：1-2
    
    # 特殊删帧数量
    DROP_COUNT_MIN_SPECIAL: int = 1   # 特殊情况下每次最少丢弃帧数。建议范围：4-5
    DROP_COUNT_MAX_SPECIAL: int = 1   # 特殊情况下每次最多丢弃帧数。建议范围：4-5
    
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
    BLUR_SIGMA: float = 0.2          # 高斯模糊强度。建议范围：0.5-2.0

class TransitionConfig:
    """过渡效果配置"""
    FADE_DURATION: float = 0.74      # 淡入淡出持续时间(秒)。建议范围：0.5-1.0
    COUNT_MIN: int = 1               # 最少插入过渡帧数。建议范围：1-2
    COUNT_MAX: int = 2               # 最多插入过渡帧数。建议范围：2-3
    DURATION: float = 0.1            # 中间过渡效果持续时间(秒)。建议范围：0.1-0.3
    
    # 删帧处过渡效果配置
    FRAME_DROP_TRANSITION_DURATION: float = 0.2  # 删帧处过渡效果持续时间(秒)。建议范围：0.05-0.15
    FRAME_DROP_FADE_TYPE: str = "crossfade"      # 删帧处过渡效果类型：crossfade, fade_in_out

# ===== process8.py 配置类 =====


# ===== 编码参数配置 =====
class EncodingConfig:
    """编码配置参数"""
    # 基础编码参数
    BASE_CRF = 20               # 基础CRF值,控制整体视频质量,越小质量越好(范围18-28)
    MIN_CRF = 13               # 最小CRF值,用于高质量场景
    MAX_CRF = 28               # 最大CRF值,用于低质量场景
    
    # 码率控制参数
    TARGET_BITRATE = 3000      # 目标码率(Kbps),期望的平均码率
    MIN_BITRATE = 1000         # 最小码率,防止质量过低
    MAX_BITRATE = 5000         # 最大码率,防止文件过大
    
    # 量化参数
    BASE_QP = 26               # 基础量化参数,控制压缩程度(范围20-32)
    MIN_QP = 20               # 最小量化参数,用于重要内容
    MAX_QP = 32               # 最大量化参数,用于不重要内容
    
    # 编码器预设
    PRESET = 'medium'          # 编码速度预设,可选:ultrafast,superfast,veryfast,faster,fast,medium,slow,slower,veryslow
    TUNE = 'film'             # 编码优化类型,可选:film,animation,grain,stillimage,fastdecode,zerolatency

# ===== GOP参数配置 =====
class GOPConfig:
    """GOP相关参数"""
    MIN_GOP_SIZE = 30          # 最小GOP大小,防止GOP过小导致码率浪费
    MAX_GOP_SIZE = 250         # 最大GOP大小,防止GOP过大影响随机访问
    MAX_B_FRAMES = 3           # 最大B帧数量,控制编码延迟和压缩效率
    SCENE_THRESHOLD = 0.12     # 降低场景变化阈值以提高检测灵敏度
    REF_FRAMES = 3             # 参考帧数量,影响压缩效率和内存使用

# ===== 关键帧参数配置 =====
class KeyframeConfig:
    """关键帧相关参数"""
    MIN_INTERVAL = 15          # 增加最小关键帧间隔,避免关键帧过密
    MAX_INTERVAL = 200         # 减小最大关键帧间隔,确保更好的随机访问性能
    MOTION_THRESHOLD = 0.06    # 降低运动检测阈值,提高运动检测灵敏度
    CONTENT_THRESHOLD = 0.08   # 降低内容变化阈值,更容易检测到内容变化
    QUALITY_THRESHOLD = 0.25    # 大幅降低质量阈值,使其更容易检测到关键帧
    FEATURE_WEIGHT = 0.7       # 增加特征权重,更注重内容特征
    EFFICIENCY_WEIGHT = 0.3    # 相应减少效率权重

# ===== 性能参数配置 =====
class PerformanceConfig:
    """性能相关参数"""
    BASE_SAMPLE_INTERVAL = 6   # 略微增加采样间隔,在性能和精度间取得平衡
    BATCH_SIZE = 12           # 适当增加批处理大小
    MAX_WORKERS = 6           # 增加工作线程数,提高并行度
    BLOCK_SIZE = 24           # 增加分析块大小,在精度和性能间取得平衡

# ===== 复杂度阈值配置 =====
class ComplexityConfig:
    """复杂度阈值参数"""
    THRESHOLDS = {
        'low': 0.3,            # 低复杂度阈值,低于此值使用低码率和高压缩比
        'medium': 0.6,         # 中复杂度阈值,介于低高之间使用标准参数
        'high': 0.8            # 高复杂度阈值,高于此值使用高码率和低压缩比
    }
    
    BLOCK_THRESHOLDS = {
        'low': 0.2,            # 低复杂度块阈值
        'medium': 0.5,         # 中复杂度块阈值
        'high': 0.8            # 高复杂度块阈值
    }

# ===== 边框配置 =====
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
    

# ===== 移动线条配置 =====
class LineConfig:
    """移动线条配置"""
    # 线条基本属性
    LINE_WIDTH = 1              # 线条宽度（像素）
    LINE_OPACITY = 0.2          # 线条透明度
    LINE_COLOR = (0, 0, 0)      # 线条颜色（黑色）
    
    # 线条移动属性
    LINE_SPEED = 8              # 线条移动速度（1-10）
    

# ===== 画面移动配置 =====
class MovementConfig:
    """画面随机移动配置"""
    # 移动基本属性
    MAX_DISTANCE_RATIO = 0.005    # 最大移动距离（相对于视频宽度的比例，如0.005表示宽度的0.5%）
    MOVE_SPEED = 1.0            # 移动速度（0.1-1.0，值越大速度越快）
    # 移动间隔配置
    MIN_INTERVAL = 0.0          # 最小移动间隔（秒）
    MAX_INTERVAL = 0.1          # 最大移动间隔（秒）

# ===== 功能开关配置 =====
class FeatureFlags:
    """功能开关配置"""
    ENABLE_BITRATE_CONTROL = True    # 启用智能码率控制,根据场景复杂度动态调整码率
    ENABLE_ADAPTIVE_QP = True        # 启用自适应量化,根据内容特征调整量化参数
    ENABLE_GOP_OPTIMIZE = True       # 启用GOP优化,优化关键帧分布和B帧数量
    ENABLE_KEYFRAME_OPTIMIZE = False  # 启用关键帧优化,智能选择关键帧位置
    ENABLE_SCENE_DETECTION = True    # 启用场景检测,检测视频场景变化
    ENABLE_PARALLEL_PROCESSING = True  # 启用并行处理,使用多线程提高性能
    
class Config:
    """静态配置类"""
    # 功能开关
    ENABLE_TRANSITIONS: bool = False    # 是否启用过渡帧功能
    ENABLE_SCALE: bool = True         # 是否启用视频放大功能

    ENABLE_BORDER = False         # 是否启用边框效果
    ENABLE_LINES = False         # 是否启用线条效果

    ENABLE_FRAME_DROP_HEAD_END: bool = True    # 是否启用删除首尾帧功能

    ENABLE_FRAME_DROP: bool = True    # 是否启用删帧功能
    ENABLE_FRAME_DROP_TRANSITIONS: bool = True  # 是否启用删帧处的过渡效果

    ENABLE_SPEED: bool = True         # 是否启用加速功能
    ENABLE_MUTE: bool = True          # 是否启用静音功能
    ENABLE_MD5_MODIFY: bool = True    # 是否启用MD5修改功能
    ENABLE_ROTATE: bool = True        # 是否启用旋转功能
    ENABLE_BLUR: bool = True          # 是否启用模糊功能
    ENABLE_FPS: bool = True           # 是否启用帧率调整功能
    ENABLE_MOVEMENT: bool = True       # 是否启用画面随机移动功能

    ENABLE_COLOR: bool = False         # 是否启用颜色调整功能
    ENABLE_MIRROR: bool = True        # 是否启用镜像翻转功能

    # 子配置类
    VIDEO = VideoConfig
    FRAME = FrameConfig
    EFFECT = EffectConfig
    TRANSITION = TransitionConfig
    BORDER = BorderConfig      # 边框配置
    LINE = LineConfig          # 线条配置
    MOVEMENT = MovementConfig  # 画面移动配置
    
    # process8.py 配置类
    FEATURE_FLAGS = FeatureFlags
    ENCODING = EncodingConfig
    GOP = GOPConfig
    KEYFRAME = KeyframeConfig
    PERFORMANCE = PerformanceConfig
    COMPLEXITY = ComplexityConfig

class VideoProcessError(Exception):
    """视频处理错误的基类"""
    pass

class InvalidParameterError(VideoProcessError):
    """参数无效错误"""
    pass

class ProcessingError(VideoProcessError):
    """处理过程中的错误"""
    pass

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