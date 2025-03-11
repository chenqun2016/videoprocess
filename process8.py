"""视频处理工具

包含以下功能:
1. 智能码率控制
2. 自适应量化参数
3. GOP结构优化
4. 关键帧重定位
"""

import os
import random
import time
import shutil
from moviepy.editor import VideoFileClip, vfx, concatenate_videoclips
import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
import logging
from dataclasses import dataclass
import math
from scipy.spatial.distance import cosine
import functools
from concurrent.futures import ThreadPoolExecutor
import subprocess

# ===== 功能开关配置 =====
class FeatureFlags:
    """功能开关配置"""
    ENABLE_BITRATE_CONTROL = True    # 启用智能码率控制,根据场景复杂度动态调整码率
    ENABLE_ADAPTIVE_QP = True        # 启用自适应量化,根据内容特征调整量化参数
    ENABLE_GOP_OPTIMIZE = True       # 启用GOP优化,优化关键帧分布和B帧数量
    ENABLE_KEYFRAME_OPTIMIZE = True  # 启用关键帧优化,智能选择关键帧位置
    ENABLE_SCENE_DETECTION = True    # 启用场景检测,检测视频场景变化
    ENABLE_PARALLEL_PROCESSING = True  # 启用并行处理,使用多线程提高性能

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
    MOTION_THRESHOLD = 0.12    # 降低运动检测阈值,提高运动检测灵敏度
    CONTENT_THRESHOLD = 0.15   # 降低内容变化阈值,更容易检测到内容变化
    QUALITY_THRESHOLD = 0.5    # 大幅降低质量阈值,使其更容易检测到关键帧
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
        'low': 0.2,            # 块低复杂度阈值,用于块级别的复杂度判断
        'medium': 0.4,         # 块中复杂度阈值,用于自适应量化
        'high': 0.7            # 块高复杂度阈值,用于保护重要内容
    }

# ===== 日志配置 =====
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Scene:
    """场景数据类"""
    start_time: float          # 开始时间
    end_time: float           # 结束时间
    start_frame: int          # 开始帧
    end_frame: int           # 结束帧
    features: np.ndarray      # 场景特征
    complexity: float         # 场景复杂度
    transition_type: str = 'cut'  # 转场类型

@dataclass
class SceneComplexity:
    """场景复杂度数据类"""
    motion_score: float = 0.0    # 运动复杂度
    detail_score: float = 0.0    # 细节复杂度
    color_score: float = 0.0     # 颜色复杂度
    overall_score: float = 0.0   # 综合评分

class BitrateController:
    """智能码率控制器"""
    
    def __init__(self):
        self.config = EncodingConfig()
        self.complexity_config = ComplexityConfig()
        self.enabled = FeatureFlags.ENABLE_BITRATE_CONTROL
        self.scene_stats = {}
    
    def analyze_frame_complexity(self, frame: np.ndarray) -> SceneComplexity:
        """分析单帧的复杂度
        
        Args:
            frame: 输入帧(numpy数组)
            
        Returns:
            SceneComplexity: 场景复杂度指标
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 1. 运动复杂度(使用Sobel算子检测边缘)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            motion_score = (np.abs(sobelx) + np.abs(sobely)).mean() / 255
            
            # 2. 细节复杂度(使用拉普拉斯算子)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            detail_score = np.abs(laplacian).mean() / 255
            
            # 3. 颜色复杂度
            color_score = frame.std(axis=(0,1)).mean() / 255
            
            # 4. 计算综合评分
            overall_score = (motion_score + detail_score + color_score) / 3
            
            return SceneComplexity(
                motion_score=motion_score,
                detail_score=detail_score,
                color_score=color_score,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"分析帧复杂度时出错: {str(e)}")
            return SceneComplexity()
    
    def calculate_optimal_bitrate(self, complexity: float) -> int:
        """根据场景复杂度计算最优码率
        
        Args:
            complexity: 场景复杂度得分(0-1)
            
        Returns:
            int: 推荐的码率(Kbps)
        """
        try:
            # 基于复杂度进行码率调整
            if complexity <= self.complexity_config.THRESHOLDS['low']:
                # 低复杂度场景使用较低码率
                bitrate = self.config.MIN_BITRATE + (self.config.TARGET_BITRATE - self.config.MIN_BITRATE) * complexity / self.complexity_config.THRESHOLDS['low']
            elif complexity <= self.complexity_config.THRESHOLDS['medium']:
                # 中等复杂度场景使用目标码率
                bitrate = self.config.TARGET_BITRATE
            else:
                # 高复杂度场景使用较高码率
                bitrate = self.config.TARGET_BITRATE + (self.config.MAX_BITRATE - self.config.TARGET_BITRATE) * (complexity - self.complexity_config.THRESHOLDS['medium']) / (1 - self.complexity_config.THRESHOLDS['medium'])
            
            # 确保码率在合理范围内
            bitrate = max(self.config.MIN_BITRATE, min(self.config.MAX_BITRATE, bitrate))
            return int(bitrate)
            
        except Exception as e:
            logger.error(f"计算最优码率时出错: {str(e)}")
            return self.config.TARGET_BITRATE
    
    def get_encoding_params(self, complexity: float) -> Dict[str, str]:
        """根据场景复杂度生成编码参数
        
        Args:
            complexity: 场景复杂度得分(0-1)
            
        Returns:
            Dict[str, str]: FFmpeg编码参数
        """
        try:
            # 计算最优码率
            bitrate = self.calculate_optimal_bitrate(complexity)
            
            # 根据复杂度调整CRF值
            crf = self.config.BASE_CRF
            if complexity <= self.complexity_config.THRESHOLDS['low']:
                crf += 2  # 低复杂度场景可以适当降低质量
            elif complexity >= self.complexity_config.THRESHOLDS['high']:
                crf -= 2  # 高复杂度场景需要提高质量
                
            # 确保CRF在有效范围内
            crf = max(self.config.MIN_CRF, min(self.config.MAX_CRF, crf))
            
            # 生成编码参数
            params = {
                'codec': 'libx264',
                'bitrate': f"{bitrate}k",
                'crf': str(crf),
                'preset': self.config.PRESET,
                'tune': self.config.TUNE,
                # 可以根据需要添加更多参数
            }
            
            return params
            
        except Exception as e:
            logger.error(f"生成编码参数时出错: {str(e)}")
            return {'codec': 'libx264', 'crf': str(self.config.BASE_CRF)}

class AdaptiveQuantizer:
    """自适应量化参数控制器"""
    
    def __init__(self):
        self.config = EncodingConfig()
        self.perf_config = PerformanceConfig()
        self.complexity_config = ComplexityConfig()
        self.enabled = FeatureFlags.ENABLE_ADAPTIVE_QP
        self.qp_range = {
            'min': self.config.MIN_QP,
            'max': self.config.MAX_QP
        }
    
    def analyze_block(self, block: np.ndarray) -> float:
        """分析视频块的复杂度
        
        Args:
            block: 视频块数据(numpy数组)
            
        Returns:
            float: 块复杂度得分(0-1)
        """
        try:
            if block.size == 0:
                return 0.0
                
            # 转换为灰度
            if len(block.shape) == 3:
                block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
            
            # 1. 计算方差(反映块内像素变化)
            variance = np.var(block) / 255
            
            # 2. 计算边缘强度
            sobelx = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_score = np.mean(edge_magnitude) / 255
            
            # 3. 计算纹理复杂度
            texture_score = cv2.Laplacian(block, cv2.CV_64F).var() / 255
            
            # 综合评分
            complexity = (variance + edge_score + texture_score) / 3
            return min(1.0, max(0.0, complexity))
            
        except Exception as e:
            logger.error(f"分析视频块时出错: {str(e)}")
            return 0.0
    
    def calculate_frame_qp(self, frame: np.ndarray) -> List[List[int]]:
        """计算帧的量化参数矩阵
        
        Args:
            frame: 输入帧
            
        Returns:
            List[List[int]]: 量化参数矩阵
        """
        try:
            if not self.enabled:
                return [[self.config.BASE_QP]]
            
            h, w = frame.shape[:2]
            blocks_h = math.ceil(h / self.perf_config.BLOCK_SIZE)
            blocks_w = math.ceil(w / self.perf_config.BLOCK_SIZE)
            
            qp_matrix = []
            for i in range(blocks_h):
                qp_row = []
                for j in range(blocks_w):
                    # 提取块
                    y1 = i * self.perf_config.BLOCK_SIZE
                    y2 = min((i + 1) * self.perf_config.BLOCK_SIZE, h)
                    x1 = j * self.perf_config.BLOCK_SIZE
                    x2 = min((j + 1) * self.perf_config.BLOCK_SIZE, w)
                    block = frame[y1:y2, x1:x2]
                    
                    # 分析块复杂度
                    complexity = self.analyze_block(block)
                    
                    # 根据复杂度调整QP
                    if complexity <= self.complexity_config.BLOCK_THRESHOLDS['low']:
                        qp = self.config.BASE_QP + 4  # 低复杂度区域,增大QP
                    elif complexity <= self.complexity_config.BLOCK_THRESHOLDS['medium']:
                        qp = self.config.BASE_QP + 2  # 中等复杂度,略增大QP
                    elif complexity <= self.complexity_config.BLOCK_THRESHOLDS['high']:
                        qp = self.config.BASE_QP      # 高复杂度,保持基础QP
                    else:
                        qp = self.config.BASE_QP - 2  # 非常复杂,降低QP
                    
                    # 确保QP在有效范围内
                    qp = max(self.qp_range['min'], min(self.qp_range['max'], qp))
                    qp_row.append(qp)
                
                qp_matrix.append(qp_row)
            
            return qp_matrix
            
        except Exception as e:
            logger.error(f"计算量化参数矩阵时出错: {str(e)}")
            return [[self.config.BASE_QP]]

class GOPOptimizer:
    """GOP结构优化器"""
    
    def __init__(self):
        self.config = GOPConfig()
        self.enabled = FeatureFlags.ENABLE_GOP_OPTIMIZE
    
    def detect_scene_changes(self, video: VideoFileClip) -> List[int]:
        """检测场景变化
        
        Args:
            video: 输入视频
            
        Returns:
            List[int]: 场景变化的帧位置列表
        """
        try:
            scene_changes = []
            prev_frame = None
            frame_count = 0
            
            # 采样检测场景变化
            for t in range(0, int(video.duration * video.fps), 2):  # 每2帧采样一次
                frame = video.get_frame(t / video.fps)
                
                if prev_frame is not None:
                    # 计算帧差异
                    if len(frame.shape) == 3:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                    else:
                        frame_gray = frame
                        prev_gray = prev_frame
                    
                    # 计算帧差
                    diff = cv2.absdiff(frame_gray, prev_gray)
                    diff_score = np.mean(diff) / 255
                    
                    # 检测场景变化
                    if diff_score > self.config.SCENE_THRESHOLD:
                        scene_changes.append(frame_count)
                
                prev_frame = frame
                frame_count += 2
            
            logger.info(f"检测到 {len(scene_changes)} 个场景变化")
            return scene_changes
            
        except Exception as e:
            logger.error(f"检测场景变化时出错: {str(e)}")
            return []
    
    def optimize_gop_structure(self, video: VideoFileClip) -> Dict:
        """优化GOP结构
        
        Args:
            video: 输入视频
            
        Returns:
            Dict: GOP参数
        """
        try:
            if not self.enabled:
                return self._get_default_gop_params()
            
            # 1. 检测场景变化
            scene_changes = self.detect_scene_changes(video)
            
            # 2. 计算GOP大小
            gop_sizes = []
            b_frame_counts = []
            prev_scene = 0
            
            for scene in scene_changes:
                distance = scene - prev_scene
                
                # 根据场景长度调整GOP大小
                if distance < self.config.MIN_GOP_SIZE:
                    gop_size = self.config.MIN_GOP_SIZE
                elif distance > self.config.MAX_GOP_SIZE:
                    # 将长场景分成多个GOP
                    num_gops = math.ceil(distance / self.config.MAX_GOP_SIZE)
                    gop_size = distance // num_gops
                else:
                    gop_size = distance
                
                # 根据GOP大小调整B帧数量
                if gop_size < 30:
                    b_frames = 1  # 小GOP使用较少B帧
                elif gop_size < 60:
                    b_frames = 2  # 中等GOP使用适中B帧
                else:
                    b_frames = self.config.MAX_B_FRAMES  # 大GOP使用最大B帧
                
                gop_sizes.append(gop_size)
                b_frame_counts.append(b_frames)
                prev_scene = scene
            
            # 处理最后一个场景
            final_distance = int(video.duration * video.fps) - prev_scene
            if final_distance > 0:
                if final_distance > self.config.MAX_GOP_SIZE:
                    num_gops = math.ceil(final_distance / self.config.MAX_GOP_SIZE)
                    gop_size = final_distance // num_gops
                else:
                    gop_size = final_distance
                
                gop_sizes.append(gop_size)
                b_frame_counts.append(self.config.MAX_B_FRAMES)
            
            # 3. 生成GOP参数
            gop_params = {
                'scene_changes': scene_changes,
                'gop_sizes': gop_sizes,
                'b_frame_counts': b_frame_counts,
                'ref_frames': self.config.REF_FRAMES
            }
            
            # 输出统计信息
            if gop_sizes:
                avg_gop_size = sum(gop_sizes) / len(gop_sizes)
                avg_b_frames = sum(b_frame_counts) / len(b_frame_counts)
                logger.info(f"平均GOP大小: {avg_gop_size:.1f}")
                logger.info(f"平均B帧数量: {avg_b_frames:.1f}")
            
            return gop_params
            
        except Exception as e:
            logger.error(f"优化GOP结构时出错: {str(e)}")
            return self._get_default_gop_params()
    
    def _get_default_gop_params(self) -> Dict:
        """获取默认GOP参数"""
        return {
            'scene_changes': [],
            'gop_sizes': [self.config.MAX_GOP_SIZE],
            'b_frame_counts': [self.config.MAX_B_FRAMES],
            'ref_frames': self.config.REF_FRAMES
        }

class KeyframeOptimizer:
    """关键帧优化器"""
    
    def __init__(self):
        self.config = KeyframeConfig()
        self.perf_config = PerformanceConfig()
        self.enabled = FeatureFlags.ENABLE_KEYFRAME_OPTIMIZE
    
    def optimize_keyframe_positions(self, video: VideoFileClip, 
                                  gop_params: Dict) -> Dict:
        """优化关键帧位置"""
        try:
            if not self.enabled:
                logger.info("关键帧优化已禁用")
                return gop_params
            
            # 1. 初始化参数
            keyframe_positions = []
            prev_frame = None
            frame_count = 0
            last_keyframe = -1
            total_frames = int(video.duration * video.fps)
            
            logger.info("开始分析帧...")
            
            # 2. 分析每一帧
            for t in range(0, total_frames, self.perf_config.BASE_SAMPLE_INTERVAL):
                frame = video.get_frame(t / video.fps)
                
                if prev_frame is not None:
                    # 计算帧重要性得分
                    importance = self._calculate_frame_importance(frame)
                    
                    # 计算运动得分
                    motion_score = self._analyze_motion(prev_frame, frame)
                    
                    # 综合评分
                    frame_score = (importance * self.config.FEATURE_WEIGHT + 
                                 motion_score * (1 - self.config.FEATURE_WEIGHT))
                    
                    # 判断是否设置关键帧
                    distance_from_last = frame_count - last_keyframe
                    
                    if (distance_from_last >= self.config.MIN_INTERVAL and
                        frame_score > self.config.QUALITY_THRESHOLD and
                        (frame_score > self.config.CONTENT_THRESHOLD or
                         motion_score > self.config.MOTION_THRESHOLD or
                         distance_from_last >= self.config.MAX_INTERVAL)):
                        keyframe_positions.append(frame_count)
                        last_keyframe = frame_count
                
                prev_frame = frame
                frame_count += self.perf_config.BASE_SAMPLE_INTERVAL
                
                # 每处理100帧输出一次进度
                if frame_count % 100 == 0:
                    logger.info(f"已分析 {frame_count}/{total_frames} 帧...")
            
            # 3. 更新GOP参数并输出统计信息
            # 即使没有找到关键帧也输出统计信息
            logger.info("\n关键帧优化统计:")
            logger.info(f"- 原始场景变化数: {len(gop_params['scene_changes'])}")
            
            if keyframe_positions:
                # 计算GOP大小
                gop_sizes = []
                prev_pos = 0
                for pos in keyframe_positions:
                    gop_sizes.append(pos - prev_pos)
                    prev_pos = pos
                
                # 添加最后一个GOP
                final_gop = total_frames - prev_pos
                if final_gop > 0:
                    gop_sizes.append(final_gop)
                
                # 输出详细统计信息
                avg_gop_size = sum(gop_sizes) / len(gop_sizes)
                min_gop = min(gop_sizes)
                max_gop = max(gop_sizes)
                std_gop = np.std(gop_sizes)
                keyframe_density = len(keyframe_positions)/total_frames*100
                
                logger.info(f"- 优化后关键帧数: {len(keyframe_positions)}")
                logger.info(f"- 平均GOP大小: {avg_gop_size:.1f} 帧")
                logger.info(f"- 最小GOP大小: {min_gop} 帧")
                logger.info(f"- 最大GOP大小: {max_gop} 帧")
                logger.info(f"- GOP大小标准差: {std_gop:.1f}")
                logger.info(f"- 关键帧密度: {keyframe_density:.2f}%")
                logger.info(f"- 平均关键帧间隔: {1/keyframe_density*100:.1f} 帧")
                
                # 分析关键帧分布
                intervals = np.diff(keyframe_positions)
                logger.info("\n关键帧间隔分布:")
                logger.info(f"- 最短间隔: {min(intervals)} 帧")
                logger.info(f"- 最长间隔: {max(intervals)} 帧")
                logger.info(f"- 间隔标准差: {np.std(intervals):.1f}")
                
                # 更新GOP参数
                gop_params['keyframe_positions'] = keyframe_positions
                gop_params['gop_sizes'] = gop_sizes
            else:
                # 如果没有找到关键帧，输出原因
                logger.info("- 未找到满足条件的关键帧")
                logger.info(f"- 当前质量阈值: {self.config.QUALITY_THRESHOLD}")
                logger.info(f"- 当前内容阈值: {self.config.CONTENT_THRESHOLD}")
                logger.info(f"- 当前运动阈值: {self.config.MOTION_THRESHOLD}")
                logger.info("- 建议调整阈值参数以检测更多关键帧")
            
            # 分析关键帧质量设置
            logger.info("\n关键帧质量设置:")
            logger.info(f"- 质量阈值: {self.config.QUALITY_THRESHOLD}")
            logger.info(f"- 内容阈值: {self.config.CONTENT_THRESHOLD}")
            logger.info(f"- 运动阈值: {self.config.MOTION_THRESHOLD}")
            logger.info(f"- 最小间隔: {self.config.MIN_INTERVAL} 帧")
            logger.info(f"- 最大间隔: {self.config.MAX_INTERVAL} 帧")
            
            return gop_params
            
        except Exception as e:
            logger.error(f"优化关键帧位置时出错: {str(e)}")
            return gop_params
    
    def _calculate_frame_importance(self, frame: np.ndarray) -> float:
        """计算帧的重要性得分"""
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 1. 计算边缘强度
            edges = cv2.Canny(frame, 100, 200)
            edge_score = np.mean(edges) / 255
            
            # 2. 计算纹理复杂度
            texture = cv2.Laplacian(frame, cv2.CV_64F)
            texture_score = np.std(texture) / 255
            
            # 3. 计算亮度分布
            brightness_score = np.std(frame) / 255
            
            # 综合评分
            importance = (edge_score * 0.4 +      # 边缘最重要
                        texture_score * 0.3 +     # 纹理次之
                        brightness_score * 0.3)    # 亮度最次
            
            return min(1.0, max(0.0, importance))
            
        except Exception as e:
            logger.error(f"计算帧重要性时出错: {str(e)}")
            return 0.0
    
    def _analyze_motion(self, frame1: np.ndarray, 
                       frame2: np.ndarray) -> float:
        """分析两帧之间的运动程度"""
        try:
            # 转换为灰度图
            if len(frame1.shape) == 3:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            if len(frame2.shape) == 3:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # 计算运动幅度
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_score = np.mean(magnitude) / 100  # 归一化
            
            return min(1.0, max(0.0, motion_score))
            
        except Exception as e:
            logger.error(f"分析运动程度时出错: {str(e)}")
            return 0.0

def suppress_moviepy_warnings(func):
    """装饰器：抑制moviepy的资源清理警告"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            # 忽略moviepy的资源清理错误
            import sys
            sys.stderr = open(os.devnull, 'w')
            sys.stderr = sys.__stderr__
    return wrapper

class VideoProcessor:
    """视频处理器"""
    
    def __init__(self):
        self.bitrate_controller = BitrateController()
        self.quantizer = AdaptiveQuantizer()
        self.gop_optimizer = GOPOptimizer()
        self.keyframe_optimizer = KeyframeOptimizer()
    
    def _analyze_video(self, video: VideoFileClip) -> Tuple[float, float]:
        """分析视频复杂度和量化参数
        
        Args:
            video: 输入视频
            
        Returns:
            Tuple[float, float]: (平均复杂度, 平均量化参数)
        """
        total_complexity = 0
        frame_count = 0
        total_qp = 0
        sample_interval = max(1, int(video.fps))
        
        for t in range(0, int(video.duration * video.fps), sample_interval):
            frame = video.get_frame(t / video.fps)
            
            # 1. 分析场景复杂度
            complexity = self.bitrate_controller.analyze_frame_complexity(frame)
            total_complexity += complexity.overall_score
            
            # 2. 计算量化参数矩阵
            qp_matrix = self.quantizer.calculate_frame_qp(frame)
            avg_qp = np.mean(qp_matrix)
            total_qp += avg_qp
            
            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"已分析 {frame_count} 帧...")
        
        # 计算平均值
        avg_complexity = total_complexity / frame_count if frame_count > 0 else 0
        avg_qp = total_qp / frame_count if frame_count > 0 else self.quantizer.config.BASE_QP
        
        logger.info(f"视频平均复杂度: {avg_complexity:.3f}")
        logger.info(f"平均量化参数: {avg_qp:.1f}")
        
        return avg_complexity, avg_qp
    
    def _get_default_gop_params(self) -> Dict:
        """获取默认的GOP参数"""
        return {
            'scene_changes': [],
            'gop_sizes': [self.gop_optimizer.config.MAX_GOP_SIZE],
            'b_frame_counts': [self.gop_optimizer.config.MAX_B_FRAMES],
            'ref_frames': self.gop_optimizer.config.REF_FRAMES
        }
    
    def _get_default_encoding_params(self) -> Dict[str, str]:
        """获取默认的编码参数"""
        return {
            'codec': 'libx264',
            'bitrate': f"{self.bitrate_controller.config.TARGET_BITRATE}k",
            'crf': str(self.bitrate_controller.config.BASE_CRF),
            'preset': self.bitrate_controller.config.PRESET,
            'tune': self.bitrate_controller.config.TUNE
        }
    
    def process_video(self, input_path: str, output_path: str):
        """处理视频"""
        video = None
        try:
            logger.info(f"开始处理视频: {input_path}")
            start_time = time.time()
            
            # 加载视频并静音
            video = VideoFileClip(input_path)
            video = video.without_audio()
            
            # 1. 分析视频复杂度 (如果启用码率控制或自适应量化)
            if (FeatureFlags.ENABLE_BITRATE_CONTROL or 
                FeatureFlags.ENABLE_ADAPTIVE_QP):
                avg_complexity, avg_qp = self._analyze_video(video)
            else:
                avg_complexity = 0
                avg_qp = self.quantizer.config.BASE_QP
            
            # 2. 优化GOP结构
            if FeatureFlags.ENABLE_GOP_OPTIMIZE:
                gop_params = self.gop_optimizer.optimize_gop_structure(video)
            else:
                gop_params = self._get_default_gop_params()
            
            # 3. 优化关键帧位置
            if FeatureFlags.ENABLE_KEYFRAME_OPTIMIZE:
                logger.info("开始优化关键帧位置...")
                gop_params = self.keyframe_optimizer.optimize_keyframe_positions(
                    video, gop_params
                )
                logger.info("关键帧优化完成")
            
            # 4. 获取编码参数
            if FeatureFlags.ENABLE_BITRATE_CONTROL:
                encoding_params = self.bitrate_controller.get_encoding_params(
                    avg_complexity
                )
            else:
                encoding_params = self._get_default_encoding_params()
            
            # 添加量化参数设置
            encoding_params['qmin'] = str(self.quantizer.qp_range['min'])
            encoding_params['qmax'] = str(self.quantizer.qp_range['max'])
            encoding_params['aq-mode'] = '2'  # 启用自适应量化
            encoding_params['aq-strength'] = '1.0'  # 自适应量化强度
            
            logger.info(f"使用编码参数: {encoding_params}")
            
            # 构建ffmpeg参数
            ffmpeg_params = [
                '-crf', encoding_params['crf'],
                '-qmin', encoding_params['qmin'],
                '-qmax', encoding_params['qmax'],
                '-aq-mode', encoding_params['aq-mode'],
                '-aq-strength', encoding_params['aq-strength'],
                # 修复GOP相关参数
                '-g', str(int(np.mean(gop_params['gop_sizes']))),  # GOP大小
                '-bf', str(int(np.mean(gop_params['b_frame_counts']))),  # B帧数量
                '-refs', str(gop_params['ref_frames']),  # 参考帧数量
                '-sc_threshold', '40',  # 场景切换检测阈值
                '-keyint_min', str(self.gop_optimizer.config.MIN_GOP_SIZE)  # 修正这里：使用config中的MIN_GOP_SIZE
            ]
            
            # 获取原始文件名（不含扩展名）
            original_filename = os.path.splitext(os.path.basename(input_path))[0]
            # 修改输出路径，保持目录不变，只修改文件名
            output_dir = os.path.dirname(output_path)
            output_path = os.path.join(output_dir, f"{original_filename}.mp4")
            
            # 保存处理后的视频
            video.write_videofile(
                output_path,
                codec=encoding_params['codec'],
                bitrate=encoding_params['bitrate'],
                preset=encoding_params['preset'],
                ffmpeg_params=ffmpeg_params
            )
            
            process_time = time.time() - start_time
            logger.info(f"视频处理完成,耗时: {process_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"处理视频时出错: {str(e)}")
            raise
            
        finally:
            # 更安全的资源清理
            try:
                # 关闭主视频
                if video is not None:
                    try:
                        if hasattr(video, 'close'):
                            video.close()
                    except:
                        pass
                        
                # 强制清理
                import gc
                gc.collect()
                
            except:
                pass

class SceneReorderer:
    """场景重排序器"""
    
    def __init__(self):
        self.min_scene_duration = 1.0  # 最小场景时长(秒)
        self.max_scenes = 20          # 最大场景数
        self.feature_dim = 512        # 特征维度
        self.transition_duration = 0.5  # 转场时长(秒)
        
        # 场景特征提取器
        self.feature_extractor = cv2.dnn.readNetFromCaffe(
            'models/scene_feature_deploy.prototxt',
            'models/scene_feature.caffemodel'
        ) if os.path.exists('models/scene_feature.caffemodel') else None
    
    def extract_scene_features(self, frame: np.ndarray) -> np.ndarray:
        """提取场景特征
        
        Args:
            frame: 输入帧
            
        Returns:
            np.ndarray: 场景特征向量
        """
        try:
            if self.feature_extractor is None:
                # 如果没有预训练模型，使用简单的特征
                frame = cv2.resize(frame, (224, 224))
                features = frame.mean(axis=(0,1))
                return features / np.linalg.norm(features)
            
            # 使用预训练模型提取特征
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, (224, 224),
                (104, 117, 123), swapRB=True
            )
            self.feature_extractor.setInput(blob)
            features = self.feature_extractor.forward()
            return features.flatten()
            
        except Exception as e:
            logger.error(f"提取场景特征时出错: {str(e)}")
            return np.zeros(self.feature_dim)
    
    def split_scenes(self, video: VideoFileClip, 
                    scene_changes: List[int]) -> List[Scene]:
        """将视频分割为场景
        
        Args:
            video: 输入视频
            scene_changes: 场景变化点列表
            
        Returns:
            List[Scene]: 场景列表
        """
        try:
            scenes = []
            fps = video.fps
            
            # 添加视频开始点
            if not scene_changes or scene_changes[0] > 0:
                scene_changes.insert(0, 0)
            
            # 添加视频结束点
            if scene_changes[-1] < video.duration * fps:
                scene_changes.append(int(video.duration * fps))
            
            # 分割场景
            for i in range(len(scene_changes) - 1):
                start_frame = scene_changes[i]
                end_frame = scene_changes[i + 1]
                
                start_time = start_frame / fps
                end_time = end_frame / fps
                
                # 跳过太短的场景
                if end_time - start_time < self.min_scene_duration:
                    continue
                
                # 提取场景中间帧的特征
                mid_time = (start_time + end_time) / 2
                mid_frame = video.get_frame(mid_time)
                features = self.extract_scene_features(mid_frame)
                
                # 计算场景复杂度
                complexity = np.std(mid_frame) / 255
                
                scene = Scene(
                    start_time=start_time,
                    end_time=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    features=features,
                    complexity=complexity
                )
                scenes.append(scene)
                
                if len(scenes) >= self.max_scenes:
                    break
            
            return scenes
            
        except Exception as e:
            logger.error(f"分割场景时出错: {str(e)}")
            return []
    
    def calculate_scene_similarity(self, scene1: Scene, 
                                 scene2: Scene) -> float:
        """计算两个场景的相似度
        
        Args:
            scene1: 第一个场景
            scene2: 第二个场景
            
        Returns:
            float: 相似度得分(0-1)
        """
        try:
            # 计算特征向量的余弦相似度
            similarity = 1 - cosine(scene1.features, scene2.features)
            
            # 考虑场景复杂度差异
            complexity_diff = abs(scene1.complexity - scene2.complexity)
            
            # 综合评分
            score = similarity * (1 - complexity_diff)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"计算场景相似度时出错: {str(e)}")
            return 0.0
    
    def optimize_scene_order(self, scenes: List[Scene]) -> List[Scene]:
        """优化场景顺序
        
        Args:
            scenes: 输入场景列表
            
        Returns:
            List[Scene]: 重排序后的场景列表
        """
        try:
            if not scenes:
                return []
            
            # 计算场景相似度矩阵
            n = len(scenes)
            similarity_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        similarity_matrix[i][j] = self.calculate_scene_similarity(
                            scenes[i], scenes[j]
                        )
            
            # 使用贪心算法重排序场景
            used = set()
            result = []
            current = 0  # 从第一个场景开始
            
            while len(result) < n:
                result.append(scenes[current])
                used.add(current)
                
                if len(used) == n:
                    break
                
                # 找到最相似的未使用场景
                next_scene = -1
                max_similarity = -1
                
                for i in range(n):
                    if i not in used:
                        similarity = similarity_matrix[current][i]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            next_scene = i
                
                current = next_scene if next_scene != -1 else min(set(range(n)) - used)
            
            # 随机调整一些场景顺序以增加变化
            if len(result) > 3:
                num_swaps = len(result) // 3
                for _ in range(num_swaps):
                    i = random.randint(0, len(result)-2)
                    result[i], result[i+1] = result[i+1], result[i]
            
            return result
            
        except Exception as e:
            logger.error(f"优化场景顺序时出错: {str(e)}")
            return scenes
    
    def create_transition(self, scene1: Scene, scene2: Scene, 
                        video: VideoFileClip) -> VideoFileClip:
        """创建场景转场效果
        
        Args:
            scene1: 第一个场景
            scene2: 第二个场景
            video: 原始视频
            
        Returns:
            VideoFileClip: 转场片段
        """
        try:
            # 提取两个场景的结束和开始部分
            end_clip = video.subclip(
                max(0, scene1.end_time - self.transition_duration),
                scene1.end_time
            )
            start_clip = video.subclip(
                scene2.start_time,
                min(video.duration, scene2.start_time + self.transition_duration)
            )
            
            # 创建交叉淡入淡出效果
            transition = concatenate_videoclips([
                end_clip.crossfadeout(self.transition_duration),
                start_clip.crossfadein(self.transition_duration)
            ])
            
            return transition
            
        except Exception as e:
            logger.error(f"创建转场效果时出错: {str(e)}")
            return video.subclip(scene2.start_time, scene2.start_time + 0.1)
    
    def reorder_scenes(self, video: VideoFileClip, 
                      scene_changes: List[int]) -> VideoFileClip:
        """重新排序场景"""
        try:
            logger.info("开始场景重排序...")
            
            # 1. 分割场景
            scenes = self.split_scenes(video, scene_changes)
            logger.info(f"识别出 {len(scenes)} 个有效场景")
            
            if not scenes:
                logger.warning("没有检测到有效场景，返回原始视频")
                return video
            
            # 2. 优化场景顺序
            reordered_scenes = self.optimize_scene_order(scenes)
            logger.info("场景重排序完成")
            
            # 3. 重组视频
            clips = []
            try:
                for i, scene in enumerate(reordered_scenes):
                    # 提取场景片段
                    scene_clip = video.subclip(scene.start_time, scene.end_time)
                    clips.append(scene_clip)
                    
                    # 添加转场效果
                    if i < len(reordered_scenes) - 1:
                        transition = self.create_transition(
                            scene, reordered_scenes[i + 1], video
                        )
                        clips.append(transition)
                
                # 合并所有片段
                final_video = concatenate_videoclips(clips)
                
                # 设置最终视频的属性
                final_video.fps = video.fps
                final_video.duration = sum(clip.duration for clip in clips)
                
                logger.info("视频重组完成")
                return final_video
                
            except Exception as e:
                logger.error(f"重组视频时出错: {str(e)}")
                # 清理clips
                for clip in clips:
                    try:
                        clip.close()
                    except:
                        pass
                return video
                
            finally:
                # 确保清理所有临时片段
                for clip in clips:
                    try:
                        clip.close()
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"重排序场景时出错: {str(e)}")
            return video

def clear_output_folder(folder: str):
    """清空输出文件夹
    
    Args:
        folder: 输出文件夹路径
    """
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"删除文件时出错: {str(e)}")
    else:
        os.makedirs(folder)
    logger.info(f"已清空输出文件夹: {folder}")

def test_video_processing(input_folder: str, output_folder: str) -> Dict:
    """测试视频处理功能
    
    Args:
        input_folder: 输入文件夹
        output_folder: 输出文件夹
        
    Returns:
        Dict: 测试结果数据
    """
    test_results = {
        'total_videos': 0,
        'successful_videos': 0,
        'failed_videos': 0,
        'processing_times': [],
        'compression_ratios': [],
        'quality_metrics': [],
        'error_logs': []
    }
    
    try:
        # 获取所有视频文件
        video_files = [f for f in os.listdir(input_folder)
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        test_results['total_videos'] = len(video_files)
        
        # 处理每个视频
        processor = VideoProcessor()
        for video_file in video_files:
            try:
                input_path = os.path.join(input_folder, video_file)
                # 获取原始文件名（不含扩展名）
                original_filename = os.path.splitext(video_file)[0]
                output_path = os.path.join(output_folder, f"{original_filename}.mp4")
                
                # 记录原始文件大小
                original_size = os.path.getsize(input_path)
                
                # 记录开始时间
                start_time = time.time()
                
                # 处理视频
                logger.info(f"开始处理视频: {video_file}")
                processor.process_video(input_path, output_path)
                
                # 记录处理时间
                process_time = time.time() - start_time
                test_results['processing_times'].append(process_time)
                
                # 计算压缩率
                output_size = os.path.getsize(output_path)
                compression_ratio = original_size / output_size
                test_results['compression_ratios'].append(compression_ratio)
                
                # 记录成功
                test_results['successful_videos'] += 1
                
                # 记录处理数据
                logger.info(f"视频 {video_file} 处理完成:")
                logger.info(f"- 处理时间: {process_time:.2f}秒")
                logger.info(f"- 压缩率: {compression_ratio:.2f}x")
                logger.info(f"- 输出大小: {output_size/1024/1024:.2f}MB")
                
            except Exception as e:
                test_results['failed_videos'] += 1
                test_results['error_logs'].append(f"{video_file}: {str(e)}")
                logger.error(f"处理视频 {video_file} 时出错: {str(e)}")
        
        # 计算统计数据
        if test_results['processing_times']:
            avg_time = np.mean(test_results['processing_times'])
            avg_ratio = np.mean(test_results['compression_ratios'])
            
            logger.info("\n测试结果汇总:")
            logger.info(f"总视频数: {test_results['total_videos']}")
            logger.info(f"成功处理: {test_results['successful_videos']}")
            logger.info(f"处理失败: {test_results['failed_videos']}")
            logger.info(f"平均处理时间: {avg_time:.2f}秒")
            logger.info(f"平均压缩率: {avg_ratio:.2f}x")
            
            if test_results['error_logs']:
                logger.info("\n错误日志:")
                for error in test_results['error_logs']:
                    logger.info(error)
        
        return test_results
        
    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")
        return test_results

def main():
    """主函数"""
    try:
        # 清空输出目录
        output_folder = "result2"
        clear_output_folder(output_folder)
        
        # 获取输入视频列表
        input_folder = "result1"  # 修改输入目录
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"输入文件夹 {input_folder} 不存在")
        
        # 运行测试
        logger.info("开始视频处理测试...")
        test_results = test_video_processing(input_folder, output_folder)
        
        if test_results['total_videos'] == 0:
            raise FileNotFoundError(f"在 {input_folder} 中未找到视频文件")


        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
