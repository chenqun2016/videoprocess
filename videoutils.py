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
from dataclasses import dataclass
import cv2
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
import logging
from scipy.spatial.distance import cosine
from PIL import Image, ImageDraw

from config import Config, VideoProcessError, InvalidParameterError, ProcessingError, validate_config, EncodingConfig, GOPConfig, KeyframeConfig, PerformanceConfig, ComplexityConfig, FeatureFlags, BorderConfig, LineConfig

# 配置日志
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
            similarity = 1 - cosine_similarity([scene1.features], [scene2.features])[0][0]
            
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
        # 获取原始帧
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
def process_video_with_effects(input_path, output_path):
    """处理视频的主函数 (process9.py)
    
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
