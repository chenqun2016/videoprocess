"""视频处理工具

包含以下功能:
1. 智能码率控制
2. 自适应量化参数
3. GOP结构优化
4. 关键帧重定位
"""

import os
import time
import shutil
from moviepy.editor import VideoFileClip
import numpy as np
import logging
import functools
from typing import Dict, Tuple, List, Optional
import subprocess
import sys

# 从videoutils.py导入视频处理相关类
from videoutils import (
    Scene, SceneComplexity, BitrateController, AdaptiveQuantizer,
    GOPOptimizer, KeyframeOptimizer, SceneReorderer, clear_output_folder
)

# 从config.py导入配置类
from config import Config, FeatureFlags, EncodingConfig, GOPConfig, KeyframeConfig, PerformanceConfig, ComplexityConfig

# ===== 日志配置 =====
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def suppress_moviepy_warnings(func):
    """装饰器：抑制moviepy的资源清理警告"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 保存原始stderr
        old_stderr = sys.stderr
        # 重定向stderr
        sys.stderr = None
        
        try:
            # 调用原始函数
            return func(*args, **kwargs)
        finally:
            # 恢复stderr
            sys.stderr = sys.__stderr__
    return wrapper

class VideoProcessor:
    """视频处理器"""
    
    def __init__(self):
        self.bitrate_controller = BitrateController()
        self.quantizer = AdaptiveQuantizer()
        self.gop_optimizer = GOPOptimizer()
        self.keyframe_optimizer = KeyframeOptimizer()
        self.scene_reorderer = SceneReorderer()
    
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
        """获取默认GOP参数"""
        return self.gop_optimizer._get_default_gop_params()
    
    def _get_default_encoding_params(self) -> Dict[str, str]:
        """获取默认编码参数"""
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

def test_video_processing(input_folder: str, output_folder: str) -> Dict:
    """测试视频处理功能
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        
    Returns:
        Dict: 处理结果统计
    """
    # 清空输出文件夹
    clear_output_folder(output_folder)
    
    # 初始化处理器
    processor = VideoProcessor()
    
    # 统计信息
    stats = {
        'total_videos': 0,
        'processed_videos': 0,
        'failed_videos': 0,
        'total_time': 0,
        'avg_time': 0,
        'details': []
    }
    
    # 获取所有视频文件
    video_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            video_files.append(os.path.join(input_folder, file))
    
    stats['total_videos'] = len(video_files)
    logger.info(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    for i, video_file in enumerate(video_files):
        logger.info(f"\n处理视频 {i+1}/{len(video_files)}: {video_file}")
        
        # 构建输出路径
        filename = os.path.basename(video_file)
        output_path = os.path.join(output_folder, filename)
        
        # 处理视频
        start_time = time.time()
        success = False
        error_msg = ""
        
        try:
            processor.process_video(video_file, output_path)
            success = True
            stats['processed_videos'] += 1
        except Exception as e:
            error_msg = str(e)
            logger.error(f"处理视频失败: {error_msg}")
            stats['failed_videos'] += 1
        
        # 计算处理时间
        process_time = time.time() - start_time
        stats['total_time'] += process_time
        
        # 记录详细信息
        stats['details'].append({
            'file': video_file,
            'success': success,
            'time': process_time,
            'error': error_msg
        })
        
        logger.info(f"视频 {i+1} 处理{'成功' if success else '失败'}, 耗时: {process_time:.2f}秒")
    
    # 计算平均处理时间
    if stats['processed_videos'] > 0:
        stats['avg_time'] = stats['total_time'] / stats['processed_videos']
    
    # 输出统计信息
    logger.info("\n处理统计:")
    logger.info(f"总视频数: {stats['total_videos']}")
    logger.info(f"成功处理: {stats['processed_videos']}")
    logger.info(f"处理失败: {stats['failed_videos']}")
    logger.info(f"总耗时: {stats['total_time']:.2f}秒")
    logger.info(f"平均耗时: {stats['avg_time']:.2f}秒/视频")
    
    return stats

def main():
    """主函数"""
    # 设置输入输出路径
    input_folder = "input"
    output_folder = "output"
    
    # 测试视频处理
    test_video_processing(input_folder, output_folder)

if __name__ == "__main__":
    main()
