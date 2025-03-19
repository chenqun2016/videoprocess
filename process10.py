"""统一视频处理工具

整合了以下功能:
1. 基础视频处理 (process7.py)
   - 修改MD5
   - 静音处理
   - 删除头尾帧
   - 丢帧处理
   - 视频加速
   - 镜像翻转
   - 颜色调整
   - 旋转处理
   - 区域模糊
   - 帧率调整

2. 高级编码优化 (process8.py)
   - 智能码率控制
   - 自适应量化参数
   - GOP结构优化
   - 关键帧重定位

3. 视觉效果增强 (process9.py)
   - 添加圆角黑色边框
   - 添加移动线条效果
"""

import os
import random
import time
import shutil
import hashlib
import numpy as np
import sys
import logging
from moviepy.editor import VideoFileClip
import functools
from typing import Dict, Tuple, List, Optional

# 从videoutils.py导入视频处理相关函数和类
from videoutils import (
    # process7.py相关
    process_video,
    
    # process8.py相关
    Scene, SceneComplexity, BitrateController, AdaptiveQuantizer,
    GOPOptimizer, KeyframeOptimizer, SceneReorderer,
    
    # process9.py相关
    process_video_with_effects, add_black_border, add_moving_line, save_video,
    
    # 视频处理函数
    modify_md5, mute_video, remove_head_tail_frames, drop_frames,
    speed_up_video, mirror_video, adjust_color, rotate_video, blur_region, adjust_fps,
    scale_video, random_movement,
    
    # 通用函数
    clear_output_folder
)

# 从config.py导入配置类
from config import (
    Config, FeatureFlags, EncodingConfig, GOPConfig, KeyframeConfig, 
    PerformanceConfig, ComplexityConfig, VideoProcessError, InvalidParameterError, 
    ProcessingError
)

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
    """统一视频处理器"""
    
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

def process_pipeline(input_path, output_folder, enable_steps=None):
    """完整的视频处理流水线
    
    Args:
        input_path: 输入视频路径
        output_folder: 输出文件夹路径
        enable_steps: 启用的处理步骤 {1: 基础处理, 2: 高级编码, 3: 视觉效果}
        
    Returns:
        str: 最终输出视频路径
    """
    if enable_steps is None:
        enable_steps = {1, 2, 3}  # 默认启用所有步骤
    
    try:
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取原始文件名（不含扩展名）
        original_filename = os.path.splitext(os.path.basename(input_path))[0]
        
        # 加载原始视频
        print(f"加载视频: {input_path}")
        video = VideoFileClip(input_path)
        
        # 步骤1: 基础视频处理 (process7.py)
        if 1 in enable_steps:
            print(f"步骤1: 基础视频处理")
            
            # 应用基础处理函数（不保存）
            if Config.ENABLE_MD5_MODIFY:
                video = modify_md5(video)
            
            if Config.ENABLE_MUTE:
                video = mute_video(video)
            
            if Config.ENABLE_FRAME_DROP_HEAD_END:
                video = remove_head_tail_frames(video)
            
            if Config.ENABLE_FRAME_DROP:
                video = drop_frames(video)
            
            if Config.ENABLE_SPEED:
                video = speed_up_video(video)
            
            if Config.ENABLE_MIRROR:
                video = mirror_video(video)
            
            if Config.ENABLE_COLOR:
                video = adjust_color(video)
            
            if Config.ENABLE_ROTATE:
                # rotate_video可能返回元组(video_clip, crop_info)
                rotate_result = rotate_video(video)
                if isinstance(rotate_result, tuple) and len(rotate_result) > 0:
                    video = rotate_result[0]  # 获取视频对象
                else:
                    video = rotate_result
            
            if Config.ENABLE_BLUR:
                try:
                    video = blur_region(video)
                except Exception as e:
                    print(f"警告: 应用模糊效果时出错: {str(e)}")
                    # 继续处理，不中断流程
            
            if Config.ENABLE_FPS:
                video = adjust_fps(video)
                    
            if Config.ENABLE_MOVEMENT:
                try:
                    video = random_movement(video)
                except Exception as e:
                    print(f"警告: 应用随机移动效果时出错: {str(e)}")
                    # 继续处理，不中断流程

            if Config.ENABLE_SCALE:
                try:
                    video = scale_video(video)
                except Exception as e:
                    print(f"警告: 应用放大效果时出错: {str(e)}")
                    # 继续处理，不中断流程
        
        # 步骤2: 高级编码优化 (process8.py)
        if 2 in enable_steps:
            print(f"步骤2: 高级编码优化")
            
            # 创建处理器实例
            processor = VideoProcessor()
            
            # 分析视频复杂度 (如果启用码率控制或自适应量化)
            if (FeatureFlags.ENABLE_BITRATE_CONTROL or 
                FeatureFlags.ENABLE_ADAPTIVE_QP):
                avg_complexity, avg_qp = processor._analyze_video(video)
            else:
                avg_complexity = 0
                avg_qp = processor.quantizer.config.BASE_QP
            
            # 优化GOP结构
            if FeatureFlags.ENABLE_GOP_OPTIMIZE:
                gop_params = processor.gop_optimizer.optimize_gop_structure(video)
            else:
                gop_params = processor._get_default_gop_params()
            
            # 优化关键帧位置
            if FeatureFlags.ENABLE_KEYFRAME_OPTIMIZE:
                logger.info("开始优化关键帧位置...")
                gop_params = processor.keyframe_optimizer.optimize_keyframe_positions(
                    video, gop_params
                )
                logger.info("关键帧优化完成")
            
            # 获取编码参数（用于最终保存）
            if FeatureFlags.ENABLE_BITRATE_CONTROL:
                encoding_params = processor.bitrate_controller.get_encoding_params(
                    avg_complexity
                )
            else:
                encoding_params = processor._get_default_encoding_params()
            
            # 添加量化参数设置
            encoding_params['qmin'] = str(processor.quantizer.qp_range['min'])
            encoding_params['qmax'] = str(processor.quantizer.qp_range['max'])
            encoding_params['aq-mode'] = '2'  # 启用自适应量化
            encoding_params['aq-strength'] = '1.0'  # 自适应量化强度
            
            logger.info(f"使用编码参数: {encoding_params}")
            
            # 构建ffmpeg参数（用于最终保存）
            ffmpeg_params = [
                '-crf', encoding_params['crf'],
                '-qmin', encoding_params['qmin'],
                '-qmax', encoding_params['qmax'],
                '-aq-mode', encoding_params['aq-mode'],
                '-aq-strength', encoding_params['aq-strength'],
                # GOP相关参数
                '-g', str(int(np.mean(gop_params['gop_sizes']))),  # GOP大小
                '-bf', str(int(np.mean(gop_params['b_frame_counts']))),  # B帧数量
                '-refs', str(gop_params['ref_frames']),  # 参考帧数量
                '-sc_threshold', '40',  # 场景切换检测阈值
                '-keyint_min', str(processor.gop_optimizer.config.MIN_GOP_SIZE)  # 最小GOP大小
            ]
        else:
            # 如果不启用高级编码，使用默认参数
            encoding_params = {
                'codec': 'libx264',
                'bitrate': '3000k',
                'preset': 'medium'
            }
            ffmpeg_params = []
        
        # 步骤3: 视觉效果增强 (process9.py)
        if 3 in enable_steps:
            print(f"步骤3: 视觉效果增强")
            
            # 添加黑色边框
            if Config.ENABLE_BORDER:
                try:
                    video = add_black_border(video)
                except Exception as e:
                    print(f"警告: 添加边框时出错: {str(e)}")
                    # 继续处理，不中断流程
            
            # 添加移动线条
            if Config.ENABLE_LINES:
                try:
                    video = add_moving_line(video)
                except Exception as e:
                    print(f"警告: 添加移动线条时出错: {str(e)}")
                    # 继续处理，不中断流程
        
        # 最终输出文件路径
        final_output = os.path.join(output_folder, f"{original_filename}.mp4")
        
        # 只在最后保存一次视频
        print(f"保存最终视频: {final_output}")
        
        # 使用高级编码参数（如果步骤2启用）或默认参数保存视频
        video.write_videofile(
            final_output,
            codec=encoding_params.get('codec', 'libx264'),
            bitrate=encoding_params.get('bitrate', '3000k'),
            preset=encoding_params.get('preset', 'medium'),
            threads=4,  # 使用多线程
            ffmpeg_params=ffmpeg_params
        )
        
        # 安全关闭视频
        try:
            if hasattr(video, 'close'):
                video.close()
        except:
            pass
        
        # 强制清理
        import gc
        gc.collect()
        
        print(f"视频处理完成: {final_output}")
        return final_output
        
    except Exception as e:
        print(f"处理视频时出错: {str(e)}")
        # 确保视频对象被关闭
        try:
            if 'video' in locals() and video is not None and hasattr(video, 'close'):
                video.close()
        except:
            pass
        raise

def main():
    """主函数"""
    try:
        # 设置输入和输出文件夹
        input_folder = "input"
        output_folder = "output"
        
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
        
        # 处理结果统计
        results = {
            'total': len(video_files),
            'success': 0,
            'failed': 0
        }
        
        # 处理每个视频
        for i, video_file in enumerate(video_files):
            print(f"\n处理视频 {i+1}/{len(video_files)}: {video_file}")
            
            # 检查文件名长度，如果过长则重命名
            if len(os.path.basename(video_file)) > 100:  # 设置一个合理的长度阈值
                file_ext = os.path.splitext(video_file)[1]
                new_filename = f"renamed_{hashlib.md5(os.path.basename(video_file).encode()).hexdigest()[:10]}{file_ext}"
                new_input_path = os.path.join(os.path.dirname(video_file), new_filename)
                print(f"文件名过长，重命名为: {new_filename}")
                # 复制文件而不是重命名，以保留原始文件
                shutil.copy2(video_file, new_input_path)
                video_file = new_input_path
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 执行完整处理流水线
                # 可以通过修改enable_steps参数来控制启用哪些处理步骤
                # 例如: 只进行基础处理 enable_steps={1}
                # 例如: 只进行视觉效果增强 enable_steps={3}
                # 例如: 跳过高级编码优化 enable_steps={1, 3}
                output_file = process_pipeline(video_file, output_folder, enable_steps={1, 2, 3})
                
                # 计算处理时间
                process_time = time.time() - start_time
                
                # 更新统计信息
                results['success'] += 1
                
                print(f"视频 {i+1} 处理成功，耗时: {process_time:.2f}秒")
                
                # 如果是重命名的文件，处理完成后删除
                if "renamed_" in os.path.basename(video_file):
                    try:
                        os.remove(video_file)
                        print(f"已删除临时文件: {video_file}")
                    except Exception as e:
                        print(f"删除临时文件时出错: {str(e)}")
                
            except Exception as e:
                # 更新统计信息
                results['failed'] += 1
                
                print(f"处理视频 {video_file} 时出错: {str(e)}")
                # 记录详细错误信息
                import traceback
                print(f"详细错误信息:\n{traceback.format_exc()}")
                # 继续处理下一个视频
                continue
        
        # 输出处理结果统计
        print("\n处理结果统计:")
        print(f"总视频数: {results['total']}")
        print(f"处理成功: {results['success']}")
        print(f"处理失败: {results['failed']}")
        
        if results['success'] > 0:
            print("\n所有视频处理完成")
        else:
            print("\n警告: 所有视频处理均失败")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        # 记录详细错误信息
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
