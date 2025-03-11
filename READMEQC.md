python process7.py

视频基础去重：

# 视频处理工具 (process7.py)

这是一个功能强大的视频处理工具，可以对视频进行多种效果处理，包括MD5修改、静音、帧处理、特效等。

## 功能特点

1. **MD5修改**
   - 通过修改最后一帧实现MD5值的改变
   - 不影响视频的实际内容和观看体验

2. **音频处理**
   - 视频静音功能

3. **帧处理**
   - 删除视频开头和结尾指定帧数
   - 随机间隔丢帧
   - 帧率调整

4. **视频效果**
   - 镜像翻转
   - 视频加速
   - 颜色调整
   - 视频旋转（自动处理黑边）
   - 随机区域模糊
   - 淡入淡出效果（支持自定义透明度）

## 配置参数

### 视频编码配置 (VideoConfig)
```python
CODEC = 'libx264'          # 视频编码器
PRESET = 'medium'          # 编码速度预设（ultrafast到veryslow）
CRF = "23"                # 视频质量控制（0-51，越小质量越好）
PROFILE = "high"          # 编码配置（baseline/main/high）
LEVEL = "4.0"             # 编码等级（3.0/3.1/4.0/4.1/4.2）
PIXEL_FORMAT = "yuv420p"  # 像素格式
```

### 帧处理配置 (FrameConfig)
```python
REMOVE_HEAD_FRAMES = 10    # 删除开头帧数（建议：5-15）
REMOVE_TAIL_FRAMES = 10    # 删除结尾帧数（建议：5-15）
DROP_INTERVAL_MIN = 30     # 最小丢帧间隔（建议：30-45）
DROP_INTERVAL_MAX = 60     # 最大丢帧间隔（建议：45-60）
DROP_COUNT_MIN = 1         # 每次最少丢弃帧数（建议：1-2）
DROP_COUNT_MAX = 2         # 每次最多丢弃帧数（建议：2-3）
FPS_ADJUST_MIN = -2.0      # 帧率调整最小值（建议：-3到0）
FPS_ADJUST_MAX = 2.0       # 帧率调整最大值（建议：0到3）
```

### 效果配置 (EffectConfig)
```python
SPEED_FACTOR = 1.1        # 视频加速倍数（建议：1.0-1.5）
COLOR_FACTOR_MIN = 0.95   # 颜色调整最小倍数（建议：0.9-1.0）
COLOR_FACTOR_MAX = 1.05   # 颜色调整最大倍数（建议：1.0-1.1）
ROTATE_ANGLE_MIN = -2.0   # 最小旋转角度（建议：-5到0度）
ROTATE_ANGLE_MAX = 2.0    # 最大旋转角度（建议：0到5度）
ROTATE_SCALE_PADDING = 0.002  # 旋转放大补偿（建议：0.001-0.005）
ROTATE_MAX_SCALE = 1.1    # 最大放大比例
BLUR_REGION_MIN = 0.1     # 模糊区域最小比例（建议：0.05-0.15）
BLUR_REGION_MAX = 0.2     # 模糊区域最大比例（建议：0.15-0.25）
BLUR_SIGMA = 1.0          # 高斯模糊强度（建议：0.5-2.0）
```

### 过渡效果配置 (TransitionConfig)
```python
FADE_DURATION = 0.74      # 淡入淡出持续时间（建议：0.5-1.0秒）
COUNT_MIN = 1             # 最少插入过渡帧数（建议：1-2）
COUNT_MAX = 2             # 最多插入过渡帧数（建议：2-3）
DURATION = 0.1            # 中间过渡效果持续时间（建议：0.1-0.3秒）
```

### 边缘裁剪配置 (EdgeCropConfig)
```python
BASE = 0.01              # 基础裁剪比例
ANGLE_FACTOR = 0.01      # 角度系数（最终裁剪比例 = 基础比例 + |角度| * 角度系数）
MAX = 0.05               # 最大裁剪比例
```

### 功能开关配置
```python
ENABLE_MD5_MODIFY = True    # MD5修改功能
ENABLE_MUTE = True          # 静音功能
ENABLE_FRAME_DROP = True    # 删帧功能
ENABLE_MIRROR = True        # 镜像翻转功能
ENABLE_SPEED = True         # 加速功能
ENABLE_RESIZE = True        # 放大功能
ENABLE_COLOR = True         # 颜色调整功能
ENABLE_ROTATE = True        # 旋转功能
ENABLE_BLUR = True          # 模糊功能
ENABLE_FPS = True           # 帧率调整功能
ENABLE_TRANSITIONS = True    # 过渡帧功能
ENABLE_EDGE_CROP = True     # 边缘裁剪功能
```

## 使用方法

1. **准备工作**
   - 创建`input`文件夹，放入要处理的视频文件
   - 确保视频格式为mp4、avi或mov

2. **运行程序**
   ```bash
   python process7.py
   ```

3. **处理流程**
   - 程序会自动创建`videos_cut`输出文件夹
   - 依次处理`input`文件夹中的所有视频
   - 处理后的视频将保存在`videos_cut`文件夹中
   - 输出文件名为随机5位数字

4. **处理步骤**
   1. 验证配置参数
   2. 加载视频文件
   3. 按配置的功能顺序处理视频
   4. 保存处理后的视频
   5. 自动清理资源

## 错误处理

程序包含完善的错误处理机制：

1. **参数验证**
   - 视频文件有效性检查
   - 配置参数范围检查
   - 输入输出路径检查

2. **异常类型**
   - `VideoProcessError`: 基础错误类
   - `InvalidParameterError`: 参数无效错误
   - `ProcessingError`: 处理过程错误

3. **错误信息**
   - 详细的错误原因说明
   - 处理步骤耗时统计
   - 失败原因追踪

## 注意事项

1. 确保有足够的磁盘空间
2. 视频处理可能需要较长时间
3. 可以通过修改配置参数调整处理效果
4. 建议在处理大量视频前先测试单个视频
5. 如遇到内存不足，可以调整视频编码参数

## 性能优化

1. 使用numpy向量化操作
2. 优化的帧处理算法
3. 高效的视频旋转和缩放
4. 智能的内存管理
5. 并行处理支持
