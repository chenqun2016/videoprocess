import os
import cv2
import shutil

def clear_directory(path):
    """清空指定目录下的所有文件"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_first_frame(video_path, save_path):
    """获取视频的第一帧并保存为图片"""
    try:
        # 确保路径格式正确
        video_path = os.path.abspath(video_path)
        save_path = os.path.abspath(save_path)
        
        print(f"处理视频文件: {video_path}")
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return False
        
        # 读取第一帧
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"无法读取视频帧: {video_path}")
            return False
            
        # 检查帧的有效性
        if frame.size == 0:
            print("帧数据为空")
            return False
            
        print(f"帧大小: {frame.shape}")
        print(f"尝试保存图片到: {save_path}")
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存帧为图片
        save_success = cv2.imwrite(save_path, frame)
        if not save_success:
            print(f"保存图片失败: {save_path}")
            return False
        
        print(f"成功保存图片到: {save_path}")
        # 释放视频对象
        cap.release()
        return True
    except Exception as e:
        print(f"处理视频时出错 {video_path}: {str(e)}")
        return False

def main():
    # 定义目录路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_dir, "input")
    picture_dir = os.path.join(base_dir, "picture")
    
    # 清空picture文件夹
    clear_directory(picture_dir)
    
    # 支持的视频格式
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
    
    # 遍历视频目录
    for idx, filename in enumerate(os.listdir(video_dir), 1):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(video_dir, filename)
            # 使用简单的数字编号作为输出文件名
            picture_path = os.path.join(picture_dir, f"frame_{idx:03d}.jpg")
            
            # 获取并保存首帧
            if get_first_frame(video_path, picture_path):
                print(f"成功处理视频: {filename}")
            else:
                print(f"处理视频失败: {filename}")

if __name__ == "__main__":
    main()
