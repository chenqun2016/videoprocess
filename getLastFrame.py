import os
import cv2
import shutil

def clear_directory(path):
    """清空指定目录下的所有文件"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_last_frame(video_path, save_path):
    """获取视频的最后一帧并保存为图片"""
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return False
        
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"视频帧数无效: {video_path}")
            return False
        
        # 设置读取位置到最后一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        
        # 读取最后一帧
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取视频尾帧: {video_path}")
            return False
        
        # 保存帧为图片
        cv2.imwrite(save_path, frame)
        
        # 释放视频对象
        cap.release()
        return True
    except Exception as e:
        print(f"处理视频时出错 {video_path}: {str(e)}")
        return False

def main():
    # 定义目录路径
    video_dir = "input"
    picture_dir = "picture"
    
    # 清空picture文件夹
    clear_directory(picture_dir)
    
    # 支持的视频格式
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
    
    # 遍历视频目录
    for filename in os.listdir(video_dir):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(video_dir, filename)
            # 生成图片保存路径（使用同名但改为.jpg后缀）
            picture_name = os.path.splitext(filename)[0] + '_last.jpg'
            picture_path = os.path.join(picture_dir, picture_name)
            
            # 获取并保存尾帧
            if get_last_frame(video_path, picture_path):
                print(f"成功处理视频: {filename}")
            else:
                print(f"处理视频失败: {filename}")

if __name__ == "__main__":
    main()
