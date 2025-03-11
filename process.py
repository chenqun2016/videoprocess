import os
import subprocess
import sys

# 全局开关参数，控制是否运行各个脚本
class ProcessConfig:
    # 各个处理脚本的开关
    RUN_PROCESS7 = True        # 是否运行process7.py
    RUN_PROCESS8 = True        # 是否运行process8.py
    RUN_PROCESS9 = True        # 是否运行process9.py

    RUN_GET_FIRST_FRAME = False # 是否运行getFirstFrame.py
    RUN_GET_LAST_FRAME = False  # 是否运行getLastFrame.py
    RUN_CUT_VIDEO = False       # 是否运行cutVideo/cutVideo.py
    
    # 各个脚本的路径
    PROCESS7_PATH = "process7.py"
    PROCESS8_PATH = "process8.py"
    GET_FIRST_FRAME_PATH = "getFirstFrame.py"
    GET_LAST_FRAME_PATH = "getLastFrame.py"
    CUT_VIDEO_PATH = os.path.join("cutVideo", "cutVideo.py")
    PROCESS9_PATH = "process9.py"

def run_script(script_path, script_name):
    """运行指定的Python脚本并打印状态信息"""
    print(f"\n正在执行 {script_name}...")
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"\n{script_name} 执行完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{script_name} 执行失败，返回码: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n{script_name} 执行出错: {str(e)}")
        return False

def main():
    """主函数，按顺序执行各个处理脚本"""
    config = ProcessConfig()
    
    # 1. 运行 process7.py
    if config.RUN_PROCESS7:
        if not run_script(config.PROCESS7_PATH, "process7.py"):
            print("process7.py 执行失败，流程中断")
            return
    else:
        print("\n已跳过 process7.py")
    
    # 2. 运行 process8.py
    if config.RUN_PROCESS8:
        if not run_script(config.PROCESS8_PATH, "process8.py"):
            print("process8.py 执行失败，流程中断")
            return
    else:
        print("\n已跳过 process8.py")

    # 3. 运行 process9.py
    if config.RUN_PROCESS9:
        if not run_script(config.PROCESS9_PATH, "process9.py"):
            print("process9.py 执行失败，流程中断")
            return
    else:
        print("\n已跳过 process9.py")
    
    
    # 4. 运行 getFirstFrame.py
    if config.RUN_GET_FIRST_FRAME:
        if not run_script(config.GET_FIRST_FRAME_PATH, "getFirstFrame.py"):
            print("getFirstFrame.py 执行失败，流程中断")
            return
    else:
        print("\n已跳过 getFirstFrame.py")
    
    # 5. 运行 getLastFrame.py
    if config.RUN_GET_LAST_FRAME:
        if not run_script(config.GET_LAST_FRAME_PATH, "getLastFrame.py"):
            print("getLastFrame.py 执行失败，流程中断")
            return
    else:
        print("\n已跳过 getLastFrame.py")
    
    # 6. 运行 cutVideo/cutVideo.py
    if config.RUN_CUT_VIDEO:
        if not run_script(config.CUT_VIDEO_PATH, "cutVideo/cutVideo.py"):
            print("cutVideo/cutVideo.py 执行失败，流程中断")
            return
    else:
        print("\n已跳过 cutVideo/cutVideo.py")
    
    print("\n所有处理脚本执行完毕！")


if __name__ == "__main__":
    main()
