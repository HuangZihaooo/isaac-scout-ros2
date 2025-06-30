import os
import hydra
import rclpy
import torch
import time
import math
import argparse
from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole Navigation environment.")

# 添加 AppLauncher 的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析命令行参数
args_cli = parser.parse_args()

# 使用参数启动应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):

    while simulation_app.is_running():
        print("Simulation is running")
        time.sleep(1)
    simulation_app.close()
            
if __name__ == "__main__":
    run_simulator()
    