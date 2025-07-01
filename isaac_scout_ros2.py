import os
import hydra
import rclpy
import torch
import time
import math
import argparse
from isaaclab.app import AppLauncher

# 添加argparse参数
parser = argparse.ArgumentParser(description="Scout机器人仿真环境运行程序")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# 追加AppLauncher cli参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余所有内容如下"""

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from scout.scout_env import ScoutSceneCfg
from scout.scout_ctrl import get_simple_scout_actions, get_movement_pattern
import env.sim_env as sim_env

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    """运行仿真器主函数"""
    
    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    
    # 创建场景
    scene_cfg = ScoutSceneCfg(num_envs=cfg.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 加载环境（在场景创建后）
    print("正在加载仿真环境:", cfg.env_name)
    if (cfg.env_name == "office"):
        sim_env.create_office_env(cfg.local_assets)
        print("已成功加载办公室环境")
    elif (cfg.env_name == "hospital"):
        sim_env.create_hospital_env(cfg.local_assets)
        print("已成功加载医院环境")
    elif (cfg.env_name == "warehouse"):
        sim_env.create_warehouse_env(cfg.local_assets)
        print("已成功加载仓库环境")
    elif (cfg.env_name == "warehouse-forklifts"):
        sim_env.create_warehouse_forklifts_env(cfg.local_assets)
        print("已成功加载带叉车的仓库环境")
    elif (cfg.env_name == "warehouse-shelves"):
        sim_env.create_warehouse_shelves_env(cfg.local_assets)
        print("已成功加载带货架的仓库环境")
    elif (cfg.env_name == "full-warehouse"):
        sim_env.create_full_warehouse_env(cfg.local_assets)
        print("已成功加载完整仓库环境")
    
    # 重置仿真
    sim.reset()
    print("[INFO]: Setup complete...")
    
    # 初始化ROS2
    rclpy.init()
    
    # 运行仿真循环
    run_simulation_loop(sim, scene)
    
    # 清理资源
    rclpy.shutdown()
    simulation_app.close()
    print("仿真已关闭")

def run_simulation_loop(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """运行仿真循环"""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    print("开始Scout运动仿真...")
    print("运动模式: 直线 -> 转弯 -> 波浪运动 (循环)")
    
    while simulation_app.is_running():
        # 每500步重置一次
        if count % 500 == 0:
            # 重置计数器
            if count > 0:  # 避免第一次输出
                print(f"\n[INFO]: 重置Scout状态... (仿真时间: {sim_time:.1f}s)")
            count = 0
            
            # 重置Scout到初始状态
            root_state = scene["Scout"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            
            # 设置Scout的位置和速度
            scene["Scout"].write_root_pose_to_sim(root_state[:, :7])
            scene["Scout"].write_root_velocity_to_sim(root_state[:, 7:])
            
            # 重置关节状态
            joint_pos, joint_vel = (
                scene["Scout"].data.default_joint_pos.clone(),
                scene["Scout"].data.default_joint_vel.clone(),
            )
            scene["Scout"].write_joint_state_to_sim(joint_pos, joint_vel)
            
            # 清理内部缓冲区
            scene.reset()
        
        # 获取当前运动模式
        movement_mode = get_movement_pattern(count)
        
        # 生成Scout控制动作
        action = get_simple_scout_actions(sim_time, movement_mode)
        
        # 显示当前状态（每50步显示一次）
        if count % 50 == 0:
            print(f"\r时间: {sim_time:.1f}s, 模式: {movement_mode:8s}, "
                  f"轮速: [{action[0][0]:.1f}, {action[0][1]:.1f}, {action[0][2]:.1f}, {action[0][3]:.1f}]", 
                  end='', flush=True)
        
        # 应用控制命令到Scout
        scene["Scout"].set_joint_velocity_target(action)
        
        # 更新仿真
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)
        
        # 简单的时间控制
        time.sleep(0.01)

if __name__ == "__main__":
    run_simulator()
    