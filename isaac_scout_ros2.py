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

# 追加AppLauncher cli参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余所有内容如下"""

import torch

# 导入Scout相关模块
from scout.scout_env import ScoutRSLEnvCfg, camera_follow
import env.sim_env as sim_env
# import scout.scout_sensors as scout_sensors
import omni
import carb
import scout.scout_ctrl as scout_ctrl
# import ros2.scout_ros2_bridge as scout_ros2_bridge
# 导入isaaclab相关模块
from isaaclab.sim import SimulationCfg, SimulationContext

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    """运行仿真器主函数"""
    
    # Scout环境设置
    scout_env_cfg = ScoutRSLEnvCfg()
    # scout_env_cfg.scene.num_envs = cfg.num_envs
    scout_env_cfg.decimation = math.ceil(1./scout_env_cfg.sim.dt/cfg.freq)
    scout_env_cfg.sim.render_interval = scout_env_cfg.decimation
    # scout_ctrl.init_base_vel_cmd(cfg.num_envs)
    # env, policy = scout_ctrl.get_scout_policy(scout_env_cfg)
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])


    # 仿真环境选择（在环境创建后加载）
    if (cfg.env_name == "obstacle-dense"):
        sim_env.create_obstacle_dense_env()  # 障碍物密集环境
    elif (cfg.env_name == "office"):
        sim_env.create_office_env(cfg.local_assets)  # 办公室环境
    elif (cfg.env_name == "hospital"):
        sim_env.create_hospital_env(cfg.local_assets)  # 医院环境  
    elif (cfg.env_name == "warehouse"):
        sim_env.create_warehouse_env(cfg.local_assets)  # 仓库环境
    elif (cfg.env_name == "warehouse-forklifts"):
        sim_env.create_warehouse_forklifts_env(cfg.local_assets)  # 带叉车的仓库环境
    elif (cfg.env_name == "warehouse-shelves"):
        sim_env.create_warehouse_shelves_env(cfg.local_assets)  # 带货架的仓库环境
    elif (cfg.env_name == "full-warehouse"):
        sim_env.create_full_warehouse_env(cfg.local_assets)  # 完整仓库环境

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # 传感器设置
    # sm = scout_sensors.SensorManager(cfg.num_envs)
    # lidar_annotators = sm.add_rtx_lidar()
    # cameras = sm.add_camera(cfg.freq)

    # 键盘控制
    # system_input = carb.input.acquire_input_interface()
    # system_input.subscribe_to_keyboard_events(
    #     omni.appwindow.get_default_app_window().get_keyboard(), scout_ctrl.sub_keyboard_event)
    
    # ROS2桥接
    rclpy.init()
    # dm = scout_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg)

    # 运行仿真
    sim_step_dt = float(scout_env_cfg.sim.dt * scout_env_cfg.decimation)
    # obs, _ = env.reset()
    
    print("仿真环境已启动，使用环境:", cfg.env_name)
    
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():            
            # 控制关节
            # actions = policy(obs)

            # 步进环境
            # obs, _, _, _ = env.step(actions)

            # ROS2数据发布
            # dm.pub_ros2_data()
            # rclpy.spin_once(dm)

            # 相机跟随
            # if (cfg.camera_follow):
                # camera_follow(env)

            # 限制循环时间
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, sim_step_dt/elapsed_time)
        
        # perform step
        sim.step()
        print(f"\r步进时间: {actual_loop_time*1000:.2f}ms, 实时因子: {rtf:.2f}", end='', flush=True)
    
    # dm.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    run_simulator()
    