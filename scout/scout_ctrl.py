import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv

# 全局变量存储基础速度命令
base_vel_cmd_data = None

def init_base_vel_cmd(num_envs):
    """初始化基础速度命令"""
    global base_vel_cmd_data
    base_vel_cmd_data = torch.zeros((num_envs, 3), dtype=torch.float32, device="cuda:0")

def base_vel_cmd(env: ManagerBasedRLEnv):
    """获取基础速度命令"""
    global base_vel_cmd_data
    if base_vel_cmd_data is None:
        init_base_vel_cmd(env.num_envs)
    return base_vel_cmd_data

def get_scout_policy(env_cfg):
    """获取Scout策略（临时实现）"""
    from isaaclab.envs import ManagerBasedRLEnv
    
    # 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 创建简单的零策略（后续可以替换为实际的RL策略）
    def zero_policy(obs):
        return torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    
    return env, zero_policy

def sub_keyboard_event(event, *args, **kwargs):
    """键盘事件处理（临时空实现）"""
    pass 

def get_simple_scout_actions(sim_time, mode="straight"):
    """
    生成简单的Scout控制动作
    
    Args:
        sim_time: 仿真时间
        mode: 运动模式 ("straight", "turn", "wave")
    
    Returns:
        四个轮子的速度控制命令
    """
    if mode == "straight":
        # 直线运动：所有轮子相同速度
        linear_vel = 2.0  # 2 m/s
        angular_vel = 0.0
    elif mode == "turn":
        # 转弯运动：差速转弯
        linear_vel = 1.0  # 1 m/s
        angular_vel = 0.5  # 0.5 rad/s
    elif mode == "wave":
        # 波浪运动：前进 + 正弦转弯
        linear_vel = 1.5
        angular_vel = 0.8 * np.sin(2 * np.pi * 0.5 * sim_time)  # 正弦波转弯
    else:
        linear_vel = 0.0
        angular_vel = 0.0
    
    # 使用我们之前定义的函数计算轮子速度
    from scout.scout_env import calculate_wheel_velocities
    wheel_vels = calculate_wheel_velocities(linear_vel, angular_vel)
    
    return torch.tensor([wheel_vels], dtype=torch.float32)

def get_movement_pattern(count):
    """
    根据计数返回运动模式
    """
    cycle = count % 300  # 每300步一个周期
    
    if cycle < 100:
        return "straight"    # 前100步直线
    elif cycle < 200:
        return "turn"        # 接下来100步转弯
    else:
        return "wave"        # 最后100步波浪运动 