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