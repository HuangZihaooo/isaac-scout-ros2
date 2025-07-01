from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from pathlib import Path

# 获取当前脚本的目录，然后向上一级到项目根目录
current_file = Path(__file__)
project_root = current_file.parent.parent  # 向上两级：scout -> isaac-scout-ros2
usd_path = str(project_root / "model" / "scout.usd")

SCOUT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "front_left_wheel": 0.0,
            "front_right_wheel": 0.0,
            "rear_left_wheel": 0.0,
            "rear_right_wheel": 0.0,
        },
        pos=(0.0, 0.0, 0.2),  # Scout放置在地面上方
    ),
    # 四轮差速控制配置
    actuators={
        "wheel_actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel"],  # 匹配所有轮子关节
            effort_limit_sim=100.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,      # 速度控制模式，刚度设为0
            damping=10.0,       # 适当的阻尼
        ),
    },
)

@configclass
class ScoutSceneCfg(InteractiveSceneCfg):
    """Scout场景配置"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", 
        spawn=sim_utils.GroundPlaneCfg()
    )

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Scout机器人
    Scout = SCOUT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Scout")

def calculate_wheel_velocities(linear_vel, angular_vel, wheel_base=0.498, track_width=0.498):
    """
    计算四轮差速小车的轮子速度
    
    Args:
        linear_vel: 线速度 (m/s)
        angular_vel: 角速度 (rad/s)
        wheel_base: 轴距 (前后轮距离)
        track_width: 轮距 (左右轮距离)
    
    Returns:
        四个轮子的速度 [front_left, front_right, rear_left, rear_right]
    """
    # 对于四轮差速，前后轮具有相同的控制逻辑
    # 左轮速度 = 线速度 - 角速度 * 轮距/2
    # 右轮速度 = 线速度 + 角速度 * 轮距/2
    
    left_wheel_vel = linear_vel - angular_vel * track_width / 2.0
    right_wheel_vel = linear_vel + angular_vel * track_width / 2.0
    
    # 四轮差速：前后轮使用相同的控制
    return [left_wheel_vel, right_wheel_vel, left_wheel_vel, right_wheel_vel]
