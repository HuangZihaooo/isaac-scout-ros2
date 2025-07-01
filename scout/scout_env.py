from isaaclab.scene import InteractiveSceneCfg
# from isaaclab_assets.robots.agilex import AGILEX_SCOUT_2_CFG

from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg
from isaacsim.core.utils.viewports import set_camera_view
import numpy as np
from scipy.spatial.transform import Rotation as R
import scout.scout_ctrl as scout_ctrl


@configclass
class ScoutSimCfg(InteractiveSceneCfg):
    """Scout仿真场景配置"""
    
    # # 地面
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300.0, 300.0)),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0, 0, 1e-4)
    #     )
    # )
    
    # # 灯光设置
    # light = AssetBaseCfg(
    #     prim_path="/World/Light",
    #     spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    # )
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/DomeLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    # )

    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # Scout机器人配置
    # agilex_scout: ArticulationCfg = AGILEX_SCOUT_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Scout")
    
    # Scout接触传感器（如果需要的话）
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Scout/.*_wheel", history_length=3, track_air_time=True)

    # Scout高度扫描仪
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Scout/base_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20)), 
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), 
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )

@configclass
class ActionsCfg:
    """环境动作配置"""
    joint_pos = mdp.JointPositionActionCfg(asset_name="agilex_scout", joint_names=[".*"])

@configclass
class ObservationsCfg:
    """观察配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略组观察"""

        # 观察项（保持顺序）
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,
                               params={"asset_cfg": SceneEntityCfg(name="agilex_scout")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,
                               params={"asset_cfg": SceneEntityCfg(name="agilex_scout")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    params={"asset_cfg": SceneEntityCfg(name="agilex_scout")},
                                    noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
        # 速度命令
        base_vel_cmd = ObsTerm(func=scout_ctrl.base_vel_cmd)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                            params={"asset_cfg": SceneEntityCfg(name="agilex_scout")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,
                            params={"asset_cfg": SceneEntityCfg(name="agilex_scout")})
        actions = ObsTerm(func=mdp.last_action)
        
        # 高度扫描
        height_scan = ObsTerm(func=mdp.height_scan,
                              params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                              clip=(-1.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # 观察组
    policy: PolicyCfg = PolicyCfg()

@configclass
class CommandsCfg:
    """MDP命令配置"""
    base_vel_cmd = mdp.UniformVelocityCommandCfg(
        asset_name="agilex_scout",
        resampling_time_range=(0.0, 0.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )

@configclass
class EventCfg:
    """事件配置"""
    pass

@configclass
class RewardsCfg:
    """奖励配置"""
    pass

@configclass
class TerminationsCfg:
    """终止条件配置"""
    pass

@configclass
class CurriculumCfg:
    """课程配置"""
    pass

@configclass
class ScoutRSLEnvCfg(ManagerBasedRLEnvCfg):
    """Scout环境配置"""
    # 场景设置
    scene = ScoutSimCfg(num_envs=2, env_spacing=2.0)

    # 基本设置
    # observations = ObservationsCfg()
    # actions = ActionsCfg()
    
    # 虚拟设置
    # commands = CommandsCfg()
    # rewards = RewardsCfg()
    # terminations = TerminationsCfg()
    # events = EventCfg()
    # curriculum = CurriculumCfg()

    def __post_init__(self):
        # 查看器设置
        self.viewer.eye = [-4.0, 0.0, 5.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        # 步进设置
        self.decimation = 8  # 步进

        # 仿真设置
        self.sim.dt = 0.005  # 仿真步进每
        self.sim.render_interval = self.decimation  
        self.sim.disable_contact_processing = True
        self.sim.render.antialiasing_mode = None

        # RSL环境控制设置
        self.episode_length_s = 20.0  # 可以忽略
        self.is_finite_horizon = False
        # self.actions.joint_pos.scale = 0.25

        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt

def camera_follow(env):
    """相机跟随功能"""
    if (env.unwrapped.scene.num_envs == 1):
        robot_position = env.unwrapped.scene["agilex_scout"].data.root_state_w[0, :3].cpu().numpy()
        robot_orientation = env.unwrapped.scene["agilex_scout"].data.root_state_w[0, 3:7].cpu().numpy()
        rotation = R.from_quat([robot_orientation[1], robot_orientation[2], 
                                robot_orientation[3], robot_orientation[0]])
        yaw = rotation.as_euler('zyx')[0]
        yaw_rotation = R.from_euler('z', yaw).as_matrix()
        set_camera_view(
            yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position,
            robot_position
        )
