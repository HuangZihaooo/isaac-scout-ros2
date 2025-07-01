from isaacsim.core.utils.prims import define_prim, get_prim_at_path
try:
    import isaacsim.storage.native as nucleus_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.nucleus as nucleus_utils
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter
from isaaclab.terrains import TerrainGeneratorCfg
import omni.replicator.core as rep
from env.terrain_cfg import HfUniformDiscreteObstaclesTerrainCfg

import os
def get_assets_root_path():
    """obtain Isaac Sim assets root path"""
    # 1. try to get from environment variable (recommended)
    env_path = os.environ.get("ISAAC_ASSETS_PATH")
    if env_path and os.path.exists(env_path):
        print(f"Using environment variable ISAAC_ASSETS_PATH: {env_path}")
        return env_path
    
    # 2. try to get from user provided default path
    default_path = "/home/hzzz/isaac/assets/Assets/Isaac/4.5"
    if os.path.exists(default_path):
        print(f"Using default path: {default_path}")
        return default_path

    # 3. throw exception if all methods fail
    raise FileNotFoundError(
        "Cannot find Isaac Sim assets directory. Please try the following solutions:\n"
        "1. Set environment variable: export ISAAC_ASSETS_PATH=/your/path\n" 
        "2. Confirm Isaac Sim is correctly installed\n"
        "3. Check if the path exists: /home/hzzz/isaac/assets/Assets/Isaac/4.5"
    )

def create_obstacle_sparse_env():
    add_semantic_label()
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(1.0, 2.0),
                num_obstacles=100 ,
                obstacles_distance=2.0,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,     
    )
    TerrainImporter(terrain) 

def create_obstacle_medium_env():
    add_semantic_label()
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(1.0, 2.0),
                num_obstacles=200 ,
                obstacles_distance=2.0,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,     
    )
    TerrainImporter(terrain) 


def create_obstacle_dense_env():
    add_semantic_label()
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/obstacleTerrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(50, 50),
            color_scheme="height",
            sub_terrains={"t1": HfUniformDiscreteObstaclesTerrainCfg(
                seed=0,
                size=(50, 50),
                obstacle_width_range=(0.5, 1.0),
                obstacle_height_range=(1.0, 2.0),
                num_obstacles=400,
                obstacles_distance=2.0,
                border_width=5,
                avoid_positions=[[0, 0]]
            )},
        ),
        visual_material=None,     
    )
    TerrainImporter(terrain) 

def add_semantic_label():
    """添加语义标签到地面"""
    ground_plane = rep.get.prims("/World/GroundPlane")
    with ground_plane:
        # 添加语义标签
        rep.modify.semantics([("class", "floor")])

def create_office_env(local_assets):
    """创建办公室仿真环境"""
    add_semantic_label()
    # 获取Isaac资源根路径
    if local_assets:
        assets_root_path = get_assets_root_path()
    else:
        assets_root_path = nucleus_utils.get_assets_root_path()
    # 检查是否已经存在Office prim
    prim = get_prim_at_path("/World/Office")
    # 创建Office prim
    prim = define_prim("/World/Office", "Xform")
    # 设置办公室USD文件路径
    asset_path = assets_root_path + "/Isaac/Environments/Office/office.usd"
    # 添加引用到办公室环境
    prim.GetReferences().AddReference(asset_path)

def create_hospital_env(local_assets):
    """创建医院仿真环境"""
    add_semantic_label()
    if local_assets:
        assets_root_path = get_assets_root_path()
    else:
        assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Hospital")
    prim = define_prim("/World/Hospital", "Xform")
    asset_path = assets_root_path + "/Isaac/Environments/Hospital/hospital.usd"
    prim.GetReferences().AddReference(asset_path)

def create_warehouse_env(local_assets):
    """创建仓库仿真环境"""
    add_semantic_label()
    if local_assets:
        assets_root_path = get_assets_root_path()
    else:
        assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    prim.GetReferences().AddReference(asset_path)

def create_warehouse_forklifts_env(local_assets):
    """创建带叉车的仓库仿真环境"""
    add_semantic_label()
    if local_assets:
        assets_root_path = get_assets_root_path()
    else:
        assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
    prim.GetReferences().AddReference(asset_path)

def create_warehouse_shelves_env(local_assets):
    """创建带货架的仓库仿真环境"""
    add_semantic_label()
    if local_assets:
        assets_root_path = get_assets_root_path()
    else:
        assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
    prim.GetReferences().AddReference(asset_path)

def create_full_warehouse_env(local_assets):
    """创建完整仓库仿真环境"""
    add_semantic_label()
    if local_assets:
        assets_root_path = get_assets_root_path()
    else:
        assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
    prim.GetReferences().AddReference(asset_path)
