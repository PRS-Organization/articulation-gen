import trimesh
import numpy as np
from trimesh.proximity import ProximityQuery
from trimesh.collision import CollisionManager


def load_model(path):
    """加载并合并模型（处理Scene对象）"""
    model = trimesh.load(path)
    if isinstance(model, trimesh.Scene):
        # 提取所有子Mesh并合并
        meshes = list(model.geometry.values())
        if not meshes:
            raise ValueError("模型中没有几何体")
        return trimesh.util.concatenate(meshes)
    return model


def load_mesh(file_path):
    loaded = load_model(file_path)
    if isinstance(loaded, trimesh.Scene):
        return load_model(loaded.file_name)
    return loaded


# def load_and_simplify_mesh(file_path, ratio=0.3):
#     """
#     加载并简化网格（按百分比减少面数）。
#
#     参数:
#     file_path (str): 模型文件路径
#     ratio (float): 保留的面数百分比（0到1之间，默认0.3）
#
#     返回:
#     trimesh.Trimesh: 简化后的网格
#     """
#     # 加载模型并提取 Trimesh 对象
#     loaded = trimesh.load(file_path)
#     if isinstance(loaded, trimesh.Scene):
#         mesh = list(loaded.geometry.values())[0]
#     else:
#         mesh = loaded



def check_intersection_and_distance(mesh1, mesh2):
    """
    计算两个网格的交集和距离（兼容 trimesh 4.4.9）。

    参数:
    mesh1 (trimesh.Trimesh): 第一个网格
    mesh2 (trimesh.Trimesh): 第二个网格

    返回:
    tuple: (has_intersection: bool, distance: float)
        - has_intersection: 是否有交集（通过顶点距离判断）
        - distance: 若无交集则为最近距离（单位：米），否则为0
    """
    # 创建 ProximityQuery 对象
    query_mesh2 = ProximityQuery(mesh2)
    query_mesh1 = ProximityQuery(mesh1)

    # 1. 判断是否有交集（通过符号距离）
    # 计算 mesh1 的顶点到 mesh2 的符号距离
    distances1 = query_mesh2.signed_distance(mesh1.vertices)
    has_intersection = np.any(distances1 < 0)  # 任何负值表示顶点在内部

    if not has_intersection:
        # 计算 mesh2 的顶点到 mesh1 的符号距离
        distances2 = query_mesh1.signed_distance(mesh2.vertices)
        has_intersection = np.any(distances2 < 0)

    if has_intersection:
        return (True, 0.0)

    # 2. 计算无交集时的最小距离
    # 取两个方向的最小正距离
    min_dist1 = np.min(np.abs(distances1)) if not np.all(distances1 < 0) else np.inf
    min_dist2 = np.min(np.abs(query_mesh1.signed_distance(mesh2.vertices))) if not np.all(distances1 < 0) else np.inf
    min_distance = min(min_dist1, min_dist2)

    return (False, min_distance)


def check_collision(mesh1, mesh2):
    """
    判断两个网格是否发生碰撞/接触。

    参数:
    mesh1 (trimesh.Trimesh): 第一个网格
    mesh2 (trimesh.Trimesh): 第二个网格

    返回:
    bool: True 表示有碰撞，False 表示无碰撞
    """
    # 创建碰撞管理器
    manager = CollisionManager()

    # 将两个网格添加到管理器中
    manager.add_object("mesh1", mesh1)
    manager.add_object("mesh2", mesh2)

    # 检查内部碰撞（两个网格之间的碰撞）
    return manager.in_collision_internal()


def is_mesh2_in_mesh1(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh) -> bool:
    """
    判断 mesh2 的包围盒是否完全在 mesh1 的包围盒内。

    参数:
    mesh1 (trimesh.Trimesh): 第一个网格
    mesh2 (trimesh.Trimesh): 第二个网格

    返回:
    bool: True 表示 mesh2 的包围盒完全在 mesh1 内，False 表示否
    """
    # 获取两个网格的轴对齐包围盒（AABB）的 bounds（min, max）
    box1_min, box1_max = mesh1.bounding_box.bounds
    box2_min, box2_max = mesh2.bounding_box.bounds

    # 判断 mesh2 的包围盒是否在 mesh1 的包围盒范围内
    # 所有维度的 min 必须 >= mesh1 的 min，max 必须 <= mesh1 的 max
    return (
            (box2_min >= box1_min).all() and  # mesh2 的最小坐标 >= mesh1 的最小坐标
            (box2_max <= box1_max).all()  # mesh2 的最大坐标 <= mesh1 的最大坐标
    )


if __name__ == '__main__':

    # 加载 GLB 文件
    mesh1 = load_mesh("-1.glb")
    mesh2 = load_mesh("5.glb")
    mesh1 = mesh1.simplify_quadric_decimation(percent=0.3)
    mesh2 = mesh2.simplify_quadric_decimation(percent=0.3)
    # 检测碰撞
    has_collision = check_collision(mesh1, mesh2)

    print(f"是否有碰撞/接触: {has_collision}")
    # mesh1 = load_and_simplify_mesh("-microwave_1_button_button_0_ty_black.glb", ratio=0.5)
    # mesh2 = load_and_simplify_mesh("microwave_1_button_button_0_ty_black.glb", ratio=0.5)
    # # 合并两个网格为一个场景
    # scene = trimesh.Scene()
    # scene.add_geometry(mesh1, node_name="model1")
    # scene.add_geometry(mesh2, node_name="model2")
    #
    # # 显示坐标轴（通过快捷键 `a` 切换）
    # scene.show()

    # --- 计算位移向量 ---
    # 指定法向量（用户提供的方向）
    normal_vector = np.array([0.03975599, -0.00465019, 0.9991986])

    # 归一化法向量（确保单位向量）
    unit_dir = normal_vector / np.linalg.norm(normal_vector)

    # 位移距离（0.1 米）
    distance = - 0.1

    # 计算位移向量
    translation_vector = unit_dir * distance

    # --- 应用平移 ---
    # 创建移动后的 mesh2 副本（保留原始 mesh2）
    moved_mesh2 = mesh2.copy()
    moved_mesh2.apply_translation(translation_vector)
    # print(check_intersection_and_distance(mesh1, mesh2))
    # print(check_intersection_and_distance(mesh1, moved_mesh2))
    has_collision = check_collision(mesh1, mesh2)

    print(f"是否有碰撞/接触: {has_collision}")
    has_collision = check_collision(mesh1, moved_mesh2)

    print(f"是否有碰撞/接触: {has_collision}")
    # --- 可视化对比 ---
    # 创建场景合并两个模型（原始 mesh2 和移动后的 moved_mesh2）
    # 创建场景并添加几何体
    scene = trimesh.Scene()
    scene.add_geometry(mesh1, node_name="model1")
    scene.add_geometry(mesh2, node_name="original_mesh2")
    scene.add_geometry(moved_mesh2, node_name="moved_mesh2")

    # 显示坐标轴（按 `a` 快捷键切换）
    scene.show()

    # 检测是否包含
    result = is_mesh2_in_mesh1(mesh1, mesh2)
    print(f"mesh2 是否在 mesh1 的包围盒内: {result}")

    # 检测是否包含
    result = is_mesh2_in_mesh1(mesh1, moved_mesh2)
    print(f"mesh2 是否在 mesh1 的包围盒内: {result}")