import json
import numpy as np
import open3d as o3d


def convert_trimesh_to_blender(points):
    """
    将 Trimesh 坐标系的点转换为 Blender 坐标系
    :param points: numpy数组，形状为 (N, 3)，每个元素为 [x, y, z]
    :return: 转换后的 numpy数组，形状为 (N, 3)
    """
    # 创建转换后的点数组
    blender_points = np.zeros_like(points)

    # 直接赋值转换
    blender_points[:, 0] = points[:, 0]  # X 轴保持不变
    blender_points[:, 1] = -points[:, 2]  # Y 轴 = -Trimesh 的 Z 轴
    blender_points[:, 2] = points[:, 1]  # Z 轴 = Trimesh 的 Y 轴

    return blender_points


def save_and_visualize_points(points, output_ply_path="output/points.ply", show=True):
    """
    将点云保存为 PLY 文件并可视化
    :param points: numpy数组，形状为 (N, 3)，转换后的点坐标（如 Blender 或 Trimesh 坐标）
    :param output_ply_path: PLY 文件保存路径
    :param show: 是否显示点云（默认 True）
    """
    # 创建 Open3D 的 PointCloud 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 保存为 PLY 文件
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"点云已保存至：{output_ply_path}")

    # 可视化（可选）
    if show:
        o3d.visualization.draw_geometries([pcd],
                                          window_name="Point Cloud Visualization",
                                          width=1000, height=800)


def fit_plane_and_normal(obj_blender_points,
                         distance_threshold=0.01,
                         ransac_n=3,
                         num_iterations=1000):
    """
    拟合点云平面并计算法向量方向
    :param obj_blender_points: numpy数组，形状为(N,3)，Blender坐标系下的点云
    :param distance_threshold: RANSAC平面拟合的距离阈值（默认0.01米）
    :param ransac_n: 每次采样的点数（平面拟合至少需要3个点）
    :param num_iterations: RANSAC迭代次数
    :return: 法向量方向的单位向量（numpy数组，形状(3,)）
    """
    # 创建Open3D的PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_blender_points)

    # 使用RANSAC拟合平面
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    # 提取平面方程参数：ax + by + cz + d = 0
    a, b, c, d = plane_model

    # 计算法向量并单位化
    normal_vector = np.array([a, b, c])
    normal_unit = normal_vector / np.linalg.norm(normal_vector)

    # 打印结果
    print("=== 平面拟合结果 ===")
    print(f"平面方程：{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"法向量（单位化）：{normal_unit}")
    print(f"共{len(obj_blender_points)}个点，其中{len(inliers)}个被选为内点")

    return normal_unit


def normalize_normal_to_axis(normal_dir, base_blender_points, axis=True):
    """
    将法向量标准化到最近的坐标轴方向，并根据点云中心调整方向
    :param normal_dir: numpy数组，形状(3,)，原始法向量方向（单位向量）
    :param base_blender_points: numpy数组，形状(N,3)，点云数据
    :param axis: bool，是否进行坐标轴标准化
    :return: numpy数组，形状(3,)，标准化后的方向轴（如 [0,1,0] 或 [-1,0,0]）
    """
    if not axis:
        return normal_dir  # 直接返回原方向（单位向量）

    # 计算点云中心点
    center = np.mean(base_blender_points, axis=0)

    # 找到与法向量最接近的坐标轴
    x_abs = abs(normal_dir[0])
    y_abs = abs(normal_dir[1])
    z_abs = abs(normal_dir[2])

    max_val = max(x_abs, y_abs, z_abs)

    if max_val == x_abs:
        axis_dir = np.array([np.sign(normal_dir[0]), 0, 0])
    elif max_val == y_abs:
        axis_dir = np.array([0, np.sign(normal_dir[1]), 0])
    else:
        axis_dir = np.array([0, 0, np.sign(normal_dir[2])])

    # 确保是单位向量（虽然已经是±1）
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 判断方向是否指向中心点，若指向则反转方向
    dot_with_center = np.dot(axis_dir, center)
    if dot_with_center > 0:
        print("other direction")
        axis_dir *= -1  # 反转方向，使其远离中心点

    # 返回整数形式的方向轴（如 [0,1,0]）
    return axis_dir.astype(int)


if __name__ == '__main__':
    file_path = "points.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    base_points = data["base"]
    obj_points = data["object"]
    points_array = np.array(base_points)
    base_blender_points = convert_trimesh_to_blender(points_array)
    points_array = np.array(obj_points)
    obj_blender_points = convert_trimesh_to_blender(points_array)
    # save_and_visualize_points(
    #     points=obj_blender_points,
    #     output_ply_path="blender_points.ply",
    #     show=True
    # )

    normal_dir = fit_plane_and_normal(obj_blender_points)
    print("\n法向量方向（返回值）：", normal_dir)

    result = normalize_normal_to_axis(normal_dir, base_blender_points, axis=True)
    print("标准化后的方向轴：", result)