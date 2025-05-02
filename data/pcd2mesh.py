import open3d as o3d
import numpy as np
import os


def read_point_cloud(file_path):
    """读取PLY点云文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在！")
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"点云信息：{np.array(pcd.points).shape} 个点")
    return pcd


def estimate_normals(pcd, radius=0.1, max_nn=30):
    """估计点云法线"""
    # 设置搜索参数（半径和最大邻域点数）
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    pcd.estimate_normals(search_param)
    print("法线已估计完成！")
    return pcd


def surface_reconstruction(pcd, depth=8):
    """泊松表面重建"""
    print("开始泊松表面重建...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    print("重建完成！")
    return mesh


def visualize_and_save_mesh(mesh, output_path="output_mesh.ply"):
    """可视化网格并保存"""
    o3d.visualization.draw_geometries([mesh],
                                      window_name="重建后的网格",
                                      width=800, height=600)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"网格已保存到：{output_path}")


def main():
    input_path = "test.ply"  # 输入点云文件路径
    output_path = "output_mesh.ply"  # 输出网格文件路径

    # 1. 读取点云
    pcd = read_point_cloud(input_path)

    # 2. 估计法线（关键步骤！）
    pcd = estimate_normals(pcd, radius=0.1, max_nn=30)  # 调整radius和max_nn参数

    # 3. 表面重建
    mesh = surface_reconstruction(pcd, depth=8)  # 推荐depth=8~12

    # 4. 可视化并保存网格
    visualize_and_save_mesh(mesh, output_path)


if __name__ == "__main__":
    main()