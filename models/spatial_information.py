import os
import json
import numpy as np
import open3d as o3d


class PointProcessor:
    def __init__(self, points_dict, file_dir):
        """
        初始化处理器
        :param points_dict: 标签到三维点列表的字典，格式如 {"标签": [[x1,y1,z1], [x2,y2,z2], ...]}
        """
        self.points_dict = points_dict
        self.info_dict = dict()
        self.base_dir = file_dir

    def get_labels(self):
        """获取所有标签列表"""
        return list(self.points_dict.keys())

    def process_label(self, label):
        """
        处理指定标签的点云数据
        :param label: 要处理的标签
        :return: 包含包围盒、中心点和尺寸的字典
        """
        if label not in self.points_dict:
            raise ValueError(f"标签 {label} 不存在")

        points = self.points_dict[label]
        if not points:
            raise ValueError(f"标签 {label} 没有点数据")

        # 转换为numpy数组方便计算
        import numpy as np
        points_array = np.array(points)

        # 计算包围盒
        min_coords = points_array.min(axis=0)
        max_coords = points_array.max(axis=0)
        center = (min_coords + max_coords) / 2

        # 计算尺寸
        dimensions = max_coords - min_coords

        return {
            "bounding_box": {
                "min": min_coords.tolist(),
                "max": max_coords.tolist()
            },
            "center": center.tolist(),
            "dimensions": dimensions.tolist()
        }

    def print_dimension_info(self, label):
        """打印指定标签的尺寸信息"""
        result = self.process_label(label)
        # 存储结果到info_dict
        self.info_dict[label] = {
            "bounding_box": {
                "min": list(result["bounding_box"]["min"]),
                "max": list(result["bounding_box"]["max"])
            },
            "center": list(result["center"]),
            "scale": list(result["dimensions"])
        }

        print(f"标签 {label} 的尺寸信息：")
        print(f"包围盒最小坐标：{result['bounding_box']['min']}")
        print(f"包围盒最大坐标：{result['bounding_box']['max']}")
        print(f"中心点坐标：{result['center']}")
        print(f"尺寸（长×宽×高）：{result['dimensions']}")

        points = self.points_dict[label]
        self.vis_point_cloud(points)

    def vis_point_cloud(self, points):
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)
        assert points.shape[1] == 3, "输入必须为N×3的点云数据"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd = self.filter_point_cloud(
            pcd,
            nb_neighbors=50,
            std_ratio=1.0
        )
        print(f"原始点云中点的数量: {len(pcd.points)}")
        # 可选：添加坐标轴并保存图片
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        o3d.visualization.draw_geometries([pcd, coordinate_frame],
                                          window_name="Point Cloud Viewer",
                                          zoom=0.8,
                                          front=[0.1, 0.2, 0.3],  # 自定义视角
                                          lookat=[0, 0, 0],
                                          up=[0, 1, 0])

    def filter_point_cloud(
            self,
            points: o3d.geometry.PointCloud,
            nb_neighbors: int = 50,
            std_ratio: float = 1.0
    ) -> o3d.geometry.PointCloud:
        """
        对Open3D点云进行统计滤波（Statistical Outlier Removal），移除离群点。

        Args:
            points (o3d.geometry.PointCloud): 输入的点云对象。
            nb_neighbors (int): 统计邻域点数（默认50）。
                - 每个点计算到邻域内 `nb_neighbors` 个点的平均距离。
            std_ratio (float): 标准差倍数阈值（默认1.0）。
                - 超过全局均值 + std_ratio × 标准差的点会被移除。（值越小，滤波越严格）

        Returns:
            o3d.geometry.PointCloud: 过滤后的点云对象。
        """
        # 直接执行统计滤波
        filtered_cloud, _ = points.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return filtered_cloud

    def save_information(self, file_path="3d_part_information.json") -> None:
        path = os.path.join(self.base_dir, file_path)
        with open(path, "w") as f:
            json.dump(self.info_dict, f, indent=4)
# =====================================


if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",  # Parent directory
        "..",  # Grandparent directory (two levels up)
        "config",  # Enter the 'config' folder
        "3d_render_video.json"  # Target file
    )
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    asset_3d = config["model_path"]
    video_dir = config["output_dir"]
    models_dir = os.path.join(video_dir, "models")
    # loaded = np.load(frames_file, allow_pickle=True)

    points_json = os.path.join(models_dir, "label_vertices_coordinates.json")
    with open(points_json, "r", encoding="utf-8") as file:
        points_dict = json.load(file)

    processor = PointProcessor(points_dict, video_dir)
    processor.print_dimension_info("5")
    processor.save_information()
    # segmenter = ModelSegmenter(
    #     model_path=asset_3d,
    #     labels_data_path=vertices_file,
    #     output_dir=models_dir
    # )
