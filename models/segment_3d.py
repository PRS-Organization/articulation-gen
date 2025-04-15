import os
import bpy
import json
import numpy as np
from pathlib import Path
import bmesh
# import open3d as o3d


class ModelSegmenter:
    def __init__(self, model_path, labels_data_path, output_dir="render_output"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.labels_data, self.vertex_counts = self.load_data(labels_data_path)
        self.vertex_labels = self.labels_data
        self.obj = None
        self.original_vertices = None  # 用于存储原始顶点坐标
        self.base_list = list()

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, labels_data_path="./"):
        print("Loading labels data...", labels_data_path)
        vertices_data = np.load(labels_data_path, allow_pickle=True)
        vertices_dict = vertices_data["vertices_label"].item()
        # # 关键修复：将所有键转换为字符串类型
        #
        # for key, value_list in vertices_dict.items():
        #     print(key)
        # json_dict = {
        #     str(key): [int(x) for x in value_list]  # 直接转换键和列表元素
        #     for key, value_list in vertices_dict.items() }
        # print(type(vertices_dict))
        # # 新增：保存为JSON [[1]]
        # json_save_path = Path(labels_data_path).parent / "vertices_label.json"
        # with open(json_save_path, 'w') as f:
        #     json.dump(json_dict, f, indent=4)
        # print(f"顶点标签数据已保存为: {json_save_path}")

        # 打印字典信息
        print(f"字典类型: {type(vertices_dict)}")
        counts = list()
        for key, value in vertices_dict.items():
            print(f"键类型: {type(key)}, 键: {key}, 值类型: {type(value)}, 值长度: {len(value)}, 值元素: {type(value[0])}")
            counts += value
        return vertices_dict, counts

    def process(self):
        """主处理流程"""
        self.import_model()
        self.prepare_vertex_data()
        self.export_all_labels()

    def import_model(self):
        """导入并处理模型"""
        self._clear_scene()

        ext = self.model_path.suffix[1:].lower()
        try:
            import_func = {
                'obj': bpy.ops.import_scene.obj,
                'glb': bpy.ops.import_scene.gltf,
                'gltf': bpy.ops.import_scene.gltf,
                'fbx': bpy.ops.import_scene.fbx
            }[ext]
            import_func(filepath=str(self.model_path))
        except KeyError:
            raise ValueError(f"Unsupported format: {ext}")
        except Exception as e:
            self._clear_scene()
            raise RuntimeError(f"模型加载失败: {str(e)}")

        # 获取有效网格对象
        self.obj = self.get_valid_mesh_object()
        self.original_vertices = self.obj.data.vertices  # 保存原始顶点数据
        vertex_indices = [v.index for v in self.original_vertices]
        result_set = set(vertex_indices).difference(set(self.vertex_counts))
        self.base_list = list(result_set)
        print(len(self.base_list), len(self.original_vertices))
        self.base_list = [idx for idx in self.base_list if idx < len(self.original_vertices)]
        print(len(self.base_list))

    def get_valid_mesh_object(self):
        """获取有效的网格对象"""
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                return obj
            elif obj.type == 'EMPTY' and obj.children:
                for child in obj.children:
                    if child.type == 'MESH':
                        return child
        raise RuntimeError("GLB文件中未找到有效的MESH对象")

    def prepare_vertex_data(self):
        """验证顶点索引有效性"""
        max_vertex_idx = len(self.obj.data.vertices) - 1
        for label, indices in self.labels_data.items():
            invalid_indices = [i for i in indices if i < 0 or i > max_vertex_idx]
            if invalid_indices:
                raise ValueError(f"标签 {label} 包含无效顶点索引：{invalid_indices}")

    def export_all_labels(self):
        """导出所有标签的子模型"""
        co_dict = dict()
        for label in self.labels_data.keys():
            points_co = self.export_submodel(label)
            co_dict[str(label)] = points_co
        # base_points = [list(self.original_vertices[idx].co) for idx in self.base_list]
        bm = bmesh.new()
        bm.from_mesh(self.obj.data)
        bm.verts.ensure_lookup_table()

        points = []
        base_points = []
        for idx in self.base_list:
            try:
                co = list(bm.verts[idx].co)
                base_points.append(co)
            except Exception as E: continue
        co_dict[str(-1)] = base_points

        print(type(self.base_list), self.base_list[0])
        save_path = self.output_dir / f"label_vertices_coordinates.json"
        with open(save_path, 'w') as f:
            json.dump(co_dict, f, indent=4)

    # def export_submodel(self, label):
    #     """导出指定标签的子模型"""
    #     # 直接获取顶点索引列表
    #     selected = self.labels_data.get(label, [])
    #
    #     if not selected:
    #         print(f"未找到标签 {label} 的顶点")
    #         return
    #
    #     # 保存顶点坐标到JSON（可选）
    #     vertex_coords = {str(idx): self.original_vertices[idx].co.to_tuple() for idx in selected}
    #     save_path = self.output_dir / f"label_{label}_vertices.json"
    #     with open(save_path, 'w') as f:
    #         json.dump(vertex_coords, f, indent=4)
    #
    #     # 选择顶点
    #     bpy.ops.object.mode_set(mode='OBJECT')
    #     bpy.ops.object.select_all(action='DESELECT')
    #     self.obj.select_set(True)
    #     bpy.context.view_layer.objects.active = self.obj
    #
    #     bpy.ops.object.mode_set(mode='EDIT')
    #     bpy.ops.mesh.select_all(action='DESELECT')
    #     bpy.ops.object.mode_set(mode='OBJECT')
    #
    #     for v_idx in selected:
    #         if v_idx < len(self.obj.data.vertices):
    #             self.obj.data.vertices[v_idx].select = True
    #         else:
    #             print(f"警告：顶点索引 {v_idx} 超出范围")
    #
    #     # 分离并导出
    #     bpy.ops.object.mode_set(mode='EDIT')
    #     bpy.ops.mesh.separate(type='SELECTED')
    #     bpy.ops.object.mode_set(mode='OBJECT')
    #
    #     # 获取分离出的新对象
    #     new_objs = [obj for obj in bpy.context.selected_objects if obj != self.obj]
    #     if not new_objs:
    #         print(f"标签 {label} 分离模型失败")
    #         return
    #
    #     new_obj = new_objs[0]
    #
    #     # 保存新对象
    #     output_path = self.output_dir / f"label_{label}_model.glb"
    #
    #     # 处理材质和贴图
    #     for mat in new_obj.data.materials:
    #         if mat.use_nodes:
    #             for node in mat.node_tree.nodes:
    #                 if node.type == 'TEX_IMAGE' and node.image:
    #                     node.image.pack()
    #
    #     # 导出GLB
    #     bpy.ops.export_scene.gltf(
    #         filepath=str(output_path),
    #         export_format='GLB',
    #         use_selection=True,
    #         export_materials='EXPORT',
    #         export_draco_mesh_compression_enable=True,
    #         export_texture_dir=str(self.output_dir)
    #     )
    #
    #     # 清理临时对象
    #     bpy.data.objects.remove(new_obj, do_unlink=True)
    #     print(f"已导出标签 {label} 的模型到：{output_path}")

    def export_submodel(self, label):

        # 选择顶点
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        self.obj.select_set(True)
        bpy.context.view_layer.objects.active = self.obj

        # 使用Bmesh确保精确选择 [[2]]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')  # 切换模式以更新选择状态
        selected = self.labels_data.get(label, [])
        # 严格筛选有效顶点
        valid_indices = [v_idx for v_idx in selected if v_idx < len(self.obj.data.vertices)]

        if not valid_indices:
            print(f"标签 {label} 无有效顶点")
            return

        # 使用Bmesh进行精确选择 [[2]]
        bm = bmesh.new()
        bm.from_mesh(self.obj.data)
        bm.verts.ensure_lookup_table()

        points = []
        for v_idx in valid_indices:
            if v_idx < len(bm.verts):  # 检查BMesh的顶点数量
                try:
                    points.append(list(bm.verts[v_idx].co))
                except ValueError: pass

        # 清除所有选择并精确选择目标顶点
        for vert in bm.verts:
            vert.select = False
        for v_idx in valid_indices:
            bm.verts[v_idx].select = True

        bm.to_mesh(self.obj.data)
        bm.free()

        # 分离并导出 [[3]][[6]]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='SELECTED')
        bpy.ops.object.mode_set(mode='OBJECT')

        # 获取并处理新对象
        new_objs = [obj for obj in bpy.context.selected_objects if obj != self.obj]
        if not new_objs:
            print(f"标签 {label} 分离模型失败")
            return

        new_obj = new_objs[0]

        # 导出前验证顶点数量 [[4]]
        if len(new_obj.data.vertices) != len(valid_indices):
            print(f"警告：导出顶点数不匹配（预期{len(valid_indices)}，实际{len(new_obj.data.vertices)}）")

        # 导出GLB [[6]]
        output_path = self.output_dir / f"label_{label}_model.glb"
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format='GLB',
            use_selection=True,
            export_materials='EXPORT',
            export_draco_mesh_compression_enable=True,
            export_texture_dir=str(self.output_dir)
        )

        # ===== 新增：使用Open3D导出点云PLY文件 =====
        ply_output_path = self.output_dir / f"label_{label}_points.ply"

        # 提取顶点坐标（直接使用原始顶点数据）

        # points = [list(self.original_vertices[idx].co) for idx in valid_indices]
        # points = np.array(
        #     [self.original_vertices[idx].co for idx in valid_indices],
        #     dtype=np.float32
        # )

        # # 创建Open3D点云对象
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        #
        # # 保存PLY文件（二进制格式）
        # o3d.io.write_point_cloud(str(ply_output_path), pcd, write_ascii=False)
        # print(f"已导出标签 {label} 的点云到：{ply_output_path}")
        # ==========================================

        # 清理
        bpy.data.objects.remove(new_obj, do_unlink=True)
        print(f"已导出标签 {label} 的模型到：{output_path}")
        return points

    def _clear_scene(self):
        """清空当前场景"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)


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
    vertices_file = os.path.join(video_dir, "vertices_segmentation.npz")
    # loaded = np.load(frames_file, allow_pickle=True)

    models_dir = os.path.join(video_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    segmenter = ModelSegmenter(
        model_path=asset_3d,
        labels_data_path=vertices_file,
        output_dir=models_dir
    )
    segmenter.process()