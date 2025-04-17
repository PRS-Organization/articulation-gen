import bpy
import json
import bmesh
from math import radians
from mathutils import Euler, Vector
from mathutils.geometry import barycentric_transform


class ModelProcessor:
    def __init__(self, config):
        self.config = config
        self.target_obj = None
        self.rotation_order = [
            (0, 0, 0),  # 原角度
            (0, radians(90), 0),  # Y轴+90
            (0, radians(-90), 0),  # Y轴-90
            (radians(90), 0, 0),  # X轴+90
            (radians(-90), 0, 0),  # X轴-90
            (0, 0, radians(90)),  # Z轴+90
            (0, 0, radians(-90))  # Z轴-90
        ]

    def process(self):
        self._clear_scene()
        self._load_model()
        json_data = self._load_json()
        model_id = self.config['model_path'].split('/')[-1].split('.')[0]
        target_config = json_data.get(model_id, None)
        if not target_config:
            raise KeyError(f"未找到模型ID {model_id} 的配置信息")

        self._process_rotation(target_config['scale'])
        self._scale_to_match(target_config['scale'])
        self._position_to_center(target_config['center'])
        self._apply_transformations()
        self._save_model_to_disk()

    def _load_json(self):
        with open(self.config['json_path'], 'r') as f:
            return json.load(f)

    def _process_rotation(self, target_scale):
        best_error = float('inf')
        best_rotation = self.rotation_order[0]

        for angles in self.rotation_order:
            # 临时应用旋转
            # print(angles, Euler(angles))
            self.target_obj.rotation_quaternion = Euler(angles).to_quaternion()
            bpy.context.view_layer.update()
            print("rot_quat", self.target_obj.rotation_quaternion)
            current_size = self._get_bounding_box_size()

            # 计算比例误差
            error = self._calculate_proportion_error(current_size, target_scale)
            print(error)
            if error < best_error:
                best_error = error
                best_rotation = angles
        print('Best rotation: ', best_rotation)
        # 应用最佳旋转
        bpy.context.view_layer.update()
        self.target_obj.rotation_quaternion = Euler(best_rotation).to_quaternion()

    def _calculate_proportion_error(self, current_size, target_size):
        current_ratio = [a / b if b != 0 else 0 for a, b in zip(current_size, target_size)]
        target_ratio = [1.0, 1.0, 1.0]
        return sum([(a - b) ** 2 for a, b in zip(current_ratio, target_ratio)])

    def _get_bounding_box_size(self):
        # 计算世界坐标系下的包围盒尺寸
        bpy.context.view_layer.update()
        obj = self.target_obj
        obj_matrix = obj.matrix_world
        bbox = [obj_matrix @ Vector(v) for v in obj.bound_box]
        min_coord = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
        max_coord = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
        return (max_coord - min_coord).to_tuple()

    def _scale_to_match(self, target_scale):
        # for i in range(3):
        current_size = self._get_bounding_box_size()
        # 计算每个轴的缩放比例（目标尺寸 / 当前尺寸）
        scale_factors = [
            t / s if s != 0 else 1.0  # 避免除以零
            for s, t in zip(current_size, target_scale)
        ]
        print(scale_factors, self.target_obj.scale, '========')
        # 应用独立缩放
        self.target_obj.scale = tuple(scale_factors)
        # self.target_obj.scale = tuple([0.3, 0.3, 0.3])
        # 直接修改顶点坐标

        # 烘焙变换并更新场景
        # 强制更新场景确保变换生效
        self._apply_transformations()
        update_size = self._get_bounding_box_size()
        print(current_size, update_size, '---------')
        print(target_scale)
        # scale_factors = [t / s if s != 0 else 1 for s, t in zip(current_size, target_scale)]
        # average_scale = sum(scale_factors) / 3

    def _position_to_center(self, target_center):
        current_center = self._get_bounding_box_center()
        self.target_obj.location = Vector(target_center) - (current_center - self.target_obj.location)

    def _get_bounding_box_center(self):
        obj = self.target_obj
        obj_matrix = obj.matrix_world
        bbox = [obj_matrix @ Vector(v) for v in obj.bound_box]
        return Vector((
            (min(v.x for v in bbox) + max(v.x for v in bbox)) / 2,
            (min(v.y for v in bbox) + max(v.y for v in bbox)) / 2,
            (min(v.z for v in bbox) + max(v.z for v in bbox)) / 2
        ))

    def _apply_transformations(self):
        # 应用变换并更新数据
        bpy.context.view_layer.update()
        self.target_obj.select_set(True)
        bpy.ops.object.transform_apply(  # 应用所有变换到物体
            location=True,
            rotation=True,
            scale=True
        )
        # 再次更新场景
        bpy.context.view_layer.update()

    def _clear_scene(self):
        """清空场景"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)

    def _load_model(self):
        """加载并标准化模型"""
        ext = self.config['model_path'].split('.')[-1].lower()
        import_func = {
            'obj': bpy.ops.import_scene.obj,
            'glb': bpy.ops.import_scene.gltf,
            'gltf': bpy.ops.import_scene.gltf,
            'fbx': bpy.ops.import_scene.fbx
        }.get(ext, None)

        if not import_func:
            raise ValueError(f"不支持的格式: {ext}")

        try:
            bpy.ops.object.select_all(action='DESELECT')
            import_func(filepath=self.config['model_path'])
            imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

            if not imported_objects:
                raise RuntimeError("没有找到有效的网格物体")

            bpy.context.view_layer.objects.active = imported_objects[0]

            if len(imported_objects) > 1:
                bpy.ops.object.mode_set(mode='OBJECT')
                for obj in imported_objects:
                    obj.select_set(True)
                bpy.ops.object.join()
                self.target_obj = bpy.context.active_object
            else:
                self.target_obj = imported_objects[0]

            # 进入对象模式
            bpy.ops.object.mode_set(mode='OBJECT')

        except Exception as e:
            self._clear_scene()
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _save_model_to_disk(self):
        """将处理后的模型保存到本地，默认文件名为 {label}.gltf"""
        # 获取模型ID（从JSON键或文件名）
        model_id = self.config['model_path'].split('/')[-1].split('.')[0]
        output_path = f"{model_id}.glb"  # 默认保存为glb格式

        # 确保导出路径正确
        export_func = {
            'glb': bpy.ops.export_scene.gltf,
            'gltf': bpy.ops.export_scene.gltf,
            'obj': bpy.ops.export_scene.obj,
            'fbx': bpy.ops.export_scene.fbx
        }.get('glb', bpy.ops.export_scene.gltf)  # 默认使用glb格式

        # 导出设置（以glb为例）
        export_settings = {
            'export_format': 'GLB',
            'export_yup': True,  # 根据需求调整坐标系
            'export_apply': True  # 应用所有变换
        }

        try:
            # 选择目标对象
            bpy.ops.object.select_all(action='DESELECT')
            self.target_obj.select_set(True)
            bpy.context.view_layer.objects.active = self.target_obj

            # 执行导出
            export_func(
                filepath=output_path,
                **export_settings
            )
            print(f"模型已保存至: {output_path}")
        except Exception as e:
            raise RuntimeError(f"导出失败: {str(e)}")

if __name__ == "__main__":
    config = {
        'model_path': './data/5.glb',
        'json_path': './3d_part_information.json'
    }

    processor = ModelProcessor(config)
    processor.process()