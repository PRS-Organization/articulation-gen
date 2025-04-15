import bpy
import math
from mathutils import Vector
import os
from mathutils import Euler


class ModelProcessor:
    def __init__(self, config):
        self.config = config
        self.target_obj = None

    def process(self):
        self._clear_scene()
        self._load_model()
        self._normalize_model()
        self._rotate_model()
        self.move_origin_without_moving_object()
        self._export_model()

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

        except Exception as e:
            self._clear_scene()
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _normalize_model(self):
        """标准化模型尺寸"""
        bbox = [self.target_obj.matrix_world @ Vector(corner) for corner in self.target_obj.bound_box]
        bbox_min = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
        bbox_max = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))

        current_size = bbox_max - bbox_min
        max_dim = max(current_size)
        scale_factor = 2.0 / max_dim if max_dim != 0 else 1.0
        self.target_obj.scale *= scale_factor
        bpy.context.view_layer.update()

    def _rotate_model(self):
        """绕Z轴旋转90度"""
        self.target_obj.rotation_euler.z = math.radians(90)
        bpy.context.view_layer.update()

        """输出原点的世界坐标，并将原点沿X轴移动指定距离"""
        obj = self.target_obj

        # 获取原点的世界坐标（即物体的location属性）
        # origin_world = obj.location
        # print(f"原点的世界坐标: {origin_world}")
        # delta_x = - 0.5
        # # 方法1: 直接修改物体的location属性（移动原点）
        # obj.location.x += delta_x  # 沿世界X轴移动0.5单位
        # print(f"新原点的世界坐标: {obj.location}")

    #     # 获取物体的局部Z轴方向向量（世界坐标系）
    #     local_z_axis = self.target_obj.matrix_world.to_3x3() @ Vector((0, 0, 1))
    #     # 计算移动向量
    #     translation = local_z_axis.normalized() * delta
    #     # 应用移动
    #     self.target_obj.location += translation
    #     bpy.context.view_layer.update()

    def move_origin_without_moving_object(self, axis='Z', delta=0.5):
        """
        通用函数：沿指定轴移动原点，物体位置不变

        :param axis: 轴向（'X', 'Y', 'Z'）
        :param delta: 移动量（正数为轴正方向）
        """
        # obj = self.target_obj
        # # bpy.context.view_layer.objects.active = obj
        # bpy.ops.object.mode_set(mode='OBJECT')
        #
        # # 确定移动方向
        # if axis.upper() == 'X':
        #     translation = (-delta, 0, 0)
        #     obj.location.x += delta
        # elif axis.upper() == 'Y':
        #     translation = (0, -delta, 0)
        #     obj.location.y += delta
        # elif axis.upper() == 'Z':
        #     translation = (0, 0, -delta)
        #     obj.location.z += delta
        # else:
        #     raise ValueError("Axis must be 'X', 'Y', or 'Z'")
        #
        # # 反向移动几何数据
        # mat = Matrix.Translation(translation)
        # obj.data.transform(mat)
        # bpy.context.view_layer.update()

        # ------------------------------------------
        # 获取物体的局部Z轴方向向量（世界坐标系）
        # local_z_axis = self.target_obj.matrix_world.to_3x3() @ Vector((0, 0, 1))
        # # 计算移动向量
        # translation = local_z_axis.normalized() * delta
        # # 应用移动
        # self.target_obj.location += translation
        # bpy.context.view_layer.update()

    #     ------------------------------------------
        # 将角度转换为弧度
        angle_radians = math.radians(90)
        print("-------", self.target_obj.rotation_mode)
        # 获取当前旋转模式（欧拉角或四元数）
        if self.target_obj.rotation_mode == 'QUATERNION':
            # 四元数模式：绕世界Z轴旋转
            rot_quat = Euler((0, 0, angle_radians)).to_quaternion()
            self.target_obj.rotation_quaternion = rot_quat
        else:
            # 欧拉角模式：直接设置Z轴角度（默认绕世界轴）
            self.target_obj.rotation_euler.z = angle_radians

        # 更新场景
        bpy.context.view_layer.update()

    def _export_model(self):
        """导出模型（覆盖原文件）"""
        ext = self.config['model_path'].split('.')[-1].lower()
        export_path = self.config['model_path']

        # 确保目录存在
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        # 选择对象
        bpy.ops.object.select_all(action='DESELECT')
        self.target_obj.select_set(True)
        bpy.context.view_layer.objects.active = self.target_obj

        try:
            if ext == 'obj':
                bpy.ops.export_scene.obj(
                    filepath=export_path,
                    use_selection=True,
                    use_materials=False,
                    use_overwrite=True
                )
            elif ext in ('glb', 'gltf'):
                bpy.ops.export_scene.gltf(
                    filepath=export_path,
                    export_format='GLB' if ext == 'glb' else 'GLTF_SEPARATE',
                    use_selection=True,
                    export_draco_mesh_compression_enable=False
                )
            elif ext == 'fbx':
                bpy.ops.export_scene.fbx(
                    filepath=export_path,
                    use_selection=True,
                    bake_space_transform=True
                )
            else:
                raise ValueError(f"不支持的导出格式: {ext}")

            print(f"成功导出模型到: {export_path}")

        except Exception as e:
            raise RuntimeError(f"导出失败: {str(e)}")


if __name__ == "__main__":
    config = {
        'model_path': './micro_new.glb'
    }

    processor = ModelProcessor(config)
    processor.process()