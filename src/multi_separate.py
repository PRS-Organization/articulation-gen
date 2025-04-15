import bpy
import math
from mathutils import Vector

class ModelProcessor:
    def __init__(self, config):
        self.config = config
        self.target_obj = None

    def _clear_scene(self):
        """清空场景"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)

    def _print_object_info(self, obj):
        """打印单个物体的详细信息"""
        # 位置和缩放
        location = obj.location
        scale = obj.scale

        # 处理旋转（兼容四元数和欧拉角）
        if obj.rotation_mode == 'QUATERNION':
            rotation = obj.rotation_quaternion.to_euler()
        else:
            rotation = obj.rotation_euler
        rotation_deg = [math.degrees(rad) for rad in rotation]

        # 原点的世界坐标（即物体的位置）
        origin_world = obj.location

        # 格式化输出
        print(f"名称: {obj.name}")
        print(f"位置 (世界坐标): X={location.x:.3f}, Y={location.y:.3f}, Z={location.z:.3f}")
        print(f"旋转 (度): X={rotation_deg[0]:.1f}°, Y={rotation_deg[1]:.1f}°, Z={rotation_deg[2]:.1f}°")
        print(f"缩放: X={scale.x:.3f}, Y={scale.y:.3f}, Z={scale.z:.3f}")
        print(f"原点世界坐标: {origin_world}")
        print("-" * 40)

    def _print_all_objects_info(self):
        """打印场景中所有有效网格物体的信息"""
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        if not mesh_objects:
            print("场景中没有有效的网格物体")
            return

        print("\n=== 模型包含的对象信息 ===")
        for obj in mesh_objects:
            self._print_object_info(obj)

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

            # 打印导入前的所有对象信息
            self._print_all_objects_info()

            # 合并多个网格物体
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

    def process(self):
        """主处理流程"""
        self._clear_scene()
        self._load_model()
        # 打印最终合并后的对象信息（如果需要）
        self._print_all_objects_info()


if __name__ == "__main__":
    config = {
        'model_path': './label_5_model.glb'
    }

    processor = ModelProcessor(config)
    processor.process()