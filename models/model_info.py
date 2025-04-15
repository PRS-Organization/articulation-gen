import bpy
import math

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

    def _print_transform_info(self):
        """打印物体的Transform信息（位置/旋转/缩放）"""
        if not self.target_obj or self.target_obj.type != 'MESH':
            print("没有有效的网格物体")
            return

        # 获取位置和缩放
        location = self.target_obj.location
        scale = self.target_obj.scale

        # 处理旋转（兼容四元数和欧拉角）
        if self.target_obj.rotation_mode == 'QUATERNION':
            # 转换四元数为欧拉角（以度为单位）
            rotation = self.target_obj.rotation_quaternion.to_euler()
            rotation_deg = [math.degrees(rad) for rad in rotation]
        else:
            # 直接获取欧拉角并转换为度
            rotation_deg = [math.degrees(rad) for rad in self.target_obj.rotation_euler]

        # 格式化输出
        print("== Transform 信息 ==")
        print(f"位置 (Location): "
              f"X={location.x:.3f}, Y={location.y:.3f}, Z={location.z:.3f}")
        print(f"旋转 (Rotation): "
              f"X={rotation_deg[0]:.1f}°, Y={rotation_deg[1]:.1f}°, Z={rotation_deg[2]:.1f}°")
        print(f"缩放 (Scale): "
              f"X={scale.x:.3f}, Y={scale.y:.3f}, Z={scale.z:.3f}")

    def process(self):
        """主处理流程"""
        self._clear_scene()
        self._load_model()
        self._print_transform_info()  # 打印Transform信息


if __name__ == "__main__":
    config = {
        'model_path': './micro_new.glb'
    }

    processor = ModelProcessor(config)
    processor.process()