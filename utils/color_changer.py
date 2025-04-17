import bpy
import os


def import_model(file_path):
    """根据文件扩展名导入模型（支持 .glb 格式）"""
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return None

    # 根据文件扩展名选择导入方式
    if file_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=file_path)
    elif file_path.endswith(".stl"):
        bpy.ops.import_mesh.stl(filepath=file_path)
    elif file_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=file_path)
    elif file_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=file_path)  # GLB/GLTFSupport
    else:
        print(f"错误：不支持的文件格式（{os.path.splitext(file_path)[1]}）！")
        return None

    # 获取导入的最后一个对象
    imported_objects = [obj for obj in bpy.context.selected_objects]
    if not imported_objects:
        print("错误：导入失败，未找到对象！")
        return None
    return imported_objects[-1]


def set_silver_material(target_obj):
    """为对象设置银色材质"""
    material_name = "Silver_Material"
    material = bpy.data.materials.get(material_name)
    if not material:
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True

    # 配置材质节点（Principled BSDF）
    bsdf_node = material.node_tree.nodes.get('Principled BSDF')
    if bsdf_node:
        # 设置银色参数
        bsdf_node.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
        bsdf_node.inputs['Metallic'].default_value = 1.0
        bsdf_node.inputs['Roughness'].default_value = 0.1

    # 应用材质到对象
    if target_obj.data.materials:
        target_obj.data.materials[0] = material
    else:
        target_obj.data.materials.append(material)
    print("成功设置银色材质！")


def main():
    # 直接指定模型路径（无需命令行参数）
    model_path = "data/5.glb"  # 修改此处路径

    # 导入模型
    target_obj = import_model(model_path)
    if not target_obj:
        return

    # 设置材质
    set_silver_material(target_obj)

    print(f"处理完成：{model_path}")


if __name__ == "__main__":
    main()