import bpy
import os
from pathlib import Path
from PIL import Image, ImageDraw


def export_textures_and_uv(model_path, output_dir="exports"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清除当前场景
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 导入模型
    try:
        ext = model_path.suffix.lower()
        if ext == ".obj":
            bpy.ops.import_scene.obj(filepath=str(model_path))
        elif ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=str(model_path))
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(model_path))
        else:
            raise ValueError(f"不支持的格式: {ext}")
    except Exception as e:
        raise RuntimeError(f"模型导入失败: {str(e)}")

    # 获取网格对象
    mesh_obj = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break
    if not mesh_obj:
        raise RuntimeError("未找到网格对象")

    # 选中对象
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)

    # 导出贴图
    textures_dir = output_dir / "textures"
    textures_dir.mkdir(exist_ok=True)

    for material in mesh_obj.data.materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                image = node.image
                image_path = textures_dir / (image.name + ".png")
                image.save_render(filepath=str(image_path))
                print(f"已保存贴图: {image_path}")

    # 导出UV映射（使用PIL手动绘制）
    uv_dir = output_dir / "uv_maps"
    uv_dir.mkdir(exist_ok=True)
    uv_filepath = uv_dir / "uv_layout.png"

    # 获取UV数据
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = mesh_obj.data
    uv_layer = mesh.uv_layers.active.data if mesh.uv_layers else None

    if not uv_layer:
        print("警告：未找到UV数据")
        return

    # 创建空白图像
    image_size = (2048, 2048)
    image = Image.new("RGB", image_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 绘制UV线框
    for poly in mesh.polygons:
        uv_coords = [uv_layer[loop_idx].uv for loop_idx in poly.loop_indices]
        points = [
            (uv.x * image_size[0], (1 - uv.y) * image_size[1])
            for uv in uv_coords
        ]
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            draw.line([start, end], fill=(0, 0, 0), width=2)

    # 保存图像
    image.save(uv_filepath)
    print(f"UV映射已保存到: {uv_filepath}")


if __name__ == "__main__":
    model_path = Path("micro1.glb")  # 替换为你的模型路径
    export_textures_and_uv(model_path, output_dir="output")