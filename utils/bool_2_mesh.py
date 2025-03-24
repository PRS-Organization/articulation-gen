import bpy
import bmesh
from mathutils import Vector


def load_model(filepath):
    """加载OBJ模型并返回对象"""
    # 清除现有选择
    bpy.ops.object.select_all(action='DESELECT')

    # 导入模型
    bpy.ops.import_scene.obj(filepath=filepath)

    # 获取导入的首个对象
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.select_set(True)

    # 应用变换（确保坐标正确）
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    return imported_obj


def calculate_volume(obj):
    """计算网格对象的体积"""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    volume = bm.calc_volume()
    bm.free()
    return volume


def perform_boolean_operation(target_obj, other_obj, operation_type):
    """执行布尔运算并返回结果对象"""
    # 复制目标对象以避免修改原始数据
    target_copy = target_obj.copy()
    target_copy.data = target_obj.data.copy()
    target_copy.name = f"{target_obj.name}_{operation_type}"
    bpy.context.collection.objects.link(target_copy)

    # 设置操作参数
    bpy.context.view_layer.objects.active = target_copy
    target_copy.select_set(True)
    other_obj.select_set(True)

    # 添加布尔修改器
    bool_mod = target_copy.modifiers.new("Boolean", 'BOOLEAN')
    bool_mod.operation = operation_type
    bool_mod.object = other_obj

    # 应用修改器
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)

    # 清除选择
    other_obj.select_set(False)

    return target_copy


def save_model(obj, filepath):
    """保存模型为OBJ文件"""
    # 清除选择并选择目标对象
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # 导出为OBJ
    bpy.ops.export_scene.obj(
        filepath=filepath,
        use_selection=True,
        axis_forward='Y',
        axis_up='Z'
    )


def main():
    # 设置文件路径
    cube_path = "cube.obj"  # 需要修改为实际路径
    cube2_path = "cube2.obj"  # 需要修改为实际路径
    output_diff = "cube3.obj"
    output_intersect = "cube_intersect.obj"
    output_union = "cube_union.obj"

    try:
        # 加载模型
        cube = load_model(cube_path)
        cube2 = load_model(cube2_path)

        # 计算原始体积
        vol_cube = calculate_volume(cube)
        vol_cube2 = calculate_volume(cube2)
        print(f"原始模型体积：\n"
              f"- Cube: {vol_cube:.2f} 立方单位\n"
              f"- Cube2: {vol_cube2:.2f} 立方单位")

        # 执行布尔运算
        # 1. 差集（Cube2 - Cube）
        diff_result = perform_boolean_operation(cube2, cube, 'DIFFERENCE')
        vol_diff = calculate_volume(diff_result)
        print(f"差集体积：{vol_diff:.2f} 立方单位")
        save_model(diff_result, output_diff)

        # 2. 交集（Cube & Cube2）
        intersect_result = perform_boolean_operation(cube2, cube, 'INTERSECT')
        vol_intersect = calculate_volume(intersect_result)
        print(f"交集体积：{vol_intersect:.2f} 立方单位")
        save_model(intersect_result, output_intersect)

        # 3. 并集（Cube | Cube2）
        union_result = perform_boolean_operation(cube2, cube, 'UNION')
        vol_union = calculate_volume(union_result)
        print(f"并集体积：{vol_union:.2f} 立方单位")
        save_model(union_result, output_union)

    except Exception as e:
        print(f"错误发生: {str(e)}")
        return

    finally:
        # 清理所有临时对象（可选）
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)


if __name__ == "__main__":
    main()