import bpy
import bmesh
from collections import deque


def load_model(filepath):
    """加载OBJ模型并返回对象"""
    # 清除现有对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 导入模型
    bpy.ops.import_scene.obj(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    print(f"模型已加载：{obj.name}")
    return obj


def find_connected_components(bm):
    """检测连通组件，返回每个组件的顶点集合"""
    visited = set()
    components = []

    for vert in bm.verts:
        if vert not in visited:
            component = set()
            queue = deque([vert])
            visited.add(vert)
            component.add(vert)

            while queue:
                current_vert = queue.popleft()
                for edge in current_vert.link_edges:
                    neighbor = edge.other_vert(current_vert)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

    return components


def check_connectivity(bm):
    """检查模型是否连通"""
    components = find_connected_components(bm)
    if len(components) > 1:
        print(f"警告：模型存在 {len(components)} 个连通组件，不连通！")
        return False
    else:
        print("模型是连通的！")
        return True


def is_closed(bm):
    """检测模型是否封闭（无孔洞）"""
    for edge in bm.edges:
        if len(edge.link_faces) < 2:  # 存在未闭合的边（属于单个面）
            return False
    print("模型是封闭的！")
    return True


def repair_holes(bm, obj):
    """修复孔洞"""
    bmesh.ops.holes_fill(bm, edges=bm.edges)
    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bmesh.update_edit_mesh(obj.data)  # 传入原始 Mesh 对象 obj.data
    print("孔洞已修复！")


def repair_connectivity(bm, obj):
    """修复连通性（连接分离组件）"""
    # 获取所有顶点坐标
    verts = list(bm.verts)
    min_dist = float('inf')
    pair = (None, None)

    # 寻找最近的顶点对（跨组件）
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            dist = (verts[i].co - verts[j].co).length
            if dist < min_dist:
                min_dist = dist
                pair = (verts[i], verts[j])

    if pair[0] and pair[1]:
        bmesh.ops.connect_vert_pair(bm, verts=[pair[0], pair[1]])
        print(f"已连接最近顶点：{pair[0].index} <-> {pair[1].index}")

    bm.normal_update()
    bm.verts.ensure_lookup_table()
    bmesh.update_edit_mesh(obj.data)  # 传入原始 Mesh 对象 obj.data


def repair_topology(bm, obj):
    """综合修复拓扑问题（孔洞+连通性）"""
    # 1. 修复孔洞
    if not is_closed(bm):
        repair_holes(bm, obj)

    # 2. 修复连通性
    if not check_connectivity(bm):
        repair_connectivity(bm, obj)

    # 最终更新
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data)  # 传入原始 Mesh 对象 obj.data


def save_model(obj, filepath):
    """保存模型覆盖原文件"""
    # 切换回对象模式
    bpy.ops.object.mode_set(mode='OBJECT')

    # 导出为OBJ并覆盖原文件
    bpy.ops.export_scene.obj(
        filepath=filepath,
        use_selection=True,
        path_mode='RELATIVE',
        axis_forward='Y',
        axis_up='Z'
    )
    print(f"文件已保存：{filepath}")


def main():
    # 模型路径
    filepath = "cube1.obj"

    # 加载模型
    obj = load_model(filepath)

    # 进入编辑模式
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)  # 获取 BMesh 对象

    # 检查封闭性
    is_closed_flag = is_closed(bm)

    # 检查连通性
    is_connected_flag = check_connectivity(bm)

    # 如果存在孔洞或不连通，进行修复
    if not is_closed_flag or not is_connected_flag:
        repair_topology(bm, obj)  # 传递 BMesh 和原始对象
        # 修复后重新检查
        if is_closed(bm) and check_connectivity(bm):
            print("拓扑修复成功！")
        else:
            print("警告：修复后仍存在问题！")

    # 退出编辑模式
    bpy.ops.object.mode_set(mode='OBJECT')

    # 保存模型
    save_model(obj, filepath)


if __name__ == "__main__":
    main()