import rembg
from PIL import Image
import os


def remove_background(input_path):
    # 生成输出文件名（自动添加_1后缀）
    base_name, ext = os.path.splitext(input_path)
    output_path = f"{base_name}_1{ext}"

    # 打开原始图片
    with open(input_path, "rb") as f:
        img = Image.open(f).convert("RGBA")

    # 去除背景
    output_img = rembg.remove(img)

    # 保存结果
    output_img.save(output_path)
    print(f"背景已去除，结果保存为: {output_path}")


# 使用示例（只需提供输入文件名）
remove_background("button1.png")