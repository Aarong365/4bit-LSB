import os
from PIL import Image


def recover_image(stego_image_path, output_recovered_image_path):
    """
    恢复隐写图像为近似原图。
    由于隐写过程覆盖了每个像素通道的低 4 位信息，
    这里将低 4 位设为 0，获得一幅近似原图的图像。
    """
    # 打开隐写图像并转换为 RGB 模式
    img = Image.open(stego_image_path).convert("RGB")
    pixels = img.load()
    width, height = img.size

    # 遍历每个像素，将低 4 位设为 0
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            r_recovered = r & 0xF0  # 保留高 4 位，低 4 位置 0
            g_recovered = g & 0xF0
            b_recovered = b & 0xF0
            pixels[x, y] = (r_recovered, g_recovered, b_recovered)

    # 保存恢复后的图像（建议使用 PNG 格式以避免压缩损失）
    img.save(output_recovered_image_path, format="PNG")
    print(f"恢复图像已保存为 {output_recovered_image_path}")


def process_recovery(input_folder, output_folder):
    """
    遍历输入文件夹中的所有隐写图像，
    对每张图像恢复为近似原图，并将恢复后的图像保存到输出文件夹中。
    """
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 支持的图像文件扩展名
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                recover_image(input_path, output_path)
            except Exception as e:
                print(f"处理 {filename} 失败: {e}")


if __name__ == '__main__':
    # 输入文件夹：之前隐写后输出的文件夹
    input_folder = "E:\hxr\make_datasets\datasets\lsb\output_images"  # 请修改为你实际的隐写图像文件夹路径
    # 输出文件夹：恢复图像将保存在该文件夹下
    output_folder = "recovered_images"

    process_recovery(input_folder, output_folder)
