import os
import random
import string
from PIL import Image


def embed_data(cover_image_path, secret_message, output_image_path):
    """
    使用 4-bit LSB 算法将 secret_message 嵌入 cover_image_path 指定的图像中，
    并将隐写后的图像保存到 output_image_path。
    """
    # 1. 读取图像并转换为 RGB 模式
    cover_img = Image.open(cover_image_path).convert("RGB")
    pixels = cover_img.load()

    # 2. 将秘密信息转换为字节，前4字节存储信息长度（假定信息长度不会超过 2^32）
    secret_bytes = secret_message.encode('utf-8')
    secret_length = len(secret_bytes)
    length_bytes = secret_length.to_bytes(4, byteorder='big')  # 固定 4 字节记录长度
    data_with_length = length_bytes + secret_bytes

    # 将整个数据转换为二进制比特串（每个字节8位）
    bit_string = ''.join([f"{byte:08b}" for byte in data_with_length])
    total_bits = len(bit_string)

    width, height = cover_img.size
    num_channels = 3  # RGB图像
    capacity_bits = width * height * num_channels * 4  # 每个通道嵌入4位数据

    if total_bits > capacity_bits:
        raise ValueError("待嵌入的信息太大，无法放入当前图像中！")

    # 3. 嵌入过程：按像素、按通道替换低4位
    bit_index = 0
    for y in range(height):
        for x in range(width):
            if bit_index >= total_bits:
                break

            r, g, b = pixels[x, y]
            new_channels = []
            for channel_val in (r, g, b):
                if bit_index >= total_bits:
                    new_channels.append(channel_val)
                    continue

                # 取出接下来的4位
                bits_to_write = bit_string[bit_index:bit_index + 4]
                bits_to_write_val = int(bits_to_write, 2)

                # 保留当前通道的高4位，再加上新的低4位
                high_4_bits = channel_val & 0xF0
                new_channel_val = high_4_bits | bits_to_write_val
                new_channels.append(new_channel_val)
                bit_index += 4

            pixels[x, y] = tuple(new_channels)
        if bit_index >= total_bits:
            break

    # 4. 保存隐写后的图像（建议使用 PNG 格式，避免有损压缩导致数据丢失）
    cover_img.save(output_image_path, format="PNG")
    print(f"图像 {os.path.basename(cover_image_path)} 处理完成，输出保存为 {output_image_path}")


def generate_random_secret(capacity_bytes):
    """
    根据图像可嵌入的字节数（capacity_bytes）生成一个随机的秘密文本。
    由于 embed_data 函数中会额外嵌入4字节记录信息长度，
    因此实际可用的秘密信息字节数为 capacity_bytes - 4。
    """
    available_bytes = capacity_bytes - 4
    if available_bytes < 1:
        raise ValueError("图像容量不足，无法嵌入秘密信息")

    # 设定最小秘密长度（例如16字节），如果可用空间不足则使用全部
    min_length = 16 if available_bytes >= 16 else 1
    # 随机生成一个介于 min_length 和 available_bytes 之间的长度
    secret_length = random.randint(min_length, available_bytes)
    # 生成随机字符串（字符由大小写字母和数字构成）
    secret_message = ''.join(random.choices(string.ascii_letters + string.digits, k=secret_length))
    return secret_message


def process_dataset(input_folder, output_folder):
    """
    遍历输入文件夹中的所有图像，使用 4-bit LSB 算法将随机生成的秘密文本嵌入图像中，
    并将结果保存到输出文件夹中。
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
                # 预先打开图像计算其嵌入容量（单位：字节）
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    width, height = img.size
                    capacity_bits = width * height * 3 * 4
                    capacity_bytes = capacity_bits // 8
                    if capacity_bytes <= 4:
                        print(f"图像 {filename} 太小，无法嵌入信息。")
                        continue

                # 为当前图像生成一个随机的秘密文本
                secret_message = generate_random_secret(capacity_bytes)
                print(f"图像 {filename} 的秘密信息长度：{len(secret_message)} 字节")

                # 使用 embed_data 嵌入秘密信息
                embed_data(input_path, secret_message, output_path)
            except Exception as e:
                print(f"处理 {filename} 失败: {e}")


if __name__ == '__main__':
    # 配置输入和输出文件夹路径
    input_folder = "E:\hxr\paper_codefile\datasets\MSCOCO\Hidden_test\\val\\val_class"  # 存放数据集图像的文件夹
    output_folder = "output_stego_images"  # 隐写后的图像将保存到此文件夹

    process_dataset(input_folder, output_folder)
