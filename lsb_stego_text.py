
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本代码实现基于4bit-LSB方法的文本隐写：
1. 将固定的秘密文本嵌入到每个封面图像中生成隐写图像
2. 从隐写图像中提取出文本，并与原始文本比对（计算比特错误率）
3. 分别评估封面图与隐写图的图像质量（PSNR、SSIM、MAE、RMSE）
4. 计算比特每像素（bpp）：每个封面图像实际嵌入的比特数除以图像像素数

输出结构（均在 output_root 下生成）：
  - cover      : 原始封面图（复制）
  - secret     : 保存秘密文本（文本文件）
  - stego      : 隐写图像（将文本嵌入封面图）
  - secret_rev : 提取恢复的秘密文本（文本文件）
"""

import os, math, glob
import numpy as np
import cv2
from PIL import Image


#############################################
# 4bit-LSB 隐写文本实现部分
#############################################

def embed_text_in_image(cover_image_path, secret_text, output_image_path):
    """
    将 secret_text 文本嵌入到封面图像中，采用4bit-LSB隐写方法。
    嵌入流程：
      1. 将 secret_text 按 UTF-8 编码为字节流，并在前面添加4字节头（表示字节长度）
      2. 将整个数据转换为二进制字符串，每个字节8位
      3. 检查封面图像的嵌入容量（cover_width * cover_height * 3 * 4 bits）
      4. 逐像素将每个像素3通道的低4位替换为数据4位
    """
    # 读取封面图像
    cover_img = Image.open(cover_image_path).convert("RGB")
    cover_width, cover_height = cover_img.size
    cover_pixels = cover_img.load()

    # 将秘密文本编码为字节流
    secret_bytes = secret_text.encode('utf-8')
    secret_length = len(secret_bytes)
    header = secret_length.to_bytes(4, byteorder='big')  # 4字节头
    data = header + secret_bytes

    # 转换为二进制字符串
    bit_string = ''.join(format(byte, '08b') for byte in data)
    total_bits = len(bit_string)

    # 计算封面图嵌入容量（单位：bit）
    capacity_bits = cover_width * cover_height * 3 * 4
    if total_bits > capacity_bits:
        raise ValueError("秘密文本数据太大，无法嵌入当前封面图像中！")

    bit_index = 0
    for y in range(cover_height):
        for x in range(cover_width):
            if bit_index >= total_bits:
                break
            r, g, b = cover_pixels[x, y]
            new_r, new_g, new_b = r & 0xF0, g & 0xF0, b & 0xF0
            # 红色通道
            if bit_index < total_bits:
                bits = bit_string[bit_index:bit_index + 4]
                if len(bits) < 4: bits = bits.ljust(4, '0')
                new_r |= int(bits, 2)
                bit_index += 4
            # 绿色通道
            if bit_index < total_bits:
                bits = bit_string[bit_index:bit_index + 4]
                if len(bits) < 4: bits = bits.ljust(4, '0')
                new_g |= int(bits, 2)
                bit_index += 4
            # 蓝色通道
            if bit_index < total_bits:
                bits = bit_string[bit_index:bit_index + 4]
                if len(bits) < 4: bits = bits.ljust(4, '0')
                new_b |= int(bits, 2)
                bit_index += 4
            cover_pixels[x, y] = (new_r, new_g, new_b)
        if bit_index >= total_bits:
            break

    cover_img.save(output_image_path, format="PNG")
    print(f"嵌入成功：{cover_image_path} -> {output_image_path}")
    return total_bits  # 返回总嵌入比特数


def extract_text_from_image(stego_image_path):
    """
    从隐写图像中提取出隐藏的文本信息。
    提取流程：
      1. 读取隐写图像，依次提取每个像素通道低4位组成二进制字符串
      2. 前32位为头信息，表示秘密文本字节长度
      3. 根据长度提取后续比特串，并转换为字节，再用 utf-8 解码为字符串
    """
    stego_img = Image.open(stego_image_path).convert("RGB")
    stego_pixels = stego_img.load()
    width, height = stego_img.size

    bits = []
    for y in range(height):
        for x in range(width):
            r, g, b = stego_pixels[x, y]
            bits.append(format(r & 0x0F, '04b'))
            bits.append(format(g & 0x0F, '04b'))
            bits.append(format(b & 0x0F, '04b'))
    bit_string = ''.join(bits)

    header_bits = bit_string[:32]
    text_length = int(header_bits, 2)
    total_text_bits = text_length * 8
    text_bits = bit_string[32:32 + total_text_bits]

    text_bytes = bytearray()
    for i in range(0, len(text_bits), 8):
        byte = int(text_bits[i:i + 8].ljust(8, '0'), 2)
        text_bytes.append(byte)
    try:
        recovered_text = text_bytes.decode('utf-8')
    except Exception as e:
        recovered_text = ""
        print("解码错误：", e)
    return recovered_text

# 评估指标测试部分（图像指标）
def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    def ssim_channel(a, b):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu_a = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
        mu_b = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
        sigma_a = cv2.filter2D(a ** 2, -1, window)[5:-5, 5:-5] - mu_a ** 2
        sigma_b = cv2.filter2D(b ** 2, -1, window)[5:-5, 5:-5] - mu_b ** 2
        sigma_ab = cv2.filter2D(a * b, -1, window)[5:-5, 5:-5] - mu_a * mu_b
        ssim_map = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / (
                    (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2))
        return ssim_map.mean()

    if len(img1.shape) == 2:
        return ssim_channel(img1, img2)
    elif img1.shape[2] == 3:
        ssim_vals = [ssim_channel(img1[:, :, i], img2[:, :, i]) for i in range(3)]
        return np.mean(ssim_vals)
    else:
        raise ValueError("不支持的图像格式。")

def calculate_mae(img1, img2):
    return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))

def calculate_rmse(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return math.sqrt(mse)

def evaluate_image_pairs(folder_GT, folder_Gen, description):
    files = sorted(glob.glob(os.path.join(folder_GT, '*')))
    psnr_list, ssim_list, mae_list, rmse_list = [], [], [], []
    for gt_path in files:
        base_name = os.path.basename(gt_path)
        gen_path = os.path.join(folder_Gen, base_name)
        if not os.path.exists(gen_path):
            print(f"警告：未找到生成图像 {gen_path}，跳过。")
            continue
        img_GT = cv2.imread(gt_path)
        img_Gen = cv2.imread(gen_path)
        if img_GT is None or img_Gen is None:
            print(f"警告：读取 {base_name} 时出错，跳过。")
            continue
        if img_GT.shape != img_Gen.shape:
            print(f"警告：{base_name} 尺寸不匹配，跳过。")
            continue
        psnr_val = calculate_psnr(img_GT, img_Gen)
        ssim_val = calculate_ssim(img_GT, img_Gen)
        mae_val = calculate_mae(img_GT, img_Gen)
        rmse_val = calculate_rmse(img_GT, img_Gen)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        mae_list.append(mae_val)
        rmse_list.append(rmse_val)
        print(
            f"{base_name:20s} PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}, MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}")
    if psnr_list:
        print(f"\n【{description}】平均指标：")
        print(
            f"PSNR: {np.mean(psnr_list):.2f} dB, SSIM: {np.mean(ssim_list):.4f}, MAE: {np.mean(mae_list):.2f}, RMSE: {np.mean(rmse_list):.2f}")
    else:
        print("无有效图像用于评估。")


# 计算比特每像素 (bpp) 的函数
def compute_bpp(cover_image_path, secret_text):
    """
    根据封面图像尺寸和秘密文本长度计算比特每像素 (bpp)：
      bpp = ((4 + len(secret_text_bytes)) * 8) / (width * height)
    其中 4 字节头信息固定表示秘密文本长度。
    """
    cover_img = Image.open(cover_image_path).convert("RGB")
    width, height = cover_img.size
    secret_bytes = secret_text.encode('utf-8')
    total_bits = (4 + len(secret_bytes)) * 8
    bpp = total_bits / (width * height)
    return bpp

# 数据集处理主流程
def process_dataset(input_cover_folder, secret_text, output_root):
    """
    对输入封面图数据集中的每张图，固定嵌入相同的 secret_text 文本，
    生成四个输出文件夹（均在 output_root 下）：
      - cover      : 原始封面图（复制）
      - secret     : 保存 secret_text（文本文件，与封面图对应）
      - stego      : 隐写图像（将文本嵌入封面图）
      - secret_rev : 从隐写图像中提取恢复的文本（文本文件）
    同时，计算每个图像的 bpp 值，并最终输出平均 bpp。
    """
    out_cover = os.path.join(output_root, "cover")
    out_secret = os.path.join(output_root, "secret")
    out_stego = os.path.join(output_root, "stego")
    out_secret_rev = os.path.join(output_root, "secret_rev")
    for folder in [out_cover, out_secret, out_stego, out_secret_rev]:
        os.makedirs(folder, exist_ok=True)

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    cover_files = sorted([f for f in os.listdir(input_cover_folder) if f.lower().endswith(valid_ext)])

    bpp_list = []

    for file in cover_files:
        cover_path = os.path.join(input_cover_folder, file)
        out_cover_path = os.path.join(out_cover, file)
        out_secret_path = os.path.join(out_secret, os.path.splitext(file)[0] + ".txt")
        out_stego_path = os.path.join(out_stego, file)
        out_secret_rev_path = os.path.join(out_secret_rev, os.path.splitext(file)[0] + ".txt")

        try:
            # 保存原始封面图（复制）
            Image.open(cover_path).convert("RGB").save(out_cover_path)
            # 保存秘密文本（原始文本）
            with open(out_secret_path, 'w', encoding='utf-8') as f:
                f.write(secret_text)
            # 嵌入文本，并获取总嵌入比特数
            total_bits = embed_text_in_image(cover_path, secret_text, out_stego_path)
            # 提取恢复文本
            recovered_text = extract_text_from_image(out_stego_path)
            with open(out_secret_rev_path, 'w', encoding='utf-8') as f:
                f.write(recovered_text)
            # 评估文本恢复情况
            print(f"【{file}】", end=' ')
            evaluate_texts(secret_text, recovered_text)
            # 计算 bpp
            bpp = compute_bpp(cover_path, secret_text)
            bpp_list.append(bpp)
            print(f"{file} 的 bpp: {bpp:.4f} bits per pixel")
        except Exception as e:
            print(f"处理 {file} 时发生错误：{e}")

    if bpp_list:
        print(f"\n平均 bpp: {np.mean(bpp_list):.4f} bits per pixel")

    print("\n评估封面图与隐写图之间的图像指标：")
    evaluate_image_pairs(out_cover, out_stego, "Cover-Stego")
    print("\n注意：秘密文本的评估采用比特错误率计算，上述已输出各文件恢复情况。")


def evaluate_texts(original_text, recovered_text):
    """
    计算原始文本与恢复文本的比特错误率（BER）。
    将文本按 utf-8 编码为二进制后逐位比较，输出比特错误率和是否完全匹配。
    """
    orig_bytes = original_text.encode('utf-8')
    rec_bytes = recovered_text.encode('utf-8')
    max_len = max(len(orig_bytes), len(rec_bytes))
    orig_bits = ''.join(format(b, '08b') for b in orig_bytes).ljust(max_len * 8, '0')
    rec_bits = ''.join(format(b, '08b') for b in rec_bytes).ljust(max_len * 8, '0')
    errors = sum(ob != rb for ob, rb in zip(orig_bits, rec_bits))
    ber = errors / (max_len * 8)
    match = (errors == 0)
    print(f"文本比特错误率: {ber * 100:.2f}%, 完全匹配: {match}")
    return ber, match


if __name__ == '__main__':
    # 设置输入封面图文件夹路径
    input_cover_folder = "E:\hxr\paper_codefile\datasets\DIV2K_256\Training\HR\hr_images"  # 请根据实际路径修改
    # 固定的秘密文本
    secret_text = "这是一个测试文本 hello world 123451616316  21415432654745686598"
    # 输出根目录（将生成 cover, secret, stego, secret_rev 四个文件夹）
    output_root = "image_lsb_stego_text"

    process_dataset(input_cover_folder, secret_text, output_root)
