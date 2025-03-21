#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本代码实现基于论文中4bit-LSB方法的图像隐写：
1. 将秘密图像（保持原始尺寸）取高4位，并打包后嵌入到封面图像中生成隐写图像
2. 从隐写图像中提取出秘密图像，解包并将4位数据乘以16还原至近似原始8位图像
3. 分别评估封面图与隐写图（Cover-Stego）以及秘密图与恢复图（Secret-Recovery）的图像质量指标（PSNR、SSIM、MAE、RMSE）
4. 计算比特每像素（bpp）：((头信息12字节 + 打包后秘密数据字节数)*8)/(封面图像像素数)
5. 将平均评估指标保存到txt文件中
"""

import os, math, glob, io
import numpy as np
import cv2
from PIL import Image

#############################################
# 辅助函数：打包和解包4位数据
#############################################
def pack_4bit_array(arr):
    """
    将一维数组（值范围0~15）中每两个4位数据打包成一个字节。
    如果元素个数为奇数，则最后一个元素高4位，低4位填0。
    """
    flat = arr.flatten()
    n = flat.shape[0]
    if n % 2 != 0:
        flat = np.append(flat, 0)
    packed = (flat[0::2] << 4) | flat[1::2]
    return packed.tobytes()

def unpack_4bit_bytes(data, total_elements):
    """
    将打包后的字节数据解包为一维数组（uint8），每个值0~15。
    total_elements: 需要返回的4位数个数。
    """
    unpacked = []
    for byte in data:
        high = byte >> 4
        low = byte & 0x0F
        unpacked.extend([high, low])
    return np.array(unpacked, dtype=np.uint8)[:total_elements]

#############################################
# 4bit-LSB 图像隐写实现部分（论文方法）
#############################################

def embed_image_in_image(cover_image_path, secret_image_path, output_image_path):
    """
    将秘密图像嵌入到封面图像中，采用论文中4bit-LSB隐写方法：
      1. 读取封面图像（RGB）和秘密图像（RGB，保持原尺寸）。
      2. 对秘密图像，每个像素取高4位：((pixel & 0xF0) >> 4)；打包为字节流。
      3. 构造头信息（12字节）：依次存储秘密图像宽、高、通道数（大端编码）。
      4. 将头信息与打包后的秘密数据拼接，转换为二进制字符串。
      5. 检查封面图像嵌入容量（cover_width * cover_height * 3 * 4 bit），若不足则报错。
      6. 逐像素将封面图像每个像素3通道的低4位替换为数据的4位。
    """
    cover_img = Image.open(cover_image_path).convert("RGB")
    cover_width, cover_height = cover_img.size
    cover_pixels = cover_img.load()

    secret_img = Image.open(secret_image_path).convert("RGB")
    secret_array = np.array(secret_img)  # (H, W, 3)
    secret_4bit = ((secret_array & 0xF0) >> 4).flatten()  # 值在0~15
    packed_secret = pack_4bit_array(secret_4bit)
    # 构造头信息：存储秘密图像宽、高、通道数，每项4字节，共12字节
    header = (secret_img.width.to_bytes(4, byteorder='big') +
              secret_img.height.to_bytes(4, byteorder='big') +
              (3).to_bytes(4, byteorder='big'))
    data = header + packed_secret

    bit_string = ''.join(format(byte, '08b') for byte in data)
    total_bits = len(bit_string)

    capacity_bits = cover_width * cover_height * 3 * 4
    if total_bits > capacity_bits:
        raise ValueError("秘密图像数据太大，无法嵌入当前封面图像中！")

    bit_index = 0
    for y in range(cover_height):
        for x in range(cover_width):
            if bit_index >= total_bits:
                break
            r, g, b = cover_pixels[x, y]
            new_r, new_g, new_b = r & 0xF0, g & 0xF0, b & 0xF0
            if bit_index < total_bits:
                bits = bit_string[bit_index:bit_index+4]
                if len(bits) < 4:
                    bits = bits.ljust(4, '0')
                new_r |= int(bits, 2)
                bit_index += 4
            if bit_index < total_bits:
                bits = bit_string[bit_index:bit_index+4]
                if len(bits) < 4:
                    bits = bits.ljust(4, '0')
                new_g |= int(bits, 2)
                bit_index += 4
            if bit_index < total_bits:
                bits = bit_string[bit_index:bit_index+4]
                if len(bits) < 4:
                    bits = bits.ljust(4, '0')
                new_b |= int(bits, 2)
                bit_index += 4
            cover_pixels[x, y] = (new_r, new_g, new_b)
        if bit_index >= total_bits:
            break

    cover_img.save(output_image_path, format="PNG")
    print(f"嵌入成功：{cover_image_path} 与 {secret_image_path} -> {output_image_path}")
    return total_bits

def extract_image_from_image(stego_image_path, output_secret_image_path):
    """
    从隐写图像中提取出秘密图像。
      1. 读取隐写图像，逐像素提取每个像素通道低4位构成二进制字符串。
      2. 前96位为头信息，依次解析秘密图像的宽、高、通道数。
      3. 根据解析出的尺寸计算总共需要的4位数个数 = width * height * channels，
         并提取剩余比特串，分组转换为字节。
      4. 解包得到秘密图像每个像素每通道的4位数据，重塑成原图尺寸，再乘以16还原。
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

    header_bits = bit_string[:96]  # 12字节头信息
    header_bytes = bytearray()
    for i in range(0, 96, 8):
        header_bytes.append(int(header_bits[i:i+8], 2))
    secret_width = int.from_bytes(header_bytes[0:4], byteorder='big')
    secret_height = int.from_bytes(header_bytes[4:8], byteorder='big')
    secret_channels = int.from_bytes(header_bytes[8:12], byteorder='big')

    total_elements = secret_width * secret_height * secret_channels  # 每个像素对应一个4位数据
    # 每字节包含2个4位数据，因此打包数据长度为 total_elements/2 字节，总比特数为 (total_elements/2)*8
    total_packed_bits = (total_elements // 2) * 8
    packed_bits = bit_string[96:96 + total_packed_bits]

    packed_data = bytearray()
    for i in range(0, len(packed_bits), 8):
        byte = int(packed_bits[i:i+8].ljust(8, '0'), 2)
        packed_data.append(byte)
    packed_data = bytes(packed_data)

    secret_4bit = unpack_4bit_bytes(packed_data, total_elements)
    secret_8bit = (secret_4bit.astype(np.uint16) * 16).astype(np.uint8)
    recovered_array = secret_8bit.reshape((secret_height, secret_width, secret_channels))
    recovered_img = Image.fromarray(recovered_array, mode="RGB")
    recovered_img.save(output_secret_image_path)
    print(f"提取成功：{stego_image_path} -> {output_secret_image_path}")
    return recovered_img

#############################################
# 评估指标测试部分（图像指标）
#############################################

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    def ssim_channel(a, b):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu_a = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
        mu_b = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
        sigma_a = cv2.filter2D(a**2, -1, window)[5:-5, 5:-5] - mu_a**2
        sigma_b = cv2.filter2D(b**2, -1, window)[5:-5, 5:-5] - mu_b**2
        sigma_ab = cv2.filter2D(a*b, -1, window)[5:-5, 5:-5] - mu_a*mu_b
        ssim_map = ((2*mu_a*mu_b + C1) * (2*sigma_ab + C2)) / ((mu_a**2+mu_b**2+C1)*(sigma_a+sigma_b+C2))
        return ssim_map.mean()
    if len(img1.shape)==2:
        return ssim_channel(img1, img2)
    elif img1.shape[2]==3:
        ssim_vals = [ssim_channel(img1[:,:,i], img2[:,:,i]) for i in range(3)]
        return np.mean(ssim_vals)
    else:
        raise ValueError("不支持的图像格式。")

def calculate_mae(img1, img2):
    return np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))

def calculate_rmse(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)
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
        mae_val  = calculate_mae(img_GT, img_Gen)
        rmse_val = calculate_rmse(img_GT, img_Gen)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        mae_list.append(mae_val)
        rmse_list.append(rmse_val)
        print(f"{base_name:20s} PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}, MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}")
    if psnr_list:
        avg_metrics = {
            "PSNR": np.mean(psnr_list),
            "SSIM": np.mean(ssim_list),
            "MAE": np.mean(mae_list),
            "RMSE": np.mean(rmse_list)
        }
        print(f"\n【{description}】平均指标：")
        print(f"PSNR: {avg_metrics['PSNR']:.2f} dB, SSIM: {avg_metrics['SSIM']:.4f}, MAE: {avg_metrics['MAE']:.2f}, RMSE: {avg_metrics['RMSE']:.2f}")
        return avg_metrics
    else:
        print("无有效图像用于评估。")
        return None

#############################################
# 计算比特每像素 (bpp) 的函数
#############################################
def compute_bpp(cover_image_path, secret_image_path):
    """
    根据封面图像尺寸和秘密图像数据大小计算比特每像素 (bpp)：
      bpp = ((12 + packed_secret_length) * 8) / (width * height)
    其中 12 字节为头信息，packed_secret_length为打包后秘密图像数据的字节数。
    """
    cover_img = Image.open(cover_image_path).convert("RGB")
    width, height = cover_img.size
    secret_img = Image.open(secret_image_path).convert("RGB")
    secret_array = np.array(secret_img)
    secret_4bit = ((secret_array & 0xF0) >> 4).flatten()
    packed_secret = pack_4bit_array(secret_4bit)
    total_bits = (12 + len(packed_secret)) * 8
    bpp = total_bits / (width * height)
    return bpp

#############################################
# 数据集处理主流程（图像隐写版）
#############################################
def process_dataset(input_cover_folder, input_secret_folder, output_root):
    """
    对输入数据集中每对封面图和秘密图（按文件名对应）进行处理，
    生成四个输出文件夹（均在 output_root 下）：
      - cover      : 保存原始封面图（复制）
      - secret     : 保存原始秘密图（复制）
      - stego      : 隐写图像（将秘密图嵌入封面图）
      - secret_rev : 从隐写图像中提取恢复的秘密图
    同时，计算每对图像的 bpp 值，并输出平均 bpp，
    以及分别评估封面图与隐写图和秘密图与恢复图之间的图像质量指标，
    最后将平均指标保存到txt文件中。
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
        base_name, _ = os.path.splitext(file)
        secret_path = None
        for ext in valid_ext:
            candidate = os.path.join(input_secret_folder, base_name + ext)
            if os.path.exists(candidate):
                secret_path = candidate
                break
        if secret_path is None:
            print(f"警告：未找到与 {file} 对应的秘密图，跳过。")
            continue

        out_cover_path = os.path.join(out_cover, file)
        out_secret_path = os.path.join(out_secret, os.path.basename(secret_path))
        out_stego_path = os.path.join(out_stego, file)
        out_secret_rev_path = os.path.join(out_secret_rev, os.path.basename(secret_path))

        try:
            Image.open(cover_path).convert("RGB").save(out_cover_path)
            Image.open(secret_path).convert("RGB").save(out_secret_path)
            total_bits = embed_image_in_image(cover_path, secret_path, out_stego_path)
            extract_image_from_image(out_stego_path, out_secret_rev_path)
            bpp = compute_bpp(cover_path, secret_path)
            bpp_list.append(bpp)
            print(f"{file} 的 bpp: {bpp:.4f} bits per pixel")
        except Exception as e:
            print(f"处理 {file} 时发生错误：{e}")

    if bpp_list:
        avg_bpp = np.mean(bpp_list)
        print(f"\n平均 bpp: {avg_bpp:.4f} bits per pixel")
    else:
        avg_bpp = None

    print("\n评估封面图与隐写图之间的图像指标：")
    cover_stego_metrics = evaluate_image_pairs(out_cover, out_stego, "Cover-Stego")
    print("\n评估秘密图与恢复图之间的图像指标：")
    secret_recovery_metrics = evaluate_image_pairs(out_secret, out_secret_rev, "Secret-Recovery")

    # 保存平均指标到文件
    metrics_file = os.path.join(output_root, "evaluation_avg.txt")
    with open(metrics_file, "w",  encoding="utf-8") as f:
        f.write("Cover-Stego 平均指标：\n")
        if cover_stego_metrics:
            f.write(f"PSNR: {cover_stego_metrics['PSNR']:.2f} dB\n")
            f.write(f"SSIM: {cover_stego_metrics['SSIM']:.4f}\n")
            f.write(f"MAE: {cover_stego_metrics['MAE']:.2f}\n")
            f.write(f"RMSE: {cover_stego_metrics['RMSE']:.2f}\n")
        else:
            f.write("无有效图像用于评估。\n")
        f.write("\nSecret-Recovery 平均指标：\n")
        if secret_recovery_metrics:
            f.write(f"PSNR: {secret_recovery_metrics['PSNR']:.2f} dB\n")
            f.write(f"SSIM: {secret_recovery_metrics['SSIM']:.4f}\n")
            f.write(f"MAE: {secret_recovery_metrics['MAE']:.2f}\n")
            f.write(f"RMSE: {secret_recovery_metrics['RMSE']:.2f}\n")
        else:
            f.write("无有效图像用于评估。\n")
        if avg_bpp is not None:
            f.write(f"\n平均 bpp: {avg_bpp:.4f} bits per pixel\n")
        else:
            f.write("无有效图像计算bpp。\n")
    print(f"平均指标已保存到文件：{metrics_file}")

if __name__ == '__main__':
    input_cover_folder = r"E:\hxr\paper_codefile\datasets\DIV2K_small\Validation\HR\hr_images"
    input_secret_folder = r"E:\hxr\paper_codefile\datasets\DIV2K_256\Training\HR\hr_images"
    output_root = "paper_lsb_stego_img"
    process_dataset(input_cover_folder, input_secret_folder, output_root)
