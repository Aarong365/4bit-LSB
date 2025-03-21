#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量计算两个文件夹下图像的比特准确率，并将测试结果写入TXT日志。
假设两文件夹中同名文件互为对应图像，以便对比。
使用示例：
  python calculate_folder_bit_accuracy.py --folder_orig /path/to/orig --folder_rev /path/to/rev
"""

import argparse
import os
import numpy as np
from PIL import Image

def calculate_bit_accuracy(original_image_path, recovered_image_path):
    """
    计算原始图像与恢复图像之间的比特准确率（百分比）。
    1) 读取并转换为RGB模式；
    2) flatten后调用 np.unpackbits 展开为二进制；
    3) 逐比特比较，计算错误率并返回 (1 - 错误率)×100。
    """
    orig = np.array(Image.open(original_image_path).convert("RGB"), dtype=np.uint8)
    rec = np.array(Image.open(recovered_image_path).convert("RGB"), dtype=np.uint8)

    if orig.shape != rec.shape:
        raise ValueError(f"图像尺寸不匹配：{original_image_path} vs {recovered_image_path}")

    orig_bits = np.unpackbits(orig.flatten())
    rec_bits = np.unpackbits(rec.flatten())

    error_bits = np.sum(orig_bits != rec_bits)
    total_bits = orig_bits.size
    accuracy = (1 - error_bits / total_bits) * 100
    return accuracy

def evaluate_folder_pairs(folder_orig, folder_rev, output_log):
    """
    在folder_orig和folder_rev中寻找同名文件，对每对图像计算比特准确率。
    结果既打印在命令行，也保存到output_log日志文件中。
    返回所有匹配文件的平均准确率。
    """
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    orig_files = sorted([
        f for f in os.listdir(folder_orig)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])

    if not orig_files:
        print(f"[警告] 在 {folder_orig} 中未找到任何有效图像文件。")
        return None

    # 记录所有输出行，最终写入日志
    log_lines = []
    log_lines.append(f"原始文件夹: {folder_orig}\n恢复文件夹: {folder_rev}\n\n")

    accuracies = []
    for file_name in orig_files:
        orig_path = os.path.join(folder_orig, file_name)
        rev_path  = os.path.join(folder_rev,  file_name)  # 同名文件
        if not os.path.exists(rev_path):
            warn_line = f"[警告] {rev_path} 不存在，跳过。"
            print(warn_line)
            log_lines.append(warn_line + "\n")
            continue

        try:
            acc = calculate_bit_accuracy(orig_path, rev_path)
            accuracies.append(acc)
            result_line = f"{file_name:20s} 比特准确率: {acc:.2f}%"
            print(result_line)
            log_lines.append(result_line + "\n")
        except Exception as e:
            err_line = f"[错误] 比对 {file_name} 时出错: {e}"
            print(err_line)
            log_lines.append(err_line + "\n")

    # 若有成功结果，则计算平均值
    if accuracies:
        avg_acc = np.mean(accuracies)
        avg_line = f"\n平均比特准确率: {avg_acc:.2f}%\n"
        print(avg_line)
        log_lines.append(avg_line)
    else:
        avg_acc = None
        no_line = "\n没有成功比较的图像，无法计算平均比特准确率。\n"
        print(no_line)
        log_lines.append(no_line)

    # 将log_lines写入日志文件
    with open(output_log, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    return avg_acc

def main():
    import argparse

    parser = argparse.ArgumentParser(description="批量计算两个文件夹下图像的比特准确率，并输出到TXT日志。")
    parser.add_argument("--folder_orig", type=str, required=True,
                        help="原始（或秘密）图像所在文件夹路径")
    parser.add_argument("--folder_rev", type=str, required=True,
                        help="恢复（或解密）图像所在文件夹路径")
    parser.add_argument("--output_log", type=str, default="bit_accuracy_log.txt",
                        help="输出日志文件名（可含路径），默认 bit_accuracy_log.txt")
    args = parser.parse_args()

    folder_orig = args.folder_orig
    folder_rev = args.folder_rev
    output_log = args.output_log

    if not os.path.isdir(folder_orig):
        print(f"错误：{folder_orig} 不是有效的文件夹。")
        return
    if not os.path.isdir(folder_rev):
        print(f"错误：{folder_rev} 不是有效的文件夹。")
        return

    evaluate_folder_pairs(folder_orig, folder_rev, output_log)


if __name__ == "__main__":
    main()
