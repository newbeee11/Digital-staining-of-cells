import os

from sympy import false

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


# ================== 配置 ==================
FAKE_DIR = r"E:\baseline\pix2pix_infer_256\ESY0359993"
REAL_DIR = r"D:\lunwen\rengongL00\ESY0359993"

SAVE_CSV = True
CSV_NAME = r"E:\baseline\pix2pix_infer_256\9993metrics_per_image.csv"

NUM_WORKERS = 8   # 8 或 12 都可以试
# ========================================


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def center_crop_to_same_size(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = min(h1, h2)
    w = min(w1, w2)

    def crop_center(img, th, tw):
        h0, w0 = img.shape[:2]
        y1 = max((h0 - th) // 2, 0)
        x1 = max((w0 - tw) // 2, 0)
        return img[y1:y1 + th, x1:x1 + tw]

    return crop_center(img1, h, w), crop_center(img2, h, w)


# 🚀 加速版计算
def calc_fast(fake_path, real_path):
    fake = cv2.imread(str(fake_path))
    real = cv2.imread(str(real_path))

    if fake is None or real is None:
        raise FileNotFoundError(f"读取失败: {fake_path} 或 {real_path}")

    fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
    real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)

    if fake.shape != real.shape:
        fake, real = center_crop_to_same_size(fake, real)

    # PSNR
    psnr = peak_signal_noise_ratio(real, fake, data_range=255)

    # ⚡ 灰度SSIM（关键提速）
    fake_gray = cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY)
    real_gray = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)

    ssim = structural_similarity(real_gray, fake_gray, data_range=255)

    return psnr, ssim


def save_metrics_csv(rows, csv_name):
    with open(csv_name, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    t0 = time.time()

    fake_dir = Path(FAKE_DIR)
    real_dir = Path(REAL_DIR)

    # 防止填错路径
    if str(fake_dir.resolve()) == str(real_dir.resolve()):
        raise ValueError("FAKE_DIR 和 REAL_DIR 不能是同一个文件夹！")

    fake_files = sorted(list_images(fake_dir))
    real_files = sorted(list_images(real_dir))

    fake_map = {p.name: p for p in fake_files}
    real_map = {p.name: p for p in real_files}

    names = sorted(set(fake_map.keys()) & set(real_map.keys()))

    print(f"AI图数量: {len(fake_files)}")
    print(f"真实图数量: {len(real_files)}")
    print(f"匹配到 {len(names)} 对图像")

    if len(names) == 0:
        raise ValueError("没有匹配到同名图像")

    psnr_list = []
    ssim_list = []
    rows = []

    # 🚀 多线程加速
    def process(name):
        p, s = calc_fast(fake_map[name], real_map[name])
        return name, p, s

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process, names), total=len(names)))

    for name, p, s in results:
        psnr_list.append(p)
        ssim_list.append(s)
        rows.append({"name": name, "psnr": p, "ssim": s})

    mean_ssim = float(np.mean(ssim_list))
    mean_psnr = float(np.mean(psnr_list))

    if SAVE_CSV:
        save_metrics_csv(rows, CSV_NAME)
        print(f"逐张结果已保存到: {CSV_NAME}")

    print("\n========== 最终结果 ==========")
    print(f"SSIM : {mean_ssim:.4f}")
    print(f"PSNR : {mean_psnr:.4f} dB")
    print(f"总耗时: {time.time() - t0:.2f} 秒")


if __name__ == "__main__":
    main()