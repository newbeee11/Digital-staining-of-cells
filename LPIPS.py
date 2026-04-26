import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

import torch
import lpips


# ================== 手动配置 ==================
FAKE_DIR = r"E:\baseline\duiqi\munit\ESY0359993\src_center"
REAL_DIR = r"D:\lunwen\duiqi1\ESY0359993\tgt_center"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NET = "alex"   # 可选: 'alex', 'vgg', 'squeeze'
# ============================================


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def load_rgb(path):
    """
    读取图像并转为 RGB
    """
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def img_to_tensor(img, device):
    """
    HWC uint8 RGB -> torch tensor
    LPIPS要求输入范围为 [-1, 1]
    shape: [1, 3, H, W]
    """
    img = img.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def match_pairs(fake_dir, real_dir):
    """
    按文件名匹配 fake 和 real
    """
    fake_files = list_images(fake_dir)
    real_files = list_images(real_dir)

    real_dict = {p.stem: p for p in real_files}
    pairs = []

    for f in fake_files:
        if f.stem in real_dict:
            pairs.append((f, real_dict[f.stem]))

    return pairs


def main():
    print(f"设备: {DEVICE}")
    print(f"LPIPS网络: {NET}")

    pairs = match_pairs(FAKE_DIR, REAL_DIR)
    print(f"匹配成功: {len(pairs)} 对")

    if len(pairs) == 0:
        print("没有匹配到同名图像，请检查文件名是否一致。")
        return

    # 初始化 LPIPS 模型
    loss_fn = lpips.LPIPS(net=NET).to(DEVICE)
    loss_fn.eval()

    scores = []

    with torch.no_grad():
        for fake_path, real_path in tqdm(pairs, desc="计算 LPIPS"):
            fake_img = load_rgb(fake_path)
            real_img = load_rgb(real_path)

            # 如果尺寸不一致，resize real 到 fake 的尺寸
            if fake_img.shape != real_img.shape:
                real_img = cv2.resize(real_img, (fake_img.shape[1], fake_img.shape[0]), interpolation=cv2.INTER_AREA)

            fake_tensor = img_to_tensor(fake_img, DEVICE)
            real_tensor = img_to_tensor(real_img, DEVICE)

            score = loss_fn(fake_tensor, real_tensor)
            score = score.item()
            scores.append(score)

    scores = np.array(scores, dtype=np.float32)

    print("\n" + "=" * 50)
    print(f"样本数: {len(scores)}")
    print(f"平均 LPIPS: {scores.mean():.6f}")
    print(f"标准差: {scores.std():.6f}")
    print(f"最小值: {scores.min():.6f}")
    print(f"最大值: {scores.max():.6f}")
    print("=" * 50)

    # 保存每张图的结果
    out_txt = Path(FAKE_DIR).parent / "lpips_scores.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for (fake_path, real_path), s in zip(pairs, scores):
            f.write(f"{fake_path.name}\t{real_path.name}\t{s:.6f}\n")
        f.write("\n")
        f.write(f"count\t{len(scores)}\n")
        f.write(f"mean\t{scores.mean():.6f}\n")
        f.write(f"std\t{scores.std():.6f}\n")
        f.write(f"min\t{scores.min():.6f}\n")
        f.write(f"max\t{scores.max():.6f}\n")

    print(f"结果已保存到: {out_txt}")


if __name__ == "__main__":
    main()