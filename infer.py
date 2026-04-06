import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

# ---------------- 项目依赖 ----------------
from neighborV6 import EnhancedResNetGenerator
from centerV7 import DualResNetGenerator
from blendertest import OpenCVPatchBlender

# ---------------- 配置 ----------------
BASE_INPUT_DIR = r"D:\lunwen\aiL00"
# BASE_OUTPUT_DIR = r"C:\Users\WH\Desktop\6"
# INPUT_DIRS = ["WHN-L00","L00-521","L00-522","L00-523","LAC-L00" ]           # 原始输入目录
PREPROCESS_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_WORKERS = 8
PATCH_SIZE = 640
TARGET_SIZE = 512
OVERLAP = 64
UPSCALE_SIZE = (1024, 1024)

def get_output_dir(input_dir,base_dir):
    base = Path(base_dir)
    return base /"Results"/ f"{input_dir}Results/L00", base /"original"/ f"{input_dir}original"

# ==============================================================
#                        模型加载
# ==============================================================

def load_models(device="cuda", checkpoint_dir="checkpoints"):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    G = EnhancedResNetGenerator().to(device)
    G_refine = DualResNetGenerator().to(device)

    def find_latest_checkpoint(prefix):
        if not os.path.exists(checkpoint_dir):
            return None
        ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith(".pth")]
        if not ckpts:
            return None
        try:
            ckpts = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        except Exception:
            ckpts = sorted(ckpts)
        return os.path.join(checkpoint_dir, ckpts[-1])

    checkpoint_path = find_latest_checkpoint("supervised_checkpoint_epoch")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "G_state_dict" in checkpoint:
            G.load_state_dict(checkpoint["G_state_dict"])
            print(f"Loaded G from {checkpoint_path}")
        if "G_refine_state_dict" in checkpoint:
            G_refine.load_state_dict(checkpoint["G_refine_state_dict"])
            print(f"Loaded G_refine from {checkpoint_path}")
    return G, G_refine


# ==============================================================
#                        辅助函数
# ==============================================================

def center_crop_from_upsampled(img, target_size):
    """从3倍上采样后的neighbor中裁剪中心区域"""
    upsampled = F.interpolate(img, scale_factor=3, mode='bilinear', align_corners=False)
    _, _, H, W = upsampled.shape
    target_h, target_w = target_size
    start_h = (H - target_h) // 2
    start_w = (W - target_w) // 2
    center_cropped = upsampled[:, :, start_h:start_h + target_h, start_w:start_w + target_w]
    return center_cropped, upsampled

def get_position(fname):
    """解析列和行"""
    base_name = os.path.splitext(fname)[0]
    if base_name.startswith("B") and base_name.endswith("C"):
        try:
            hex_col = base_name[3:5]
            hex_row = base_name[6:8]
            col = int(hex_col, 16)
            row = int(hex_row, 16)
            return col, row
        except (ValueError, IndexError):
            return None, None
    return None, None

def get_neighbors(files_set, col, row):
    """返回9邻居，缺失时用 None 占位"""
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            n_col, n_row = col + dx, row + dy
            hex_c = f"{n_col:02X}"
            hex_r = f"{n_row:02X}"
            expected_name = f"B00{hex_c}0{hex_r}C.jpg"
            if expected_name in files_set:
                neighbors.append(expected_name)
            else:
                neighbors.append(None)
    return neighbors

def create_grid_image(input_dir, neighbor_names, size=(512,512), center_overlap=(640,640)):
    """拼3x3网格"""
    grid_w, grid_h = size[0] * 3, size[1] * 3
    grid_image = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for idx, name in enumerate(neighbor_names):
        if name is None:
            img = Image.new("RGB", size, (255, 255, 255))
        else:
            img_path = Path(input_dir) / name
            img = Image.open(img_path).convert("RGB").resize(size, Image.LANCZOS)
        row, col = idx // 3, idx % 3
        x_offset, y_offset = col * size[0], row * size[1]
        grid_image.paste(img, (x_offset, y_offset))

    target_w, target_h = center_overlap
    cx, cy = grid_w // 2, grid_h // 2
    x0, y0 = cx - target_w // 2, cy - target_h // 2
    x1, y1 = x0 + target_w, y0 + target_h
    center_patch = grid_image.crop((x0, y0, x1, y1))
    full_grid_resized = grid_image.resize(size, Image.LANCZOS)
    return full_grid_resized, center_patch


# ==============================================================
#                        Dataset
# ==============================================================

class NeighborDataset(Dataset):
    def __init__(self, input_dir, size=(512, 512), overlap_pix=(640,640), transform=None):
        self.input_dir = Path(input_dir)
        self.size = size
        self.overlap_pix = overlap_pix
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        self.files_set = set(self.image_files)
        self.samples = []

        for fname in self.image_files:
            col, row = get_position(fname)
            if col is None:
                continue
            neighbors = get_neighbors(self.files_set, col, row)
            self.samples.append((fname, neighbors))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, neighbors = self.samples[idx]
        neighbor_grid_img, center_patch = create_grid_image(self.input_dir, neighbors, self.size)
        neighbor_tensor = self.transform(neighbor_grid_img) if self.transform else transforms.ToTensor()(neighbor_grid_img)
        center_tensor = self.transform(center_patch) if self.transform else transforms.ToTensor()(center_patch)
        return center_tensor, neighbor_tensor, fname


# ==============================================================
#                        推理阶段
# ==============================================================

def run_pipeline_dataloader(input_dir, save_dir=''):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"输入目录不存在 {input_dir}")
        return False

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, G_refine = load_models(device=device)
    G.eval()
    G_refine.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    dataset = NeighborDataset(input_dir, PREPROCESS_SIZE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    original_save_dir = Path(save_dir)
    os.makedirs(original_save_dir, exist_ok=True)

    target_size = (640, 640)
    with torch.no_grad():
        for centers, neighbors, fnames in tqdm(dataloader, desc="推理中"):
            centers, neighbors = centers.to(device, non_blocking=True), neighbors.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                # fake_neighbors, _ = G(neighbors, return_feats=True)
                # fake_center_refined, _ = center_crop_from_upsampled(fake_neighbors, target_size)
                # refined_output, _ = G_refine(centers, fake_center_refined, return_feats=True)
                fake_neighbors, *other_info = G(neighbors, return_feats=True)
                fake_center_refined, _ = center_crop_from_upsampled(fake_neighbors, target_size)
                refined_output, *extra_info = G_refine(centers, fake_center_refined, return_feats=True)

            refined_imgs = (refined_output.cpu() * 0.5 + 0.5).clamp(0, 1)

            for i, fname in enumerate(fnames):
                col, row = get_position(fname)
                if col is None:
                    continue
                orig_img = refined_imgs[i].permute(1, 2, 0).numpy()
                orig_img_uint8 = (orig_img * 255).astype(np.uint8)
                orig_pil = Image.fromarray(orig_img_uint8)
                orig_pil.save(original_save_dir / fname)

    print("=== DataLoader 推理完成，PatchBlender 开始后处理 ===")
    return True


# ==============================================================
#                        后处理阶段
# ==============================================================

def run_blender(input_dir, save_dir="output/processed_centers"):
    PATCH_SIZE = 640
    TARGET_SIZE = 512
    OVERLAP = 64
    blender = OpenCVPatchBlender(patch_size=PATCH_SIZE, target_size=TARGET_SIZE, overlap=OVERLAP)

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    for fname in tqdm(image_files, desc="添加patches"):
        base_name = os.path.splitext(fname)[0]
        if base_name.startswith("B") and base_name.endswith("C"):
            try:
                col = int(base_name[3:5], 16)
                row = int(base_name[6:8], 16)
            except:
                continue
        else:
            continue

        img_path = Path(input_dir) / fname
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue

            if img.shape[0] != PATCH_SIZE or img.shape[1] != PATCH_SIZE:
                print(f"图像 {fname} 尺寸不正确: {img.shape}, 期望: {PATCH_SIZE}x{PATCH_SIZE}")
                continue

            blender.add_patch(img, col, row, fname)
        except Exception as e:
            print(f"处理图像 {fname} 时出错: {e}")
            continue

    if blender.patches:
        print("开始处理所有patches...")
        smoothed_centers = blender.process_all()
        blender.save_centers(smoothed_centers, save_dir)
        print(f"=== 处理完成，结果已保存到 {save_dir} ===")
    else:
        print("没有找到可处理的patches")


# ==============================================================
#                        主执行入口
# ==============================================================

# ==============================================================
#                        主执行入口
# ==============================================================

# ==============================================================
#                        主执行入口
# ==============================================================

# ==============================================================
#                        主执行入口
# ==============================================================

# ==============================================================
#                        主执行入口
# ==============================================================

if __name__ == "__main__":
    def find_l00_folders(root_path, max_depth=5):
        """递归查找所有名为 L00 的文件夹"""
        l00_folders = []

        def search(current_path, current_depth):
            if current_depth > max_depth:
                return

            if current_path.name == "L00" and current_path.is_dir():
                # 检查是否有图片文件
                try:
                    image_files = [f for f in os.listdir(current_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                    b_files = [f for f in image_files if f.startswith("B") and f.endswith("C.jpg")]
                    if b_files:
                        l00_folders.append(current_path)
                        # 显示找到的L00文件夹信息
                        rel_path = current_path.relative_to(root_path) if current_path.is_relative_to(
                            root_path) else current_path
                        print(f"  ✅ 找到L00: {rel_path} ({len(b_files)}个B*C文件)")
                except:
                    # 即使无法访问也添加到列表
                    l00_folders.append(current_path)

            # 递归搜索子目录
            try:
                for item in current_path.iterdir():
                    if item.is_dir():
                        search(item, current_depth + 1)
            except:
                pass

        search(root_path, 0)
        return l00_folders


    # 1. 运行预处理
    print("=" * 80)
    print("🚀 开始预处理阶段 (解压+清理+复制)")
    print("=" * 80)

    try:
        current_dir = Path(__file__).parent
        if (current_dir / "jieya.py").exists():
            import jieya

            # 先检查是否有压缩文件
            base_path = Path(BASE_INPUT_DIR)
            print(f"\n📁 目标目录: {BASE_INPUT_DIR}")

            # 列出所有压缩文件
            supported_exts = ['.zip', '.rar', '.tar', '.gz', '.bz2', '.7z', '.tgz', '.tbz2']
            archives = []
            for ext in supported_exts:
                archives.extend(base_path.glob(f"*{ext}"))

            if archives:
                print(f"📦 找到 {len(archives)} 个压缩文件:")
                for archive in archives[:10]:  # 只显示前10个
                    try:
                        rel_path = archive.relative_to(base_path) if archive.is_relative_to(base_path) else archive
                        print(f"  - {rel_path}")
                    except:
                        print(f"  - {archive.name}")
                if len(archives) > 10:
                    print(f"  ... 还有{len(archives) - 10}个文件未显示")
            else:
                print("📭 未找到压缩文件，可能已解压或无需解压")

            # 执行预处理
            print("\n⚙️  开始预处理...")
            jieya.preprocess_input_directory(BASE_INPUT_DIR)
            print("✅ 预处理完成")
        else:
            print("❌ 未找到jieya.py文件")
            print("请确保 jieya.py 文件与主程序在同一目录")
    except Exception as e:
        print(f"❌ 预处理出错: {e}")
        import traceback

        traceback.print_exc()

    # 2. 查找所有 L00 文件夹
    print("\n" + "=" * 80)
    print("🔍 查找L00文件夹")
    print("=" * 80)

    base_path = Path(BASE_INPUT_DIR)
    l00_folders = find_l00_folders(base_path, max_depth=5)

    if not l00_folders:
        print("❌ 未找到L00文件夹")
        print("\n可能的原因:")
        print("1. 压缩文件没有正确解压")
        print("2. L00文件夹不存在或不包含B*C.jpg文件")
        print("3. 搜索深度不够，可以增加max_depth参数")
        print("4. 目录结构不符合预期")
        sys.exit(0)

    # 3. 处理每个L00文件夹
    print("\n" + "=" * 80)
    print("🤖 开始模型推理")
    print("=" * 80)

    processed_count = 0
    failed_count = 0

    for idx, l00_dir in enumerate(l00_folders):
        print(f"\n[{idx + 1}/{len(l00_folders)}] 处理: {l00_dir.name}")

        # 获取相对路径以便显示
        try:
            rel_path = l00_dir.relative_to(base_path) if l00_dir.is_relative_to(base_path) else l00_dir
            print(f"  路径: {rel_path}")
        except:
            print(f"  路径: {l00_dir}")

        # 创建临时目录
        temp_dir = l00_dir.parent / f"{l00_dir.name}_temp"
        temp_dir.mkdir(exist_ok=True)

        try:
            # 检查是否有图片
            image_files = [f for f in os.listdir(l00_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            b_files = [f for f in image_files if f.startswith("B") and f.endswith("C.jpg")]

            if not b_files:
                print(f"  ⚠️  无B*C图片，跳过")
                failed_count += 1
                continue

            print(f"  📊 {len(b_files)}个B*C文件")

            # 执行推理
            success = run_pipeline_dataloader(str(l00_dir), save_dir=str(temp_dir))

            if success:
                # 后处理
                run_blender(str(temp_dir), str(l00_dir))
                print(f"  ✅ 完成")
                processed_count += 1
            else:
                print(f"  ❌ 推理失败")
                failed_count += 1

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            failed_count += 1

        finally:
            # 清理临时目录
            import shutil

            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

    # 4. 输出结果
    print("\n" + "=" * 80)
    print("📊 处理结果汇总")
    print("=" * 80)
    print(f"✅ 成功处理: {processed_count} 个文件夹")
    print(f"❌ 处理失败: {failed_count} 个文件夹")
    print(f"📁 总计发现: {len(l00_folders)} 个文件夹")

    if processed_count > 0:
        print("🎉 处理完成！")
    else:
        print("⚠️  没有文件夹被成功处理")

    sys.exit(0)