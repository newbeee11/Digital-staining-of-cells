import cv2
import numpy as np
import os
from pathlib import Path
from typing import List

class OpenCVPatchBlender:
    def __init__(self, patch_size: int = 640, target_size: int = 512, overlap: int = 64, upscale_size=(1024, 1024)):
        """
        OpenCV 多频段融合器（带 overlap feather mask）
        Args:
            patch_size: 输入 patch 尺寸 (默认 640)
            target_size: 中心区域尺寸 (默认 512)
            overlap: 重叠区域大小 (默认 64)
            upscale_size: 输出上采样尺寸 (默认 1024x1024)
        """
        self.patch_size = patch_size
        self.target_size = target_size
        self.overlap = overlap
        self.upscale_size = upscale_size
        self.center_start = (patch_size - target_size) // 2
        self.center_end = self.center_start + target_size

        self.patches = []
        self.coordinates = []  # 中心区域左上角坐标
        self.filenames = []

        # MultiBandBlender
        self.blender = cv2.detail_MultiBandBlender()
        self.blender.setNumBands(4)

    def _generate_mask(self) -> np.ndarray:
        """生成带 overlap 的 feather mask"""
        mask = np.zeros((self.patch_size, self.patch_size), np.float32)
        o = self.overlap
        c = self.center_start
        t = self.target_size

        # 中心区域全白
        mask[c:c+t, c:c+t] = 1.0

        # 上下渐变
        for i in range(o):
            val = (i+1)/o
            mask[i, o:-o] = val          # top
            mask[-i-1, o:-o] = val       # bottom

        # 左右渐变
        for j in range(o):
            val = (j+1)/o
            mask[o:-o, j] = val          # left
            mask[o:-o, -j-1] = val       # right

        return (mask * 255).astype(np.uint8)

    def add_patch(self, patch: np.ndarray, col: int, row: int, filename: str = None):
        """添加 patch"""
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            raise ValueError(f"Patch 尺寸必须为 {self.patch_size}x{self.patch_size}, 实际: {patch.shape}")
        if patch.dtype != np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)

        self.patches.append(patch)

        # 步长 = target_size
        step = self.target_size
        x = col * step
        y = row * step
        self.coordinates.append((x, y))
        self.filenames.append(filename or f"patch_{len(self.patches)-1}")

        print(f"添加 patch: {filename}, 网格坐标: ({col},{row}), 中心坐标: ({x},{y})")

    def process_all(self) -> List[np.ndarray]:
        """处理所有 patch，返回融合后的中心区域（已上采样）"""
        if not self.patches:
            return []

        # 计算 canvas 大小
        min_x = min(x - self.center_start for x, y in self.coordinates)
        min_y = min(y - self.center_start for x, y in self.coordinates)
        max_x = max(x - self.center_start + self.patch_size for x, y in self.coordinates)
        max_y = max(y - self.center_start + self.patch_size for x, y in self.coordinates)

        canvas_width = int(max_x - min_x)
        canvas_height = int(max_y - min_y)
        self.blender.prepare((0, 0, canvas_width, canvas_height))
        print(f"Canvas 大小: {canvas_width}x{canvas_height}")

        # mask
        mask = self._generate_mask()

        # feed
        for patch, (cx, cy) in zip(self.patches, self.coordinates):
            x = int(cx - self.center_start - min_x)
            y = int(cy - self.center_start - min_y)

            # 🚩 转 int16 (CV_16SC3)
            patch_feed = patch.astype(np.int16)

            self.blender.feed(patch_feed, mask, (x, y))

        # 融合
        result_canvas = np.zeros((canvas_height, canvas_width, 3), np.int16)
        result_mask = np.zeros((canvas_height, canvas_width), np.uint8)
        self.blender.blend(result_canvas, result_mask)

        # 🚩 转回 uint8
        result_canvas = np.clip(result_canvas, 0, 255).astype(np.uint8)

        # 提取中心区域并上采样
        centers = []
        for cx, cy in self.coordinates:
            x = int(cx - min_x)
            y = int(cy - min_y)
            center = result_canvas[y + self.center_start:y + self.center_end,
                                   x + self.center_start:x + self.center_end]

            # 🚩 上采样到 1024×1024
            center_up = cv2.resize(center, self.upscale_size, interpolation=cv2.INTER_CUBIC)
            centers.append(center_up)

        print("融合完成 + 上采样")
        return centers

    def save_centers(self, centers: List[np.ndarray], output_dir: str):
        """保存中心区域（已上采样）"""
        os.makedirs(output_dir, exist_ok=True)
        for i, center in enumerate(centers):
            filename = self.filenames[i]
            output_path = os.path.join(output_dir, f"{Path(filename).stem}.jpg")
            cv2.imwrite(output_path, center)
            print(f"保存: {output_path}")


# ================== 使用示例 ==================
if __name__ == "__main__":
    blender = OpenCVPatchBlender(patch_size=640, target_size=512, overlap=64, upscale_size=(1024, 1024))

    # 2x2 随机 patch 示例
    for row in range(2):
        for col in range(2):
            patch = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            blender.add_patch(patch, col, row, filename=f"patch_{row}_{col}.jpg")

    # 融合 + 上采样
    centers = blender.process_all()

    # 保存
    blender.save_centers(centers, "output/processed_centers")
