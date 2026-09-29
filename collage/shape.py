"""形状提取：图片 → 布尔掩码 + 几何信息。

对外只暴露 load_mask()；它内部处理三件事：
    ① 白底图 / 透明 PNG 两条路自动选择
    ② 连通域去噪（默认保留所有主连通域 —— 素材常含多个分离主体）
    ③ 计算掩码的几何信息（包围盒、面积、质心）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class ShapeInfo:
    """掩码 + 由它导出的几何量（下游排版全靠这几个数）。"""
    mask: np.ndarray          # bool HxW
    width: int
    height: int
    area: int                 # 掩码像素数
    bbox: tuple[int, int, int, int]   # x0, y0, w, h

    @property
    def center(self) -> tuple[int, int]:
        ys, xs = np.where(self.mask)
        return int(xs.mean()), int(ys.mean())


def load_mask(path_or_img, white_thresh: int = 242, alpha_thresh: int = 12,
              min_ratio: float = 0.012, min_px: int = 2000,
              largest_only: bool = False, fill_holes: bool = True) -> ShapeInfo:
    """白底图 / 透明 PNG → ShapeInfo。

    min_ratio / min_px  连通域保留阈值（两个都满足才留）
    largest_only        只保留最大连通域（素材是单一主体时更快）
    fill_holes          ★ 填充内部孔洞 —— 插画内部有大量线条和封闭区域，
                        不填充会碎成一堆轮廓碎片（剪影填充必须是实心的）
    """
    im = Image.open(path_or_img).convert("RGB") if isinstance(path_or_img, (str, bytes)) \
        else path_or_img.convert("RGB")
    arr = np.asarray(im)
    alpha = None
    try:
        a2 = np.asarray(Image.open(path_or_img).convert("RGBA")) if isinstance(path_or_img, (str, bytes)) else None
        if a2 is not None:
            alpha = a2[..., 3]
    except Exception:
        alpha = None

    if alpha is not None and (alpha < 250).mean() > 0.01:
        mask = alpha > alpha_thresh
    else:
        mask = arr.min(axis=2).astype(np.int16) < white_thresh

    mask = ndimage.binary_opening(mask, iterations=1)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        if largest_only:
            keep_id = int(np.argmax(sizes)) + 1
            mask = (lab == keep_id)
        else:
            keep = np.zeros(n + 1, dtype=bool)
            keep[1:][sizes >= max(mask.sum() * min_ratio, min_px)] = True
            mask = keep[lab]

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("掩码为空：检查白底阈值或图片内容")
    bbox = (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return ShapeInfo(mask=mask, width=arr.shape[1], height=arr.shape[0],
                     area=int(mask.sum()), bbox=bbox)
