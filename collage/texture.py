"""质感生成：磨砂噪点、细胞噪声（Worley）、纯色。

★ 细胞噪声在大画幅上直接算会爆内存（26 个点 × 整块像素的数组），
  所以先在 192px 小样上算一次并缓存，再平铺到目标尺寸。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

_TILE_PX = 192          # 小样边长
_tile_cache: dict = {}


def grain(w: int, h: int, strength: float = 0.12,
          rng: np.random.Generator | None = None) -> np.ndarray:
    """磨砂：高斯噪点 → 亮度乘数，形状 (h, w)。"""
    rng = rng or np.random.default_rng()
    return np.clip(1.0 + rng.normal(0.0, 1.0, (h, w)) * strength, 0.6, 1.4)


def _worley_tile(n_cells: int, seed: int) -> np.ndarray:
    """在 _TILE_PX 小样上算一次二阶距离差（细胞边界）。"""
    key = (n_cells, seed)
    if key in _tile_cache:
        return _tile_cache[key]
    r = np.random.default_rng(seed)
    pts = r.random((n_cells, 2)) * _TILE_PX
    yy, xx = np.mgrid[0:_TILE_PX, 0:_TILE_PX].astype(np.float32)
    dists = np.stack([np.hypot(xx - px, yy - py) for px, py in pts])
    srt = np.partition(dists, 1, axis=0)
    edge = srt[1] - srt[0]
    edge = (edge / (edge.max() + 1e-6)).astype(np.float32)
    _tile_cache[key] = edge
    return edge


def worley(w: int, h: int, n_cells: int, seed: int = 0) -> np.ndarray:
    """细胞噪声：0..1，1 = 细胞边界。按小样平铺，代价与目标尺寸无关。"""
    base = _worley_tile(n_cells, seed)
    ny, nx = int(np.ceil(h / _TILE_PX)), int(np.ceil(w / _TILE_PX))
    return np.tile(base, (ny, nx))[:h, :w]


def tile(w: int, h: int, base_rgb, mode: str, seed: int = 0,
         accent_rgb=(178, 52, 40)) -> Image.Image:
    """生成一块底纹。

    mode: grain  磨砂
          cell   细胞噪声（= base 打底 + accent 描边界）
          ink    深色块（用于反白块底）
          flat   纯色
    """
    base = np.array(base_rgb, dtype=float)[None, None, :] * np.ones((h, w, 1))
    if mode == "grain":
        arr = base * grain(w, h, 0.12)[..., None]
    elif mode == "cell":
        e = worley(w, h, max(10, (w * h) // 420), seed)
        t = np.clip(e * 3.2, 0, 1)[..., None]
        arr = base * (1 - t) + np.array(accent_rgb, dtype=float)[None, None, :] * t
        arr = arr * grain(w, h, 0.06)[..., None]
    elif mode == "ink":
        arr = base * grain(w, h, 0.05)[..., None]
    else:
        arr = base
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
