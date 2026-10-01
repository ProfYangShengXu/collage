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


# ★ 细胞密度：撒在 _TILE_PX 小样上的点数，必须是【常数】。
# 曾用 `n_cells = max(10, (w*h)//420)`（按块面积），但 _worley_tile 永远在 _TILE_PX×_TILE_PX
# 的小样上撒点 —— 两个坐标系不一致，导致细胞大小随块尺寸反向变化：50×50 的块只有 10 个点，
# 细胞直径被放大到 68px（看着像贴的马赛克瓷砖）；700×400 的块有 666 个点，细胞仅 8px（细纹）。
# 四叉树在主体周围分出的正是小块，于是"马赛克"全冒在主图边上。固定成常数后所有块纹理一致。
# n_cells = _TILE_PX² / (π · r²)。调这个常数即可改细胞粗细：
#   直径 40px → 29 ｜ 28px → 60 ｜ 23px → 90 ｜ 13px → 278
# ★ 2026-10-01：初值 278（13px）—— 用户反馈「怎么全是磨砂」，13px 在 1188px 画布上细到看不出
# 网状骨架，缩小后与 grain 白噪声无法区分。参考样例（examples/tiger-shikigami.jpg）的细胞约 20px（1080 画布），
# 折算到本图约 22px，故取 90（≈23px）。
_CELL_N = 90


# ★ 细胞边界混入 accent 的上限。
# 0.45 时底纹弱到看不见；1.0（原值）时整片发花。0.80 用于 cell 块数量少的情况；
# ★ 2026-10-01 降到 0.55 —— 块放大后仍被判「无意义组件还是太碎」，根因不是块尺寸而是
# 【块间对比】：cell 块与 grain 块的明暗差让大块也像贴瓷砖。降对比比缩块更治本。
# 参考样例的底纹对比来自【深色素材本身】，而这里的 accent 是从浅色素材采样来的亮蓝，
# 同样比例下会更跳 —— 所以上限要按素材亮度自适应，不能照抄。
_CELL_MIX = 0.55


def halftone(w: int, h: int, pitch: int = 9, angle: float = 18.0,
             seed: int = 0, var: float = 0.45) -> np.ndarray:
    """半调网点：旋转网格上的圆点。返回 0..1，1 = 点内（要上色）。

    印刷语汇里最核心的一种质感（Riso 风格全靠它）。做法是把坐标旋转后取模得到
    网格坐标，再按到格心的距离判定圆 —— 所有运算都是整幅向量的，没有逐像素循环。
    `var` 给网点直径加低频起伏，模拟专色上墨不匀；纯常数会显得像程序生成的贴图。
    """
    a = np.deg2rad(angle)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xr = xx * np.cos(a) + yy * np.sin(a)
    yr = -xx * np.sin(a) + yy * np.cos(a)
    fx = (xr % pitch) / pitch - 0.5
    fy = (yr % pitch) / pitch - 0.5
    d = np.hypot(fx, fy) * 2.0                       # 0(格心) .. ~1.4(格角)

    # 低频起伏：在粗网格上撒随机值再放大，得到平滑的"上墨不匀"
    rng = np.random.default_rng(seed)
    gh, gw = max(2, h // 48), max(2, w // 48)
    low = rng.random((gh, gw)).astype(np.float32)
    low = np.asarray(Image.fromarray((low * 255).astype(np.uint8))
                     .resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    radius = 0.62 + (low - 0.5) * 2 * var
    return np.clip((radius - d) * 8.0, 0.0, 1.0)


def scanline(w: int, h: int, pitch: int = 4) -> np.ndarray:
    """扫描线：每 pitch 行一条细线。返回 0..1，1 = 线。"""
    yy = np.arange(h, dtype=np.float32)[:, None]
    line = np.clip(1.0 - (yy % pitch) / max(1.0, pitch * 0.35), 0.0, 1.0)
    return np.repeat(line, w, axis=1)


def fiber(w: int, h: int, strength: float = 0.10, seed: int = 0) -> np.ndarray:
    """纸张纤维：各向异性噪声 → 亮度乘数，形状 (h, w)。

    与 grain 的区别是【方向性】：先在横向上细、纵向上粗的网格里撒点再放大，
    得到沿一个方向拉长的纹理，像未涂布纸的纤维。grain 是各向同性的，放大后只会
    变成均匀沙点，缺少这种方向感。
    """
    rng = np.random.default_rng(seed)
    gh, gw = max(2, h // 6), max(2, w // 40)        # 纵向密、横向疏 → 横向拉长
    small = rng.normal(0.0, 1.0, (gh, gw)).astype(np.float32)
    img = Image.fromarray(((small * 0.25 + 0.5) * 255).astype(np.uint8))
    big = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    return np.clip(1.0 + (big - 0.5) * 2.0 * strength, 0.6, 1.4)


def cell_soften(base, e, accent_rgb):
    """细胞边界按 _CELL_MIX 上限混入 accent（供 tile 复用）。"""
    t = (np.clip(e * 3.2, 0, 1) * _CELL_MIX)[..., None]
    return base * (1 - t) + np.array(accent_rgb, dtype=float)[None, None, :] * t


def tile(w: int, h: int, base_rgb, mode: str, seed: int = 0,
         accent_rgb=(178, 52, 40)) -> Image.Image:
    """生成一块底纹。

    mode: grain     磨砂（各向同性白噪声）
          cell      细胞噪声（Worley 边界）
          halftone  半调网点（印刷/Riso 的核心语汇）
          scanline  扫描线（每 4 行一条）
          paper     纸张纤维（各向异性，横向拉长）
          riso      套印错位（双网点层 + 偏移 + 颗粒）
          ink       深色块（用于反白块底）
          flat      纯色
    """
    base = np.array(base_rgb, dtype=float)[None, None, :] * np.ones((h, w, 1))
    acc = np.array(accent_rgb, dtype=float)[None, None, :]

    if mode == "grain":
        arr = base * grain(w, h, 0.12)[..., None]
    elif mode == "cell":
        e = worley(w, h, _CELL_N, seed)
        arr = cell_soften(base, e, accent_rgb) * grain(w, h, 0.06)[..., None]
    elif mode == "halftone":
        # 网点用 accent 上色，浓度上限沿用 _CELL_MIX（同一条"底纹不许抢主体"的约束）
        m = (halftone(w, h, seed=seed) * _CELL_MIX)[..., None]
        arr = base * (1 - m) + acc * m
        arr = arr * grain(w, h, 0.05)[..., None]
    elif mode == "scanline":
        m = (scanline(w, h) * _CELL_MIX * 0.55)[..., None]
        arr = base * (1 - m) + acc * m
        arr = arr * grain(w, h, 0.04)[..., None]
    elif mode == "paper":
        arr = base * fiber(w, h, 0.10, seed)[..., None]
    elif mode == "riso":
        # 套印错位：两层不同角度的网点，其中一层横向偏移几像素（模拟套印不准）
        m1 = halftone(w, h, pitch=9, angle=18.0, seed=seed)
        m2 = np.roll(halftone(w, h, pitch=9, angle=18.0, seed=seed + 1), 3, axis=1)
        m = np.clip(m1 + m2 * 0.8, 0, 1)[..., None] * _CELL_MIX
        arr = base * (1 - m) + acc * m
        arr = arr * grain(w, h, 0.07)[..., None]
    elif mode == "ink":
        arr = base * grain(w, h, 0.05)[..., None]
    else:
        arr = base
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
