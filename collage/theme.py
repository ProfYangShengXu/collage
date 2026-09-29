"""配色派生：从原图采样主色，生成整套排版配色。

为什么不用固定配色：换一张素材（色调完全不同）时，固定配色会打架。
从原图派生后，素材换掉、配色自动跟随，代码不用改。

★ 选主色时用「饱和度 × 明度」打分，而不是只看饱和度 ——
  否则会被深色轮廓/暗部拉偏（实测：只看饱和度会选出深棕而非亮黄）。
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


def _to_hsv(rgb) -> tuple[float, float, float]:
    r, g, b = (c / 255.0 for c in rgb[:3])
    return colorsys.rgb_to_hsv(r, g, b)


def _from_hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(r * 255), int(g * 255), int(b * 255))


@dataclass
class Theme:
    """一套排版配色。"""
    accent: tuple[int, int, int]          # 强调色（主色压暗）
    ink: tuple[int, int, int]             # 墨色（文字 / 反白块底）
    paper: tuple[int, int, int]           # 纸色（底）
    tints: list = field(default_factory=list)   # 底纹基色（由浅到深若干档）
    accent_raw: tuple[int, int, int] = (0, 0, 0)  # 未处理的采样原色

    def as_dict(self) -> dict:
        return {"accent": self.accent, "ink": self.ink,
                "paper": self.paper, "tints": self.tints,
                "accent_raw": self.accent_raw}


def theme_from_image(img, mask, n_colors: int = 8, verbose: bool = False) -> Theme:
    """从原图（掩码内区域）采样主色，派生排版配色。

    img   PIL Image (RGB)
    mask  bool ndarray，True = 主体（只从主体取色，避免背景白底干扰）
    """
    arr = np.asarray(img.convert("RGB"))
    px = arr[mask]
    if len(px) < 100:
        px = arr.reshape(-1, 3)

    # ① 中位切分量化取主色簇
    strip = Image.fromarray(px.reshape(-1, 1, 3).astype(np.uint8), "RGB")
    q = strip.quantize(colors=n_colors, method=Image.MEDIANCUT)
    palette = q.getpalette()
    cands = [(cnt, tuple(palette[i * 3: i * 3 + 3])) for cnt, i in q.getcolors()]

    # ② 选主色：饱和度 × 明度^2.2（排除近灰与暗部）
    def _score(c):
        _, s, v = _to_hsv(c)
        return s * (v ** 2.2)

    colored = [(n, c) for n, c in cands
               if _to_hsv(c)[1] > 0.10 and _to_hsv(c)[2] > 0.25]
    accent = max(colored, key=lambda t: _score(t[1]))[1] if colored \
        else max(cands, key=lambda t: t[0])[1]

    # ③ 派生整套
    h, s, v = _to_hsv(accent)
    th = Theme(
        accent     = _from_hsv(h, min(0.95, s * 1.05), max(0.42, v * 0.82)),
        ink        = _from_hsv(h, min(0.55, s * 0.85), 0.11),
        paper      = _from_hsv(h, 0.045, 0.975),
        tints      = [_from_hsv(h, 0.060, 0.945),
                      _from_hsv(h, 0.075, 0.915),
                      _from_hsv(h, 0.045, 0.975),
                      _from_hsv(h, min(0.10, s * 0.20), 0.925),
                      _from_hsv(h, min(0.12, s * 0.22), 0.895)],
        accent_raw = accent,
    )
    if verbose:
        print(f"[theme] accent={accent} -> ink={th.ink} paper={th.paper}")
    return th
