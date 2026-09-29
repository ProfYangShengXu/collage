"""渲染层：三种呈现范式 + 信息层。

三种范式共用同一套「排版层」，区别只在与形状的关系：

    剪影填充    排版 ∩ 剪影          → 排版本身构成形状
    原图加背景  排版铺满 + 原图贴上去 → 原图是主角，排版是背景
    背景负形    排版 ∩ ¬剪影         → 形状以留白负形出现

范式二和三的排版层必须用「贴边版式」——
用版心版式会被剪影整个挖掉，只剩碎片。
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .layout import (FONTS, LayoutSpec, build_edge_layout, build_layout,
                     text_size)
from .shape import ShapeInfo
from .theme import Theme

PARADIGMS = ("silhouette", "photo", "negative")

LABELS = {
    "silhouette": "剪影填充",
    "photo": "原图加背景",
    "negative": "背景负形",
}


def _add_info(img: Image.Image, spec: LayoutSpec, theme: Theme) -> Image.Image:
    """画布四角的信息层（标题 / 副标题 / 元信息）。"""
    d = ImageDraw.Draw(img)
    W, H = img.size
    M = int(W * 0.055)

    d.rectangle([M - 6, M - 6, M + int(W * 0.42), M + 108], fill=theme.paper)
    h1 = 40 if W > 800 else 26
    d.text((M, M), spec.title, font=FONTS.get(spec.font_bold, h1), fill=theme.ink)
    d.text((M, M + int(h1 * 1.25)), spec.title_sub,
           font=FONTS.get(spec.font_mono, max(10, int(h1 * 0.4))), fill=theme.accent)

    for k, ln in enumerate(spec.meta):
        f = FONTS.get(spec.font_mono, 13)
        b = d.textbbox((0, 0), ln, font=f)
        y = H - M - (len(spec.meta) - k) * 22
        d.rectangle([W - M - (b[2] - b[0]) - 6, y - 6, W - M + 6, y + 18],
                    fill=theme.paper)
        d.text((W - M - (b[2] - b[0]), y), ln, font=f, fill=theme.ink)
    return img


def render(shape: ShapeInfo, spec: LayoutSpec, theme: Theme,
           paradigm: str = "silhouette", source_img: Image.Image | None = None,
           add_info: bool = True) -> Image.Image:
    """渲染一张成图。

    source_img  范式二需要（原图）；其余范式可省
    """
    from PIL import Image as _I
    W, H = shape.width, shape.height
    blank = _I.new("RGB", (W, H), theme.paper)
    mask = _I.fromarray((shape.mask * 255).astype(np.uint8)).convert("L")

    if paradigm == "silhouette":
        layer = build_layout(shape, spec, theme, add_contour=True)
        out = _I.composite(layer, blank, mask)

    elif paradigm == "photo":
        if source_img is None:
            raise ValueError("paradigm='photo' 需要 source_img")
        base = build_edge_layout(shape, spec, theme)
        halo = _I.fromarray(
            (ndimage.binary_dilation(shape.mask, iterations=18) * 255).astype(np.uint8)
        ).convert("L")
        base = _I.composite(blank, base, halo)          # 纸色衬底让主体跳出
        subj = _I.new("RGB", (W, H), theme.paper)
        subj.paste(source_img.convert("RGB"), (0, 0))
        out = _I.composite(subj, base, mask)

    elif paradigm == "negative":
        layer = build_edge_layout(shape, spec, theme)
        inv = _I.fromarray(((~shape.mask) * 255).astype(np.uint8)).convert("L")
        out = _I.composite(layer, blank, inv)

    else:
        raise ValueError(f"未知范式: {paradigm}（可选 {PARADIGMS}）")

    return _add_info(out, spec, theme) if add_info else out


def render_all(shape: ShapeInfo, spec: LayoutSpec, theme: Theme,
               source_img: Image.Image | None = None,
               out_dir: str = ".", prefix: str = "", verbose: bool = True) -> dict:
    """一次渲染三种范式，返回 {范式: 路径}。"""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for p in PARADIGMS:
        img = render(shape, spec, theme, p, source_img)
        fname = f"{prefix}{LABELS[p]}.png" if prefix else f"{LABELS[p]}.png"
        path = os.path.join(out_dir, fname)
        img.save(path)
        paths[p] = path
        if verbose:
            print(f"  {LABELS[p]:8s} -> {path}  ({os.path.getsize(path)//1024} KB)")
    return paths
