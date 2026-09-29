"""版式层：把组件摆进形状。

两种版式（对应不同范式）：
    build_layout()       版心在形状内 —— 组件构成形状（用于「剪影填充」）
    build_edge_layout()  版式贴画布边缘 —— 避开主体（用于「原图加背景」「背景负形」）

组件尺寸由【面积比 × 形状面积】反推，不是拍脑袋定的比例。
文字一律撑满组件（字号由组件尺寸反算），这是「文字贴合组件」的落地。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .partition import RATIOS, Rect, assign_levels, quadtree
from .shape import ShapeInfo
from .texture import tile

# 字体按平台自动解析（不写死 Windows 路径 —— Linux/macOS clone 下来会崩）
from . import fonts as _fonts

# 底纹模式轮换：flat = 留白（不贴质感）
MODES_LAYOUT = ["grain", "flat", "flat", "cell", "flat", "grain", "flat",
                "flat", "flat", "cell", "flat", "flat", "grain", "flat", "flat"]
MODES_EDGE = ["grain", "cell", "flat", "grain", "cell", "grain", "flat", "cell",
              "grain", "cell", "grain", "flat", "cell", "grain", "cell"]


@dataclass
class LayoutSpec:
    """一次排版的全部参数。"""
    title: str = "TITLE"
    title_sub: str = "subtitle"
    sub: str = "SUB"
    sub_en: str = "LABEL"
    body: str = ""
    meta: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    # 留空 = 按平台自动解析（见 fonts.py）
    font_bold: str | None = None
    font_reg: str | None = None
    font_serif: str | None = None
    font_mono: str | None = None

    def __post_init__(self):
        for attr, kind in (("font_bold", "bold"), ("font_reg", "reg"),
                           ("font_serif", "serif"), ("font_mono", "mono")):
            if getattr(self, attr) is None:
                setattr(self, attr, _fonts.resolve(kind))

    title_bleed: float = 1.16      # 主标题字号倍率（>1 = 出血被裁切）
    offx_title: float = -0.045     # 主标题错位（相对版心宽）
    offx_sub: float = -0.07        # 副标题错位
    offx_body: float = 0.30        # 正文错位


class _Fonts:
    """字号 → 字体对象 缓存（逐帧场景必须缓存）。"""

    def __init__(self):
        self._c: dict = {}

    def get(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        k = (path, max(6, int(size)))
        if k not in self._c:
            self._c[k] = ImageFont.truetype(k[0], k[1])
        return self._c[k]


FONTS = _Fonts()
_PROBE = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def text_size(txt: str, path: str, size: int) -> tuple[int, int, tuple]:
    b = _PROBE.textbbox((0, 0), txt, font=FONTS.get(path, size))
    return b[2] - b[0], b[3] - b[1], b


def _wrap(draw, txt: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for ch in txt:
        if draw.textbbox((0, 0), cur + ch, font=font)[2] > max_w and cur:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def _knockout(canvas, draw, box, paper, pad=6, thresh=0.04):
    """文字衬底：只在文字区确实压到质感块时，挖一块统一底色。

    压在纸色上就不动 —— 否则会留下一个近色矩形边界。
    """
    x0, y0, x1, y1 = box
    bx0, by0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    bx1, by1 = min(canvas.width, int(x1 + pad)), min(canvas.height, int(y1 + pad))
    if bx1 <= bx0 or by1 <= by0:
        return
    reg = np.asarray(canvas.crop((bx0, by0, bx1, by1))).astype(np.int16)
    dev = np.abs(reg - np.array(paper, np.int16)).max(axis=2)
    if (dev > 8).mean() > thresh:
        draw.rectangle([bx0, by0, bx1, by1], fill=paper)


def _base_canvas(shape: ShapeInfo, rects, theme, modes, out_size=None):
    W, H = shape.width, shape.height
    c = Image.new("RGB", (W, H), theme.paper)
    for i, r in enumerate(rects):
        mode = modes[i % len(modes)]
        if r.w < 8 or r.h < 8 or mode == "flat":
            continue
        base = theme.ink if mode == "ink" else theme.tints[i % len(theme.tints)]
        c.paste(tile(r.w, r.h, base, mode, 1000 + i, theme.accent), (r.x, r.y))
    return c


def _contour(canvas: Image.Image, shape: ShapeInfo, color) -> Image.Image:
    """形状轮廓线：留白时靠它辨认形状（细线，不抢内容）。"""
    edge = shape.mask & ~ndimage.binary_erosion(shape.mask, iterations=2)
    arr = np.asarray(canvas).copy()
    arr[edge] = color
    return Image.fromarray(arr)


def build_layout(shape: ShapeInfo, spec: LayoutSpec, theme, add_contour=True):
    """版心版式：组件构成形状（用于「剪影填充」范式）。"""
    W, H = shape.width, shape.height
    rects = quadtree(shape)
    levels = assign_levels(rects, RATIOS, shape)
    canvas = _base_canvas(shape, rects, theme, MODES_LAYOUT)
    d = ImageDraw.Draw(canvas)

    # 版心：形状中部最宽处的可用宽度
    bx, by, bw, bh = shape.bbox
    best_w, best_y = 0, by + bh // 2
    for yy in range(by + int(bh * 0.22), by + int(bh * 0.58)):
        xs = np.where(shape.mask[yy])[0]
        if len(xs) >= 2 and int(xs[-1] - xs[0]) > best_w:
            best_w, best_y = int(xs[-1] - xs[0]), yy
    COL_W = int(best_w * 0.80)
    xs = np.where(shape.mask[best_y])[0]
    COL_X = int((xs[0] + xs[-1]) / 2) - COL_W // 2

    # ★ 组件尺寸由面积反推
    A = shape.area
    h_title = (RATIOS["title"] * A / sum(RATIOS.values())) / COL_W
    h_sub = (RATIOS["sub"] * A / sum(RATIOS.values())) / COL_W

    # 主标题：英文按实测宽度反推字号（中文字宽≈字号的假设对英文不成立）
    t_size = 10
    while text_size(spec.title, spec.font_bold, t_size + 1)[0] <= COL_W * 0.96:
        t_size += 1
    t_size = int(t_size * spec.title_bleed)
    t_w, t_h, _ = text_size(spec.title, spec.font_bold, t_size)
    t_x = COL_X + int(COL_W * spec.offx_title)
    t_y = best_y - int(bh * 0.11)

    pad = int(t_size * 0.14)
    d.rectangle([t_x - pad, t_y - pad, t_x + t_w + pad, t_y + t_h + pad], fill=theme.ink)
    d.text((t_x, t_y), spec.title, font=FONTS.get(spec.font_bold, t_size), fill=theme.paper)

    s2 = max(11, int(t_size * 0.11))
    ew, eh, _ = text_size(spec.title_sub, spec.font_mono, s2)
    ey = t_y + t_h + pad + 12
    d.text((t_x + t_w + pad - ew - 8, ey), spec.title_sub,
           font=FONTS.get(spec.font_mono, s2), fill=theme.accent)
    rule_y = ey + eh + 16

    # 副标题方框
    sub_w = int(COL_W * 0.56)
    sub_h = max(56, int(h_sub))
    sub_x = COL_X + int(COL_W * spec.offx_sub)
    sub_y = rule_y + int(t_size * 0.10)
    _knockout(canvas, d, (sub_x, sub_y, sub_x + sub_w, sub_y + sub_h), theme.paper, 5)
    d.rectangle([sub_x, sub_y, sub_x + sub_w, sub_y + sub_h], outline=theme.ink, width=3)
    pin = int(sub_h * 0.20)
    sc = int((sub_h - pin * 2) * 0.80)
    while text_size(spec.sub, spec.font_bold, sc)[0] > sub_w * 0.40 and sc > 10:
        sc -= 1
    cw, ch, _ = text_size(spec.sub, spec.font_bold, sc)
    d.text((sub_x + pin + 4, sub_y + (sub_h - ch) // 2), spec.sub,
           font=FONTS.get(spec.font_bold, sc), fill=theme.ink)
    sen = 10
    while text_size(spec.sub_en, spec.font_mono, sen + 1)[0] <= sub_w * 0.44:
        sen += 1
    sew, seh, _ = text_size(spec.sub_en, spec.font_mono, sen)
    d.text((sub_x + sub_w - pin - sew - 4, sub_y + (sub_h - seh) // 2), spec.sub_en,
           font=FONTS.get(spec.font_mono, sen), fill=theme.accent)

    # 正文：错位到右侧
    bs = max(12, int(W * 0.0155))
    bp = int(bs * 1.1)
    body_w = int(COL_W * 0.62)
    body_x = COL_X + int(COL_W * spec.offx_body)
    body_y = sub_y + sub_h + int(t_size * 0.22)
    if spec.body:
        lines = _wrap(_PROBE, spec.body, FONTS.get(spec.font_serif, bs), body_w)
        body_h = len(lines) * int(bs * 1.8) + bp
        _knockout(canvas, d, (body_x, body_y, body_x + body_w, body_y + body_h),
                  theme.paper, 7)
        d.line([body_x, body_y + 2, body_x + int(bs * 4), body_y + 2], fill=theme.accent, width=3)
        for k, ln in enumerate(lines):
            d.text((body_x, body_y + 10 + k * int(bs * 1.8)), ln,
                   font=FONTS.get(spec.font_serif, bs), fill=(52, 50, 48))

    # 微型标签
    for tag, (tx, ty) in zip(spec.tags,
                             [(t_x, t_y - 46), (t_x + t_w - 30, rule_y + 6),
                              (sub_x, sub_y - 30), (body_x + body_w - 40, body_y + 6),
                              (t_x + int(t_w * 0.5), t_y + t_h + pad + 6)]):
        d.text((tx, ty), tag, font=FONTS.get(spec.font_mono, 11), fill=theme.ink)

    if add_contour:
        canvas = _contour(canvas, shape, theme.ink)
    return canvas


def build_edge_layout(shape: ShapeInfo, spec: LayoutSpec, theme):
    """贴边版式：组件放画布边缘、避开主体（用于「原图加背景」「背景负形」）。"""
    W, H = shape.width, shape.height
    rects = quadtree(shape)
    canvas = _base_canvas(shape, rects, theme, MODES_EDGE)
    # 形状之外的画布也铺底纹（否则负形范式会大片空白）
    bx, by, bw, bh = shape.bbox
    for i, (rx, ry, rw, rh) in enumerate([
            (0, 0, W, max(0, by - 4)), (0, by + bh + 4, W, max(0, H - by - bh - 4)),
            (0, by, max(0, bx - 4), bh), (bx + bw + 4, by, max(0, W - bx - bw - 4), bh)]):
        if rw > 8 and rh > 8:
            canvas.paste(tile(rw, rh, theme.tints[(i + 2) % len(theme.tints)],
                              MODES_EDGE[(i + 4) % len(MODES_EDGE)], 2000 + i,
                              theme.accent), (rx, ry))
    d = ImageDraw.Draw(canvas)
    M = int(W * 0.055)

    # 主标题：底部，反白块，字号极大
    ts = max(20, int(W * 0.10))
    while text_size(spec.title, spec.font_bold, ts)[0] > W * 0.86 and ts > 12:
        ts -= 1
    tw, th, _ = text_size(spec.title, spec.font_bold, ts)
    s2 = max(11, int(ts * 0.13))
    ew, eh, _ = text_size(spec.title_sub, spec.font_mono, s2)
    block_h = 26 + th + 14 + eh + 14
    tx = M
    ty = H - M - block_h + 26
    d.rectangle([tx - 8, ty - 26, tx + tw + 8, ty + th + 14 + eh + 22], fill=theme.paper)
    d.line([tx - 6, ty - 14, tx + tw + 6, ty - 14], fill=theme.ink, width=6)
    d.text((tx, ty), spec.title, font=FONTS.get(spec.font_bold, ts), fill=theme.ink)
    d.text((tx, ty + th + 10), spec.title_sub, font=FONTS.get(spec.font_mono, s2),
           fill=theme.accent)
    d.line([tx - 6, ty + th + 10 + eh + 12, tx + tw + 6, ty + th + 10 + eh + 12],
           fill=theme.ink, width=2)

    # 副标题方框：右下
    sw, sh = int(W * 0.30), int(W * 0.072)
    sx, sy = W - M - sw, H - M - sh - 62
    d.rectangle([sx - 5, sy - 5, sx + sw + 5, sy + sh + 5], fill=theme.paper)
    d.rectangle([sx, sy, sx + sw, sy + sh], outline=theme.ink, width=3)
    pin = int(sh * 0.20)
    sc = int((sh - pin * 2) * 0.82)
    while text_size(spec.sub, spec.font_bold, sc)[0] > sw * 0.40 and sc > 10:
        sc -= 1
    cw, ch, _ = text_size(spec.sub, spec.font_bold, sc)
    d.text((sx + pin + 4, sy + (sh - ch) // 2), spec.sub,
           font=FONTS.get(spec.font_bold, sc), fill=theme.ink)
    sen = 10
    while text_size(spec.sub_en, spec.font_mono, sen + 1)[0] <= sw * 0.46:
        sen += 1
    sew, seh, _ = text_size(spec.sub_en, spec.font_mono, sen)
    d.text((sx + sw - pin - sew - 4, sy + (sh - seh) // 2), spec.sub_en,
           font=FONTS.get(spec.font_mono, sen), fill=theme.accent)

    # 正文：右上
    if spec.body:
        bs = max(12, int(W * 0.0165))
        bp = int(bs * 1.2)
        bw2 = int(W * 0.30)
        lines = _wrap(_PROBE, spec.body, FONTS.get(spec.font_serif, bs), bw2)
        bx2, by2 = W - M - bw2, M + 30
        d.rectangle([bx2 - 6, by2 - 6, W - M + 6, by2 + len(lines) * int(bs * 1.85) + bp],
                    fill=theme.paper)
        d.line([bx2, by2, bx2 + int(bs * 4), by2], fill=theme.accent, width=3)
        for k, ln in enumerate(lines):
            d.text((bx2, by2 + 10 + k * int(bs * 1.85)), ln,
                   font=FONTS.get(spec.font_serif, bs), fill=(52, 50, 48))
    return canvas
