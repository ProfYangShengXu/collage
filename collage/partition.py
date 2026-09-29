"""四叉树分区：把形状切成一组大小不一、互不重叠的矩形。

为什么是四叉树而不是均匀网格：
    均匀网格产生等大的块，视觉上呆板；四叉树递归分割天然产生大小错落的块面，
    也就是设计语言里的「不平均网格」。这是本项目取代「逐字填充」的关键一步。

面积比在矩形集合上分配（而不是在像素上），所以 3:1:1:9 这类比例能控得住。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shape import ShapeInfo

# 四层面积比（主标题 : 副标题 : 正文 : 无意义）—— 用户定的设计参数
RATIOS = {"title": 3, "sub": 1, "body": 1, "fill": 9}


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int
    fill: float          # 该矩形内的形状占比

    @property
    def area(self) -> int:
        return self.w * self.h


def quadtree(shape: ShapeInfo, min_side_ratio: float = 0.058,
             min_fill: float = 0.06, max_depth: int = 6,
             accept_range: tuple[float, float] = (0.62, 0.88),
             max_area_ratio: float = 0.10,
             seed: int = 17) -> list[Rect]:
    """递归四等分，返回矩形集合。

    min_side_ratio  最小边长 = 图宽 × 该比例（大图不会切出巨型块）
    min_fill        填充率低于此值直接丢弃
    accept_range    ★ 接受阈值随机区间 —— 让块面积拉开，避免大量等大块
    max_area_ratio  ★ 单块面积上限（占形状面积）。没有这条，简单形状
                    （圆、方块）会只切出 4 个巨型块，副标题/正文层
                    因预算装不下任何块而空掉。
    """
    m = shape.mask
    H, W = m.shape
    integ = m.cumsum(0).cumsum(1)
    max_area = shape.area * max_area_ratio

    def fill_ratio(rx, ry, rw, rh) -> float:
        x2, y2 = min(rx + rw, W), min(ry + rh, H)
        rx, ry = max(rx, 0), max(ry, 0)
        if x2 <= rx or y2 <= ry:
            return 0.0
        s = integ[y2 - 1, x2 - 1]
        if rx > 0:
            s -= integ[y2 - 1, rx - 1]
        if ry > 0:
            s -= integ[ry - 1, x2 - 1]
        if rx > 0 and ry > 0:
            s += integ[ry - 1, rx - 1]
        return float(s) / ((x2 - rx) * (y2 - ry))

    min_side = max(46, int(W * min_side_ratio))
    rng = np.random.default_rng(seed)
    out: list[Rect] = []

    def rec(rx, ry, rw, rh, d):
        f = fill_ratio(rx, ry, rw, rh)
        if f < min_fill:
            return
        too_big = rw * rh > max_area and d < max_depth
        thresh = accept_range[0] + rng.random() * (accept_range[1] - accept_range[0])
        if (f >= thresh and not too_big) or d >= max_depth or min(rw, rh) <= min_side:
            out.append(Rect(rx, ry, rw, rh, f))
            return
        hw, hh = rw // 2, rh // 2
        rec(rx,      ry,      hw,    hh,    d + 1)
        rec(rx + hw, ry,      rw - hw, hh,    d + 1)
        rec(rx,      ry + hh, hw,    rh - hh, d + 1)
        rec(rx + hw, ry + hh, rw - hw, rh - hh, d + 1)

    bx, by, bw, bh = shape.bbox
    rec(bx, by, bw, bh, 0)
    return out


def assign_levels(rects: list[Rect], ratios: dict[str, int],
                  shape: ShapeInfo, seed: int = 31) -> list[str]:
    """按面积比把矩形分配给各层级。

    主标题优先落在「厚」处（距边缘远），避免大字卡在细枝末节里。
    """
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(shape.mask)
    total = sum(r.area for r in rects)
    budget = {k: total * v / sum(ratios.values()) for k, v in ratios.items()}
    acc = {k: 0.0 for k in ratios}
    level = [None] * len(rects)

    order = sorted(range(len(rects)),
                   key=lambda i: -dist[rects[i].y + rects[i].h // 2,
                                       rects[i].x + rects[i].w // 2])
    for i in order:
        if acc["title"] >= budget["title"] * 0.85:
            break
        if rects[i].area < total * 0.03:
            continue
        if acc["title"] + rects[i].area > budget["title"] * 1.18:
            continue
        level[i] = "title"
        acc["title"] += rects[i].area

    rest = sorted([i for i in range(len(rects)) if level[i] is None],
                  key=lambda i: -rects[i].area)
    for lvl in ("sub", "body"):
        for i in rest:
            if level[i] is not None:
                continue
            if rects[i].area < total * 0.012:
                continue
            if acc[lvl] + rects[i].area > budget[lvl] * 1.30:
                continue
            level[i] = lvl
            acc[lvl] += rects[i].area
            if acc[lvl] >= budget[lvl] * 0.80:
                break

    for i in range(len(rects)):
        if level[i] is None:
            level[i] = "fill"
    return level
