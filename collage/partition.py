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
# ★ 2026-10-01：曾误改成 3:3:3:9（以为三者等权），用户更正为 **9:3:1:1**
# （无意义 : 主标题 : 副标题 : 正文）—— 主标题是副标题/正文的 3 倍，不是等权。
# 副标题与正文各占 1 份，两者同权。总 14 份。
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


def sliced_blocks(shape: ShapeInfo, n_cuts: int = 7, gap: int = 24,
                  seed: int = 17, min_side: int = 240) -> list[tuple[int, int, int, int, str]]:
    """递归二分切割 + 间隔：主块独占 + 小块填充，块间留白。返回 [(x,y,w,h,kind)]。

    ★ 为什么不用网格 / 四叉树：这两种切出来的块尺寸都太接近，排满后就是"一片方格"。
    用户要的是**构成感**：一个半调网点的大方块独占，若干小正方形/长方形填充其余，
    块与块之间要有**整体的间隔**（露出统一底色，画面才透气）——
    "两个破碎画面中间要有整体的画面间隔"。

    做法：从整张画布开始，反复挑一块（按面积加权，大块优先）沿随机方向、随机位置切一刀。
    二分切割的块面积呈指数分布，天然产出"一大 + 若干小"的层次，而不是均匀网格。
    最后每块向内收缩 gap/2，于是所有块之间都有一条统一宽度的底色缝。
    """
    W, H = shape.width, shape.height
    rng = np.random.default_rng(seed)
    blocks: list[tuple[int, int, int, int]] = [(0, 0, W, H)]

    for _ in range(n_cuts):
        cand = [i for i, b in enumerate(blocks) if min(b[2], b[3]) > min_side]
        if not cand:
            break
        # 按面积加权挑：大块更容易被继续切，于是小块能长出来
        wts = np.array([blocks[i][2] * blocks[i][3] for i in cand], dtype=float)
        i = cand[int(rng.choice(len(cand), p=wts / wts.sum()))]
        x, y, w, h = blocks.pop(i)
        t = float(rng.uniform(0.34, 0.66))
        if w >= h:
            cut = max(min_side // 2, int(w * t))
            blocks.append((x, y, cut, h))
            blocks.append((x + cut, y, w - cut, h))
        else:
            cut = max(min_side // 2, int(h * t))
            blocks.append((x, y, w, cut))
            blocks.append((x, y + cut, w, h - cut))

    # ★ 内缩 gap/2 → 块与块之间留下统一宽度的底色缝，这就是"整体间隔"
    out: list[tuple[int, int, int, int, str]] = []
    for (x, y, w, h) in blocks:
        g = gap // 2
        nw, nh = w - gap, h - gap
        if nw < 12 or nh < 12:
            continue
        out.append((x + g, y + g, nw, nh, ""))

    out.sort(key=lambda b: -(b[2] * b[3]))
    # ★ 分档要按【可见面积】排，不能按总面积 —— 主体（人物）会遮掉大块，
    # 若按总面积选主块，主调质感很可能整块落在人物背后，等于白做。
    # 实测：按总面积选时主块被主体挡掉大半，画面上只剩边缘的碎纹理。
    m = shape.mask
    def _visible(b):
        x, y, w, h = b[0], b[1], b[2], b[3]
        sub = m[max(0, y):y + h, max(0, x):x + w]
        return int((~sub).sum()) if sub.size else 0

    out.sort(key=_visible, reverse=True)
    for k, b in enumerate(out):
        if k == 0:
            out[k] = (b[0], b[1], b[2], b[3], "hero")      # 主块独占半调网点
        else:
            a = _visible(b)
            out[k] = (b[0], b[1], b[2], b[3],
                      "mid" if a > 60000 else ("small" if a > 22000 else "tiny"))
    return out


def graded_grid(shape: ShapeInfo, cols: int = 4, rows: int = 7,
                seed: int = 17, split_prob: float = 0.55,
                small_ratio: float = 0.34) -> list[tuple[int, int, int, int, str]]:
    """分级网格：大正方形为主 + 小块点缀。返回 [(x, y, w, h, kind)]。

    ★ 为什么取代四叉树：四叉树递归会切出 37px 到 476px 连续分布的块 —— 尺寸无分级，
    视觉上就是"一片均匀的方格"，缺少节奏（用户原话：「纯方格均匀排布缺少变化」）。
    真正的拼贴语言是【两档】：大块主导画面、小块做点缀，尺寸对比明确。

    做法：先把画布按 cols×rows 切成大格（接近正方形），每个大格以 split_prob 的概率
    被切出一条更小的块（长方形或正方形）。于是产出两类东西：
      big   大格本身，占绝大多数面积 —— 给半调网点这类"主调"质感
      small 切出来的小块 —— 给细胞噪声或纯色底，作为节奏点
    小块的位置沿切缝随机，所以不会排成规整的行列。
    """
    W, H = shape.width, shape.height
    cw, ch = W / cols, H / rows
    rng = np.random.default_rng(seed)
    out: list[tuple[int, int, int, int, str]] = []

    for r in range(rows):
        for c in range(cols):
            x0, y0 = int(c * cw), int(r * ch)
            x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
            w, h = x1 - x0, y1 - y0
            if rng.random() >= split_prob:
                out.append((x0, y0, w, h, "big"))
                continue
            # 沿较长的边切：先决定横切还是竖切
            if w >= h:
                cut = int(w * (1 - small_ratio * (0.6 + 0.8 * rng.random())))
                cut = max(int(w * 0.25), min(cut, int(w * 0.78)))
                out.append((x0, y0, cut, h, "big"))              # 大块（竖长方形）
                sx, sy, sw, sh = x0 + cut, y0, w - cut, h
            else:
                cut = int(h * (1 - small_ratio * (0.6 + 0.8 * rng.random())))
                cut = max(int(h * 0.25), min(cut, int(h * 0.78)))
                out.append((x0, y0, w, cut, "big"))              # 大块（横长方形）
                sx, sy, sw, sh = x0, y0 + cut, w, h - cut
            # 小块：一半概率再对半切成更方的两块，形成"小正方形"
            if rng.random() < 0.5 and min(sw, sh) > 40:
                if sw >= sh:
                    out.append((sx, sy, sw // 2, sh, "small"))
                    out.append((sx + sw // 2, sy, sw - sw // 2, sh, "small"))
                else:
                    out.append((sx, sy, sw, sh // 2, "small"))
                    out.append((sx, sy + sh // 2, sw, sh - sh // 2, "small"))
            else:
                out.append((sx, sy, sw, sh, "small"))
    return out


def coverage_partition(shape: ShapeInfo, min_side: int = 96, max_side: int = 470,
                       seed: int = 17, split_prob: float = 0.62
                       ) -> list[tuple[int, int, int, int, str]]:
    """铺满画布的【纯几何】分块：与形状无关，尺寸有变化。返回 [(x,y,w,h,kind)]。

    ★ 为什么不能复用 quadtree：quadtree 的分裂由 `fill_ratio`（形状填充率）驱动 ——
    形状复杂处细切、简单处早早停止，这是为【剪影填充】设计的。用它铺底纹会出问题：
    min_side 一旦调大，所有块都在同一深度被 `min(rw,rh) <= min_side` 截断，
    于是块尺寸完全一致（实测 0.14→全是 148×238、0.28→全是 297×476），
    视觉上就是「没有面积变化」。

    底纹要的是【铺满画布 + 尺寸有层次】，所以这里用【随机停止】驱动分裂：
    块小于 min_side 必停；块小于 max_side 时按 split_prob 随机决定停不停 ——
    同一层的块有的继续切、有的就地停下，于是大小错落。
    """
    W, H = shape.width, shape.height
    rng = np.random.default_rng(seed)
    out: list[tuple[int, int, int, int, str]] = []

    def rec(x: int, y: int, w: int, h: int):
        if w < 8 or h < 8:
            return
        if min(w, h) <= min_side:
            out.append((x, y, w, h, "small"))
            return
        if w * h <= max_side * max_side and rng.random() < split_prob:
            out.append((x, y, w, h, "mid"))
            return
        hw, hh = w // 2, h // 2
        rec(x, y, hw, hh)
        rec(x + hw, y, w - hw, hh)
        rec(x, y + hh, hw, h - hh)
        rec(x + hw, y + hh, w - hw, h - hh)

    rec(0, 0, W, H)
    return out


def quadtree(shape: ShapeInfo, min_side_ratio: float = 0.26,
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

    ★ 2026-10-01 min_side_ratio 0.058 → 0.26。原值在 1188×2112 上切出 **239 块**
    （最小 37×60），块太小导致整片底纹碎成马赛克（用户：「背景太碎了」）。参考样例
    （见 examples/tiger-shikigami.jpg）的底纹块边长约 200–400px（1080 画布），是【大块拼贴】而非碎块。
    实测本图各档：0.058→239 块(37×60)；0.10→127 块(74×119)；0.14~0.22→55 块(148×238)；
    0.28→16 块(297×476)。
    ★ 再提到 0.26：0.22 的 55 块仍被判「还是太碎」。注意——单纯放大块并不能解决"碎"，
    真正的元凶是【块间对比】（见 MODES_EDGE 与 _CELL_MIX 的注释），这里只是配合收敛。
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
