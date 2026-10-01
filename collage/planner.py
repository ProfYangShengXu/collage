"""版面决策：由【填充区域】直接推导组件位置。

为什么要有这个模块
------------------
在此之前，贴边版式的坐标全是硬编码常数（`W*0.30`、`M + 30`、`H - M - sh - 62`）。
换一张素材，这些数就全部失效 —— 表现为文字被主体的手/头发吃掉、整块组件凭空消失
（因为 photo 范式的合成是 `composite(subj, base, mask)`：剪影内用原图，落在里面的文字
直接被覆盖）。实测：同一份代码换素材后，副标题框整块消失、正文被切成 `en research in
large la…` 这样的碎片。

解决思路：不再"猜坐标再验证"，而是【直接读填充区域】算坐标。
mask 的每一行就是一条布尔序列；从设计基准线往右扫到第一个主体像素，就是这个位置
"还能放多宽"。版面决策因此退化成「在 mask 上找一条足够高、足够宽的空白竖直带」。

为什么不用装箱/最优化求解器
--------------------------
不需要。真实约束只有两个：① 组件必须完全在主体之外 ② 全部左对齐到同一条基准线。
这两条一旦写进搜索过程，剩下的只是"带够不够高"，用游程扫描 O(H) 就能判定。
引入通用求解器只会把可解释性换掉，换不来更好的版面。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Slot:
    x: int
    y: int
    w: int
    h: int


def free_width_at(mask: np.ndarray, x: int) -> np.ndarray:
    """以 x 为左界，逐行返回"向右连续空白"的宽度。

    mask  True = 主体（不可放文字）
    返回  int 数组，长度 H；某行若 x 处本身就是主体，则该行宽度为 0。
    """
    H, W = mask.shape
    sub = mask[:, x:]                       # 只看 x 右侧
    # 逐行找到第一个 True 的位置；全空白则为剩余宽度
    out = np.zeros(H, dtype=np.int32)
    for y in range(H):
        row = sub[y]
        nz = np.flatnonzero(row)            # 主体像素的列号
        out[y] = int(nz[0]) if nz.size else row.size
    return out


def band_for(mask: np.ndarray, x: int, w: int, h: int,
             y_lo: int = 0, y_hi: int | None = None) -> tuple[int, int] | None:
    """在左界 x、宽 w 的条件下，找一条高度 >= h 的空白竖直带。

    返回 (y0, y1)；找不到返回 None。多条则取最长的。
    """
    H, _ = mask.shape
    y_hi = H if y_hi is None else min(y_hi, H)
    if x + w > mask.shape[1]:
        return None
    # 该行在 [x, x+w) 内是否全空白
    ok = ~mask[:, x:x + w].any(axis=1)

    best: tuple[int, int] | None = None
    y = max(0, y_lo)
    while y < y_hi:
        if ok[y]:
            y0 = y
            while y < y_hi and ok[y]:
                y += 1
            if y - y0 >= h and (best is None or (y - y0) > (best[1] - best[0])):
                best = (y0, y)
        else:
            y += 1
    return best


def _subtract(taken: list[tuple[int, int]], y0: int, y1: int,
              gap: int) -> int | None:
    """在 [y0, y1) 内避开已占用的区间，返回可用的起始 y；放不下返回 None。

    已占用区间会各自向上下扩张 gap，保证组件之间始终有间距（先定间距，再排内容）。
    """
    pads = sorted((a - gap, b + gap) for a, b in taken)
    cand = y0
    for pa, pb in pads:
        if pb <= cand:
            continue
        if pa >= y1:
            break
        cand = pb                      # 被占了，跳到该占用区间的下方
        if cand >= y1:
            return None
    return cand


def _overlaps_taken(taken: list[tuple[int, int]], y0: int, y1: int, gap: int) -> bool:
    """新窗口 [y0, y1) 是否与已占用区间冲突（各自外扩 gap 当间距）。"""
    for a, b in taken:
        if y0 < b + gap and a - gap < y1:
            return True
    return False


def plan_column(mask: np.ndarray, comps: list[tuple[str, int, int]],
                x: int, gap: int, w_max: int | None = None,
                y_lo: int = 0, y_hi: int | None = None,
                allow_shrink: bool = True,
                max_overlap: float = 0.0) -> dict[str, Slot]:
    """把一组组件排进「以 x 为左界的同一栏」，全部左对齐。

    comps        [(name, w, h), ...]  按优先级降序（先放的优先拿到空间）
    gap          组件之间的垂直间距
    w_max        栏宽上限（组件宽度会被 clamp 到它以内）
    allow_shrink 是否允许【逐组件】收窄宽度。
                 ★ 默认 True 会让不同组件拿到不同宽度 —— 左边界齐、右边界不齐，
                 视觉上就是"正文比副标题框宽出一截"。要栏内整齐，调用方应传 False，
                 并自己在外面把栏宽降到所有组件都能用的那一档（见 layout.build_edge_layout）。
    max_overlap  组件与主体的重叠占比上限。0 = 必须完全落在空白里；
                 用户许可「主标题只能 20% 的部分盖住原画」→ 传 0.20。

    返回  {name: Slot}；放不下的组件不会出现在结果里（调用方降级处理）。
    """
    H, W = mask.shape
    y_hi = H if y_hi is None else min(y_hi, H)
    out: dict[str, Slot] = {}
    taken: list[tuple[int, int]] = []
    max_w = min(w_max, W - x) if w_max else W - x
    shrinks = (1.0, 0.85, 0.70, 0.55, 0.45) if allow_shrink else (1.0,)
    integ = _integral(mask)
    # ★ 游标：组件按 comps 顺序依次排，每个都从【上一个的下方】开始找。
    # 这是"阅读顺序"（主标题 → 副标题 → 正文）的机械保证 —— 否则后一个组件可能
    # 在 y 更小的位置找到空位，爬到前一个上面去（用户明确要求过「正文不能在标题上面」）。
    cursor = max(0, y_lo)

    for name, w, h in comps:
        w = int(min(w, max_w))
        if w <= 0 or h <= 0:
            continue

        # ★ 2026-10-01 重写：原来靠 band_for 找「整条带全空白」再避开已占用区间，
        # 那条路径有两个坑 —— ① 换素材后空白不足时组件直接消失；② 找下一条带时用当前带
        # 的 y1 作上界，等于把自己关死在已被占满的带里。
        # 现在统一成「自上而下逐 y 试放」：每次判定两件事 —— 与已占用区间不冲突（间距），
        # 与主体的重叠不超过 max_overlap。逻辑单一，降级链（收窄宽度）在外层。
        placed = False
        for shrink in shrinks:
            w2 = int(w * shrink)
            if w2 < 40 or x + w2 > W:
                continue
            for y in range(cursor, max(0, y_hi - h) + 1):
                if _overlaps_taken(taken, y, y + h, gap):
                    continue
                if overlap_ratio(mask, x, y, w2, h, integ) <= max_overlap:
                    out[name] = Slot(x, y, w2, h)
                    taken.append((y, y + h))
                    cursor = y + h + gap
                    placed = True
                    break
            if placed:
                break
    return out


def _integral(mask: np.ndarray) -> np.ndarray:
    """mask 的积分图（带 1px 零边），用于 O(1) 查询任意矩形的重叠量。"""
    return np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def overlap_ratio(mask: np.ndarray, x: int, y: int, w: int, h: int,
                  integ: np.ndarray | None = None) -> float:
    """矩形与【主体】的重叠面积占比。0 = 完全落在空白里。"""
    H, W = mask.shape
    x2, y2 = min(x + w, W), min(y + h, H)
    x, y = max(x, 0), max(y, 0)
    if x2 <= x or y2 <= y:
        return 0.0
    if integ is None:
        integ = _integral(mask)
    s = integ[y2, x2] - integ[y, x2] - integ[y2, x] + integ[y, x]
    return float(s) / float((x2 - x) * (y2 - y))


def best_band(mask: np.ndarray, x: int, w: int, h: int,
              y_lo: int = 0, y_hi: int | None = None,
              max_overlap: float = 0.0) -> tuple[int, float] | None:
    """在 [y_lo, y_hi) 内找高 h 的窗口，使与主体的重叠占比 <= max_overlap。

    按 y 递增扫描，返回**最靠上**的合格位置（符合自上而下的阅读顺序）。
    返回 (y, 实际重叠比)；找不到返回 None。

    ★ 为什么要有 max_overlap：原本要求组件 100% 落在空白里（max_overlap=0）。
    换素材后形状覆盖从 47% 涨到 71%，空白不足 → 组件【静默消失】。
    用户给的许可（原话）：「主标题只能 20% 的部分盖住原画」——
    把 0.20 写进约束，空白紧张时组件仍有位置可放，而不是被丢掉。
    """
    H, W = mask.shape
    y_hi = H if y_hi is None else min(y_hi, H)
    if x + w > W:
        return None
    integ = _integral(mask)
    lo, hi = max(0, y_lo), max(0, y_hi - h)
    for y in range(lo, hi + 1):
        r = overlap_ratio(mask, x, y, w, h, integ)
        if r <= max_overlap:
            return y, r
    return None


def free_ratio(mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """任意矩形的空白占比（积分图 O(1)）。给测试和诊断用。"""
    H, W = mask.shape
    x2, y2 = min(x + w, W), min(y + h, H)
    x, y = max(x, 0), max(y, 0)
    if x2 <= x or y2 <= y:
        return 0.0
    free = ~mask
    integ = free.cumsum(0).cumsum(1)
    s = integ[y2 - 1, x2 - 1]
    if x > 0:
        s -= integ[y2 - 1, x - 1]
    if y > 0:
        s -= integ[y - 1, x2 - 1]
    if x > 0 and y > 0:
        s += integ[y - 1, x - 1]
    return float(s) / ((x2 - x) * (y2 - y))
