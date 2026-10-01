"""版式层：把组件摆进形状。

两种版式（对应不同范式）：
    build_layout()       版心在形状内 —— 组件构成形状（用于「剪影填充」）
    build_edge_layout()  版式贴画布边缘 —— 避开主体（用于「原图加背景」「背景负形」）

组件尺寸由【面积比 × 形状面积】反推，不是拍脑袋定的比例。
文字一律撑满组件（字号由组件尺寸反算），这是「文字贴合组件」的落地。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import hashlib
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .partition import (RATIOS, Rect, assign_levels, coverage_partition,
                        graded_grid, quadtree, sliced_blocks)
from .planner import _integral, overlap_ratio, plan_column
from .shape import ShapeInfo
from .texture import tile

# 字体按平台自动解析（不写死 Windows 路径 —— Linux/macOS clone 下来会崩）
from . import fonts as _fonts

# 底纹模式轮换：flat = 留白（不贴质感）
# ★ 2026-10-01 定稿。贴边区（MODES_EDGE）的演变，两个极端都试过：
#   · 逐块轮换 cell 占 40% → 块小的时候碎成马赛克
#   · 只留 grain、几乎不放 cell → 底纹变成均匀砂点，认不出质感
#   正解：块够大（见 quadtree.min_side_ratio）时多种质感交替反而是参考样例的效果。
#   用户要求「找点别的无意义质感组件」→ 引入印刷语汇的四种（见 texture.tile 的 mode 表）：
#     halftone 半调网点 / scanline 扫描线 / paper 纸纤维 / riso 套印错位
#   全部走 _CELL_MIX 同一个浓度上限，保证「底纹是背景、不抢主体」这条约束不被破坏。
#   版心区（MODES_LAYOUT）仍保留 flat 作为留白节奏，那是「剪影填充」范式的设计语言。
MODES_LAYOUT = ["grain", "flat", "flat", "cell", "flat", "grain", "flat",
                "flat", "flat", "cell", "flat", "flat", "grain", "flat", "flat"]
MODES_EDGE = ["paper", "halftone", "grain", "paper", "scanline", "grain", "paper",
              "halftone", "grain", "paper", "riso", "grain", "paper", "cell", "grain"]


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
    # ★ 主标题块的左上角 y（由 build_edge_layout 规划后写入，draw_title_block 读取）。
    # 主标题必须【画在合成之后】（否则被原图吃掉），但位置又必须和其他组件一起统一规划
    # （否则会出现「正文在标题上面」这种层级倒置）。所以位置在这里过一手：layout 规划、
    # render 落笔。None = 沿用默认的底部位置。
    title_y: int | None = None
    # ★ 标题块反白：True = 黑底白字，False = 白底黑字。
    # 用户看过两种并置后定的：**要白底黑字**（原话「黑底白字换回白底黑字」）。
    # 反白那条路留着不改 —— 换素材时深色底纹多的话还要用。
    title_invert: bool = False
    # ★ 已排好的文字组件矩形 [(x, y, w, h, name), ...]，由 build_edge_layout 写入。
    # 极小信息层（draw_micro_info）读它来"紧贴"排布 —— 用户要求小字贴着文字框走，
    # 而不是散落在固定锚点上（散落版本变成 8 个孤立小块，太碎）。
    micro_boxes: list = field(default_factory=list)

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
    """按像素宽度换行。★ 英文优先在空格处断，不切断单词；中文无空格，自动退回逐字符。

    逐字符硬断对中文正确（字与字之间可断），对英文是错的：实测把
    "language models" 断成 "mode|ls."、"reinforcement" 断成 "reinforcemen|t"。
    先按空格切词，放不下才换行；单个词本身就超宽时（超长 URL、长单词）才退回硬断。
    """
    def w_of(s: str) -> int:
        return draw.textbbox((0, 0), s, font=font)[2]

    lines: list[str] = []
    cur = ""
    for word in txt.split(" "):
        if not word:
            cur = cur or " "
            continue
        probe = (cur + " " + word) if cur else word
        if w_of(probe) <= max_w:
            cur = probe
            continue
        if cur:
            lines.append(cur)
            cur = ""
        if w_of(word) <= max_w:
            cur = word
            continue
        # 单个词超宽：退回逐字符硬断，避免整词溢出画布
        piece = ""
        for ch in word:
            if w_of(piece + ch) > max_w and piece:
                lines.append(piece)
                piece = ch
            else:
                piece += ch
        cur = piece
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


def _mode_for_rect(r, W: int, H: int, modes) -> str:
    """按块所在【区域】分配质感，而不是按块序号轮流。

    ★ 用户同时要「找点别的质感组件」和「不要碎」，这两个诉求天然冲突 ——
    逐块轮换多种质感，本身就是"处处在变化"。解法是让同一种质感占住一大片：
    把画布划成 3×3 区域，同一区域内的所有块用同一种质感，于是质感成片、边界只在
    区域之间出现（3×3 = 最多 9 次变化，而不是 55 块 55 次）。
    """
    zones = (("paper", "halftone", "grain"),        # 上：纸 / 网点 / 磨砂
             ("scanline", "paper", "halftone"),     # 中：扫描线 / 纸 / 网点
             ("grain", "riso", "cell"))             # 下：磨砂 / 套印 / 细胞
    cx = min(0.999, (r.x + r.w / 2) / max(1, W))
    cy = min(0.999, (r.y + r.h / 2) / max(1, H))
    return zones[int(cy * 3)][int(cx * 3)]


def assign_texture_by_area(rects, hero: str = "cell",
                           hero_ratio: float = 0.45,
                           blank_ratio: float = 0.45,
                           others=("halftone",)):
    """按【面积】分配质感与留白。

    ★ 用户定的配比（原话）：「9311 之外加一个 9 的留白部分」+「纹理限制在两种」—— 即
        留白 9 : 底纹 9 : 主标题 3 : 副标题 1 : 正文 1   （总 23）
    底纹与留白各占一半，画面才有呼吸；**全部画面只用两种纹理**：
    半调网点（主调，占底纹的 90%）+ 细胞噪声（点缀）。
    质感种类一多就碎 —— 这是用户反复确认过的结论，不要再往 others 里加东西。

    所以这里：
        hero_ratio   面积 → 主调质感（半调网点）
        blank_ratio  面积 → 留白（flat = 不贴纹理，直接露出纸色）
        剩余面积      → 第二种纹理点缀（cell）

    实现要点是**按面积累加**而不是按块数 —— 块的面积差异极大，按块数分配会让小碎块
    吃掉大量配额，主调反而不占主导。做法：块按面积降序，从头累加。
    因此大块落在底纹档、中块落在留白档，两者都成片，而不是散点。
    """
    n = len(rects)
    if n == 0:
        return []
    areas = []
    for r in rects:
        a = r.area if hasattr(r, "area") else (r[2] * r[3])
        areas.append(a)
    total = float(sum(areas)) or 1.0

    modes = ["flat"] * n
    order = sorted(range(n), key=lambda i: -areas[i])
    acc = 0.0
    k = 0
    t_hero = total * hero_ratio
    t_blank = total * (hero_ratio + blank_ratio)
    for idx in order:
        if acc < t_hero:
            modes[idx] = hero                      # 底纹（主调）
        elif acc < t_blank:
            modes[idx] = "flat"                    # 留白（露出纸色）
        else:
            modes[idx] = others[k % len(others)]   # 其他质感点缀
            k += 1
        acc += areas[idx]
    return modes


_STOP = frozenset((
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "with", "on",
    "that", "is", "are", "be", "by", "as", "at", "it", "its", "from", "so",
    "this", "these", "those", "was", "were", "not", "but", "than", "then",
))


def micro_tokens(spec, n: int = 6) -> list[str]:
    """从 spec 派生【极小信息字】。

    ★ 一个字符串都不是手写的 —— 全部由代码从现有文案派生。换素材、换文案、
    换语言都自动生成，不需要人配内容。三个派生源：
      ① `title` 派生的【稳定编号】：md5 取模。同一输入永远同一个号（可复现），
         不同标题号不同 —— 看起来像"编号/参考号"，实际是确定性哈希。
      ② `title_sub` 的首字母缩写："OPEN-WEIGHT REASONING MODELS" → "OWRM"。
      ③ `body` 里的长词：去停用词、去重、按长度降序取前几个 → 大写。
         长词通常是实词（models / architectures / reinforcement），
         短词多为 the/of/and 这类虚词，按长度筛比维护一份"关键词表"更稳。
    """
    out: list[str] = []
    t = (spec.title or "").upper()
    if t:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
        out.append("REF-%03d" % (h % 997))
        out.append("NO.%02d" % (h % 97))
    sub = [w for w in (spec.title_sub or "").replace("-", " ").split() if w]
    if sub:
        out.append("".join(w[0].upper() for w in sub)[:6])
    words = [w.strip(".,;:()\u2014").upper() for w in (spec.body or "").split()]
    words = [w for w in words if w and w.lower() not in _STOP and len(w) > 4]
    seen: set[str] = set()
    for w in sorted(words, key=lambda x: -len(x)):
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= n:
            break
    return out[:n]


def _place_one_rect(mask, target: float, seed: int, extra=None, blocked=None,
                    far_from=None, min_sep: float = 0.0):
    """放【一个】面积为 target 的纯色矩形：宽高比随机、与 blocked 零重叠。

    mask     主体掩码（只用来算"贴着主体"的距离基准）
    blocked  禁止占用的区域（主体 + 已排好的文字框 + 先前放的色块）。
             与 mask 分开传，是因为"要避让的"和"要靠近的"不是同一批东西 ——
             避开文字框，但仍然要贴着主体站。
    far_from 已放色块的中心点列表；配合 min_sep 强制块与块之间拉开距离。
             ★ 用户原话：「如果要多个块的话需要有间隔距离，大概 1/2 画布」。
    """
    H, W = mask.shape          # ★ numpy 的 shape 是 (行, 列) = (H, W)，别写反
    blk = mask if blocked is None else blocked
    rng = np.random.default_rng(seed)
    integ = _integral(blk)

    ys, xs = np.where(mask)
    if len(xs):
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    else:
        bx0, by0, bx1, by1 = 0, 0, W - 1, H - 1

    # ★ 宽高比档：接近正方形 → 1:4 为止。
    # 原先一路给到 0.10，方形档全放不下时算法会用最细长的比例硬凑面积 ——
    # 实测蓝发那张凑出 118×1181 的一条线（面积对，但已经不像"块"了）。
    # 下限收到 0.25：宁可少占点面积/少放一块，也要保持"块"的形态。
    ratios = [1.0, 0.85, 0.70, 0.55, 0.42, 0.32, 0.25]
    ratios = ratios[int(rng.integers(0, 2)):]

    for ar in ratios:
        w = int(round((target * ar) ** 0.5))
        h = int(round((target / ar) ** 0.5))
        if w < 24 or h < 24 or w > W or h > H:
            continue
        step = max(10, min(w, h) // 10)
        best = None
        for x in range(0, W - w + 1, step):
            for y in range(0, H - h + 1, step):
                if extra is not None and extra[y:y + h, x:x + w].any():
                    continue
                if overlap_ratio(blk, x, y, w, h, integ) > 0.0:
                    continue
                ccx, ccy = x + w / 2.0, y + h / 2.0
                # 块间强制间隔
                if far_from and min_sep > 0:
                    dmin = min(((ccx - px) ** 2 + (ccy - py) ** 2) ** 0.5
                               for px, py in far_from)
                    if dmin < min_sep:
                        continue
                dx = max(bx0 - ccx, 0.0, ccx - bx1)
                dy = max(by0 - ccy, 0.0, ccy - by1)
                d_subj = (dx * dx + dy * dy) ** 0.5
                if best is None or d_subj < best[0]:
                    best = (d_subj, x, y, w, h)
        if best is not None:
            return best[1], best[2], best[3], best[4]
    return None


def _free_regions(blocked, min_area: int):
    """把空白区按【连通域】切开，返回 [(面积, 掩码), ...] 按面积降序。

    ★ 为什么要按连通域分：主体把画面切成了几块互不相连的空白（左上、右下、右中）。
    色块如果只在"离主体最近的候选点"里挑，会被全部吸到同一块区域里堆成一列
    （实测蓝发那张三个橙块全挤在右缘）。先切区域、再一块一个区域，才是真正的"分散"。
    """
    free = ~blocked
    lab, n = ndimage.label(free)
    if n == 0:
        return []
    out = []
    for i in range(1, n + 1):
        m = (lab == i)
        a = int(m.sum())
        if a >= min_area:
            out.append((a, m))
    out.sort(key=lambda t: -t[0])
    return out


def comp_blocks(shape: ShapeInfo, ratio: float = 1.0 / 18.0, seed: int = 17,
                max_parts: int = 4, avoid=None,
                min_sep_ratio: float = 0.5) -> list[tuple[int, int, int, int]]:
    """高饱和互补色【纯色块】：一块一个空白区，块间强制拉开距离。

    ★ 用户原话：「无意义组件里加一个和采样色的高饱和互补色纯色组件，形状随机，
    位置靠近主体不覆盖」+「互补色块如果很多的话分散一点」+
    「互补色块太大了，逻辑换成固定占画面18分之一」+
    「如果要多个块的话需要有间隔距离，大概 1/2 画布」。
    **目标面积 = W*H/18（固定）**；**块与块的中心距 >= 0.5 × 画布短边**。

    ★ 算法：把空白区（主体与文字框之外）按【连通域】切开 →
    逐块放置，优先用还没放过的区域，同优先级下先大区域。
    区域数只当优先级、不当门槛（一个大区域可以容纳多块）。
    块间距离用 min_sep 硬约束 —— 这是"分散"的机械保证，比"优先未用区域"更硬。
    放不满就减少块数，总面积随之分摊。
    """
    W, H = shape.width, shape.height
    target = (W * H) * ratio
    # "大概 1/2 画布"取画布【短边】的一半做中心距下限。
    # 用短边而不是长边：长边的一半（1056）在 4 块时会直接无解。
    min_sep = min_sep_ratio * min(W, H)
    blk = shape.mask.copy()
    if avoid is not None:
        blk = blk | avoid
    best_effort: list[tuple[int, int, int, int]] = []

    for n in range(1, max_parts + 1):
        per = target / n
        regions = _free_regions(blk, int(per * 0.45))
        if not regions:
            continue
        rects: list[tuple[int, int, int, int]] = []
        occ = np.zeros_like(shape.mask)
        centers: list[tuple[float, float]] = []
        used: dict[int, int] = {}          # 区域被用了几次
        for i in range(n):
            # 优先用还没放过的区域 → 配合 min_sep 一起保证分散；同优先级下先大区域
            order = sorted(range(len(regions)),
                           key=lambda ri: (used.get(ri, 0), -regions[ri][0]))
            for ri in order:
                blk_i = blk | ~regions[ri][1] | occ
                r = _place_one_rect(shape.mask, per, seed + i * 131, blocked=blk_i,
                                    far_from=centers, min_sep=min_sep)
                if r is None:
                    continue
                x, y, w, h = r
                occ[y:y + h, x:x + w] = True
                centers.append((x + w / 2.0, y + h / 2.0))
                used[ri] = used.get(ri, 0) + 1
                rects.append(r)
                break
            else:
                break                       # 所有区域都放不下这一块，放弃这档
        if len(rects) == n:
            return rects
        if len(rects) > len(best_effort):
            best_effort = rects

    return best_effort


def _vt_line(d, x, y, text, font, fill, lh, gap=4):
    """竖排一行：逐字符向下画。每个字符各自居中到同一竖轴。"""
    for i, ch in enumerate(text):
        b = d.textbbox((0, 0), ch, font=font)
        d.text((x - (b[2] - b[0]) // 2, y + i * lh), ch, font=font, fill=fill)
    return y + len(text) * lh + gap


def _micro_attached(d, integ, mask, box, tokens, font, theme, W, H, gap=10):
    """把一个 token 组【紧贴】在文字框外侧。横排、竖排都试，取第一个放得下的。

    ★ 用户原话：「极小信息层紧贴文字框，可以横可以竖」。
    原先的实现是 8 个固定锚点撒在四边 —— 结果变成 8 个孤立小方块，画面"到处都是文字块"
    （违反"不要有太碎的变化"）。改成贴着组件走之后：小字天然成组、跟着组件移动、
    数量也降到每框一组。

    候选顺序（先横后竖，先右后下）：右侧横排 → 下方横排 → 左侧竖排 → 上方竖排。
    每个候选都要求与主体【零重叠】，并且与已画的小字不打架。
    """
    bx, by, bw, bh = box
    txt_h = "  ".join(tokens)
    txt_v = " ".join(tokens)

    cands = []
    # ① 右侧 · 横排
    bb = d.textbbox((0, 0), txt_h, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    cands.append(("h", bx + bw + gap, by, tw, th))
    # ② 下方 · 横排
    cands.append(("h", bx, by + bh + gap, tw, th))
    # ③ 左侧 · 竖排
    lh = int(th * 1.25)
    vw = max((d.textbbox((0, 0), c, font=font)[2] for c in txt_v), default=6)
    vh = len(txt_v) * lh
    cands.append(("v", bx - gap - vw, by, vw, vh))
    # ④ 上方 · 竖排
    cands.append(("v", bx + bw - vw, by - gap - vh, vw, vh))

    for kind, x, y, w, h in cands:
        if x < 4 or y < 4 or x + w > W - 4 or y + h > H - 4:
            continue
        if overlap_ratio(mask, x, y, w, h, integ) > 0.0:
            continue
        if kind == "v":
            _vt_line(d, x + w // 2, y, txt_v, font, theme.accent, lh)
        else:
            d.rectangle([x - 3, y - 3, x + w + 3, y + h + 3], fill=theme.paper)
            d.text((x, y), txt_h, font=font, fill=theme.accent)
        return True
    return False


def draw_micro_info(canvas, shape: ShapeInfo, spec, theme, size: int = 9):
    """极小信息层：紧贴每个文字框的外侧，横排或竖排自适应。

    ★ 内容 100% 由 `micro_tokens()` 从文案派生，没有手写字符串。
    ★ 位置不再是固定锚点，而是贴着 `spec.micro_boxes`（已排好的文字组件）走 ——
    "紧贴文字框，可以横可以竖"（用户原话）。这样小字成了组件的附属标注：
    组件移到哪它跟到哪，且天然成组，不会散成一地碎块。
    """
    d = ImageDraw.Draw(canvas)
    W, H = canvas.size
    integ = _integral(shape.mask)
    f = FONTS.get(spec.font_mono, size)
    tokens = micro_tokens(spec, n=8)
    if not tokens:
        return canvas

    ri = 0
    for box in (spec.micro_boxes or []):
        bx, by, bw, bh, _name = box
        chunk = tokens[ri:ri + 2]
        ri += 2
        if not chunk:
            break
        _micro_attached(d, integ, shape.mask, (bx, by, bw, bh), chunk, f, theme, W, H)
    return canvas


def _base_canvas(shape: ShapeInfo, rects, theme, modes=None, out_size=None):
    """底纹层：四叉树分块 + 质感按面积主次分配 + 块间留缝。

    ★ 2026-10-01 定稿。这块走过的路，都记在这里免得再走回去：
      · 四叉树 + 极小块（min_side_ratio=0.058 → 239 块 37×60）→ 碎成马赛克
      · 取消分块、整片连续纹理 → 层次全失
      · 四叉树 + 逐块轮换质感 → 「纯方格均匀排布缺少变化」
      · 分级网格 / 二分切割 → 「更丑了」（块紧贴拼满，没有间隔）
    用户最终定的方案：**还是四叉树**，但质感要【按面积分主次】——
    一个质感（半调网点）占绝对主导，与其他质感 9:1。见 `assign_texture_by_area`。
    """
    W, H = shape.width, shape.height
    # ★ 留白色必须与文字块衬底【同一色值】。原先这里用 theme.tints[0]（明度 0.945），
    # 而 draw_title_block / 副标题框 / 正文块的衬底用 theme.paper（明度 0.975）——
    # 两个都是"白"但不是同一个白，并置时能看出色差（用户：「留白颜色和文本框里的留白颜色统一」）。
    c = Image.new("RGB", (W, H), theme.paper)
    tier = assign_texture_by_area(rects) if modes is None else modes
    for i, r in enumerate(rects):
        if hasattr(r, "w"):
            x, y, w, h = r.x, r.y, r.w, r.h
        else:
            x, y, w, h = r[0], r[1], r[2], r[3]
        if w < 8 or h < 8:
            continue
        mode = tier[i] if i < len(tier) else "flat"
        if mode == "flat":
            continue                      # 纯色 = 直接露出底层 tints[0]
        base = theme.ink if mode == "ink" else theme.tints[i % len(theme.tints)]
        c.paste(tile(w, h, base, mode, 1000 + i, theme.accent), (x, y))
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


_PAD = 8          # ★ 所有文本组件衬底的统一内边距。
# 对齐不是"各自的数字差不多"就行 —— 原先主标题块衬底左伸 8px、副标题框 5px、正文块 6px，
# 三者的**文字**左缘虽然都是 M，但**衬底边框**差到 3px，并置时一眼看出没对齐。
# 三个组件共用这一个常量，改一处全对齐。


def title_block_size(W: int, spec: LayoutSpec):
    """主标题块的度量：返回 (ts, tw, t_bottom, s2, eh, block_h)。

    ★ 抽出来共用：`build_edge_layout` 要用它把主标题放进统一规划，`draw_title_block`
    要用它落笔。两处各算一套必然漂移（改了一边忘了另一边）。
    """
    ts = max(20, int(W * 0.10))
    while text_size(spec.title, spec.font_bold, ts)[0] > W * 0.86 and ts > 12:
        ts -= 1
    tw, th, tb = text_size(spec.title, spec.font_bold, ts)
    # ★ `text_size` returns (b[2]-b[0], b[3]-b[1]) — the NET glyph height, which excludes the
    # ascender gap above it. But `d.text((x, y))` anchors at the TOP of that box, so the title
    # actually occupies down to `y + b[3]`. Advancing the next element by `th` therefore overlaps
    # it by (b[3] - th) every time, and the error grows with the font size: measured on a 171px
    # title, b[3]=183 vs th=133 — the subtitle collided with the title by 40px (25px at 118px).
    # Use the bottom edge for stacking.
    t_bottom = tb[3]
    s2 = max(11, int(ts * 0.13))
    ew, eh, _ = text_size(spec.title_sub, spec.font_mono, s2)
    block_h = 26 + t_bottom + 14 + eh + 14
    return ts, tw, t_bottom, s2, eh, block_h


def draw_title_block(canvas, spec: LayoutSpec, theme):
    """底部主标题块：极字号大字 + 小字 + 反白衬底 + 上下横线。

    ★ 为什么要单独抽出来：这一段原先写在 `build_edge_layout` 里，而「原图加背景」范式的合成是
    `composite(subj, base, mask)` —— 剪影【内】用原图、剪影【外】才用 base。标题块画在 base 上，
    一旦落进剪影就被原图整个吃掉（实测主标题被左下角的道具遮掉下半截）。抽出来由 `render()` 在
    合成【之后】调用，标题就压在最上层 —— 用户明确要求「主标题可以盖住一部分画没关系」。

    ★ 位置由 `spec.title_y` 决定（build_edge_layout 规划时写入），None 时回落到画布底部。
    """
    d = ImageDraw.Draw(canvas)
    W, H = canvas.size
    M = int(W * 0.055)
    ts, tw, t_bottom, s2, eh, block_h = title_block_size(W, spec)

    tx = M
    ty = spec.title_y if spec.title_y is not None else (H - M - block_h + 26)
    # ★ 反白：黑底白字 / 白底黑字二选一。用户要求两种都要有 —— 并置才有对比。
    bg, fg = (theme.ink, theme.paper) if spec.title_invert else (theme.paper, theme.ink)
    d.rectangle([tx - _PAD, ty - 26, tx + tw + _PAD, ty + t_bottom + 14 + eh + 22], fill=bg)
    d.line([tx - _PAD + 2, ty - 14, tx + tw + _PAD - 2, ty - 14], fill=fg, width=6)
    d.text((tx, ty), spec.title, font=FONTS.get(spec.font_bold, ts), fill=fg)
    d.text((tx, ty + t_bottom + 10), spec.title_sub, font=FONTS.get(spec.font_mono, s2),
           fill=theme.accent)
    d.line([tx - _PAD + 2, ty + t_bottom + 10 + eh + 12,
            tx + tw + _PAD - 2, ty + t_bottom + 10 + eh + 12], fill=fg, width=2)
    return canvas


def build_edge_layout(shape: ShapeInfo, spec: LayoutSpec, theme):
    """贴边版式：组件放画布边缘、避开主体（用于「原图加背景」「背景负形」）。

    ★ 主标题【不】在这里画 —— 见 `draw_title_block`。其余组件（副标题框、正文）仍在此绘制，
    它们被原图遮住属于该范式的风格。
    """
    W, H = shape.width, shape.height
    # ★ 底纹用【纯几何分块】（coverage_partition），不用 quadtree ——
    # quadtree 的分裂由形状填充率驱动，尺寸会趋同（实测 0.28 档全是 297×476，
    # 用户问「为什么这个四叉树没有面积变化」）。底纹要铺满画布 + 尺寸有层次，
    # 得用随机停止驱动的分裂，与形状无关。
    # 质感仍按面积分主次：半调网点占 90%（绝对主导），其余质感分剩下 10%。
    rects = coverage_partition(shape)
    canvas = _base_canvas(shape, rects, theme)
    # ★ 形状外不再单独铺条。`_base_canvas` 现在整片打底、已覆盖全画布；原先这里再铺四个边条，
    # 用的是与四叉树块不同的基色和模式，等于在同一片背景上又叠一层不一样的纹理，接缝全露。
    # （整片打底对「背景负形」范式同样成立，不会出现大片空白。）
    d = ImageDraw.Draw(canvas)
    M = int(W * 0.055)
    # ★ 互补色块不在这里画 —— 必须先排完文字，才能知道要避开哪些区域。
    # 见本函数末尾（plan 完成后）：色块避让【主体 + 文字框】两样东西。
    # 原先在这里直接画，结果色块只避让主体、不知道文字排到哪，
    # 实测奶龙那张蓝色块和标题栏/正文块挤在同一条左栏里。
    # ★★★ 2026-10-01 版面重构：从「各自贴角」改为「共享一条左基准线」。
    # 依据用户给的三条排版标准（对齐/层级/间距）：
    #   · 对齐 —— 对齐的作用不是整齐，而是让元素之间建立关系。原先四个组件各自贴各自的角
    #     （副标题贴右、正文贴右、主标题贴左），彼此没有任何共享边线，画面"散"。
    #     现在正文、副标题框、主标题全部左对齐到 x = M，只有右侧留给主体和出血大字。
    #   · 层级 —— 第一级必须唯一。原先左上角（_add_info）和底部（draw_title_block）各有一套
    #     标题且字号差 3 倍，等于两个第一级。现由 render() 关掉左上那套。
    #   · 间距 —— 先定间距再排内容，而不是"元素各占一块、剩下的是缝"。
    COL_X = M
    # 栏宽是【上限】。★ 注意主标题块本身可能比栏宽还宽（它的字号由画布宽反推），
    # 所以 w_max 要取两者较大值，否则主标题会被栏宽硬压窄。
    ts_, tw_, tb_, s2_, eh_, block_h_ = title_block_size(W, spec)
    TITLE_W = tw_ + 16
    COL_W = max(int(W * 0.46), TITLE_W)

    # ★★★ 2026-10-01 版面重构（二）—— 按用户两条要求：
    #   ① 「正文不能在标题上面」：原先正文在顶、主标题在底，层级倒置。
    #      改为自上而下的阅读顺序：主标题 → 副标题 → 正文。
    #   ② 「优化算法使其能适应不同比例的各个组件」：原先所有组件被强制同宽（0.46W），
    #      但主标题块天然更宽、正文天然更矮。改为各组件【按自身内容定宽】，只统一左边界；
    #      右边界不齐正是参考样例（MAGMA 海报）的做法 —— 共享一条左边线，字宽各异。
    # 物理约束：本图左栏 y 400–1600 被主体占满（空白 0–28%），无法容纳连续竖排，
    # 所以采用「顶部信息区（标题+副标题） + 底部信息区（正文）」两段式，顺序仍然正确。
    # ★★★ 2026-10-01：分区限制放开 + 重叠约束落地。
    # 用户两条要求：
    #   ① 「主标题只能 20% 的部分盖住原画」→ MAX_OVERLAP = 0.20（原来是 0，必须全空白）
    #   ② 换素材后组件【静默消失】（47% 覆盖 → 71% 覆盖时副标题框和正文都没画出来）
    # 原先把副标题限死在顶部 19%、正文限死在底部 14% —— 那是为素材 A 调的常数，
    # 换素材就失效。现在改为【全画布搜索 + 优先级排序】：顺序仍是主标题 → 副标题 → 正文，
    # 由 plan_column 的游标机械保证（后一个只在前一个下方找），不再靠分区常数。
    MAX_OVERLAP = 0.20
    sub_h = max(56, int(W * 0.062))
    gap = max(10, int(H * 0.012))

    # ★ 组件目标面积按 RATIOS 9:3:1:1（无意义:主标题:副标题:正文）推导：
    # 副标题与正文各占主标题的 1/3。这是"面积"的比例，不是"相同宽度"。
    # 实测反推：主标题块 610×198 ≈ 120,780 → 目标 40,260；
    #   副标题框 546×74 = 40,404 ✓ 宽度用满栏即可
    #   正文块不能用满栏（546 宽只需 74px 高，字号会掉到 11px 不可读），
    #   改用较窄的栏（≈0.34W），让高度落在可读字号区间：404×100 ≈ 40,400 ✓
    TITLE_AREA = TITLE_W * block_h_
    TARGET = TITLE_AREA / 3.0
    BAR_W = max(160, int(W * 0.46))
    sub_h = max(56, int(TARGET / BAR_W))
    sub_w = min(BAR_W, max(COL_W, int(W * 0.40)))
    body_w = max(160, int(W * 0.34))

    # ---- 主标题（第一优先，最上）→ 副标题（紧随其下）----
    col_w = max(COL_W, TITLE_W)
    slots_top = plan_column(shape.mask, [("title", TITLE_W, block_h_), ("sub", sub_w, sub_h)],
                            COL_X, gap, w_max=col_w, y_lo=0, y_hi=H,
                            allow_shrink=False, max_overlap=MAX_OVERLAP)
    if "title" in slots_top:
        spec.title_y = slots_top["title"].y
        spec.micro_boxes = [(COL_X, spec.title_y, TITLE_W, block_h_, "title")]

    # ---- 正文（第三优先，在副标题之下）----
    # 字号从小到大试，取第一个"面积达标且放得下"的档；降级链：缩字号 → 收窄 → 不画。
    slots_body = {}
    body_bs, body_lines = 0, []
    if spec.body:
        y_body_lo = 0
        if "sub" in slots_top:
            y_body_lo = slots_top["sub"].y + slots_top["sub"].h + gap
        elif "title" in slots_top:
            y_body_lo = slots_top["title"].y + slots_top["title"].h + gap
        for bs in range(max(12, int(W * 0.019)), 9, -1):
            lines = _wrap(_PROBE, spec.body, FONTS.get(spec.font_serif, bs), body_w)
            bh = len(lines) * int(bs * 1.85) + int(bs * 1.2)
            if body_w * bh < TARGET * 0.75:      # 面积不足 → 字号还没够大
                continue
            slots = plan_column(shape.mask, [("body", body_w, bh)], COL_X, gap,
                                w_max=body_w, y_lo=y_body_lo, allow_shrink=True,
                                max_overlap=MAX_OVERLAP)
            if "body" in slots:
                slots_body = slots
                body_bs, body_lines = bs, lines
                break
    if not slots_top and not slots_body:
        return canvas

    # ★★★ 先把【所有】文字框的位置定下来并记录，再画色块。
    # 原先正文/副标题的框是在各自绘制时才 append 的，而色块画在它们之前 ——
    # 于是色块只避让了主标题，对正文和副标题毫不知情（实测挤在同一条左栏）。
    # 现在：规划 → 记录全部框 → 画色块 → 画文字。z 序不变，但避让信息完整。
    if body_lines and "body" in slots_body:
        _s = slots_body["body"]
        _bh_ = len(body_lines) * int(body_bs * 1.85) + int(body_bs * 1.2)
        spec.micro_boxes.append((_s.x, _s.y, _s.w, _bh_, "body"))
    if "sub" in slots_top:
        _s = slots_top["sub"]
        spec.micro_boxes.append((_s.x, _s.y, _s.w, _s.h, "sub"))

    # ★ 互补色【纯色块】—— 无意义组件的新成员。
    # 颜色由 theme.comp 派生（采样色色相 +180°、饱和度顶 0.92），
    # 面积固定 = 画布 1/18（用户定的：1/7 时太抢），贴着主体站但既不压主体、
    # 也不压任何文字框；多块时优先分散到不同空白区域。
    if spec.micro_boxes:
        _avoid = np.zeros_like(shape.mask)
        for _x, _y, _w, _h, _n in spec.micro_boxes:
            _x0, _y0 = max(0, _x - _PAD * 3), max(0, _y - _PAD * 3)
            _x1, _y1 = min(W, _x + _w + _PAD * 3), min(H, _y + _h + _PAD * 3)
            _avoid[_y0:_y1, _x0:_x1] = True
        _cbs = comp_blocks(shape, avoid=_avoid)
        if os.environ.get("COLLAGE_DEBUG"):
            print("[comp] 文字框 %s" % [(x, y, w, h, n) for x, y, w, h, n in spec.micro_boxes])
            print("[comp] 色块 %d 个 %s" % (len(_cbs), _cbs))
        for _bx, _by, _bw, _bh in _cbs:
            d.rectangle([_bx, _by, _bx + _bw, _by + _bh], fill=theme.comp)

    # 正文（底部区）
    if body_lines and "body" in slots_body:
        s = slots_body["body"]
        bx2, by2 = s.x, s.y
        lines1 = _wrap(_PROBE, spec.body, FONTS.get(spec.font_serif, body_bs), s.w)
        bw2 = s.w
        bs = body_bs
        bp = int(bs * 1.2)
        d.rectangle([bx2 - _PAD, by2 - _PAD, bx2 + bw2 + _PAD,
                     by2 + len(lines1) * int(bs * 1.85) + bp], fill=theme.paper)
        d.line([bx2, by2, bx2 + int(bs * 4), by2], fill=theme.accent, width=3)
        for k, ln in enumerate(lines1):
            d.text((bx2, by2 + 10 + k * int(bs * 1.85)), ln,
                   font=FONTS.get(spec.font_serif, bs), fill=(52, 50, 48))

    # 副标题方框（顶部区）
    if "sub" not in slots_top:
        return canvas
    s = slots_top["sub"]
    sw, sh, sx, sy = s.w, s.h, s.x, s.y
    d.rectangle([sx - _PAD, sy - _PAD, sx + sw + _PAD, sy + sh + _PAD], fill=theme.paper)
    d.rectangle([sx, sy, sx + sw, sy + sh], outline=theme.ink, width=3)
    pin = int(sh * 0.20)
    GAP = int(sw * 0.08)                      # ★ 固定的标签间距，先给间距再分宽度
    avail = sw - pin * 2 - GAP                # 两个标签可用的总宽
    sc = int((sh - pin * 2) * 0.82)
    while text_size(spec.sub, spec.font_bold, sc)[0] > avail * 0.55 and sc > 10:
        sc -= 1
    cw, ch, _ = text_size(spec.sub, spec.font_bold, sc)
    d.text((sx + pin, sy + (sh - ch) // 2), spec.sub,
           font=FONTS.get(spec.font_bold, sc), fill=theme.ink)
    sen = 10
    while text_size(spec.sub_en, spec.font_mono, sen + 1)[0] <= avail * 0.45:
        sen += 1
    sew, seh, _ = text_size(spec.sub_en, spec.font_mono, sen)
    d.text((sx + sw - pin - sew, sy + (sh - seh) // 2), spec.sub_en,
           font=FONTS.get(spec.font_mono, sen), fill=theme.accent)
    return canvas
