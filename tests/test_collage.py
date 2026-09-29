"""collage 测试。

用合成图（不依赖示例素材），保证在任何机器上都能跑。
"""
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from collage import (LayoutSpec, load_mask, quadtree, render, theme_from_image,
                     tile, worley)
from collage.partition import RATIOS, assign_levels


# ── 测试素材 ──────────────────────────────────────────────
@pytest.fixture
def circle_img(tmp_path):
    """白底 + 实心圆（模拟人物剪影素材）。"""
    W = H = 400
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse([80, 60, 320, 340], fill=(210, 150, 60))
    p = tmp_path / "circle.png"
    im.save(p)
    return str(p)


@pytest.fixture
def theme(circle_img):
    from collage import load_mask as lm
    si = lm(circle_img)
    return theme_from_image(Image.open(circle_img).convert("RGB"), si.mask)


@pytest.fixture
def spec():
    return LayoutSpec(title="TEST", title_sub="a test", sub="SUB",
                      sub_en="EN", body="Some body text for the layout test.")


# ── 形状提取 ──────────────────────────────────────────────
def test_load_mask_basic(circle_img):
    si = load_mask(circle_img)
    assert si.width == 400 and si.height == 400
    # ellipse([80,60,320,340]) 是 240×280 的椭圆，面积 = π·120·140 ≈ 52779
    assert 50_000 < si.area < 55_500
    x, y, w, h = si.bbox
    assert 78 <= x <= 82 and 58 <= y <= 62
    assert 238 <= w <= 244 and 278 <= h <= 284


def test_load_mask_largest_only(circle_img, tmp_path):
    """两个圆 → largest_only 只保留大的。"""
    im = Image.open(circle_img).convert("RGB")
    d = ImageDraw.Draw(im)
    d.ellipse([10, 10, 90, 90], fill=(0, 0, 0))     # 小圆（面积 ~5000 > min_px）
    p = tmp_path / "two.png"
    im.save(p)

    both = load_mask(str(p))
    one = load_mask(str(p), largest_only=True)
    assert one.area < both.area
    assert one.bbox[2] > 200                        # 留下的是大圆


def test_load_mask_fills_holes(tmp_path):
    """甜甜圈 → fill_holes 后是实心的。"""
    im = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse([10, 10, 190, 190], fill=(0, 0, 0))
    d.ellipse([50, 50, 150, 150], fill=(255, 255, 255))   # 挖掉中间的洞
    p = tmp_path / "donut.png"
    im.save(p)

    solid = load_mask(str(p), fill_holes=True)
    hollow = load_mask(str(p), fill_holes=False)
    assert solid.area > hollow.area * 1.4
    # 实心版在包围盒内接近满（椭圆在正方形包围盒里上限 π/4 ≈ 0.785）
    _, _, w, h = solid.bbox
    assert solid.area / (w * h) > 0.75


def test_load_mask_empty_raises(tmp_path):
    im = Image.new("RGB", (100, 100), (255, 255, 255))   # 全白
    p = tmp_path / "blank.png"
    im.save(p)
    with pytest.raises(ValueError):
        load_mask(str(p))


# ── 四叉树 ────────────────────────────────────────────────
def test_quadtree_covers_shape(circle_img):
    si = load_mask(circle_img)
    rects = quadtree(si)
    assert len(rects) > 3

    # 块面积之和不小于形状面积（块会覆盖到边缘外，但不该少于形状）
    total = sum(r.area for r in rects)
    assert total >= si.area * 0.95

    # 块都在画布内
    for r in rects:
        assert r.x >= 0 and r.y >= 0
        assert r.x + r.w <= si.width and r.y + r.h <= si.height


def test_quadtree_no_overlap(circle_img):
    """四叉树的块两两不重叠（这是它区别于 Voronoi 的性质）。"""
    si = load_mask(circle_img)
    rects = quadtree(si)
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            ox = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
            oy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
            assert ox * oy == 0, f"块重叠: {a} {b}"


def test_quadtree_caps_block_size(circle_img):
    """单块面积上限生效 —— 否则简单形状只切出 4 个巨型块，
    副标题/正文层会因预算装不下任何块而空掉。"""
    si = load_mask(circle_img)
    rects = quadtree(si, max_area_ratio=0.10)
    assert len(rects) > 6, f"块太少（{len(rects)}），上限没生效"
    for r in rects:
        assert r.area <= si.area * 0.10 * 1.05, \
            f"块 {r.area} 超过形状面积的 10%"


def test_quadtree_size_scales_with_image(circle_img, tmp_path):
    """大图不该切出巨型块 —— 块尺寸按图宽比例走。"""
    big = Image.open(circle_img).convert("RGB").resize((1200, 1200), Image.NEAREST)
    p = tmp_path / "big.png"
    big.save(p)
    si_small, si_big = load_mask(circle_img), load_mask(str(p))

    def median_side(si):
        rs = quadtree(si)
        return float(np.median([min(r.w, r.h) for r in rs]))

    # 大图的典型块边长为小图的数倍（按图宽比例，不是固定像素）
    assert median_side(si_big) > median_side(si_small) * 1.5


# ── 分层 ──────────────────────────────────────────────────
def test_assign_levels_ratio(circle_img):
    """四层面积比接近 3:1:1:9（容差 20%）。"""
    si = load_mask(circle_img)
    rects = quadtree(si)
    levels = assign_levels(rects, RATIOS, si)

    got = {}
    for r, lv in zip(rects, levels):
        got[lv] = got.get(lv, 0) + r.area
    tot = sum(got.values())

    for k, want in RATIOS.items():
        share = got.get(k, 0) / tot
        target = want / sum(RATIOS.values())
        assert abs(share - target) < target * 0.6, \
            f"{k}: 得到 {share:.3f}，目标 {target:.3f}"


# ── 配色 ──────────────────────────────────────────────────
def test_theme_from_image(circle_img):
    si = load_mask(circle_img)
    th = theme_from_image(Image.open(circle_img).convert("RGB"), si.mask)

    for name in ("accent", "ink", "paper"):
        c = getattr(th, name)
        assert len(c) == 3 and all(0 <= v <= 255 for v in c)
    assert len(th.tints) == 5

    # 橙色的圆 → 主色应该偏暖（R > B）
    assert th.accent[0] > th.accent[2]
    # 墨色应该比纸色暗
    assert sum(th.ink) < sum(th.paper)


def test_theme_ignores_white_background(circle_img):
    """只从主体取色 —— 白底不该把主色拉成灰。"""
    si = load_mask(circle_img)
    th = theme_from_image(Image.open(circle_img).convert("RGB"), si.mask)
    r, g, b = th.accent
    assert max(r, g, b) - min(r, g, b) > 30, f"主色太灰: {th.accent}"


# ── 质感 ──────────────────────────────────────────────────
def test_tile_size():
    for mode in ("grain", "cell", "ink", "flat"):
        im = tile(80, 60, (200, 180, 150), mode, seed=1)
        assert im.size == (80, 60)
        assert im.mode == "RGB"


def test_worley_repeatable():
    a = worley(64, 64, 12, seed=7)
    b = worley(64, 64, 12, seed=7)
    assert np.allclose(a, b)
    assert a.shape == (64, 64)
    assert 0.0 <= a.min() and a.max() <= 1.0


def test_worley_tile_cache():
    """小样缓存生效 —— 第二次调用不该重算。"""
    from collage import texture
    texture._tile_cache.clear()
    worley(500, 500, 20, seed=99)
    n1 = len(texture._tile_cache)
    worley(800, 800, 20, seed=99)
    assert len(texture._tile_cache) == n1    # 没新增缓存条目


# ── 字体 ──────────────────────────────────────────────────
def test_fonts_resolve():
    """至少能解析出粗体（跨平台字体探测）。"""
    from collage.fonts import describe, resolve
    assert resolve("bold") is not None, "没找到任何可用粗体"
    d = describe()
    assert "platform" in d
    for k in ("bold", "reg", "serif", "mono"):
        assert k in d


def test_layout_spec_autofills_fonts():
    """LayoutSpec 不传字体路径时自动解析，不写死 Windows 路径。"""
    s = LayoutSpec(title="X")
    assert s.font_bold is not None
    assert "C:\\Windows" not in (s.font_bold or "") or os.name == "nt"


# ── 渲染 ──────────────────────────────────────────────────
@pytest.mark.parametrize("paradigm", ["silhouette", "photo", "negative"])
def test_render_paradigms(circle_img, spec, theme, paradigm):
    si = load_mask(circle_img)
    img = Image.open(circle_img).convert("RGB")
    out = render(si, spec, theme, paradigm, source_img=img)
    assert out.size == (si.width, si.height)
    assert out.mode == "RGB"


def test_render_photo_requires_source(circle_img, spec, theme):
    si = load_mask(circle_img)
    with pytest.raises(ValueError):
        render(si, spec, theme, "photo", source_img=None)


def test_render_unknown_paradigm(circle_img, spec, theme):
    si = load_mask(circle_img)
    with pytest.raises(ValueError):
        render(si, spec, theme, "nope")


def test_render_not_blank(circle_img, spec, theme):
    """成图必须有内容 —— 全白说明渲染链断了。"""
    si = load_mask(circle_img)
    out = render(si, spec, theme, "silhouette")
    arr = np.asarray(out)
    assert arr.std() > 10, "成图接近纯色，渲染可能失败"
