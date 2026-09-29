# -*- coding: utf-8 -*-
"""
collage · 形状内拼贴排版

把文字与图形组件拼贴进任意形状（人物剪影、物品轮廓）的排版引擎。
"""

from .shape import load_mask, ShapeInfo
from .theme import theme_from_image
from .partition import quadtree
from .texture import grain, worley, tile
from .layout import build_layout, LayoutSpec
from .render import render, PARADIGMS

__version__ = "0.1.0"
__all__ = [
    "load_mask", "ShapeInfo",
    "theme_from_image",
    "quadtree",
    "grain", "worley", "tile",
    "build_layout", "LayoutSpec",
    "render", "PARADIGMS",
]
