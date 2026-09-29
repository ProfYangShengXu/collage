"""字体解析：跨平台找可用字体。

为什么不写死路径：`C:\\Windows\\Fonts\\...` 在 Linux / macOS 上不存在，
陌生人 clone 下来第一件事就是崩。这里按平台依次探测，
都没找到就退回 PIL 的默认字体（难看但能跑）。
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

# 每类字体的候选路径，按平台优先级排列
_BOLD = [
    r"C:\Windows\Fonts\msyhbd.ttc",                      # Windows 微软雅黑 Bold
    "/System/Library/Fonts/PingFang.ttc",                # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansSC-Bold.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_REG = [
    r"C:\Windows\Fonts\msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_SERIF = [
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
]
_MONO = [
    r"C:\Windows\Fonts\consola.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


@lru_cache(maxsize=None)
def _first_existing(paths: tuple[str, ...]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def resolve(kind: str = "bold") -> str | None:
    """按类别找一个存在的字体文件路径；找不到返回 None（使用者退回默认字体）。

    kind: bold / reg / serif / mono
    """
    table = {"bold": _BOLD, "reg": _REG, "serif": _SERIF, "mono": _MONO}
    if kind not in table:
        raise ValueError(f"未知字体类别 {kind}（可选 {list(table)}）")
    return _first_existing(tuple(table[kind]))


def describe() -> dict:
    """诊断用：当前每类字体实际解析到什么。"""
    found = {k: resolve(k) for k in ("bold", "reg", "serif", "mono")}
    found["platform"] = sys.platform
    return found
