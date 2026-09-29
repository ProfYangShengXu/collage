"""命令行入口：collage <图片> [选项]

两种调用方式都通：
    python -m collage examples/tiger.jpg --title "TIGER SHIKIGAMI"
    collage examples/tiger.jpg --title "TIGER SHIKIGAMI"   （pip install -e . 之后）
"""
from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

from .layout import LayoutSpec
from .render import LABELS, PARADIGMS, render_all
from .shape import load_mask
from .theme import theme_from_image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collage",
        description="把文字与图形组件拼贴进任意形状（剪影 / 轮廓）的排版引擎",
    )
    p.add_argument("image", nargs="?", help="输入图（白底或透明 PNG）")
    p.add_argument("-o", "--out", default="out", help="输出目录（默认 out/）")
    p.add_argument("--title", default="TITLE", help="主标题")
    p.add_argument("--title-sub", default="subtitle", help="主标题下方的英文小字")
    p.add_argument("--sub", default="SUB", help="副标题方框里的字")
    p.add_argument("--sub-en", default="LABEL", help="副标题方框右侧的标签")
    p.add_argument("--body", default="", help="正文段落")
    p.add_argument("--meta", action="append", default=[],
                   help="右下角元信息，可重复传多次")
    p.add_argument("--tags", default="01,2026,DRAFT",
                   help="散落的微型标签，逗号分隔")
    p.add_argument("--threshold", type=int, default=242,
                   help="白底阈值（默认 242，背景偏灰时调高）")
    p.add_argument("--largest-only", action="store_true",
                   help="只保留最大连通域（素材是单一主体时更快）")
    p.add_argument("--paradigm", choices=list(PARADIGMS) + ["all"], default="all",
                   help="只出某一种范式（默认三种都出）")
    p.add_argument("--prefix", default="", help="输出文件名前缀")
    p.add_argument("--layout-font", default=None,
                   help="标题字体路径（默认按平台自动解析）")
    p.add_argument("--check-fonts", action="store_true",
                   help="只打印各字体解析结果后退出")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.check_fonts:
        from .fonts import describe
        for k, v in describe().items():
            print(f"  {k:10s} {v or '（未找到，将退回 PIL 默认字体）'}")
        return 0

    if args.image is None:
        build_parser().print_help()
        return 2

    if not os.path.exists(args.image):
        print(f"找不到输入图：{args.image}", file=sys.stderr)
        return 2

    print(f"[1/3] 提取形状：{args.image}")
    shape = load_mask(args.image, white_thresh=args.threshold,
                      largest_only=args.largest_only)
    bx, by, bw, bh = shape.bbox
    print(f"      画布 {shape.width}x{shape.height}  形状覆盖 "
          f"{shape.area / (shape.width * shape.height) * 100:.1f}%  "
          f"包围盒 {bw}x{bh}")

    print("[2/3] 采样配色（从原图派生，不写死）")
    img = Image.open(args.image).convert("RGB")
    theme = theme_from_image(img, shape.mask, verbose=True)

    spec = LayoutSpec(
        title=args.title, title_sub=args.title_sub,
        sub=args.sub, sub_en=args.sub_en, body=args.body,
        meta=args.meta or [f"{os.path.basename(args.image)}"],
        tags=[t.strip() for t in args.tags.split(",") if t.strip()],
        font_bold=args.layout_font,
    )

    print(f"[3/3] 渲染（{args.paradigm}）")
    if args.paradigm == "all":
        paths = render_all(shape, spec, theme, source_img=img,
                           out_dir=args.out, prefix=args.prefix)
        print(f"\n完成，{len(paths)} 张 -> {os.path.abspath(args.out)}")
    else:
        from .render import render
        os.makedirs(args.out, exist_ok=True)
        out = render(shape, spec, theme, args.paradigm, img)
        path = os.path.join(args.out, f"{args.prefix}{LABELS[args.paradigm]}.png")
        out.save(path)
        print(f"\n完成 -> {os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
