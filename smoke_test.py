# -*- coding: utf-8 -*-
"""smoke_test.py —— 真实 PaddleOCR 冒烟测试：渲染含形近字的中文图片并识别。"""
import os
import sys

os.environ["HOME"] = os.path.join(os.getcwd(), ".smoke_home")
os.environ.setdefault("PADDLE_OCR_USE_GPU", "0")

from PIL import Image, ImageDraw, ImageFont

import app as app_module

FONT = "/usr/share/fonts/SourceHanSansCN-Medium.otf"


def make_text_image(text, font_path=FONT, size=52, pad=24):
    """用系统思源黑体把文字渲染成白底黑字图片（模拟二值化较好的印刷/盖章字体）。"""
    font = ImageFont.truetype(font_path, size)
    # 先量尺寸
    tmp = Image.new("RGB", (10, 10), (255, 255, 255))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(0, 0, 0))
    return img


def main():
    print("[1/3] 初始化 PaddleOCR 并加载中文模型（首次会自动下载，约几十MB）...", flush=True)
    engine_ver = None
    try:
        ocr, major = app_module._get_ocr()
        import paddleocr
        engine_ver = f"paddleocr {paddleocr.__version__} (主版本 {major})"
    except Exception as e:
        print(f"[FATAL] PaddleOCR 初始化失败: {e}", file=sys.stderr)
        return 1
    print("[2/3] 引擎就绪:", engine_ver, flush=True)

    tests = [
        "沭阳县迎宾大道 沐阳",
        "12345 67890 ABC",
        "江苏沭阳 农商银行",
    ]
    print("[3/3] 开始识别 ...", flush=True)
    for text in tests:
        img = make_text_image(text)
        try:
            res = app_module._run_ocr(img)
        except Exception as e:
            print(f"  [{text}] 识别异常: {e}", flush=True)
            continue
        rows = [f"{it['text']}({it['score']*100:.1f}%)" for it in res["items"]]
        print(f"  ─ 原图文字: {text}", flush=True)
        print(f"    OCR 结果: {'  '.join(rows) if rows else '(未识别到文本)'}", flush=True)
        print(f"    耗时 {res['elapsed_ms']}ms, {res['count']} 行", flush=True)
    print("SMOKE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
