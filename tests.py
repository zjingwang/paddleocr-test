# -*- coding: utf-8 -*-
"""
tests.py —— app.py 的本地自测脚本（无需真实 PaddleOCR 也可运行大部分用例）

覆盖范围：
  1. HTTP 接口层的正常路径 / 各类异常路径（用假 _run_ocr 替换真实推理）
  2. CORS 预检与响应头
  3. Base64 / 图片字节解码与格式校验的边界情况
  4. PaddleOCR 2.x 与 3.x 两套 API 的版本适配逻辑（注入假 paddleocr 模块）
运行： .venv/bin/python tests.py
"""
import io
import sys
import types

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module


# ============================================================================
# 工具：生成测试图片
# ============================================================================
def make_png_bytes(size=(320, 120), color=(255, 255, 255)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_png_rgba_bytes(size=(64, 64)):
    """RGBA 透明 PNG：模拟 Canvas 导出的手写图，验证白底合成逻辑不抛错且尺寸不变"""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================================
# 占位推理：让接口测试不依赖真实模型
# ============================================================================
FAKE_RESULT = {
    "count": 2,
    "elapsed_ms": 12.3,
    "avg_score": 0.9,
    "engine": "fake-1.0",
    "items": [
        {"text": "沭阳县迎宾大道", "score": 0.99, "box": [[10, 10], [200, 10], [200, 40], [10, 40]]},
        {"text": "沐阳", "score": 0.71, "box": [[10, 50], [50, 50], [50, 80], [10, 80]]},
    ],
}


def patch_ocr_for_http():
    """把 app 模块内的推理替换为假实现，避免接口测试触发真实模型；
    同时保留原函数引用，供第 6 节版本适配测试恢复使用。"""
    global _real_get_ocr, _real_run_ocr
    _real_get_ocr = app_module._get_ocr
    _real_run_ocr = app_module._run_ocr
    app_module._run_ocr = lambda img: dict(FAKE_RESULT)
    app_module._get_ocr = lambda: (object(), 3)


def restore_ocr_functions():
    app_module._get_ocr = _real_get_ocr
    app_module._run_ocr = _real_run_ocr


client = TestClient(app_module.app, raise_server_exceptions=False)

RESULTS = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    if cond:
        RESULTS["pass"] += 1
        print(f"  [PASS] {name}")
    else:
        RESULTS["fail"] += 1
        print(f"  [FAIL] {name}  {detail}")


# ============================================================================
# 1. 健康探活
# ============================================================================
patch_ocr_for_http()
r = client.get("/api/health")
check("GET /api/health -> 200", r.status_code == 200, r.text)
check("health 含 status 字段", "status" in r.json())


# ============================================================================
# 2. 上传文件接口：正常路径 + 各类异常
# ============================================================================
r = client.post("/api/ocr-file", files={"file": ("a.png", make_png_bytes(), "image/png")})
j = r.json()
check("POST /api/ocr-file 正常 -> 200", r.status_code == 200, r.text)
check("返回 success=true", j.get("success") is True)
check("返回 items 长度=2", len(j.get("items", [])) == 2)
check("文本内容正确", j["items"][0]["text"] == "沭阳县迎宾大道")
check("置信度字段为数字", isinstance(j["items"][0]["score"], float))

# 非图片内容 → 400
r = client.post("/api/ocr-file", files={"file": ("bad.txt", b"hello not an image", "text/plain")})
check("上传非图片 -> 400", r.status_code == 400, r.text)
check("错误信息含'无法解析'", "无法解析" in r.json().get("detail", ""))

# 0 字节文件 → 400
r = client.post("/api/ocr-file", files={"file": ("empty.png", b"", "image/png")})
check("上传空文件 -> 400", r.status_code == 400, r.text)

# 缺少 file 字段 → 422 (FastAPI 校验)
r = client.post("/api/ocr-file")
check("缺少文件字段 -> 422", r.status_code == 422, f"got {r.status_code}")

# 超过 20MB → 413
big = b"x" * (20 * 1024 * 1024 + 1)
r = client.post("/api/ocr-file", files={"file": ("big.png", big[:10], "image/png")})
# 用足够大的真实 PNG 字节触发大小校验
png20m = make_png_bytes(size=(1, 1)) * 0  # 占位
import struct
# 直接构造 20MB+1 的元数据文件
fake_big_png = make_png_bytes(size=(1, 1)) + b"\x00" * (20 * 1024 * 1024)
r = client.post("/api/ocr-file", files={"file": ("big.png", fake_big_png, "image/png")})
check("超大文件 -> 413", r.status_code == 413, f"got {r.status_code}: {r.text}")


# ============================================================================
# 3. Canvas 接口：正常路径 + 各类异常
# ============================================================================
import base64

png_b64 = base64.b64encode(make_png_rgba_bytes()).decode("ascii")
data_url = f"data:image/png;base64,{png_b64}"

r = client.post("/api/ocr-canvas", json={"image_base64": data_url})
check("POST /api/ocr-canvas 正常 -> 200", r.status_code == 200, r.text)
check("canvas 返回 success=true", r.json().get("success") is True)

# 纯 base64（无 data: 前缀）也应兼容
r = client.post("/api/ocr-canvas", json={"image_base64": png_b64})
check("纯 base64 兼容 -> 200", r.status_code == 200, r.text)

# 非法 base64 字符 → 400
r = client.post("/api/ocr-canvas", json={"image_base64": "!!not-base64!!"})
check("非法 base64 -> 400", r.status_code == 400, r.text)
check("错误信息含'Base64'", "Base64" in r.json().get("detail", ""))

# 缺字段 → 400
r = client.post("/api/ocr-canvas", json={})
check("缺 image_base64 -> 400", r.status_code == 400, r.text)

# 非 dict 请求体 → 400 或 422（FastAPI 结构校验），核心要求是“不得 500”
r = client.post("/api/ocr-canvas", content="[1,2,3]", headers={"Content-Type": "application/json"})
check("数组请求体 -> 4xx（非500）", r.status_code in (400, 422), f"got {r.status_code}: {r.text[:120]}")

r = client.post("/api/ocr-canvas", content="null", headers={"Content-Type": "application/json"})
check("null 请求体 -> 4xx（非500）", r.status_code in (400, 422), f"got {r.status_code}: {r.text[:120]}")

# 正常 base64 但内容不是图片 → 400（PIL 解码失败）
r = client.post("/api/ocr-canvas", json={"image_base64": base64.b64encode(b"not an image at all").decode()})
check("内容非图片 -> 400", r.status_code == 400, r.text)


# ============================================================================
# 4. CORS 跨域
# ============================================================================
r = client.options(
    "/api/ocr-canvas",
    headers={
        "Origin": "null",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    },
)
check("CORS 预检 -> 200", r.status_code == 200, f"got {r.status_code}")
check("预检响应带 allow-origin=*", r.headers.get("access-control-allow-origin") == "*",
      r.headers.get("access-control-allow-origin"))
check("预检响应带 allow-headers", r.headers.get("access-control-allow-headers") is not None)

r = client.post("/api/ocr-canvas", json={"image_base64": data_url}, headers={"Origin": "null"})
check("实际请求响应带 allow-origin=*", r.headers.get("access-control-allow-origin") == "*",
      r.headers.get("access-control-allow-origin"))


# ============================================================================
# 5. 透明图白底合成（真实 _load_image_from_bytes 逻辑）
# ============================================================================
rgba = make_png_rgba_bytes(size=(100, 100))
img = app_module._load_image_from_bytes(rgba, source="测试")
check("RGBA 图已转为 RGB", img.mode == "RGB", img.mode)
check("尺寸保持不变", img.size == (100, 100), img.size)


# ============================================================================
# 6. PaddleOCR 版本适配：注入两个假模块分别模拟 3.x 与 2.x
#    （先恢复被 HTTP 测试替换掉的真实 _get_ocr/_run_ocr 函数）
# ============================================================================
restore_ocr_functions()
def inject_fake_paddleocr(version, cls_impl):
    fake = types.ModuleType("paddleocr")
    fake.__version__ = version
    fake.PaddleOCR = cls_impl
    sys.modules["paddleocr"] = fake
    return fake


def make_fake_paddleocr_3x(captured):
    class PaddleOCR:
        def __init__(self, **kw):
            captured["kwargs"] = kw
        def predict(self, arr):
            class Res:
                @property
                def json(self):
                    return {"res": {
                        "rec_texts": ["沭阳", "沐阳"],
                        "rec_scores": [np.float32(0.99), np.float32(0.71)],
                        "rec_polys": np.array([[[0, 0], [10, 0], [10, 10], [0, 10]],
                                               [[0, 20], [10, 20], [10, 30], [0, 30]]], dtype=np.float32),
                    }}
            return [Res()]
    return PaddleOCR


def make_fake_paddleocr_2x(captured):
    class PaddleOCR:
        def __init__(self, **kw):
            captured["kwargs"] = kw
        def ocr(self, arr, cls=True):
            return [[
                ([[0, 0], [10, 0], [10, 10], [0, 10]], ("沭阳", np.float32(0.98))),
                ([[0, 20], [10, 20], [10, 30], [0, 30]], ("沐阳", np.float32(0.66))),
            ], None]
    return PaddleOCR


def reset_ocr_cache():
    app_module._OCR_INSTANCE = None
    app_module._OCR_MAJOR_VERSION = 0


# ---- 3.x 分支 ----
reset_ocr_cache()
cap3 = {}
inject_fake_paddleocr("3.2.1", make_fake_paddleocr_3x(cap3))
ocr3, ver3 = app_module._get_ocr()
check("3.x 检测主版本=3", ver3 == 3, ver3)
check("3.x 使用 use_textline_orientation=True", cap3["kwargs"].get("use_textline_orientation") is True, cap3["kwargs"])
out3 = app_module._run_ocr(Image.new("RGB", (64, 64), (255, 255, 255)))
check("3.x 推理结果条数=2", out3["count"] == 2, out3)
check("3.x 文本解析正确", [i["text"] for i in out3["items"]] == ["沭阳", "沐阳"])
check("3.x 分数为 float", all(isinstance(i["score"], float) for i in out3["items"]))
check("3.x 坐标已转 list（可 JSON 序列化）", isinstance(out3["items"][0]["box"], list), type(out3["items"][0]["box"]))

# ---- 2.x 分支 ----
reset_ocr_cache()
cap2 = {}
inject_fake_paddleocr("2.7.3", make_fake_paddleocr_2x(cap2))
ocr2, ver2 = app_module._get_ocr()
check("2.x 检测主版本=2", ver2 == 2, ver2)
check("2.x 使用 use_angle_cls=True", cap2["kwargs"].get("use_angle_cls") is True, cap2["kwargs"])
out2 = app_module._run_ocr(Image.new("RGB", (64, 64), (255, 255, 255)))
check("2.x 推理结果条数=2", out2["count"] == 2, out2)
check("2.x 文本解析正确（跳过 None 页）", [i["text"] for i in out2["items"]] == ["沭阳", "沐阳"])

# 清掉假模块，恢复现场
sys.modules.pop("paddleocr", None)
reset_ocr_cache()

# ============================================================================
# 汇总
# ============================================================================
print("\n" + "=" * 60)
print(f"总计：通过 {RESULTS['pass']} 项，失败 {RESULTS['fail']} 项")
print("=" * 60)
sys.exit(1 if RESULTS["fail"] else 0)
