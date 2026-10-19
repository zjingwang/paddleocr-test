# 🖥️ PaddleOCR 识别准确率测试台

一个 **完全本地离线** 的 PaddleOCR 中文识别测试工具，带图形化网页界面。面向**财务审计**场景，专门用来验证 PaddleOCR 对中文**形近字**（如「沭阳 / 沐阳」）和**连笔手写体**的识别准确率，为正式接入生产对账脚本提供评估依据。

- 后端：Python + FastAPI（调用 PaddleOCR，`lang="ch"` 中文高精度 + 文本行方向校正）
- 前端：单个 HTML 文件，双击即可打开（拖拽上传图片 / 画板手写 / 实时展示结果与置信度）
- 兼容 PaddleOCR 2.x 与 3.x 两代 API，自动适配，无需改代码

---

## 📁 目录结构

```
paddleocr-test/
├── app.py            # 后端服务（FastAPI + PaddleOCR + 跨域 + 异常处理）
├── index.html        # 前端网页（单文件，双击即用）
├── requirements.txt  # Python 依赖清单
├── tests.py          # 后端自测脚本（无需真实模型）
├── smoke_test.py     # 真实模型冒烟测试（需要已下载模型）
├── sample_print.png  # 测试样例：印刷体「沭阳县迎宾大道 沐阳 12345」
├── sample_hand.png   # 测试样例：手写风「沐阳沭阳」（魏碑字体+倾斜）
└── LICENSE           # Apache-2.0 开源协议
```

---

## 🚀 快速开始（第一次使用请看这里）

### 0️⃣ 前置要求

- 已安装 **Python 3.10 或更高版本**（官网下载即可：https://www.python.org/downloads/）
- 能联网（安装依赖；PaddleOCR 模型首次运行也会联网下载）

> Windows 用户在下面的命令行中请使用 `py` 或 `python`（需在“命令提示符 cmd”或 PowerShell 中执行）。

### 1️⃣ 安装依赖

打开终端，进入本项目目录后执行：

```bash
pip install -r requirements.txt
```

> 依赖约几百 MB（含 paddlepaddle 深度学习框架），请耐心等待安装完成。
> 若下载慢，可换国内镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 2️⃣ 启动后端

在本目录执行：

```bash
python app.py
```

看到类似下面的提示就说明成功了（默认监听 8000 端口）：

```
========================================================
  PaddleOCR 识别准确率测试后端已启动
  前端默认请求后端地址: http://127.0.0.1:8000
========================================================
```

> **首次启动较慢（5~30 秒）**：后端会加载/下载 PaddleOCR 中文模型，之后每次启动就快了。
> 想换端口：`python app.py --port 9000`（前端需同步修改，见下方 FAQ）。

### 3️⃣ 打开前端网页

**直接双击** 项目里的 `index.html` 文件，浏览器会自动打开页面。无需安装任何 Web 服务器。

> 页面顶部有一个状态徽标：🟢 表示后端已连接；🔴 表示连不上（请确认第 2 步的后端正在运行，或点击徽标重试）。

### 4️⃣ 开始测试（两种方式）

**方式 A：上传图片识别**
1. 把发票 / 快递单 / 手写单据图片**拖拽**到左侧「① 上传图片文件」区域（或点击选择文件）；
2. 选择后自动开始识别，右侧立即展示结果。

**方式 B：画板手写识别**
1. 在左侧「② 手写识别」的画板上用**鼠标 / 手指 / 触控笔**书写文字（连笔、形近字都可以）；
2. 点击 **「识别画板文字」** 按钮；
3. 结果同样显示在右侧，并且画板上会**叠加彩色检测框**，方便对照识别位置。

**怎么看结果（右侧面板）**
- 每一行 = 一行识别文本 + 一个**置信度评分**（百分比）；
- 🟢 绿色 ≥ 90%：可信；🟡 黄色 75%~90%：一般；🔴 红色 < 75%：**建议人工复核**（审计重点）；
- 顶部统计区显示：识别行数、平均置信度、耗时、低置信行数；
- 「复制全部文本」一键复制结果；点单行「复制」可复制单条。

> 🎯 想快速验证？没有发票图片也能测：直接用仓库自带的 `sample_print.png`（印刷体）和 `sample_hand.png`（手写风）上传即可。画板推荐先写：沭阳、沐阳、已、己、末、采、彩 这些形近字。

---

## 🛡️ 完全离线部署（审计内网必读）

PaddleOCR 模型在**本机没有缓存**时才会联网下载。要做到断网也能用：

1. **在联网的电脑上**执行一次预热（下载并校验全部中文模型）：
   ```bash
   python app.py --warmup
   ```
2. 把模型缓存目录拷贝到内网机器，路径参考：
   - Windows：`C:\Users\你的用户名\.paddleocr` 和 `.paddlex`
   - Linux / macOS：`~/.paddleocr` 和 `~/.paddlex`
3. 在内网机器上正常 `python app.py` 即可，全程不发起任何网络请求。

后端默认 CPU 推理，稳定可靠；若内网机器有 NVIDIA GPU 且装有 CUDA，可加环境变量 `PADDLE_OCR_USE_GPU=1` 提速。

---

## 📡 API 接口（供进阶使用 / 对接生产脚本）

| 接口 | 方法 | 请求格式 | 返回 |
|---|---|---|---|
| `/api/ocr-file` | POST | `multipart/form-data`，字段名 `file` | `{success, count, avg_score, elapsed_ms, engine, items:[{text, score, box}]}` |
| `/api/ocr-canvas` | POST | `{"image_base64": "data:image/png;base64,..."}` | 同上 |
| `/api/health` | GET | 无 | `{status, engine, detail}` |

- 错误情况返回 `{success: false, detail: "中文原因"}`，HTTP 状态码 400/413/422/500。
- `box` 是文本框四点坐标 `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`，可直接用于留痕/定位。

把 `app.py` 里的 `_run_ocr(PIL.Image)` 函数单独拿出来，就能在自己的 Python 生产脚本里直接复用：
```python
from app import _run_ocr
from PIL import Image
result = _run_ocr(Image.open("发票.jpg"))
for it in result["items"]:
    print(it["text"], it["score"])
```

---

## 🧪 自测

```bash
pip install httpx            # 仅运行自测需要
python tests.py              # 37 项接口/异常/CORS/版本适配自测，全部 PASS 即正常
python smoke_test.py         # 真实模型冒烟测试（需已联网下载过模型）
```

---

## ❓ 常见问题（FAQ）

**Q1：打开网页后提示「后端未连接」？**
后端服务没启动，或端口不对。先执行 `python app.py`；如果换了端口（如 9000），请打开 `index.html`，找到开头 `const API_BASE = "http://127.0.0.1:8000";` 把端口改成 9000 后刷新页面。

**Q2：点击识别后提示「未检测到 PaddleOCR」？**
依赖没装全。执行 `pip install -r requirements.txt` 后重启后端。

**Q3：首次识别很慢，还打印一堆下载进度？**
正常，模型首次加载/下载需要时间。若在断网机器上使用，请先按上文「完全离线部署」操作。

**Q4：报错 `ConvertPirAttribute2RuntimeAttribute`？**
这是 paddlepaddle 3.x 在 CPU 上开启 oneDNN 的已知兼容问题，本项目已默认关闭规避。如果你的 paddle 版本无此问题且想提速，设环境变量 `PADDLE_OCR_USE_MKLDNN=1`。

**Q5：手写识别结果不理想？**
先确认字迹清晰、一笔一画；连笔字可多写几个样本对比。本工具就是用来评估不同写法下准确率的，结果不理想也属于有价值的结论。

**Q6：这些识别结果会不会上传到外部？**
不会。模型与推理全部在本机完成，只会从页面向你本机的后端发请求，不经过任何第三方服务器。

---

## 📜 协议

本项目基于 [Apache License 2.0](LICENSE) 开源，可自由使用、修改、商用。
