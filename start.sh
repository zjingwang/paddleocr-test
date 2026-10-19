#!/usr/bin/env bash
# ============================================================
#  启动 PaddleOCR 测试后端
#
#  用法：
#    ./start.sh          # 默认端口 8000（前端 index.html 默认连这个端口）
#    ./start.sh 9000     # 指定端口 9000（需同步修改 index.html 里的 API_BASE）
#
#  说明：
#    - 模型缓存放固定在项目内 .smoke_home 目录（已被 .gitignore 忽略），
#      首次运行会自动下载一次，之后每次启动都直接复用，不会重复下载。
#    - 本脚本只影响这个项目的运行环境，不修改你机器的全局配置。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8000}"

# 优先使用项目内 venv，其次用系统 python3
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

# 把 paddle / paddlex 的模型缓存指向项目内可写目录（持久复用）
export HOME="$PWD/.smoke_home"
mkdir -p "$HOME"

echo "▶ 启动后端：端口 ${PORT}，模型缓存：${HOME}（首次运行会自动下载模型）"
exec "$PY" app.py --port "$PORT"
