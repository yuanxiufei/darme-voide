# Drama Studio —— 生产镜像（**Python 后端版**，2026-09-15 重写）
#
# 与旧版的差别（旧版是 Node 时代产物，随 `backend/` 一起退役）：
#   · 运行时从 `node:20-slim` + tsx 换成 `python:3.12-slim` + uvicorn；Node **只**留在前端构建阶段
#   · 端口 5789 → **5790**；CMD 从 `tsx backend/src/index.ts` 换成 `uvicorn app.main:app`
#   · `skills/`、`scripts/`、`local_services/` 2026-09-15 统一并入 `backend-py/app/`
#   · Python 侧自己托管前端静态产物与 SPA 回退（见 `backend-py/app/main.py` 第 6 条路由）
#
# ⚠️ **本文件在本机未实测**：这台机器没装 Docker（见 `.codebuddy/memory`），
#    只做了静态核对（路径/端口/产物目录与代码里的常量逐一对齐）。首次真构建请留意：
#    `FRONTEND_DIST = PROJECT_ROOT/frontend/dist`、数据根 = `PROJECT_ROOT/data`（`PROJECT_ROOT`
#    由 `backend-py/app/core/config.py` 上溯三级得到 ⇒ 容器内必须是 `/app/backend-py/app/...`）。

# ── Stage 1: 构建前端静态产物 ──────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run generate

# ── Stage 2: 运行时（Python，不带 Node）────────────────────────
FROM python:3.12-slim

# ffmpeg：合成 / 拼接 / 抽帧 / 校色 / 参考图压缩 / 连续性 QC 全靠它（Python 侧刻意不引入 Pillow/OpenCV）
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖先装（利用层缓存：改代码不必重装依赖）
COPY backend-py/requirements.txt ./backend-py/
RUN pip install --no-cache-dir -r backend-py/requirements.txt

# 应用代码（`skills/`、`scripts/`、`local_services/` 现在都在 `app/` 之内 ⇒ 由这一条覆盖）
# ⚠️ `agent/`、`mcp/` 是 `app/` 的**子包**（2026-09-15 方案 A 定稿：代码归代码包）⇒ 这一条 COPY 已覆盖它们；
#    `tests/dockerfile_contract_test.py` 守着「顶层包不再需要单独 COPY」。
COPY backend-py/app ./backend-py/app

# 配置：示例配置直接落成运行时配置（与原 Dockerfile 同策略；真实配置请挂载覆盖）
COPY configs/config.example.yaml ./configs/config.yaml

# 前端静态产物：Python 侧按 `PROJECT_ROOT/frontend/dist` 找它
COPY --from=frontend-build /app/frontend/.output/public ./frontend/dist

# 数据根（SQLite + static）：PROJECT_ROOT/data ⇒ /app/data
RUN mkdir -p /app/data/static

WORKDIR /app/backend-py

ENV PYTHONUNBUFFERED=1
ENV PORT=5790

EXPOSE 5790
VOLUME ["/app/data"]

# 用 exec 形式（不是 shell）以便正确接收 SIGTERM；⚠️ 改端口要**同时**改这里与上面的 EXPOSE
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5790"]
