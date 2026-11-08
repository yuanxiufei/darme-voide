# 🎬 Drama Studio - AI 短剧生成平台

<div align="center">

**基于 TypeScript 全栈的 AI 短剧自动化生产平台**

[![Node Version](https://img.shields.io/badge/Node.js-22.x-339933?style=flat&logo=node.js)](https://nodejs.org)
[![Vue Version](https://img.shields.io/badge/Vue-3.x-4FC08D?style=flat&logo=vue.js)](https://vuejs.org)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[功能特性](#功能特性) • [快速开始](#快速开始) • [部署指南](#部署指南)

</div>

---

## 📖 项目简介

Drama Studio 是一个基于 AI 的短剧自动化生产平台，实现从剧本生成、角色设计、分镜制作到视频合成的全流程自动化。

### 🎯 核心价值

- **🤖 AI 驱动**：使用大语言模型解析剧本，提取角色、场景和分镜信息
- **🎨 智能创作**：AI 绘图生成角色形象和场景背景
- **📹 视频生成**：基于文生视频和图生视频模型自动生成分镜视频
- **🔄 工作流**：完整的短剧制作工作流，从创意到成片一站式完成

### 🛠️ 技术架构

```
frontend/   — Nuxt 3 + Vue 3 + TypeScript (纯 CSS，无 UI 框架)；Web 端（3013）
desktop/    — Tauri 2 桌面壳 + 独立 Vite 前端（3014）。前端是 frontend/app 的**拷贝**
              （**不共享 node_modules、不跨工程 import**）：物理隔离由
              desktop/frontend/scripts/check-isolation.mjs 守卫，逐字一致性由
              两端各一份 scripts/check-parity.mjs 守卫（两份逐字相同，差异全部登记并写明理由）
backend/    — ❌ 已于 2026-09-15 **删除**（Node/Hono 后端；能力全部迁至 backend-py/，
              守卫靠 backend-py/tests/frozen_ts_source.py 的 TS 快照继续做「Python vs TS 当初」比对）
configs/    — config.yaml 配置文件 + ComfyUI 工作流/提示词资源
data/       — **运行时数据根**（= 你的作品目录，备份它）。顶层只放目录、不放散文件：
              `db/` 所有 SQLite（`drama.db` 作品库 ＋ 它的 `.bak-*` ＋ `library-overlay.sqlite`
              知识库编辑/上传台账，**两者都不可重建**）、`static/` 图片/音频/视频、`traces/` Agent
              轨迹、`knowledge/` 知识闭环派生库、`engine-dryrun/` 干跑凭据、`materialized/` LLM
              反量化权重缓存（可重建，几 GB 级）。逐层说明见 `data/README.md`。
              可被设置页「数据存储」整体切到别的盘（写仓库根的 `.data-root`）⇒ 这里只放
              **跟着数据根走**的东西。
corpus/     — **随仓库分发的语料**（入库，clone 就有）。`playbook/` 137 份方法原料，
              是知识库检索与 `search_script_library(source="playbook")` 的源（纪律见其 README）；
              `prompt-corpus/` 可选，第三方版权、不入库。
indexes/    — **知识库派生索引**（不入库，可 `--redo`/`--all` 重建，但要跑数小时）。
              `library-ocr.sqlite`／`library-extra.sqlite`／`library-scenes.sqlite` 三只 sqlite。
models/     — **本仓携带的模型素材**（不入库，可重下）。按用途分目录：`ocr/` OCR 权重（1.8 GiB）、
              `tokenizers/` 三份 Qwen 词表（18 MB）。⚠️ 与 `<dataRoot>/models`（model_manager 下载的
              几十 GB 出图权重）**不是一处**：那边是运行时数据、跟着数据根走。见 `models/README.md`。
backend-py/app/        — 后端代码包（装配 main.py + 平台层 core/ + HTTP 层 routers/ + 业务层 services/
                        + Agent 运行时 agent/ + MCP 客户端 mcp/；层间**单向依赖**由
                        backend-py/tests/layering_test.py 机械守卫）
backend-py/app/skills/     — Agent 技能（自有 SKILL.md + 外部技能库，约定见 backend-py/app/skills/README.md）
backend-py/app/local_services/ — 本地服务根（模型工具链 `git clone` 的第三方服务 + 项目自带 h3 8765 薄封装）
docs/       — 设计与审计文档（分层/契约/迁移证据都落这里，不散在代码注释里）
benchmarks/ — Agent 评估基准用例（4 份 JSON）
logs/       — 运行日志暂存区（`backend.out.log` 那类「手动起服务 `/` 跑全量」的屏幕输出，
              **全可删**、永不入库；2026-11-07 之前它们摊在仓库根上）。见 `logs/README.md`。
```

### 🎥 作品展示 / Demo Videos

体验 AI 短剧生成效果：

<div align="center">

**示例作品**

> 演示视频请在本地启动后通过浏览器访问体验。

</div>

---

## ✨ 功能特性

### 🎭 角色管理

- ✅ AI 生成角色形象
- ✅ 批量角色生成
- ✅ 角色图片上传和管理
- ✅ 角色音色分配与试听

### 🎬 分镜制作

- ✅ AI 自动拆解分镜脚本
- ✅ 场景描述和镜头设计
- ✅ 分镜图片生成（文生图）
- ✅ 宫格图生成、切分与分配
- ✅ 帧类型选择（首帧/尾帧/分镜板）

### 🎥 视频生成

- ✅ 图生视频自动生成
- ✅ TTS 配音生成
- ✅ FFmpeg 单镜头合成（视频 + 音频 + 字幕）
- ✅ 整集拼接导出

### 📦 资源管理

- ✅ 素材库统一管理
- ✅ 本地存储支持
- ✅ 任务进度追踪

### 🤖 AI Agents

内置 5 类 Agent（Python 侧 `backend-py/app/agent/`），支持数据库配置和 Skill 扩展：

| Agent | 职责 |
|---|---|
| `script_rewriter` | 小说 → 格式化剧本改写 |
| `extractor` | 角色 + 场景智能提取与去重 |
| `storyboard_breaker` | 剧本 → 分镜序列拆解 |
| `voice_assigner` | 角色音色自动分配 |
| `grid_prompt_generator` | 角色/场景/宫格图提示词生成 |

### 🔌 多厂商适配

| 类型 | 支持厂商 |
|---|---|
| **文本** | OpenAI 兼容（OpenAI / OpenRouter / 火山引擎 / 阿里 / MiniMax / chatfire）、Gemini、Ollama（走原生 `/api/chat`，不走 OpenAI 兼容端点） |
| **图片** | OpenAI、Gemini、MiniMax、火山引擎、阿里、第三方代理（chatfire）、本地 SD WebUI（`local-sd`）、本地 ComfyUI（`comfyui`） |
| **视频** | MiniMax（本地 H3 改 baseUrl 指向 localhost 同一条路）、火山引擎/Seedance、Vidu、阿里 |
| **TTS** | MiniMax、CosyVoice、本机薄封装 `local-tts`（edge-tts 在线音色 / Windows SAPI 离线） |

> 注册表的唯一真源是 `backend-py/app/services/adapters/registry.py`（四个 `*_adapters` 字典）；
> 上表是它的镜像，**新增厂商只改那里**。

---

## 🚀 快速开始

### 📋 环境要求

| 软件 | 版本要求 | 说明 |
|---|---|---|
| **Node.js** | 22.x（推荐 22.20.0） | **仅前端**（Nuxt 构建 / dev） |
| **npm** | 9+ | 前端包管理 |
| **Python** | 3.12（`backend-py/.venv` 已备好） | **后端**（FastAPI + SQLAlchemy，唯一后端） |
| **FFmpeg** | 4.0+ | 视频处理（**必需**） |

#### 安装 FFmpeg

**macOS:**

```bash
brew install ffmpeg
```

**Ubuntu/Debian:**

```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
从 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载并配置环境变量

验证安装：

```bash
ffmpeg -version
```

### ⚙️ 配置文件

复制并编辑配置文件：

```bash
cp configs/config.example.yaml configs/config.yaml
```

配置文件格式（`configs/config.yaml`）：

```yaml
app:
  name: "Drama Studio API"

storage:
  type: "local"
  local_path: "./data/static"
  base_url: "http://localhost:5790/static"

ai:
  default_text_provider: "openai"
  default_image_provider: "openai"
  default_video_provider: "doubao"
```

> **说明**：AI 服务的具体 API Key 和模型参数在 Web 界面的「设置」页面中配置。
>
> **配置优先级**：环境变量 > `config.yaml` > 默认值。常用环境变量：`PORT`、`HOST`、`CORS_ORIGINS`（逗号分隔）、`DB_PATH`、`STORAGE_PATH`、`CONFIG_PATH`（自定义配置文件路径）。

### 📥 安装依赖

```bash
# 获取源码（git clone 或下载解压后进入项目根目录）
# git clone <你的仓库地址>
cd darme-voide

# 安装后端依赖（Python 3.12，与 Dockerfile 的 python:3.12-slim 同基线；
# Node 后端已于 2026-09-15 删除，没有 npm install 这一步）
cd backend-py
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
cd ..

# 安装前端依赖
cd frontend && npm install && cd ..
```

### 🎯 启动项目

#### 方式一：开发模式（推荐）

前后端分离，支持热重载：

```bash
# 终端1：启动后端（Python）
cd backend-py
.venv\Scripts\python.exe -m uvicorn app.main:app --port 5790

# 终端2：启动前端
cd frontend
npm run dev
```

- 前端地址: `http://localhost:3013`
- 后端 API: `http://localhost:5790/api/v1`
- 前端自动代理 `/api` 和 `/static` 到后端

#### 方式二：单服务模式

后端同时提供 API 和前端静态文件：

```bash
# 1. 构建前端
cd frontend && npm run generate

# 2. 启动后端（单服务：同时提供 API 与前端静态文件）
cd ../backend-py && .venv\Scripts\python.exe -m uvicorn app.main:app --port 5790
```

访问: `http://localhost:5790`

### 🗄️ 数据库

数据库表在首次启动时自动创建，无需手动迁移。默认路径 `data/db/drama.db`（2026-11-07 从 `data/` 根下
迁入 `db/`，旧位置会在启动时自动搬过来），可通过环境变量覆盖：

```bash
DB_PATH=/path/to/your.db .venv\Scripts\python.exe -m uvicorn app.main:app --port 5790   # 在 backend-py/ 下
```

---

## 📦 部署指南

### 🐳 Docker 部署（推荐）

#### 方式一：Docker Compose（推荐）

```bash
# 启动服务
docker compose up -d

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

#### 方式二：Docker 命令

```bash
# 从 Docker Hub 运行
docker run -d \
  --name drama-studio \
  -p 5790:5790 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs/config.yaml:/app/configs/config.yaml \
  --restart unless-stopped \
  drama-studio/drama-studio:latest

# 查看日志
docker logs -f drama-studio
```

> **注意**：Linux 用户需添加 `--add-host=host.docker.internal:host-gateway` 以访问宿主机服务

**本地构建**（可选）：

```bash
docker build -t drama-studio:latest .
docker run -d --name drama-studio -p 5790:5790 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs/config.yaml:/app/configs/config.yaml \
  drama-studio:latest
```

**Docker 部署优势：**

- ✅ 开箱即用，内置 FFmpeg 和默认配置
- ✅ 前后端合并为单镜像、单端口
- ✅ 环境一致性，避免依赖问题
- ✅ `data/` 目录 volume 挂载，数据持久化

#### 🔗 访问宿主机服务（Ollama / 本地模型）

容器内可通过 `http://host.docker.internal:端口号` 访问宿主机服务。

**配置步骤：**

1. 宿主机启动服务（监听所有接口）：

   ```bash
   export OLLAMA_HOST=0.0.0.0:11434 && ollama serve
   ```

2. 在 Web 界面「设置 → AI 服务配置」中填写：
   - Base URL: `http://host.docker.internal:11434/v1`
   - Provider: `openai`
   - Model: `qwen2.5:latest`

---

### 🏭 传统部署方式

```bash
# 1. 构建前端
cd frontend && npm run generate && cd ..

# 2. 启动后端（Python；默认 5790，会直接服务 frontend/dist）
cd backend-py && .venv\Scripts\python.exe -m uvicorn app.main:app --port 5790
```

需要上传到服务器的文件：

```
backend-py/       # 唯一的后端（Python；Node 后端已于 2026-09-15 删除）
frontend/dist/    # 前端构建产物
configs/config.yaml
data/             # 数据目录（首次运行自动创建）
backend-py/app/skills/           # Agent 技能文件
```

#### Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5790;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

---

## 🎨 技术栈

### 后端

- **运行时**: Python 3.12
- **Web 框架**: FastAPI（+ SQLAlchemy Core）
- **数据库**: SQLite（WAL；`database.path` / `DATA_ROOT` 可切）
- **AI Agent**: 自研 Agent 运行时 + OpenAI 兼容协议（`backend-py/app/agent/`）
- **视频处理**: FFmpeg（CLI：合成 / 拼接 / 校色 / 抽帧 / 连续性 QC）
- **图片处理**: FFmpeg（**刻意不依赖 Pillow/Sharp**，见 `backend-py/README.md`）

### 前端

- **框架**: Nuxt 3 (SPA 模式)
- **语言**: Vue 3 + TypeScript
- **路由**: 文件路由 (Vue Router 4)
- **样式**: 纯 CSS + CSS Variables (暗色主题)
- **图标**: Lucide Vue

---

## 📝 常见问题

### Q: Docker 容器如何访问宿主机的 Ollama？

A: 使用 `http://host.docker.internal:11434/v1` 作为 Base URL。注意：
1. 宿主机 Ollama 需监听 `0.0.0.0`：`export OLLAMA_HOST=0.0.0.0:11434 && ollama serve`
2. Linux 用户使用 `docker run` 需添加：`--add-host=host.docker.internal:host-gateway`

### Q: FFmpeg 未安装或找不到？

A: 确保 FFmpeg 已安装并在 PATH 环境变量中。运行 `ffmpeg -version` 验证。Docker 部署已内置 FFmpeg。

### Q: 前端无法连接后端 API？

A: 检查后端是否启动，端口是否正确。开发模式下前端代理配置在 `frontend/nuxt.config.ts`。

### Q: 数据库表未创建？

A: 后端会在首次启动时自动创建所有表，检查日志确认初始化是否成功。

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

常用检查命令：

```bash
cd backend-py && .venv\Scripts\python.exe tests\run_all.py    # 后端自检（套件/项数以 tests\run_all.py 的 TESTS 为准）
cd ../frontend && npm run build
```

---


> _"让 AI 帮我们做更有创造力的事"_

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star！**

