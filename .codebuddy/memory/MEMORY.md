# MEMORY — 长期记忆

## 项目
Drama Studio（`darme-voide`）：AI 剧本/分镜/视频生成。Nuxt 3 前端（`frontend/app`）+ Hono 后端（`backend`），TypeScript 全栈。

## 关键端口 / 环境
- 后端 Hono + tsx，默认端口 5789（`config.ts`，`PORT` 环境变量可覆盖）；前端 Nuxt 3 默认 3013（`nuxt.config.ts` proxy `/api`、`/static` → `localhost:5789`）。
- 无登录页；前端为 Nuxt 页面路由：`/`（剧集列表）、`/settings`、`/drama/[id]`、`/library/*`（characters/scenes/costumes/weapons）。
- 本地 SQLite：默认数据根目录 `data/`（`drama.db` + `static/`），优先级 `.data-root` 标记文件 > `DATA_ROOT` > config.yaml `database.path` 目录 > `./data`。
- Docker 未装/不在 PATH；不依赖 postgres/redis/qdrant。

## 本地模型「下载+管理」闭环（阶段 1–7，已完成）
- Web UI 后端 `backend/src/routes/localModels.ts` + `aiConfigs.ts`；前端 `settings.vue` + `useApi.ts`。
- CLI `scripts/model_manager.py` + `configs/models.json`（15 模型 + 4 节点）。
- 能力：扫描、删除（Ollama/本地文件）、下载（HF / hf-mirror / ModelScope 三源）、断点续传（`.part` + Range）、跳过已存在、ModelScope 签名 URL 兜底。
- 三源默认分支：modelscope=`master`，hf 系列=`main`；前端 `hfSource` 持久化 localStorage（key `darme.hfSource`）。
- 全部通过 tsc / lint / py_compile / 后端运行时冒烟四层验证。

## 本地 H3 视频推理（剩余工作，需真实 GPU）
- ComfyUI(8188) 承担 H3 DiT/VAE/text_encoder 显存驻留与计算；`8765` 是推理薄封装（复用 `minimax` adapter，`POST /v1/video_generation` + `GET /v1/video_generation/task/:id`）。
- GPU 卸载走 ComfyUI `POST /free`（`gpu-manager.ts` 中 `unloadBaseUrl` 与 `baseUrl` 分离）。
- 模型扫描：H3 权重判 `runtime='h3'`、`baseUrl='http://localhost:8765'`；provider=`minimax`、model=`hailuo-02`（vram 15GB）。
- 自定义节点包 `ComfyUI-MiniMaxH3` 已装于 `D:/Comfy-Desktop/ComfyUI-Installs/.../custom_nodes/`。
- H3 双路线：FL2VA（动作/空镜/首尾帧）、Ref2VA（参考图/视频/音频）；六键 Bible `CHAR_ID/SPEAKER_ID/VOICE_ID/LOCATION_ID/COSTUME_ID/STYLE_ID` 跨集锁定。
- 方法论文档 `docs/local-h3-video-system.md`。

## H3「Turbo」加速路线（2026-08-31 调研修正）
- Turbo 有两条路线：**Turbo LoRA**（~744MB 适配器挂 H3-Base）vs **lightx2v 蒸馏**（独立模型 ModelTC/Minimax-H3-Turbo）。
- `molbal/MiniMax-H3-Turbo-GGUF` = 蒸馏模型 GGUF（非 LoRA），文件 `minimax_h3_fl2v_turbo_4step_v1.0_768p_{Q4_0 11.4GB / Q8_0 21.4GB / Q8_CR 20.2GB}.gguf`；元数据 `general.architecture` 仍为 `minimax_h3`（与 base 同名）。
- 该 repo README 要求 **ComfyUI GGUF loader（molbal fork）+ ComfyUI-MiniMax-H3-Turbo 采样节点**，未提 sd.cpp。
- sd.cpp 官方 `docs/minimax_h3.md` 只支持 base H3（time-embedder DiT / AdaLN curve-table 权重自动分派），**无证据支持蒸馏 Turbo**。
- 路线判定：A=Turbo 蒸馏 GGUF+sd.cpp（未证实）；B=原版 GGUF（unsloth/leejet）+sd.cpp（4×V100 已验证，20–60min）；C=Turbo LoRA/int8 ComfyUI（最成熟，1–3min）。
- 兼容性实测脚本 `scripts/sd_h3_compat_probe.py`（零依赖，读 GGUF 元数据 → 对照 sd.cpp 支持矩阵 → 可选 `--live` 真机 load）。
- 编排脚本 `scripts/sd_h3_pipeline.py`（doctor/download/build/probe，复用 model_manager 断点续传；Windows 编译 flag `-DSD_CUDA=ON`、`cmake --build --config Release`）。

## 本机 GPU / CUDA 环境实测（2026-08-31）
- GPU：**NVIDIA RTX A5000**，nvidia-smi 报 23028 MiB（≈22 GiB）；驱动 610.88，`CUDA UMD 13.3`。
- PyTorch `2.13.0+cu130`，`cuda_available=True`，cudnn 92000。
- **无 CUDA Toolkit（nvcc 缺失）**：`CUDA_PATH` 环境变量为空、`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` 目录为空、PATH 无 nvcc。
- 关键概念：`nvcc`（CUDA Toolkit 编译器）≠ CUDA runtime（驱动/PyTorch 的 cuXXX pip 包只带 runtime+cudnn，不带 nvcc）。→「torch.cuda 可用」不代表「能源码编译 CUDA 版 sd.cpp」。
- 后果：路线 A/B 的 sd.cpp 需 nvcc 编 CUDA 版，当前只能 CPU 版（极慢）；路线 C（ComfyUI Desktop，本机已在跑、显存 22.4/23 GiB 占用中）不依赖 nvcc，是现成可用路线。

## 用户操作规范
- 上下文过大时必须按阶段拆分、每阶段只读 1 文件只改 1 处、逐步验证，避免一次塞入过多内容。

## Git 提交约定（本机）
- 本机无默认 git user 身份（user.name/email 全局与系统均为空），commit 需临时指定：`git -c user.name=yuanxf -c user.email=yuanxf@wedoctor.com commit ...`。
- PowerShell 传中文 commit message 会乱码（GBK/UTF-8 转换损坏字节），统一用英文 commit message。
- 提交拆分风格：按「整理类 chore / 功能类 feat」分组，每类一个逻辑提交。
