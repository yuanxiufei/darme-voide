"""跑完全部后端自检（每迁完一块请跑这个）。

**本文件的 ``TESTS`` 就是套件权威清单**（README 里那棵树只是摘录，别去数它）。

规模：**以本文件下面的 `TESTS` 为唯一权威** ✓（⚠️ 不再在这里写死总数 ✗ —— 逐项罗列会随
增删而腐烂 ✓，本文件自己就被它咬过：下面那行曾是 **82 套 / 2911 项** ✗，早过期好几轮 ✓）。
最近一次**全量实测**：**120 套 / 3912 项 / 0 失败**（2026-09-24 ✓：**119 套按 `SUMMARY:` 收敛出
3912 项** ✓、另 1 套是常量守卫型（`OK: 镜像常量漂移 0 条` ✓ 无项数 ✓）+ `skip 7` ✓。
⇒ 本轮较上一跑 **+1 套 / +15 项** ✓（= 配音契约 `voice_contract` 15 条 ✓）。⚠️ **记账须与实测
对得上** ✗：我曾在日志把 `engine_cache_guard` 写成 **12/12** ✓✗，实测是 **20/20** ✓ 已改正 ✓。
⚠️ 上一段括号里的 104/3629 是**更早一跑**的细节 ✓（数字真 ✓ 但别与本行混读 ✗）；本轮新增见下一段 ✓）。
⭐⭐ 修掉一条**产品侧竞态** ✓ ——
H3 服务此前是「**先写 `succeeded`** ✓、再在 `finally` 里 `/free` + 写 `freed_vram`」✗✗ ⇒ 中间那一瞬
调用方读到的是「成功了但 `freed_vram=null`」✓✗（API **自相矛盾** ✓）。⇒ 现在**先卸载、再落终态** ✓
（终态与 `freed_vram` 在**同一次 `_update`** 里落地 ✓，两条执行体都改了 ✓）；同轮补判据：失败路径也
`/free` ✓（`h3_stage2_test` 20 → 21 项 ✓）+ `dit.H3_PACK_FACTS` 两条「给错也不报错」的事实的**逐值**判据 ✓
（**跨度表 (1,4,4,4,4) 循环** ✓ / 立体声 **channel-major 的落点是 `w` 两端** ✓）+
**混合形态的模态标签** ✓ 与三条「混了不报错」的**网格规则** ✓ —— 104 套按 `SUMMARY:` 收敛出 3629 项 ✓，
另 1 套是常量守卫型（打印 `OK: 镜像常量漂移 0 条` ✓ 无项数 ✓））。
⭐ 2026-09-24 新增**六套**（都是「逆向出来的可用项」落地 ✓）：加速链 **30 条** ✓ + H3 prompt 契约
**37 条** ✓ + 权重守卫（内嵌元数据契约 ✓ + 未实现布局族**具名**拒绝 ✓）**18 条** ✓ + 导演稿分段
**26 条** ✓ + 联合 AV 潜变量容器（鸭子类型 ✓ 零依赖 ✓）**16 条** ✓ + 超清模式规划（1.5× 先 2× ✓ /
4× 小分块 ✓）**16 条** ✓ + 混合加载计划（fl2va 基底 + ref2va 的 adaLN 覆盖 ✓）**11 条** ✓ +
档位表（步数 / 分辨率 / 加速件 / 显存建议 / **必备模型** ✓）**16 条** ✓ +
两条**接缝**（加速链 ↔ 真 `/object_info` ✓ 8 条 / H3 prompt 提交前那道关 ✓ 10 条 ✓）✓
⇒ `TESTS` 登记 105 → **115** ✓（**登记数是数出来的** ✓；项数以「最近一次全量实测」为准 ✓ **不推算** ✗）。
⚠️ 上一轮：105 套 / 3623 项 ✓（H3 混合打包形态 ✓ + 掩码坐标空间补注 ✓）
—— 按规则**不推算** ✗，数字取自刚跑出来的汇总 ✓。
⚠️⚠️ **并发跑全量会让个别套件假红** ✗（2026-09-22 实测 ✓✗）：两个回归同时在跑时
`h3_stage2_test` 报 `19/20` ✓✗（单独复跑**两次都 20/20** ✓ ⇒ 当时判成「资源竞争不是代码」✗）。
⚠️ **2026-09-24 更正** ✗：那**不是**并发的问题 ✗✗ —— 是**产品侧的终态竞态** ✓（`h3` 服务先写
`succeeded` ✓ 再 `/free` ✓ ⇒ 中间态「成功但 `freed_vram=null`」✓✗）；并发/负载只是**让它更容易露** ✓。
⇒ 已修（先卸载再落终态 ✓）并加了判据 ✓；⭐ 教训：**「单独复跑就好了」≠「不是代码问题」** ✗✗，
它可能只是**概率**问题 ✓ —— 该做的是**问"什么条件下会露"** ✓，而不是收工 ✓。
⇒ 判据：① 看有没有 `结论：` 行 ✓；② 有红先数进程 ✓（>1 就清 ✓）③ **单独连跑 3 次** ✓
（一次就绿说明不了什么 ✗）—— 仍绿才去看代码里"什么时候写终态"这类**时序** ✗。
⚠️⚠️ **跑之前先把 ffmpeg 的真实 bin 目录「前置」到 PATH** ✗（2026-09-22 实测踩到 ✓✗）：
Windows 上 WinGet 装的 ``…\\Microsoft\\WinGet\\Links\\ffmpeg.exe`` 是**应用别名（重解析点）** ✗ ——
在本进程里可能 `lexists=True` 但 `exists=False` ✓✗，于是两条路一起坏：
① `shutil.which("ffmpeg")` 返回 `None` ✓（引擎侧会说「找不到 ffmpeg ✗」✗ —— 其实装了 ✓）；
② **裸名** spawn（`subprocess.run(["ffmpeg", …])` ✓ 产品代码那种写法 ✓）直接
`OSError: [WinError 448] 无法遍历该路径，因为它包含不受信任的装入点` ✗✗
⇒ `image_generation` / `consistency_qc` / `technical_qc` / `color_grade` / `compressed_data_url`
**5 套会红** ✓，看着像代码坏了 ✗✗。⇒ 对策：把
``…\\WinGet\\Packages\\Gyan.FFmpeg_…\\ffmpeg-*-full_build\\bin`` **前置**到 `PATH` ✓
（**前置**才行 ✗ —— 追加只修好 ① ✗，② 仍会命中坏别名 ✓；判据是 `where.exe ffmpeg` 指到真实 bin ✓）。
⚠️ 前端两条自检与**构建命令**耦合 ✗（2026-09-22 踩过 ✓）：`frontend/.output/public/index.html` 是
`npm run **generate**` 的产物 ✓（Dockerfile 用的也是它 ✓）⇒ 只跑 `npm run **build**` 会把它**覆盖掉** ✓✗
（`dockerfile_contract_test` 当场红 ✓）⇒ 改完前端**该跑的清单**：`typecheck` ✓ + `generate` ✓。
⚠️⚠️ **别同时开多个全量回归** ✗：并发会互相抢 CPU（表现为"卡在某套很久" ✓✗）；
判据是日志里有没有 `结论：` 行 ✓ —— **半截日志不算跑过** ✗（2026-09-21 实测踩到 ✓）。
⚠️ 自检**汇总行格式有硬要求** ✗：必须是 `SUMMARY: n/m passed` ✓（`run_all.py` 按这个前缀收敛项数 ✓）
—— 写成「n/m 项通过」会在总表里显示**空摘要** ✓✗（2026-09-20 实测踩过 ✓ 已修 4 个套件 ✓）。
⚠️ 与上一轮（78 套 / 2806 项）**独立互证** ✓：2806 + 11（`safetensors_crosscheck_test` ✓）
+ 8（`engine_pipeline_test` 90 → 98 ✓）+ 26（`engine_dit_test` ✓）+ 39（`engine_io_test` ✓）
+ 21（`engine_text_test` ✓）= **2911** ✓✓（两条路径同一个数 ✓）。
⚠️ **批次日志标签不可复用** ✗：`runbatch.py` 按标签写 `tmp/<label>.txt` ✓ ⇒ 复用旧标签会读到
**上一次会话的陈旧结果** ✓（2026-09-17 实测踩到过一次 ✓）⇒ 每次换新标签 ✓。
⚠️ 其后又**新增** `engine_segments_test`（长视频分段 ✓ 54 项 ✓）⇒ 现为 **83 套**、总数**待实测** ✗
（按规则**不推算** ✗ —— 跑一次全量即可补齐 ✓）。
那一轮的完整复测**没跑成** ✗（环境连跳多次 ✗）⇒ 按规则**不推算总数** ✗：
本轮改为逐个实测**受影响的**套件（管线 98/98 ✓、交叉验证 11/11 ✓、加载计划 28/28 ✓、
体检 34/34 ✓、算法 26/26 ✓）✓。**下次跑一次全量即可补齐** ✓。
⚠️ 每轮都与上一轮**独立互证** ✓（两条路径得出同一个数 ✓）：2762 − 46 + 65（引导 ✓）− 65 + 82
（首帧条件 ✓）− 82 + 90（torch 依赖闸门 ✓）= **2806** ✓✓。
⚠️ **活体类套件的项数随本地服务在否浮动** ✗：全停时 `local_services_live_test.py` 记 9 项、
`h3_backend_live_test.py` 记 0 项（都显式 SKIP ✓）；把 H3 薄封装（8765）起来后变 14 项与 8 项 ✓。
⇒ **别把数字当硬约束，自己跑一遍看汇总** ✓。
⚠️ **不做算术推算** ✗（09-16 按「2526+8」算错过 13 项 ✓ 教训在案）⇒ 一律以**刚跑出来的汇总**为准 ✓。
⚠️ **活体类套件的项数随本地服务在否浮动** ✗：全停时 `local_services_live_test.py` 记 9 项、
`h3_backend_live_test.py` 记 0 项（都显式 SKIP ✓）；把 H3 薄封装（8765）起来后变 14 项与 8 项
⇒ 合计 **+13**（这个差也已实测过 ✓）⇒ **别把数字当硬约束，自己跑一遍看汇总** ✓。
跳过条数一律显式打印（`（skip n：服务未启动）`）⇒ **不要把这几个数字当硬约束，自己跑一遍看汇总** ✓）（删 `backend/` 后实测；比删库前少 2 项
**显式 `[skip]`**，见 README；另**新增 2 个套件**，都是「删库后 Python 成为唯一后端」才存在的不变量：
``frontend_api_coverage_test.py``（前端每个调用点都在后端路由表里）与 ``contract_mirror_test.py``
（``frontend/app/types/contracts.ts`` 的字段在后端源码里都能找到 —— 共享契约只许单向同步：后端改 ⇒ 镜像跟）。各套件**各自独立进程**、都不碰真实库：

* 契约冒烟 ``smoke_test.py``（跑在**数据库副本**上，覆盖每一条已注册端点）
* 纯逻辑/适配层：``adapters_test.py``、``vendor_errors_test.py``（``MockTransport``，零真实网络）、
  ``text_generation_test.py``、``prompt_storyboard_test.py``、``grid_prompt_test.py`` 等
* 迁移域回归：各 ``*_route_test.py`` / ``*_generate_test.py``（事件与落盘都打桩）
* 机械守卫：``route_parity_test.py``（路径 0 遮蔽 + 九道镜像常量漂移）、
  ``freeze_snapshot_test.py``（TS 快照反漂移）、``parity_diff_test.py``（Node↔Python 差分比较器）
* ⚠️ 少数套件会用 **ffmpeg 现场造真实媒体**（``consistency_qc_test.py`` / ``color_grade_test.py`` /
  ``compressed_data_url_test.py``）⇒ 机器上没有 ffmpeg 时这几套会失败，属环境依赖（见 README）。

用法::

    ./.venv/Scripts/python.exe tests/run_all.py          # 全量
    ./.venv/Scripts/python.exe tests/<某个>_test.py      # 单跑（改哪儿跑哪儿）

⚠️ 单跑很快，全量偏慢（分钟级）；若被上层环境截断，可按 ``run_all.TESTS[a:b]`` 切片分批跑，
   结论等价（本项目实测过：三批 = 61 套件一次全绿）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TESTS = [
    ("契约冒烟", "smoke_test.py"),
    ("适配器层", "adapters_test.py"),
    ("厂商错误归因", "vendor_errors_test.py"),
    ("文本生成", "text_generation_test.py"),
    ("图片生成链路", "image_generation_test.py"),
    ("视频生成链路", "video_generation_test.py"),
    ("TTS / 音色复刻", "tts_generation_test.py"),
    ("分镜 prompt + 图谱", "prompt_storyboard_test.py"),
    ("宫格 prompt + 运镜", "grid_prompt_test.py"),
    ("逐镜路由 + videos", "videos_route_test.py"),
    ("单镜合成 compose", "compose_test.py"),
    ("整集拼接 merge", "merge_test.py"),
    ("宫格 prompt/切分 grid", "grid_route_test.py"),
    ("图谱/图片/回调 misc", "misc_routes_test.py"),
    ("AI 音色 ai-voices", "ai_voices_test.py"),
    ("预设框架 preset-framework", "preset_framework_test.py"),
    ("Agent 协议/工具基座", "agent_protocol_test.py"),
    ("Agent 工具集", "agent_tools_test.py"),
    ("Agent 音色工具", "agent_voice_tools_test.py"),
    ("Agent 剧本/语料工具", "agent_script_corpus_test.py"),
    ("Agent 提取工具", "agent_extract_tools_test.py"),
    ("Agent 分镜工具", "agent_storyboard_tools_test.py"),
    ("Agent 运行时", "agent_runtime_test.py"),
    ("Agent 聊天域", "agent_route_test.py"),
    ("MCP 接入层 + 路由", "mcp_test.py"),
    ("Agent 出厂提示词", "agent_prompts_test.py"),
    ("评测打分器 + 基准目录", "evaluation_scorer_test.py"),
    ("Skill 解析 + 加载", "skills_test.py"),
    ("评测执行器 + 端点", "evaluation_route_test.py"),
    ("基准资产搬迁 + 评测 CLI", "eval_cli_test.py"),
    ("GPU 显存管理器 + 2 端点", "gpu_manager_test.py"),
    ("GPU 租约接线（text / tts + image/video 长租约）", "gpu_lease_wiring_test.py"),
    ("差分对拍比较器（Node↔Python 工具）", "parity_diff_test.py"),
    ("多集节奏相位 + 端点", "rhythm_phase_test.py"),
    ("镜头 QC 打分 + 端点", "qc_scoring_test.py"),
    ("审片重跑闭环 + 端点", "qc_retry_test.py"),
    ("设置分镜首尾帧 + 抽帧泛化", "set_frame_test.py"),
    ("重生成镜头帧 + 端点", "regenerate_frame_test.py"),
    ("图像连续性 QC + 端点", "consistency_qc_test.py"),
    ("技术维度 QC + 接线", "technical_qc_test.py"),
    ("宫格 Agent 提示词 + 端点", "grid_agent_prompt_test.py"),
    ("像素级校色（ffmpeg）", "color_grade_test.py"),
    ("子 Agent 调度工具", "subagent_test.py"),
    ("请求日志中间件", "http_logger_test.py"),
    ("参考图压缩（ffmpeg）", "compressed_data_url_test.py"),
    ("TS 源码快照反漂移", "freeze_snapshot_test.py"),
    ("Agent 创建器 + generate", "creator_test.py"),
    ("提示词优化器 + optimize", "optimizer_test.py"),
    ("评测调度器 + 端点", "evaluation_scheduler_test.py"),
    ("SSE 总线 + 尾帧提取", "sse_hub_frames_test.py"),
    ("全自动管线 + SSE 端点", "auto_pipeline_test.py"),
    ("本地模型扫描 + 11 端点", "local_models_test.py"),
    ("角色生成链路 8 端点", "characters_generate_test.py"),
    ("物品/场景出图 2 端点", "props_scenes_generate_test.py"),
    ("分镜 TTS/出图/LLM 5 端点", "storyboards_generate_test.py"),
    ("剧集续写端点", "episodes_continue_script_test.py"),
    ("导出服务 + EDL/ZIP 2 端点", "export_service_test.py"),
    ("剪映草稿导出", "jianying_draft_test.py"),
    ("QC 报告 + 联系表 2 端点", "qc_report_test.py"),
    ("时代背景提炼 + 风格提炼 2 端点", "era_style_distill_test.py"),
    ("数据根切换 + 存储 2 端点", "storage_change_test.py"),
    ("生产镜像布局一致性", "dockerfile_contract_test.py"),
    ("前端调用点 ↔ 后端路由覆盖", "frontend_api_coverage_test.py"),
    ("共享契约镜像（contracts.ts ↔ 后端）", "contract_mirror_test.py"),
    ("层级守卫（app/ 内单向依赖）", "layering_test.py"),
    ("资产布局守卫（三类资产在 app/ 内 + 常量权威）", "assets_layout_test.py"),
    ("本地 H3 链接缝契约（适配器 ↔ 8765 薄封装）", "h3_chain_test.py"),
    ("本地服务活体接缝（ollama 真推理；其余按设计 SKIP）", "local_services_live_test.py"),
    ("H3 全栈活体（配置 → 服务层 → 真 8765 → 落库 + task_id 交叉证明）", "h3_backend_live_test.py"),
    ("CosyVoice 接缝契约（JSON↔form/multipart、裸 PCM→WAV；真 HTTP stub 上游）", "cosyvoice_seam_test.py"),
    ("ComfyUI 能力（客户端 + UI→API 工作流转换；真 HTTP stub ComfyUI）", "h3_comfyui_test.py"),
    ("H3 阶段2 闭环（body → 组装/注入 → 提交 → 轮询 → 取片 → /files → /free）", "h3_stage2_test.py"),
    ("ComfyUI 能力门面（系统/目录/队列/历史/作业/媒体 + 通用工作流）", "comfyui_capability_test.py"),
    ("自研引擎·加速链（配置解析 + 提交前校验 + 接线；零依赖）", "engine_accel_chain_test.py"),
    ("自研引擎·H3 prompt 契约（声明 / 跳过判据 / 标签 / 任务选择；零依赖）",
     "engine_conditioning_test.py"),
    ("自研引擎·权重守卫（内嵌元数据契约 + 未实现布局族具名拒绝；零依赖）",
     "engine_checkpoint_guards_test.py"),
    ("自研引擎·导演稿分段（5 种标记 / 4.5 字每秒 / 断点优先级 / ≤15s 合并；零依赖）",
     "engine_script_parse_test.py"),
    ("自研引擎·联合 AV 潜变量容器（认出视频流 / 保类型换流；鸭子类型、零依赖）",
     "engine_latent_container_test.py"),
    ("自研引擎·超清模式规划（契约校验 / 1.5×先 2× / 4× 小分块；零依赖）",
     "engine_upscale_test.py"),
    ("自研引擎·超清放大器网络（V2 主干 + V3 因子化注意力；T 不变 / 残差恒等 / 严格装载）",
     "engine_upscale_net_test.py"),
    ("自研引擎·段级音频合成（长度守恒 / 偏移 / 裁剪 / 三支削波口径；纯样本、零依赖）",
     "engine_audio_mix_test.py"),
    ("自研引擎·缓存完整性守卫（头自洽 / 偏移吃满 / 头哈希指纹 / 源数对齐；零依赖）",
     "engine_cache_guard_test.py"),
    ("自研引擎·参考素材指纹（顺序无关 / 两种形态都认 / 采样率入指纹 / 出错退化成不相等）",
     "engine_cache_key_test.py"),
    ("配音契约（8 维定序 / 单值↔预设 ↔ 向量 / 未知情绪拒 / 不归一化 / 语速不钳位；零依赖）",
     "voice_contract_test.py"),
    ("自研引擎·混合加载计划（fl2va 基底 + ref2va 的 adaLN 覆盖；零依赖）",
     "engine_hybrid_merge_test.py"),
    ("自研引擎·档位表（步数 / 分辨率 / 加速件 / 显存建议 / 必备模型；零依赖）",
     "engine_tiers_test.py"),
    ("H3 加速链接缝（链校验 + 接线 → 真 /object_info；假客户端、零网络）",
     "h3_accel_wiring_test.py"),
    ("H3 prompt 接缝（全局提示词 + 标签转换 + 编号校验 + 声明追加）",
     "h3_prompt_wiring_test.py"),
    ("自研引擎·算法层（σ 调度 / 帧网格与像素预算 / 采样循环；零依赖）", "engine_core_test.py"),
    ("自研引擎·权重体检（纯 Python 读 safetensors + 就绪报告；零依赖）", "engine_inventory_test.py"),
    ("自研引擎·管线编排（阶段/事件/取消/错误归因 + 干跑后端；零依赖）", "engine_pipeline_test.py"),
    ("自研引擎·加载计划（量化配套/层号连续性/显存排班；零依赖）", "engine_loader_test.py"),
    ("自研引擎·GGUF 读取器（头格式/张量表/截断检测/量化方案名；零依赖）", "engine_gguf_test.py"),
    ("自研引擎·H3 键名核对器（键全集/形状关系/PDD 头库/curve 变体；零依赖 torch-free）", "engine_h3_keys_test.py"),
    ("自研引擎·BPE 分词器（真词表 tokenizer.json/vocab+merges；往返恒等；零依赖离线）", "engine_tokenizer_test.py"),
    ("自研引擎·反量化（fp8/int8 × 四种布局 × 两种 scale 方向；判不出来就拒绝；接进装载）", "engine_quant_test.py"),
    ("自研引擎·分词器总入口（形态嗅探/自研优先/参考回退/批量+LRU/离线开关；互校）", "engine_tokenizer_hub_test.py"),
    ("自研引擎·自研 Unigram/WordPiece/Metaspace（Viterbi/fuse_unk/整词 UNK；与参考逐例同 id）",
     "engine_tokenizer_own_test.py"),
    ("自研引擎·transformers 运行时改造（离线兜底含 Auto 工厂/缓存目录/降噪/计数/可撤；幂等）",
     "engine_tokenizers_tuning_test.py"),
    ("自研引擎·上机前自检入口（依赖/权重就绪/加载计划/真权重预检 + 词表；JSON 与退出码）",
     "engine_readiness_script_test.py"),
    ("safetensors 交叉验证（纯 Python 读取器 vs 官方库；缺库则显式 SKIP）", "safetensors_crosscheck_test.py"),
    ("自研引擎·真模型层（DiT 前向/条件生效/权重往返/差异报告；CPU 可验）", "engine_dit_test.py"),
    ("自研引擎·H3 形态积木（RMSNorm / SwiGLU / 18 路 adaLN / 正弦时间嵌入 / RoPE 旋转不变量 / "
     "双 fp32 输出头 / PDD 头库 —— 已接进 `TorchBackend` 双流路径 ✓ `H3_FORM_TODO` 已清零 ✓）",
     "engine_h3_form_test.py"),
    ("自研引擎·音频 VAE（32 kHz 立体声 ⇄ 潜变量；**真写 wav + 标准库读回核对** ✓）",
     "engine_audio_vae_test.py"),
    ("自研引擎·H3 双流接进管线（**真 mp4 + 真 wav**；参考块四类入口 ✓ / 能力自述逐条报缺 / "
     "取整口径必填 / 当场拒绝 / 单流默认路径一字未动 ✓）", "engine_dual_stream_test.py"),
    ("自研引擎·解码与落盘（VAE 编解码 + **真 mp4/wav 用 ffprobe/标准库复核** + 整链出片）", "engine_io_test.py"),
    ("自研引擎·文本编码（真 TE + 注入式 tokenizer + 截断回报 + 整链 TE→DiT→VAE→mp4）", "engine_text_test.py"),
    ("自研引擎·长视频分段（网格长度/重叠接缝/保留帧守恒 + 首帧落 PNG 传递）", "engine_segments_test.py"),
    ("自研引擎·命名映射（预设改名/合并顺序/**干跑报告**缺哪些键 + 往返装载逐位一致）", "engine_mappings_test.py"),
    ("短剧提示词生产契约（占位符**解析成具体内容** + Mx-Shell 五段质感层；零依赖）",
     "prompt_contract_test.py"),
    ("资产清单 + 验收门（依赖拓扑/成环点名/逐镜阻断/**付费生成前的门**；零依赖）",
     "asset_gate_test.py"),
    ("连续性表（§6-§11：道具时间线/越轴/线索提前暴露/转场动机/可删动作；零依赖；**事前**）",
     "continuity_test.py"),
    ("Agent 上下文预算（估算/机械压缩/**tool 往返成对**/摘要交给调用方；零依赖）",
     "agent_context_test.py"),
    ("开跑前体检（四块串成一次调用：占位符→质感→连续性→验收门 + 按成本排序的行动项）",
     "preflight_test.py"),
    ("请求体守卫（`read_json` 已保证 dict ⇒ 路由里的 isinstance 判断是**死代码**；AST 检出）",
     "read_json_guard_test.py"),
    ("分镜→连续性适配器（只映射真实列、**不补默认值** + 契约字段的 schema 缺口清单）",
     "storyboard_continuity_test.py"),
    ("体检取数（按 episodeId 组装；逐镜 asset_status→逐资产验收 + 空集**不给绿灯**）",
     "preflight_source_test.py"),
    ("连续性写入（词汇**拒收而非忽略** / 幂等替换 / **全有或全无** / 归属校验；临时库 ✓）",
     "continuity_store_test.py"),
    ("生产链端到端（真工具写状态 → 取数 → 适配 → 判定 → 门；**填齐就当绿** ✓ 临时库 ✓）",
     "chain_e2e_test.py"),
    ("路径 + 常量守卫", "route_parity_test.py"),
]


def syntax_problems(root: Path) -> list[str]:
    """**AST 预检**整棵源码树 ✓ ⇒ ``["相对路径:行 说明", …]`` ✓（空 = 全过 ✓）。

    ⚠️⚠️ 为什么要它 ✗（2026-09-20 加 ✓）：那一天里「中文文案里嵌了半角双引号 ``"``」把
    `SyntaxError` 带进来**四次** ✓✗（本仓 MEMORY 早写着这条 ✓ 我还是犯了 ✓）。那种错的**代价很阴** ✗：
    套件**连一行都不输出** ✓ ⇒ 表现只是"没有 SUMMARY"✓，看起来像"被判后台 / 输出被吞"✓✗
    —— 每次都得另外跑一次才看出来 ✓。⇒ 开跑前统一 parse ✓，一有问题**当场点名 `文件:行`** ✓
    （"响亮"而不是静默 ✓ —— 本仓那条判据 ✓）。
    """
    import ast  # noqa: PLC0415 —— 只在预检用 ✓

    bad: list[str] = []
    for path in sorted(Path(root).rglob("*.py")):
        if {".venv", "__pycache__"} & set(path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), str(path))
        except SyntaxError as err:
            bad.append(f"{path.relative_to(root)}:{err.lineno} {err.msg}")
    return bad


def main() -> int:
    here = Path(__file__).resolve().parent
    failures: list[str] = []

    syntax_bad = syntax_problems(here.parent)
    if syntax_bad:
        for item in syntax_bad:
            print("SYNTAX " + item)
        print(f"\nSYNTAX 预检失败：{len(syntax_bad)} 个文件有语法错 ✗（先修它们再跑套件 ✓）")
        return 1

    # ⚠️ Windows 上的双坑（都踩过）：
    # 1. 子进程 stdout 是**管道**时，Python 按**本地代码页（GBK）**输出 ⇒ 若按 UTF-8 解码
    #    会得到 U+FFFD，再打印回 GBK 控制台直接 UnicodeEncodeError 崩掉。
    #    这里给子进程强制 PYTHONIOENCODING=utf-8，两边编码就对齐了。
    # 2. 自身 stdout 也加 errors="replace"，保证任何字符都不会让汇总步骤崩。
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover
        pass

    for label, filename in TESTS:
        completed = subprocess.run(
            [sys.executable, str(here / filename)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(here.parent),
            env=child_env,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        summary = ""
        for line in output.splitlines():
            if line.startswith("SUMMARY:") or line.startswith("OK:") or line.startswith("FAIL:"):
                summary = line.strip()
        status = "OK  " if completed.returncode == 0 else "FAIL"
        print(f"{status}  {label:<16} {summary}")
        if completed.returncode != 0:
            failures.append(label)
            # 失败时把 FAIL 行打出来，方便直接定位
            for line in output.splitlines():
                if line.startswith("FAIL") or line.startswith("  DRIFT") or line.startswith("  SHADOW"):
                    print(f"        {line.strip()}")

    print()
    if failures:
        print(f"结论：{len(failures)} 项未通过 -> {'、'.join(failures)}")
        return 1
    print(f"结论：{len(TESTS)} 项自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
