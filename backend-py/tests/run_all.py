"""跑完全部后端自检（每迁完一块请跑这个）。

**本文件的 ``TESTS`` 就是套件权威清单**（README 里那棵树只是摘录，别去数它）。

规模（2026-09-17 实测）：**82 个套件 / 2911 项断言**（82 套 / 2911 项 / 0 失败 ✓
（m1 0-40 → 1822 项、m2 40-60 → 582 项、m3 60-85 → 507 项，三批 `_BAD=0` ✓，范围互不重叠 ✓；
与 `run_all.TESTS` 的权威套件数 **82** 一致 ✓；本地服务全停时测得 ✓）。
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
    ("自研引擎·算法层（σ 调度 / 帧网格与像素预算 / 采样循环；零依赖）", "engine_core_test.py"),
    ("自研引擎·权重体检（纯 Python 读 safetensors + 就绪报告；零依赖）", "engine_inventory_test.py"),
    ("自研引擎·管线编排（阶段/事件/取消/错误归因 + 干跑后端；零依赖）", "engine_pipeline_test.py"),
    ("自研引擎·加载计划（量化配套/层号连续性/显存排班；零依赖）", "engine_loader_test.py"),
    ("safetensors 交叉验证（纯 Python 读取器 vs 官方库；缺库则显式 SKIP）", "safetensors_crosscheck_test.py"),
    ("自研引擎·真模型层（DiT 前向/条件生效/权重往返/差异报告；CPU 可验）", "engine_dit_test.py"),
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


def main() -> int:
    here = Path(__file__).resolve().parent
    failures: list[str] = []

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
