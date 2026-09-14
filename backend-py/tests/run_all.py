"""跑完全部后端自检（每迁完一块请跑这个）。

五个自检各自独立、**都不碰真实库**：

* ``smoke_test.py``          契约冒烟（跑在数据库副本上）
* ``adapters_test.py``       适配器层（纯函数，无需网络/密钥）
* ``vendor_errors_test.py``  厂商错误归因 + 重试（``MockTransport``，零真实网络）
* ``text_generation_test.py``文本生成纯逻辑（提示词 / 规则拆分器）
* ``route_parity_test.py``   路径守卫（0 遮蔽）+ 镜像常量漂移守卫

用法::

    ./.venv/Scripts/python.exe tests/run_all.py
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
