#!/usr/bin/env python3
"""一次性：把既有模型文件（硬链接）迁移到 ComfyUI Desktop 共享库。

> 本文件是 ``scripts/migrate_models.ps1`` 的移植（2026-09-15，PowerShell 版已删）。
> 语义逐条对齐：**硬链接**（不复制字节）、跳过 ``.cache`` 目录、目标已存在先删、
> 源不存在就 SKIP。

⚠️ 路径里的 ``D:\\code\\ComfyUI`` / ``D:\\Comfy-Desktop`` 是**本机一次性事实**，不是配置项 ——
   换机器请改下面的 ``FILES`` / ``DIRS``。它不进产品运行时，只在下述场景手工跑一次。

用法::

    python backend-py/app/scripts/migrate_models.py

退出码：0 = 全部处理完（含 SKIP）；1 = 有硬链接创建失败。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

COMFY_SRC = Path(r"D:\code\ComfyUI\ComfyUI\models")
COMFY_SHARED = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models")

#: 单文件硬链接：(源, 目标)
FILES: tuple[tuple[Path, Path], ...] = (
    (COMFY_SRC / "unet" / "flux1-dev.safetensors",
     COMFY_SHARED / "unet" / "flux1-dev.safetensors"),
    (COMFY_SRC / "unet" / "z_image_turbo_bf16.safetensors",
     COMFY_SHARED / "unet" / "z_image_turbo_bf16.safetensors"),
    (COMFY_SRC / "vae" / "ae.safetensors",
     COMFY_SHARED / "vae" / "ae.safetensors"),
)

#: 整目录硬链接迁移：(源目录, 目标目录)
DIRS: tuple[tuple[Path, Path], ...] = (
    (COMFY_SRC / "vae" / "flux_vae", COMFY_SHARED / "vae" / "flux_vae"),
    (COMFY_SRC / "LLM" / "qwen3-30b-a3b-gptq", COMFY_SHARED / "llm" / "qwen3-30b-a3b-gptq"),
)


def _hard_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    os.link(source, target)


def _link_directory(source: Path, target: Path) -> None:
    """把源目录下所有文件（跳过 ``.cache``）硬链接到目标目录，保持相对结构。"""
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if ".cache" in path.parts:
            continue
        _hard_link(path, target / path.relative_to(source))


def main() -> int:
    failed = 0
    for source, target in FILES:
        if not source.exists():
            print(f"SKIP (missing src) {source}")
            continue
        try:
            _hard_link(source, target)
            print(f"OK {target}")
        except OSError as exc:
            failed += 1
            print(f"FAIL {target} -> {exc}")

    for source, target in DIRS:
        if not source.is_dir():
            print(f"SKIP (missing src dir) {source}")
            continue
        try:
            _link_directory(source, target)
            print(f"OK {target}/**")
        except OSError as exc:
            failed += 1
            print(f"FAIL {target} -> {exc}")

    print("ALL DONE" if not failed else f"ALL DONE（{failed} 个失败）")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
