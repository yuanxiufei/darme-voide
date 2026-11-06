"""下载并校验 PaddleOCR-VL 权重（供进程内 transformers 推理使用，不需要 paddle）。

为什么自带下载器而不是让用户手敲 ``huggingface-cli``
=====================================================

1. **官方站点在本机直连超时**（实测 12 s 无响应），必须走 ``hf-mirror.com``。把
   镜像地址写进代码，省得每台机器各配一遍 ``HF_ENDPOINT`` 还配错 —— 而配错的症状
   是"卡住不动"，极难自查。
2. **这个仓库同时发布两套权重**：Paddle 格式（``*.pdparams`` / ``*.pth``）与 HF 格式
   （``*.safetensors``）。本仓不装 paddle（见 ``ocr_vlm_local`` 的说明），Paddle 那
   半**永远用不上**，必须按白名单排除，否则多下约 1.8 GB 死文件。
3. ``--check`` 让"权重齐没齐"在跑 OCR **之前**就能问，而不是等第一张可疑页触发
   升级时才报错 —— 那时已经烧掉几十分钟 OCR。

用法::

    python app/scripts/corpus/ocr_vlm_fetch.py --check     # 只查，不联网（默认动作）
    python app/scripts/corpus/ocr_vlm_fetch.py --fetch     # 走镜像下载

``--check`` 退出码：齐 ⇒ 0，不齐 ⇒ 1（可直接接进启动脚本）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

__all__ = ["REPO", "MIRROR", "ALLOW_PATTERNS", "IGNORE_PATTERNS", "REQUIRED_FILES",
           "default_dir", "inspect", "fetch"]

#: 上游仓库。默认取 **1.6**（2026-05-28）：与初版**同一架构、同一 0.9B 体积**
#: （``config.json`` 与 ``model.safetensors`` 逐字节同尺寸，加载代码零改动），
#: 但 OmniDocBench v1.6 96.33%（初版在 v1.5 上 92.56），且定向加强了中文生僻字、
#: 古籍、印章、跨页表格与扫描件鲁棒性 —— 正是语料批处理最需要的那几项。
#: 初版仍可用 ``--repo PaddlePaddle/PaddleOCR-VL`` 取回作对照。
#: transformers 5.x 已原生收录（``paddleocr_vl`` 三个 model_type），所以纯 torch 可跑。
REPO = "PaddlePaddle/PaddleOCR-VL-1.6"

#: 本机能连通的唯一镜像（官方站点超时，实测见模块文档）
MIRROR = "https://hf-mirror.com"

#: 只要 HF 那套：权重 + 三份配置 + tokenizer。``*.jinja`` 是 chat template。
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]

#: Paddle / ONNX / 旧式 bin 一律不要（体积大且本仓不读）
IGNORE_PATTERNS = ["*.pth", "*.pdparams", "*.onnx", "*.bin", "*.h5", "*.msgpack"]

#: 缺任何一个都算没就绪（``config.json`` 缺 ⇒ ``from_pretrained`` 直接抛）
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json")


def default_dir() -> Path:
    """默认落点：``<仓库根>/data/models/PaddleOCR-VL-1.6``（与 ``data/ocr`` 同级）。"""
    backend_py = Path(__file__).resolve().parents[3]  # corpus → scripts → app → backend-py
    return backend_py.parent / "data" / "models" / "PaddleOCR-VL-1.6"


def inspect(dest: Path | str | None = None) -> dict:
    """盘上现状：文件数、总字节、缺哪些必需文件、有没有真权重。**只读，不联网**。"""
    path = Path(dest) if dest else default_dir()
    files: list[Path] = []
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
    total = sum(item.stat().st_size for item in files)
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    weights = [item for item in files if item.suffix.lower() == ".safetensors"]
    if not weights:
        #: 分片权重通常带 ``model.safetensors.index.json``，单文件则是 ``model.safetensors``
        missing.append("*.safetensors")
    return {
        "path": path.as_posix(),
        "exists": path.is_dir(),
        "files": len(files),
        "bytes": total,
        "gib": round(total / 2 ** 30, 2),
        "missing": missing,
        "weights": [item.name for item in weights],
        "ready": not missing,
    }


def fetch(dest: Path | str | None = None, *, repo: str = REPO, mirror: str = MIRROR,
          revision: str | None = None) -> dict:
    """从镜像拉权重（增量：已存在的文件不会重下）。返回 ``inspect()`` 的结果。"""
    path = Path(dest) if dest else default_dir()
    path.mkdir(parents=True, exist_ok=True)
    if mirror:
        current = (os.environ.get("HF_ENDPOINT") or "").strip()
        #: 只覆盖"没配"和"配成官方站点"两种情况：用户显式配了自己的镜像/代理就不动他
        if not current or "huggingface.co" in current:
            os.environ["HF_ENDPOINT"] = mirror
    #: ⚠️ 必须**在设完环境变量之后**才 import：``huggingface_hub`` 的 ENDPOINT 在
    #: 模块导入时就固化成常量，先 import 再设环境变量 ⇒ 照样打官方站点、照样超时。
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(path),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )
    return inspect(path)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(
        description="下载/校验 PaddleOCR-VL 权重（走镜像，只要 transformers 用的 safetensors）")
    parser.add_argument("--dest", default="", help="落点目录（默认 data/models/PaddleOCR-VL）")
    parser.add_argument("--repo", default=REPO, help=f"上游仓库（默认 {REPO}）")
    parser.add_argument("--mirror", default=MIRROR,
                        help="下载镜像；传空字符串则用官方站点（默认走镜像）")
    parser.add_argument("--revision", default=None, help="分支/commit（默认仓库默认分支）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="只查盘上现状，不联网（默认）")
    group.add_argument("--fetch", action="store_true", help="走镜像下载")
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    args = parser.parse_args(argv)

    dest = Path(args.dest) if args.dest else default_dir()
    if args.fetch:
        try:
            info = fetch(dest, repo=args.repo, mirror=args.mirror, revision=args.revision)
        except Exception as exc:  # noqa: BLE001 —— 网络错一律给人话，不吐栈
            print(f"✗ 下载失败：{type(exc).__name__}: {exc}")
            print(f"  镜像：{args.mirror or '官方站点'}；可先试 --check 看已下到哪一步")
            return 1
    else:
        info = inspect(dest)

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"落点：{info['path']}")
        print(f"现状：{info['files']} 个文件 / {info['gib']} GiB"
              f"（权重 {len(info['weights'])} 份）")
        if info["ready"]:
            print("✓ 权重就绪，可开 OCR_VLM_BACKEND=local")
        else:
            print(f"✗ 未就绪，缺：{', '.join(info['missing'])}")
            print("  下载：python -X utf8 app/scripts/corpus/ocr_vlm_fetch.py --fetch")
    return 0 if info["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
