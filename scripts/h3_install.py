"""H3 安装工具兼容 shim —— 数据源已统一到 configs/models.json，通用化由 model_manager.py 承担。

说明
----
原 h3_install.py 内置的 H3 量化模型清单（8 个）与节点清单（4 个）已迁移到
`configs/models.json`（category=video + runtime=comfyui 即 H3 模型；nodes 即节点）。
本文件保留旧 CLI（doctor/download/download-model/install-nodes）与库接口，避免旧用法
broken；数据一律从 models.json 读取，增删模型无需再改本文件。

新能力（多类模型：文本/图片/视频/TTS；ollama/git/manual 安装；add-model/remove-model）
请直接使用 `scripts/model_manager.py`：

    python scripts/model_manager.py list
    python scripts/model_manager.py download --required
    python scripts/model_manager.py doctor
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 让 `from scripts.h3_install import ...` 与直接 `python scripts/h3_install.py` 均可工作
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_manager import (  # noqa: E402
    check_bin,
    download_url,
    load_catalog,
    resolve_paths,
)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


@dataclass(frozen=True)
class H3Model:
    key: str
    filename: str
    url: str
    kind: str
    size_gib: float
    required: bool = False
    note: str = ""


@dataclass
class H3Node:
    name: str
    repo: str
    purpose: str
    required: bool = False


def _load_from_catalog():
    """从 configs/models.json 构建 H3_MODELS / H3_NODES（保持原库接口）。"""
    cat = load_catalog()
    models: dict[str, H3Model] = {}
    for m in cat.get("models", []):
        if m.get("category") == "video" and m.get("runtime") == "comfyui":
            models[m["key"]] = H3Model(
                m["key"],
                m["filename"],
                m["url"],
                m["kind"],
                float(m.get("size_gib") or 0),
                bool(m.get("required")),
                m.get("note", ""),
            )
    nodes = [
        H3Node(n["name"], n["repo"], n.get("purpose", ""), bool(n.get("required")))
        for n in cat.get("nodes", [])
    ]
    return models, nodes


H3_MODELS, H3_NODES = _load_from_catalog()


def _resolve_dirs(root, models_dir, nodes_dir):
    """统一路径解析：CLI > 环境变量 > configs/model-paths.json > 探测/默认。"""
    paths = resolve_paths(
        cli_comfyui=root, cli_models=models_dir, cli_nodes=nodes_dir
    )
    return (
        Path(paths.comfyui_root) if paths.comfyui_root else Path(),
        Path(paths.models_dir) if paths.models_dir else Path(),
        Path(paths.nodes_dir) if paths.nodes_dir else Path(),
    )


def _human(nbytes: float) -> str:
    n = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def download_model(model: H3Model, root=None, models_dir=None,
                   force: bool = False, timeout: float = 120) -> str:
    _, mdir, _ = _resolve_dirs(root, models_dir, None)
    dest_dir = mdir / model.kind
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / model.filename
    download_url(model.url, str(dest), force=force, timeout=timeout, size_gib=model.size_gib)
    return str(dest)


def download_recommended(root=None, models_dir=None, force: bool = False) -> list[str]:
    return [download_model(m, root, models_dir=models_dir, force=force)
            for m in H3_MODELS.values() if m.required]


def download_all(root=None, models_dir=None, force: bool = False) -> list[str]:
    return [download_model(m, root, models_dir=models_dir, force=force)
            for m in H3_MODELS.values()]


def install_node(node: H3Node, root=None, nodes_dir=None) -> str:
    if not check_bin("git"):
        raise RuntimeError("未找到 git，请先安装 git 并加入 PATH")
    _, _, ndir = _resolve_dirs(root, None, nodes_dir)
    ndir.mkdir(parents=True, exist_ok=True)
    dest = ndir / Path(node.repo.rstrip("/")).name
    if dest.exists():
        print(f"[跳过] {node.name} 已存在于 {dest}")
        return str(dest)
    print(f"[安装] {node.name} -> {node.repo}")
    subprocess.run(["git", "clone", "--depth", "1", node.repo, str(dest)], check=True)
    req = dest / "requirements.txt"
    if req.exists():
        print(f"  [提示] {node.name} 有依赖：{req}，请在 ComfyUI 环境 pip install -r 后重启")
    return str(dest)


def install_nodes(root=None, only: Optional[list[str]] = None, nodes_dir=None) -> list[str]:
    targets = H3_NODES if not only else [n for n in H3_NODES if n.name in only]
    return [install_node(n, root, nodes_dir=nodes_dir) for n in targets]


def doctor(root=None, models_dir=None, nodes_dir=None) -> dict:
    root, mdir, ndir = _resolve_dirs(root, models_dir, nodes_dir)
    report: dict = {"comfyui_root": str(root), "checks": []}

    def _add(name, ok, detail):
        report["checks"].append({"item": name, "ok": ok, "detail": detail})
        print(f"  [{'OK' if ok else '!!'}] {name}: {detail}")

    print(f"ComfyUI 根目录: {root}")
    print(f"模型目录: {mdir}")
    print(f"节点目录: {ndir}")
    _add("ComfyUI 底座", (root / "main.py").exists() or (root / "comfy").exists(),
         "存在" if (root / "main.py").exists() or (root / "comfy").exists() else "缺失")
    _add("git", check_bin("git"), shutil.which("git") or "未找到")

    try:
        free = shutil.disk_usage(mdir).free
        need = sum(m.size_gib for m in H3_MODELS.values() if m.required) * (1 << 30)
        _add("磁盘空间", free >= need, f"剩余 {_human(free)}，推荐配置需约 {_human(need)}")
    except Exception as e:  # noqa: BLE001
        _add("磁盘空间", False, f"无法检测: {e}")

    for m in H3_MODELS.values():
        p = mdir / m.kind / m.filename
        ok = p.exists() and p.stat().st_size > 0
        _add(f"模型 {m.key}", ok,
             f"{p.name} {'已就绪' if ok else '缺失'} ({m.size_gib} GiB{'，推荐' if m.required else '，可选'})")

    for n in H3_NODES:
        p = ndir / Path(n.repo.rstrip("/")).name
        _add(f"节点 {n.name}", p.exists(), f"{'已安装' if p.exists() else '未安装'} — {n.purpose}")

    ok_all = all(c["ok"] for c in report["checks"])
    print("\n结论:", "环境就绪" if ok_all else "存在缺失，请按上述 !! 项补装")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--comfyui", help="ComfyUI 根目录")
    common.add_argument("--models-dir", help="模型根目录（默认 <ComfyUI>/models）")
    common.add_argument("--nodes-dir", help="节点根目录（默认 <ComfyUI>/custom_nodes）")

    p = argparse.ArgumentParser(prog="h3_install", description="H3 本地量化栈安装（兼容 shim，数据源 configs/models.json）")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", parents=[common], help="环境体检")
    d = sub.add_parser("download", parents=[common], help="下载 24GB 推荐配置")
    d.add_argument("--all", action="store_true", help="下载全部清单（含可选）")
    d.add_argument("--force", action="store_true", help="强制覆盖")
    dm = sub.add_parser("download-model", parents=[common], help="下载单个模型")
    dm.add_argument("--key", required=True, choices=list(H3_MODELS), help="模型 key")
    dm.add_argument("--force", action="store_true", help="强制覆盖")
    n = sub.add_parser("install-nodes", parents=[common], help="安装 H3 自定义节点")
    n.add_argument("--only", action="append", help="只装指定节点（可重复）")

    args = p.parse_args(argv)
    try:
        if args.cmd == "doctor":
            doctor(args.comfyui, models_dir=args.models_dir, nodes_dir=args.nodes_dir)
        elif args.cmd == "download":
            (download_all if args.all else download_recommended)(
                args.comfyui, models_dir=args.models_dir, force=args.force)
        elif args.cmd == "download-model":
            download_model(H3_MODELS[args.key], args.comfyui,
                           models_dir=args.models_dir, force=args.force)
        elif args.cmd == "install-nodes":
            install_nodes(args.comfyui, only=args.only, nodes_dir=args.nodes_dir)
        return 0
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
