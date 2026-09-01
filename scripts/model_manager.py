#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drama Studio 通用本地模型安装与管理工具（Python 标准库零依赖）。

模型清单外置为 configs/models.json，支持多类模型（文本/图片/视频/TTS）
与多种安装方式（runtime）：
  - comfyui : HTTP 下载权重到 ComfyUI/models/<kind>/（支持断点续传）
  - ollama  : ollama pull / rm
  - git     : git clone 到 local_services/<key>（如 CosyVoice 2 独立服务）
  - manual  : 仅打印部署指引，不自动安装（如 Wan 2.6 GGUF）

用法：
  python scripts/model_manager.py list [--category X] [--runtime Y] [--missing]
  python scripts/model_manager.py download [--key ...] [--category ...] [--required] [--all] [--force]
  python scripts/model_manager.py remove --key ...
  python scripts/model_manager.py doctor
  python scripts/model_manager.py install-nodes [--only ...]
  python scripts/model_manager.py add-model --key ... --name ... --category ... --runtime ... [...]
  python scripts/model_manager.py remove-model --key ...
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from urllib.parse import quote
from collections import namedtuple

# ---- Windows UTF-8 输出 ----
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CATALOG_PATH = os.path.join(PROJECT_ROOT, "configs", "models.json")
PATHS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "model-paths.json")

# ---- 路径配置（高度可配置化） ----
# 优先级：命令行参数 > 环境变量 > configs/model-paths.json > 默认/探测
Paths = namedtuple("Paths", ["comfyui_root", "models_dir", "nodes_dir", "local_services_dir"])

# 仅作为「探测回退」，真正的目录以配置/环境变量/参数为准
COMFYUI_CANDIDATES = [
    "D:/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI",
    "D:/Comfy-Desktop/ComfyUI-Shared",
    "D:/code/ComfyUI/ComfyUI",
    "D:/code/ComfyUI",
    "D:/ComfyUI/ComfyUI",
    "D:/ComfyUI",
    "C:/ComfyUI",
]


def load_paths_config(path=PATHS_CONFIG_PATH):
    """读取 configs/model-paths.json（不存在或非法时返回空 dict）。"""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def detect_comfyui():
    """返回 ComfyUI 根目录（含 models/ 与 custom_nodes/），找不到返回 None。"""
    for cand in COMFYUI_CANDIDATES:
        if cand and os.path.isdir(os.path.join(cand, "models")):
            return cand
    return None


def resolve_paths(cli_comfyui=None, cli_models=None, cli_nodes=None, cli_services=None):
    """统一解析本地模型路径，返回 Paths。优先级：CLI > 环境变量 > 配置文件 > 默认。"""
    cfg = load_paths_config()
    comfyui_root = (
        cli_comfyui
        or os.environ.get("COMFYUI_PATH")
        or cfg.get("comfyui_root")
        or detect_comfyui()
        or ""
    )
    default_models = os.path.join(comfyui_root, "models") if comfyui_root else ""
    default_nodes = os.path.join(comfyui_root, "custom_nodes") if comfyui_root else ""
    default_services = os.path.join(PROJECT_ROOT, "local_services")
    return Paths(
        comfyui_root,
        cli_models or os.environ.get("MODELS_DIR") or cfg.get("models_dir") or default_models,
        cli_nodes or os.environ.get("NODES_DIR") or cfg.get("nodes_dir") or default_nodes,
        cli_services or os.environ.get("LOCAL_SERVICES_DIR") or cfg.get("local_services_dir") or default_services,
    )


def load_catalog(path=CATALOG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_catalog(catalog, path=CATALOG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---- 状态检测 ----
def ollama_installed():
    """返回已安装的 ollama 模型名集合。"""
    try:
        r = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            return set()
        names = set()
        for line in r.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                names.add(parts[0])
        return names
    except Exception:
        return set()


def model_status(m, paths):
    """返回 (状态, 详情路径)。状态：installed/missing/manual/unknown。"""
    runtime = m.get("runtime")
    if runtime == "comfyui":
        path = os.path.join(paths.models_dir or "", m.get("kind", ""), m.get("filename", ""))
        ok = bool(paths.models_dir) and os.path.isfile(path) and os.path.getsize(path) > 0
        return ("installed" if ok else "missing", path)
    if runtime == "ollama":
        ref = m.get("ref", "")
        installed = ref in ollama_installed()
        return ("installed" if installed else "missing", "ollama:" + ref)
    if runtime == "git":
        dest = os.path.join(paths.local_services_dir, m["key"])
        ok = os.path.isdir(dest) and bool(os.listdir(dest))
        return ("installed" if ok else "missing", dest)
    if runtime == "manual":
        return ("manual", "")
    return ("unknown", "")


def _human_gib(gib):
    try:
        g = float(gib)
        if g >= 1:
            return f"{g:.2f} GiB"
        return f"{g * 1024:.0f} MiB"
    except (TypeError, ValueError):
        return "-"


def _status_mark(status):
    return {
        "installed": "OK  ",
        "missing": "MISS",
        "manual": "MAN ",
        "unknown": "?   ",
    }.get(status, "?   ")


# ---- 下载来源（与 Web UI 三源对齐） ----
DOWNLOAD_SOURCES = {
    "hf": "https://huggingface.co",
    "hf_mirror": "https://hf-mirror.com",
    "modelscope": "https://modelscope.cn",
}


def default_revision(source):
    """默认分支：ModelScope 用 master，HF 系列用 main。"""
    return "master" if source == "modelscope" else "main"


def build_download_url(source, repo, revision, file_path):
    """按来源构造下载直链（modelscope 走 /models/.../resolve 302 重定向）。"""
    enc_rev = quote(revision, safe="")
    enc_path = "/".join(quote(seg, safe="") for seg in file_path.split("/"))
    if source == "modelscope":
        return f"https://modelscope.cn/models/{repo}/resolve/{enc_rev}/{enc_path}"
    base = DOWNLOAD_SOURCES.get(source, DOWNLOAD_SOURCES["hf"])
    return f"{base}/{repo}/resolve/{enc_rev}/{enc_path}"


def resolve_download_url(m, source):
    """根据模型条目 + 来源解析下载 URL。

    优先用结构化字段 repo/file_path 构建（支持三源切换）；否则回退旧 url 字段，
    hf_mirror 时对 huggingface.co 做前缀替换。
    """
    repo = m.get("repo")
    file_path = m.get("file_path")
    if repo and file_path:
        return build_download_url(source, repo, default_revision(source), file_path)
    url = m.get("url", "")
    if source == "hf_mirror" and url.startswith("https://huggingface.co"):
        return url.replace("https://huggingface.co", "https://hf-mirror.com", 1)
    return url


def get_model_scope_download_url(repo, revision, file_path):
    """通过 ModelScope API 获取单文件签名下载 URL（LFS 大文件稳定下载）。

    失败返回 None，由调用方回退到 /resolve/ 302 直链。
    """
    try:
        url = (
            "https://modelscope.cn/api/v1/models/" + quote(repo, safe="/")
            + "/repo?FilePath=" + quote(file_path, safe="")
            + "&Revision=" + quote(revision, safe="")
        )
        with http_open(url, timeout=15) as resp:
            data = json.load(resp)
        u = data.get("Data", {}).get("Url") if isinstance(data, dict) else None
        return u if isinstance(u, str) and u else None
    except Exception:
        return None


# ---- 下载 ----
def http_open(url, headers=None, timeout=120):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)


def download_url(url, dest, force=False, timeout=120, size_gib=None):
    """断点续传下载；返回 True 成功/已就绪。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and not force:
        if size_gib is None:
            print(f"    已存在（跳过）: {dest}")
            return True
        expected = int(float(size_gib) * 1024 ** 3)
        if os.path.getsize(dest) >= expected:
            print(f"    已存在且大小达标（跳过）: {dest}")
            return True

    resume_from = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    print(f"    下载: {url}")
    with http_open(url, headers, timeout) as resp:
        total = None
        cl = resp.headers.get("Content-Length")
        if cl:
            total = int(cl) + (resume_from if resume_from else 0)
        mode = "ab" if resume_from > 0 else "wb"
        done = resume_from
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r    {done / 1024 ** 3:.2f}/{total / 1024 ** 3:.2f} GiB ({pct}%)", end="")
        print()
    return True


def install_comfyui_model(m, paths, force=False, source="hf"):
    path = os.path.join(paths.models_dir, m.get("kind", ""), m.get("filename", ""))
    url = resolve_download_url(m, source)
    if not url:
        print("    [跳过] 无可用下载地址（缺 url 或 repo/file_path）")
        return False
    # ModelScope 优先取签名 URL 下载 LFS 大文件，失败回退 /resolve/ 302 直链
    if source == "modelscope" and m.get("repo") and m.get("file_path"):
        signed = get_model_scope_download_url(
            m["repo"], default_revision(source), m["file_path"]
        )
        if signed:
            url = signed
    return download_url(url, path, force=force, size_gib=m.get("size_gib"))


def install_ollama_model(m):
    ref = m["ref"]
    print(f"    ollama pull {ref}")
    try:
        subprocess.run(["ollama", "pull", ref], check=True)
    except subprocess.CalledProcessError as e:
        print(f"    [失败] ollama pull {ref} 返回 {e.returncode}")
        return False
    return True


def install_git_model(m, paths):
    dest = os.path.join(paths.local_services_dir, m["key"])
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"    已存在（跳过）: {dest}")
        return True
    os.makedirs(paths.local_services_dir, exist_ok=True)
    print(f"    git clone {m['repo']} -> {dest}")
    try:
        subprocess.run(["git", "clone", m["repo"], dest], check=True)
    except subprocess.CalledProcessError as e:
        print(f"    [失败] git clone 返回 {e.returncode}")
        return False
    return True


def install_model(m, paths, force=False, source="hf"):
    runtime = m.get("runtime")
    print(f"  [{m['key']}] {m.get('name', m['key'])} (runtime={runtime})")
    if runtime == "comfyui":
        if not paths.models_dir:
            print("    [跳过] 未配置模型目录，请设置 models_dir / MODELS_DIR / COMFYUI_PATH")
            return False
        return install_comfyui_model(m, paths, force, source)
    if runtime == "ollama":
        return install_ollama_model(m)
    if runtime == "git":
        return install_git_model(m, paths)
    if runtime == "manual":
        print(f"    [需手动] {m.get('note', '')}")
        return False
    print("    [跳过] 未知 runtime")
    return False


# ---- 删除 ----
def remove_model_files(m, paths):
    runtime = m.get("runtime")
    if runtime == "comfyui":
        if not paths.models_dir:
            return False
        path = os.path.join(paths.models_dir, m.get("kind", ""), m.get("filename", ""))
        if os.path.isfile(path):
            os.remove(path)
            print(f"    已删除: {path}")
            return True
        print(f"    不存在: {path}")
        return False
    if runtime == "ollama":
        ref = m["ref"]
        print(f"    ollama rm {ref}")
        subprocess.run(["ollama", "rm", ref], check=False)
        return True
    if runtime == "git":
        dest = os.path.join(paths.local_services_dir, m["key"])
        if os.path.isdir(dest):
            shutil.rmtree(dest)
            print(f"    已删除目录: {dest}")
            return True
        print(f"    不存在: {dest}")
        return False
    print("    [跳过] manual/unknown runtime 无自动删除")
    return False


# ---- 节点 ----
def install_node(n, paths):
    dest = os.path.join(paths.nodes_dir, n["name"])
    if os.path.isdir(dest) and os.listdir(dest):
        print(f"  已存在（跳过）: {n['name']}")
        return True
    print(f"  git clone {n['repo']} -> {dest}")
    try:
        subprocess.run(["git", "clone", n["repo"], dest], check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [失败] {n['name']} clone 返回 {e.returncode}")
        return False
    return True


# ---- doctor ----
def check_bin(name):
    return shutil.which(name) is not None


def doctor(paths):
    print("== 底座检查 ==")
    print(f"  ComfyUI       : {paths.comfyui_root or '未检测到（请设 COMFYUI_PATH）'}")
    print(f"  模型目录      : {paths.models_dir or '未配置'}")
    print(f"  节点目录      : {paths.nodes_dir or '未配置'}")
    print(f"  本地服务目录  : {paths.local_services_dir or '未配置'}")
    print(f"  git           : {'OK' if check_bin('git') else 'MISS'}")
    print(f"  ollama        : {'OK' if check_bin('ollama') else 'MISS'}")
    ollama_models = ollama_installed()
    if ollama_models:
        print(f"    ollama 已拉取: {', '.join(sorted(ollama_models))}")
    print(f"  Python        : {sys.version.split()[0]}")

    if paths.comfyui_root:
        free = shutil.disk_usage(paths.comfyui_root).free
        print(f"  磁盘剩余(ComfyUI盘): {free / 1024 ** 3:.1f} GiB")


def main():
    parser = argparse.ArgumentParser(description="Drama Studio 通用本地模型安装与管理工具")
    common_paths = argparse.ArgumentParser(add_help=False)
    common_paths.add_argument("--comfyui-dir", help="ComfyUI 根目录（含 models/ 与 custom_nodes/）")
    common_paths.add_argument("--models-dir", help="模型根目录")
    common_paths.add_argument("--nodes-dir", help="节点根目录")
    common_paths.add_argument("--local-services-dir", help="本地服务根目录（git 类模型/独立服务）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", parents=[common_paths], help="列出模型清单与安装状态")
    p_list.add_argument("--category", help="按类别筛选（text/image/video/tts）")
    p_list.add_argument("--runtime", help="按安装方式筛选（comfyui/ollama/git/manual）")
    p_list.add_argument("--missing", action="store_true", help="仅显示未安装项")

    p_dl = sub.add_parser("download", parents=[common_paths], help="下载/安装模型")
    p_dl.add_argument("--key", nargs="+", help="指定模型 key")
    p_dl.add_argument("--category", help="按类别批量安装")
    p_dl.add_argument("--required", action="store_true", help="安装全部 required 模型")
    p_dl.add_argument("--all", action="store_true", help="安装全部可自动安装的模型")
    p_dl.add_argument("--force", action="store_true", help="强制重新下载")
    p_dl.add_argument("--source", choices=["hf", "hf_mirror", "modelscope"], default="hf",
                      help="comfyui 权重下载来源：hf / hf_mirror / modelscope（默认 hf）")

    p_rm = sub.add_parser("remove", parents=[common_paths], help="删除已安装的模型文件")
    p_rm.add_argument("--key", nargs="+", required=True, help="模型 key")

    sub.add_parser("doctor", parents=[common_paths], help="体检（底座/磁盘/模型/节点）")

    p_node = sub.add_parser("install-nodes", parents=[common_paths], help="安装 ComfyUI 自定义节点")
    p_node.add_argument("--only", nargs="+", help="仅安装指定节点名")
    p_node.add_argument("--required", action="store_true", help="仅安装 required 节点")

    sub.add_parser("paths", parents=[common_paths], help="显示当前路径解析结果（CLI>环境变量>配置文件>默认）")

    p_add = sub.add_parser("add-model", help="添加模型条目到 configs/models.json")
    p_add.add_argument("--key", required=True)
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--category", required=True)
    p_add.add_argument("--runtime", required=True, choices=["comfyui", "ollama", "git", "manual"])
    p_add.add_argument("--kind", help="comfyui: 子目录（diffusion_models/text_encoders/vae/checkpoints）")
    p_add.add_argument("--filename", help="comfyui: 文件名")
    p_add.add_argument("--url", help="comfyui: 下载地址（缺省时用 repo/file_path 结构化构建）")
    p_add.add_argument("--ref", help="ollama: 模型标识")
    p_add.add_argument("--repo", help="git: 仓库地址；comfyui: 结构化 owner/name（配合 --file-path 支持三源）")
    p_add.add_argument("--file-path", help="comfyui: 仓库内相对路径（配合 --repo 使用）")
    p_add.add_argument("--size", type=float, help="大小 GiB")
    p_add.add_argument("--required", action="store_true", help="标记为必需")
    p_add.add_argument("--note", default="", help="备注")
    p_add.add_argument("--tag", nargs="+", default=[], help="标签")

    p_rmm = sub.add_parser("remove-model", help="从 configs/models.json 移除模型条目")
    p_rmm.add_argument("--key", required=True, help="模型 key")

    args = parser.parse_args()
    catalog = load_catalog()
    models = catalog.get("models", [])
    nodes = catalog.get("nodes", [])
    paths = resolve_paths(
        cli_comfyui=getattr(args, "comfyui_dir", None),
        cli_models=getattr(args, "models_dir", None),
        cli_nodes=getattr(args, "nodes_dir", None),
        cli_services=getattr(args, "local_services_dir", None),
    )

    if args.cmd == "paths":
        print("== 路径解析结果（CLI > 环境变量 > configs/model-paths.json > 默认/探测） ==")
        print(f"  ComfyUI 根目录  : {paths.comfyui_root or '未检测到'}")
        print(f"  模型目录        : {paths.models_dir or '未配置'}")
        print(f"  节点目录        : {paths.nodes_dir or '未配置'}")
        print(f"  本地服务目录    : {paths.local_services_dir or '未配置'}")
        return

    if args.cmd == "list":
        print(f"模型清单（共 {len(models)} 个）")
        print(f"{'状态':<5} {'key':<28} {'类别':<8} {'方式':<9} {'大小':<11} {'必需':<5} 名称")
        for m in models:
            if args.category and m.get("category") != args.category:
                continue
            if args.runtime and m.get("runtime") != args.runtime:
                continue
            status, detail = model_status(m, paths)
            if args.missing and status == "installed":
                continue
            flag = "*" if m.get("required") else ""
            print(
                f"{_status_mark(status):<5} {m['key']:<28} {m.get('category', '-'):<8} "
                f"{m.get('runtime', '-'):<9} {_human_gib(m.get('size_gib')):<11} {flag:<5} {m.get('name', '')}"
            )
        if args.missing or not args.category:
            print(f"\n节点清单（共 {len(nodes)} 个）")
            for n in nodes:
                dest = os.path.join(paths.nodes_dir if paths.nodes_dir else "", n["name"])
                ok = paths.comfyui_root and os.path.isdir(dest) and os.listdir(dest)
                print(f"  {'OK  ' if ok else 'MISS'} {n['name']:<36} {n.get('purpose', '')}")
        return

    if args.cmd == "download":
        targets = []
        if args.key:
            keymap = {m["key"]: m for m in models}
            targets = [keymap[k] for k in args.key if k in keymap]
        elif args.category:
            targets = [m for m in models if m.get("category") == args.category]
        elif args.required:
            targets = [m for m in models if m.get("required")]
        elif args.all:
            targets = [m for m in models if m.get("runtime") != "manual"]
        else:
            print("请指定 --key / --category / --required / --all")
            return
        if not targets:
            print("无匹配模型")
            return
        ok = 0
        for m in targets:
            if install_model(m, paths, force=args.force, source=args.source):
                ok += 1
        print(f"\n完成：成功/就绪 {ok}/{len(targets)}")
        return

    if args.cmd == "remove":
        keymap = {m["key"]: m for m in models}
        for k in args.key:
            m = keymap.get(k)
            if not m:
                print(f"[未知 key] {k}")
                continue
            print(f"[{k}]")
            remove_model_files(m, paths)
        return

    if args.cmd == "doctor":
        doctor(paths)
        print("\n== 模型状态 ==")
        for m in models:
            status, detail = model_status(m, paths)
            print(f"  {_status_mark(status)} {m['key']:<28} {detail}")
        print("\n== 节点状态 ==")
        for n in nodes:
            dest = os.path.join(paths.nodes_dir if paths.nodes_dir else "", n["name"])
            ok = paths.comfyui_root and os.path.isdir(dest) and os.listdir(dest)
            print(f"  {'OK  ' if ok else 'MISS'} {n['name']}")
        return

    if args.cmd == "install-nodes":
        if not paths.comfyui_root:
            print("未检测到 ComfyUI，请设置 COMFYUI_PATH 或 --comfyui-dir")
            return
        targets = nodes
        if args.only:
            nmap = {n["name"]: n for n in nodes}
            targets = [nmap[x] for x in args.only if x in nmap]
        elif args.required:
            targets = [n for n in nodes if n.get("required")]
        if not targets:
            print("无匹配节点")
            return
        for n in targets:
            install_node(n, paths)
        return

    if args.cmd == "add-model":
        if any(m["key"] == args.key for m in models):
            print(f"key 已存在: {args.key}")
            return
        entry = {
            "key": args.key,
            "name": args.name,
            "category": args.category,
            "runtime": args.runtime,
            "size_gib": args.size,
            "required": bool(args.required),
            "tags": args.tag,
            "note": args.note,
        }
        if args.kind:
            entry["kind"] = args.kind
        if args.filename:
            entry["filename"] = args.filename
        if args.url:
            entry["url"] = args.url
        if args.ref:
            entry["ref"] = args.ref
        if args.repo:
            entry["repo"] = args.repo
        if args.file_path:
            entry["file_path"] = args.file_path
        entry = {k: v for k, v in entry.items() if v not in (None, "", [])}
        catalog["models"].append(entry)
        save_catalog(catalog)
        print(f"已添加: {args.key}")
        return

    if args.cmd == "remove-model":
        before = len(catalog["models"])
        catalog["models"] = [m for m in catalog["models"] if m["key"] != args.key]
        if len(catalog["models"]) == before:
            print(f"未找到 key: {args.key}")
            return
        save_catalog(catalog)
        print(f"已移除: {args.key}")
        return


if __name__ == "__main__":
    main()
