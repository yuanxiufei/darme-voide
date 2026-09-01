#!/usr/bin/env python3
"""sd_h3_pipeline.py —— sd.cpp × MiniMax-H3 Turbo GGUF 一键编排脚本。

把「路线 A（Turbo 蒸馏 GGUF + sd.cpp）」的验证流程串成一条流水线：

    doctor   →   环境体检（git/cmake/MSVC/CUDA/磁盘/sd.cpp/GGUF）
    download →   下载 Q4_0 Turbo GGUF（hf-mirror，断点续传）
    build    →   git clone --recursive + cmake -DSD_CUDA=ON 编译 sd.cpp
    probe    →   调用 sd_h3_compat_probe.py 做静态判定（--live 真机 load）

用法：
    python scripts/sd_h3_pipeline.py                 # 顺序执行 doctor→download→build→probe
    python scripts/sd_h3_pipeline.py doctor          # 单步：仅体检
    python scripts/sd_h3_pipeline.py download        # 单步：仅下载
    python scripts/sd_h3_pipeline.py build           # 单步：仅编译
    python scripts/sd_h3_pipeline.py probe [--live]  # 单步：仅判定
    python scripts/sd_h3_pipeline.py --help

依赖：Python 3.8+ 标准库 + 同目录 model_manager.py（复用 download_url/check_bin）。
编译依赖：Git、CMake 3.x+、Visual Studio 2019/2022（C++ 桌面开发）、CUDA Toolkit。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model_manager import check_bin, download_url  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ---- 事实常量 ----
SD_CPP_REPO = "https://github.com/leejet/stable-diffusion.cpp"
SD_CPP_DIR = os.path.join(PROJECT_ROOT, "sd-cpp")
GGUF_DIR = os.path.join(PROJECT_ROOT, "models", "gguf")

TURBO_REPO = "molbal/MiniMax-H3-Turbo-GGUF"
TURBO_FILE = "minimax_h3_fl2v_turbo_4step_v1.0_768p_Q4_0.gguf"
TURBO_SIZE_GIB = 11.4
MIRROR_URL = "https://hf-mirror.com/{repo}/resolve/main/{file}"
HF_URL = "https://huggingface.co/{repo}/resolve/main/{file}"

PROBE_SCRIPT = os.path.join(SCRIPT_DIR, "sd_h3_compat_probe.py")

# sd-cli 完整 vid_gen 命令模板（docs/minimax_h3.md 权威形式，含 vae/audio-vae/llm）
SD_CLI_TEMPLATE = (
    'sd-cli -M vid_gen '
    '--diffusion-model "{gguf}" '
    '--vae "<video_vae.safetensors>" '
    '--audio-vae "<audio_vae.safetensors>" '
    '--llm "<qwen3vl_32b_minimax_h3-Q4_K_M.gguf>" '
    '-p "<prompt>" --cfg-scale 1.0 -W 864 -H 480 '
    '--diffusion-fa --offload-to-cpu --rng cpu --fps 24 --video-frames 56'
)


# ---- 工具 ----
def _which(name: str) -> str:
    return shutil.which(name) or ""


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _human(n: int) -> str:
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} TiB"


def locate_sd_cli() -> str:
    """定位 sd-cli.exe：优先 build/bin/Release，回退全 build 递归查找。"""
    for cand in (
        os.path.join(SD_CPP_DIR, "build", "bin", "Release", "sd-cli.exe"),
        os.path.join(SD_CPP_DIR, "build", "bin", "sd-cli.exe"),
    ):
        if os.path.isfile(cand):
            return cand
    build_dir = os.path.join(SD_CPP_DIR, "build")
    if os.path.isdir(build_dir):
        for root, _dirs, files in os.walk(build_dir):
            for f in files:
                if f.lower() in ("sd-cli.exe", "sd.exe"):
                    return os.path.join(root, f)
    return ""


def detect_gpus() -> list:
    """通过 nvidia-smi 检测 GPU 型号与显存（MiB）。失败返回空列表。"""
    try:
        r = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                  "--format=csv,noheader,nounits"], timeout=10)
        if r.returncode != 0:
            return []
        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0]:
                name = parts[0]
                mem = int(float(parts[1])) if len(parts) > 1 and parts[1] else 0
                gpus.append((name, mem))
        return gpus
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def detect_cuda_toolkit() -> str:
    """检测 CUDA Toolkit 编译器 nvcc（非 CUDA runtime）。

    nvcc 只存在于完整 CUDA Toolkit 安装中；PyTorch 的 pip CUDA 运行时
    （cuXXX 包）只带 runtime/cudnn，不带 nvcc。因此「torch.cuda 可用」
    ≠「能从源码编译 CUDA 版 sd.cpp」。
    """
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(base):
        vers = sorted(
            (d for d in os.listdir(base)
             if d.lower().startswith("v") and os.path.isdir(os.path.join(base, d))),
            reverse=True,
        )
        for v in vers:
            cand = os.path.join(base, v, "bin", "nvcc.exe")
            if os.path.isfile(cand):
                return cand
    return ""


def detect_torch_cuda() -> str:
    """检测 PyTorch CUDA 运行时可用性（仅运行时，不代表有 nvcc）。"""
    try:
        code = ("import torch; "
                "print('torch %s | cuda_avail=%s | cuda=%s' % "
                "(torch.__version__, torch.cuda.is_available(), torch.version.cuda))")
        r = _run([sys.executable, "-c", code], timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        return ""
    except (OSError, subprocess.SubprocessError):
        return ""


# ---- 步骤 1：doctor ----
def doctor() -> dict:
    checks = []
    def _add(name, ok, detail):
        checks.append({"item": name, "ok": ok, "detail": detail})
        print(f"  [{'OK' if ok else '!!'}] {name:<14} {detail}")

    print("== GPU / CUDA 运行时 ==")
    gpus = detect_gpus()
    if gpus:
        desc = "、".join(f"{n}（{m / 1024:.0f} GiB）" for n, m in gpus)
        _add("GPU", True, desc)
    else:
        _add("GPU", False, "未检测到 NVIDIA GPU（nvidia-smi 不可用）")

    tc = detect_torch_cuda()
    _add("PyTorch CUDA", bool(tc), tc or "未检测到 torch（无 CUDA runtime）")

    print("== 编译工具链 ==")
    _add("git", bool(_which("git")), _which("git") or "未找到（需 git clone）")
    _add("cmake", bool(_which("cmake")), _which("cmake") or "未找到（需 CMake 3.x+）")
    nvcc = detect_cuda_toolkit()
    _add("CUDA Toolkit", bool(nvcc),
         nvcc or "未找到 nvcc（缺 CUDA Toolkit，无法编译 CUDA 版 sd.cpp）")

    # MSVC 检测：优先 vswhere
    msvis = False
    vsw = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.isfile(vsw):
        try:
            r = _run([vsw, "-latest", "-products", "*",
                      "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                      "-property", "installationPath"], timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                msvis = True
        except (OSError, subprocess.SubprocessError):
            pass
    _add("MSVC (VS2019/2022)", msvis,
         "检测到 VS" if msvis else "未检测到（需安装「使用 C++ 的桌面开发」）")

    print("== 磁盘空间 ==")
    try:
        free = shutil.disk_usage(SD_CPP_DIR if os.path.isdir(SD_CPP_DIR) else PROJECT_ROOT).free
        need = int(TURBO_SIZE_GIB * 1024 ** 3) + 5 * 1024 ** 3  # GGUF + 编译产物余量
        _add("磁盘空间", free >= need, f"剩余 {_human(free)}（需约 {_human(need)}）")
    except OSError as e:
        _add("磁盘空间", False, f"无法检测：{e}")

    print("== sd.cpp 状态 ==")
    _add("源码 clone", os.path.isdir(os.path.join(SD_CPP_DIR, "ggml")),
         f"{SD_CPP_DIR}")
    sd_cli = locate_sd_cli()
    _add("sd-cli 编译", bool(sd_cli), sd_cli or f"未编译（{SD_CPP_DIR}\\build）")

    print("== GGUF 状态 ==")
    gguf = os.path.join(GGUF_DIR, TURBO_FILE)
    ok = os.path.isfile(gguf) and os.path.getsize(gguf) >= int(TURBO_SIZE_GIB * 1024 ** 3)
    _add("Turbo GGUF", ok, f"{gguf} {'已就绪' if ok else '缺失'}（{TURBO_SIZE_GIB} GiB）")

    ok_all = all(c["ok"] for c in checks)
    print(f"\n结论：{'全部就绪，可直接 probe --live' if ok_all else '存在缺失，按 !! 项补齐'}")
    return {"checks": checks, "all_ok": ok_all}


# ---- 步骤 2：download ----
def download(source: str, force: bool) -> str:
    base = MIRROR_URL if source == "mirror" else HF_URL
    url = base.format(repo=TURBO_REPO, file=TURBO_FILE)
    dest = os.path.join(GGUF_DIR, TURBO_FILE)
    os.makedirs(GGUF_DIR, exist_ok=True)
    print(f"下载 {TURBO_FILE}（{TURBO_SIZE_GIB} GiB）到 {dest}")
    print(f"源：{url}")
    ok = download_url(url, dest, force=force, size_gib=TURBO_SIZE_GIB)
    return dest if ok else ""


# ---- 步骤 3：build ----
def build(force: bool) -> str:
    if not check_bin("git"):
        raise RuntimeError("未找到 git，请先安装并加入 PATH")
    if not check_bin("cmake"):
        raise RuntimeError("未找到 cmake，请先安装 CMake 3.x+")

    os.makedirs(SD_CPP_DIR, exist_ok=True)
    if not os.path.isdir(os.path.join(SD_CPP_DIR, "ggml")):
        print(f"git clone --recursive {SD_CPP_REPO} -> {SD_CPP_DIR}")
        _run(["git", "clone", "--recursive", SD_CPP_REPO, SD_CPP_DIR], check=True)
    else:
        print(f"源码已存在，跳过 clone：{SD_CPP_DIR}")

    build_dir = os.path.join(SD_CPP_DIR, "build")
    if os.path.isfile(os.path.join(build_dir, "CMakeCache.txt")) and not force:
        print("CMake 已配置，跳过 configure（--force 可重配）")
    else:
        print(f"cmake -S {SD_CPP_DIR} -B {build_dir} -DSD_CUDA=ON")
        _run(["cmake", "-S", SD_CPP_DIR, "-B", build_dir,
              "-DSD_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"], check=True)

    print("cmake --build build --config Release --parallel")
    _run(["cmake", "--build", build_dir, "--config", "Release",
          "--parallel"], check=True)

    sd_cli = locate_sd_cli()
    if not sd_cli:
        raise RuntimeError(f"编译完成但未找到 sd-cli.exe，请检查 {build_dir}")
    print(f"sd-cli 就绪：{sd_cli}")
    return sd_cli


# ---- 步骤 4：probe ----
def probe(live: bool, gguf: str, sd_dir: str) -> int:
    if not os.path.isfile(PROBE_SCRIPT):
        raise RuntimeError(f"缺少 {PROBE_SCRIPT}")
    cmd = [sys.executable, PROBE_SCRIPT, "--gguf", gguf]
    if sd_dir:
        cmd += ["--sd-dir", sd_dir]
    if live:
        cmd.append("--live")
    r = _run(cmd)
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    return r.returncode


# ---- 主流程 ----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="sd_h3_pipeline",
        description="sd.cpp × MiniMax-H3 Turbo GGUF 一键验证流水线",
    )
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("doctor", help="环境体检")
    d = sub.add_parser("download", help="下载 Q4_0 Turbo GGUF")
    d.add_argument("--source", choices=("mirror", "hf"), default="mirror",
                   help="下载源（默认 hf-mirror）")
    d.add_argument("--force", action="store_true")
    b = sub.add_parser("build", help="clone + 编译 sd.cpp（-DSD_CUDA=ON）")
    b.add_argument("--force", action="store_true")
    p = sub.add_parser("probe", help="运行兼容性判定")
    p.add_argument("--live", action="store_true", help="真机 load 测试")
    p.add_argument("--gguf", default=os.path.join(GGUF_DIR, TURBO_FILE))
    p.add_argument("--sd-dir", default=SD_CPP_DIR)
    args = ap.parse_args(argv)

    # 默认（无子命令）：顺序跑全流程
    if not args.cmd:
        doctor()
        print("\n" + "=" * 50)
        if input("\n继续执行 download→build→probe？[y/N] ").strip().lower() != "y":
            return 0
        dest = download("mirror", force=False)
        if not dest:
            return 1
        build(force=False)
        gguf = os.path.join(GGUF_DIR, TURBO_FILE)
        return probe(live=False, gguf=gguf, sd_dir=SD_CPP_DIR)

    try:
        if args.cmd == "doctor":
            doctor()
        elif args.cmd == "download":
            download(args.source, args.force)
        elif args.cmd == "build":
            build(args.force)
        elif args.cmd == "probe":
            return probe(args.live, args.gguf, args.sd_dir)
        return 0
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
