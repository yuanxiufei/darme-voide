#!/usr/bin/env python3
"""sd.cpp × MiniMax-H3 Turbo GGUF 兼容性实测（零依赖，纯标准库）。

回答一个问题：sd.cpp 能不能加载「Turbo 蒸馏版」H3 GGUF？

背景：H3 Turbo 有 Turbo LoRA（~744MB 适配器）与 lightx2v 蒸馏（独立模型）两条路线。
molbal/MiniMax-H3-Turbo-GGUF 是蒸馏模型 GGUF，文件为
minimax_h3_fl2v_turbo_4step_v1.0_768p_{Q4_0,Q8_0,Q8_CR}.gguf，
元数据 general.architecture 同样声明为 minimax_h3（与 base 同名）。
sd.cpp 官方 docs/minimax_h3.md 只支持 base H3（time-embedder DiT / AdaLN curve-table，
从权重自动识别），README 要求 ComfyUI GGUF loader + Turbo 采样节点，未提 sd.cpp。
=> arch 同名 ≠ 一定能加载，需实证。

用法：
  python scripts/sd_h3_compat_probe.py --gguf <path.gguf>          # 静态判定
  python scripts/sd_h3_compat_probe.py --gguf <path.gguf> --live   # 真机 load
  python scripts/sd_h3_compat_probe.py --list                      # 支持矩阵
"""
from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ARCH_MINIMAX_H3 = "minimax_h3"
TURBO_REPO = "molbal/MiniMax-H3-Turbo-GGUF"
TURBO_FILES = [
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_Q4_0.gguf",
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_Q8_0.gguf",
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_Q8_CR.gguf",
]
BASE_FILES = [
    "minimax_h3_fl2va_pruned-Q4_K_M.gguf",
    "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
]
SD_CPP_VIDEO_ARCHS = {
    "minimax_h3": "MiniMax H3 base（time-embedder DiT / AdaLN curve-table）",
    "wan": "Wan 2.1/2.6",
    "hunyuan_video": "腾讯混元视频",
    "flux": "Flux 系列",
}

GGUF_MAGIC = 0x46554747
SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
              10: "<Q", 11: "<q", 12: "<d"}
LOAD_OK = ("loaded", "arch:", "architecture", "total tensors")
LOAD_REJECT = ("unsupported", "not supported", "unknown arch", "failed to load",
               "invalid gguf", "gguf_get")


def _u32(d, o): return struct.unpack_from("<I", d, o)[0], o + 4
def _u64(d, o): return struct.unpack_from("<Q", d, o)[0], o + 8


def _read_string(d, o):
    n, o = _u64(d, o)
    return d[o:o + n].decode("utf-8", "replace"), o + n


def _read_value(d, o, t):
    if t == 8:
        return _read_string(d, o)
    if t == 9:  # ARRAY：仅跳过
        et, o = _u32(d, o)
        cnt, o = _u64(d, o)
        for _ in range(cnt):
            _, o = _read_value(d, o, et)
        return None, o
    if t == 7:
        return bool(d[o]), o + 1
    sz = SCALAR_SIZE.get(t)
    if sz is None:
        raise ValueError(f"未知 GGUF 类型 {t}")
    val = struct.unpack(SCALAR_FMT[t], d[o:o + sz])[0]
    return val, o + sz


def parse_gguf(path, max_bytes=16 * 1024 * 1024):
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    if len(data) < 24 or struct.unpack_from("<I", data, 0)[0] != GGUF_MAGIC:
        raise ValueError("不是有效 GGUF")
    version = struct.unpack_from("<I", data, 4)[0]
    tensor_count = struct.unpack_from("<Q", data, 8)[0]
    kv_count = struct.unpack_from("<Q", data, 16)[0]
    meta, off = {}, 24
    for _ in range(kv_count):
        key, off = _read_string(data, off)
        t, off = _u32(data, off)
        val, off = _read_value(data, off, t)
        meta[key] = val
    return {"version": version, "tensor_count": tensor_count, "meta": meta}


def find_sd_cli(sd_dir=None):
    cands = [shutil.which(n) or "" for n in ("sd-cli", "sd-cli.exe", "sd", "sd.exe")]
    if sd_dir:
        for sub in (Path(sd_dir), Path(sd_dir) / "build",
                    Path(sd_dir) / "build" / "bin",
                    Path(sd_dir) / "build" / "examples" / "cli"):
            if sub.is_dir():
                for n in ("sd-cli", "sd-cli.exe", "sd", "sd.exe"):
                    if (sub / n).exists():
                        cands.insert(0, str(sub / n))
    return next((c for c in cands if c), None)


def detect_gpu():
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        try:
            r = subprocess.run([nvsmi, "--query-gpu=name,memory.total",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                return True, r.stdout.strip().replace("\n", "; ")
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return True, f"torch.cuda: {torch.cuda.get_device_name(0)}"
    except ImportError:
        return False, "无 nvidia-smi 且无 torch"
    return False, "无 GPU"


def live_test(sd_cli, gguf, timeout):
    out = str(Path(gguf).with_suffix("")) + "_probe.mp4"
    cmd = [sd_cli, "-M", "vid_gen", "--diffusion-model", gguf,
           "-p", "a test shot, one person speaking", "-o", out]
    r = {"command": cmd, "returncode": None, "stage": "unknown", "tail": ""}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        r["returncode"] = p.returncode
        r["tail"] = (p.stderr or "")[-2000:]
        low = ((p.stdout or "") + (p.stderr or "")).lower()
        r["stage"] = ("rejected" if any(m in low for m in LOAD_REJECT)
                      else "loaded" if any(m in low for m in LOAD_OK)
                      else "unclear")
    except subprocess.TimeoutExpired:
        r["stage"], r["tail"] = "timeout", f"超 {timeout}s（可能已加载在采样，或 OOM）"
    except OSError as e:
        r["stage"], r["tail"] = "spawn_failed", str(e)
    return r


def print_list():
    print("sd.cpp 已知视频架构：")
    for a, d in SD_CPP_VIDEO_ARCHS.items():
        print(f"  {a:<16} {d}")
    print("\n[路线 B 已验证] unsloth/MiniMax-H3-GGUF:")
    for f in BASE_FILES:
        print(f"  - {f}")
    print(f"[路线 A 待验证] {TURBO_REPO}:")
    for f in TURBO_FILES:
        print(f"  - {f}")
    print("\narch == minimax_h3 只是必要条件；蒸馏版需 --live 实证，被拒则落回路线 C。")


def main(argv=None):
    ap = argparse.ArgumentParser(description="sd.cpp × H3 Turbo GGUF 兼容性实测")
    ap.add_argument("--gguf")
    ap.add_argument("--sd-dir")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        print_list()
        return 0

    gguf = args.gguf
    if not gguf:
        for hint in TURBO_FILES + BASE_FILES:
            for base in (Path.cwd(), Path.cwd() / "models"):
                if (base / hint).exists():
                    gguf = str(base / hint)
                    break
            if gguf:
                break
        if not gguf:
            print("未找到本机 GGUF，请下载后重试：")
            for f in TURBO_FILES:
                print(f"  https://hf-mirror.com/{TURBO_REPO}/resolve/main/{f}")
            return 1
        print(f"[自动探测] {gguf}")

    try:
        info = parse_gguf(gguf)
    except (OSError, ValueError) as e:
        print(f"[错误] 读取 GGUF 失败：{e}", file=sys.stderr)
        return 1

    meta = info["meta"]
    arch = str(meta.get("general.architecture", "") or "")
    name = str(meta.get("general.name", "") or "")
    is_turbo = "turbo" in gguf.lower() or "fl2v" in gguf.lower()
    gpu_ok, gpu_detail = detect_gpu()
    sd_cli = find_sd_cli(args.sd_dir)

    result = {
        "file": gguf, "arch": arch or "(缺失)", "name": name or "(缺失)",
        "tensor_count": info["tensor_count"], "gguf_version": info["version"],
        "is_turbo": is_turbo, "gpu": gpu_ok, "gpu_detail": gpu_detail,
        "sd_cli": sd_cli,
    }

    if arch != ARCH_MINIMAX_H3:
        result["status"] = "unsupported_arch"
        result["reason"] = f"架构 {arch!r} != {ARCH_MINIMAX_H3!r}，sd.cpp 不认。"
    elif is_turbo:
        result["status"] = "arch_match_uncertain"
        result["reason"] = ("arch==minimax_h3 但为 Turbo 4-step 蒸馏版；"
                            "sd.cpp 按权重分派 DiT/AdaLN，蒸馏版可能被拒，需真机验证。")
        result["suggest"] = "加 --live 真机 load 测试"
    else:
        result["status"] = "arch_match_base"
        result["reason"] = "arch==minimax_h3 且非 turbo，大概率是 sd.cpp 官方支持的 base H3。"
        result["suggest"] = "按 docs/minimax_h3.md 用 sd-cli 加载"

    if args.live and sd_cli and gpu_ok:
        result["live"] = live_test(sd_cli, gguf, args.timeout)
    elif args.live:
        result["live"] = {"stage": "skipped",
                          "reason": f"缺 sd-cli({sd_cli}) 或 GPU({gpu_ok})"}

    if args.json:
        print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n=== 静态判定 ===")
        for k in ("file", "arch", "name", "tensor_count", "gguf_version",
                  "is_turbo", "status"):
            print(f"  {k:<14} {result.get(k)}")
        print(f"  {'reason':<14} {result.get('reason')}")
        print(f"  {'suggest':<14} {result.get('suggest')}")
        print(f"\n=== 环境 ===")
        print(f"  GPU         {gpu_ok}  ({gpu_detail})")
        print(f"  sd-cli      {sd_cli or '(未找到)'}")
        if "live" in result:
            print(f"\n=== 真机 load ===")
            print(f"  stage       {result['live']['stage']}")
            print(f"  tail        {result['live'].get('tail', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
