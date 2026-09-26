#!/usr/bin/env python3
r"""**CLI 编码守卫** ✓ —— 拦「裸跑就崩」的测试 / 脚本 ✗。

为什么需要
  2026-09-25 实测事故 ✓✗：给上机准备的那条命令
  ``python app/scripts/h3_readiness.py`` **在本机一个字都打不出来** ✗ ——
  脚本满屏 ``✓ / ✗``，而 Windows 中文控制台是 **GBK** ✓（PEP 686 的 UTF-8 默认要
  **py3.15** ✓，本机 py3.14.5 ✗）⇒ 第一条 ``print`` 就 ``UnicodeEncodeError`` ✗。

  更坏的是**四层掩盖** ✓✗（这才是真教训 ✓）：
    1. 脚本崩 ⇒ ``--json`` 没输出 ⇒ 测试 ⑩ 判 FAIL；
    2. 而测试**自己**崩在打印 FAIL 原因上（详情里的 ``⇒`` 也是 GBK 外字符 ✗）
       ⇒ 只看得到 ``EXIT=1`` ✗，「哪条 FAIL」全丢 ✗✗；
    3. 全量自检 `run_all.py` 又给子进程灌了 ``PYTHONIOENCODING=utf-8`` ✓
       ⇒ **133 套全绿** ✓，缺的那行永远不暴露 ✗；
    4. 于是「**上机当天**才发现跑不起来」✓✗。

  记忆层的冻结按「环境变了才失效」口径留着，这类**静默掩盖**必须变成机械可查 ✓。

判定口径（与当年补丁 `_apply_gbk_fix.py` / 探针同口径 ✓ —— 只此一份 ✗）
  三条同时成立 ⇒ 报致命 ✗：
    ① 源码里出现 ``CODEC``（**显式 cp936** ✓）编不了的字符；
    ② 且真的会往 stdout 打（含 ``print(`` ✓）；
    ③ 且入口**没有** ``reconfigure`` ✓。

  ⚠️ ``CODEC`` **写死 cp936 而不是读 ``sys.stdout.encoding``** ✓：这是**目标运行平台**的硬约束 ✓，
  读本机编码会让守卫在 UTF-8 机器 / CI 上**静默失效** ✗（守卫不许随环境翻脸 ✓✗）。
  ②是近似（不查该字符是否真在 print 里 ✓）⇒ **宁严勿漏** ✓。

用法::

    python backend-py/app/scripts/check_cli_encoding.py
    python backend-py/app/scripts/check_cli_encoding.py --root <backend-py 目录>   # 负向实证用 ✓
    python backend-py/app/scripts/check_cli_encoding.py --list                    # 只列命中，不判成败 ✓

退出码：0 = 无命中；1 = 有命中（致命 N 处）。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

#: ⚠️ 显式目标编码 ✗（不读 `sys.stdout.encoding` ✓ —— 理由见文件头 ✓）
CODEC = "cp936"

#: 扫描范围（相对 backend-py ✓）：测试与 CLI 脚本 ✓
GLOBS = ("tests/*_test.py", "app/scripts/**/*.py")

MARK = "reconfigure"


def unencodable(text: str, codec: str = CODEC) -> str:
    """文本里该编码**编不了**的去重字符（升序 ✓）。"""
    bad: set[str] = set()
    for char in text:
        try:
            char.encode(codec)
        except UnicodeEncodeError:
            bad.add(char)
    return "".join(sorted(bad))


def offenders(root: pathlib.Path) -> list[tuple[str, str]]:
    """返回 [(相对路径, 触发的字符们)] ✓（空 = 干净 ✓）。"""
    hits: list[tuple[str, str]] = []
    for pattern in GLOBS:
        for path in sorted(root.glob(pattern)):
            source = path.read_text(encoding="utf-8")
            bad = unencodable(source)
            if bad and "print(" in source and MARK not in source:
                hits.append((path.relative_to(root).as_posix(), bad[:16]))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="CLI 编码守卫（裸跑崩不崩 ✓）")
    parser.add_argument("--root", help="扫描根（默认 backend-py ✓，负向实证用 ✓）")
    parser.add_argument("--list", action="store_true", help="只列命中，恒以 0 退出 ✓")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve() if args.root \
        else pathlib.Path(__file__).resolve().parents[2]
    if not root.is_dir():
        print(f"扫描根不存在：{root} ✗")
        return 1

    hits = offenders(root)
    label = {"cp936": "GBK（Windows 中文控制台 ✓）"}[CODEC]
    if args.list:
        for name, bad in hits:
            print(f"  {name:<58} {bad}")
        print(f"\n共 {len(hits)} 处（仅供参考 ✓ 未判成败 ✓）")
        return 0

    if hits:
        print(f"✗ 这些文件在 {label} 下裸跑会崩 ⇒ 入口需补 "
              f"`sys.stdout.reconfigure(encoding=\"utf-8\", errors=\"replace\")` ✓")
        print("  ⚠️ **别**改写成 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)` ✗ ——"
              " 它丢掉原 wrapper（刷新顺序与调用方乱 ✗），且本守卫认不出它 ⇒ 会一直报 ✓✗：")
        for name, bad in hits:
            print(f"  致命：{name}（触发字符 {bad} ✓）")
        print(f"\n致命 {len(hits)} 处 ✗（口径：含 {label} 外字符 + 有 print + 无 reconfigure ✓）")
        return 1

    print(f"✓ 裸跑编码：无命中 ✓（扫 {sum(len(list(root.glob(p))) for p in GLOBS)} 个文件 ✓ "
          f"口径 = 含 {label} 外字符 + 有 print + 无 reconfigure ✓）")
    print("致命 0 处")
    return 0


if __name__ == "__main__":
    # ⚠️ 本文件自己也在扫描范围内 ✓ ⇒ 这行是**必须**的 ✗（否则守卫自己先崩 ✓✗）。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
