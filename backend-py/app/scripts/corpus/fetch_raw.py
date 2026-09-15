#!/usr/bin/env python3
"""语料采集（只下载原始文件，不解析）—— 第一阶段。

> 本文件是 ``backend-py/scripts/corpus/fetch-raw.mjs`` 的**逐条对齐移植**（2026-09-15，Node 版已删）。
> 只把 ``fetch`` 换成 ``urllib``（并补一个 User-Agent，GitHub raw 对无 UA 的请求更易 403），
> 其余行为、落盘路径、退出码保持不变。

为什么单独一步
  下载受网络波动影响最大，且**必须可重跑**；解析则依赖对每个源结构的实测。
  把两者分开：本脚本只负责「把原始字节拿到本地并记账」，解析器随后按实测定稿。

落盘约定（沿用项目既有规矩，见 ``backend-py/app/scripts/README.md``）
  ``data/prompt-corpus/<源 id>/raw/<原始相对路径>``   ← 原始文件（该目录已 gitignore）
  来源 / 许可 / 采集时间以 ``SOURCES`` 表 + ``tmp/fetch-raw.log`` 留痕。

⚠️ 原始文件因第三方版权**不得入库**（CC BY 4.0 要求署名）⇒ 溯源按本文件的 ``SOURCES`` 表重建。

用法::

    python backend-py/app/scripts/corpus/fetch_raw.py             # 跳过已存在的文件
    python backend-py/app/scripts/corpus/fetch_raw.py --force     # 重新下载
    python backend-py/app/scripts/corpus/fetch_raw.py --only=<id 前缀>   # 只跑某个源

退出码：0 = 全部成功（含跳过）；1 = 有源失败（失败清单在日志末尾）。
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # depth-adjusted-to-app
CORPUS = ROOT / "data" / "prompt-corpus"
FORCE = "--force" in sys.argv
ONLY = next((arg[7:] for arg in sys.argv if arg.startswith("--only=")), "")
USER_AGENT = "voide-darme-corpus-fetch/1.0"

#: 只收**许可证明确可商用或可内部使用**的源。
#: ⚠️ 已核实但**刻意不收**的源（勿随手加回来）：
#:   - TIP-I2V                CC BY-NC 4.0（禁商用）→ 只能内部统计，不进语料库
#:   - Semonxue/awesome-video-prompts   无 LICENSE（4.88 GB）→ 仅内部参考
#:   - HitPaw-Official / geekjourneyx / fantasylights  实测为空壳（0 条数据）
#: 见 ``docs/video-prompt-data-sources.md``。
SOURCES = (
    {
        "id": "seedance-prompt-ericgood",
        "label": "Seedance Prompt (Ericgood)",
        "license": "CC-BY-4.0",
        "repo": "https://github.com/Ericgood/seedance-prompt",
        "base": "https://raw.githubusercontent.com/Ericgood/seedance-prompt/main/",
        # prompts.json 是唯一数据文件；assets/ 与 videos/ 是媒体，**不采**
        "files": ("prompts.json", "LICENSE"),
    },
    {
        "id": "awesome-seedance-2-5-flaqai",
        "label": "Awesome Seedance 2.5 (flaqai)",
        "license": "MIT",
        "repo": "https://github.com/flaqai/awesome_seedance_2_5",
        "base": "https://raw.githubusercontent.com/flaqai/awesome_seedance_2_5/main/",
        # 120 个场景分布在 7 个 markdown 里；i18n/ 15 语言暂不采（体积大、增益低）
        "files": (
            "prompts/README.md",
            "prompts/prompt-library.md",
            "prompts/extended-scenarios.md",
            "prompts/advanced-workflows.en.md",
            "prompts/creative-techniques.en.md",
            "prompts/genre-social-experiments.en.md",
            "prompts/multilingual-pack.md",
            "LICENSE",
        ),
    },
)

_log: list[str] = []


def say(message: str) -> None:
    _log.append(message)
    print(message)


def _iso_now() -> str:
    """``new Date().toISOString()`` 的同款形态（毫秒 + ``Z``）。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def fetch_one(url: str, dest: Path) -> tuple[bool, int]:
    """下载单个文件到 dest；已存在且非 ``--force`` 时跳过（保证可重入）。"""
    if dest.exists() and not FORCE:
        return True, dest.stat().st_size
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - 固定 https 源
        payload = response.read()
    if not payload:
        raise RuntimeError("empty body")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return False, len(payload)


def main() -> int:
    failed: list[str] = []
    say(f"# 语料采集 {_iso_now()}")
    say(f"# force={str(FORCE).lower()} only={ONLY or '(all)'}")

    for source in SOURCES:
        if ONLY and not source["id"].startswith(ONLY):
            continue
        say("")
        say(f"=== {source['id']}  [{source['license']}]  {source['repo']}")
        for name in source["files"]:
            dest = CORPUS / source["id"] / "raw" / name
            try:
                skipped, size = fetch_one(source["base"] + name, dest)
                say(f"  {'skip' if skipped else 'get '}  {str(size).rjust(8)} B  {name}")
            except (urllib.error.URLError, OSError, RuntimeError) as exc:
                failed.append(f"{source['id']}/{name}")
                say(f"  FAIL  {name}  -> {exc}")

    say("")
    say(f"# 失败 {len(failed)} 个：{', '.join(failed)}" if failed else "# 全部成功")

    log_path = ROOT / "tmp" / "fetch-raw.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(_log), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
