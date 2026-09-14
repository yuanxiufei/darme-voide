"""S7 自检：参考图压缩 ``read_image_as_compressed_data_url``（ffmpeg 实现）。

对齐原 TS（``sharp``）的语义：等比缩到「长边 ≤ maxWidth×maxHeight」的框内、**不放大** →
**有 alpha 则 flatten 白底** → JPEG → ``data:image/jpeg;base64,…``。

⭐ 关键用例是「**全透明图 ⇒ 纯白**」：这一条能区分「flatten 白底」（正确）与「直接丢掉 alpha」
（会得到黑或原色，是错的）—— 也是选 ``geq`` 而不是 ``format=rgb24`` 的理由。

⚠️ 已知差异（无法消除，已记录在 ``quality_to_qscale``）：编码器不同（mjpeg vs mozjpeg）
⇒ **字节体积不相等**（同视觉质量下 ffmpeg 约大 1.4~1.7×），本套件只锁**尺寸/语义/单调性**。

运行::

    ./.venv/Scripts/python.exe tests/compressed_data_url_test.py
"""
from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="refimg_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.file_storage import (  # noqa: E402
    get_storage_root,
    quality_to_qscale,
    read_image_as_compressed_data_url,
)

_R: list[tuple[str, bool, object]] = []

#: 夹具：名字 -> (lavfi 源, 额外参数)
#: ⚠️ alpha 夹具**不能**用 `color=black@0.0`：那个 `@` 语法在本机 ffmpeg 上不生效
#: （实测解出来 alpha=255，是**不透明黑**）⇒ 必须用 `colorchannelmixer=aa=` 显式设 alpha 平面。
_FIXTURES: dict[str, tuple[str, list[str]]] = {
    "big.png": ("testsrc2=s=1024x768", []),
    "small.png": ("color=0x3366cc:s=200x100", []),
    "tall.png": ("testsrc2=s=600x1200", []),
    "opaque.png": ("color=red:s=64x64", []),
    "alpha_half.png": ("color=red:s=64x64", ["-vf", "format=rgba,colorchannelmixer=aa=0.5"]),
    "alpha_zero.png": ("color=black:s=64x64", ["-vf", "format=rgba,colorchannelmixer=aa=0"]),
}


def check(name: str, condition: object, detail: object = "") -> None:
    _R.append((name, bool(condition), detail))


def fresh(name: str) -> str:
    directory = Path(get_storage_root()) / "images"
    directory.mkdir(parents=True, exist_ok=True)
    source, extra = _FIXTURES[name]
    pix_fmt_arg = ["-pix_fmt", "rgba"] if name.startswith("alpha_") else []
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", source,
                    *extra, *pix_fmt_arg, "-frames:v", "1", "-y", str(directory / name)],
                   check=True, capture_output=True)
    return f"images/{name}"


def alpha_of(rel_path: str) -> int:
    """夹具的**首像素 alpha 值**（只看 pix_fmt 不够 —— `format=rgba` 也可能 alpha 全是 255）。"""
    absolute = str(Path(get_storage_root()) / rel_path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", absolute, "-vf", "format=rgba", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgba", "-"], check=True, capture_output=True).stdout
    return raw[3]


def pix_fmt(rel_path: str) -> str:
    absolute = str(Path(get_storage_root()) / rel_path)
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt", "-of", "csv=p=0", absolute],
        check=True, capture_output=True, text=True).stdout.strip()


def decode(data_url: str) -> bytes:
    assert data_url.startswith("data:image/jpeg;base64,"), data_url[:40]
    return base64.b64decode(data_url.split(",", 1)[1])


def inspect(jpeg: bytes) -> tuple[int, int, tuple[int, int, int]]:
    """返回 (宽, 高, 平均 RGB)。"""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        handle.write(jpeg)
        path = handle.name
    try:
        size = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", path],
            check=True, capture_output=True, text=True).stdout.strip().split(",")
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=1:1,format=rgb24",
             "-frames:v", "1", "-f", "rawvideo", "-"],
            check=True, capture_output=True).stdout
        return int(size[0]), int(size[1]), (raw[0], raw[1], raw[2])
    finally:
        os.remove(path)


def run(name: str, options: dict | None = None) -> tuple[int, int, tuple[int, int, int]]:
    return inspect(decode(asyncio.run(
        read_image_as_compressed_data_url(fresh(name), options))))


def near(actual: tuple[int, int, int], expected: tuple[int, int, int], tol: int = 6) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(actual, expected))


def main() -> int:  # noqa: C901
    # ── 1. 尺寸语义 ──
    width, height, _ = run("big.png")
    check("尺寸: 1024×768 -> 长边夹到 768 且**保持比例**（768×576）",
          (width, height) == (768, 576), (width, height))
    width, height, _ = run("small.png")
    check("尺寸: 小图**不放大**（200×100 原样，对齐 withoutEnlargement）",
          (width, height) == (200, 100), (width, height))
    width, height, _ = run("tall.png")
    check("尺寸: 竖图按**高**受限（600×1200 -> 384×768）", (width, height) == (384, 768),
          (width, height))
    width, height, _ = run("big.png", {"maxWidth": 400})
    check("尺寸: 覆盖 maxWidth=400（且 maxHeight 仍是 768）-> 400×300",
          (width, height) == (400, 300), (width, height))
    check("尺寸: 显式 0 视为「该维不约束」（nullish 语义，不是回退默认）",
          run("small.png", {"maxWidth": 0, "maxHeight": 0})[:2] == (200, 100))

    # ── 2. alpha：flatten 白底（关键语义）──
    check("前置: 两个 alpha 夹具的 **alpha 值**分别 0 / 128（只看 pix_fmt 会被假夹具骗过）",
          alpha_of(fresh("alpha_zero.png")) == 0
          and abs(alpha_of(fresh("alpha_half.png")) - 128) <= 2,
          (alpha_of("images/alpha_zero.png"), alpha_of("images/alpha_half.png")))
    check("alpha: **全透明图 -> 纯白**（flatten 白底；丢 alpha 会得到黑/原色）",
          near(run("alpha_zero.png")[2], (255, 255, 255)), run("alpha_zero.png")[2])
    half = run("alpha_half.png")[2]
    check("alpha: 50% 红 + 白底 -> 约 (255,127,127)（线性合成）",
          near(half, (255, 127, 127), 10), half)
    check("alpha: 无 alpha 的纯红图**不被 flatten**（仍是纯红）",
          near(run("opaque.png")[2], (255, 0, 0), 10), run("opaque.png")[2])

    # ── 3. quality 与映射 ──
    high = len(decode(asyncio.run(read_image_as_compressed_data_url(
        fresh("big.png"), {"quality": 95}))))
    low = len(decode(asyncio.run(read_image_as_compressed_data_url(
        fresh("big.png"), {"quality": 20}))))
    check("质量: quality 越大体积越大（95 > 20，单调性）", high > low, (high, low))
    check("映射: quality_to_qscale 是反向标度（100->2 最好，0->31 最差，68->10）",
          quality_to_qscale(100) == 2 and quality_to_qscale(0) == 31
          and quality_to_qscale(68) == 10, [quality_to_qscale(v)
                                            for v in (100, 85, 68, 20, 0)])
    check("映射: 非法/缺失 quality 回退到 68 的口径（不抛错）",
          quality_to_qscale(None) == 10 and quality_to_qscale("abc") == 10)

    # ── 4. options 缺省与输出形态 ──
    check("缺省: options=None / {} 等价（768 长边 + q68）",
          run("big.png", None)[:2] == (768, 576) and run("big.png", {})[:2] == (768, 576))
    raw = asyncio.run(read_image_as_compressed_data_url(fresh("opaque.png")))
    check("形态: 前缀 `data:image/jpeg;base64,` 且可解出合法 JPEG（FFD8 魔数）",
          raw.startswith("data:image/jpeg;base64,") and decode(raw)[:2] == b"\xff\xd8", raw[:30])

    # ── 5. 失败路径与临时文件 ──
    images = Path(get_storage_root()) / "images"
    try:
        asyncio.run(read_image_as_compressed_data_url("images/does-not-exist.png"))
        check("失败: 文件不存在 -> 抛错（对齐 sharp 抛错）", False, "未抛错")
    except Exception as exc:  # noqa: BLE001
        leftover = [p.name for p in images.glob("does-not-exist*")]
        check("失败: 抛错且**不留 .ref.jpg 临时文件**", not leftover, (str(exc)[:60], leftover))
    leftovers = [p.name for p in images.glob("*.ref.jpg")]
    check("成功路径也**不留临时文件**（用完即删）", not leftovers, leftovers)

    failed = [item for item in _R if not item[1]]
    for name, passed, detail in _R:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_R) - len(failed)}/{len(_R)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
