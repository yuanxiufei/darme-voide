"""S7 自检：像素级校色（``services/color_grade.py``，ffmpeg 实现）。

**基准值来自真实 sharp 实测**（``backend/probe-sharp.cjs``，2026-09-15 跑出来的），
不是照公式推的 —— 其中两条结论直接改变了实现：

* ``.gamma()`` 在**无 resize** 的链里是 **no-op**（128→127、200,100,50→199,9?,4?，只差 ±1 取整）
  ⇒ Python 侧**有意跳过** ⑥⑦ 两步（照抄公式去 ``pow()`` 反而会引入 Node 没有的色调偏移）；
* ``modulate({brightness})`` 是 **L\\* 感知域乘法**（128→199，不是 192）⇒ 两侧有约 3% 差异，
  本实现用 RGB 乘法近似并**显式记录**该差异。

⚠️ 校色是**原地改写**文件的 ⇒ 每条用例前**重建夹具**（``fresh()``），否则前一条的结果会
污染后一条（本文件第一版就是这么踩的）。

运行::

    ./.venv/Scripts/python.exe tests/color_grade_test.py
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import os

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="cgrade_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.color_grade import (  # noqa: E402
    apply_color_grade_to_file,
    build_filter_chain,
)
from app.services.file_storage import get_storage_root  # noqa: E402

_R: list[tuple[str, bool, object]] = []

#: 夹具定义：名字 -> (lavfi 源, 额外参数)
_FIXTURES: dict[str, tuple[str, list[str]]] = {
    "gray.png": ("color=0x808080:s=16x16", []),      # 中性 128
    "light.png": ("color=0xC8C8C8:s=16x16", []),     # 中性 200
    "color.png": ("color=0xC86432:s=16x16", []),     # 200,100,50
    "alpha.png": ("color=0xC86432:s=16x16", ["-vf", "format=rgba"]),
    "photo.jpg": ("color=0x808080:s=16x16", []),
}


def check(name: str, condition: object, detail: object = "") -> None:
    _R.append((name, bool(condition), detail))


def fresh(name: str) -> str:
    """重建夹具（原地校色会改写文件 ⇒ 每条用例前都要重来一次）。"""
    source, extra = _FIXTURES[name]
    directory = Path(get_storage_root()) / "images"
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", source, *extra,
                    "-frames:v", "1", "-y", str(directory / name)],
                   check=True, capture_output=True)
    return f"images/{name}"


def _probe(rel_path: str, vf: str) -> bytes:
    absolute = str(Path(get_storage_root()) / rel_path)
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", absolute, "-vf", vf, "-frames:v", "1",
         "-f", "rawvideo", "-"], check=True, capture_output=True).stdout


def avg_rgb(rel_path: str) -> tuple[int, int, int]:
    out = _probe(rel_path, "scale=1:1,format=rgb24")
    return out[0], out[1], out[2]


def avg_yuv(rel_path: str) -> tuple[int, int, int]:
    out = _probe(rel_path, "scale=1:1,format=yuv444p")
    return out[0], out[1], out[2]


def pix_fmt(rel_path: str) -> str:
    absolute = str(Path(get_storage_root()) / rel_path)
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt", "-of", "csv=p=0", absolute],
        check=True, capture_output=True, text=True).stdout.strip()


def run_grade(rel_path: str, params: dict | str | None) -> str:
    raw = params if isinstance(params, str) or params is None else json.dumps(params)
    return asyncio.run(apply_color_grade_to_file(rel_path, raw))


def graded_rgb(fixture: str, params: dict) -> tuple[int, int, int]:
    """重建夹具 -> 校色 -> 返回平均 RGB。"""
    return avg_rgb(run_grade(fresh(fixture), params))


def near(actual: tuple[int, int, int], expected: tuple[int, int, int], tol: int = 2) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(actual, expected))


def main() -> int:  # noqa: C901
    # ── 0. 中性短路：连文件都不碰 ──
    target = fresh("gray.png")
    before = (Path(get_storage_root()) / target).read_bytes()
    for neutral in (None, "{}", '{"skinTone":0}', '{"toneMapping":{"gamma":50}}',
                    '{"shadowsHighlights":{"shadows":100,"highlights":100}}'):
        check(f"中性/仅 gamma({neutral}) -> 返回同一路径",
              run_grade(target, neutral) == target)
    check("中性/仅 gamma 参数 -> **文件零改动**（gamma 两段是 no-op ⇒ 链为空、不跑 ffmpeg）",
          (Path(get_storage_root()) / target).read_bytes() == before, avg_rgb(target))
    check("坏 JSON -> 原样返回（parse 失败走 None 分支）",
          run_grade(target, "{oops") == target)

    neutral = {"colorCalibration": {"red": 0, "green": 0, "blue": 0},
               "toneMapping": {"gamma": 0}, "whiteBalance": {"temperature": 0, "tint": 0},
               "exposure": 0, "saturation": 0, "contrast": 0, "skinTone": 0,
               "shadowsHighlights": {"shadows": 0, "highlights": 0}}
    check("链构造: 中性参数返回 None（连 ffmpeg 都不起）", build_filter_chain(neutral) is None)
    mixed = build_filter_chain({**neutral, "exposure": 10, "saturation": 10})
    check("链构造: 曝光走 lutrgb、饱和度夹一段 eq",
          mixed is not None and "lutrgb" in mixed and "eq=saturation" in mixed, mixed)

    # ── 1. 精确基准（与 sharp 实测逐值一致）──
    got = graded_rgb("gray.png", {"colorCalibration": {"red": 50}})
    check("校准 red+50 @128 -> **192**,128,128（sharp 实测 192,128,128）",
          near(got, (192, 128, 128)), got)
    got = graded_rgb("color.png", {"colorCalibration": {"red": 50}})
    check("校准 red+50 @200,100,50 -> 255(钳位),100,50", near(got, (255, 100, 50)), got)
    got = graded_rgb("light.png", {"contrast": 50})
    check("对比度+50 @200 -> **236**（v*1.5-64，sharp 实测 236）", near(got, (236, 236, 236)), got)
    got = graded_rgb("color.png", {"contrast": 50})
    check("对比度+50 @200,100,50 -> 236,86,11（sharp 实测同值）", near(got, (236, 86, 11)), got)
    got = graded_rgb("gray.png", {"contrast": 50})
    check("对比度+50 @中灰 -> **不动**（锁中灰 128，sharp 同）", near(got, (128, 128, 128)), got)
    got = graded_rgb("gray.png", {"whiteBalance": {"temperature": 100}})
    check("色温+100 -> R 升 / B 降（128*1.2 / 128*0.8 ⇒ 154/102）", near(got, (154, 128, 102), 3), got)
    got = graded_rgb("gray.png", {"whiteBalance": {"tint": 100}})
    check("色调+100 -> R,B 升 / G 降（品红方向 ⇒ 141/115/141）",
          near(got, (141, 115, 141), 3), got)
    got = graded_rgb("gray.png", {"skinTone": 100})
    check("肤色 100 -> R>128、B<128（暖化 + 保护红润）",
          got[0] > 130 and got[1] > 128 and got[2] < 126, got)

    # ── 2. 近似项：曝光 ──
    got = graded_rgb("gray.png", {"exposure": 50})
    check("曝光+50 @128 -> 本实现 ≈192；**Node 实测 199**（L* 感知乘法 ⇒ 已知 3% 差异）",
          188 <= got[0] <= 196 and got[0] == got[1] == got[2], got)

    # ── 3. 饱和度（跨通道）──
    color = fresh("color.png")
    before_yuv = avg_yuv(color)
    run_grade(color, {"saturation": 100})
    after_yuv = avg_yuv(color)
    check("饱和度+100 @有色图 -> 彩度扩大（|U-128|+|V-128| 增大）",
          abs(after_yuv[1] - 128) + abs(after_yuv[2] - 128)
          > abs(before_yuv[1] - 128) + abs(before_yuv[2] - 128), (before_yuv, after_yuv))
    got = graded_rgb("light.png", {"saturation": 100})
    # ±3 容差：rgb→yuv→缩放彩度→rgb 的往返取整（灰图本身没有彩度可放大）
    check("饱和度+100 @中性灰 -> 仍中性（不会凭空造彩度）", max(got) - min(got) <= 4, got)

    # ── 4. 格式 / alpha / 原地 ──
    alpha = fresh("alpha.png")
    check("前置: 夹具确实是带 alpha 的 png（rgba）", pix_fmt(alpha) == "rgba", pix_fmt(alpha))
    run_grade(alpha, {"contrast": 20})
    check("alpha: 校色后**丢掉 alpha**（对齐 sharp 的 removeAlpha）",
          pix_fmt(alpha) == "rgb24", pix_fmt(alpha))
    photo = fresh("photo.jpg")
    run_grade(photo, {"contrast": 30})
    check("格式保留: jpg 入 -> 仍可解码为 jpg，且**采样对齐 libvips 的 4:2:0**",
          pix_fmt(photo) in ("yuvj420p", "yuv420p"), pix_fmt(photo))

    # ── 5. 失败路径：不留临时文件 ──
    missing = "images/does-not-exist.png"
    try:
        run_grade(missing, {"contrast": 20})
        check("失败: 文件不存在 -> 抛错", False, "未抛错")
    except Exception as exc:  # noqa: BLE001 —— 路径解析或 ffmpeg 任一环节
        leftover = [p.name for p in (Path(get_storage_root()) / "images").glob("does-not-exist*")]
        check("失败: 抛错且**不留 .grade 临时文件**（不会写出半成品）",
              not leftover, (str(exc)[:70], leftover))

    failed = [item for item in _R if not item[1]]
    for name, passed, detail in _R:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_R) - len(failed)}/{len(_R)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
