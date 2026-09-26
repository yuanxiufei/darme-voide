"""**素材磁盘操作** ✓ —— 扫图 / 改名 / 转格式 / 标签读写（LoRAMaster 移植 ✓）。

出处（参考实现 ✓）：

* 改名 —— ``reference/lora/LoRAMaster/dataset_manager/image_rename.py`` 的 ``run_rename()`` ✓
  （第 87~127 行 ✓）；
* 转格式 —— 同目录 ``image_convert.py`` 的 ``convertSuffix()`` ✓（第 86~192 行 ✓）；
* 扫图与标签读写 —— ``AutoCaptioning.py`` 的 ``loadDataSets()`` ✓（第 539~576 行 ✓）与
  ``save_caption_to_txt()`` ✓（第 319~323 行 ✓）。

## ⚠️ **改了参考实现四个真缺陷** ✓（每一处都指得出行号 ✓，不是"我觉得更好" ✗）

1. **备份目录会自己抄自己** ✗✗（``image_convert.py`` 第 108~115 行 ✓）：
   参考把备份建在 ``<素材目录>/loramaster_backup`` ✓ —— 也就是**被备份目录的里面** ✓✗。
   ``shutil.copytree(src, src/xxx)`` 会先把 ``src/xxx`` 建出来 ✓，然后扫描 ``src`` 时
   **正好看见它** ✓ ⇒ 递归下去直到路径超长报错 ✓✗（备份要么报错、要么白跑 ✓）。
   本仓挪到**同级**：``<素材目录的父目录>/<素材目录名>_loramaster_backup_<时间戳>`` ✓。
2. **改名会覆盖别的文件** ✗（同上第 120 行 ✓）：``os.rename`` 在 POSIX 上**静默覆盖**同名文件 ✓✗、
   在 Windows 上抛错 ✓ ⇒ 同一份代码两种行为 ✓✗。本仓先**算出全部目标名并查冲突** ✓，
   有冲突就**一张都不动**直接报错 ✓（两阶段：先规划、后落地 ✓）。
3. **改名不管 ``.txt`` 标签** ✗（同上 ✓）：参考只改图片 ✓ ⇒ 改完名，
   与图片同名的标签文件就**全部对不上了** ✓✗（素材没丢，标签丢了 ✓ 更隐蔽 ✓）。
   本仓默认**连同名 ``.txt`` 一起改** ✓（开关 ``rename_caption`` ✓，可在界面上关掉 ✓）。
4. **转格式写死删原图** ✗（同上第 126 行 ``remove_original = True`` ✓）：
   用完就把素材删了 ✓✗。本仓把它提成参数 ✓，**默认不删** ✓（见 ``fields.py`` 第 2 条 ✓）。

⚠️ 另外两条**如实记录**的差异 ✓（不算缺陷 ✓，但现象不一样 ✓，不写清会让人对不上 ✓）：

* **目标格式与源格式相同** ⇒ 参考会"读进来再原样写回去"一回 ✓（等于重压缩一遍 ✓✗）；
  本仓直接**跳过并计数** ✓（``skipped`` ✓）。
* **``quality``** 只对 JPEG / WebP 这类**有损**格式有意义 ✓ ⇒ 本仓只在这两种格式上传 ✓
  （参考实现压根没传过 ``quality`` ✓ —— 它声明了 ``quality = 95`` 却没用 ✓✗）。
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..errors import LoraTrainConfigError
from .fields import CAPTION_IMAGE_EXTENSIONS, CONVERT_INPUT_EXTENSIONS

#: 收到一条进展就调一次 ✓（不许抛 ✗）
Emit = Callable[[str], None]
#: 该不该停 ✓（每张图问一次 ✓）
ShouldStop = Callable[[], bool]

#: 有损格式 ✓（只有它们认 ``quality`` ✓）
_LOSSY_FORMATS = {"JPEG", "WEBP"}

#: 目标后缀 → PIL 格式名 ✓（参考实现只特判了 ``JPG -> JPEG`` ✓，见 ``image_convert.py`` 第 157~159 行 ✓）
_FORMAT_ALIASES = {"JPG": "JPEG", "TIF": "TIFF", "JPEG": "JPEG", "PNG": "PNG", "BMP": "BMP",
                   "TIFF": "TIFF", "WEBP": "WEBP"}


def _noop(_message: str) -> None:
    return None


# ---------------------------------------------------------------------------
# 扫图与标签读写 ✓（对应 AutoCaptioning.loadDataSets / save_caption_to_txt ✓）
# ---------------------------------------------------------------------------


def require_directory(path: str | Path, *, what: str = "素材文件夹") -> Path:
    """把用户给的目录**验成真有这个目录** ✓ ⇒ 绝对路径 ✓；不是目录就报错点名 ✓。"""
    raw = str(path or "").strip()
    if not raw:
        raise LoraTrainConfigError(f"没填{what} ✗")
    target = Path(raw).expanduser()
    if not target.is_dir():
        raise LoraTrainConfigError(f"{what}不是个目录 ✗：{target}（存在={target.exists()}）")
    return target.resolve()


def list_images(directory: Path, *, extensions: Sequence[str] = CAPTION_IMAGE_EXTENSIONS,
                recursive: bool = False) -> list[Path]:
    """列图 ✓（按名字排序 ✓ —— 参考实现也 ``files.sort()`` ✓，顺序决定编号 ✓）。"""
    wanted = {ext.lower() for ext in extensions}
    walker: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    found = [p for p in walker if p.is_file() and p.suffix.lower() in wanted]
    found.sort(key=lambda p: str(p).lower())
    return found


def caption_path_for(image: Path) -> Path:
    """图片 → 与它同名的 ``.txt`` ✓（参考实现是 ``splitext(basename)[0] + ".txt"`` ✓）。"""
    return image.with_suffix(".txt")


def read_caption(image: Path) -> str:
    """读标签 ✓；没有标签文件 ⇒ 空串 ✓（**这不是兜底** ✗：未打标就是没有标签 ✓）。"""
    target = caption_path_for(image)
    if not target.is_file():
        return ""
    return target.read_text(encoding="utf-8", errors="replace").strip()


def write_caption(image: Path, text: str) -> Path:
    """写标签 ✓ ⇒ 落了哪个文件 ✓（覆盖写 ✓，与参考实现一致 ✓）。

    ⚠️ 图**不存在**就报错 ✗（本仓加的 ✓）：否则一个写错的路径会在素材目录里留下
    一堆**没有对应图片的孤儿 ``.txt``** ✓✗（打标工具扫不到它们 ✓，但训练脚本会读到 ✓✗）——
    宁可当场报错 ✓。
    """
    if not image.is_file():
        raise LoraTrainConfigError(
            f"要写标签的图不存在 ✗：{image}\n"
            "⇒ 不写孤儿 .txt ✓（打标/触发词只在**扫到的图**上写 ✓）"
        )
    target = caption_path_for(image)
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


# ---------------------------------------------------------------------------
# 改名 ✓（对应 image_rename.run_rename ✓）
# ---------------------------------------------------------------------------


@dataclass
class RenamePlan:
    """改名**规划**的结果 ✓ —— 落地之前就把它算完 ✓（见模块头第 2 条 ✓）。"""

    #: ``(原路径, 新路径)`` ✓
    moves: list[tuple[Path, Path]] = field(default_factory=list)
    #: ``(原标签路径, 新标签路径)`` ✓
    caption_moves: list[tuple[Path, Path]] = field(default_factory=list)
    #: 已经叫对了名字、不用动 ✓
    unchanged: list[Path] = field(default_factory=list)
    #: 会撞上的目标路径 ✓（**非空 ⇒ 一张都不许动** ✓）
    conflicts: list[Path] = field(default_factory=list)
    #: 扫到的图片总数 ✓
    scanned: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "scanned": self.scanned,
            "renamed": len(self.moves),
            "captions": len(self.caption_moves),
            "unchanged": len(self.unchanged),
            "conflicts": [str(p) for p in self.conflicts],
        }


def plan_rename(directory: Path, *, suffix: str, prefix: str, digits: int,
                rename_caption: bool = True,
                extensions: Sequence[str] = CAPTION_IMAGE_EXTENSIONS) -> RenamePlan:
    """算出「哪些图要改成什么」（**只算不动** ✓）✓ ⇒ :class:`RenamePlan` ✓。

    ⚠️ 冲突判定用的是**改名之后那张表** ✓：只在"要搬进去"的位子**被别人占着**时报冲突 ✓；
    「那个位子属于另一张**自己也在这批里挪窝**的图」**不算冲突** ✓ ——
    那种情况交给 :func:`apply_rename` 先腾位 ✓（见该函数说明 ✓），而**不是**让整批停下来 ✗。
    ⚠️ 判"被占着"用的是 ``exists()`` ✓（不是"在扫描到的图里" ✗）：
    一个叫 ``0002.jpg`` 的**目录**、或者后缀大小写古怪的东西 ✓，同样是"过不去的位子" ✓✗。
    """
    if digits <= 0:
        raise LoraTrainConfigError(f"命名位数要是正整数 ✗：{digits}")
    normalized = suffix.strip().lstrip(".")
    if not normalized:
        raise LoraTrainConfigError("没填素材格式 ✗（如 jpg）")

    wanted = f".{normalized.lower()}"
    # ⚠️ 用 ``suffix()`` 比后缀而不是 ``Path.match("*.jpg")`` ✗：后者在 Windows 上不分大小写、
    #    在 Linux 上分 ✓✗ ⇒ 「同一份素材两台机器扫出来的张数不一样」✓✗
    # ⚠️ 这里**不**套 :data:`CAPTION_IMAGE_EXTENSIONS` ✗：改名的目标后缀是用户随便填的 ✓，
    #    参考实现也是 ``glob(f"*{suffix}")`` 不给白名单 ✓ ⇒ 保持一致 ✓。
    images = [p for p in list_images(directory, extensions=(wanted,)) if p.suffix.lower() == wanted]
    plan = RenamePlan(scanned=len(images))

    targets: list[tuple[Path, Path]] = []
    for index, image in enumerate(images, start=1):
        new_name = f"{prefix}{str(index).zfill(digits)}.{normalized}"
        new_path = image.with_name(new_name)
        if new_path == image:
            plan.unchanged.append(image)
            continue
        targets.append((image, new_path))

    # ⚠️ 撞车检查 ✓：把"改完之后有哪些路径"整个算出来 ✓，重名就报冲突 ✓
    moving = {old for old, _ in targets}
    seen: dict[Path, Path] = {}
    for old, new in targets:
        # 位子被**不会挪窝**的东西占着（含同名目录 ✓）⇒ 冲突 ✓；
        # 被"这批里也要挪窝的图"占着 ⇒ 不算冲突 ✓（apply_rename 会先腾位 ✓）
        if new.exists() and new not in moving:
            plan.conflicts.append(new)
            continue
        if new in seen:
            plan.conflicts.append(new)  # 两个原文件想叫同一个名字 ✓（理论上做不到 ✓，照样查 ✓）
            continue
        seen[new] = old

    if plan.conflicts:
        return plan

    plan.moves = targets
    if rename_caption:
        for old, new in targets:
            old_caption = caption_path_for(old)
            if old_caption.is_file():
                plan.caption_moves.append((old_caption, caption_path_for(new)))
    return plan


def apply_rename(plan: RenamePlan, *, emit: Emit | None = None) -> dict[str, int]:
    """把 :func:`plan_rename` 的结果真正落到磁盘 ✓ ⇒ 计数 ✓。

    ⚠️ 规划里有冲突 ⇒ **直接报错、一个文件都不动** ✗✗（见模块头第 2 条 ✓）。

    ⚠️ ⭐ **先腾位、再挪窝** ✗✗（本仓补的一步 ✓，理由是真会丢数据 ✓）：
    ``Path.replace`` 是**覆盖**语义 ✓ ⇒ 若「某张图要占的位子」正被「另一张**自己也要挪窝**的图」
    占着 ✓，那条 ``replace`` 会**先把被占位那张盖掉** ✓✗ ⇒ 等轮到它挪窝时，挪走的是**刚盖上去的别人** ✓✗✗
    （原内容已经没了 ✓）。⇒ 凡是这种"自己也要让位"的目标 ✓，先把占位那份挪到**同目录临时名** ✓
    （同目录 ⇒ 不跨盘 ⇒ 还是元数据操作 ✓ 秒级 ✓），轮到时从临时名挪到正式名 ✓。
    标签 ``.txt`` 一起进一起出 ✓（临时名保留后缀 ⇒ :func:`caption_path_for` 照样算得对 ✓）。
    """
    log = emit or _noop
    if plan.conflicts:
        raise LoraTrainConfigError(
            "目标名会覆盖已有文件 ✗，所以**一个都没改**：\n  - "
            + "\n  - ".join(str(p) for p in plan.conflicts[:10])
            + (f"\n（共 {len(plan.conflicts)} 处）" if len(plan.conflicts) > 10 else "")
        )

    sources = {old for old, _ in plan.moves}
    staged: dict[Path, Path] = {}
    for index, (_old, new) in enumerate(plan.moves):
        if new not in sources or not new.exists():
            continue
        temp = _staging_name(new, index)
        log(f"先腾位：{new.name} → {temp.name}（它自己也要挪窝 ✓，所以先让开 ✓）")
        new.replace(temp)
        staged[new] = temp
        old_caption = caption_path_for(new)
        if old_caption.is_file():
            old_caption.replace(caption_path_for(temp))

    done = 0
    captions = 0
    for old, new in plan.moves:
        start = staged.get(old, old)
        start.replace(new)
        log(f"改名：{old.name} → {new.name}")
        done += 1
        if plan.caption_moves:
            caption = caption_path_for(start)
            if caption.is_file():
                caption.replace(caption_path_for(new))
                captions += 1
    for image in plan.unchanged:
        log(f"跳过（名字本来就对 ✓）：{image.name}")
    log(f"完成：改名 {done} 张 ✓、标签 {captions} 个 ✓、跳过 {len(plan.unchanged)} 张 ✓")
    return {"renamed": done, "captions": captions, "unchanged": len(plan.unchanged)}


def _staging_name(target: Path, index: int) -> Path:
    """腾位用的**同目录**临时名 ✓（保留后缀 ✓ ⇒ :func:`caption_path_for` 照样算得对 ✓）。"""
    candidate = target.with_name(f"{target.stem}.lora-staging-{index}{target.suffix}")
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = target.with_name(f"{target.stem}.lora-staging-{index}-{counter}{target.suffix}")
    return candidate


def run_rename(directory: Path, *, suffix: str, prefix: str, digits: int,
               rename_caption: bool = True, emit: Emit | None = None) -> dict[str, object]:
    """规划 + 落地 ✓ ⇒ 事实 ✓（含**规划本身**的计数 ✓）。"""
    log = emit or _noop
    plan = plan_rename(directory, suffix=suffix, prefix=prefix, digits=digits,
                       rename_caption=rename_caption)
    log(f"扫到 {plan.scanned} 张 *.{suffix.strip().lstrip('.')} ✓，"
        f"其中 {len(plan.moves)} 张要改名 ✓、{len(plan.unchanged)} 张名字已对 ✓")
    if plan.conflicts:
        # ⚠️ **必须**走 :func:`apply_rename` 把那个冲突错误抛出来 ✗✗ —— 冲突时 ``plan.moves``
        #    是空的 ✓ ⇒ 如果这里按"没东西要改"直接 return ✓，用户会看到一句"没有需要改名的文件" ✓✗
        #    而**真实原因是撞车** ✓✗（静默失败正是本仓纪律不许的 ✗）。文案只有一处 ✓。
        apply_rename(plan, emit=log)
    if not plan.moves:
        log("没有需要改名的文件 ✓")
        return plan.as_dict()
    counts = apply_rename(plan, emit=log)
    return {**plan.as_dict(), **counts}


# ---------------------------------------------------------------------------
# 转格式 ✓（对应 image_convert.convertSuffix ✓）
# ---------------------------------------------------------------------------


def backup_directory(dataset: Path) -> Path:
    """备份目录 ✓ —— 放在**素材目录的同级** ✓（见模块头第 1 条 ✓）。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return dataset.parent / f"{dataset.name}_loramaster_backup_{stamp}"


def require_pillow():  # pragma: no cover - 只在真跑的时候用 ✓
    try:
        from PIL import Image  # noqa: PLC0415 - 惰性导入 ✓（只做改名/扫图的路径不该被 PIL 拖着 ✓）
        return Image
    except ImportError as err:
        raise LoraTrainConfigError(
            "转格式需要 Pillow ✗ —— 当前解释器里 import PIL 失败 ✓："
            f"{err}\n⇒ 装到**跑后端的那个**解释器里（pip install pillow）✓"
        ) from err


def run_convert(dataset: Path, *, target_suffix: str, backup: bool = True,
                remove_original: bool = False, overwrite: bool = True, quality: int = 95,
                emit: Emit | None = None, should_stop: ShouldStop | None = None) -> dict[str, object]:
    """递归把素材转成目标格式 ✓ ⇒ 计数事实 ✓（``scanned/converted/skipped/failed`` ✓）。

    ⚠️ 顺序**必须**是「先备份 ✓、再转 ✓、最后才（可选地 ✓）删原图」✗✗：
    参考实现就是这个顺序 ✓，本仓保持 ✓ —— 反过来会在没备份的情况下先删 ✓✗。

    ⚠️ ``should_stop`` 是**本仓加的** ✓（参考没有 ✗）：一次转几万张要跑很久 ✓✗，
    用户在界面上点停却停不下来是很糟的体验 ✓。中断点放在**每张图之间** ✓ ⇒
    停下来时已经转好的那些是完整的 ✓（而且备份还在 ✓）✓。
    """
    log = emit or _noop
    stop = should_stop or (lambda: False)
    Image = require_pillow()

    normalized = target_suffix.strip().lstrip(".").lower()
    if not normalized:
        raise LoraTrainConfigError("没填目标格式 ✗")
    pil_format = _FORMAT_ALIASES.get(normalized.upper())
    if pil_format is None:
        raise LoraTrainConfigError(
            f"不认识的目标格式 ✗：{normalized!r}；本仓认这些：{sorted(_FORMAT_ALIASES)}"
        )
    quality = max(1, min(100, int(quality)))

    if backup:
        destination = backup_directory(dataset)
        log(f"开始备份到（素材目录**之外** ✓）：{destination}")
        shutil.copytree(dataset, destination)
        log(f"备份完成 ✓：{destination}")
    else:
        log("⚠️ 没开备份 ✓ —— 转格式会**就地**改文件 ✓（原图是否保留只由「转完删掉原文件」决定 ✓）")

    wanted = {ext.lower() for ext in CONVERT_INPUT_EXTENSIONS}
    scanned = converted = skipped = 0
    stopped = False
    failures: list[tuple[str, str]] = []
    for source in sorted(dataset.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in wanted:
            continue
        if stop():
            stopped = True
            log(f"收到停止 ✓ —— 已在 {scanned} 张处收手（已转好的都完整 ✓，备份也在 ✓）")
            break
        scanned += 1
        if source.suffix.lower() == f".{normalized}":
            skipped += 1
            log(f"跳过（已经是 {normalized} ✓）：{source.name}")
            continue
        target = source.with_suffix(f".{normalized}")
        if target.exists() and not overwrite:
            skipped += 1
            log(f"跳过（目标已存在 ✓）：{target.name}")
            continue
        try:
            with Image.open(source) as image:
                if getattr(image, "is_animated", False):
                    # 动图取第一帧 ✓（参考实现第 150~154 行 ✓）—— 多帧一律只留第一帧 ✓
                    try:
                        image.seek(0)
                    except (OSError, EOFError):
                        pass
                rgb = image.convert("RGB")
                save_kwargs = {"quality": quality} if pil_format in _LOSSY_FORMATS else {}
                rgb.save(target, pil_format, **save_kwargs)
        except Exception as err:  # noqa: BLE001 - 一张图失败不该中断整批 ✓（但**要记下来** ✓）
            failures.append((str(source), f"{type(err).__name__}: {err}"))
            log(f"✗ 转换失败：{source.name} —— {type(err).__name__}: {err}")
            continue
        converted += 1
        log(f"✓ 转换：{source.name} → {target.name}")
        if remove_original and source.resolve() != target.resolve():
            try:
                source.unlink()
                log(f"  已删原文件：{source.name}")
            except OSError as err:
                failures.append((str(source), f"删除原文件失败：{err}"))
                log(f"  ✗ 删原文件失败：{source.name} —— {err}")

    log(f"----- {'已停止' if stopped else '完成'} ----- 扫到 {scanned} ✓、转了 {converted} ✓、"
        f"跳过 {skipped} ✓、失败 {len(failures)} ✓")
    for path, reason in failures[:20]:
        log(f"  ✗ {path} => {reason}")
    return {"scanned": scanned, "converted": converted, "skipped": skipped,
            "failed": len(failures), "failures": [{"path": p, "reason": r} for p, r in failures[:20]],
            "targetSuffix": normalized, "quality": quality, "stopped": stopped,
            "backup": str(backup_directory(dataset)) if backup else None,
            "removedOriginal": bool(remove_original)}


# ---------------------------------------------------------------------------
# 打标结果统计 ✓
# ---------------------------------------------------------------------------


def caption_stats(directory: Path) -> dict[str, int]:
    """这个素材目录的标签覆盖情况 ✓（给界面用 ✓）。"""
    images = list_images(directory)
    labeled = sum(1 for image in images if caption_path_for(image).is_file())
    return {"images": len(images), "labeled": labeled, "missing": len(images) - labeled}
