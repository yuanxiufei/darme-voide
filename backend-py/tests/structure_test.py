"""**结构守卫**（2026-10-30 立）：把「分层规范」从纸上搬进判据 ✓。

## 为什么要有这一套

仓库的结构约定一直写在 ``backend-py/README.md`` §目录结构 与 ``engine/__init__.py`` 模块头里 ✓，
但**没人核** ✗ —— 于是同类事故反复出现，而且每次都长得一样 ✓：

* 2026-09 把 ``scripts/`` 从仓库根搬进 ``app/scripts/`` ✓，根目录却留下一个只剩 ``.pyc`` 的空壳 ✗
  （README 明写「仓库根没有 ``scripts/`` 了」✗ —— 纸面对了，盘上没对）；
* ``app/services/agents/`` 整体改名成 ``app/agent/`` ✓，三个模块的说明还指着旧路径 ✗
  （``services/agents/skills.py`` 早就不存在 ✓，读者照着找一辈子也找不到 ✗）；
* ``engine/`` 有 63 个模块 ✓，``__all__`` 只登记了 55 个 ✗ ⇒ 8 个模块在「能力清单」上不存在 ✗
  （``bridge`` / ``facade`` / ``runtime`` / ``lora`` / ``quant_nvfp4`` / ``tokenizer_gguf`` /
  ``vae_h3`` / ``keyframes`` ✓），谁按 ``__all__`` 找功能就会漏掉它们 ✗。

这四条的共同点：**都没有任何测试会失败** ✗。本套就是补这四个洞 ✓。

## 判据（每条都反向验过：喂坏数据必须报 ✓）

㊀ :func:`registry_problems` —— ``MODULE_GROUPS`` 与盘上模块**双向相等**、无重复登记 ✓
㊁ :func:`heavy_imports` —— ``GROUP_NUMERIC`` 顶层**不许** ``import torch`` ✓
   （那是这组唯一的卖点：**不装 torch 也能测** ✓，破了就等于把 10 个模块偷偷降级 ✗）
㊂ :func:`docstring_problems` —— 每个 engine 模块的模块头必须真写清职责 ✓（≥8 词 ✓）
㊃ :func:`root_problems` —— 仓库根不许有 ``scripts/`` 空壳 ✓、不许有空目录 ✓
㊄ :func:`dangling_refs` —— 说明/注释里点名的 ``app/…py`` **必须存在** ✓（防路径腐烂 ✓）
㊅ :func:`unwired_problems` —— 每个 engine 模块都得有**生产**调用者 ✓，或登记进
   ``UNWIRED`` 写明「该接到谁身上、为什么现在允许断着」✓
   （这类的坑不是占地方，是**看起来像已经能用** ✗ —— 2026-10-30 实测 3 个模块只在测试里活着 ✓）
㊆ :func:`map_gaps` —— ``docs/feature-map.md`` 要点到 ``app/`` 下**每一个** ``.py`` ✓
   （㊄ 只管「点到的必须存在」✗ 反向没人管 ⇒ 2026-10-30 实测 27 个模块从没被点名 ✓）
㊇ :func:`test_list_gaps` —— 盘上每份 ``tests/*_test.py`` 都要进 ``run_all.py::TESTS`` ✓
   （漏登记不是少跑一次，是**永远不跑还没人知道** ✗ —— 2026-10-31 实测
   ``lora_train_test.py``（85 项）在清单外待了一个多月 ✓）
㊈ :func:`summary_gaps` —— 登记进清单的套件，收尾行必须是 ``SUMMARY: n/m passed`` ✓
   （``run_all`` 只认这个字面量；写成 ``SUMMARY 12/12 通过 ✓`` / 「全部通过」⇒
   全量表里这套的**项数恒为 0**、退出码却仍 0 ✗ —— 2026-11-01 实测 9 套中招 ✓）
㊉ :func:`hygiene_problems` —— **生成物不许入库**：``.gitignore`` 必须**整目录**挡住
   ``backend-py/output/``（真跑产物）、``backend-py/_bench/``（草稿区）、
   ``models/tokenizers/``（16 MB 词表）✓；``backend-py/`` **包根只许**有
   ``README.md`` / ``requirements*.txt`` / ``_run_qwen_image.py`` ✓
   （2026-11-04 实测：根上躺着 5 个「跑完就删」的探针 ✗、三处生成区 36+95+9 件全在忽略之外
   ⇒ 一次 ``git add -A`` 就整片入库 ✗ —— 这类坑不报错、只是**悄悄变胖** ✓）

运行::

    ./.venv/Scripts/python.exe tests/structure_test.py
"""
from __future__ import annotations

import ast
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import MODULE_GROUPS  # noqa: E402
from app.services.engine import __all__ as ENGINE_ALL  # noqa: E402

ENGINE_DIR = BACKEND_PY / "app" / "services" / "engine"
APP_DIR = BACKEND_PY / "app"
REPO_ROOT = BACKEND_PY.parent

#: 顶层 import 了这些就算「重依赖」✓（装了才跑 ⇒ 不许出现在 zero-dep 组 ✓）
HEAVY_ROOTS = ("torch", "torchvision", "transformers", "safetensors", "numpy", "PIL", "cv2")

#: 模块头的**职责下限** ✓（8 词以下＝没写 ✓，不是「写得简洁」✗）
MIN_DOC_WORDS = 8

#: ㊉ 仓库里**允许长大的生成区**（不是代码、不是配置 ⇒ 必须被 ``.gitignore`` 整目录挡住 ✓）。
#: 2026-11-04 实测三处都在忽略之外：``output/`` 36 件真跑产物 ✓、``_bench/`` 95 件草稿 ✓、
#: ``models/tokenizers/`` 16 MB 词表（代码零引用 ✓）⇒ 一次 ``git add -A`` 就整片入库 ✗。
GENERATED_DIRS = {
    "backend-py/output/": "引擎真跑产物（PNG / MP4 / 一次性 JSON 快照）",
    "backend-py/_bench/": "调试草稿区（一次性 probe 脚本与其账）",
    "models/tokenizers/": "下载的分词器词表（约 16 MB，代码零引用）",
}

#: ㊉ ``backend-py/`` **包根只许**这几个松散文件 ✓。
#: ``.py`` 落在包根＝既不在 ``app/scripts/``（工具链）、也不在 ``_bench/``（草稿区）⇒
#: 没人知道它该不该跑 ✗；2026-11-04 实测根上正躺着 5 个「跑完就删」的探针 ✗。
PKG_ROOT_FILES = frozenset({"README.md", "requirements.txt", "requirements-ocr.txt",
                            "_run_qwen_image.py"})

#: 说明里**故意**点名的路径（本来就不该存在 ✓ —— 只有「讲历史」与「造夹具」两种正当理由 ✓）
#:
#: ⚠️ 这条清单本身是个诚实的妥协 ✓：本判据**分不清**「腐烂的旧路径 ✗」与「讲历史必须点名 ✓」——
#: 两者写法完全一样 ✓。所以宁可留几位具名的例外，也不许把判据放宽成「找不到就算了」✗
#: （那等于把㊄ 直接废掉 ✓）。新条目必须写清为什么该路径不该存在 ✓：
#: * ``app/config.py`` —— 讲「配置早已搬去 ``app/core/config.py``」的历史 ✓
#: * ``app/scripts/fake.py`` —— 反验夹具里编的假路径 ✓（见 :func:`case_refs`）
#: * ``app/agent/skill_parser.py`` —— 2026-10-30 **删掉**的转发壳 ✓（说明里必须点名，
#:   否则读者只会看到一个凭空消失的模块 ✗）
_REF_ALLOW = ("app/config.py", "app/scripts/fake.py", "app/agent/skill_parser.py")

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


#: ㊅ 「只在测试里活着」的 engine 模块 ⇒ **欠账清单**（不是许可清单 ✗）。
#:
#: 每一条都必须回答两问：**它本来该接到谁身上** ✓、**为什么现在允许断着** ✓。
#: 表里的模块＝「已实现、口径已被测试钉死、但生产代码 0 处 import」——
#: 这类模块最危险的不是占地方，而是**看起来像已经能用** ✗（2026-10-30 三份体检结论）。
UNWIRED: dict[str, str] = {
    "image_ops": "图像算子库（重采样/裁剪/贴回）；预期接入 SDXL 与 DiT 的解码后处理 ⇒ 等通用骨架前向落地再接",
    "mappings": "权重键名映射表；预期接入 loader / hybrid_load 的跨族命名转换 ⇒ 等第二族骨架接入时用",
    # ⚠️ 2026-11-06 **已删** Music3 那条欠账（`te_music3` 与 `vae_music3` ✓）：
    #    第十族的装配线（`runtime._ensure_music3_loaded`）、音频 plan（`pipeline._fill_music3_plan`）、
    #    条件编码（`torch_backend._encode_music3_condition`）、解码（`decode` 的 `music3-real` 支）
    #    与出盘（`write` 的纯音频支）都已接通 ✓ —— 三处都由 `pipeline` 真 import ✓。
    #    守它们的判据换成了 `engine_music3_wiring_test`（A–G 七组：两把时间尺子 / 秒→帧三档 /
    #    引导散文 / 清单口径 / 接线顺序 / 装配拒绝面 / 路由键表 ✓）。
    # ⚠️ 同族的 `dit_music3` **不在**这张表 ✓ —— 它接在 `dit_build` 的 `music3` 行上
    #    （`BUILDERS` 静态 import ⇒ 建壳与前向口径都走那一行 ✓）；写进来就会被反向判
    #    「该删条目」✗（白名单只增不减就成谎话 ✗）。

    # ⚠️ 2026-11-01 **已删** `qwen25vl_vision` 那条欠账 ✓ —— 它就是「Edit 2511 通路动工时
    #    第一件事」✓：`torch_backend.encode_text` 里参考图过视觉塔 ⇒ `image_embeds` ⇒
    #    `<|image_pad|>` 整段替换 ✓；`denoise` 每一步带 `ref_latents` ✓；`phase_parts` 让
    #    视觉塔只在 encode 段在场 ✓（`runtime._ensure_qwen_image_loaded` 的 `is_edit` 分支 ✓）。
    #    守它的判据换成了 `engine_qwen25vl_vision_test` ⑪ 组（接线点源码级核对 ✓）。
    # ⚠️ 2026-11-01 **已删** `win_io` 那条欠账 ✓：它现在接在 `safetensors.TensorReader` 上 ✓
    #    （`_open_raw` 建无缓冲读口、`read` / `read_many` 走它、建不起来就退回 seek+read ✓）；
    #    真权重 A/B：`z_image_bf16.safetensors` 11.46 GiB / 453 张量，2.88 ⇒ 3.43 GiB/s ✓，
    #    逐字节相同 ✓（`engine_win_io_test` ⑨ 组）。
    # ⚠️ 「prefetch」已于 2026-11-01 **接上**（`weights._load_streaming` 的逐张量读 ⇒ 见下方说明 ✓）：
    #    接线点是「读第 i 件时把第 i+1..i+2 件交给后台读」✓，作用域由 `_prefetch_scope` 管 ✓，
    #    出函数即 `close(wait=True)` ⇒ **跨阶段预取被明确否掉** ✗（换装第一步就是 `release_part`，
    #    预取的字节会在「退干净」后指向已经不要的东西 ✗✗，且峰值要多吃一整件 ✗）。
    # ⚠️ 「scheduler」不在本表 ✓：它已经接在 `torch_backend` 上（同日在飞的那批工作 ✓）——
    #    写进来反而会被反向判「该删条目」✗（白名单只增不减就成谎话 ✗）。
    # ⚠️ 2026-11-01 **已删** H3 文本口三份（`qwen3vl_te` / `qwen3vl_vision` / `h3_encode` ✓）：
    #    它们现在都接在 `torch_backend.attach_h3_text` 上 ✓ —— 同一个 int8 文件里装两半塔 ✓、
    #    由 `h3_encode.combine` 合成一个编码器 ✓、`_encode_h3_condition` 出 `hidden` ＋ `tags` ✓；
    #    装配线在 `runtime.ensure_loaded(stage="h3")` 的第四条（`textWeights` ✓）。
    #    ⚠️ 只剩一件事**仍然断着** ✗：`ref2va` 的**音频/视频参考**（见 `_h3_pieces` 里两条当场拒 ✓），
    #    那不是"模块没接线"，是通路本身没做 ⇒ 该报在调用口，不该挂在这张表上 ✗。
    # ⚠️ 2026-10-31 已删 `dit_lumina` 那条欠账 ✓：它现在接在 `torch_backend._build_zimage_module`
    #    上 ✓（判族 ⇒ `zimage` ⇒ `LuminaDiT` ✓；前向口径按 `_ditFamily` 分 ✓）。
    # ⚠️ 2026-10-31 已删 `latent_formats` 那条欠账 ✓：`vae_wan.py` 接线时按族取规格 ✓
    #    （`latent_formats.QwenImage21` 的 mean/std ⇒ 解码前反缩放 ✓）⇒ 它不再是「0 处 import」✓。
    # ⚠️ 2026-11-01 **已删** `dit_seedvr2` 那条欠账 ✓：它现在接在 `dit_build.BUILDERS["seedvr2"]` 上 ✓
    #    （`SKELETON_MODULES` 静态 import ⇒ 拼错当场 ImportError ✓；契约 = 5 维 / σ 批形态 /
    #    口径 `seedvr2` ✓）。接线的门票是**真实 3B fp16 权重前向 `maxdiff == 0`** ✓
    #    （`engine_seedvr2_test` ⑫ 形状反推、⑬ 族表接线、⑭ 真装载 ✓）。
    #    ⚠️ 2026-11-01 当天**那条通路也做完了** ✓：`stage="seedvr2"` 装配线在
    #    `runtime._ensure_seedvr2_loaded` ✓、17 通道条件在
    #    `torch_backend._encode_seedvr2_conditioning`（退化 16 ＋ mask 1 ✓）里成形 ✓、
    #    plan 在 `pipeline._fill_seedvr2_plan` ✓、`denoise` 按 `_ditFamily` 走条件分支 ✓
    #    ⇒ 这一族现在**端到端可跑** ✓（判据：`engine_seedvr2_test` 的装配/条件/编解码三组 ✓）。
    # ⚠️ 2026-11-01 **已删** `vae_seedvr2` 那条欠账 ✓：这一族的**两半一起接上了** ✓✗
    #    （`attach_seedvr2_vae` ✓ ＋ `derive latent` 走 `_encode_pixels` ✓ ＋ `decode` 的
    #    `seedvr2-real` 分支 ✓ ＋ `runtime._ensure_seedvr2_loaded` 装配线 ✓ ＋
    #    `pipeline._fill_seedvr2_plan` ✓ ＋ 17 通道条件在 `_encode_seedvr2_conditioning` 里成形 ✓）。
    #    前向口径与 ComfyUI `comfy/ldm/seedvr/vae.py` 逐位对拍过（fp32/CPU，encode/decode maxdiff 0.0 ✓，
    #    见 `tests/engine_seedvr2_vae_test.py` ⑧ 黄金数值 ✓）。⇒ 接线时**别**只补 decode ✗：
    #    编码侧取「前 16 通道（均值）」✓、反缩放 **×0.9152** 一个标量 ✓（不是 mean/std 组 ✓）。

    # ⚠️ `umt5_te` / `umt5_tokenizer` 两条欠账已删（2026-11-02 ✓）：UMT5 塔与其 unigram 分词器
    #    已经接进 `torch_backend` 的 Wan 装配线 ✓ ⇒ 再留在这张表里就是「白名单只增不减」✗
    #    （结构测试的 ㊅ 会当场报「已经接上线 ⇒ 该删条目」✓）。
}


# ══════════════════════════════════════════════════════════════════════════
# 纯判据（**不读盘** ⇒ 才喂得进坏数据 ✓）
# ══════════════════════════════════════════════════════════════════════════
def registry_problems(files: set[str], groups: dict[str, tuple[str, ...]]) -> list[str]:
    """㊀ 登记表 vs 盘上文件名 ⇒ 问题清单 ✓（空 = 干净 ✓）。"""
    flat = [name for group in groups.values() for name in group]
    problems: list[str] = []
    for name, times in sorted(Counter(flat).items()):
        if times > 1:
            problems.append(f"`{name}` 被登记了 {times} 次（一组一个家 ✓，重复登记等于边界不清 ✗）")
    missing = sorted(set(files) - set(flat))
    if missing:
        problems.append("盘上有、登记表里没有：" + "、".join(f"`{m}`" for m in missing))
    extra = sorted(set(flat) - set(files))
    if extra:
        problems.append("登记表里有、盘上没有：" + "、".join(f"`{e}`" for e in extra))
    return problems


def heavy_imports(path: Path, heavy: tuple[str, ...] = HEAVY_ROOTS) -> set[str]:
    """㊁ 某文件**顶层** import 的重依赖 ✓（函数内惰性 import 不算 ✓ —— 那是允许的写法 ✓）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            found.add(node.module.split(".")[0])
    return found & set(heavy)


def doc_problem(path: Path, minimum: int = MIN_DOC_WORDS) -> str:
    """㊂ 模块头是否真写清职责 ✓（返回空串 = 合格 ✓）。"""
    text = path.read_text(encoding="utf-8")
    words = len((ast.get_docstring(ast.parse(text)) or "").split())
    if words >= minimum:
        return ""
    return f"`{path.name}` 的模块头只有 {words} 词（下限 {minimum} ⇒ 职责没写清楚 ✗）"


def root_problems(root: Path, empties: list[str] | None = None) -> list[str]:
    """㊃ 仓库根的空壳与空目录 ✓（``empties`` 可注入 ⇒ 才反向验得动 ✓）。"""
    problems: list[str] = []
    if (root / "scripts").exists():
        problems.append("仓库根又有 `scripts/` 了（README 明写「仓库根没有 scripts/ 了」✗）")
    found = empties if empties is not None else [
        p.name for p in root.iterdir() if p.is_dir() and not any(p.iterdir())
    ]
    for name in found:
        problems.append(f"仓库根有**空目录** `{name}/`（空壳只会让人误会「这里该放东西」✗）")
    return problems


def hygiene_problems(gitignore: str, pkg_root: Path,
                     allowed: frozenset[str] = PKG_ROOT_FILES,
                     required: dict[str, str] = GENERATED_DIRS) -> list[str]:
    """㊉ 生成物不许入库 + 包根不许散落脚本 ✓（两个入参都可注入 ⇒ 反向验得动 ✓）。

    ⚠️ 判 ``.gitignore`` 时**先剔注释行**再逐字比对 ✓：否则「把规则注释掉、注释里还留着路径」
    也会被判成合格 ✗ —— 那正是最像修好的坏法 ✓。
    """
    problems: list[str] = []
    rules = {ln.strip() for ln in gitignore.splitlines()
             if ln.strip() and not ln.strip().startswith("#")}
    for rule, why in sorted(required.items()):
        if rule not in rules:
            problems.append(f"`.gitignore` 少了 `{rule}` 这条（{why} ⇒ "
                            f"下次 `git add -A` 就把它整片入库 ✗）")
    for path in sorted(pkg_root.iterdir()):
        if path.is_dir() or path.name in allowed:
            continue
        if path.suffix in {".log", ".err"}:  # 运行日志：`*.log` 已挡，留在根上不碍事 ✓
            continue
        problems.append(f"`backend-py/{path.name}` 散在包根（工具链进 `app/scripts/`、"
                        f"一次性草稿进 `_bench/`、真跑入口才留根；包根只许 "
                        f"{sorted(allowed)} ✗）")
    return problems


_REF = re.compile(r"(?:backend-py/)?(app/[A-Za-z0-9_./-]+\.py)")
#: 不带 ``app/`` 前缀的写法（**正是当年漏网的那种** ✓：``services/agents/skills.py`` ✗）
_REF_BARE = re.compile(r"(?<![\w/.])((?:services|routers|agent|core|scripts|configs|local_services)"
                       r"/[A-Za-z0-9_./-]+\.py)")


def _resolves(ref: str, root: Path) -> bool:
    """``ref`` 在**各种书写习惯**下是否落在盘上 ✓（有 ``app/`` 前缀的、没有前缀的、带 ``backend-py/`` 的 ✓）。"""
    bases = (root, root.parent, root / "app", root.parent / "backend-py")
    return any((base / ref).exists() for base in bases)


def _absolute_module(where: str, level: int, module: str) -> str:
    """把 ``ImportFrom`` 的 ``(level, module)`` 归一成**绝对点分路径** ✓。

    ⚠️ 这步是 ㊅ 的第二个坑 ✓：``app/services/*`` 习惯写 ``from .engine import cache_key`` ✗ ——
    只看字面 ``module`` 的话，「已接线」的模块会被**误报成孤儿** ✓（2026-10-30 实测：``cache_key`` /
    ``audio_mix`` / ``llm_backend`` 三处全是相对写法 ✓）。
    """
    if not level:
        return module
    parts = Path(where).with_suffix("").parts
    package = list(parts[:-1])  # 当前文件所在的包
    drop = level - 1
    if drop:
        package = package[: len(package) - drop]
    return ".".join([*package, module]) if module else ".".join(package)


def engine_callers(sources: dict[str, str], modules: tuple[str, ...]) -> dict[str, set[str]]:
    """㊅ 每个 engine 模块被**谁** import ✓（``sources`` ＝ ``{相对路径: 源码}`` ⇒ 可注入 ✓）。

    认全部写法 ✓：绝对（``from app.services.engine import X`` / ``from app.services.engine.X import …``）
    与相对（services 层 ``from .engine import X``、engine 包内 ``from . import X`` / ``from .X import …``）✓。
    —— 相对写法**必须先归一成绝对路径**再判 ✓，否则接好的线会被当成没接 ✗。
    自身引用自己不算调用者 ✗。
    """
    names = set(modules)
    callers: dict[str, set[str]] = {name: set() for name in modules}
    for where, text in sources.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue  # 别人家的坏文件不该让守卫崩 ✓
        relative = where.replace("\\", "/")
        stem = Path(relative).stem
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            absolute = _absolute_module(relative, node.level, node.module or "")
            hits: set[str] = set()
            if absolute.startswith("app.services.engine."):
                hits.add(absolute.rsplit(".", 1)[1])
            elif absolute in ("app.services.engine", "services.engine", "engine"):
                hits.update(alias.name for alias in node.names)
            for hit in hits & names:
                if Path(relative).parent.name == "engine" and stem == hit:
                    continue
                callers[hit].add(relative)
    return callers


def unwired_problems(modules: tuple[str, ...], callers: dict[str, set[str]],
                     allow: dict[str, str] = UNWIRED) -> list[str]:
    """㊅ 「生产代码 0 处 import」的模块 ⇒ 问题清单 ✓（空 = 全部接了线 ✓）。

    ⚠️ 两个方向都要判 ✓（只判一个方向，白名单就会**永久腐烂** ✗）：
    * 没接线、也没登记 ⇒ 报「已实现未接线」✓；
    * **登记了、但已经接上线** ⇒ 报「那条欠账该删了」✓（不然一年后没人知道哪些还是真的）。
    """
    problems: list[str] = []
    for name in modules:
        production = sorted(w for w in callers.get(name, set()) if not w.startswith("tests/"))
        if name in allow:
            if production:
                problems.append(f"`{name}` **已经接上线**（`{production[0]}` ✓）⇒ "
                                f"`UNWIRED` 里那条欠账该删了 ✗（白名单只增不减就成谎话 ✗）")
            continue
        if production:
            continue
        problems.append(f"`{name}` 生产代码 0 处 import（只在测试里活着 ⇒ 要么接线，"
                        f"要么写进 `structure_test.UNWIRED` 说明欠账 ✗）")
    return problems


def dangling_refs(texts: list[tuple[str, str]], root: Path = BACKEND_PY,
                  allow: tuple[str, ...] = _REF_ALLOW) -> list[str]:
    """㊄ 说明里点名的模块路径必须真实存在 ✓（``texts`` ＝ ``[(在哪, 正文)]`` ✓）。"""
    problems: list[str] = []
    for where, text in texts:
        for pattern in (_REF, _REF_BARE):
            for match in pattern.finditer(text):
                ref = match.group(1)
                if any(token in ref for token in allow):
                    continue
                if not _resolves(ref, root):
                    problems.append(f"`{where}` 指向不存在的 `{ref}` ⇒ 路径腐烂 ✗")
    return problems


# ══════════════════════════════════════════════════════════════════════════
# ① 登记表 ⇄ 盘上
# ══════════════════════════════════════════════════════════════════════════
def case_registry() -> None:
    on_disk = {p.stem for p in ENGINE_DIR.glob("*.py")} - {"__init__", "__main__"}
    problems = registry_problems(on_disk, MODULE_GROUPS)
    check(f"㊀ 登记表与盘上**双向相等**（{len(on_disk)} 个模块，一组一个家 ✓）",
          not problems, problems)
    check("㊀ `__all__` 由登记表拼成（不是另抄一份 ✓ —— 抄的迟早会漂 ✗）",
          list(ENGINE_ALL) == [n for g in MODULE_GROUPS.values() for n in g],
          f"{len(ENGINE_ALL)} vs 登记 {sum(len(g) for g in MODULE_GROUPS.values())}")
    check("㊀ `__all__` 无重复项",
          len(ENGINE_ALL) == len(set(ENGINE_ALL)), None)

    # 反验：谎报一个盘上不存在的模块名、并把真的藏掉一个 ⇒ 两侧都必须报
    broken = dict(MODULE_GROUPS)
    broken["numeric"] = MODULE_GROUPS["numeric"] + ("不存在的模块",)
    bad = registry_problems(on_disk, broken)
    check("㊀ 反验：登记了盘上没有的模块 ⇒ 报「有登记没文件」",
          any("登记表里有、盘上没有" in p for p in bad), bad)
    ghost = set(on_disk) | {"幽灵模块"}
    bad = registry_problems(ghost, MODULE_GROUPS)
    check("㊀ 反验：盘上多一个没登记的模块 ⇒ 报「有文件没登记」",
          any("盘上有、登记表里没有" in p for p in bad), bad)


# ══════════════════════════════════════════════════════════════════════════
# ② 零依赖组：顶层不许拖重依赖
# ══════════════════════════════════════════════════════════════════════════
def case_zero_dep() -> None:
    dirty = {name: sorted(heavy_imports(ENGINE_DIR / f"{name}.py")) for name in MODULE_GROUPS["numeric"]}
    dirty = {name: mods for name, mods in dirty.items() if mods}
    check("㊁ `numeric` 组顶层**零重依赖**（这组的卖点：不装 torch 也能测 ✓）",
          not dirty, dirty)
    # 反验：判据得真能看见 `import torch` ✓（不能因为它现在恰好干净就永远说 PASS ✗）
    with tempfile.TemporaryDirectory() as tmp:
        heavy = Path(tmp) / "拖重依赖.py"
        heavy.write_text("import torch\n", encoding="utf-8")
        check("㊁ 反验：顶层 `import torch` 必须被抓出来", "torch" in heavy_imports(heavy), None)
        lazy = Path(tmp) / "惰性依赖.py"
        lazy.write_text("def build():\n    import torch\n    return torch\n", encoding="utf-8")
        check("㊁ 反验：函数内惰性 import **不算**（那是允许的写法 ✓，别把它也判红 ✗）",
              not heavy_imports(lazy), None)


# ══════════════════════════════════════════════════════════════════════════
# ③ 每个模块都得写清自己是干什么的
# ══════════════════════════════════════════════════════════════════════════
def case_docstrings() -> None:
    problems = [msg for msg in (doc_problem(p) for p in sorted(ENGINE_DIR.glob("*.py"))) if msg]
    check("㊂ 63 个 engine 模块都写清了职责（模块头 ≥8 词 ✓）", not problems, problems)

    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / "无说明模块.py"
        stub.write_text("x = 1\n", encoding="utf-8")
        check("㊂ 反验：没有模块头的文件 ⇒ 判据报错",
              bool(doc_problem(stub)), None)
        brief = Path(tmp) / "太短模块.py"
        brief.write_text('"""短。"""\nx = 1\n', encoding="utf-8")
        check("㊂ 反验：模块头只有几个字 ⇒ 也算没写（不是「简洁」✗）",
              bool(doc_problem(brief)), doc_problem(brief))


# ══════════════════════════════════════════════════════════════════════════
# ④ 仓库根不留空壳
# ══════════════════════════════════════════════════════════════════════════
def case_root() -> None:
    problems = root_problems(REPO_ROOT)
    check("㊃ 仓库根没有 `scripts/` 空壳、没有空目录（README 的承诺要在盘上成立 ✓）",
          not problems, problems)
    check("㊃ 反验：注入一个空目录 ⇒ 判据报错",
          bool(root_problems(REPO_ROOT, empties=["某个空壳"])), None)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 说明里的路径不许腐烂
# ══════════════════════════════════════════════════════════════════════════
def case_refs() -> None:
    # **正本判据面**＝`app/` 下所有模块的说明/注释（`services/agents/` 那类事故就发生在这里 ✓）
    texts = [(p.relative_to(BACKEND_PY).as_posix(), p.read_text(encoding="utf-8", errors="replace"))
             for p in sorted(APP_DIR.rglob("*.py"))]
    problems = dangling_refs(texts)
    check(f"㊄ `app/` 里 {len(texts)} 份源码点名的 `app/…py` 全部真实存在（防路径腐烂 ✓）",
          not problems, problems[:8])

    # **文档面**＝功能分类地图（`docs/feature-map.md` 在 backend-py 之外 ⇒ 单独取 ✓）
    # 它是全仓**点名 `app/…py` 最多**的一份文档（300+ 处）⇒ 也最容易烂，必须同一条判据管住 ✓
    doc = BACKEND_PY.parent / "docs" / "feature-map.md"
    if doc.exists():
        doc_problems = dangling_refs([(f"docs/{doc.name}",
                                      doc.read_text(encoding="utf-8", errors="replace"))])
        check("㊄ 功能分类地图点名的 `app/…py` 全部真实存在（与代码同生共死 ✓）",
              not doc_problems, doc_problems[:8])

    fake = [("某模块.py", "见 `app/services/agents/skills.py` 的 `load_agent_skills` ✓")]
    check("㊄ 反验：指向已改名的旧路径 ⇒ 必须报（这正是 2026-09 那次改名的遗留 ✗）",
          bool(dangling_refs(fake)), None)
    ok = [("某模块.py", "见 `app/services/engine/inventory.py` ✓")]
    check("㊄ 反验：指向真路径 ⇒ 不报（判据不许见谁都喊 ✗）",
          dangling_refs(ok) == [], dangling_refs(ok))


# ══════════════════════════════════════════════════════════════════════════
# ⑥ 「已实现」不等于「已接线」
# ══════════════════════════════════════════════════════════════════════════
def case_wiring() -> None:
    modules = tuple(name for group in MODULE_GROUPS.values() for name in group)
    files = [p for base in ("app", "tests") for p in sorted((BACKEND_PY / base).rglob("*.py"))
             if "__pycache__" not in p.parts]
    sources = {p.relative_to(BACKEND_PY).as_posix(): p.read_text(encoding="utf-8", errors="replace")
               for p in files}
    callers = engine_callers(sources, modules)
    problems = unwired_problems(modules, callers)
    check(f"㊅ {len(modules)} 个 engine 模块都有**生产**调用者（或已登记为欠账 ✓）", not problems, problems)
    check(f"㊅ 欠账表 `UNWIRED` 没有僵尸条目（{len(UNWIRED)} 条，接上线就该删 ✗）",
          not (set(UNWIRED) - set(modules)), sorted(set(UNWIRED) - set(modules)))

    lone = {"tests/engine_x_test.py": "from app.services.engine import 只被测试用\n"}
    check("㊅ 反验：模块只在测试里被 import ⇒ 必须报（这正是「看起来能用」的坑 ✗）",
          bool(unwired_problems(("只被测试用",), engine_callers(lone, ("只被测试用",)))), None)
    wired = {"app/services/foo.py": "from app.services.engine import x\n",
             "tests/engine_x_test.py": "from app.services.engine import x\n"}
    check("㊅ 反验：生产代码 import 了 ⇒ 不报（判据不许见谁都喊 ✗）",
          not unwired_problems(("x",), engine_callers(wired, ("x",))), None)
    relative = {"app/services/foo.py": "from .engine import x\n"}
    check("㊅ 反验：`from .engine import x` 也算接线（相对写法不归一 ⇒ 已接的线会被误报 ✗）",
          not unwired_problems(("x",), engine_callers(relative, ("x",))), None)
    check("㊅ 反验：登记进 `UNWIRED` ⇒ 不报（欠账据实登记就放行 ✓，但表里必须写理由 ✓）",
          not unwired_problems(("只被测试用",), engine_callers(lone, ("只被测试用",)),
                               allow={"只被测试用": "理由"}), None)
    stale = unwired_problems(("x",), engine_callers(wired, ("x",)), allow={"x": "理由"})
    check("㊅ 反验：欠账表里的模块**已经接上线** ⇒ 必须报「该删条目」（白名单只增不减就成谎话 ✗）",
          bool(stale), stale)


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 功能分类地图必须**点到每一个模块**
# ══════════════════════════════════════════════════════════════════════════
def map_gaps(app_dir: Path, doc_text: str) -> list[str]:
    """文档没点到的 ``app/**.py``（「覆盖完整」唯一可核验的形式 ✓）。

    ``__init__.py`` 不点名 ✓：它由所在目录代表，写进表格只会灌水 ✗。
    ``app/skills/**/scripts/*.py`` **要**点名 ✓：后端不 import 它们，但「这个能力在哪个文件」
    必须查得到 ✓（目前 3 个，逐份点名比目录规则清楚）。
    """
    gaps = []
    for path in sorted(app_dir.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        rel = path.relative_to(app_dir.parent).as_posix()
        if rel not in doc_text:
            gaps.append(rel)
    return gaps


def case_coverage() -> None:
    doc = REPO_ROOT / "docs" / "feature-map.md"
    if not doc.exists():
        check("㊆ 功能分类地图存在（`docs/feature-map.md`）", False, "文件不在盘上")
        return
    text = doc.read_text(encoding="utf-8", errors="replace")
    total = len([p for p in APP_DIR.rglob("*.py")
                 if "__pycache__" not in p.parts and p.name != "__init__.py"])
    gaps = map_gaps(APP_DIR, text)
    check(f"㊆ 功能分类地图点到 `app/` 全部 {total} 个模块（新模块没归类 ⇒ FAIL ✗）",
          not gaps, gaps[:10])
    check("㊆ 反验：文档漏掉一个模块 ⇒ 必须报（漏掉就等于「这个功能没人认识」✗）",
          map_gaps(APP_DIR, text.replace("app/core/cpu_budget.py", "", 1)) != [], None)
    check("㊆ 反验：`__init__.py` 不要求点名（包由所在目录代表 ✓）",
          not [g for g in gaps if g.endswith("__init__.py")], None)


TESTS_DIR = BACKEND_PY / "tests"

#: 盘上有、但**故意**不进全量清单的套件（空着＝一份都没漏 ✓）
#:
#: 这一栏存在的唯一理由：有的套件要真 GPU / 真权重 / 真外部服务 ✓，塞进 `run_all`
#: 会把「一条命令验完」变成「跑一晚上还不一定过」✗。登记时必须写清**什么时候该手工跑** ✓
#: —— 写「太慢」这类空话不算 ✓（那样等于把判据废掉 ✗）。
NOT_IN_RUN_ALL: dict[str, str] = {
    #: ⚠️ 本表**只放"故意不进全量"的套件** ✓，而且每条都要写清「**什么时候手工跑**」✓✗。
    #: 2026-11-01 曾临时登记 `engine_scheduler_test.py`（当时它自红 ✓），同日它转绿（45/45 ✓）
    #: 且已进 `run_all` ⇒ **按本表自己的规矩删掉那行** ✓ —— 白名单只增不减就成谎话 ✗✗。
}


def registered_tests(run_all_path: Path) -> list[str]:
    """``run_all.py::TESTS`` 里登记的套件名（``ast`` 读，**不 import** ✓）。

    保留重复 ✓：判「同一套登记两遍」要用它，去重成 set 就看不出来了 ✗。
    """
    tree = ast.parse(run_all_path.read_text(encoding="utf-8", errors="replace"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(getattr(target, "id", "") == "TESTS" for target in node.targets):
            return [str(row[1]) for row in ast.literal_eval(node.value)]
    raise AssertionError(f"{run_all_path.name} 里找不到 TESTS 常量")


def test_list_gaps(tests_dir: Path, listed: list[str],
                   allowed: dict[str, str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """(盘上有但没登记, 登记了但盘上不在, 重复登记, 白名单里已过期的)。

    「没登记」这类的坑不是少跑一次 ✓，是**永远不跑还没人知道** ✗ ——
    2026-10-31 实测 `lora_train_test.py`（85 项）就这么在清单外待了 35 天 ✓。
    """
    on_disk = {path.name for path in tests_dir.glob("*_test.py")}
    known = set(listed)
    unlisted = sorted(on_disk - known - set(allowed))
    missing = sorted(name for name in known if not (tests_dir / name).exists())
    duplicated = sorted(name for name, count in Counter(listed).items() if count > 1)
    stale = sorted(set(allowed) - (on_disk - known))
    return unlisted, missing, duplicated, stale


def summary_gaps(tests_dir: Path, listed: list[str]) -> list[str]:
    """㊈ 登记的套件里**收尾行口径**不对的（``run_all`` 只认 ``SUMMARY: n/m passed`` ✓）。

    ⚠️ 这类的坑是**静默丢账** ✗✗（2026-11-01 实测 9 套中招 ✓）：收尾写成
    ``SUMMARY 12/12 通过 ✓`` / ``全部通过`` / ``12/12 通过`` 都**不是** `run_all` 认的那个字面量
    ⇒ 全量表里这些套件的项数**恒为 0** ✓✗（"少一套就是凭空消失"✓，而且退出码还是 0，
    看起来一切正常 ✗）。判据故意只盯**字面量** ✓：`SUMMARY:` 是 `run_all` 解析的那一段 ✓，
    别的写法无论多漂亮都进不了账 ✓。
    """
    gaps: list[str] = []
    for name in listed:
        path = tests_dir / name
        if not path.exists():
            continue
        if "SUMMARY:" not in path.read_text(encoding="utf-8", errors="replace"):
            gaps.append(name)
    return gaps


def _unregistered_fixture() -> list[str]:
    """反验夹具：盘上两份、清单只写一份 ⇒ 必须报出另一份 ✓。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "aaa_test.py").write_text("", encoding="utf-8")
        (root / "bbb_test.py").write_text("", encoding="utf-8")
        (root / "run_all.py").write_text(
            "TESTS = [('说明', 'aaa_test.py')]\n", encoding="utf-8")
        gaps = test_list_gaps(root, registered_tests(root / "run_all.py"), {})
        return gaps[0]


def case_test_list() -> None:
    listed = registered_tests(TESTS_DIR / "run_all.py")
    unlisted, missing, duplicated, stale = test_list_gaps(TESTS_DIR, listed, NOT_IN_RUN_ALL)
    total = len(list(TESTS_DIR.glob("*_test.py")))
    check(f"㊇ 盘上 {total} 份 `tests/*_test.py` 全部进了全量清单（漏登记 ⇒ 永远不跑 ✗）",
          not unlisted, unlisted)
    check("㊇ 清单点名的套件都在盘上（否则全量一跑就崩 ✗）", not missing, missing)
    check("㊇ 没有重复登记（同一套跑两遍＝白花时间 ✗）", not duplicated, duplicated)
    check("㊇ `NOT_IN_RUN_ALL` 没有过期条目（已进清单/文件已删 ⇒ 该删这行 ✗）",
          not stale, stale)
    check("㊇ 反验：盘上多出一份没登记的套件 ⇒ 必须报出它",
          _unregistered_fixture() == ["bbb_test.py"], None)
    check("㊇ 反验：登记进 `NOT_IN_RUN_ALL` 的就**不**报（白名单真能压住 ✓）",
          test_list_gaps(TESTS_DIR, listed, {**NOT_IN_RUN_ALL,
                                            unlisted[0]: "反验用"})[0] == unlisted[1:]
          if unlisted else True, None)
    # ── ㊈ 收尾口径：`run_all` 只认 `SUMMARY: n/m passed` ────────────────────
    gaps = summary_gaps(TESTS_DIR, listed)
    check(f"㊈ 清单里 {len(listed)} 套的收尾行都打成 `run_all` 认的口径 "
          f"`SUMMARY: n/m passed`（写别的 ⇒ 总表里**项数恒为 0** ✗✗）", not gaps, gaps)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "bad_test.py").write_text('print(f"SUMMARY {n}/{m} 通过 ✓")\n', encoding="utf-8")
        (root / "ok_test.py").write_text('print(f"SUMMARY: {n}/{m} passed")\n', encoding="utf-8")
        check("㊈ 反验：`SUMMARY 12/12 通过 ✓`（缺冒号 / 不是 passed）⇒ 必须报 ✗"
              "（2026-11-01 实测 9 套就栽在这个写法上，退出码仍是 0 ✓✗）",
              summary_gaps(root, ["bad_test.py"]) == ["bad_test.py"], None)
        check("㊈ 反验：`SUMMARY: n/m passed` ⇒ 不报（判据不许见谁都喊 ✗）",
              summary_gaps(root, ["ok_test.py"]) == [], None)


# ══════════════════════════════════════════════════════════════════════════
# ⑩ 生成物不许入库、包根不许散落脚本
# ══════════════════════════════════════════════════════════════════════════
def case_hygiene() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8", errors="replace")
    problems = hygiene_problems(gitignore, BACKEND_PY)
    check(f"㊉ 三处生成区（`output/` `_bench/` `models/tokenizers/`）都被 `.gitignore` 整目录挡住、"
          f"`backend-py/` 包根没有散落脚本 ✓", not problems, problems)

    stripped = "\n".join(ln for ln in gitignore.splitlines()
                         if ln.strip() != "backend-py/output/")
    leaked = hygiene_problems(stripped, BACKEND_PY)
    check("㊉ 反验：把 `backend-py/output/` 这条规则删掉 ⇒ 必须报"
          "（36 件真跑产物下一次 `git add -A` 就入库 ✗）",
          len(leaked) == 1 and "backend-py/output/" in leaked[0], leaked)
    commented = gitignore.replace("backend-py/output/", "# backend-py/output/", 1)
    check("㊉ 反验：规则被**注释掉**（注释里还留着路径）⇒ 仍必须报（最像修好的坏法 ✗）",
          bool(hygiene_problems(commented, BACKEND_PY)), None)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "README.md").write_text("", encoding="utf-8")
        (root / "requirements.txt").write_text("", encoding="utf-8")
        (root / "_run_qwen_image.py").write_text("", encoding="utf-8")
        (root / "backend.err.log").write_text("", encoding="utf-8")
        check("㊉ 反验：白名单内的文件 + 运行日志 ⇒ 不报（判据不许见谁都喊 ✗）",
              hygiene_problems(gitignore, root) == [], hygiene_problems(gitignore, root))
        (root / "_probe_real_chunk.py").write_text("", encoding="utf-8")
        stray = hygiene_problems(gitignore, root)
        check("㊉ 反验：包根多出一个一次性探针 ⇒ 必须报（2026-11-04 实测根上有 5 个 ✗）",
              len(stray) == 1 and "_probe_real_chunk.py" in stray[0], stray)


def main() -> int:
    case_registry()
    case_zero_dep()
    case_docstrings()
    case_root()
    case_refs()
    case_wiring()
    case_coverage()
    case_test_list()
    case_hygiene()
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      ↳ {detail}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed"
          + ("  ❌" if failed else "  ✅"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
