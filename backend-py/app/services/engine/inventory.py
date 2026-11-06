"""引擎的**权重体检 / 就绪报告**（零依赖 ✓）—— 跑之前先回答「这台机器能不能跑、缺什么」。

## 它解决什么真问题

本机的现实是：H3 那套要 **8 个组件、主权重单个 19.53 GiB** ✓，而且是「几十 GB 下载」级别 ✗。
没有这份体检时，用户只能**跑到一半**才发现某个组件根本没装 ✗（甚至更糟：装了个**截断的文件**，
加载时才炸 ✗）。所以本模块把三件事提前：

1. **在不在**（按 ``configs/models.json`` 的清单逐条对 ``models_dir`` ✓）；
2. **完不完整**（safetensors 头部自述的字节数与文件大小是否自洽 ⇒ 截断当场发现 ✓）；
3. **够不够显存**（权重字节是**精确值** ✓；激活是**估算**且明确标注 ✗ 不承诺 ✓）。

## 事实来源（不猜 ✓）

* 清单：``configs/models.json``（``models[]`` 每项含 ``kind`` / ``filename`` / ``file_path`` /
  ``size_gib`` / ``required`` ✓；⚠️ ``file_path`` 是**远端仓库内**的路径 ✗ —— 只用来拼下载直链 ✓，
  **本地落点只看 ``kind`` + ``filename``** ✓，见 :func:`component_path` ✓）；
* 目录：``configs/model-paths.json`` 的 ``models_dir`` ✓（解析优先级由
  :func:`app.services.local_model_scan.get_model_paths` 统一提供 ✓ 与本仓其它地方同一份来源 ✓）；
* 校验：:mod:`app.services.engine.safetensors`（纯 Python ✓）与
  :mod:`app.services.engine.gguf`（纯 Python ✓，2026-09-20 起支持 ✓）。
* H3 键名核对：:mod:`app.services.engine.h3_keys`（纯 Python ✓ torch-free ✓
  2026-09-20 起支持 ✓）—— 文件形态是 H3 时（`h3_form.looks_like_h3_form` ✓），
  直接对照参考结构核「键全集 + 形状关系」✓（真权重到手**插上就验** ✓，
  不用先建 50 层模型 ✗）。

⚠️ GGUF 此前只报「在/不在 + 大小」✗（读取器未实现 ✗）—— **已补齐** ✓（2026-09-20 ✓）：
现在 ``.gguf`` 走 :mod:`.gguf` 真体检（张量表 / 截断 / 量化方案 ✓）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..local_model_scan import get_model_paths
from . import gguf as gguf_mod
from . import h3_form
from . import h3_keys
from . import safetensors as st
from . import tiers as tiers_mod

__all__ = [
    "DEFAULT_CAPACITY_GIB",
    "candidate_roots",
    "catalog_entry",
    "component_candidates",
    "component_path",
    "component_status",
    "estimate_vram",
    "COMPONENT_SLOTS",
    "assembly_for_stage",
    "load_catalog",
    "readiness",
    "resolve_component",
    "upscale_status",
]

#: 本机目标显卡（A5000 24GB ✓）—— 只用于算「余量」，可覆盖 ✓
DEFAULT_CAPACITY_GIB = 24.0

#: 各阶段的必需组件筛选规则（``category`` ＋ **可选** ``tags`` ✓）
#:
#: ⚠️ ``tags`` 是 2026-10-31 加的**可选**精筛 ✗✗（不加它 ``zimage`` 就得跟 SDXL 共用
#: ``category="image"`` ✓ ⇒ ``readiness("zimage")`` 会把 `sdxl_base` 也列进来 ✓✗，
#: 「Z-Image 就绪」于是变成「image 分类里有东西」✓✗ —— 两条链的组件根本不是一套 ✓）。
#: 语义：``category`` 先筛（必须一致 ✓），``tags`` 非空时再要求**交集非空** ✓。
#: 不写 ``tags`` 的阶段行为**一字不变** ✓（``image`` 仍列全部分类件 ✓）。
STAGE_FILTERS: dict[str, dict[str, Any]] = {
    "h3": {"category": "video"},
    "image": {"category": "image"},
    "audio": {"category": "audio"},
    "text": {"category": "text"},
    #: ⭐ Z-Image 单列一档 ✓（2026-10-31 ✓）：三份权重是**一条链** ✓ —— DiT ＋ Qwen3-4B ＋ Flux VAE ✓，
    #: 少一份都出不了图 ✓，而它们与 SDXL 四件套**没有交集** ✗ ⇒ 不能只按 ``category`` 筛 ✓。
    #:
    #: ``requireAll`` = 「本档筛到的件**全都要**」✓ —— 与清单里那三条的 ``required: false`` 配套 ✓：
    #: ⚠️ 若清单直接把它们标 ``required: true`` ✗，``readiness("image")`` 会把 SDXL（6.94）与
    #: Z-Image（19.27）**加在一起 = 25.73 GiB** ✓✗ —— 两套是**互斥备选** ✗（要么跑 SDXL、要么跑 Z-Image ✓），
    #: 加出来的数没有任何一条链用得上 ✓✗，还会让 image 档在 24GB 卡上凭空"装不下" ✓✗。
    #: ⇒ 「这条链的件必须齐」这一条判据只属于 ``zimage`` 档 ✓，与 `image` 档的必需件彻底分开 ✓。
    "zimage": {"category": "image", "tags": ["zimage"], "requireAll": True},
    #: ⭐ Qwen-Image **2.1** 单列一档 ✓（2026-10-31 ✓）：同样三份**一条链** ✓ ——
    #: DiT 649 键 ＋ Qwen3.5-9B ＋ 64 通道 VAE ✓；与 Z-Image / SDXL 都**没有交集** ✗。
    #: ⚠️ 不与 ``zimage`` 合档 ✗✗：两族三件都叫「DiT / 文本塔 / VAE」✓、形状也"看着能对上" ✓，
    #: 但 `text_dim` 2560 vs 4096 ✓、VAE 16 vs 64 通道 ✓ ⇒ 合档会把 **19.27 + 16.5 ≈ 35.8 GiB**
    #: 加在一起 ✓✗（没有任何一条链用得上 ✓，还会让 24GB 卡凭空报装不下 ✓✗）。
    "qwen_image21": {"category": "image", "tags": ["qwen21"], "requireAll": True},
    #: ⭐ **Qwen-Image 2512 / Edit 2511**（2026-10-31 加 ✓）。与 `qwen_image21` 必须**分档** ✗✗：
    #: 两族的文本塔（Qwen2.5 **3584** vs Qwen3.5 **4096** ✓）与 VAE（**16** vs **64** 通道 ✓）
    #: 一件都不通用 ✓ —— 合档会把 38.05 + 16.1 + 0.24 ≈ **54.4 GiB** 与 2.1 的 28.68 GiB
    #: **加在一起** ✓✗（多报 23.7 GiB ✓，24GB 卡凭空"装不下" ✓✗，而真装的时候只装一条链 ✓）。
    #: ⚠️ 两者族名只差一个 `21` ✗（`qwen_image` vs `qwen_image21` ✓）—— tags 用 `qwen2512` 而非
    #: `qwen_image` ✓，避免将来有人拿 `startswith("qwen_image")` 判族把 2.1 一起卷进来 ✓✗。
    "qwen_image": {"category": "image", "tags": ["qwen2512"], "requireAll": True},
    #: ⭐ **Krea2**（2026-11-01 加 ✓）。必须**单列** ✗✗：三件里有两件与既有档**同文件不同族** ✓✗ ——
    #: VAE 就是 2512 那个 ``qwen_image_vae`` ✓（16 通道 ✓），DiT 却是 **12.82B 参数的单流**
    #: ✓（2512 是 38.05 GiB 双流 ✓），文本塔是 **Qwen3-VL-4B 的 12 层栈** ✓（2512 是 Qwen2.5-VL-7B 末层 ✓）。
    #: ⚠️ 合进 ``qwen_image`` 档 ⇒ ``readiness("qwen_image")`` 会把 12.57 + 5.08 加进 2512 那条
    #: **38.05 GiB** 的账里 ✓✗（多报 17.6 GiB ✓，24GB 卡凭空"装不下" ✓✗，而真装时只装一条链 ✓）。
    #: ⚠️ ``krea2`` 与 ``qwen2512`` 的 tags **不许互相包含** ✓（同 ``qwen_image`` 那条注释的理由 ✓）。
    "krea2": {"category": "image", "tags": ["krea2"], "requireAll": True},
    #: ⭐ **SEEDVR2**（2026-11-01 加 ✓）。``category`` 是 **`video`** ✓（与 H3 同类 ✓），但它与 H3
    #: **一件都不通用** ✗✗：H3 三件（DiT ＋ Qwen3-VL 文本塔 ＋ 视频/音频双 VAE ✓），SEEDVR2 两件
    #: （NaDiT ＋ 250 键专用 VAE ✓、**没有文本塔** ✗）✓。
    #: ⚠️ 并进 ``h3`` 档 ⇒ ``readiness("h3")`` 会多算一份 6.8 GiB ✓✗，且「H3 就绪」会被 SEEDVR2 的
    #: VAE 冒充 ✓✗（两条链各自有 VAE ✓，体积与键数都不同 ✓）。
    #: ⚠️ ``tags`` 用 ``seedvr2`` ✓（**不**用 ``video`` ✓ —— 那个词被 H3 的三件占了 ✓✗）。
    "seedvr2": {"category": "video", "tags": ["seedvr2"], "requireAll": True},
    #: ⭐ **Wan 2.2 I2V**（2026-11-02 加 ✓）。``category`` 是 **`video`** ✓（与 H3 / SEEDVR2 同类 ✓），
    #: 但三件都不通用 ✗✗：H3 是 16 通道文生视频 ＋ Qwen3-VL 文本塔 ✓、SEEDVR2 没有文本塔 ✓；
    #: 这一族是 **36 通道**（首帧条件）＋ **UMT5-XXL** ＋ **Wan2.1 VAE** ＋ 双专家 ✓。
    #: ⚠️ 并进 ``h3`` / ``seedvr2`` 档 ⇒ ``readiness`` 会把 26.62 ＋ 10.58 ＋ 0.24 三笔加到别的链上
    #: ✓✗（24GB 卡凭空"装不下" ✓，而真装时只装一条链 ✓）。
    #: ⚠️ ``tags`` 用 ``wan22`` ✓（**不**用 ``wan`` ✗ —— 清单里 `wan26_i2v_14b_gguf` 那条也带 ``wan`` ✓，
    #: 那是 GGUF 另走 ``text`` 那条路 ✓，卷进来会多算 15 GiB ✓✗）。
    "wan22": {"category": "video", "tags": ["wan22"], "requireAll": True},
    #: ⭐ **Music3 文生音乐**（2026-11-06 加 ✓）—— 本仓唯一一条**输出不是画面**的线 ✓✗。
    #:
    #: ``category`` 是 **`audio`** ✓（清单里那一类此前**一条都没有** ✓）。
    #: ⚠️ 本档的「三件必须齐」**只能**靠 ``requireAll`` ✓✗：``STAGE_FILTERS["audio"]`` 只有
    #: ``category`` ✓，而三件的 ``required`` 是 ``false`` ✓（与 zimage 同套路 ✓）⇒
    #: 走 ``audio`` 档问就绪**仍然是**"空必需集 ⇒ `ready=True`" ✓✗（那是 ``audio`` 档既有的口径 ✓，
    #: 本轮**不**顺手改它 ✗ —— 改它会把"audio 档到底指什么"这个更大的问题卷进来 ✓）。
    #: 「Music3 齐不齐」**只从 ``music3`` 档问** ✓：``stage_entries("music3")`` 按
    #: ``tags:["music3"]`` 挑三件 ＋ ``requireAll`` 把它们全判成必需 ✓。
    #:
    #: ⚠️ **三件都不许与别的档合并** ✗✗：
    #: · DiT 4.58 GiB ✓ 是**音频** 3 维潜变量 ``(b,128,T)`` ✓，与任何出图/出片族维度都不同 ✓；
    #: · 语义塔 8.57 GiB ✓ **不是编码器** ✗（自回归生成器 ✓，`output_dim` 32768 ＝ 8×4096 ✓）；
    #: · DAV 0.20 GiB ✓ 是**音频**解码器 ✓（128ch ⇒ 44.1k 立体声 ✓，反缩放**恒等** ✓）。
    #: 合档会多报 13.34 GiB 到别的链上 ✓✗（24GB 卡凭空"装不下" ✓）。
    #:
    #: ⚠️ ``requireAll: True`` 与清单里三件的 ``required: false`` **必须配套** ✓（同 zimage 那条注释 ✓）：
    #: 直接标 ``required: true`` 会让 ``readiness("audio")`` 把它们当"audio 档的必需件"✓，
    #: 而 audio 档的口径本来就是"本族三件齐不齐"✓ ⇒ 由 ``requireAll`` 表达才对 ✓。
    "music3": {"category": "audio", "tags": ["music3"], "requireAll": True},
}


def catalog_path() -> Path:
    """``configs/models.json`` 的路径（与 ``scripts/model_manager.py`` 同一份 ✓）。"""
    return Path(__file__).resolve().parents[3].parent / "configs" / "models.json"


def load_catalog() -> dict[str, Any]:
    """读清单 ✓（缺失/非法 ⇒ 返回空表而不是抛错 ✓ —— 体检本身不该把服务搞崩 ✓）。"""
    path = catalog_path()
    if not path.exists():
        return {"models": [], "nodes": [], "error": f"清单不存在：{path}"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        return {"models": [], "nodes": [], "error": f"清单读不了：{err}"}
    if not isinstance(data, dict):
        return {"models": [], "nodes": [], "error": "清单不是对象 ✗"}
    models = data.get("models")
    return {
        "models": [item for item in models if isinstance(item, dict)] if isinstance(models, list) else [],
        "nodes": data.get("nodes") if isinstance(data.get("nodes"), list) else [],
        "error": None,
    }


def models_dir() -> Path | None:
    """清单里声明的模型根目录（空 ⇒ ``None`` ✓）。"""
    value = str(get_model_paths().get("models_dir") or "").strip()
    return Path(value) if value else None


def component_path(entry: dict[str, Any], root: Path | None = None) -> Path | None:
    """按清单算出该组件的**期望落点** ✓ ⇒ ``<根>/<kind>/<文件名>`` ✓（落点**只此一份口径** ✗）。

    ⚠️ ``file_path`` **不是**本地落点 ✗✗（2026-09-26 修 ✓）：它是**远端仓库内**的路径 ✓ ——
    只用来拼下载直链（``scripts/model_manager.py`` 拿 ``repo`` + ``file_path`` 拼出
    ``…/resolve/main/<file_path>`` ✓）。本地落点由**安装器**定 ✓，而它的装 / 查 / 删**三处**算的
    都是 ``models_dir/<kind>/<文件名>`` ✓ ⇒ 那才是唯一事实来源 ✓。

    例子（也是最容易混的那个 ✓）：``stabilityai/stable-diffusion-xl-base-1.0`` 这个 HF 仓库**是平的** ✓
    ⇒ ``file_path`` = ``sd_xl_base_1.0.safetensors`` ✓（照它算本地落点，就会去找
    ``<models_dir>/sd_xl_base_1.0.safetensors`` ✓✗ —— 那是**安装器永远不会写**的位置 ✓），
    而盘上（与 ComfyUI 的惯例一致 ✓）它在 ``<models_dir>/checkpoints/`` 里 ✓。
    以前两者混用 ⇒ 体检 / 「按 key 体检」/ hybrid-merge 的**按 key 解析**全都指错地方 ✓✗
    （本机实测：扫描看得见、装载看不见 ✓✗）。
    """
    root = root if root is not None else models_dir()
    if root is None:
        return None
    filename = str(entry.get("filename") or "").strip()
    if not filename:
        # ⚠️ 连**文件名**都没有 ⇒ 算不出来就是算不出来 ✗：本地叫什么名字**没有事实来源** ✓
        #    （``file_path`` 也救不了 ✓ —— 它是**远端**路径，见上 ✓）。
        return None
    kind = str(entry.get("kind") or "").strip()
    return root / (f"{kind}/{filename}" if kind else filename)


def catalog_entry(key: str) -> dict[str, Any] | None:
    """按 ``key`` 取清单条目 ✓（取不到 ⇒ ``None`` ✓ —— "报"还是"跳过"由调用方定 ✓）。"""
    wanted = str(key or "").strip()
    if not wanted:
        return None
    return next((item for item in load_catalog()["models"] if item.get("key") == wanted), None)


def candidate_roots() -> list[tuple[Path, str]]:
    """清单落点的**候选根** ✓ ⇒ ``[(根, 来源说明)]`` ✓（去重保序 ✓）。

    根表**只此一份** ✗ —— 直接摊平 :func:`app.services.local_model_scan.get_default_roots` ✓
    （``models_dir`` 排第一 ✓ / 用户显式加的目录 ✓ / 各生态落点 ✓ / **每个** ComfyUI 类目录的
    ``models/`` ✓ —— 见那份函数自己的说明 ✓）。

    ⚠️ 这里**一个路径字面量都没有** ✗：根全部来自用户配置或动态探测 ✓（口径同
    ``tests/local_models_test.py`` 的静态守卫 ✓）。⚠️ 探测根**只用来对文件名** ✗，
    绝不"挑一个看着像的权重" ✓ —— 见 :func:`component_candidates` ✓。
    """
    from ..local_model_scan import get_default_roots  # 局部 import ✓：清单层不把扫描层拖到 import 期 ✓

    declared = models_dir()
    declared_key = os.path.normcase(str(declared)) if declared is not None else ""
    roots: list[tuple[Path, str]] = []
    for raw in get_default_roots():
        path = Path(raw)
        source = ("清单 configs/model-paths.json 的 models_dir"
                  if declared_key and os.path.normcase(str(path)) == declared_key
                  else "扫描根 local_model_scan.get_default_roots()")
        roots.append((path, source))
    return roots


def component_candidates(entry: dict[str, Any],
                         roots: Sequence[tuple[Path, str]] | None = None) -> list[tuple[Path, str]]:
    """清单条目 ⇒ **有序候选落点** ✓ ⇒ ``[(文件, 来源说明)]`` ✓。

    每个候选根下试 ``<根>/<kind>/<文件名>`` ✓（ComfyUI 那套「按类别分子目录」的惯例 ✓ ——
    本机那份 SDXL 就在 ``…/models/checkpoints/`` 里 ✓，也正是安装器的落点 ✓；
    ⚠️ 根**本身就已经是** ``<kind>`` 那个目录时**只试平的** ✓，见下 ✓）。

    ⚠️ 这里**不再**拿 ``file_path`` 当本地落点 ✗✗（2026-09-26 修 ✓，口径同 :func:`component_path` ✓）：
    它是**远端仓库内**的路径 ✓，而"上游那个 HF 仓库平不平"跟**本地**落点**根本没关系** ✓✗
    （SDXL 那个仓库就是平的 ✓，本地仍在 ``checkpoints/`` ✓）。以前两种布局并列 ⇒ 候选表凭空
    翻一倍 ✓✗，其中一半还在找 ``<根>/sd_xl_base_1.0.safetensors`` 这种**不可能存在**的形状 ✓✗。

    ⚠️ **文件名一律来自清单** ✗（``filename`` ✓）⇒ 这里**没有猜名字** ✓：
    猜中的代价是加载到**别的**权重 ⇒「装得进去、跑得出来、结果不对」✓✗
    （见 :func:`app.services.engine.bridge.resolve_dit_path` 的说明 ✓）。
    """
    filename = str(entry.get("filename") or "").strip()
    if not filename:
        return []
    kind = str(entry.get("kind") or "").strip()
    layout = f"{kind}/{filename}" if kind else filename
    table = list(roots) if roots is not None else candidate_roots()
    found: list[tuple[Path, str]] = []
    for root, source in table:
        # ⚠️ 根**本身就已经是** ``<kind>`` 那个目录时**别再拼一层** ✗✗ —— 本机的根表里，每个
        #    ComfyUI 类别目录（``checkpoints`` / ``clip`` / …）**各自都是一条根** ✓ ⇒ 再拼
        #    ``<kind>/`` 只会造出 ``…/checkpoints/checkpoints/…`` 这种**不可能存在**的形状 ✓✗
        #    （候选表翻一倍还全是噪声 ✓）。⇒ 这种根下**只试平的那一种** ✓。
        here = filename if (kind and root.name == kind) else layout
        found.append((root / here, source))
    return found


def resolve_component(entry: dict[str, Any], root: Path | None = None,
                      *, roots: Sequence[tuple[Path, str]] | None = None) -> tuple[Path | None, str]:
    """清单条目 ⇒ **盘上那一份** + 来源说明 ✓（取不到 ⇒ ``(None, "")`` ✓ —— 不猜 ✓）。

    ⚠️ 搜索域**由调用方圈定** ✗✗（三种写法，语义互不相同 ✓）：
    ① 什么都不给 ⇒ 域 = ``models_dir`` 的清单落点 ✓ + **动态探测**根（:func:`candidate_roots` ✓）
       —— 生产侧就是这个用法 ✓；
    ② ``root``（自检注入临时目录 ✓）⇒ 域 = **只有它** ✗：再往下翻真实磁盘，会把「钉住的那份没装」
       变成「其实装了**别的**」✓✗（体检结果随机器变 ✓✗，本仓铁律：自检要能在任何机器上给出
       同一个结论 ✓）；
    ③ ``roots``（**根表** ✓ —— 形参口径与 :func:`component_candidates` 的 ``roots`` **同一份** ✓）
       ⇒ 域 = **那张表** ✓✗ —— ⚠️ 此时**不再**另取 ``models_dir`` ✗：调用方既然圈定了域，再往里塞
       一条它**没说过**的落点，等于替它做主 ✓✗（自检注入临时根时，结论会随**本机** ``models_dir``
       里恰好有什么而变 ✓✗ —— 同一份自检在两台机器上给两个答案，正是本仓最忌的那个 ✓）。

    ``roots`` 必须传**根表**而不是**候选文件表** ✗：后者会被当成根再摊一层
    （``…/checkpoints/sd_xl_base.safetensors/checkpoints/…`` ✓✗ —— 曾经的真 bug ✓）。
    """
    if roots is not None:
        table: Sequence[tuple[Path, str]] = list(roots)
        declared = component_path(entry, root) if root is not None else None
    elif root is not None:
        table = []
        declared = component_path(entry, root)
    else:
        table = candidate_roots()
        declared = component_path(entry)
    if declared is not None and declared.exists():
        return declared, "清单落点"
    for found, source in component_candidates(entry, table):
        if declared is not None and found == declared:
            continue
        if found.exists():
            return found, source
    return None, ""


def _audit_h3_form(status: dict[str, Any], tensors: Mapping[str, Sequence[int] | None]) -> None:
    """文件形态是 H3 时（`h3_form.looks_like_h3_form` ✓）做**键名核对** ✓（2026-09-20 起）。

    好处：真权重到手那一刻就能答「这份 checkpoint 能不能装进 ``H3FormTrunk``」✓ ——
    不用先建 50 层模型再让 ``load_module_weights`` 报 missing ✗。
    核不通过（缺键 / 形状不符 / 结构矛盾）⇒ 进 ``problems`` ⇒ **阻断**就绪 ✓。
    """
    if not tensors or not h3_form.looks_like_h3_form(set(tensors)):
        return
    try:
        audit = h3_keys.audit_h3_checkpoint(tensors)
    except Exception as err:  # noqa: BLE001 —— 核对器自身炸了也是「结论」的一种 ✓
        status["problems"].append(f"H3 键名核对器异常：{err}")
        return
    status["h3Audit"] = audit.to_dict()
    status["problems"].extend(audit.problems)
    for item in audit.missing[:6]:
        status["problems"].append(f"H3 键名核对：缺 `{item}` ✗")
    for item in audit.shape_mismatch[:4]:
        status["problems"].append(
            f"H3 键名核对：`{item['key']}` 形状 {item['got']} ≠ 期望 {item['expected']} ✗")
    if not audit.ok:
        status["verified"] = False


def component_status(entry: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """单个组件的体检结论（不抛错 ✓ —— 缺文件也是**结论**的一种 ✓）。

    ⚠️ **"盘上那一份"只有一处解析** ✗✗（2026-09-27 修 ✓，用户当场指出 ✓）：不给 ``root`` 时走
    :func:`resolve_component` ✓ —— 也就是**引擎加载时**用的那份（``models_dir`` 清单落点优先 ✓
    → 动态探测根 ✓）；给了 ``root`` ⇒ **只认那个根** ✓（自检钉根 ⇒ 结论不随机器变 ✓）。

    以前这里**只看** ``models_dir`` ✗✗ ⇒ **同一件事两个说法** ✓✗：体检说「缺 ``dit_fl2va_int8`` ✗、
    还要下 19.53 GiB」，可盘上那份**明明就在**探测到的 ComfyUI 共享目录里 ✓、引擎加载也真会用它 ✓
    （实测：``…/ComfyUI-Shared/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`` ✓）
    ⇒ 报告把"已经就绪"说成"没下载" ✓✗ —— 报告与实底打架，比没有报告更坏 ✗。
    """
    expected_path = component_path(entry)
    if root is not None:
        path, source = component_path(entry, root), "调用方钉的模型根 ✓"
    else:
        path, source = resolve_component(entry)
    expected_gib = entry.get("size_gib")
    status: dict[str, Any] = {
        "key": entry.get("key"),
        "name": entry.get("name"),
        "kind": entry.get("kind"),
        "category": entry.get("category"),
        "required": bool(entry.get("required")),
        "path": str(path) if path else None,
        #: ⚠️ 清单算出来的**期望落点**（``models_dir`` 那份 ✓）—— 与实际找到的 ``path`` **分开报** ✗：
        #: 不分开 ⇒ 读者答不出「这份是不是本仓自己那份」✓✗（口径见 :func:`readiness` 的
        #: ``resolvedElsewhere`` ✓）。
        "expectedPath": str(expected_path) if expected_path else None,
        #: 这份是**在哪儿**找到的 ✓（清单落点 / 动态探测根 ✓）—— 缺件时给的是"找过哪儿" ✓。
        "source": source if path is not None else "",
        "present": False,
        "bytes": 0,
        "expectedGiB": expected_gib,
        "verified": False,
        "problems": [],
    }
    if path is None:
        status["problems"].append("清单里没有可解析的路径（filename/file_path 皆空，或 models_dir 未配置 ✗）")
        return status
    if not path.exists():
        # ⚠️ 文案里必须说清"找过哪里" ✗：只说「未安装」⇒ 用户没法判断该往哪儿放 ✓✗
        #    （2026-09-27 实测：盘上明明有，报告却只说"未安装" ✓）。
        #    ⚠️ 钉了 ``root`` 就不许再提探测根 ✗（那等于偷偷翻真实磁盘 ✓✗ —— 见 :func:`resolve_component` 的域口径 ✓）。
        where = (f"清单落点 {path} ✗" if root is not None
                 else f"清单落点 {expected_path} 与动态探测根都没有 ✓")
        status["problems"].append(f"未安装 ✗（{where} —— 期望落点见 expectedPath ✓）")
        return status
    try:
        status["bytes"] = path.stat().st_size
    except OSError as err:
        status["problems"].append(f"读不到文件：{err}")
        return status
    status["present"] = True

    if path.suffix.lower() == ".gguf":
        # ⚠️ 2026-09-20 起有**真读取器** ✓（此前只报「在/不在 + 大小」✗）⇒ 与 safetensors
        #    同一口径：结构坏 ⇒ 阻断 ✗；量化方案名来自 ``general.file_type`` ✓。
        info = gguf_mod.inspect(path)
        status["problems"].extend(info.problems)
        status["verified"] = info.ok
        status["tensorCount"] = info.tensor_count
        status["dtypeCounts"] = info.dtype_counts
        status["quantScheme"] = info.quant_scheme
        status["biggest"] = [{"name": t.name, "shape": t.shape, "dtype": t.type_name}
                             for t in info.biggest(5)]
        _audit_h3_form(status, {name: tensor.shape for name, tensor in info.tensors.items()})
    else:
        info = st.inspect(path)
        status["problems"].extend(info.problems)
        status["verified"] = info.ok
        status["tensorCount"] = info.tensor_count
        status["dtypeCounts"] = info.dtype_counts
        status["biggest"] = [{"name": t.name, "shape": t.shape, "dtype": t.dtype}
                             for t in info.biggest(5)]
        _audit_h3_form(status, {name: tensor.shape for name, tensor in info.tensors.items()})

    actual_gib = round(status["bytes"] / 1024 ** 3, 2)
    status["actualGiB"] = actual_gib
    status["sizeMismatch"] = False
    if isinstance(expected_gib, (int, float)) and float(expected_gib) > 0:
        # 大小偏差 >5% ⇒ **可疑**（下载中断/下错文件都会这样 ✓）
        # ⚠️ 刻意与「结构损坏」分开：结构问题（截断/头部矛盾）会**阻断**就绪 ✗，
        #    大小偏差只进 `suspectRequired`（因为社区量化版/重新导出会让体积合法地变 ✓）。
        delta = abs(actual_gib - float(expected_gib)) / float(expected_gib)
        if delta > 0.05:
            status["sizeMismatch"] = True
            status["problems"].append(
                f"大小与清单不符：实测 {actual_gib} GiB vs 清单 {expected_gib} GiB"
                f"（偏差 {delta * 100:.1f}% ⇒ 可能没下完或下错 ⚠️）")
    return status


def upscale_status(*, root: Path | None = None, key: str = "", path: str = "") -> dict[str, Any]:
    """**超清放大器**的可用性 ✓（2026-09-24 接 ✓）—— 让「超清到底能不能开」在**跑之前**就看得见 ✓✗。

    ⚠️ 三段都要**能说清「没查」** ✗✗（本仓铁律：**没查 ≠ 通过** ✓）：
    1. 没给 ``path`` / ``key`` ⇒ ``checked=False`` ✓ + 说明**怎么让它可查** ✓（不是通过 ✓✗）；
    2. 清单里没有这个组件 ⇒ ``checked=True`` + ``present=False`` ✓ + **期望的格式串** ✓
       （不然只能靠猜要什么权重 ✓✗）；
    3. 权重在盘上 ⇒ 读它的**内嵌契约** ✓：读不出来 ⇒ 给**具名**原因 ✓（回退普通模式是有理由的 ✓）。
    """
    from . import upscale as upscale_mod   # 局部 import ✓：只有这条查询路需要它 ✓（避免清单层拉着引擎层 ✓）

    resolved = str(path or "").strip()
    if not resolved and key:
        entry = next((item for item in load_catalog()["models"] if item.get("key") == key), None)
        if entry is not None:
            # ⚠️ **同一条解析** ✗（2026-09-27 修 ✓）：给了 ``root`` ⇒ 只认那个根 ✓；否则
            #    清单落点优先 → 动态探测根 ✓。以前只看 ``models_dir`` ✗ ⇒ 盘上那份会被当成"不在"
            #    ✓✗（就是"同一件事两个说法"那个毛病 ✓，见 :func:`component_status` 的说明 ✓）。
            found = (component_path(entry, root) if root is not None
                     else resolve_component(entry)[0])
            resolved = "" if found is None else str(found)
    report: dict[str, Any] = {
        "checked": False, "present": False, "key": str(key or ""), "path": resolved or None,
        "contract": None, "contractError": None, "format": upscale_mod.CHECKPOINT_FORMAT,
        "requiresMaxTileAbove2x": True, "notes": [],
    }
    if not resolved:
        report["notes"].append(
            "**没查** ✗：没给放大器权重（``upscale.path`` / ``upscale.key`` 或清单里的组件 ✓）"
            "⇒ 超清计划会**回退普通模式**并给出理由 ✓ —— ⚠️ 别把这条读成「超清可用」✗✗")
        return report
    target = Path(resolved)
    if not target.exists():
        report.update({"checked": True, "notes": [
            f"清单/入参指向的放大器权重**不在盘上** ✗：{target} ⇒ 超清会回退普通模式 ✓（把文件放上去再查 ✓）"]})
        return report
    report["checked"] = True
    report["present"] = True
    try:
        metadata, _header = st.read_header(target)
        # ⚠️ 读 + **结构核** ✓（``load_upscaler_contract``）：原先只读不核 ✗ ⇒ 缺字段的契约会走到
        #    下面那句 ``contract['base_config']['in_channels']`` 上**直接 KeyError** ✓✗
        #    （本接口的职责是"具名拒绝"✓，不是把异常抛给它自己的调用方 ✗）。
        contract = upscale_mod.load_upscaler_contract(metadata)
    except Exception as err:  # noqa: BLE001 —— 读不了/契约不合法 ⇒ **具名拒绝** ✓ 不猜 ✓
        contract = None
        report["contractError"] = f"{type(err).__name__}: {err}"
    report["contract"] = contract
    if contract is None:
        report["notes"].append(
            "内嵌契约读不出来 ✗ ⇒ ``plan_upscale`` 会**回退普通模式**（理由就是上面那条 ✓）"
            "—— ⚠️ 不是「文件坏了」的笼统话：真因在 ``contractError`` 里 ✓✗")
    else:
        report["notes"].append(
            f"契约 OK ✓（``in_channels={contract['base_config']['in_channels']}`` ✓、"
            f"``width={contract['config']['width']}`` / ``heads={contract['config']['heads']}`` ✓）"
            f"⇒ 2× 可做 ✓；⚠️ **>2× 要调用方显式给 ``maxTile``** ✗（本仓不猜安全块大小 ✓）")
    report["bytes"] = target.stat().st_size
    return report


def stage_entries(stage: str = "h3") -> list[dict[str, Any]]:
    """某阶段该看哪些清单条目 ✓ —— **筛选口径只此一份** ✗✗。

    ⚠️ 这是 2026-10-31 抽出来的单一真源 ✗：抽之前 :func:`readiness` 与
    ``loader.plan_stage`` **各写了一遍** ``category`` 筛选 ✓✗，两份都没认 ``tags`` ——
    加 ``zimage`` 档（靠 ``tags=["zimage"]`` 与 SDXL 区分）当场暴露：``plan_stage("zimage")``
    会把 `sdxl_base` 也列进去 ✓，而 ``required`` 又只取清单字段（那三条是 ``false`` ✓）
    ⇒ ``ready = all([]) = True`` ✓✗ —— **三份权重一份都没有时它说"就绪"** ✓✗，
    比 :func:`loader.residency_plan` 里那条「必需组件全缺却 `fits: true`」更坏 ✓。

    规则（与 :data:`STAGE_FILTERS` 逐条对应 ✓）：``category`` 必筛 ⇒ ``tags`` 交集非空则再筛 ⇒
    ``requireAll`` 时把筛出来的**副本**的 ``required`` 覆盖为 ``True``（清单文件与别的档不受影响 ✓）。
    """
    entries = load_catalog()["models"]
    rule = STAGE_FILTERS.get(stage)
    if not rule:
        return list(entries)
    entries = [item for item in entries if item.get("category") == rule["category"]]
    wanted_tags = set(rule.get("tags") or ())
    if wanted_tags:
        entries = [item for item in entries
                   if wanted_tags & set(item.get("tags") or ())]
    if rule.get("requireAll"):
        entries = [{**item, "required": True} for item in entries]
    return entries


#: 清单 ``kind`` → **装配槽位** ✓（一份表，装配 / 前端选档都读它 ✓）
#:
#: ⚠️ 不能靠"文件名里有没有 dit"猜 ✗：``ae.safetensors``（VAE）、``qwen_3_4b``（TE）都不带族名 ✓，
#: 而 ``wan_2.1_vae`` 既像 wan 又像 vae ✓ —— 唯一可靠的是**清单自己写的 kind** ✓。
COMPONENT_SLOTS: dict[str, str] = {
    "diffusion_models": "dit",
    "unet": "dit",
    "text_encoders": "text",
    "clip": "text",
    "vae": "vae",
    "checkpoints": "checkpoint",
}

#: ⚠️⚠️ **只有 H3** 的 VAE 进 ``video_vae_weights`` ✓✗ —— 别按"这档出视频"归 ✗✗：
#: Wan2.2 / SEEDVR2 虽然也出 mp4 ✓，但它们的装配线上那一格**就叫** ``image_vae_weights`` ✓
#: （实测：按"视频档"塞 ``video_vae_weights`` ⇒ 装载当场报「还没给 `imageVaeWeights`」✓✗，
#: 而 ``video_vae_weights`` 在 runtime 里只服务 ``real_video_vae`` ＝ H3 真视频 VAE 那一条 ✓）。
#: H3 本身走 :func:`bridge.assembly_from_config`（上面早返回 ✓），这张表只是把口径钉住 ✓。
_VIDEO_VAE_STAGES = ("h3",)

#: ⭐ **"这一档的『VAE』那一格叫什么参数"** ✓（2026-11-06 接 Music3 时把口径从"两选一"改成表 ✓）。
#:
#: ⚠️ 原来是 ``key = "video_vae_weights" if stage in _VIDEO_VAE_STAGES else "image_vae_weights"``
#: ✗ —— **两个分支都不是** Music3 要的那一格 ✓✗：它的解码器是 **DAV**（音频 ✓），
#: 装配线认的参数名是 ``audioVaeWeights`` ✓ ⇒ 按老写法会塞成 ``image_vae_weights`` ✓✗，
#: 而 ``_ensure_music3_loaded`` 缺 ``audio_vae_weights`` ⇒ 报「缺 `audioVaeWeights`」✓，
#: 读起来像"清单里没有 DAV"✓✗ —— 其实清单里**有** ✓，是**插错了口** ✓。
#: 这与 ``realAudioVae`` 那个坑**同形** ✓：同一个文件、两个名字、塞错不报错 ✓✗。
_SLOT_BY_STAGE_VAE: dict[str, str] = {
    "h3": "video_vae_weights",
    "music3": "audio_vae_weights",
}
#: 缺省仍回 ``image_vae_weights`` ✓（图族四档 ＋ seedvr2 / wan22 ✓ 都用这一格 ✓）。
_DEFAULT_VAE_SLOT = "image_vae_weights"


#: H3 自动装配除 DiT 之外**必须**补的两件 ✓（key 是清单里的 key ✓；词表不入清单 ✓）
#:
#: ⚠️ 文本塔这里要的是 ``int8_convrot`` 那份（**25.28 GiB** ✓），**不是** `te_nvfp4` ✗✗ ——
#: 两者都是 Qwen3-VL-32B ✓、文件名也几乎一样 ✓，但：
#: ① ``attach_h3_text`` 只认 **int8 常驻**（350 个线性层 ＋ 逐层 scale），nvfp4 件里**一个 int8
#:    权重都没有** ⇒ 装载当场拒 ✓；
#: ② 更要命的是 nvfp4 的**形状是打包的**（``q_proj`` 落成 ``[out, in/2]`` ✓）⇒
#:    `qwen3vl_te.infer_shape` 读出 hidden **2560**（真值 5120 ✓）⇒ `_check_text_encoder`
#:    报「TE 配置自述 2560 ✗ ≠ 主干要的 5120」✓ —— 读起来像"拿错文本塔"✓✗，
#:    其实是**打包没解开**（2026-11-03 实测就是这么报的 ✓）。
_H3_TEXT_KEY = "te_h3_int8_convrot"
_H3_VIDEO_VAE_KEY = "vae_video_fp16"
#: H3 的**音频** VAE ✓（`vae_audio_fp32` = 0.56 GiB 那份 ✓）——它与视频 VAE **毫无关系** ✗✗
#: （DAC 血统编码 ＋ BigVGAN 解码 ✓，见 `vae.H3_AUDIO_VAE_FACTS` ✓），不能互相顶替 ✓✗。
_H3_AUDIO_VAE_KEY = "vae_audio_fp32"


def _h3_real_parts(assembly: dict[str, Any], root: Path | None) -> tuple[list[str], list[str]]:
    """把 H3 的「真文本塔 ＋ 真视频 VAE ＋ 词表」补进 ``assembly`` ✓ ⇒ ``(notes, missing)`` ✓。

    ## 为什么必须在**装配参数**这一层补 ✗✗

    H3 原来只从清单拿 **DiT** ✓（``bridge.assembly_from_config`` 只管主干 ✓）⇒ 另外三格
    （``text_weights`` / ``tokenizer_path`` / ``real_video_vae``）**一个都不带** ✓✗
    ⇒ 运行时按"没给就挂桩"的纪律挂上**未训练**的参考 TE / VAE ✓ —— 装载、采样、写盘
    **全部成功** ✓、文件也是真 mp4 ✓，但：

    ① 画面是噪声 ✓✗；
    ② **参考 VAE 的时间压缩比是 1** ✗ ⇒ 4 倍时间压缩**不展开** ✓✗ ⇒ 计划 39 帧只写出
       **10 帧**（＝潜帧数 ✓）、时长 0.42 秒而音频 1.63 秒 ✓✗（音画不同步 ✓）。

    2026-11-03 实测就是这样：``/engine/generate`` 走 ``stage=h3``（画布点"视频"）拿到的
    mp4 只有 10 帧、note 写着"画面来自未训练参考 VAE" ✓。所以这里按清单把三格补上 ✓；
    缺件**如实**记进 ``missing`` ✓（不塞盘上不存在的路径 ✗ —— 那是"装到一半才炸" ✗）。
    """
    notes: list[str] = []
    missing: list[str] = []
    if not str(assembly.get("text_weights") or "").strip():
        entry = catalog_entry(_H3_TEXT_KEY)
        path, source = resolve_component(entry, root) if entry else (None, "")
        if path is not None and path.is_file():
            assembly["text_weights"] = str(path)
            notes.append(f"文本塔用 {path.name}（{source}）")
        else:
            missing.append(f"text:{_H3_TEXT_KEY}（清单落点没有这份文件）")
    if not str(assembly.get("tokenizer_path") or "").strip():
        from . import tokenizer_hub  # 局部 import ✓（避开循环 ✓）

        found = None
        try:
            found = tokenizer_hub.find_h3_tokenizer()
        except Exception as err:  # noqa: BLE001 —— 词表缺 ⇒ 运行时点名，这里不拦 ✓
            notes.append(f"词表探测出错：{str(err)[:80]}")
        if found and found.get("path"):
            assembly["tokenizer_path"] = str(found["path"])
            notes.append(f"词表用 {found.get('source')}（{found.get('tokenizerClass')}）")
        else:
            missing.append(
                "tokenizer（Qwen3-VL-32B 的词表 —— 放一份到 "
                f"<模型根>/tokenizers/{tokenizer_hub.H3_TOKENIZER_SUBDIR.split('/')[-1]}/ 即可自动识别）")
    if not assembly.get("real_video_vae"):
        entry = catalog_entry(_H3_VIDEO_VAE_KEY)
        path, source = resolve_component(entry, root) if entry else (None, "")
        if path is not None and path.is_file():
            #: ⚠️ 这一格**同时**打开 `real_video_vae` ✓✗：只塞 `video_vae_weights` 而不开这面旗
            #: ⇒ 运行时仍挂参考族 ✓（`_attach_video_vae(real=False)` 会把路径忽略掉 ✓✗）。
            assembly["real_video_vae"] = True
            assembly["video_vae_weights"] = str(path)
            notes.append(f"视频 VAE 用 {path.name}（{source}；装配期落 cpu、decode 前搬上卡）")
        else:
            missing.append(f"vae:{_H3_VIDEO_VAE_KEY}（清单落点没有这份文件）")
    #: ⭐ **音频 VAE**（2026-11-04 补 ✓✗）：H3 是**双流** ✓（视频 ＋ 音频 ✓），而上面的三格只管
    #: 画面 ⇒ 音轨仍由 `attach_audio_vae()` 挂**参考实现**（未经训练 ✗）⇒ 出来是噪声 ✓✗。
    #: 症状极隐蔽 ✓✗：mp4 是真的、时长也对、`ffprobe` 甚至能看到 aac 流 ✓ ——
    #: 只有**听**才发现是噪声，而回执原先只报 `vaeKind` 一个字段 ✓✗（音频侧无处可查 ✓）。
    #: ⚠️ 与视频侧**同一条纪律**：这一格必须**同时**打开 `real_audio_vae` ✓✗（只塞路径不开旗
    #: ⇒ `_attach_reference_audio_vae` 仍走参考分支 ✓）。
    if not assembly.get("real_audio_vae"):
        entry = catalog_entry(_H3_AUDIO_VAE_KEY)
        path, source = resolve_component(entry, root) if entry else (None, "")
        if path is not None and path.is_file():
            assembly["real_audio_vae"] = True
            assembly["audio_vae_weights"] = str(path)
            notes.append(f"音频 VAE 用 {path.name}（{source}；BigVGAN 解码 ✓ 0.56 GiB ✓）")
        else:
            missing.append(f"audioVae:{_H3_AUDIO_VAE_KEY}（清单落点没有这份文件）")
    return notes, missing


def assembly_for_stage(stage: str = "h3", *, root: Path | None = None) -> dict[str, Any]:
    """某档的清单条目 → ``ensure_loaded`` 的装配参数 ✓（**盘上没有的件一律不塞** ✗）。

    为什么要有它：``/engine/generate`` 原来只吃**显式路径** ✓（``ditPath`` / ``textWeights`` …），
    于是画布想跑一张图，得先知道 ``qwen_image_2.1_int8_convrot.safetensors`` 在哪 ✓✗ ——
    那是让前端复刻一遍清单解析 ✗。这里按 ``STAGE_FILTERS`` 把件挑出来，
    路径仍由 :func:`component_path` **单一真源**算 ✓。

    ⚠️ 缺件**不报错、不塞期望路径** ✗：塞一个盘上不存在的路径 ⇒ 运行时读文件时才炸 ✓✗，
    报错读起来像"权重坏了"✓；不塞 ⇒ ``_ensure_*_loaded`` 当场点名缺哪一格 ✓（含期望落点 ✓）。

    ``note`` 里回报"这一档挑到了什么、缺什么"✓ —— 前端不必再问一次清单 ✓。
    """
    # ⚠️ H3 **不在这里挑** ✗✗：它的 ``STAGE_FILTERS`` 只有 ``category=video`` ✓，
    #    会把 SEEDVR2 / Wan2.2 的件一起列出来（同一 category ✓）—— 按 kind 挑必然挑错主干 ✓✗。
    #    H3 的口径是「fl2va 主件 + 双流 VAE」，那是 :func:`bridge.assembly_from_config` 的活 ✓。
    if stage == "h3":
        try:
            from . import bridge  # 局部 import ✓（避开循环 ✓）

            assembly = dict(bridge.assembly_from_config({}, stage="h3"))
            notes, missing = _h3_real_parts(assembly, root)
            if notes or missing:
                assembly["note"] = "；".join(
                    notes + ([f"缺 {len(missing)} 件：" + "、".join(missing)] if missing else []))
            return assembly
        except Exception as err:  # noqa: BLE001 —— 探不到就如实说 ✓（不让它变成 500 ✗）
            return {"stage": "h3", "note": f"H3 自动装配没成：{str(err)[:160]}"}

    picked = [item for item in stage_entries(stage)
              if COMPONENT_SLOTS.get(str(item.get("kind") or ""))]
    # 一档里既有必需件又有可选件时**只挑必需的** ✓：``image`` 档把五条链的件全列了 ✓✗，
    # 照单全挑会得到"六个 dit"（那不是在装配一条链 ✗）。
    entries = [item for item in picked if item.get("required", True)] or picked
    assembly: dict[str, Any] = {"stage": stage}
    notes: list[str] = []
    missing: list[str] = []
    dits: list[Path] = []
    slot_paths: dict[str, Path] = {}

    for entry in entries:
        slot = COMPONENT_SLOTS[str(entry.get("kind"))]
        path = component_path(entry, root)
        if path is None:
            missing.append(f"{slot}:{entry.get('key')}（清单没落点）")
            continue
        if not path.is_file():
            missing.append(f"{slot}:{entry.get('key')}（{path.name} 不在盘上）")
            continue
        if slot == "dit":
            dits.append(path)
        else:
            slot_paths.setdefault(slot, path)

    # DiT：单件档直接给 dit_path ✓；Wan2.2 是**双专家** ✓（高噪 / 低噪各一份 ✓）
    if len(dits) == 1:
        assembly["dit_path"] = str(dits[0])
    elif len(dits) > 1:
        high = next((p for p in dits if "high" in p.name.lower()), None)
        low = next((p for p in dits if "low" in p.name.lower()), None)
        if high and low:
            assembly["dit_path"] = str(high)
            assembly["dit_path2"] = str(low)
        else:
            # 分不出高/低就按名字排序取前两件 ✓，并在 note 里**说清是猜的** ✗（不装作确定 ✓）
            ordered = sorted(dits, key=lambda p: p.name.lower())
            assembly["dit_path"] = str(ordered[0])
            assembly["dit_path2"] = str(ordered[1])
            notes.append("DiT 有两件但文件名分不出高/低 ⇒ 按名字排序取前两件（不确定 ✓）")

    if "text" in slot_paths:
        assembly["text_weights"] = str(slot_paths["text"])
    if "vae" in slot_paths:
        assembly[_SLOT_BY_STAGE_VAE.get(stage, _DEFAULT_VAE_SLOT)] = str(slot_paths["vae"])
    if "checkpoint" in slot_paths:
        # SDXL 走 ``dit_path``（它的"主干"就是 checkpoint ✓，见 ``bridge.resolve_sdxl_path`` ✓）
        assembly.setdefault("dit_path", str(slot_paths["checkpoint"]))

    # 词表：Qwen 系四档都要 ✓（不是权重 ✗ ⇒ 清单里没有 ✓）
    #
    # ⚠️ **按档问** :func:`tokenizer_hub.find_tokenizer_for_stage` ✓（目录名/仓库名认族 ✓
    # ＋ ``tokenizer_class`` 排错族 ✓ 双判据 ✓）。第一版这里**拿 H3 的 Qwen3-VL-32B 词表
    # 顶过所有档** ✗✗：四档文本塔分别是 Qwen3-4B / Qwen3.5-9B / Qwen2.5-VL-7B / Qwen3-VL-4B ✓，
    # 喂错族「装得上、编得出、id 全错」且**一条报错都没有** ✗✗。
    # 同理不许用 ``bridge.resolve_tokenizer_path`` ✗✗（那是 CLIP-BPE ✓，只有 SDXL 吃它 ✓）。
    # 找不到就**不塞** ✗ ⇒ 运行时点名缺词表 ✓，并把"该往哪放"写进 note ✓。
    #: ⚠️ ``music3`` **必须**在跳过名单里 ✗✗（2026-11-06 ✓）：它的词表**焊在语义塔权重里** ✓
    #: （safetensors 的 ``tokenizer_json`` 键 ✓），盘上**没有** ``tokenizer.json`` ✓ ——
    #: 让它去找词表 ⇒ 永远"缺一件" ✓✗，而那一件**根本不该存在** ✓（用户会去下载一份别族的 ✓✗，
    #: 装是装得上 ✓、id 语义全错 ✓ —— 见 :meth:`torch_backend.TorchBackend.attach_music3_te` ✓）。
    if stage not in ("h3", "sdxl", "seedvr2", "wan22", "music3"):
        from . import tokenizer_hub  # 局部 import ✓（避开循环 ✓）

        found = None
        probe_error = ""
        try:
            found = tokenizer_hub.find_tokenizer_for_stage(stage)
        except Exception as err:  # noqa: BLE001 —— 词表缺 ⇒ 运行时点名，这里不拦 ✓
            probe_error = str(err)[:80]
        if found and found.get("path"):
            assembly["tokenizer_path"] = str(found["path"])
            notes.append(f"词表用 {found.get('source')}（{found.get('tokenizerClass')}）")
        elif probe_error:
            missing.append(f"tokenizer（探测出错：{probe_error}）")
        else:
            spec = tokenizer_hub.STAGE_TOKENIZERS.get(stage, {})
            where = spec.get("subdir") or "该档"
            missing.append(
                f"tokenizer（{spec.get('why', '该档')}——放一份到 "
                f"<模型根>/tokenizers/{where}/ 或仓内 models/tokenizers/{where}/ 即可自动识别）")

    if notes or missing:
        assembly["note"] = "；".join(notes + ([f"缺 {len(missing)} 件：" + "、".join(missing)] if missing else []))
    return assembly


def readiness(stage: str = "h3", *, root: Path | None = None,
              capacity_gib: float = DEFAULT_CAPACITY_GIB,
              upscale_key: str = "", upscale_path: str = "") -> dict[str, Any]:
    """某阶段的**就绪报告** ✓：必需组件齐不齐、缺多少、显存够不够估。

    ``required`` 由清单的 ``required`` 字段决定 ✓（``requireAll`` 的档见 :func:`stage_entries` ✓）；
    非必需组件也会列出来（可选件 ✓），但**不影响** ``ready`` 判定 ✓。
    """
    entries = stage_entries(stage)
    #: ⚠️ 报告里要带上「清单自己的错」✓（``load_catalog`` 解析失败时 ``models`` 是空表 ✓✗ ——
    #: 不报这一栏，读者只会看到「组件一个都没有」而不知道是清单坏了 ✓✗）。
    catalog = load_catalog()

    components = [component_status(entry, root) for entry in entries]
    required = [item for item in components if item["required"]]
    missing = [item for item in required if not item["present"]]
    #: **阻断**就绪：结构上不成立（截断 / 头部矛盾 / 读不了 ✓）
    #: ⚠️ 这里**不能**再排除 ``sizeMismatch`` ✗ —— 初版加了 `and not sizeMismatch`，
    #:    结果「截断的文件」因为同时「大小也不符」而被漏出断障名单 ✗（自检 ⑳ 当场抓到 ✓）。
    #:    正确语义：**结构坏 ⇒ 一律阻断**；只有**结构完好**的文件才可能"仅可疑" ✓。
    broken = [item for item in required if item["present"] and not item["verified"]]
    #: **不阻断**：结构完好但大小与清单不符（可疑 ⚠️ —— 社区量化版/重新导出会让体积合法地变 ✓）
    suspect = [item for item in required
               if item.get("sizeMismatch") and item["verified"]]
    optional_missing = [item["key"] for item in components
                        if not item["required"] and not item["present"]]

    # ⭐ **"不在 `models_dir`、但引擎加载会用"的那些** ✓ —— ⚠️ 2026-09-27 重修 ✓✗：
    #    以前这里做的是"给 `missing` 补一份"✗，且**不**改 `ready`（口径是：`missingRequired` 按
    #    `models_dir` 说 ✓）⇒ 结果报告自相矛盾 ✓✗：说「缺 `dit_fl2va_int8` ✗、还要下 19.53 GiB」，
    #    可那份**就在**探测到的 ComfyUI 共享目录里 ✓、引擎加载也真会用它 ✓（用户当场指出 ✓）。
    #    现在 :func:`component_status` **自己**就走"清单落点优先 → 动态探测根" ✓ ⇒ 这类件**已计入**
    #    `present` / `bytes` / `requiredWeightsGiB` / `ready` ✓（因为"能不能跑"取决于引擎找不找得到 ✓）。
    #    ⇒ 本表只剩**另一个问题** ✓：哪些件**不在**本仓 `models_dir` 里（想让本仓自持 / 想整目录搬走
    #    ⇒ 得把它们收进去 ✓）。⚠️ 只对 ``root is None`` 有意义 ✗（钉了根 ⇒ 那一整个域就是调用方给的 ✓）。
    #    ⚠️ **不再走盘** ✓：判据就是 `component_status` 已经记下的 `source` ✓（体检不该为一句提示扫全盘 ✓）。
    resolved_elsewhere: dict[str, dict[str, str]] = {}
    if root is None:
        for item in components:
            source = str(item.get("source") or "")
            if item["present"] and source and source != "清单落点":
                resolved_elsewhere[str(item["key"])] = {"path": str(item.get("path") or ""),
                                                        "source": source}

    weights_bytes = sum(int(item["bytes"]) for item in components if item["required"])
    weights_gib = round(weights_bytes / 1024 ** 3, 2)
    return {
        "stage": stage,
        "catalogError": catalog.get("error"),
        "modelsDir": str(root or models_dir() or ""),
        "ready": not missing and not broken,
        "components": components,
        "missingRequired": [item["key"] for item in missing],
        "brokenRequired": [item["key"] for item in broken],
        "suspectRequired": [item["key"] for item in suspect],
        "optionalMissing": optional_missing,
        # ⭐ "**在盘上、但不在 ``models_dir``**"的那些 ✓ —— 它们**已经算进** `ready` ✓（引擎加载会用它 ✓）；
        #    本表只提醒"想让本仓自持就得收进去"✓（口径见上面那段注释 ✓）。
        "resolvedElsewhere": resolved_elsewhere,
        "resolvedElsewhereNote": (
            "这些件**已经算进** ``ready`` / ``requiredWeightsGiB`` ✓ —— 引擎加载时会在这些位置找到它们 ✓"
            "（清单落点优先 ✓ → 动态探测根 ✓，与加载同一条解析 ✓）；⚠️ 但它们**不在** ``models_dir`` ✓"
            "⇒ 想让本仓**自持**（能整目录搬走 / 换机器不靠别人的 ComfyUI ✓）就把它们收进 ``models_dir`` ✓"),
        "requiredWeightsGiB": weights_gib,
        "vram": estimate_vram(weights_bytes, capacity_gib=capacity_gib),
        "catalogNodes": len(catalog.get("nodes") or []),
        # ⭐ 档位表随就绪报告一起给 ✓（2026-09-24 接的 ✓）：步数 / 分辨率 / 加速件 /
        #    显存建议 / **必备模型** ✓ —— ⚠️ 没给容量时各档的 advised 是「**没比**」✗ 不是通过 ✓。
        "tiers": tiers_mod.summary(gpu_gib=capacity_gib if capacity_gib else None),
        # ⭐ 超清放大器可用性 ✓（2026-09-24 接的 ✓）：⚠️ 没给 key/path ⇒ ``checked=False``
        #    **不是**「超清可用」✗✗（本仓铁律：没查 ≠ 通过 ✓）。
        "upscale": upscale_status(root=root, key=upscale_key, path=upscale_path),
    }


def estimate_vram(weights_bytes: int, *, activation_bytes: int = 0,
                  safety: float = 1.25, capacity_gib: float = DEFAULT_CAPACITY_GIB) -> dict[str, Any]:
    """显存粗估 ✓ —— **权重是精确值，激活是调用方给的，乘安全系数后才是"估"** ✓。

    ⚠️ 诚实边界：真正的峰值还取决于**实现细节**（offload 策略、注意力算法、分块解码 ✓）⇒
    这里给的是「**下界 + 安全系数**」的量级判断 ✓，**不是承诺** ✗。跑起来后用 ``nvidia-smi`` 实测为准 ✓。
    """
    total = int((int(weights_bytes) + max(0, int(activation_bytes))) * max(1.0, float(safety)))
    total_gib = round(total / 1024 ** 3, 2)
    return {
        "weightsGiB": round(int(weights_bytes) / 1024 ** 3, 2),
        "activationGiB": round(max(0, int(activation_bytes)) / 1024 ** 3, 2),
        "safety": safety,
        "estimatedGiB": total_gib,
        "capacityGiB": capacity_gib,
        "fits": total_gib <= capacity_gib,
        "headroomGiB": round(capacity_gib - total_gib, 2),
        "disclaimer": "估算：权重精确、激活由调用方给；峰值还取决于 offload/注意力/分块实现 ✗ ⇒ 实测为准 ✓",
    }
