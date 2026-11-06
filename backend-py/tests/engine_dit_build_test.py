"""④⁗ 族装配表的守卫（`engine/dit_build.py`，2026-11-01）。

「抄 ComfyUI 好的那一半」的落地口径：**声明式**（一族一行）要抄 ✓，**隐式契约**（写漏不报错）
不抄 ✗。所以这张表必须自带三条断言，否则它只是把四个 `_build_*` 换成四行字典、**一点没少错**：

1. **表项完备** ✗✗：声明的模块 / 入口必须真实存在。表里写错一个函数名 ⇒ 只有走到那族才炸
   （而且报的是 `AttributeError`，看不出是表写错了）；
2. **判得出 ⇔ 建得了** ✗：`dit_family` 认得、`dit_build` 没登记 ⇒ 用户拿到"判族成功"
   却装不了（**假绿** ✓✗）。反过来登记了判不出的族 ⇒ 永远走不到（死配置）；
3. **收口** ✗：`torch_backend` 里不许再按族名分叉。加一族要动的地方必须**只有**
   `dit_family._RULES` + `dit_build.BUILDERS` 两处 —— 这条用源码断言守，别靠人记 ✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_dit_build_test.py
"""
from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
ENGINE = BACKEND_PY / "app" / "services" / "engine"
os.environ.setdefault("PROXY_TO_NODE", "0")
#: ⚠️ cp936 控制台下 ``✓`` 会 ``UnicodeEncodeError`` ⇒ 判据全过也照样崩在最后一行打印上 ✗
#: （``run_all`` 已重配 stdout，只有单独跑才露头 ⇒ 每个 ``__main__`` 测试都自带这一句 ✓）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(BACKEND_PY))

from app.services.engine import dit_build as bd  # noqa: E402
from app.services.engine import dit_family as fam  # noqa: E402
from app.services.engine import latent_formats as lf  # noqa: E402
from app.services.engine import torch_backend as tb  # noqa: E402
from app.services.engine.safetensors import SafetensorsInfo, TensorEntry  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []

#: **只判族、不建网**的族（有前向骨架但不在 DiT 装配这条路：H3 走 `h3_form`、SDXL 走
#: `sdxl_backend`）。新增族**不许**往这里塞 ✓✗（只有"族判得出来、但装配这条路根本没入口"才配）
#: ⚠️ `music3` 2026-11-02 也移出去了 ✓：`dit_music3` 已与参考 `minimax_music/dit.py` 逐位
#: 对拍 ✓（**真实 2.46B fp16 权重**下 `maxdiff = 4.8e-06` ✓、相对 1.2e-06 ✓ ⇒ 只有
#: sdpa/求和顺序的差 ✓）、且 `load_state_dict` 0 missing / 0 unexpected ✓ ⇒ 走装配表 ✓。
#: **但**「建得出 + 前向逐位对得上」≠「出得了歌」✓✗：条件来源（TE+AR）与输出侧（DAV）都还欠着 ✓
#: （`dit_build._music3_sources()` 的 `audioPipeline` 那格如实写着 ✓）。
#: ⚠️ `flux` **不在**这张表里 ✓（2026-11-01 补 ✓）：`dit_flux` 的骨架与 Flux.1-dev 的键已经
#: 逐行核过 ✓ ⇒ 它走的是**装配表**（`dit_build.BUILDERS["flux"]` ✓），不是老路 ✓。
#: ⚠️ `wan` 同一天也移出去了 ✓：`dit_wan22` 的 1095 键已与盘上 14B fp8_scaled 逐键对账 ✓
#: ⇒ 走装配表 ✓。**但**「建得出」≠「跑得通」✓✗：I2V 的 36 通道条件拼接与 28.6 GiB 常驻还欠着 ✓
#: （`dit_build._wan_sources()` 的 `i2vConcat` / `residency` 两格如实写着 ✓）。
#: ⚠️ `seedvr2` 也是同一天移出去的 ✓：`dit_seedvr2` 的 3B 键已与盘上 fp16 权重逐键对账、
#: 且**真实权重前向对拍 `maxdiff == 0`** ✓ ⇒ 走装配表 ✓。**但**「建得出 + 前向对得上」≠
#: 「端到端能出片」✓✗：17 通道 conditioning 的**来源**（现状视频 + loss mask）与 VAE 侧
#: 都还欠着 ✓（`engine_seedvr2_test` 的 ⑬⑭ 两组只核到「建壳 → 装权重 → 前向 finite」✓）。
DETECT_ONLY: frozenset[str] = frozenset({"h3", "sdxl"})


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def src_backend() -> str:
    """``torch_backend.py`` 源码 ✓（源码级断言用 ✓ —— 判据没接上线时，只有它能看出来 ✓）。"""
    return (ENGINE / "torch_backend.py").read_text(encoding="utf-8")


def _info(pairs: dict[str, Any]) -> SafetensorsInfo:
    """``{名字: 形状}`` ⇒ 一次 `inspect` 的结论（只造元信息，不写字节 ✓）。"""
    tensors: dict[str, TensorEntry] = {}
    for name, shape in pairs.items():
        tensors[name] = TensorEntry(name=name, dtype="BF16", shape=[int(x) for x in shape],
                                    begin=0, end=0)
    return SafetensorsInfo(path="<合成>", tensors=tensors)


def krea2_pairs(*, blocks: int = 2, hidden: int = 512) -> dict[str, Any]:
    """Krea2 最小骨架（键名照实测；`mod.lin` 是**裸参数**，写到 `.weight` 就判不出族 ✗）。"""
    pairs: dict[str, Any] = {}
    for n in range(blocks):
        head = f"blocks.{n}."
        pairs[head + "attn.wq.weight"] = (hidden, hidden)
        pairs[head + "attn.wk.weight"] = (hidden, hidden)
        pairs[head + "attn.wo.weight"] = (hidden, hidden)
        pairs[head + "attn.gate.weight"] = (hidden, hidden)
        pairs[head + "attn.qknorm.qnorm.scale"] = (128,)
        pairs[head + "mlp.gate.weight"] = (hidden, hidden)
        pairs[head + "mlp.up.weight"] = (hidden * 4, hidden)
        pairs[head + "mlp.down.weight"] = (hidden, hidden * 4)
        pairs[head + "prenorm.scale"] = (hidden,)
        pairs[head + "mod.lin"] = (hidden * 6,)
    return pairs


def krea2_full_pairs(*, blocks: int = 2, hidden: int = 512, channels: int = 16,
                     txtdim: int = 64, txtlayers: int = 4, tdim: int = 256
                     ) -> dict[str, Any]:
    """**能推得出结构**的 Krea2 骨架（`:func:`krea2_pairs` 只够判族，不够 `infer_shape` ✓）。

    ⚠️ 两者**不是**一个东西 ✗：判族只看头部几个键 ✓；``infer_shape`` 还要 ``first``/``last``/
    ``tmlp``/``txtmlp``/``txtfusion.projector`` 才能把「入维、patch²、文本层数」互相证出来 ✓。
    拿判族用的那半边去喂装配 ⇒ ``ShapeError: 权重里缺 `first.weight``` ✓（本次踩过 ✓）。
    """
    patch = 2
    pairs = krea2_pairs(blocks=blocks, hidden=hidden)
    pairs["first.weight"] = (hidden, channels * patch * patch)
    pairs["last.linear.weight"] = (channels * patch * patch, hidden)
    pairs["tmlp.0.weight"] = (hidden, tdim)
    pairs["txtmlp.1.weight"] = (hidden, txtdim)
    pairs["txtfusion.projector.weight"] = (1, txtlayers)
    for group in ("layerwise_blocks", "refiner_blocks"):
        head = f"txtfusion.{group}.0."
        pairs[head + "attn.wq.weight"] = (txtdim, txtdim)
        pairs[head + "attn.wk.weight"] = (txtdim, txtdim)
        pairs[head + "attn.qknorm.qnorm.scale"] = (txtdim,)
        pairs[head + "mlp.gate.weight"] = (txtdim * 2, txtdim)
    return pairs


# ══════════════════════════════════════════════════════════════════════════
# ① 表项完备：写进表的名字必须真的存在
# ══════════════════════════════════════════════════════════════════════════
def case_table_complete() -> None:
    missing: list[str] = []
    loaded = bd.skeletons()
    for key, spec in bd.BUILDERS.items():
        mod = loaded.get(spec.module)
        if mod is None:
            missing.append(f"{key}: 模块 {spec.module} 不在 SKELETONS（没有静态 import ⇒ 守卫看不见）")
            continue
        for attr in (spec.shape_fn, spec.build_fn):
            if not hasattr(mod, attr):
                missing.append(f"{key}: {spec.module} 缺 {attr}()")
        if spec.key != key:
            missing.append(f"{key}: 声明的 key={spec.key} 与表键不一致")
        if not spec.label:
            missing.append(f"{key}: label 空（报错文案会变成空族名）")
        for target, _source in spec.shortcuts:
            if not target:
                missing.append(f"{key}: shortcuts 里有空目标字段")
    check("① 每个表项的模块 / 入口 / label / 改名目标都真实存在 ✓（写错名字只有走到才炸 ✗）",
          not missing, missing)

    check("① 表项的族都判得出来（`dit_family.FAMILY_KEYS` 里有它 ✓）",
          all(key in fam.FAMILY_KEYS for key in bd.BUILDERS),
          [k for k in bd.BUILDERS if k not in fam.FAMILY_KEYS])

    #: ⚠️ 骨架模块必须是**静态 import** ✓，不许 `importlib.import_module(spec.module)` ✗✗：
    #: 按名字动态取 ⇒ 拼错要**装到那一族**才炸 ✓、结构守卫也扫不到这条线 ✓（报「0 处 import」✓）。
    src = (ENGINE / "dit_build.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    statically = {a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                  and (node.module or "").endswith("engine") for a in node.names}
    statically |= {a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                   and node.level == 1 for a in node.names}
    check("① 骨架模块是静态 import 的 ✓（守卫才看得见调用者 ✓）",
          set(bd.SKELETON_MODULES) <= statically,
          sorted(set(bd.SKELETON_MODULES) - statically))
    dynamic = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and getattr(node.func, "attr", "") == "import_module"]
    check("① `dit_build` 里没有 `import_module` 按名字取模块 ✗", not dynamic, len(dynamic))
    #: ⚠️⚠️ 骨架 import **不许在模块顶层** ✗✗（2026-11-01 二次修 ✓）：六个骨架顶层必然
    #: `import torch` ✓，而 `capabilities()`／`readiness()`／`/engine/plan` 只读 `BUILDERS`
    #: 这张**纯数据表** ✓ ⇒ 顶层 import 让「没装 torch 的环境算不出能力」✓✗（独立性 ⑤ 会红 ✓）。
    import_where: dict[str, bool] = {}
    for node in tree.body:                      #: 只扫**模块顶层**语句 ✓
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            for alias in node.names:
                import_where[alias.name] = True
    check("①″ 六个骨架**不在模块顶层** import ✗✗（改回顶层 ⇒ 独立性 ⑤ 立刻红 ✓）",
          not (set(bd.SKELETON_MODULES) & set(import_where)),
          sorted(set(bd.SKELETON_MODULES) & set(import_where)))
    inside = {a.name for node in ast.walk(tree)              #: 函数体内也算 ✓（整棵树 ✓）
              if isinstance(node, ast.ImportFrom) and node.level == 1 for a in node.names}
    check("①‴ 名字表与源码里那条 `from . import` 的六个名字**逐一对应** ✓✗（加骨架要两处同改 ✓）",
          {n for n in inside if n.startswith("dit_") and n != "dit_build"}
          == set(bd.SKELETON_MODULES),
          sorted({n for n in inside if n.startswith("dit_") and n != "dit_build"}))

    generic = loaded.get("dit_generic")
    check("① 单流骨架**只有一个实现** ✓（`SingleStreamDiT is Krea2DiT` ⇒ 同构族加一行就够，不用再抄模块 ✓）",
          getattr(generic, "SingleStreamDiT", None) is not None
          and getattr(generic, "SingleStreamDiT") is getattr(generic, "Krea2DiT"),
          {"SingleStreamDiT": getattr(generic, "SingleStreamDiT", None),
           "Krea2DiT": getattr(generic, "Krea2DiT", None)})


# ══════════════════════════════════════════════════════════════════════════
# ② 判得出 ⇔ 建得了（两侧一一对上，不许有"判得出但没通路"的假绿）
# ══════════════════════════════════════════════════════════════════════════
def case_coverage() -> None:
    known = set(fam.FAMILY_KEYS)
    built = set(bd.BUILDERS)
    check("② 判得出来、也有装配声明的：判族集合 = 装配集合 ∪ 只判不建集合 ✓",
          built | DETECT_ONLY == known,
          {"少登记(判得出但建不了)": sorted(known - built - DETECT_ONLY),
           "多余(登记了但判不出)": sorted((built | DETECT_ONLY) - known)})
    check("② 只判不建的名单与装配表**不重叠** ✓（同一个族两条通路 = 迟早走错）",
          not (built & DETECT_ONLY), sorted(built & DETECT_ONLY))
    check("② `buildable_keys()` 就是表键（没有第二份口径 ✓）",
          bd.buildable_keys() == tuple(bd.BUILDERS), bd.buildable_keys())


# ══════════════════════════════════════════════════════════════════════════
# ③ 收口：`torch_backend` 里不许再按族名分叉
# ══════════════════════════════════════════════════════════════════════════
def case_no_fork() -> None:
    src = (ENGINE / "torch_backend.py").read_text(encoding="utf-8")
    check("③ `torch_backend` 里没有 `family ==` 式分叉 ✓（加族不该再动这个文件）",
          "family == " not in src and "family in (" not in src,
          src.count("family == "))

    tree = ast.parse(src)
    thin: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
                "_build_zimage_module", "_build_qwen21_module",
                "_build_qwen_image_module", "_build_krea2_module"}:
            thin[node.name] = len(node.body)
    check("③ 四个 `_build_*` 只剩薄壳（body ≤ 3 行 ✓；再长说明逻辑又搬回来了 ✗）",
          len(thin) == 4 and all(n <= 3 for n in thin.values()), thin)

    own = (ENGINE / "dit_build.py").read_text(encoding="utf-8")
    #: ⚠️ 别用 `"torch_backend" not in own` ✗ —— 文档里点名它是**该有的**（说明加族要改哪两处），
    #: 那条断言会因为它出现一次就假失败 ✓。要判的是**import 关系**（AST），不是字面量 ✓。
    imported: list[str] = []
    for node in ast.walk(ast.parse(own)):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.endswith("torch_backend"):
                imported.append(node.module)
            imported.extend(a.name for a in node.names if a.name == "torch_backend")
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names if a.name.endswith("torch_backend"))
    check("③ `dit_build` 不反向 import `torch_backend` ✓（单向依赖，防循环；只看 AST 不看字面量）",
          not imported, imported)


# ══════════════════════════════════════════════════════════════════════════
# ④ 表驱动分派真的生效：新登记一项 ⇒ 立刻可建（不需要改 torch_backend）
# ══════════════════════════════════════════════════════════════════════════
def case_table_dispatch() -> None:
    probe = bd.FamilyBuilder(
        key="_probe", label="探针族", module="dit_generic",
        #: 契约必须是**表里有的**口径 ✓（自造名 `probe` 会被当场拒 ✓ —— 这正是本文件 ⑦′ 的判据 ✓）。
        contract=bd.ForwardContract(latent_dims=4, sigma_form="batch", latent_format="qwen_image"),
        build_fn="Krea2DiT", build_args=("shape",), meta=True,
        shape_args=("patch",), shortcuts=(("in_channels", "channels"),),
        sources=(("结构", "权重形状反推 ✓"),))
    bd.BUILDERS["_probe"] = probe
    try:
        built = bd.build("_probe", _info(krea2_full_pairs()), {})
    finally:
        bd.BUILDERS.pop("_probe", None)
    config = built[1] if built else {}
    check("④ 只在表里加一行 ⇒ 立刻建得出来（`torch_backend` 一行没动 ✓）",
          built is not None and config.get("family") == "_probe"
          and config.get("layers") == 2 and config.get("in_channels") == 16
          and config.get("features") == 512,
          {k: config.get(k) for k in ("family", "layers", "features", "in_channels")})
    check("④ 临时项已清理（别把探针族留在全局表里 ✗）", "_probe" not in bd.BUILDERS)
    #: ⚠️ 探针必须挑一个**没进装配表**的族 ✓✗ —— 原来这里用 ``wan`` ✓，2026-11-01 起
    #: Wan2.2 进了 ``BUILDERS`` ✓ ⇒ 那一刻起这条断言测的就不是「没登记」而是「登记了但推不出」
    #: 的 ``BuildError`` ✓（断言会以异常形式炸掉 ✓，不是静默假绿 ✓）。改用 ``seedvr2`` ✓，
    #: 而 SEEDVR2 当天也进了装配表 ✓ ⇒ 现在用 ``h3``（``FAMILY_KEYS`` 里有、``BUILDERS`` 没有 ✓，
    #: 走的是 ``h3_form`` 那条独立通路 ✓）。
    check("④ 没登记的族 ⇒ 返回 None（调用方走老路 ✓，不是抛异常）",
          bd.build("h3", _info(krea2_pairs()), {}) is None)


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 推不出结构 ⇒ 当场拒（**不拿出厂形状硬装**）
# ══════════════════════════════════════════════════════════════════════════
def case_build_error() -> None:
    bad = _info({"blocks.0.attn.wq.weight": (512, 512)})
    err: Exception | None = None
    try:
        bd.build("krea2", bad, {})
    except bd.BuildError as exc:                       # noqa: PERF203 —— 就是要抓它
        err = exc
    check("⑤ 推不出可装的结构 ⇒ BuildError（reason=pending，文案点明不硬装 ✓）",
          isinstance(err, bd.BuildError) and getattr(err, "reason", "") == "pending"
          and "不拿出厂形状硬装" in str(err), type(err).__name__)

    wrapped: Exception | None = None
    try:
        tb._build_krea2_module(bad, {})
    except Exception as exc:                           # noqa: BLE001 —— 要看它包成什么
        wrapped = exc
    check("⑤ 薄壳把它包成后端不可用（`TorchBackendUnavailable`，reason=pending ✓）",
          wrapped is not None and "pending" in str(getattr(wrapped, "reason", "")),
          type(wrapped).__name__)


# ══════════════════════════════════════════════════════════════════════════
# ⑥ Krea2 前缀：一处判、三处用（形状 / int8 计划 / convrot 表剥的是同一层）
# ══════════════════════════════════════════════════════════════════════════
def case_krea2_prefix() -> None:
    prefixed = {f"model.diffusion_model.{k}": v for k, v in krea2_pairs().items()}
    shapes, key_map, prefix = bd.krea2_strip(prefixed)
    check("⑥ 带前缀 ⇒ 剥掉那一层，并给出 key_map（正则里点已转义 ✓）",
          prefix == "model.diffusion_model."
          and all(not k.startswith("model.") for k in shapes)
          and key_map == {"^model\\.diffusion_model\\.": ""}, (prefix, key_map))
    plain, none_map, none_prefix = bd.krea2_strip(krea2_pairs())
    check("⑥ 无前缀 ⇒ 原样返回、key_map=None（**不猜**前缀 ✗）",
          none_prefix is None and none_map is None and len(plain) == len(krea2_pairs()))
    specs = {"model.diffusion_model.blocks.0.attn.wq": 256, "blocks.1.attn.wq": 256}
    stripped = bd.krea2_specs(specs, "model.diffusion_model.")
    check("⑥ convrot 分组表剥同一层（漏剥 ⇒ 查空 ⇒ 静默不旋转 ✓✗）",
          stripped == {"blocks.0.attn.wq": 256, "blocks.1.attn.wq": 256}, stripped)


# ══════════════════════════════════════════════════════════════════════════
# ⑦⁵ int8 常驻：**谁在 GPU 上算**这件事的开关
# ══════════════════════════════════════════════════════════════════════════


def case_int8_adapter() -> None:
    """⑦⁵ int8 **常驻**接入（2026-11-01 ✓）—— 直接决定「GPU 能不能整份吃下这件」。

    ## 这条守的是什么 ✗

    盘上 Edit 2511 是 **840 层 int8 / 19.09 GiB** ✓，22.49 GiB 的卡本可以**整份吃下** ✓✗；
    老路把它反量化成 bf16（**38.05 GiB** ✓）⇒ 只能 split ✓ —— 实测 ``gpu 15.9 / cpu 22.15 GiB`` ✓✗
    （**58% 的权重在内存里逐层爬** ✓，这就是"绝大部分计算由 CPU 承担"最硬的一条 ✓）。

    判据三层：
      ① 这一族**接上了** adapter ✓，且**只有**它接（别的族盘上没有 int8 文件 ⇒ 接了就是死配置 ✓✗）；
      ② 接的时候**不剥前缀** ✓（Edit 的 840 个键与模块名 840/840 逐字对得上 ✓，没有
         Krea2 那种 ``model.diffusion_model.`` ✓）；
      ③ 没给权重路径（**自检路** ✓）**不许**换壳 ✓✗ —— 换壳必须按**逐层** convrot 说明来 ✓，
         自检没有文件、只能按形状建 bf16 ⇒ 硬换会让自检与真装载建出**两种骨架** ✓✗。
    """
    from torch import nn  # noqa: PLC0415

    carriers = {key for key, spec in bd.BUILDERS.items()
                if getattr(spec, "adapter", None) == "int8"}
    check("⑦⁵ qwen_image 接了 int8 常驻（Edit 2511 的 840 层 int8 ⇒ 驻留 38.05→19.09 GiB ✓）",
          "qwen_image" in carriers, sorted(carriers))
    check("⑦⁵ 只有 qwen_image 接（别的族盘上没有 int8 文件 ⇒ 接了就是死配置 ✓✗）",
          carriers == {"qwen_image"}, sorted(carriers))

    adapter = bd._ADAPTERS["int8"]                       # noqa: SLF001
    shapes_in = {"transformer_blocks.0.attn.to_q": (3072, 3072)}
    got, key_map, prefix = adapter.prepare(shapes_in)
    check("⑦⁵ prepare 不剥前缀、不给 key_map（键名就是模块名 ✓）",
          got == shapes_in and key_map is None and prefix is None, (got, key_map, prefix))

    tiny = nn.Sequential(nn.Linear(8, 8))
    plan, convrot, int4 = bd._shell_int8(tiny, _info({}), None, None, None)  # noqa: SLF001
    check("⑦⁵ 没给权重路径 ⇒ 空 plan、**不换壳**（自检口径不变 ✓✗；4bit 计划同样空 ✓）",
          plan == {} and convrot == 0 and int4 == {} and isinstance(tiny[0], nn.Linear),
          (plan, convrot, int4))

    #: ⚠️ 两个 adapter **必须**共用同一条实现 ✗✗（各抄一份 ⇒ 改一处漏一处 ⇒ 静默按 bf16 物化 ✓✗）
    import inspect  # noqa: PLC0415
    krea2_src = inspect.getsource(bd._Krea2Adapter.shell)            # noqa: SLF001
    check("⑦⁵ Krea2 与 Qwen-Image 走**同一条** `_shell_int8`（不许各抄一份 ✓）",
          "_shell_int8(" in krea2_src, krea2_src.strip().splitlines()[-1][:60])


# ══════════════════════════════════════════════════════════════════════════
# ⑦ 前向契约：潜变量几维 / σ 什么形态，**只有表里那一份**
# ══════════════════════════════════════════════════════════════════════════

#: 冻结快照 —— 改契约必须**手改本表** ✓（＝一次有意的决定 ✓，而不是顺手改个数字 ✓）。
#: ⚠️ ``latent_format`` 必须是 :mod:`latent_formats` 的**规范名** ✓✗（``flux8`` / ``qwen21``
#: 这种自造名字表里没有 ✓ ⇒ 以前写错了也没人发现 ✓；现在构造期就归一化 ✓）。
CONTRACTS: dict[str, tuple[int, str, str]] = {
    "zimage": (4, "batch", "flux"),            # Flux 的 ae VAE（仿射 0.3611/0.1159 ✓）
    "qwen_image21": (4, "batch", "qwen_image21"),   # 64ch ✓
    "qwen_image": (4, "batch", "qwen_image"),  # 16ch **恒等** ✓（**不是** wan21 ✗）
    "krea2": (4, "batch", "wan21"),            # 16ch 但**真 Wan 均值/方差** ✓（文件同档、口径不同 ✓）
    "flux": (4, "batch", "flux"),
    #: ⭐ **五维**（唯一一个真建的视频族 ✓）：``[B,16,T,H/8,W/8]`` ✓ —— 主干输入是 36 通道
    #: （噪声 16 + 条件图 16 + 掩码 4 ✓），但**潜变量/噪声是 16** ✓，两个数别混 ✓✗；
    #: ``wan21`` 口径 = 16ch / 8× / **meanstd** ✓（解码前要按 ``vae_wan.latents_stats`` 反算 ✓）。
    "wan": (5, "batch", "wan21"),
    #: ⭐ **SEEDVR2 / NaDiT**：也是五维 ✓，但**口径独立** ✓（16ch / 3 维 / **scale 1.0 恒等** ✓，
    #: 既不是 ``wan21`` 的 meanstd ✓、也不是 ``qwen_image`` 的仿射 ✓ ⇒ 挂错 VAE 不会报错，
    #: 只会出一片偏色 ✓✗）。σ 是批形态且∈[0,1] ✓（``×1000`` 在骨架内部 ✓）。
    "seedvr2": (5, "batch", "seedvr2"),
    #: ⭐ **Music3（音频）**：**三维** ✓ ``[B,128,T]`` ✓ —— 本仓唯一一个不是图的族 ✓。
    #: ``minimax_music3`` 口径 = 128ch / **时间 ÷512** / **scale 恒 1.0** ✓（**没有** spatial
    #: 下采样 ✓）；σ 是批形态且∈[0,1] ✓，但 ⚠️ **方向是反的** ✓✗ —— 内部要 ``t = 1 − σ`` ✓
    #: 且输出取负 ✓（两条都在 ``dit_music3`` 自己身上 ✓，跑通了才能改写这里 ✓）。
    "music3": (3, "batch", "minimax_music3"),
}


def case_contract() -> None:
    got = {key: (spec.contract.latent_dims, spec.contract.sigma_form, spec.contract.latent_format)
           for key, spec in bd.BUILDERS.items()}
    check("⑦ 前向契约与冻结快照一致 ✓（改它就是改口径 ⇒ 必须手改这张表 ✓）",
          got == CONTRACTS, {"多/少": sorted(set(got) ^ set(CONTRACTS)),
                             "不同": {k: (got.get(k), CONTRACTS.get(k))
                                      for k in got if got.get(k) != CONTRACTS.get(k)}})

    check("⑦ `IMAGE_ONLY_FAMILIES` 是**派生**的 ✓（不再手写第二份 ⇒ 加族只改表 ✓）",
          tb.IMAGE_ONLY_FAMILIES == bd.image_only_keys()
          and tb.IMAGE_ONLY_FAMILIES == tuple(k for k, c in CONTRACTS.items() if c[0] == 4),
          {"IMAGE_ONLY_FAMILIES": tb.IMAGE_ONLY_FAMILIES, "derived": bd.image_only_keys()})

    check("⑦ 认不出的族回退到通用口径 ✓（老路 `dit.build_dit` 5 维 + 标量 σ ✓）",
          bd.contract_of(None) == bd.contract_of("没有这个族")
          and bd.contract_of(None).latent_dims == 5
          and bd.sigma_form_of("没有这个族") == "scalar",
          bd.contract_of(None))

    #: ⚠️ 拼错的枚举必须**当场拒** ✗✗：写成 `"Batch"` ⇒ `sigma_form == "batch"` 为假 ⇒
    #:    `denoise` 走 else 分支 ⇒ 拿标量喂张量口 ✓✗（不报错、只是 σ 形状没生效 ✓）。
    rejected: list[str] = []
    #: ⚠️ 2026-11-02 起 ``3`` 是**合法**的（音频 ✓ Music3 ✓）⇒ 反向验证改用 ``6`` ✓✗
    #: （继续拿 3 当反例 ⇒ 这条断言从"挡错值"变成"挡对的族" ✓✗）。
    for kwargs, why in (({"latent_dims": 6, "sigma_form": "batch", "latent_format": "x"}, "latent_dims"),
                        ({"latent_dims": 4, "sigma_form": "Batch", "latent_format": "x"}, "sigma_form"),
                        ({"latent_dims": 4, "sigma_form": "batch", "latent_format": ""}, "latent_format")):
        try:
            bd.ForwardContract(**kwargs)                 # type: ignore[arg-type]
        except ValueError:
            rejected.append(why)
    check("⑦ 契约三个字段写错都当场 ValueError ✓（`\"Batch\"` 这种大小写最容易静默 ✓✗）",
          sorted(rejected) == ["latent_dims", "latent_format", "sigma_form"], rejected)

    #: `contract` 是**必填**（无默认值 ✓）⇒ 新族不写就 import 期 TypeError，而不是继承 5 维 ✓。
    required = True
    try:
        bd.FamilyBuilder(key="x", label="X", module="dit_generic")  # type: ignore[call-arg]
        required = False
    except TypeError:
        pass
    check("⑦ `contract` 无默认值 ⇒ 新族**必须**声明 ✓（默认值会让它悄悄继承 5 维视频口径 ✓✗）",
          required)

    src = (ENGINE / "torch_backend.py").read_text(encoding="utf-8")
    #: 四处消费点：`latent_shape` / `_latent_mode`（潜变量维数）✓、`denoise`（σ 形态）✓、
    #: `decode` 门口的**口径对账**（`flux` / `seedvr2` 两族各自从契约取规范名 ✓）。
    #: ⚠️ 这里数**出现次数**是为了防「只读一处、另一处又手抄 5 维」✓✗ —— 加新消费点时
    #: 必须同时把这行数改掉 ✓，否则它会立刻红（本轮 SEEDVR2 入 `decode` 对账 3 ⇒ 4 ✓）。
    _dim_reads = src.count('dit_build.contract_of(getattr(self, "_ditFamily", None)).latent_dims == 4')
    #: ⭐ 音频那条（``== 3`` ✓ 2026-11-02 接 Music3 ✓）**只许出现在两个地方** ✓✗：
    #: `latent_shape`（给出 ``(1,128,T)`` ✓ 2026-11-06）与 `_latent_mode`（判"真形状 vs 一维占位"✗✓）；
    #: 再多一处 ⇒ 说明有人开始按"是不是音频"在 `torch_backend` 里另开分派点 ✓（每条线都长一个 ✓✗）。
    _audio_reads = src.count('dit_build.contract_of(getattr(self, "_ditFamily", None)).latent_dims == 3')
    _sigma_reads = src.count("dit_build.sigma_form_of(")
    _format_reads = src.count('_vae_format = dit_build.contract_of("')
    #: ⚠️ 2026-11-06 加 Music3 后：`contract_of` 5 ⇒ 7（音频两处）、口径读 2 ⇒ 3（`flux` / `seedvr2` /
    #: `music3` —— 音频 VAE 也要"挂错族当场拒" ✓）、`_vae_format` 3 处 ✓。
    check("⑦ `torch_backend` 七处消费点都读契约 ✓（`latent_shape`×2（图/音频）/ `_latent_mode` / "
          "`denoise` / `decode` 对账×3）",
          src.count("dit_build.contract_of(") == 7 and _sigma_reads == 1
          and _dim_reads == 2 and _audio_reads == 2 and _format_reads == 3,
          {"contract_of": src.count("dit_build.contract_of("),
           "sigma_form_of": _sigma_reads,
           "dim_reads": _dim_reads, "audio_reads": _audio_reads, "format_reads": _format_reads})


# ══════════════════════════════════════════════════════════════════════════
# ⑦′ 潜空间口径**真的被用上**（不再是「写进报告就算数」的摆设）
# ══════════════════════════════════════════════════════════════════════════
def case_latent_format() -> None:
    registered = set(lf.LATENT_FORMATS)
    unknown = {k: spec.contract.latent_format for k, spec in bd.BUILDERS.items()
               if spec.contract.latent_format not in registered}
    check("⑦′ 每族的 `latent_format` 都是**表里有的规范名** ✓（自造名 `flux8`/`qwen21` 写进来也没人拦 ✗）",
          not unknown, unknown)

    #: ⚠️ 反向验证：自造名字**必须当场拒** ✗✗ —— 以前只判非空 ✓ ⇒ `flux8` 一路过 ✓、
    #: 报告里写着不存在的口径 ✓，等真去 `get_latent_format` 才炸（离写错处几千行 ✓）。
    try:
        bad = bd.ForwardContract(latent_dims=4, sigma_form="batch", latent_format="flux8")
    except ValueError as err:
        bad = err
    check("⑦′ 自造口径名当场 ValueError ✓（`flux8` 这种最容易蒙混 ✓✗）",
          isinstance(bad, ValueError), bad)

    #: ⚠️ 大小写/别名**归一化**（`Wan21` ⇒ `wan21` ✓）：不归一 ⇒ 两边名字不同却同族 ⇒ 假报警 ✓✗。
    check("⑦′ 别名/大小写归一化成规范名 ✓（`Wan21` ⇒ `wan21` ✓）",
          bd.ForwardContract(latent_dims=4, sigma_form="batch", latent_format="Wan21").latent_format
          == "wan21")
    check("⑦′ `generic` 是**显式哨兵** ✓（老路口径未知 ⇒ 不假装知道 ✓，`latent_spec()` 给 None ✓）",
          bd.contract_of(None).latent_format == bd.GENERIC_LATENT_FORMAT
          and bd.contract_of(None).latent_spec() is None
          and bd.contract_of("krea2").latent_spec() is lf.LATENT_FORMATS["wan21"])

    #: 对账三档：同族放行 ✓ / 异族当场报 ✓ / 一边未知不假装 ✓。
    check("⑦′ 同族 ⇒ 无问题 ✓（`krea2` ↔ `wan21`、`qwen_image` ↔ `qwen_image` ✓）",
          bd.latent_format_problem("krea2", "wan21") is None
          and bd.latent_format_problem("qwen_image", "qwen_image") is None
          and bd.latent_format_problem("qwen_image21", "qwen_image21") is None)
    #: ⚠️ 这就是**最贵的那类错**：16ch 的 wan21 与 16ch 的 qwen_image **形状全同** ✓ ⇒
    #: 挂错族的 VAE ⇒ 图整体偏色/发灰 ✓、形状全对 ✓、一个错都不报 ✓✗。
    clash = bd.latent_format_problem("qwen_image", "wan21")
    check("⑦′ 异族 ⇒ 报出来 ✓（`qwen_image` 挂 `wan21` 的 VAE = 静默偏色 ✓✗）",
          isinstance(clash, str) and "wan21" in clash and "qwen_image" in clash, clash)
    check("⑦′ 一边未知 ⇒ **不拦也不假装通过** ✓（老路族/认不出的名字 ✓）",
          bd.latent_format_problem(None, "wan21") is None
          and bd.latent_format_problem("krea2", None) is None
          and bd.latent_format_problem("krea2", "没有这个口径") is None)

    #: 装配线**从契约读**反缩放族名 ✓✗（手写字面量 ⇒ 契约改了它不跟 ⇒ 同上那个静默偏色 ✓）。
    #: ⚠️ 用 **AST** 判、不用字符串搜 ✗✗：注释里正写着 `vae_family="qwen_image"` 这句解释 ✓
    #: （讲"为什么不能用 wan21" ✓），字符串搜会把**说明**当成**缺陷** ✓✗。
    rt_tree = ast.parse((ENGINE / "runtime.py").read_text(encoding="utf-8"))
    from_contract: set[str] = set()
    handwritten: list[str] = []
    for node in ast.walk(rt_tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "vae_family":
                continue
            #: 形如 `dit_build.contract_of("krea2").latent_format` ✓ —— 取到 `.latent_format`
            #: 这一层 `Attribute` 之后，再往里剥到那颗 `Call` ✓（否则永远判成"手写" ✓✗）。
            value: ast.expr = kw.value
            if isinstance(value, ast.Attribute):
                value = value.value
            ok = (isinstance(value, ast.Call)
                  and getattr(value.func, "attr", "") == "contract_of"
                  and getattr(getattr(value.func, "value", None), "id", "") == "dit_build")
            if ok:
                from_contract |= {a.value for a in value.args if isinstance(a, ast.Constant)}
            else:
                handwritten.append(ast.unparse(kw.value))
    #: H3 那条（``vae_family="h3"``）**豁免** ✓：它不走 DiT 装配表（`h3_form` 自己的 VAE ✓）。
    #: ⚠️ 2026-11-02 补：Wan 2.2 那条原本也是**手写** ``"wan21"`` ✓（值对、但那是又一份拷贝 ✗）
    #: ⇒ 已改成从 `contract_of("wan")` 读 ✓。所以现在「手写」栏**只许剩 h3 一句** ✓✗。
    check("⑦′ 四条 DiT 装配线的 `vae_family` 都从契约读 ✓（手写字面量 ✗，H3 豁免）",
          from_contract == {"qwen_image21", "krea2", "qwen_image", "wan"}
          and all(lit in ("'h3'", '"h3"') for lit in handwritten),
          {"契约读": sorted(from_contract), "手写": handwritten})

    #: ⚠️ H3 **不在** `BUILDERS` ✓ ⇒ `contract_of("h3").latent_format` 是 `generic` ✓✗ ⇒
    #: 光靠契约问不出它的口径 ✓ ⇒ 解码对账会**整条跳过** ✓（"没检查"冒充"检查通过" 的又一处 ✓）。
    check("⑦′ 非装配族（H3）的口径**能问出来** ✓（`latent_format_of` ≠ `contract_of().latent_format` ✓）",
          bd.latent_format_of("h3") == "h3_video"
          and bd.contract_of("h3").latent_format == bd.GENERIC_LATENT_FORMAT
          and bd.latent_format_of("krea2") == bd.contract_of("krea2").latent_format,
          {"h3": bd.latent_format_of("h3"), "krea2": bd.latent_format_of("krea2")})
    check("⑦′ H3 与 16ch 族互换 VAE **互相拦得住** ✓（24ch 虽会先撞形状，但口径这一问也该自己答 ✓）",
          bd.latent_format_problem("h3", "h3_video") is None
          and isinstance(bd.latent_format_problem("h3", "wan21"), str)
          and isinstance(bd.latent_format_problem("krea2", "h3_video"), str))

    #: ⚠️ **最贵的漏检**：`process="meanstd"` 的族在解码时要 **16 组 mean/std** ✓ ——
    #: 数值登记在 :func:`vae_wan.latents_stats` ✓（另一张表 ✓）。契约说得出名字、那边没登记
    #: ⇒ 装完能用、**解到一半才炸** ✓✗（报错指着 VAE 的 shape ✓，像"文件坏了" ✗）。
    from app.services.engine import vae_wan as vw  # noqa: E402  （真 torch，本组只在末尾用 ✓）
    missing_stats: list[str] = []
    wrong_channels: list[str] = []
    for key, family in sorted(bd.BUILDERS.items()):
        spec = family.contract.latent_spec()
        if spec is None or str(spec.process) != "meanstd":
            continue
        try:
            mean, std = vw.latents_stats(bd.latent_format_of(key))
        except Exception as err:                       # noqa: BLE001 —— 缺登记就是这里炸 ✓
            missing_stats.append(f"{key}（{type(err).__name__}: {err}）")
            continue
        #: ⚠️ `latents_stats` 返的是**两个 float 元组** ✓（不是张量 ✗）⇒ 数 `len()` ✓，别 `numel()` ✓。
        if len(mean) != int(spec.latent_channels) or len(std) != int(spec.latent_channels):
            wrong_channels.append(f"{key}: mean={len(mean)} std={len(std)} 应为 {spec.latent_channels}")
    check("⑦′ `meanstd` 口径的族在 `vae_wan.latents_stats` 里**都有数** ✓（少一格 = 解码时才炸 ✓✗）",
          not missing_stats, missing_stats)
    check("⑦′ 登的 mean/std **组数 = 声明的通道数** ✓（16ch 只给 4 组 ⇒ 广播错 ✓✗）",
          not wrong_channels, wrong_channels)

    #: 解码**入口**要真调它 ✓✗：只写个函数不接 ⇒ 判据等于不存在 ✓。
    check("⑦′ `decode` 解码前调对账 ✓（不接 ⇒ 判据等于不存在 ✓✗）",
          src_backend().count("dit_build.latent_format_problem(") == 1,
          src_backend().count("dit_build.latent_format_problem("))
    #: ⚠️ 单点调用还不够 ✓✗：三种已认得的 VAE（flux / wan21 / h3）都得**各自报出口径** ✓，
    #: 少一条 ⇒ 那条解码路径就是"没检查" ✓（H3 曾整条漏掉 ✓）。
    kinds = [lit for lit in ('self._vaeKind == "flux-real"', 'self._vaeKind == "wan21-real"',
                             'self._vaeKind == "h3-real"') if lit in src_backend()]
    check("⑦′ 三种 VAE 都报口径给对账 ✓（漏一条 = 那条路径没检查 ✗）", len(kinds) == 3, kinds)

    #: 报告面 ✓：口径现在是判据 ✓，`readout` 里必须看得见"解码会发生什么" ✓✗。
    line = bd.readout("krea2")
    check("⑦′ `readout` 写出前向契约（潜变量维数 · σ 形态 · 口径换算方式 ✓）",
          "前向：" in line and "wan21" in line and "meanstd" in line, line)
    flux_line = bd.readout("zimage")
    check("⑦′ `affine` 口径写成 shift/scale ✓（只写族名读不出后果 ✓）",
          "affine" in flux_line and "0.3611" in flux_line, flux_line)

    #: ⚠️ 未知口径**必须在报告里说"别当已核"** ✓✗：写个 `口径 `generic`` 就完事 ⇒ 读的人
    #: 以为核过了 ✓。
    unknown = bd._describe_latent_format(bd._GENERIC_CONTRACT)
    check("⑦′ 未知口径 ⇒ 报告里明说「不参与对账」✓（别让 `generic` 冒充 🟢）",
          "不参与对账" in unknown, unknown)


def case_derived() -> None:
    """⭐ 派生格 ✓：`config` 只收 dataclass 字段 ✗ —— `@property` 不在 `vars()` 里 ✓✗。

    2026-11-01 真权重首跑撞的正是这一格 ✓：Edit 2511 的 `ref_timestep_zero` 是 property ✓ ⇒
    `config.get("ref_timestep_zero")` 永远 None ✓ ⇒ 装配线当 t2i 装（不挂 vision/vae）✓、
    `encode_text` 整段跳过参考图 ✓、直到 `denoise` 才报「ref_latents 不在」✓ ——
    **报错落点与病因差了三层** ✓✗。所以这里守三条：
    ① 声明的派生格必须真在 shape 上（拼错当场 `BuildError` ✓ 不许静默写 None ✗）；
    ② `qwen_image` 的 config **真带**这一格 ✓；
    ③ 它**确实**是 property（不在 `vars()` 里）✓ —— 否则这条判据就白守了 ✓✗。
    """
    spec = bd.BUILDERS["qwen_image"]
    check("⑧ `qwen_image` 声明了派生格 `ref_timestep_zero` ✓（不声明 ⇒ 装配线永远判 t2i ✗）",
          "ref_timestep_zero" in spec.derived, spec.derived)

    #: 最小可推的双流 MMDiT 形状 ✓（features 256 / head_dim 128（= 默认三轴 16+56+56 ✓）/
    #: 16ch / 1 层 ✓）—— 只造**元信息** ✓，`build()` 建的是骨架 ✓，不装字节 ✓。
    base = {
        "img_in.weight": (256, 64), "proj_out.weight": (64, 256),
        "transformer_blocks.0.attn.norm_q.weight": (128,),
        "txt_norm.weight": (64,), "txt_in.weight": (256, 64),
        "transformer_blocks.0.img_mlp.net.0.proj.weight": (512, 256),
        "transformer_blocks.0.img_mlp.net.2.weight": (256, 512),
        "transformer_blocks.0.img_mod.1.weight": (1536, 256),
        "time_text_embed.timestep_embedder.linear_1.weight": (256, 256),
        "time_text_embed.timestep_embedder.linear_2.weight": (256, 256),
    }
    built = bd.build("qwen_image", _info(base), {})
    t2i_config = built[1] if built else {}
    check("⑧ t2i 档（无哨兵键）⇒ config 里 `ref_timestep_zero` 是 **False** ✓（不是 None ✗）",
          t2i_config.get("ref_timestep_zero") is False,
          {k: v for k, v in t2i_config.items() if "ref" in str(k)})

    #: ⚠️ 哨兵键**存在性**才是判据 ✓ —— 载荷是 0 元素 ✓（int8_convrot 那份实测 `shape=[0]` ✓）。
    edit = dict(base, **{"__index_timestep_zero__": (0,)})
    built_edit = bd.build("qwen_image", _info(edit), {})
    edit_config = built_edit[1] if built_edit else {}
    check("⑧ Edit 档（带哨兵键 ✓ 且 `shape=[0]` 也认 ✓）⇒ **True** ✓",
          edit_config.get("ref_timestep_zero") is True,
          {k: v for k, v in edit_config.items() if "ref" in str(k)})

    #: 派生格拼错 ⇒ 当场报 ✓（不许把 None 写进 config ✗：上层读不出「没这条通路」与「读不到」的区别 ✗）
    probe = bd.FamilyBuilder(
        key="_derived_probe", label="派生探针", module="dit_generic",
        contract=bd.ForwardContract(latent_dims=4, sigma_form="batch", latent_format="qwen_image"),
        build_fn="Krea2DiT", build_args=("shape",), meta=True,
        shape_args=("patch",), shortcuts=(("in_channels", "channels"),),
        derived=("没有这一格",))
    bd.BUILDERS["_derived_probe"] = probe
    err: Exception | None = None
    try:
        bd.build("_derived_probe", _info(krea2_full_pairs()), {})
    except bd.BuildError as exc:                        # noqa: PERF203 —— 就是要抓它
        err = exc
    finally:
        bd.BUILDERS.pop("_derived_probe", None)
    check("⑧ 派生格拼错 ⇒ 当场 `BuildError` 且点名那一格 ✓（不许写 `None` 混过去 ✗）",
          isinstance(err, bd.BuildError) and "没有这一格" in str(err), type(err).__name__)


def main() -> int:
    case_table_complete()
    case_coverage()
    case_no_fork()
    case_table_dispatch()
    case_build_error()
    case_krea2_prefix()
    case_int8_adapter()
    case_contract()
    case_latent_format()
    case_derived()

    failures = 0
    for name, ok, detail in _RESULTS:
        if ok:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}\n      {detail!r}")
    print(f"\nSUMMARY: {len(_RESULTS) - failures}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
