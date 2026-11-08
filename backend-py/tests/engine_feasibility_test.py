"""任务级可行性预检自检（G2，2026-11-07）✓✗。

这份夹具盯的是「**预检与真装载不许各持一份账**」✓ —— 也就是说：

1. 逐件需求必须走 :func:`torch_backend.stage_part_need` ✓（装配线的同一个出口 ✓）；
2. 表值必须**引用**那几本 ``*_PART_NEED_GIB`` ✓（``is`` 相等 ✓，不是抄一份数字 ✗）；
3. 「峰值」的口径必须与装配线**真怎么装**一致 ✓（整链同驻求和 / 分时换装取最大 ✓）——
   写错这个，``qwen_image21`` 会被判「22.49 的卡装不下」✓✗（而它每天都在跑 ✓），
   ``music3`` 会被判「省 8 GiB」✓✗（而它三件真同驻 ✓）。
4. ``verdict`` 四档都要**真能出现** ✓（构造四种读数 ✓，不是只在纸面上有四个词 ✓）；
5. 激活如实 ``basis="unknown"`` ✓、``perfect`` 必须降档 ✓（方案 A ✓）。

⚠️ 全部判据走**不存在的路径** ＋ 注入读数 ✓：不依赖盘上装了哪些模型 ✓、不需要 torch ✓，
换一台机器结论不变 ✓（数值全来自表 ✓）。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.engine import device_policy as dp  # noqa: E402
from app.services.engine import feasibility as feas  # noqa: E402
from app.services.engine import inventory as inv  # noqa: E402
from app.services.engine import torch_backend as tb  # noqa: E402

_FAILS: list[str] = []
_TOTAL = 0
GIB = dp.GIB

ENGINE = ROOT / "app" / "services" / "engine"


def check(label: str, ok: bool, detail: object = "") -> None:
    global _TOTAL
    _TOTAL += 1
    if ok:
        print(f"PASS  {label}")
        return
    _FAILS.append(label)
    print(f"FAIL  {label}  ->  {detail!r}")


#: 路径**故意不存在** ⇒ 走 ``part_need_gib`` 的「读不到头部 ⇒ 按表」那一条 ✓（这正是没装模型的场景 ✓）。
FAKE = {
    "stage": "fake",
    "dit_path": "no_such_dir/dit.safetensors",
    "text_weights": "no_such_dir/text.safetensors",
    "image_vae_weights": "no_such_dir/vae.safetensors",
    "video_vae_weights": "no_such_dir/vae.safetensors",
    "audio_vae_weights": "no_such_dir/vae.safetensors",
}

ALL_STAGES = ("h3", "zimage", "qwen_image21", "krea2", "seedvr2", "music3",
              "wan22", "qwen_image", "image")


def _card(free_gib: float, total_gib: float = 22.49) -> dp.Pressure:
    return dp.Pressure(totalBytes=int(total_gib * GIB), freeBytes=int(free_gib * GIB))


HOST_ROOMY = (90 * GIB, 96 * GIB)
HOST_TIGHT = (1 * GIB, 16 * GIB)


def case_tables() -> None:
    """① 登记表完备 ＋ **同一批对象**（不是第二本账）✓✗。"""
    missing_peak = [stage for stage in tb.STAGE_PART_SPECS if stage not in feas.STAGE_PEAK]
    check("①a 每个装配族都有峰值口径（STAGE_PEAK 不缺行）", not missing_peak, missing_peak)

    check("①b 峰值口径的件集 == 该族的件表（多一件/少一件都报）",
          all(set(feas.STAGE_PEAK[stage].parts) == set(tb.STAGE_PART_SPECS[stage])
              for stage in tb.STAGE_PART_SPECS),
          {stage: (feas.STAGE_PEAK[stage].parts, tuple(tb.STAGE_PART_SPECS[stage]))
           for stage in tb.STAGE_PART_SPECS
           if set(feas.STAGE_PEAK[stage].parts) != set(tb.STAGE_PART_SPECS[stage])})

    same_name = {
        "h3": tb.TorchBackend.H3_PART_NEED_GIB,
        "zimage": tb.TorchBackend.CHAIN_NEED_PARTS_GIB["zimage"],
        "qwen_image21": tb.TorchBackend.PART_NEED_GIB,
        "krea2": tb.TorchBackend.KREA2_PART_NEED_GIB,
        "seedvr2": tb.TorchBackend.SEEDVR2_PART_NEED_GIB,
        "music3": tb.TorchBackend.MUSIC3_PART_NEED_GIB,
        "wan22": tb.TorchBackend.WAN22_PART_NEED_GIB,
        "qwen_image": tb.TorchBackend.QWEN_IMAGE_PART_NEED_GIB,
    }
    bad = [stage for stage, table in same_name.items()
           if tb.TorchBackend.STAGE_NEED_TABLES.get(stage) is not table]
    check("①c 索引里的表就是那几本账本身（`is` 相等 ⇒ 不是抄的数字）", not bad, bad)
    check("①d `image`（SDXL）刻意不进表（单 checkpoint ⇒ 按文件字节）",
          "image" not in tb.TorchBackend.STAGE_NEED_TABLES,
          sorted(tb.TorchBackend.STAGE_NEED_TABLES))

    check("①e 查不到表必须**抛**、不编数（`nope` 档）",
          _raises(lambda: tb.stage_table_need("nope", "dit")), "no raise")
    check("①f 表里没有该件也要抛（krea2 没有 text2）",
          _raises(lambda: tb.stage_table_need("krea2", "text2")), "no raise")
    check("①g 能查到时返回表值本身（krea2 dit = 12.57）",
          abs(tb.stage_table_need("krea2", "dit") - 12.57) < 0.005,
          tb.stage_table_need("krea2", "dit"))


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def case_need() -> None:
    """② 逐档需求：数值来自表 ＋ 峰值口径与装配事实一致 ✓✗。"""
    need = feas.weight_need("krea2", assembly=FAKE)
    check("②a krea2 峰值 = 常驻 DiT 12.57（text/vae 换装不进峰值）",
          abs(need["gpuGiB"] - 12.57) < 0.02, need["gpuGiB"])
    check("②b krea2 三件之和 = 20.31（这个数**只有**在报「全件同驻」时才用）",
          abs(need["sumGiB"] - (12.57 + 7.5 + 0.24)) < 0.05, need["sumGiB"])
    check("②c krea2 口径标成 phased（分时换装）", need["mode"] == "phased", need["mode"])

    need = feas.weight_need("qwen_image21", assembly=FAKE)
    check("②d qwen_image21 峰值 = 文本塔 14.79（比 DiT 13.26 大）",
          abs(need["gpuGiB"] - 14.79) < 0.02, need["gpuGiB"])
    check("②e qwen_image21 三件求和 28.68 > 卡 22.49（所以**不能**用求和判它）",
          need["sumGiB"] > 22.49, need["sumGiB"])

    need = feas.weight_need("h3", assembly=FAKE)
    check("②f h3 峰值 = 文本塔 25.44（encode 那一拍最大）",
          abs(need["gpuGiB"] - 25.44) < 0.02, need["gpuGiB"])

    need = feas.weight_need("wan22", assembly=FAKE)
    check("②g wan22 峰值 = 单 DiT 26.62（双专家不并存、不许把 dit2 加上）",
          abs(need["gpuGiB"] - 26.62) < 0.02, need["gpuGiB"])

    need = feas.weight_need("zimage", assembly=FAKE)
    check("②h zimage 是**整链同驻** ⇒ 峰值 = 三件求和 19.28",
          abs(need["gpuGiB"] - 19.28) < 0.05, need["gpuGiB"])
    check("②i 且与整链账 CHAIN_NEED_GIB 对得上（两本账互相印证）",
          abs(need["gpuGiB"] - tb.TorchBackend.CHAIN_NEED_GIB["zimage"]) < 0.3,
          (need["gpuGiB"], tb.TorchBackend.CHAIN_NEED_GIB["zimage"]))

    need = feas.weight_need("music3", assembly=FAKE)
    check("②j music3 三件**同驻** ⇒ 峰值 13.35（实测 13.34）",
          abs(need["gpuGiB"] - 13.35) < 0.05, need["gpuGiB"])
    check("②k music3 峰值必须**大于**单件 DiT（写错 combine 会相等）",
          need["gpuGiB"] > need["parts"]["dit"]["gib"], need["gpuGiB"])

    need = feas.weight_need("seedvr2", assembly=FAKE)
    check("②l seedvr2 = DiT + VAE 两件同驻 7.27（这条线没有文本塔）",
          abs(need["gpuGiB"] - 7.27) < 0.05, need["gpuGiB"])

    need = feas.weight_need("image", assembly=FAKE)
    check("②m image（SDXL）按文件字节算；读不到就 0 且**写明原因**（不编数）",
          need["parts"]["dit"]["gib"] == 0.0
          and "不存在" in str(need["parts"]["dit"]["source"].get("why", "")), need["parts"])

    complete = dict(FAKE, dit_path="no_such_dir/sdxl.safetensors")
    need = feas.weight_need("image", assembly=complete)
    check("②n image 有路径就不是缺件（字节仍按表回退 0）",
          need["missing"] == [] and need["mode"] == "single", need["missing"])


def case_verdicts() -> None:
    """③ 四档**真能出现** ＋ 激活未知时 perfect 降档 ✓✗。"""
    roomy = feas.check("krea2", pressure=_card(22.0), host=HOST_ROOMY, assembly=FAKE)
    check("③a 空卡跑小链 ⇒ raw 是 perfect", roomy["verdictRaw"] == "perfect", roomy["verdictRaw"])
    check("③b 但激活未知 ⇒ 报 good 且标 downgraded",
          roomy["verdict"] == "good" and roomy["downgraded"] is True,
          (roomy["verdict"], roomy["downgraded"]))
    check("③c 降档要给出理由（不是静默改判）", bool(roomy["downgradeWhy"]), roomy["downgradeWhy"])

    good = feas.check("krea2", pressure=_card(16.0), host=HOST_ROOMY, assembly=FAKE)
    check("③d 余量 16/12.57=1.27 ⇒ good（不降档）",
          good["verdict"] == "good" and good["downgraded"] is False,
          (good["verdict"], good["downgraded"], good["verdictWhy"]))

    tight = feas.check("krea2", pressure=_card(13.0), host=HOST_ROOMY, assembly=FAKE)
    check("③e 余量不够 1 GiB ⇒ 走联合驻留 ⇒ marginal（degraded）",
          tight["verdict"] == "marginal" and tight["decision"]["residency"] == "split",
          (tight["verdict"], tight["decision"]))

    cpu = feas.check("krea2", pressure=_card(0.2), host=HOST_ROOMY, assembly=FAKE)
    check("③f 卡上空但内存够 ⇒ 整份退 CPU ⇒ marginal（不是失败）",
          cpu["verdict"] == "marginal" and cpu["decision"]["device"] == "cpu",
          (cpu["verdict"], cpu["decision"]["device"]))

    dead = feas.check("krea2", pressure=_card(0.2), host=HOST_TIGHT, assembly=FAKE)
    check("③g 卡与内存都放不下 ⇒ too_tight",
          dead["verdict"] == "too_tight", (dead["verdict"], dead["verdictWhy"]))

    forced = feas.check("krea2", pressure=_card(22.0), host=HOST_TIGHT, assembly=FAKE,
                        policy="cpu")
    check("③h `policy=cpu` 强制整份进内存 ⇒ 内存不够时如实报 too_tight",
          forced["verdict"] == "too_tight" and forced["decision"]["device"] == "cpu",
          (forced["verdict"], forced["decision"]))

    absent = feas.check("krea2", pressure=_card(22.0), host=HOST_ROOMY,
                        assembly={"dit_path": None, "text_weights": None,
                                  "image_vae_weights": None})
    check("③i 缺件 ⇒ too_tight（不是「显存不够」）且 missing 点名",
          absent["verdict"] == "too_tight" and "dit" in absent["need"]["missing"],
          (absent["verdict"], absent["need"]["missing"]))

    check("③j 不认识的策略 ⇒ 抛（不静默回落 auto）",
          _raises(lambda: feas.check("krea2", pressure=_card(22.0), host=HOST_ROOMY,
                                     assembly=FAKE, policy="turbo")), "no raise")


def case_activation() -> None:
    """④ 激活如实 unknown ✓（方案 A：不编数）✓✗。"""
    report = feas.check("krea2", pressure=_card(22.0), host=HOST_ROOMY, assembly=FAKE)
    check("④a `activation.basis` 恒为 unknown", report["activation"]["basis"] == "unknown",
          report["activation"])
    check("④b `activation.gib` 必须是 None（不许给个数凑完整）",
          report["activation"]["gib"] is None, report["activation"])
    check("④c 要写清「权重之外还要余量」这句话",
          "权重" in report["activation"]["whatItMeans"], report["activation"]["whatItMeans"])
    check("④d 回执里 `need.gpuGiB` 与 `activation` 分开两栏（别合成一个模糊数）",
          isinstance(report["need"]["gpuGiB"], float) and report["activation"]["gib"] is None,
          report["need"]["gpuGiB"])


def case_capacity() -> None:
    """⑤ `capacityGiB`（"如果我买 16G 卡"）✓：丢开真读数、按空卡算、并如实标 assumed ✓✗。"""
    report = feas.check("krea2", capacity_gib=16.0, host=HOST_ROOMY, assembly=FAKE)
    check("⑤a 假装 16 GiB ⇒ have.gpuTotalGiB = 16", report["have"]["gpuTotalGiB"] == 16.0,
          report["have"])
    check("⑤b 假装时按**空卡**算 ⇒ 可用 = 16", report["have"]["gpuFreeGiB"] == 16.0,
          report["have"])
    check("⑤c 必须标 assumed（假数据不许冒充真读数）", report["have"]["assumed"] is True,
          report["have"])

    report = feas.check("krea2", capacity_gib=8.0, host=HOST_ROOMY, assembly=FAKE)
    check("⑤d 8 GiB 卡跑 krea2 ⇒ 只能联合 ⇒ marginal",
          report["verdict"] == "marginal", (report["verdict"], report["verdictWhy"]))

    report = feas.check("krea2", capacity_gib=8.0, host=HOST_TIGHT, assembly=FAKE)
    check("⑤e 8 GiB 卡 + 32 GB 都放不下那条链 ⇒ too_tight（验收档的真实结论）",
          report["verdict"] == "too_tight", (report["verdict"], report["verdictWhy"]))


def case_ast() -> None:
    """⑥ 峰值口径必须和装配线**真怎么装**对得上（源码级对账）✓✗。"""
    runtime = (ENGINE / "runtime.py").read_text(encoding="utf-8")
    pipeline = (ENGINE / "pipeline.py").read_text(encoding="utf-8")

    #: `decide_for(("dit", ...))` 的件元组字面量（装配期真判的那一组件）
    tuples: set[tuple[str, ...]] = set()
    for node in ast.walk(ast.parse(runtime)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name != "decide_for" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, (ast.Tuple, ast.List)):
            names = [elt.value for elt in first.elts if isinstance(elt, ast.Constant)]
            if names:
                tuples.add(tuple(names))

    check("⑥a 装配线里真存在 `decide_for(('dit','text','vae'))`（music3 的同驻口径）",
          ("dit", "text", "vae") in tuples, sorted(tuples))
    check("⑥b 也真存在 `decide_for(('dit','vae'))`（seedvr2 的配对口径）",
          ("dit", "vae") in tuples, sorted(tuples))
    check("⑥c 也真存在 `decide_for(('dit',))`（分时换装那几档）",
          ("dit",) in tuples, sorted(tuples))

    check("⑥d zimage 的整链同驻走 `_zimage_device_decision`（源码里得有这个调用）",
          "_zimage_device_decision(" in runtime, "")
    check("⑥e 分时换装真有阶段边界（`_backend_phase` 转调 `enter_phase`、含 encode/decode）",
          "_backend_phase(" in pipeline and "enter_phase" in pipeline
          and '"encode"' in pipeline and '"decode"' in pipeline, "")
    check("⑥f 装了整链账就不许再手写数字（`CHAIN_NEED_GIB` 是唯一出处）",
          "CHAIN_NEED_GIB" in runtime, "")

    checksum = [stage for stage, spec in feas.STAGE_PEAK.items()
                if spec.combine not in ("sum", "max")]
    check("⑥g combine 只有 sum / max 两种（写错第三个值会在别处静默走 else）",
          not checksum, checksum)


def case_facade_router() -> None:
    """⑦ 唯一入口 ＋ 端点真的挂上 ✓✗。"""
    from app.services import engine as engine_pkg

    check("⑦a 新模块已登记进分组表（结构守卫同口径）",
          "feasibility" in engine_pkg.MODULE_GROUPS["orchestration"],
          engine_pkg.MODULE_GROUPS["orchestration"])

    from app.services.engine import facade

    check("⑦b 门面导出 `feasibility`（CLI 与 HTTP 同一份）", "feasibility" in facade.__all__,
          facade.__all__)
    data = facade.feasibility("krea2")
    check("⑦c 门面回执带 schemaVersion / apiVersion",
          "schemaVersion" in data and "apiVersion" in data, sorted(data))
    check("⑦d 门面回执就是预检的键（不是另包一层）",
          "verdict" in data and "need" in data and "activation" in data, sorted(data))
    check("⑦e 未知档从门面走时要翻译成 EngineError（不是裸异常）",
          _raises(lambda: facade.feasibility("nope")), "no raise")

    from app.routers import engine as engine_router

    paths = {getattr(route, "path", "") for route in engine_router.router.routes}
    check("⑦f 路由上真有 `GET …/feasibility`（带 `/api/v1/engine` 前缀）",
          any(path.endswith("/feasibility") for path in paths), sorted(paths))
    check("⑦g 端点写的是 `facade.feasibility(`（E1.3 唯一入口，不直调模块）",
          "facade.feasibility(" in (ROOT / "app" / "routers" / "engine.py").read_text(encoding="utf-8"),
          "")
    check("⑦h 端点参数里有 `capacityGiB`（「如果我买 16G 卡」那条路）",
          "capacityGiB" in (ROOT / "app" / "routers" / "engine.py").read_text(encoding="utf-8"), "")


def case_shape() -> None:
    """⑧ 回执形状 ＋ 自检 ＋ 与两条邻居的分工不许混 ✓✗。"""
    shape = feas.describe_shape()
    report = feas.check("seedvr2", pressure=_card(22.0), host=HOST_ROOMY, assembly=FAKE)
    check("⑧a describe_shape 的顶层键都在回执里",
          all(key in report for key in shape["top"]),
          [key for key in shape["top"] if key not in report])
    check("⑧b need 的键都在回执里",
          all(key in report["need"] for key in shape["need"]),
          [key for key in shape["need"] if key not in report["need"]])
    check("⑧c 四档名单与 VERDICTS 一致",
          set(shape["verdicts"]) == set(feas.VERDICTS), (shape["verdicts"], feas.VERDICTS))

    #: ⚠️ 分工：`plan` 用**文件字节**、本模块用**驻留 GiB** ⇒ 两者对 2.1 必须给出不同的数 ✓✗
    need = feas.weight_need("qwen_image21", assembly=FAKE)
    check("⑧d 与 `/load-plan` 的差别是**口径**不是数值（峰值 14.79 对文件字节另一套）",
          need["gpuGiB"] != need["parts"]["dit"]["gib"], (need["gpuGiB"], need["parts"]["dit"]["gib"]))

    problems = feas.selftest()
    check("⑧e 模块自检全绿", not problems, problems)

    #: ⚠️ `STAGE_FILTERS` 里**混了类别键**（`image` / `audio` / `text` ＝ 按 category 挑整类 ✓，
    #: 不是装配族 ✗）⇒ 只对**带 tags 的族键**要求登记 ✓（漏一族才该红 ✓）。
    tagged = {stage for stage, spec in inv.STAGE_FILTERS.items() if spec.get("tags")}
    check("⑧f `STAGE_PEAK` 覆盖了 inventory 的全部**族键**（类别键不算 ✓）",
          all(stage in feas.STAGE_PEAK for stage in tagged),
          sorted(tagged - set(feas.STAGE_PEAK)))


def case_estimate() -> None:
    """⑨ `estimate` **只报有据的数** ✓✗（样本不足给 ``null`` ✓、绝不给 ``0`` ✗✗）。"""
    check("⑨a 没有样本 ⇒ basis=unknown 且 runSeconds 是 None（不是 0 秒）",
          feas.estimate(stage="krea2")["basis"] == "unknown"
          and feas.estimate(stage="krea2")["runSeconds"] is None,
          feas.estimate(stage="krea2"))

    short = feas.estimate(stage="krea2", samples=[120000.0, 90000.0])
    check("⑨b 样本数不足闸门 ⇒ 仍 unknown，且 note 里写明「几条 / 闸门几条」",
          short["basis"] == "unknown" and short["runSeconds"] is None
          and str(short["samples"]) in short["note"] and str(short["minSamples"]) in short["note"],
          short)

    enough = feas.estimate(stage="krea2", samples=[120000.0, 90000.0, 300000.0])
    rs = enough.get("runSeconds") or {}
    check("⑨c 达到闸门 ⇒ basis=measured 且秒数是**正**数（min ≤ p50 ≤ p95 ≤ max）",
          enough["basis"] == "measured" and set(rs) == {"p50", "p95", "min", "max", "mean"}
          and all(value > 0 for value in rs.values())
          and rs["min"] <= rs["p50"] <= rs["p95"] <= rs["max"],
          rs)

    check("⑨d `loadSeconds` **恒** None ＋ 带 loadWhy（有无样本都一样）",
          all(feas.estimate(stage="h3", samples=samples)["loadSeconds"] is None
              and bool(feas.estimate(stage="h3", samples=samples)["loadWhy"])
              for samples in (None, [1000.0, 2000.0, 3000.0])),
          feas.estimate(stage="h3")["loadWhy"])

    dirty = feas.estimate(stage="h3",
                          samples=[0.0, -5000.0, float("nan"), float("inf"), "bad", 60000.0])
    check("⑨e 0 / 负数 / nan / inf / 非数一律剔除（记 0ms 的跑次多半是没跑就失败）",
          dirty["samples"] == 1 and dirty["basis"] == "unknown", dirty)

    sampled = feas.estimate(stage="h3", samples=[10000.0, 20000.0, 30000.0], matched=40)
    check("⑨f matched 与 samples **分开** ＋ 抽样时明说「不是就这么多」",
          sampled["matched"] == 40 and sampled["samples"] == 3
          and "抽样" in sampled["note"], sampled)

    check("⑨g `unit` 明写「整任务」（别让它被按纯采样速度读）",
          "整任务" in feas.estimate(stage="h3")["unit"], feas.estimate(stage="h3")["unit"])

    #: ⚠️ 架构约束：引擎内核**不读业务库** ✓ —— 跑次台账由调用方取 ✓、本模块纯计算 ✓。
    source = (ENGINE / "feasibility.py").read_text(encoding="utf-8")
    check("⑨h 模块源码不 import 业务库（sqlite3 / app.models / app.db）",
          not any(token in source for token in ("sqlite3", "app.models", "app.db")),
          [token for token in ("sqlite3", "app.models", "app.db") if token in source])

    tree = ast.parse(source)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    target = next((node for node in funcs if node.name == "estimate"), None)
    check("⑨i `estimate` 在模块顶层且**只做计算**（没有任何 await / IO 调用）",
          target is not None and not any(isinstance(node, ast.Await) for node in ast.walk(target)),
          target.name if target else None)


def case_size() -> None:
    """⑩ `size` 这一栏必须**委托** :func:`canvas.plan_canvas` ✓✗（自拍步长表就会说反）。"""
    from app.services.engine import canvas as canvas_mod
    from app.services.engine import facade

    check("⑩a 不给 size ⇒ None（「没问」不是「不变」）", facade._size_note("h3", None) is None)

    mismatched: dict[str, object] = {}
    for stage in sorted(canvas_mod.STAGE_CANVAS):
        mine = facade._size_note(stage, "1001x1003")
        theirs = canvas_mod.plan_canvas(stage, width=1001, height=1003)
        if (mine["snapped"] != f"{int(theirs.width)}x{int(theirs.height)}"
                or mine["step"] != int(theirs.grid)):
            mismatched[stage] = (mine.get("snapped"), mine.get("step"), theirs)
    check("⑩b 每个接画布的族，`snapped`/`step` 都与 `canvas.plan_canvas` 逐字同值",
          not mismatched, mismatched)

    #: ⭐ 这条是实测栽过的坑 ✓✗：自拍「全族 16 ＋ 一律向上吸」会报 1008×1008，
    #: 而真源是**就近**（round）＋逐族步长 ⇒ h3（32）落到 992×992 ✓。
    near = facade._size_note("h3", "1000x1000")
    check("⑩c **就近**不是向上：h3 的 1000×1000 吸到 992×992（不是 1024/1008）",
          near["snapped"] == "992x992" and near["absorbed"] is True, near)

    off = [stage for stage in ("wan22", "seedvr2", "music3")
           if stage not in canvas_mod.STAGE_CANVAS]
    notes = {stage: facade._size_note(stage, "1000x1000") for stage in off}
    check("⑩d 不接画布覆盖的族明说「不参与」且 snapped=None（不编一个数出来）",
          all(entry["snapped"] is None and "不接画布覆盖" in entry["note"]
              for entry in notes.values()), notes)

    bads = {text: facade._size_note("h3", text) for text in ("abc", "1024", "0x100", "-8x64")}
    check("⑩e 看不懂的尺寸给 `problem`（不抛异常、也不静默当 1024×1024）",
          all("problem" in entry and entry["snapped"] is None for entry in bads.values()), bads)

    #: ⚠️ 权重判据与分辨率无关 ✓（激活峰值没有表 ✗）⇒ 尺寸**不参与** verdict ✓。
    plain = facade.feasibility("krea2")
    huge = facade.feasibility("krea2", size="4096x4096")
    check("⑩f 给了极端尺寸也不改 verdict（这一栏只报几何、不做判据）",
          plain["verdict"] == huge["verdict"]
          and plain["size"] is None and huge["size"] is not None,
          (plain["verdict"], huge["verdict"], huge["size"]))

    source = (ENGINE / "facade.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = {node.targets[0].id for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    check("⑩g facade 里**没有**自造的步长表（`_SIZE_STEPS` / `_size_rule` 都不许回来）",
          not ({"_SIZE_STEPS", "_size_rule"} & assigned),
          sorted({"_SIZE_STEPS", "_size_rule"} & assigned))
    check("⑩h 吸附走 `canvas.plan_canvas(`（唯一真源）",
          "plan_canvas(" in source, "")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("== 任务级可行性预检（G2）==")
    case_tables()
    case_need()
    case_verdicts()
    case_activation()
    case_capacity()
    case_ast()
    case_facade_router()
    case_shape()
    case_estimate()
    case_size()
    passed = _TOTAL - len(_FAILS)
    print(f"\nSUMMARY: {passed}/{_TOTAL} passed")
    return 0 if passed == _TOTAL and _TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(main())
