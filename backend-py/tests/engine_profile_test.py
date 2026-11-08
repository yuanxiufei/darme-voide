"""机器档（``profile``）自检（G1，2026-11-07）✓✗。

这份夹具盯的是「**一句话 → 一组旋钮**只许有一个展开点」✓ —— 也就是：

1. 五档取值域与别名**只有一张表** ✓，不认的**当场抛** ✗（静默回落 ``auto`` 会让「我选了 L1」变成假话 ✓✗）；
2. ``applied`` 只放**真会改**的项 ✓✗ —— ``auto``/``gpu`` 两档必须是 ``{}``
   （把「等于默认值」也报成「已改」⇒ 用户以为换了档就有区别 ✓✗，还会让装载键多一格 ✓✗）；
3. 档位表**只引用默认值** ✓：本文件里不许出现 ``host_cache.HOST_USE_RATIO`` 那个数字的字面量 ✗
   （抄一遍就是第二本账 ✓✗）；
4. 档位假设**绝不当实测** ✓✗：``budgets.assumed`` 与 ``budgets.gpuGiB`` 必须分得开 ✓；
5. 接线：``profile`` 必须进**装载键** ✓（换了档却复用旧布局，且一声不响 ✓✗）、
   必须一路传到**每一条**装配线的 ``TorchBackend(...)`` ✓、``/load`` 白名单与 ``/engine/runtime`` 都要认 ✓；
6. 反向：装配线不许**绕过**档位直接调 ``device_policy`` / ``host_cache`` 的旋钮
   （G1 的意义就是「只有一处展开」✓✗）。

⚠️ 全部判据**不读盘、不装载、不需要 torch** ✓（读数由夹具注入 ✓），换一台机器结论不变 ✓。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.engine import device_policy as dp  # noqa: E402
from app.services.engine import host_cache as hc  # noqa: E402
from app.services.engine import profile as prof_mod  # noqa: E402

_FAILS: list[str] = []
_TOTAL = 0
GIB = 1 << 30

ENGINE = ROOT / "app" / "services" / "engine"
ROUTERS = ROOT / "app" / "routers"


def check(label: str, ok: bool, detail: object = "") -> None:
    global _TOTAL
    _TOTAL += 1
    if ok:
        print(f"PASS  {label}")
        return
    _FAILS.append(label)
    print(f"FAIL  {label}  ->  {detail!r}")


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _card(free_gib: float, total_gib: float = 22.49) -> dp.Pressure:
    return dp.Pressure(totalBytes=int(total_gib * GIB), freeBytes=int(free_gib * GIB))


# --------------------------------------------------------------- ① 表与纪律


def case_table() -> None:
    """① 五档取值域 ＋ 「只引用不抄数」✓✗。"""
    check("①a 五档齐全（`auto`/`gpu`/`l1`/`l2`/`cpu`）", set(prof_mod.PROFILES) == {"auto", "gpu", "l1", "l2", "cpu"},
          sorted(prof_mod.PROFILES))
    check("①b 每档 `Profile.key` 与字典键一致（拼错 ⇒ 回执与键对不上）",
          all(row.key == key for key, row in prof_mod.PROFILES.items()),
          {key: row.key for key, row in prof_mod.PROFILES.items() if row.key != key})

    #: ⚠️ 假设值只许出现在 ``l1``/``l2``/``cpu``/``gpu`` 这四行 ✓；``auto`` 一律 ``None`` ✓✗
    check("①c `auto` 档**不预设任何东西**（三格全 None ＋ 两个假设全 None）",
          prof_mod.PROFILES["auto"].policy is None
          and prof_mod.PROFILES["auto"].host_ratio is None
          and prof_mod.PROFILES["auto"].temporal_chunk is None
          and prof_mod.PROFILES["auto"].gpu_gib is None
          and prof_mod.PROFILES["auto"].host_gib is None,
          prof_mod.PROFILES["auto"])

    #: ⚠️ §5.2 的两行数字是**规格**（8+32 / 6+16）✓，改了要连文档一起改 ✓ ⇒ 这里钉死 ✓
    check("①d `l1` 假设 8 GiB 卡 / 32 GB 内存、`l2` 假设 6 GiB / 16 GB（§5.2 规格值）",
          (prof_mod.PROFILES["l1"].gpu_gib, prof_mod.PROFILES["l1"].host_gib) == (8.0, 32.0)
          and (prof_mod.PROFILES["l2"].gpu_gib, prof_mod.PROFILES["l2"].host_gib) == (6.0, 16.0),
          [(key, prof_mod.PROFILES[key].gpu_gib, prof_mod.PROFILES[key].host_gib)
           for key in ("l1", "l2")])

    #: ⚠️ 低配档**不许**钉成 ``cpu`` ✗✗：8 GiB 卡上「联合驻留」正是要救的那条路 ✓
    check("①e `l1`/`l2` 的 policy 仍是 `auto`（钉成 `cpu` ＝ 放弃用卡）",
          prof_mod.PROFILES["l1"].policy == "auto" and prof_mod.PROFILES["l2"].policy == "auto",
          [prof_mod.PROFILES[key].policy for key in ("l1", "l2")])
    check("①f `cpu` 档钉死 `cpu`（机器根本没卡 ⇒ 别让 `auto` 白等一次 OOM）",
          prof_mod.PROFILES["cpu"].policy == "cpu", prof_mod.PROFILES["cpu"].policy)
    check("①g 档位表里的 policy 都在 `device_policy.POLICIES` 内（写错一个字母 ⇒ 装载期才炸）",
          all(row.policy is None or row.policy in dp.POLICIES for row in prof_mod.PROFILES.values()),
          [row.policy for row in prof_mod.PROFILES.values()])

    #: ⚠️ 「只引用不抄数」的**机器判据** ✓✗：``HOST_USE_RATIO`` 的数值**不许**以字面量出现 ✓
    #: （抄一遍 ⇒ 那边改了这边不跟，症状＝「改了没生效」✓✗）。
    tree = ast.parse(_source(ENGINE / "profile.py"))
    default_ratio = hc.HOST_USE_RATIO
    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, float)
                and abs(node.value - default_ratio) < 1e-12]
    check(f"①h 文件里**不出现**默认配额的字面量 {default_ratio}（引用 `HOST_USE_RATIO`）",
          not literals, literals)
    check("①i `gpu` 档的配额是**引用**默认值（`is`/`==` 都成立，且不等于低配档）",
          prof_mod.PROFILES["gpu"].host_ratio == hc.HOST_USE_RATIO
          and prof_mod.PROFILES["gpu"].host_ratio != prof_mod.PROFILES["l1"].host_ratio,
          (prof_mod.PROFILES["gpu"].host_ratio, hc.HOST_USE_RATIO))


def case_normalize() -> None:
    """② 归一：别名表只有一张 ＋ 不认的**抛** ✗✗。"""
    pairs = [(None, "auto"), ("", "auto"), ("AUTO", "auto"), ("l1", "l1"), ("L2", "l2"),
             ("8g", "l1"), ("8GB", "l1"), ("low", "l1"), ("6g", "l2"), ("nocuda", "cpu"),
             ("  l1  ", "l1"), ("l1-8g", "l1")]
    bad = [(raw, prof_mod.normalize_profile(raw)) for raw, want in pairs
           if prof_mod.normalize_profile(raw) != want]
    check("②a 别名/大小写/空白一律归到同一档（`8g`→`l1`、`6g`→`l2`、`nocuda`→`cpu`）", not bad, bad)

    #: ⚠️ 不认的**必须抛** ✗✗ —— 静默回落 ``auto`` 会让用户以为档位生效了 ✓✗
    raised = {}
    for raw in ("low1", "l3", "16g", "gpu-only"):
        try:
            prof_mod.normalize_profile(raw)
            raised[raw] = "没抛"
        except ValueError as err:
            raised[raw] = "ok" if raw in str(err) else f"报错没点名 {err}"
    check("②b 不认的档位名当场抛 ValueError 且**点名**", all(v == "ok" for v in raised.values()), raised)

    #: ⚠️ 非字符串（例如 JSON 里写了数字 `8`）也是「不认」✓ —— 别把它读成 `8g` ✓✗
    try:
        prof_mod.normalize_profile(8)
        nonstr = "没抛"
    except ValueError:
        nonstr = "ok"
    check("②c 非字符串（`8`）同样抛（不许猜成 `8g`）", nonstr == "ok", nonstr)


# --------------------------------------------------------------- ③ 冲突裁决


def case_resolve() -> None:
    """③ ``applied`` 只放真会改的项 ＋ 冲突以显式 ``policy`` 为准 ✓✗。"""
    auto = prof_mod.resolve("auto")
    check("③a `auto` 档 `applied={}` ＋ `source=default`（不往装载键塞假配置）",
          auto["applied"] == {} and auto["source"] == "default", auto)
    gpu = prof_mod.resolve("gpu")
    check("③b `gpu` 档 `applied={}`（它那两格就是默认值 ⇒ 报「已改」是假话）",
          gpu["applied"] == {} and gpu["source"] == "default", gpu)

    l1 = prof_mod.resolve("l1")
    check("③c `l1` 档 `applied` 恰好是配额 0.60 ＋ 解码按 1 潜帧分时",
          l1["applied"] == {"hostRatio": 0.60, "vaeTiling": {"temporalChunk": 1}}, l1["applied"])
    check("③d `l1` 的 `applied` 里**没有** policy（它写的就是默认 `auto` ⇒ 不算改）",
          "policy" not in l1["applied"], l1["applied"])
    cpu = prof_mod.resolve("cpu")
    check("③e `cpu` 档 `applied` 一定含 `policy=cpu`（这条是真改）",
          cpu["applied"].get("policy") == "cpu", cpu["applied"])

    #: ⚠️ 冲突：显式 ``policy`` **赢** ✓ —— 但只有当档位真想改 policy 时才算「忽略」✓✗
    clash = prof_mod.resolve("cpu", policy="gpu")
    check("③f `profile=cpu` ＋ 显式 `policy=gpu` ⇒ policy 被剔除、`profileIgnored=true`",
          "policy" not in clash["applied"] and clash["ignored"] == ["policy"]
          and clash["profileIgnored"] is True, clash)
    no_clash = prof_mod.resolve("l1", policy="gpu")
    check("③g `profile=l1` ＋ 显式 `policy=gpu` ⇒ **不算冲突**（档位本来就没压 policy）",
          no_clash["ignored"] == [] and no_clash["profileIgnored"] is False, no_clash)

    tiled = prof_mod.resolve("l1", vae_tiling={"temporalChunk": 4})
    check("③h 显式 `vaeTiling.temporalChunk` 赢，档位那条进 `ignored`",
          "vaeTiling" not in tiled["applied"]
          and tiled["ignored"] == ["vaeTiling.temporalChunk"], tiled)
    only_tile = prof_mod.resolve("l1", vae_tiling={"tile": 512})
    check("③i 只给了 `tile`（没提 temporalChunk）⇒ 档位仍**补上**分时（tile 换字节、分时不换）",
          only_tile["applied"].get("vaeTiling") == {"temporalChunk": 1}, only_tile["applied"])

    check("③j `notWired` 恰好四项且每项都有理由",
          [row["knob"] for row in l1["notWired"]]
          == ["reserveFor", "gpuLayers", "readAhead", "foresight"]
          and all(len(row["why"]) > 20 for row in l1["notWired"]),
          l1["notWired"])
    check("③k `assumed` 只有两格（假设值别混进 `applied`）",
          set(l1["assumed"]) == {"gpuGiB", "hostGiB"}, l1["assumed"])


def case_describe() -> None:
    """④ 给人看的那份清单与 `resolve` **同一份真源** ✓✗。"""
    rows = prof_mod.describe()
    check("④a `describe()` 覆盖五档且键与表一致",
          [row["key"] for row in rows] == list(prof_mod.PROFILES), [row["key"] for row in rows])
    mismatch = {}
    for row in rows:
        resolved = prof_mod.resolve(row["key"])
        if row["changes"] != resolved["applied"]:
            mismatch[row["key"]] = (row["changes"], resolved["applied"])
    check("④b 每档 `changes` 与 `resolve(applied)` 逐项相同（两处各算一遍必漂）", not mismatch, mismatch)
    check("④c `auto`/`gpu` 两档 `changes` 为空（前端据此决定标不标「已改」）",
          all(not row["changes"] for row in rows if row["key"] in ("auto", "gpu")),
          {row["key"]: row["changes"] for row in rows})
    check("④d 每档都有 `why`（回执要能一句话解释这档在压什么）",
          all(isinstance(row["why"], str) and row["why"] for row in rows))


# --------------------------------------------------------------- ⑤ 预算


def case_budgets() -> None:
    """⑤ ``budgets``：哪几栏真读数 ＋ 假设与实测**分得开** ✓✗。"""
    full = prof_mod.budgets("l1", pressure=_card(7.0, 8.0),
                            host=prof_mod.HostMemory(totalBytes=32 * GIB,
                                                     availBytes=20 * GIB))
    check("⑤a 卡与内存都读得到 ⇒ `source=cuda+host`", full["source"] == "cuda+host", full)
    check("⑤b 配额＝**可用**内存 × 档位比例（20 × 0.60 = 12 GiB，不是总量 × 比例）",
          full["hostQuotaGiB"] == round(20 * 0.60, 2), full["hostQuotaGiB"])
    check("⑤c `assumed` 仍如实是 8/32（档位假设，不因为读到 8.0 就「对上」）",
          full["assumed"] == {"gpuGiB": 8.0, "hostGiB": 32.0}, full["assumed"])

    none = prof_mod.budgets("l1")
    check("⑤d 什么都测不到 ⇒ `source=none` 且数值是 `None`（**不写 0**：0 读成「一点都没有」）",
          none["source"] == "none" and none["gpuGiB"] is None and none["hostGiB"] is None
          and none["hostQuotaGiB"] is None, none)
    only_card = prof_mod.budgets("l2", pressure=_card(5.0, 6.0))
    check("⑤e 只有卡 ⇒ `source=cuda`、内存三格 `None`",
          only_card["source"] == "cuda" and only_card["hostGiB"] is None
          and only_card["hostQuotaGiB"] is None, only_card)
    only_host = prof_mod.budgets("l2", host=prof_mod.HostMemory(totalBytes=16 * GIB,
                                                               availBytes=12 * GIB))
    check("⑤f 只有内存 ⇒ `source=host`、卡三格 `None`",
          only_host["source"] == "host" and only_host["gpuGiB"] is None
          and only_host["gpuFreeGiB"] is None, only_host)

    #: ⚠️ 两个 ``tuple[int, int]`` 形状全同 ✓✗ ⇒ 顺序反了**不报错**、只把配额算虚高 ✓✗
    total, avail = 32 * GIB, 20 * GIB
    by_named = prof_mod.budgets("l1", host=prof_mod.HostMemory(totalBytes=total, availBytes=avail))
    by_tuple = prof_mod.budgets("l1", host=(total, avail))
    swapped = prof_mod.budgets("l1", host=(avail, total))
    check("⑤g `HostMemory` 与元组 `(总量, 可用)` 结果相同（带名字的入口就是为这个）",
          (by_named["hostGiB"], by_named["hostFreeGiB"]) == (by_tuple["hostGiB"],
                                                             by_tuple["hostFreeGiB"]),
          (by_named, by_tuple))
    check("⑤h 反验：把顺序写成 `(可用, 总量)` ⇒ 配额**必然不同**（这条坑不报错，只会算错）",
          swapped["hostQuotaGiB"] != by_named["hostQuotaGiB"],
          (swapped["hostQuotaGiB"], by_named["hostQuotaGiB"]))

    check("⑤i `gpu` 档的配额比例就是默认值（引用 `HOST_USE_RATIO`，不抄数）",
          prof_mod.budgets("gpu")["hostRatio"] == hc.HOST_USE_RATIO,
          prof_mod.budgets("gpu")["hostRatio"])

    hint = full["hint"]
    check("⑤j 实测 8 GiB 卡配 `l1` ⇒ 不喊（对得上）", hint is None, hint)


# --------------------------------------------------------------- ⑥ 错配提示


def case_hint() -> None:
    """⑥ ``hint``：只提示、不拦 ＋ 阈值经得起真机 ✓✗。"""
    check("⑥a `l1` 跑在 22.49 GiB 的卡上 ⇒ 必须提示（2.8 倍刚好过 2 倍阈值）",
          prof_mod.hint("l1", gpu_gib=22.49) is not None, prof_mod.hint("l1", gpu_gib=22.49))
    check("⑥b 反验：阈值不是 3 倍（3 倍会让 22.49/8 = 2.8 倍滑过去）",
          "大得多" in (prof_mod.hint("l1", gpu_gib=22.49) or ""), prof_mod.hint("l1", gpu_gib=22.49))
    check("⑥c 卡比假设**小** ⇒ 也提示（真装不下）",
          prof_mod.hint("l1", gpu_gib=4.0) is not None, prof_mod.hint("l1", gpu_gib=4.0))
    check("⑥d 内存按 10% 相对容差：32 GB 机器报 30 GiB 可用**不喊**（否则提示变噪音）",
          prof_mod.hint("l1", host_gib=30.0) is None, prof_mod.hint("l1", host_gib=30.0))
    check("⑥e 内存差得远（16 档碰上 8 GiB）⇒ 喊",
          prof_mod.hint("l2", host_gib=8.0) is not None, prof_mod.hint("l2", host_gib=8.0))
    check("⑥f `auto` 档不预设 ⇒ 永不喊（它没资格说三道四）",
          prof_mod.hint("auto", gpu_gib=1.0, host_gib=1.0) is None,
          prof_mod.hint("auto", gpu_gib=1.0, host_gib=1.0))


# --------------------------------------------------------------- ⑦ 源码纪律


def case_ast() -> None:
    """⑦ 纯函数纪律 ＋ 接线（AST 判，不靠人读）✓✗。"""
    tree = ast.parse(_source(ENGINE / "profile.py"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    check("⑦a `profile.py` **不 import torch**（纯函数 ⇒ 没有读数也能算）", "torch" not in imported, imported)
    check("⑦b `profile.py` 不读盘（没有 `open(` / `Path` / `read_text`）",
          not re.search(r"\bopen\s*\(|\bPath\s*\(|read_text\s*\(", _source(ENGINE / "profile.py")))

    runtime_src = _source(ENGINE / "runtime.py")
    runtime_tree = ast.parse(runtime_src)
    #: ⚠️ ``profile`` 必须进**装载键** ✓✗ —— 不进键就会「换了档却复用上一套布局」且不报错 ✓✗
    keys = []
    for node in ast.walk(runtime_tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_load_key":
            keys.append({kw.arg for kw in node.keywords})
    check("⑦c `_load_key` 的调用点（`ensure_loaded` ＋ 内部装配）**都带 `profile`**",
          len(keys) >= 2 and all("profile" in row for row in keys), keys)
    check("⑦d 两处建键的归一都写在**建键那一行**（`policy=_policy_key(...)`，不靠上一行赋值）",
          len([row for row in keys if "policy" in row]) == len(keys)
          and runtime_src.count("policy=_policy_key(policy)") >= 2,
          (keys, runtime_src.count("policy=_policy_key(policy)")))

    #: ⚠️ 七条装配线的 ``_ensure_*_loaded`` 都要收档位 ✓（漏一条 ⇒ 那条线上的装配期 `host_cache`
    #: 配额还是按默认 0.75 算，而回执说「已按 l1」✓✗）
    ensure_sigs = {}
    for node in ast.walk(runtime_tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_ensure_") \
                and node.name.endswith("_loaded"):
            args = [arg.arg for arg in node.args.kwonlyargs]
            ensure_sigs[node.name] = args
    missing = {name: args for name, args in ensure_sigs.items()
               if "profile_name" not in args or "profile_applied" not in args}
    check(f"⑦e 全部 {len(ensure_sigs)} 条 `_ensure_*_loaded` 都收 `profile_name`/`profile_applied`",
          ensure_sigs and not missing, missing or list(ensure_sigs))

    #: ⚠️ 构造 ``TorchBackend`` 的地方**必须**把档位传下去 ✓✗（构造期定死 ⇒ 漏传就是那条线没档位）
    backend_calls = []
    bare_calls = 0
    for node in ast.walk(runtime_tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "TorchBackend":
            kwargs = {kw.arg for kw in node.keywords}
            #: ⚠️ 零关键字那处是 ``TorchBackend().describe()`` ✓（只为取族表/自述 ✓，**不装载** ✗）
            #: ⇒ 它不需要档位 ✓；凡是**带装配参数**（`device`/`policy`/`residency`… ✓）的都必须是
            #: 装配线 ✓✗ —— 漏传档位 ⇒ 那条线装配期的 `host_cache` 配额还是 0.75 ✓，而回执说已按档位 ✓✗。
            if not kwargs:
                bare_calls += 1
                continue
            backend_calls.append(kwargs)
    check("⑦f 每处**装配用** `TorchBackend(...)` 都带 `profile_name` ＋ `profile_applied`"
          "（零关键字那处是取族表的，不算装配线）",
          backend_calls and all({"profile_name", "profile_applied"} <= row for row in backend_calls),
          {"calls": backend_calls, "bare": bare_calls})
    check("⑦f′ 反验：装配线调用点数 ≥ 8（七条 DiT 线 ＋ H3 ⇒ 少一条就是漏传）",
          len(backend_calls) >= 8, len(backend_calls))

    #: ⚠️ 出图线（SDXL）走 ``SdxlBackend``，它没有 `host_cache` 那本账 ⇒ 档位有一格**必然无效** ✓✗
    #: ⇒ 必须**看得见**（回执 `profileNote.ignored`），不许静默跳过 ✗
    check("⑦g SDXL 那条线如实报 `profileNote.ignored`（`hostRatio` 在它上面不生效）",
          "profileNote" in runtime_src and '"hostRatio"' in runtime_src, "profileNote")

    router_src = _source(ROUTERS / "engine.py")
    check("⑦h `/load` 白名单收 `profile`（三写法那批旁边）",
          re.search(r'"profile":\s*"profile"', router_src) is not None)
    check("⑦i `/load` 的归一调**同一个函数** `machine_profile.normalize_profile`（不手抄别名表）",
          "machine_profile.normalize_profile" in router_src)
    check("⑦j 归一失败转成 `StageError`（不冒 500 —— 用户只是把 `l1` 写成 `low1`）",
          re.search(r"machine_profile\.normalize_profile\(values\[\"profile\"\]\)[\s\S]{0,200}"
                    r"raise pipe\.StageError", router_src) is not None)
    check("⑦k `auto` 视为**没配**（从 `values` 里摘掉 ⇒ 换回默认档不白装一遍）",
          re.search(r'values\["profile"\] == "auto"[\s\S]{0,120}values\.pop\("profile"\)',
                    router_src) is not None)
    check("⑦l `/engine/runtime` 收 `?profile=` 且拼错回 400（不静默回落）",
          "def runtime_status(profile" in router_src
          and "machine_profile.normalize_profile(profile)" in router_src)

    facade_src = _source(ENGINE / "facade.py")
    check("⑦m 门面 `status()` 透传 ``profile``（CLI 与路由同一份事实）",
          "def status(profile" in facade_src and "status(profile=profile)" in facade_src)

    #: ⚠️ 反向：装配线**不许**自己去改 `host_cache` 的配额 ✗✗ —— 一处展开是 G1 的全部意义 ✓
    check("⑦n 反验：`profile.py` 是唯一读 `HOST_USE_RATIO` 决定「档位配额」的地方"
          "（`runtime.py` 只透传，不另算比例）",
          "HOST_USE_RATIO" not in runtime_src, "runtime.py 里出现了 HOST_USE_RATIO")

    #: ⚠️ 人面对那一层（`python -m app.services.engine status --profile l1`）也要认 ✓✗ ——
    #: 否则「8 GiB 卡」的机器上，用户只能靠 HTTP 才知道这档下还剩多少配额 ✓✗。
    main_src = _source(ENGINE / "__main__.py")
    check("⑦o CLI `status` 收了 `--profile`（只影响 budgets，不装载）",
          re.search(r'status\.add_argument\("--profile"', main_src) is not None)
    check("⑦p CLI 拼错档位 ⇒ `E_BAD_CONFIG`（退出 2）而不是 `E_INTERNAL`（退出 3）",
          'code="E_BAD_CONFIG"' in main_src and "normalize_profile(key)" in main_src)


# --------------------------------------------------------------- ⑧ 真接线


def case_runtime_wiring() -> None:
    """⑧ 不装载也要能问出「这一档下预算多少」＋ 坏档名当场拒 ✓✗。"""
    from app.services.engine import runtime as runtime_mod  # noqa: PLC0415

    engine = runtime_mod.EngineRuntime()
    status = engine.status()
    check("⑧a `status()` 带 `profile` 与 `budgets` 两栏", "profile" in status and "budgets" in status,
          sorted(status))
    check("⑧b 没装过 ⇒ `profile.source=default` 且档位是 `auto`（不假装用户选过）",
          status["profile"]["source"] == "default" and status["profile"]["profile"] == "auto",
          status["profile"])
    check("⑧c `budgets.profile` 与 `profile.profile` 一致（一处回答）",
          status["budgets"]["profile"] == status["profile"]["profile"],
          (status["budgets"]["profile"], status["profile"]["profile"]))

    asked = engine.status(profile="l1")
    check("⑧d 问 `l1` ⇒ `source=request`、配额比例 0.60（按档位算，不是按默认 0.75）",
          asked["profile"]["source"] == "request" and asked["budgets"]["hostRatio"] == 0.60,
          asked["budgets"])
    check("⑧e 问档位**不改装载状态**（只读：`loadReport` 仍为空）",
          asked["loadReport"] is None, asked["loadReport"])

    try:
        engine.status(profile="low1")
        bad = "没抛"
    except ValueError:
        bad = "ok"
    check("⑧f `status(profile=拼错)` 抛 ValueError（路由层才转 400）", bad == "ok", bad)

    try:
        engine.ensure_loaded(profile="low1")
        loaded = "没抛"
    except ValueError:
        loaded = "ok"
    except Exception as err:  # noqa: BLE001 —— 走到装载才算「没在门口拒」
        loaded = f"抛错了地方 {type(err).__name__}"
    check("⑧g `ensure_loaded(profile=拼错)` 在**装载之前**就拒（不等装 19 GiB 才炸）",
          loaded == "ok", loaded)

    #: ⚠️ 装载键必须带档位 ✓✗ —— 否则换了档复用旧布局、且不报错 ✓✗
    key_auto = runtime_mod._load_key(stage="krea2", profile="auto")
    key_l1 = runtime_mod._load_key(stage="krea2", profile="l1")
    check("⑧h 同一族换档 ⇒ **装载键不同**（换档必须重装）", key_auto != key_l1,
          (key_auto, key_l1))

    router = __import__("app.routers.engine", fromlist=["runtime_status"])
    resp = router.runtime_status(profile="low1")
    #: ⚠️ 路由回的是 ``JSONResponse`` ✓（不是 dict ✓）⇒ 判据必须读 ``body`` ✓，
    #: 直接 ``str(resp)`` 只能拿到对象地址 ✓✗（那条判据会永远假绿 ✓✗）。
    text = (resp.body or b"").decode("utf-8", "replace") if hasattr(resp, "body") else str(resp)
    check("⑧i 路由 `?profile=low1` ⇒ 400 且错误里点名坏档位（不是泛泛的 500）",
          getattr(resp, "status_code", 200) == 400 and "low1" in text
          and ("machine" in text or "profile" in text or "机器档" in text), text)

    #: ⚠️ CLI 那条路**真调一次** ✓✗（只看源码会漏掉「参数没接上」✓ —— 源码里有名字不等于跑得通 ✓）。
    from app.services.engine import __main__ as engine_cli  # noqa: PLC0415
    from app.services.engine import facade as facade_mod  # noqa: PLC0415

    args = engine_cli.build_parser().parse_args(["status", "--profile", "low1", "--json"])
    try:
        engine_cli._cmd_status(args)  # noqa: SLF001 —— 夹具就是要打这条口
        cli_bad = "没抛"
    except facade_mod.EngineError as err:
        cli_bad = "ok" if err.code == "E_BAD_CONFIG" else f"码不对 {err.code}"
    check("⑧j CLI `status --profile low1` 抛 `E_BAD_CONFIG`（退出 2，不会被当成内部错）",
          cli_bad == "ok", cli_bad)

    args_ok = engine_cli.build_parser().parse_args(["status", "--profile", "l1", "--json"])
    check("⑧k CLI `status --profile l1` 真跑通（不是只把参数收进来就算了）",
          engine_cli._cmd_status(args_ok) == 0, "退出码非 0")  # noqa: SLF001


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("== 机器档（G1）==")
    case_table()
    case_normalize()
    case_resolve()
    case_describe()
    case_budgets()
    case_hint()
    case_ast()
    case_runtime_wiring()
    passed = _TOTAL - len(_FAILS)
    print(f"\nSUMMARY: {passed}/{_TOTAL} passed")
    return 0 if passed == _TOTAL and _TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(main())
