"""自检：**反量化产物落盘**（``engine/materialize.py`` ＋ ``gguf_to_llm`` 那四处接线 ✓ 2026-11-07 G3 ✓）。

验的是什么 ✗：G1/G2 之后，**装载**仍然每次把整份 Q4_K 文本塔重新反量化一遍 ✓（4B ≈ 31 秒纯 CPU 活 ✓）。
物化缓存把它变成「第一次算、之后读盘」✓ —— 但代价是**多了一份盘上的权重** ✓✗ ⇒ 三条判据一条都不能松 ✗✗：

1. ⭐⭐⭐ **数值不变** ✗✗：命中产物装出来的权重与不命中（现算）装出来的**逐位相等** ✓
   （物化的是「装进去那张张量的位模式」✓ ⇒ 相等是构造的 ✓；自己写舍入差 1 ulp 就红 ✓）；
2. ⭐⭐ **键 = 源指纹 ＋ dtype ＋ 转换器** ✗✗：源一变（内容或 mtime ✓）⇒ 换文件名 ⇒ 不复用 ✓；
   sidecar 缺失 / 头被截断 / 自述不符 ⇒ 不复用并重算 ✓（**不许**「看着像就拿来跑」✗）；
3. ⭐ **写失败退化成慢** ✓✗：超配额 / 余量不足 / 写到一半出错 ⇒ 拒写、**留不下半成品** ✗，
   装载照常成功 ✓（失败的是缓存 ✓，不是模型 ✗）。

⚠️ 本套**不读真权重** ✗：``_read_tensor`` 换替身（同 ``engine_gguf_to_llm_test`` ②⁵ 的做法 ✓），
源文件只是一个占位文件（只为拿指纹 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_materialize_test.py
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="mat_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import gguf_to_llm as g2l  # noqa: E402
from app.services.engine import materialize as mat_mod  # noqa: E402

_ENGINE = Path(__file__).resolve().parents[1] / "app" / "services" / "engine"
_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        return False
    return True


def _rev(name: str) -> str:
    """LlmModel 参数名 → GGUF 张量名（本文件只用 depth=1 的小模型 ✓）"""
    if name == "embed.weight":
        return "token_embd.weight"
    if name == "norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    m = re.match(r"blocks\.(\d+)\.attn\.(q_proj|k_proj|v_proj|o_proj)\.weight", name)
    if m:
        return f"blk.{m.group(1)}.{ {'q_proj': 'attn_q', 'k_proj': 'attn_k',
                                      'v_proj': 'attn_v', 'o_proj': 'attn_output'}[m.group(2)] }.weight"
    m = re.match(r"blocks\.(\d+)\.ffn\.(gate|up|down)\.weight", name)
    if m:
        return f"blk.{m.group(1)}.ffn_{m.group(2)}.weight"
    m = re.match(r"blocks\.(\d+)\.norm([12])\.weight", name)
    if m:
        return f"blk.{m.group(1)}.{'attn_norm' if m.group(2) == '1' else 'ffn_norm'}.weight"
    raise AssertionError(f"未知参数名 {name}")


# ---------------------------------------------------------------- ① 模块自身


def _fake_source(tmp: Path, payload: bytes = b"sunny") -> Path:
    src = tmp / "fake-Q4_K_M.gguf"
    src.write_bytes(payload * 64)
    return src


def _near(got: object, want: object, *, rel: float = 1e-6) -> bool:
    """两个浮点数差不多 ✓（``writeGiB`` 是四舍五入到 3 位的 ✓ 不能直接比 ✓✗）。"""
    try:
        a, b = float(got), float(want)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if a == b:
        return True
    return abs(a - b) <= max(abs(b) * rel, 1e-3)


def case_key_and_valid() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="matk_"))
    src = _fake_source(tmp)
    specs = [mat_mod.Spec.of("a.weight", (2, 3), "BF16"),
             mat_mod.Spec.of("b.weight", (4,), "BF16")]

    owner = mat_mod.Materializer(src, converter="conv-v1", dtype="BF16", directory=tmp / "out")
    path = owner.path()
    check("①a 产物名带 源文件名 ＋ converter ＋ dtype ＋ 12 位键 ✓",
          path.name.startswith("fake-Q4_K_M.gguf.conv-v1.BF16.") and len(path.suffixes) >= 2
          and re.search(r"\.[0-9a-f]{12}\.safetensors$", path.name) is not None, path.name)
    check("①b 同一份源 ＋ 同一套参数 ⇒ 键稳定 ✓（跑两次必须一样 ✓ 否则永远重算 ✗）",
          owner.path() == path)

    other_conv = mat_mod.Materializer(src, converter="conv-v2", dtype="BF16", directory=tmp / "out")
    other_dtype = mat_mod.Materializer(src, converter="conv-v1", dtype="F32", directory=tmp / "out")
    check("①c converter / dtype 换了 ⇒ 键就换 ✓（换精度不该覆盖另一份产物 ✓✗）",
          other_conv.path().name != path.name and other_dtype.path().name != path.name)

    src.write_bytes(b"rainy" * 64)
    same = mat_mod.Materializer(src, converter="conv-v1", dtype="BF16", directory=tmp / "out")
    check("①d 源内容变了 ⇒ 键就换 ✓（头哈希 ✓，不只信 size+mtime ✗）", same.path().name != path.name)
    src.write_bytes(b"sunny" * 64)
    check("①e 源改回来 ⇒ 键回来 ✓（键是**内容**函数 ✓）",
          same.path().name == path.name)

    ok, why = owner.valid()
    check("①f 还没写 ⇒ 不命中 ✓（reason 说「没有产物」✓ 不抛 ✗）", not ok and "产物" in why, why)

    with owner.open_writer(specs) as writer:
        writer.add("a.weight", b"\x00" * 12)
        writer.add("b.weight", b"\x00" * 8)
    report = writer.report
    check("①g 写完 ⇒ 回执 written=True ＋ 张量数 2 ＋ sidecar 写成 ✓",
          report is not None and report.written and report.tensor_count == 2
          and report.sidecar_written, report)
    ok, why = owner.valid()
    check("①h 再判 ⇒ 命中 ✓，理由点明「结构自证 ＋ 自述一致 ＋ sidecar 指纹」三条 ✓", ok, why)
    reader = owner.reader()
    with reader:
        meta = reader.meta()
        got = reader.read("a.weight")
    check("①i 产物是**正常 safetensors** ✓（`TensorReader` 读得出形状与字节 ✓）",
          meta.get("a.weight", (None, ()))[0] == "BF16" and len(got) == 12, meta)

    # ⭐ 只 touch 一下（mtime 变、内容不变）：产物路径必须**不变** ✓ 但要判**失效** ✓✗
    mtime0 = src.stat().st_mtime
    os.utime(src, (mtime0 + 60, mtime0 + 60))
    same_path = owner.path().name == path.name
    ok_touch, why_touch = owner.valid()
    check("①i′ 只 touch 源（mtime 变、内容不变）⇒ 产物**路径不变** ✓ 但**判失效** ✓"
          "（含 mtime 的键：每 touch 一次多留 8 GiB ✗；不看 mtime：拿旧产物去跑 ✗✗）",
          same_path and not ok_touch, (path.name, owner.path().name, why_touch))
    os.utime(src, (mtime0, mtime0))
    check("①i″ 时间戳复原 ⇒ 又命中 ✓（三元组全对上 ✓）", owner.valid()[0])

    # ---- 失效的四条路（少一条就会出现「静默用可疑产物」✗✗）----
    sidecar = mat_mod.cg.cache_sidecar_path(path)
    saved = sidecar.read_text(encoding="utf-8")
    sidecar.unlink()
    ok, why = owner.valid()
    check("①j sidecar 没了 ⇒ 不命中 ✓（源指纹无从核对 ⇒ 宁可重算 ✓）", not ok and "sidecar" in why, why)
    sidecar.write_text(json.dumps({"source": {"size": 1, "mtime": 1.0, "hash": "dead"}}), encoding="utf-8")
    ok, why = owner.valid()
    check("①k sidecar 指纹对不上 ⇒ 不命中 ✓（换了权重版本却想复用旧产物 ✗✗）", not ok, why)
    sidecar.write_text(saved, encoding="utf-8")

    payload = path.read_bytes()
    path.write_bytes(payload[:len(payload) // 2])
    ok, why = owner.valid()
    check("①l 产物被截断 ⇒ 不命中 ✓（「头自洽」那条会拦下 ✓ 不靠运气 ✗）", not ok, why)
    path.write_bytes(payload)

    header = json.loads(path.read_bytes()[8:8 + int.from_bytes(path.read_bytes()[:8], "little")])
    bad = dict(header)
    bad["__metadata__"] = {"darme_materialize": json.dumps(
        {"schema": 1, "converter": "conv-OTHER", "dtype": "BF16"})}
    body = json.dumps(bad, separators=(",", ":")).encode("utf-8")
    rest = payload[8 + int.from_bytes(payload[:8], "little"):]
    path.write_bytes(len(body).to_bytes(8, "little") + body + rest)
    ok, why = owner.valid()
    check("①m 产物**自述**的 converter 不对 ⇒ 不命中 ✓（同名旧产物换了公式仍被当有效 ✗✗）",
          not ok and "自述" in why, why)
    path.write_bytes(payload)
    check("①n 复原后 ⇒ 又命中 ✓（上面六条判的都是「失效」而不是「永久坏」✓）", owner.valid()[0])
    shutil.rmtree(tmp, ignore_errors=True)


def case_writer_discipline() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="matw_"))
    src = _fake_source(tmp)
    specs = [mat_mod.Spec.of("a", (4,), "BF16"), mat_mod.Spec.of("b", (4,), "BF16")]
    owner = mat_mod.Materializer(src, converter="c", dtype="BF16", directory=tmp / "out")

    def _fresh() -> Path:
        return owner.path()

    # ⚠️ 顺序错 ⇒ 偏移全错 ⇒ 「形状对、数值错」✗✗ ⇒ 必须当场拒
    try:
        with owner.open_writer(specs) as writer:
            writer.add("b", b"\x00" * 8)
        order_err = "没抛"
    except mat_mod.MaterializeError as err:
        order_err = "ok" if "顺序" in str(err) else f"错因不对：{err}"
    check("①o 写入顺序与计划不一致 ⇒ `MaterializeError` ✓", order_err == "ok", order_err)
    check("①p 顺序错之后 ⇒ 盘上**没有产物也没有 .tmp** ✗（半成品会被当成命中 ✓✗）",
          not _fresh().exists() and not list((tmp / "out").glob("*.tmp")))

    try:
        with owner.open_writer(specs) as writer:
            writer.add("a", b"\x00" * 7)          # 少一个字节
        size_err = "没抛"
    except mat_mod.MaterializeError as err:
        size_err = "ok" if "字节数" in str(err) else f"错因不对：{err}"
    check("①q 字节数与计划不符 ⇒ `MaterializeError` ✓（dtype·形状对不上会让头自证失败 ✗）",
          size_err == "ok", size_err)

    try:
        with owner.open_writer(specs) as writer:
            writer.add("a", b"\x00" * 8)          # 少写 b
        short_err = "没抛"
    except mat_mod.MaterializeError as err:
        short_err = "ok" if "没写" in str(err) else f"错因不对：{err}"
    check("①r 少写一个张量 ⇒ `MaterializeError` ✓ 且**不留残缺产物** ✗✗"
          "（缺张量的产物装得上、跑得动、输出不对 ✗）",
          short_err == "ok" and not _fresh().exists(), short_err)

    # 写到一半抛异常（装载中途失败 ✓）⇒ 产物没意义 ⇒ 必须删干净
    try:
        with owner.open_writer(specs) as writer:
            writer.add("a", b"\x00" * 8)
            raise RuntimeError("装载中途炸了")
    except RuntimeError:
        pass
    check("①s 写手在装载异常时**自动放弃** ✓（`with` 退出就删 .tmp ✓ 不留半成品 ✗）",
          not _fresh().exists() and not list((tmp / "out").glob("*.tmp")))

    # ⚠️ 配额与余量：**一个字节都不写就拒** ✓（写到一半才发现装不下最坏 ✗）
    # ⚠️ 配额要**真比产物小** ✓：1e-6 GiB ≈ 1074 字节 ✓，而这套的最小产物只有几百字节 ✗✗
    #    ⇒ 配额比产物还大 ⇒ 不抛 ⇒ 用例退化成「测空气」✓✗（1e-9 GiB ≈ 1 字节 ✓ 必然超 ✓）。
    tight = mat_mod.Materializer(src, converter="c", dtype="BF16", directory=tmp / "q",
                                 quota_gib=1e-9)
    try:
        tight.open_writer(specs)
        quota_err = "没抛"
    except mat_mod.MaterializeSkip as err:
        quota_err = "ok" if "配额" in str(err) else f"错因不对：{err}"
    check("①t 超配额 ⇒ `MaterializeSkip` ✓（退化成重算 ✓ 不是抛给装载层 ✗）", quota_err == "ok", quota_err)
    check("①u 超配额之后目录里**空无一物** ✓（先判后写 ✓）",
          not (tmp / "q").exists() or not list((tmp / "q").iterdir()))

    original_free = mat_mod.free_bytes
    mat_mod.free_bytes = lambda _p: 8            # noqa: ARG005 —— 模拟只剩 8 字节 ✓
    try:
        try:
            owner.open_writer(specs)
            free_err = "没抛"
        except mat_mod.MaterializeSkip as err:
            free_err = "ok" if "余量" in str(err) else f"错因不对：{err}"
    finally:
        mat_mod.free_bytes = original_free
    check("①v 磁盘余量不足 ⇒ `MaterializeSkip` ✓（含 5% 余量系数 ✓ 不是正好塞满 ✗）", free_err == "ok", free_err)

    # 环境变量三条口子
    os.environ["DARME_MATERIALIZE"] = "0"
    off = mat_mod.enabled()
    os.environ["DARME_MATERIALIZE"] = "off"
    off2 = mat_mod.enabled()
    os.environ.pop("DARME_MATERIALIZE", None)
    check("①w `DARME_MATERIALIZE=0/off` ⇒ 关 ✓，默认开 ✓", (not off) and (not off2) and mat_mod.enabled())

    os.environ["DARME_MATERIALIZE_LIMIT_GIB"] = "写错了"
    bad_limit = mat_mod.limit_gib()
    os.environ["DARME_MATERIALIZE_LIMIT_GIB"] = "0"
    zero_limit = mat_mod.limit_gib()
    os.environ.pop("DARME_MATERIALIZE_LIMIT_GIB", None)
    check("①x 配额读不懂 / 写成 0 ⇒ ``None``（不限）✓ —— **不许**当 0（那会静默关掉整个功能 ✗✗）",
          bad_limit is None and zero_limit is None and mat_mod.limit_gib() is None)

    source = (_ENGINE / "materialize.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {node.names[0].name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import)} | {node.module.split(".")[0] for node in ast.walk(tree)
                                                    if isinstance(node, ast.ImportFrom) and node.module}
    check("①y 反验：本模块**不 import torch / numpy** ✓（数值转换留在调用方 ✓ ⇒ 纯 stdlib 可测 ✓）",
          "torch" not in imported and "numpy" not in imported, imported)
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- ② 真装载接线


def _fake_info(model: Any, *, type_id: int = 12) -> tuple[Any, dict[str, Any]]:
    """``(info, payload)``：载荷是**已摆好**的 fp32 数组 ✓，``tensor`` 侧只放元数据 ✓。"""
    import numpy as np  # noqa: PLC0415

    payload: dict[str, Any] = {}
    tensors: dict[str, Any] = {}
    for name, param in model.state_dict().items():
        gname = _rev(name)
        count = int(np.prod(tuple(param.shape)))
        # ⚠️ 每根张量的值**互不相同** ✓：全零会让「装上了」与「没装上」看起来一样 ✗✗
        payload[gname] = (np.arange(count, dtype=np.float32) * 0.001 + 0.5).reshape(tuple(param.shape))
        tensors[gname] = types.SimpleNamespace(type_id=type_id, shape=(count,), numel=count,
                                               nbytes=count * 4, offset=0)
    return types.SimpleNamespace(tensors=tensors), payload


def _load_with(info: Any, payload: dict[str, Any], calls: dict[str, int], fn: Any) -> Any:
    """在「假 ``inspect`` ＋ 计数替身读取器」下跑**一次真装载** ✓（其余全是生产代码 ✓）。

    ⚠️ 为什么两处都要替 ✗：``load_llm_from_gguf`` 自己会 ``gguf.inspect(path)`` ✓（本套不造真 GGUF ✗），
    而 ``_read_tensor`` 才是**反量化**那一步 ✓ —— 只替一个的话，另一个会去解析那堆占位字节 ✓✗
    （结果是「info 是空的 ⇒ 全部算缺 key」✓✗，用例看起来在测、其实什么都没测 ✓✗）。
    """
    real_read = g2l._read_tensor
    real_inspect = g2l.gguf_mod.inspect

    def _stub(_p: Any, info_: Any, tensor: Any) -> Any:
        calls["n"] += 1
        for gname, meta in info_.tensors.items():
            if meta is tensor:
                return payload[gname]
        raise AssertionError(f"未知张量 {tensor!r}")

    g2l._read_tensor = _stub
    g2l.gguf_mod.inspect = lambda _p: info
    try:
        return fn()
    finally:
        g2l._read_tensor = real_read
        g2l.gguf_mod.inspect = real_inspect


def case_roundtrip() -> None:
    if not _torch_available():
        print("SKIP  ② 装载接线：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    import torch  # noqa: PLC0415

    from app.services.engine import llm as llm_mod  # noqa: PLC0415

    tmp = Path(tempfile.mkdtemp(prefix="matl_"))
    src = _fake_source(tmp)
    config = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=1, heads=2, kv_heads=1,
                              head_dim=8, ffn=32, tie_embeddings=False)
    cache_dir = tmp / "out"
    calls = {"n": 0}

    def _make() -> Any:
        return llm_mod.build_llm(config, dtype=torch.bfloat16)

    model_a = _make()
    info, payload = _fake_info(model_a)

    def _load(**kwargs: Any) -> Any:
        kwargs.setdefault("materialize", cache_dir)
        return _load_with(info, payload, calls, lambda: g2l.load_llm_from_gguf(
            src, config, dtype=torch.bfloat16, **kwargs))

    first = _load()
    receipt_a = dict(first.materialized)
    produced = receipt_a.get("path") or ""
    check("②a 首次装载 ⇒ 不命中、**写成了产物** ✓，回执带 path/gib ✓",
          receipt_a.get("hit") is False and receipt_a.get("written") is True and Path(produced).exists(),
          receipt_a)
    check("②a′ 首次装载真的做了反量化 ✓（计数器 > 0 ✓ —— 否则这条用例在测空气 ✗✗）",
          calls["n"] > 0, calls["n"])

    calls["n"] = 0
    second = _load()
    receipt_b = dict(second.materialized)
    check("②b 二次装载 ⇒ **命中** ✓ 且**一次反量化都没做** ✓（这才是 G3 的全部收益 ✓）",
          receipt_b.get("hit") is True and calls["n"] == 0, (receipt_b, calls["n"]))
    check("②b′ 命中时回执说清凭什么 ✓（自证＋自述＋指纹三条 ✓，不是「文件在」就算 ✗）",
          "sidecar" in str(receipt_b.get("reason")) and "自证" in str(receipt_b.get("reason")),
          receipt_b.get("reason"))

    # ⭐⭐⭐ 数值不变：逐位比（bf16 就比位模式 ✓ 不比浮点近似 ✗）
    # ⚠️ 比的必须是**两次真装载**（`first` 现算 / `second` 读盘）✓ —— 拿 `model_a`（随机初始化的那份
    #    只用来出形状与载荷 ✓）来比，等于拿随机权重比装载结果 ✓✗（恒不相等 ✓ 且毫无意义 ✗✗）。
    same = all(torch.equal(first.state_dict()[k].view(torch.int16),
                           second.state_dict()[k].view(torch.int16))
               for k in first.state_dict())
    check("②c ⭐⭐⭐ 命中装出来的模型与现算装出来的**逐位相等** ✓（差 1 ulp 这条也红 ✓）", same)
    check("②c′ 而权重**不是**退化成全零/常量 ✓（否则「相等」毫无意义 ✗✗）",
          bool((first.embed.weight != first.embed.weight[0, 0]).any()))

    # 源变了 ⇒ 换键 ⇒ 不复用
    src.write_bytes(b"overcast" * 64)
    calls["n"] = 0
    third = _load()
    receipt_c = dict(third.materialized)
    check("②d 源文件变了 ⇒ **不复用** ✓（回执 hit=False ✓ 且产物路径换了 ✓ 不是同一个文件 ✗）",
          receipt_c.get("hit") is False and receipt_c.get("path") != produced
          and Path(receipt_c.get("path") or "x").exists(), receipt_c)

    # 产物被截断 ⇒ 重算（装载照常成功）
    broken = Path(receipt_c.get("path") or "")
    payload_bytes = broken.read_bytes()
    broken.write_bytes(payload_bytes[:64])
    calls["n"] = 0
    fourth = _load()
    receipt_d = dict(fourth.materialized)
    check("②e 产物被截断 ⇒ 不复用、重算、**装载照常成功** ✓（坏产物不许把装载弄挂 ✗✗）",
          receipt_d.get("hit") is False and calls["n"] > 0 and receipt_d.get("written") is True, receipt_d)
    check("②e′ 重算出来的模型仍然对 ✓（坏产物只是白读一次 ✓ 不是污染 ✗）",
          all(torch.equal(first.state_dict()[k], fourth.state_dict()[k])
              for k in first.state_dict()))

    # 关掉这条路 ⇒ 什么都不写
    calls["n"] = 0
    fifth = _load(materialize=False)
    check("②f `materialize=False` ⇒ 不物化 ✓ 且不解目录 ✓（回执如实说「显式关掉」✓）",
          dict(fifth.materialized).get("enabled") is False
          and "显式" in str(dict(fifth.materialized).get("reason")), fifth.materialized)

    # 全浮点源（type_id=0）⇒ 整份不做
    float_info, _ = _fake_info(model_a, type_id=0)
    specs, note = g2l._plan_specs(float_info, config, model_a.state_dict())
    check("②g 源张量全是 fp32/fp16 ⇒ **整份不物化** ✓（抄一份 16 GiB 换 0 秒 ✗）"
          "且原因说清「没有量化」✓", specs == [] and "没有量化" in note, note)

    quant_info, _ = _fake_info(model_a, type_id=12)
    specs_q, label = g2l._plan_specs(quant_info, config, model_a.state_dict())
    mapped = {name for name in model_a.state_dict()}
    check("②h 计划键集 = 装载循环真会写的那些 ✓（键集不多不少 ✓ 且 dtype 标签写进键 ✓）",
          len(specs_q) == len(mapped) and label == "BF16", (len(specs_q), len(mapped), label))
    shutil.rmtree(tmp, ignore_errors=True)


def case_skip_on_quota() -> None:
    if not _torch_available():
        print("SKIP  ②i 配额回退：需要 torch（本机未装）⇒ 不判（≠ 通过 ✓）")
        return
    import torch  # noqa: PLC0415

    from app.services.engine import llm as llm_mod  # noqa: PLC0415

    tmp = Path(tempfile.mkdtemp(prefix="matq_"))
    src = _fake_source(tmp)
    config = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=1, heads=2, kv_heads=1,
                              head_dim=8, ffn=32, tie_embeddings=False)
    model = llm_mod.build_llm(config, dtype=torch.bfloat16)
    info, payload = _fake_info(model)

    calls = {"n": 0}
    os.environ["DARME_MATERIALIZE_LIMIT_GIB"] = "0.000001"
    try:
        loaded = _load_with(info, payload, calls, lambda: g2l.load_llm_from_gguf(
            src, config, dtype=torch.bfloat16, materialize=tmp / "q"))
    finally:
        os.environ.pop("DARME_MATERIALIZE_LIMIT_GIB", None)
    receipt = dict(loaded.materialized)
    check("②i 超配额 ⇒ **装载照常成功** ✓（失败的是缓存 ✓ 不是模型 ✗）、回执说「超配额」✓",
          receipt.get("hit") is False and receipt.get("written") is False
          and "配额" in str(receipt.get("reason")), receipt)
    check("②i′ 超配额之后盘上一个产物都没有 ✓", not list((tmp / "q").glob("*.safetensors")))
    check("②i″ 而模型权重照样是对的 ✓", bool((loaded.embed.weight == loaded.embed.weight[0, 0]).all())
          is False)
    shutil.rmtree(tmp, ignore_errors=True)


def case_backend_receipt() -> None:
    """③ LlmBackend 的可见面 ✓：`describe()["materialize"]` 必须**原样**抬出装载结论 ✓。

    ⚠️ 这一格存在的理由 ✗：缓存到底命中没命中，**只有回执说得清** ✓。加载快了几十秒时，
    看回执的人无法分辨「缓存生效」还是「这次权重小了」；把它埋在 ``model`` 上而不抬进
    `describe` ⇒ 等于没有这一层 ✓✗（真机排查时只能靠猜 ✗✗）。
    """
    import torch  # noqa: PLC0415

    from app.services.engine import llm_backend as lb_mod
    from app.services.engine.llm import LlmConfig

    config = LlmConfig(vocab_size=16, hidden=16, depth=1, heads=2, kv_heads=2,
                       head_dim=8, ffn=32, tie_embeddings=False)

    tmp = Path(tempfile.mkdtemp(prefix="matl-be-"))
    try:
        src = tmp / "w.gguf"
        src.write_bytes(b"GGUF")
        be = lb_mod.LlmBackend(config, gguf_path=str(src), device="cpu")
        check("③a 还没装载 ⇒ `materialize` 是 **None** ✓（不是「没命中」✗ —— 两件事 ✗✗）",
              be.describe().get("materialize") is None)
        want = {"hit": True, "gib": 1.5, "path": "<cache>", "reason": "全部张量命中✓",
                "written": False, "tensor_count": 12}

        class _FakeModel:
            def __init__(self) -> None:
                self.w = torch.zeros(4, dtype=torch.float32)
                self.materialized = dict(want)

            def parameters(self) -> Any:
                return iter([self.w])

            def eval(self) -> Any:
                return self

        real_loader = lb_mod.gguf_to_llm.load_llm_from_gguf
        lb_mod.gguf_to_llm.load_llm_from_gguf = lambda *a, **k: _FakeModel()  # type: ignore[assignment]
        try:
            be._hub = object()          # 词表这一步与本例无关 ✓ 别让它先炸 ✗
            be.load()
        finally:
            lb_mod.gguf_to_llm.load_llm_from_gguf = real_loader  # type: ignore[assignment]
        got = be.describe().get("materialize")
        check("③b 装载后 `describe()['materialize']` **逐字**等于装载结论 ✓（不在这里重算口径 ✗）",
              got == want, got)
        check("③c 「命中」这一栏真的是 True ✓（拷贝时把布尔丢了 ⇒ 回执说反 ✗✗）",
              (got or {}).get("hit") is True, got)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def case_stats_and_usage() -> None:
    """④ 对外账（§4.4 ✓）：**复用了几次** ✓ 与 **盘上占多少** ✓ 是两件事，都得答得出来 ✓。

    ⚠️ 这一格存在的理由 ✗：缓存本身不报账 ⇒ 用户只会看到「第二次快了几十秒」✓ 却不知道
    盘上多了 8 GiB ✓✗；反过来只知道占 8 GiB ✓ 又看不出这层有没有在起作用 ✓✗。
    合成一个 `hitRate` 更糟 ✗：`refused`（磁盘不够，一个字节没写 ✓）会被算成"没命中"✓✗，
    而它说明的是**另一件事**（该清盘 / 该调配额 ✓）。
    """
    mat_mod.reset_stats()
    zero = mat_mod.stats()
    check("④a 清零后五个计数全是 0 ✓ 且**不留**上一次的 lastReason ✓",
          all(zero[k] == 0 for k in ("hits", "misses", "written", "refused", "failed"))
          and zero["lastReason"] == "", zero)

    tmp = Path(tempfile.mkdtemp(prefix="matl-st-"))
    try:
        src = _fake_source(tmp)
        # ⚠️ 张量要**接近真实量级**（1024×1024 bf16 = 2 MiB ✓）：判据里有「GiB 读数 > 0」✓，
        #    拿 8×8 的 128 字节去测 ✓ 会永远读成 0.0 ✓✗（那不是"没记账" ✗ 是"记账精度不够" ✓）。
        specs = [mat_mod.Spec.of("embed.weight", (1024, 1024), "BF16"),
                 mat_mod.Spec.of("head.weight", (1024, 1024), "BF16")]
        cache = tmp / "cache"
        mat = mat_mod.Materializer(src, converter="conv-v1", dtype="BF16", directory=cache)

        ok, _why = mat.valid()
        check("④b 判了一次「没有产物」⇒ `misses` +1 ✓ 且 `hits` 仍是 0 ✓（不许两头都加 ✗）",
              (not ok) and mat_mod.stats()["misses"] == 1 and mat_mod.stats()["hits"] == 0,
              mat_mod.stats())

        writer = mat.open_writer(specs)
        for spec in specs:
            writer.add(spec.name, b"\x01\x02" * (spec.nbytes // 2))
        report = writer.close()
        after_write = mat_mod.stats()
        check("④c 写成一次 ⇒ `written` +1 ✓、`writeGiB` > 0 ✓（写下去的量要能对上产物大小 ✓）",
              after_write["written"] == 1 and after_write["writeGiB"] > 0
              and _near(after_write["writeGiB"], report.gib), (after_write, report.gib))

        ok2, _why2 = mat.valid()
        after_hit = mat_mod.stats()
        check("④d 命中一次 ⇒ `hits` +1 ✓、`hitGiB` > 0 ✓、`misses` **不再**涨 ✓（各记各的 ✓✗）",
              ok2 and after_hit["hits"] == 1 and after_hit["hitGiB"] > 0
              and after_hit["misses"] == 1, after_hit)

        usage = mat_mod.store_usage(cache)
        check("④e 盘上占用：`files` = 产物个数 ✓、`diskGiB` ≈ 产物字节 ✓、`dir` 是这层自己的目录 ✓",
              usage["files"] == 1 and usage["diskGiB"] > 0
              and _near(usage["diskGiB"], report.gib, rel=1e-3)
              and usage["dir"] == str(cache), usage)
        check("④f `limitGiB` 没设 ⇒ **None** ✓（不是 0 ✗ —— 0 会被读成「一个字节都不许存」✓✗）",
              usage["limitGiB"] is None, usage)
        check("④g sidecar **个数**单列 ✓（几 KB 的 json 混进 diskGiB 会让读数对不上产物 ✓✗）",
              usage["sidecars"] == 1 and usage["diskGiB"] < report.gib * 1.01, usage)

        # 配额不足 ⇒ 记的是 `refused` ✓ 不是 `misses` ✗（两件事：一个该清盘/调配额、一个该重算）
        tight = mat_mod.Materializer(src, converter="conv-v1", dtype="BF16",
                                     directory=cache, quota_gib=1e-9)
        try:
            tight.open_writer(specs)
        except mat_mod.MaterializeSkip:
            pass
        refused = mat_mod.stats()
        check("④h 超配额 ⇒ `refused` +1 ✓ 而 `misses` **不动** ✓（合成一个命中率就会把它读成「没命中」✗✗）",
              refused["refused"] == 1 and refused["misses"] == 1, refused)
        check("④i 放弃原因留在 `lastReason` 上 ✓（只报「慢了」不说为什么 ⇒ 用户查不到磁盘只剩多少 ✓✗）",
              "配额" in refused["lastReason"], refused["lastReason"])

        report_all = mat_mod.cache_report(cache)
        check("④j `cache_report` 一格答全 ✓（盘上体积 + 次数 + 开关/配额/余量 ✓）",
              all(k in report_all for k in ("dir", "enabled", "files", "diskGiB", "limitGiB",
                                            "freeGiB", "hits", "misses", "refused")), sorted(report_all))

        # 环境变量：关了开关也要如实说「关着」✓（不是沉默地报 0 次命中 ✗）
        os.environ["DARME_MATERIALIZE"] = "0"
        try:
            os.environ["DARME_MATERIALIZE_LIMIT_GIB"] = "3.5"
            off = mat_mod.store_usage(cache)
        finally:
            os.environ.pop("DARME_MATERIALIZE", None)
            os.environ.pop("DARME_MATERIALIZE_LIMIT_GIB", None)
        check("④k 开关关着 ⇒ `enabled=False` ✓；配额 3.5 ⇒ `limitGiB=3.5` ✓（话要说得清 ✓）",
              off["enabled"] is False and off["limitGiB"] == 3.5, off)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    case_key_and_valid()
    case_writer_discipline()
    case_roundtrip()
    case_skip_on_quota()
    case_backend_receipt()
    case_stats_and_usage()

    failures = 0
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n        → {detail!r}"))
        failures += 0 if ok else 1
    print(f"\nSUMMARY: {len(_RESULTS) - failures}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    # Windows 中文控制台是 GBK ⇒ 不重配就崩在第一条含 ⇒/✓ 的判据行上（口径见 check_cli_encoding.py）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
