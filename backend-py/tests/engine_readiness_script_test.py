"""S29 自检：**上机前自检入口**（服务层 `engine_readiness` ✓ + CLI ✓ + 路由 ✓ 2026-09-21）。

它把已实现的能力串成**一个入口** ✓：依赖 ✓ → 权重就绪 ✓ → 加载计划 ✓ → **真权重预检**
（键名核对 + 结构推导 + 词表 ✓）。⚠️ 本套验的是**这个入口本身** ✓（那四块各自的细节由它们
自己的套件覆盖 ✓），以及三条**最容易写错的口径** ✗：

1. **没给权重 ⇒ 那是「没查」✗ 不是「通过」✓**（本仓那条纪律 ✓：无从体检 ≠ 绿灯 ✓）；
2. **「还要下多少」不能拿 `bytes` 求和** ✗（缺文件时它恒为 0 ⇒ 打出「还差 0.00 GiB」✓✗）；
3. ⚠️ **逻辑必须在服务层** ✗ —— 第一版全在 `app/scripts/` 里 ✓✗，而那个目录**不是包** ✓
   ⇒ 路由**引用不到** ✓✗（"只有人手敲命令才能用" = 没人调用的能力 ✓）⇒ 本条钉住
   「路由端点真的在、且与服务层同源」✓。

运行::

    ./.venv/Scripts/python.exe tests/engine_readiness_script_test.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

from app.services import engine_readiness as svc  # noqa: E402

_SCRIPT = BACKEND_PY / "app" / "scripts" / "h3_readiness.py"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def tiny_h3_weights(root: Path, *, broken: bool = False) -> Path:
    """造一份**缩小版 H3 权重** ✓（走真 `H3FormTrunk` ✓ ⇒ 键名/形状都是真的 ✓）。"""
    from app.services.engine import h3_form  # noqa: PLC0415
    from app.services.engine import weights as weights_mod  # noqa: PLC0415

    tiny = {"hidden": 32, "layers": 2, "heads": 4, "head_dim": 16, "ffn": 64,
            "text_dim": 16, "latents_dim": 4, "audio_latents_dim": 8, "patch_size": (1, 2, 2),
            "time_input_dim": 16, "time_hidden": 32, "time_dim": 32, "inv_freq_len": 2,
            "refiner_layers": 1}
    model = h3_form.H3FormTrunk(**tiny, head_banks=3)
    if broken:
        # ⚠️⚠️ **"坏"要真坏** ✗：第一版拿「层数不同」当坏 ✓✗ —— 那仍是**合法** H3 ✓
        #    （审计的结构是**从文件自身推**的 ✓ ⇒ 自洽 ✓）⇒ 它当然通过 ✓✗（自检 ⑧ 当场红 ✓）。
        #    真坏法要破**键与键之间**的关系 ✓：`mlp.fc1` 的行数必须是 `2×ffn` ✓（SwiGLU 要一分为二 ✓）
        #    ⇒ 把它改成 `ffn` 行 ✓ ⇒ 期望形状对不上 ✓ 且**真装不进去** ✓。
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415

        block = model.blocks[0]
        half = int(block.mlp.fc1.weight.shape[0]) // 2
        block.mlp.fc1.weight = nn.Parameter(torch.zeros(half, block.mlp.fc1.weight.shape[1]))
        path = weights_mod.save_module_weights(model, root / "broken_h3.safetensors")
    else:
        path = weights_mod.save_module_weights(model, root / "tiny_h3.safetensors")
    return path


def tiny_tokenizer_dir(root: Path) -> Path:
    """造一份**自研能覆盖**的词表 ✓（ByteLevel-BPE ✓）。"""
    from app.services.engine import tokenizer_bpe as tbp  # noqa: PLC0415

    target = root / "tokenizer"
    target.mkdir(parents=True, exist_ok=True)
    vocab = {char: index for index, char in enumerate(sorted(tbp.bytes_to_unicode().values()))}
    vocab.update({"he": 400, "ll": 401, "hell": 402, "hello": 403})
    (target / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": ["h e", "l l", "he ll", "hell o"]},
        "pre_tokenizer": {"type": "ByteLevel"},
    }), encoding="utf-8")
    return target


# ══════════════════════════════════════════════════════════════════════════
# ① 无权重：**「没查」不许说成「通过」** ✗
# ══════════════════════════════════════════════════════════════════════════
def case_without_weights() -> None:
    report = svc.collect()
    check("① 没给权重路径 ⇒ `weightsCheck=None` ✓ 且 `unchecked` **点名**「没查」✓"
          "（**不许**读成通过 ✗ —— 无从体检 ≠ 绿灯 ✓）",
          report["weightsCheck"] is None and report["unchecked"]
          and "没查" in report["unchecked"][0], report["unchecked"])
    check("② 四块都在 ✓：环境 / 权重就绪 / 加载计划 / 预检槽位 ✓",
          set(report) >= {"environment", "weightsReadiness", "loadPlan", "weightsCheck",
                          "ready", "blockers", "unchecked", "nextSteps"}, sorted(report))
    planned = svc.planned_gib(report["weightsReadiness"])
    missing = svc.planned_gib(report["weightsReadiness"], only_missing=True)
    check("③ 「还要下多少」按**清单 `expectedGiB`** 求和 ✓（拿 `bytes` 求和会恒为 0 ✓✗ —— "
          "第一版就是这么打出「还差 0.00 GiB」的 ✓）",
          planned >= missing >= 0, (planned, missing))
    check("④ 缺权重 ⇒ `ready=False` ✓ 且 `nextSteps` 里给出**下载目标目录** ✓（可行动 ✓）",
          report["ready"] is False
          and any("下权重" in step for step in report["nextSteps"]), report["nextSteps"])


# ══════════════════════════════════════════════════════════════════════════
# ② 给了（合成）权重：预检真跑 ✓ 好坏两侧都能分辨 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_with_weights(root: Path) -> None:
    good = tiny_h3_weights(root)
    tokenizer = tiny_tokenizer_dir(root)
    report = svc.collect(weights=str(good), tokenizer=str(tokenizer))
    # ⚠️ 变量名别叫 `check` ✗ —— 会把模块级那个 `check()` 函数**遮蔽**掉 ✓✗
    #    （当场 `TypeError: 'dict' object is not callable` ✓ 实测踩到 ✓）
    weights_check = report["weightsCheck"]
    check("⑤ ⭐ 真权重预检**真跑起来** ✓：结构 ok ✓ + 键名核对通过 ✓ + 层数/banks 从权重读出 ✓",
          bool(weights_check) and weights_check["structureOk"] and weights_check["audit"]["ok"]
          and weights_check["audit"]["depth"] == 2
          and weights_check["audit"]["headBanksVideo"] == 3,
          {key: weights_check.get(key) for key in ("structureOk", "configOk")})
    check("⑥ ⭐ **结构从权重推**：小尺寸（hidden=32 / heads=4 / modalities=3）如实报出 ✓"
          "（不是出厂常量 ✗）",
          weights_check["configOk"] and weights_check["trunkConfig"]["hidden"] == 32
          and weights_check["trunkConfig"]["heads"] == 4
          and weights_check["trunkConfig"]["layers"] == 2,
          weights_check["trunkConfig"])
    check("⑦ 词表也一并验了 ✓：`backend=own-bpe`（**自研** ✓）且词表口径给出 ✓",
          (weights_check.get("tokenizer") or {}).get("backend") == "own-bpe"
          and (weights_check.get("tokenizer") or {}).get("vocabSize", 0) > 250,
          weights_check.get("tokenizer"))

    broken = tiny_h3_weights(root, broken=True)
    broken_report = svc.collect(weights=str(broken), tokenizer=str(tokenizer))
    check("⑧ ⭐ **坏的权重能被拦下** ✓：键名核对不通过 ⇒ `ready=False` + 阻断项点名 ✓"
          "（不是「看着像通过」 ✗）",
          broken_report["weightsCheck"]["audit"]["ok"] is False
          and any("核对**没通过**" in item or "没通过" in item
                  for item in broken_report["blockers"]),
          broken_report["blockers"])

    missing_report = svc.collect(weights=str(root / "nope.safetensors"))
    check("⑨ 路径不存在 ⇒ **当成结论**报出 ✓（不抛 ✗）",
          missing_report["weightsCheck"]["error"] == "文件不存在 ✗", None)


# ══════════════════════════════════════════════════════════════════════════
# ③ CLI：`--json` 输出可被机器消费 ✓（自动化入口 ✓）
# ══════════════════════════════════════════════════════════════════════════
def case_cli(root: Path) -> None:
    completed = subprocess.run(  # noqa: S603 —— 跑的就是本仓脚本 ✓ 参数固定 ✓
        [sys.executable, str(_SCRIPT), "--json"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=str(BACKEND_PY), timeout=180, check=False)
    payload: Any = None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = None
    check("⑩ `--json` 是**合法 JSON** ✓ 且含四块 ✓（给自动化用 ✓ —— 人读的渲染另有一条 ✓）",
          isinstance(payload, dict)
          and {"environment", "weightsReadiness", "ready", "nextSteps"} <= set(payload),
          (completed.returncode, (completed.stdout or "")[:200]))
    check("⑪ 退出码**如实反映** `ready` ✓（脚本能被 CI / 上机脚本直接判成败 ✓）",
          (completed.returncode == 0) == bool(payload and payload.get("ready")),
          completed.returncode)


# ══════════════════════════════════════════════════════════════════════════
# ④ 路由：能力**真的接出去了** ✓（否则就是"只有人手敲命令才能用" ✗）
# ══════════════════════════════════════════════════════════════════════════
def case_route() -> None:
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from app.main import app  # noqa: PLC0415

    client = TestClient(app)
    response = client.get("/api/v1/production/engine-readiness")
    data = response.json().get("data") or {}
    check("⑫ ⭐ 端点**在** ✓ 且回**精简摘要**（不含逐步排班那坨 ✗ —— 那有几十个组件 ✓）",
          response.status_code == 200
          and {"ready", "blockers", "unchecked", "nextSteps", "environment", "weights",
               "vram", "hint"} <= set(data)
          and "loadPlan" not in data, sorted(data))
    check("⑬ 摘要与服务层**同源** ✓：`ready` / 缺件 / 三个 GiB 口径都对得上 ✓"
          "（不是各算一份 ✗ —— 那会漂 ✓）",
          isinstance(data.get("ready"), bool)
          and data["weights"]["remainingGiB"] <= data["weights"]["plannedGiB"] + 1e-6
          and data["hint"].count("--weights") == 1
          and isinstance(data["unchecked"], list), data.get("weights"))

    # ⚠️ 两种世界都成立 ✓：**盘上就绪**那一栏由 `inv.readiness` 决定 ✓ ⇒ 这里打桩 ✓
    #    （不依赖"这台机器上下没下过权重" ✗ —— 那会让断言随环境翻脸 ✓✗）
    original = svc.inv.readiness
    svc.inv.readiness = lambda *args, **kwargs: {          # type: ignore[assignment]
        "stage": "h3", "ready": True, "components": [], "missingRequired": [],
        "brokenRequired": [], "suspectRequired": [], "optionalMissing": [],
        "requiredWeightsGiB": 39.55, "modelsDir": "X:/models", "vram": {}, "catalogNodes": 8,
    }
    try:
        stub = client.get("/api/v1/production/engine-readiness").json()["data"]
    finally:
        svc.inv.readiness = original  # type: ignore[assignment]
    check("⑭ 权重齐（打桩 ✓）⇒ 摘要里 `remainingGiB=0` ✓ 且**不再**列「下权重」那一步 ✓"
          "（阻断项跟着实底走 ✓）",
          stub["weights"]["remainingGiB"] == 0
          and not any("下权重" in step for step in stub["nextSteps"]), stub["weights"])


# ══════════════════════════════════════════════════════════════════════════
# ⑤ 反量化结论：**三档分开说** ✗（支持 / 不支持 / **没查** ✓）+ GGUF 豁免 ✓
# ══════════════════════════════════════════════════════════════════════════
def case_quant() -> None:
    """`quant_plan` 是**纯函数** ✓ ⇒ 直接喂合成的 `load_plan` ✓（不用造真文件 ✓）。"""
    plan = {"components": [
        {"key": "dit_int8", "present": True,
         "dequantPlan": {"weights": 3, "supported": True, "layouts": {"per-row": 3},
                         "unresolved": [], "unpaired": [], "grouped": []}},
        {"key": "te_nvfp4", "present": True,
         "dequantPlan": {"weights": 2, "supported": False, "layouts": {},
                         "unresolved": [], "unpaired": [], "grouped": ["te.blk"],
                         "groupedCount": 1}},
        {"key": "vae_fp16", "present": False, "dequantPlan": {}},
        {"key": "dit_gguf", "present": True,
         "dequantPlan": {"weights": 1, "supported": False, "note": "GGUF：反量化由运行时做 ✓",
                         "unresolved": [], "unpaired": [], "grouped": []}},
    ]}
    quant = svc.quant_plan(plan)
    check("① 三档**分开报** ✓：能自动还原的进 `supported` ✓（附布局 ✓）、判不出来的进 `unsupported` ✓"
          "（附判不出的名字 ✓）、**没下载**的进 `notChecked` ✓",
          [item["key"] for item in quant["supported"]] == ["dit_int8"]
          and quant["supported"][0]["layouts"] == {"per-row": 3}
          and [item["key"] for item in quant["unsupported"]] == ["te_nvfp4"]
          and quant["notChecked"] == ["vae_fp16"], quant)

    check("② ⚠️ **GGUF 豁免** ✗：它的反量化由运行时（llama.cpp/ComfyUI-GGUF ✓）做 ⇒ "
          "`supported=False` **也不许**算「装不上」✗（否则每份 GGUF 都会把 `ready` 判死 ✓✗）",
          "dit_gguf" not in [item["key"] for item in quant["unsupported"]]
          and "dit_gguf" not in quant["notChecked"], quant)

    report = svc.collect()
    check("③ 本机现状（一个都没下）⇒ `notChecked` 非空 ✓ 且**不许**因此变阻塞 ✗"
          "（**没查 ≠ 通过** ✓：不阻塞 ✓、但也不假装绿 ✓ —— `ready` 的阻塞项里只有「权重没齐」✓）",
          report["quantPlan"]["notChecked"] and not report["quantPlan"]["supported"]
          and not any("反量化" in item for item in report["blockers"]),
          (report["quantPlan"]["notChecked"][:3], report["blockers"]))

    check("④ 结论进了**两份**消费者 ✓：`summary()`（路由 ✓）与 CLI 的 `--json` ✓"
          "（能力接不出去 = 只有人手敲命令才能用 ✗）",
          "quant" in svc.summary(report)
          and svc.summary(report)["quant"]["notChecked"] == report["quantPlan"]["notChecked"],
          svc.summary(report).get("quant"))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_without_weights()
        case_with_weights(root)
        case_cli(root)
        case_route()
        case_quant()
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      ↳ {detail}"))
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed"
          + (" ✗✗✗" if failed else " ✓"))
    return 1 if failed else 0


if __name__ == "__main__":
    # ⚠️ 与 `check_memory.py` / `h3_readiness.py` 同一写法 ✓（Windows GBK 控制台裸跑必加 ✗✗）：
    #    本文件打的是**判据名**（满屏 ✓/✗ ✓）与失败详情里的 `⇒` ✓ ⇒ 不加就「**崩在打印失败原因上**」✗✗。
    #    ⚠️ 2026-09-25 实测踩到：⑩ 那条 FAIL，可报错信息**一个字都打不出来** ✓✗
    #    （崩在 `print(f"FAIL ... ⇒ {detail}")` 的 U+21D2 上 ✗ ⇒ 只看到 `EXIT=1` ✓✗）。
    #    ⇒ 全量自检**也不会**暴露它 ✗：`run_all.py` 给子进程兜了 `PYTHONIOENCODING=utf-8` ✓。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
