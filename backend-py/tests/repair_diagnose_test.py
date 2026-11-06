"""修复诊断守卫：**从工作流反推依赖 → 与本机盘面比对**（源 `diagnostics.mjs` 重写）。

钉四件事：
1. **两种工作流格式都认** —— API 格式（顶层扁平节点表）与 UI 格式（``{"nodes":[...]}``）。
   ⚠️ 只认一种的话，另一种会静默反推出 0 条依赖、诊断报"什么都不缺"（假绿）✗。
2. **``registry.json`` 必须跳过** —— 它不是工作流，混进来会凭空多出依赖。
3. **缺件要带 ``usedBy``** —— 只说"缺这个文件"没用，得说"缺它堵了哪几条工作流"。
4. **同名多份要报 ambiguous，不替人挑** ✓。

运行::

    ./.venv/Scripts/python.exe tests/repair_diagnose_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    sys.path.insert(0, str(BP))
    from app.services import repair_diagnose as diag  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        workflows = root / "workflows"
        models = root / "models"
        models.mkdir(parents=True)

        # API 格式：模型 + LoRA 各一
        _write(workflows / "api-wf.json", {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "have_it.safetensors"}},
            "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "lack_lora.safetensors"}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}},
        })
        # UI 格式：widgets_values 里带文件名（含子目录前缀，要能剥掉）
        _write(workflows / "ui-wf.json", {
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["sub/dir/lack_ui.safetensors"]},
                {"id": 2, "type": "KSampler", "widgets_values": [12345, "randomize", 20]},
            ]
        })
        # registry.json 必须被跳过（这里塞一个只有它会引用的文件）
        _write(workflows / "registry.json", {
            "0": {"class_type": "UNETLoader", "inputs": {"unet_name": "only_in_registry.safetensors"}}
        })

        # 盘上只放 have_it；lack_* 故意不放
        (models / "have_it.safetensors").write_bytes(b"x" * 64)

        req = diag.workflow_requirements(workflows)
        check("反推: 只统计真工作流（registry 不计入）", req["workflows"] == 2, req["workflows"])
        names = {slot["file"] for slot in req["refs"].values()}
        check("反推: API 格式被认出", "have_it.safetensors" in names and "lack_lora.safetensors" in names, sorted(names))
        check("反推: UI 格式被认出且剥掉子目录", "lack_ui.safetensors" in names, sorted(names))
        check("反推: registry.json 被跳过", "only_in_registry.safetensors" not in names, sorted(names))
        check("反推: LoRA 与模型分开标",
              req["refs"]["lack_lora.safetensors"]["kind"] == "lora"
              and req["refs"]["have_it.safetensors"]["kind"] == "model",
              {k: v["kind"] for k, v in req["refs"].items()})
        check("反推: usedBy 指出归属",
              req["refs"]["lack_lora.safetensors"]["usedBy"] == ["api-wf"],
              req["refs"]["lack_lora.safetensors"]["usedBy"])

        result = diag.diagnose(workflows, roots=[str(models)])
        counts = result["counts"]
        check("诊断: 盘上有的算 present", counts["present"] == 1, counts)
        check("诊断: 缺 2 件", counts["missing"] == 2, counts)
        missing_files = {item["file"] for item in result["missing"]}
        check("诊断: 缺的正是那两件", missing_files == {"lack_lora.safetensors", "lack_ui.safetensors"}, missing_files)
        check("诊断: 阻塞工作流被点出",
              set(result["blockedWorkflows"]) == {"api-wf", "ui-wf"}, result["blockedWorkflows"])
        check("诊断: have_it 不该出现在缺口里",
              all(item["file"] != "have_it.safetensors" for item in result["missing"]))

        # 同名多份 ⇒ ambiguous，不替他挑
        (models / "dup").mkdir()
        (models / "dup" / "have_it.safetensors").write_bytes(b"x" * 64)
        again = diag.diagnose(workflows, roots=[str(models)])
        check("同名多份: 进 ambiguous", any(item["file"] == "have_it.safetensors" for item in again["ambiguous"]),
              [item["file"] for item in again["ambiguous"]])

        # 摘要必须封顶
        summary = diag.summarize(result, limit=1)
        check("摘要: missingTop 按 limit 截断", len(summary["missingTop"]) == 1, summary["missingTop"])
        check("摘要: 给出被截掉的条数", summary["missingMore"] == 1, summary["missingMore"])
        check("摘要: 带出 counts", summary["counts"]["missing"] == 2, summary["counts"])

        # 目录不存在要如实报错，不编空结果
        bad = diag.diagnose(root / "no-such-dir", roots=[str(models)])
        check("目录不存在: ok=false", bad["ok"] is False, bad)
        check("目录不存在: 有 error 文案", bool(bad.get("error")), bad.get("error"))

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(("PASS  " if passed_ok else "FAIL  ") + name + ("" if passed_ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
