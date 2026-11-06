"""一次性真跑入口：把某条出图 stage 真权重跑出一张 PNG ✓。

用法（跑完就删，别提交）：

    ./.venv/Scripts/python.exe -X utf8 _run_qwen_image.py --stage qwen_image \
        --dit "...qwen_image_2512_bf16.safetensors" --text "...qwen_2.5_vl_7b_fp8_scaled.safetensors" \
        --vae "...qwen_image_vae.safetensors" --tokenizer "...tokenizers\qwen3vl_32b" \
        --steps 8 --cfg 1.0 --mp 1.76 --ratio 1:1 --seed 20261101

⚠️ 不替调用方挑权重、不猜路径（猜错是"装齐了、跑得出、图是噪声"✗）。
⚠️ 与 HTTP 那条路**同一个入口**（`runtime.ensure_loaded` + `submit`/`wait`）——
   不走私有方法，写出来的图与 `/engine/generate` 逐字节同源。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.engine import guidance as guidance_mod          # noqa: E402
from app.services.engine import pipeline as pipe                 # noqa: E402
from app.services.engine.runtime import EngineUnavailable, engine_runtime  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="qwen_image")
    parser.add_argument("--dit", required=True)
    #: ⭐ 第二份主干 ✓（`stage=wan22` 的**低噪专家** ✓）。⚠️ 不给它 ⇒ 装配线当场拒 ✓✗
    #: （不是"接力退化成单专家" ✓ —— 那种退化会让后段细节照样糊且一条报错都没有 ✓✗）。
    parser.add_argument("--dit2", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--tokenizer", default=None)
    #: ⭐ 视频族的首帧 ✓（`stage=wan22` 必给 ✓；图族不认它 ✓）。
    parser.add_argument("--first-frame", default=None)
    parser.add_argument("--fps", type=int, default=None,
                        help="不给就用 GenerationRequest 的默认值 ✓（wan22 会把 24 换成官方 16 ✓）")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--mp", type=float, default=1.76)
    parser.add_argument("--ratio", default="1:1")
    parser.add_argument("--target", default=None,
                        help="画布档位（profiles.PROFILES ✓，如 storyboard-last 1344×768 ✓）；"
                             "给了它就别再给 --mp/--ratio ✗（两处口径二选一 ✓）")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--refs", default="", help="逗号分隔的参考图路径（Edit 用）")
    parser.add_argument("--init", default=None,
                        help="图生图的**起点参考图**（与 --denoise<1 必须成对 ✓；经 VAE 编码成起点 latent）")
    parser.add_argument("--denoise", type=float, default=1.0,
                        help="0~1 ✓；<1 必须同时给 --init（只砍 σ 而不给图 ⇒ 起点仍是纯噪声 ✗）")
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "output" / "engine_run"))
    parser.add_argument("--residency", default="split", choices=("off", "split", "park"))
    parser.add_argument("--budget", type=float, default=None, help="MiB")
    parser.add_argument("--policy", default="auto")
    parser.add_argument("--report", default=None, help="把装载报告写进这个 json")
    args = parser.parse_args()

    #: ⚠️ `submit(assembly=…)` 是**直接 kwargs 展开**给 `_ensure_loaded_internal` ✓ ⇒ 这里必须是
    #: **后端键名**（snake_case ✓）。camelCase 归一在 HTTP 路由层（`_ASSEMBLY_KEYS` ✓），进程内
    #: 直调**没有那一步** ⇒ 给 `ditPath` 会炸成 `unexpected keyword argument` ✓✗（踩过 ✓）。
    assembly = {
        "stage": args.stage,
        "dit_path": args.dit,
        "dit_path2": args.dit2,
        "text_weights": args.text,
        "image_vae_weights": args.vae,
        "tokenizer_path": args.tokenizer,
        "residency": args.residency,
        "policy": args.policy,
    }
    if args.budget:
        assembly["residency_budget_mib"] = args.budget

    _log(f"ensure_loaded start（stage={args.stage} / residency={args.residency} / policy={args.policy}）")
    started = time.perf_counter()
    report = engine_runtime.ensure_loaded(**{k: v for k, v in {
        "stage": args.stage,
        "dit_path": args.dit,
        "dit_path2": args.dit2,
        "text_weights": args.text,
        "image_vae_weights": args.vae,
        "tokenizer_path": args.tokenizer,
        "residency": args.residency,
        "residency_budget_mib": args.budget,
        "policy": args.policy,
    }.items() if v is not None})
    load_seconds = time.perf_counter() - started
    _log(f"ensure_loaded 完成 用时 {load_seconds:.1f}s")
    weights = (report or {}).get("weights") or {}
    _log("装载回执：" + json.dumps({
        "stage": (report or {}).get("stage"),
        "reused": (report or {}).get("reused"),
        "residency": weights.get("residency"),
        "device": weights.get("device"),
        "dtype": weights.get("dtype"),
        "paramCount": weights.get("paramCount"),
    }, ensure_ascii=False))
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                     encoding="utf-8")
        _log(f"装载报告已写 {args.report}")

    request = pipe.GenerationRequest(
        prompt=args.prompt,
        negative=args.negative,
        seed=args.seed,
        steps=args.steps,
        megapixels=args.mp,
        ratio=args.ratio,
        target=args.target,
        outputs_dir=args.outdir,
        stage=args.stage,
        denoise=args.denoise,
        init_image=args.init,
        #: ⚠️ 不给 `fps` 就落回 GenerationRequest 的默认 **24** ✓（wan22 会换 16 并在 warning 里
        #: **说清这一档给不了 24** ✓）。这里显式传 16 是为了让回执里不出现那条"猜你要多少"的
        #: 提醒 ✓ —— 不是把 24 偷偷改掉 ✓（改帧率的是 plan ✓，脚本只搬 ✓）。
        fps=int(args.fps or pipe.GenerationRequest.fps),
        seconds=float(args.seconds),
        first_frame=args.first_frame,
        reference_frames=tuple(x for x in args.refs.split(",") if x.strip()),
        guidance=guidance_mod.GuidanceConfig(scale=args.cfg),
    )
    plan = pipe.build_plan(request)
    _log("计划：" + json.dumps({k: v for k, v in plan.to_dict().items() if k != "sigmas"},
                               ensure_ascii=False))

    _log("submit …")
    task_id = engine_runtime.submit(request, assembly=assembly, stage=args.stage)
    last_note = ""
    started = time.perf_counter()
    while True:
        task = engine_runtime.get_task(task_id, events_limit=400) or {}
        status = task.get("status")
        events = task.get("events") or []
        if events:
            tail = events[-1]
            note = f"{tail.get('stage')}/{tail.get('label')} step={tail.get('step')}"
            if note != last_note:
                last_note = note
                elapsed = time.perf_counter() - started
                _log(f"  {elapsed:7.1f}s  {json.dumps(tail, ensure_ascii=False)[:220]}")
        if status in ("done", "failed", "cancelled"):
            break
        time.sleep(5.0)
    total = time.perf_counter() - started

    done = engine_runtime.wait(task_id, timeout=5) or task
    _log(f"结束 status={done.get('status')} 用时 {total:.1f}s")
    print(json.dumps(done, ensure_ascii=False, indent=2, default=str)[:8000], flush=True)

    result = (done.get("result") or {})
    if result.get("error"):
        _log("ERROR " + json.dumps(result["error"], ensure_ascii=False))
        return 1
    outputs = result.get("outputs") or {}
    for name, path in outputs.items():
        # ⚠️ `TorchBackend.write()` 返回的字典里**混着一句说明**（键名 `note` ✓，值是中文句子 ✗ 不是路径）
        #   —— 2026-11-02 真跑时它被当成路径去 `exists()` ✓✗ ⇒ 每次出图都多报一条
        #   「声明了但盘上没有」✓，而那行话里恰好写着「**真 PNG** ✓」✓ ⇒ 真异常会被这条假警报埋掉 ✓✗。
        #   这里按**键名**认它（不按"看起来像不像路径"猜 ✓）：只有 `note` 是说明，其余键都是产物 ✓。
        if name == "note":
            _log(f"产出说明 [{name}] {path}")
            continue
        p = Path(str(path))
        if p.exists():
            _log(f"产出 [{name}] {p} {p.stat().st_size} 字节")
        else:
            _log(f"产出 [{name}] 声明了但盘上没有：{p}")
    return 0 if done.get("status") == "done" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EngineUnavailable as err:
        _log(f"装配失败：{err}")
        raise SystemExit(2) from err
