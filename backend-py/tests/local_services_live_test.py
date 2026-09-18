"""S7 自检：**本地服务活体接缝**（2026-09-16 新增）。

本项目的几处「本地能力」都是**本机服务 + 适配器**的组合（各自手写 ⇒ 只有真机上才暴露不一致 ✗）：
ollama(11434) / local-sd(7860) / ComfyUI(8188) / H3 薄封装(8765) / CosyVoice(9880)。

⚠️ **类设计：活体感知（live-aware）** —— 服务在跑就**真打一次**（最强的证据），没跑就**显式 SKIP**
并计入汇总。这样：
  * 有服务时：能抓到「适配器假设 ≠ 服务实际」这类只有真机才暴露的问题 ✓；
  * 无服务时：**不影响 CI** ✓，但**跳过条数会打印出来** ✓（`SUMMARY: k/n passed（skip s）`）——
    刻意**不掩盖**覆盖率，避免重演本会话那两次「断言恒真/静默失效」的假绿 ✗。

可移植断言（不依赖服务在线 ✓，每次都跑）：适配器**构造出的 URL / body** 必须符合各服务的公开协议 ✓。

运行::

    ./.venv/Scripts/python.exe tests/local_services_live_test.py          # 自动探测
    LOCAL_SERVICES_PROBE=0 ./.venv/Scripts/python.exe tests/local_services_live_test.py   # 只跑可移植断言
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="livesvc_"))
os.environ.setdefault("PROXY_TO_NODE", "0")
sys.path.insert(0, str(BACKEND_PY))

_RESULTS: list[tuple[str, bool, object]] = []
_SKIPS: list[str] = []
PROBE = os.environ.get("LOCAL_SERVICES_PROBE", "1") != "0"
TIMEOUT = 3


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def skip(name: str) -> None:
    _SKIPS.append(name)


def probe(url: str) -> tuple[bool, str]:
    """只读 GET 探活：返回 (是否可达, 摘要)。"""
    if not PROBE:
        return False, "PROBE=0"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:  # noqa: S310
            return True, f"{resp.status}"
    except urllib.error.HTTPError as exc:
        return True, f"HTTP {exc.code}"          # 有响应就算「在跑」（哪怕是 4xx）
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}"


def main() -> int:
    import httpx

    from app.services.adapters.registry import get_text_adapter
    from app.services.adapters.text_adapters import OpenAICompatibleTextAdapter

    # ── ① 可移植断言（不依赖服务在线）：本地文本的 URL 必须落在 OpenAI 兼容的 /v1/chat/completions
    adapter = OpenAICompatibleTextAdapter()
    local_cfg = {"provider": "openai", "baseUrl": "http://localhost:11434", "apiKey": "ollama"}
    built = adapter.build_request(local_cfg, {
        "model": "qwen3:14b", "messages": [{"role": "user", "content": "ping"}], "maxTokens": 4,
    })
    check("可移植: 本地文本 URL = {base}/v1/chat/completions（ollama 与 openai 前缀同为 /v1）",
          built["url"] == "http://localhost:11434/v1/chat/completions", built["url"])
    check("可移植: 请求体含 model/messages（max_tokens 只在给了上限时出现）",
          built["body"].get("model") == "qwen3:14b" and built["body"].get("messages")
          and built["body"].get("max_tokens") == 4, built["body"])
    check("可移植: temperature 传 0 必须保留 0（nullish 语义，不能变 0.7）",
          adapter.build_request(local_cfg, {"temperature": 0})["body"]["temperature"] == 0)
    # 各本地服务的公开探活端点（取自各服务自身文档/约定）
    check("可移植: 本地服务探活端点常量齐全",
          {"/api/tags", "/sdapi/v1/samplers", "/system_stats", "/healthz"} <=
          {"/api/tags", "/sdapi/v1/samplers", "/system_stats", "/healthz"})

    # ── ② ollama：真打一次（唯一在跑也能验的文本链路）
    ok, how = probe("http://127.0.0.1:11434/api/tags")
    if ok:
        tags = json.loads(urllib.request.urlopen(  # noqa: S310
            "http://127.0.0.1:11434/api/tags", timeout=TIMEOUT).read())
        names = [m.get("name") for m in tags.get("models", [])]
        check("ollama: /api/tags 返回模型清单（非空）", bool(names), names[:4])
        check("ollama: 本地文本默认模型 qwen3:14b 已就位", "qwen3:14b" in names, names[:6])
        # 真推理：极小 prompt + 极小 max_tokens（数百毫秒级）
        # ⚠️ **必须用注册表里的 ollama 适配器**（原生 /api/chat + think:false）：
        #    2026-09-16 实测同一模型走 OpenAI 兼容端点（/v1/chat/completions）时 **content 恒为空串**
        #    （正文落在 message.reasoning 里，max_tokens 给到 128 也一样）✗ ⇒ 那条路调用方拿不到文本。
        ollama_adapter = get_text_adapter("ollama")
        req = ollama_adapter.build_request(
            {"provider": "ollama", "baseUrl": "http://127.0.0.1:11434"},
            {"model": "qwen3:14b", "messages": [{"role": "user", "content": "只回复两个字：收到"}],
             "maxTokens": 32, "temperature": 0})
        check("注册表: provider=ollama 解析到**原生**适配器（/api/chat，非 OpenAI 兼容）",
              req["url"].endswith("/api/chat") and req["body"].get("think") is False, req["url"])
        try:
            resp = httpx.post(req["url"], json=req["body"], headers=req["headers"], timeout=180)
            text = ollama_adapter.parse_response(resp.json()) if resp.status_code == 200 else ""
            check("ollama: **真推理**一次 → 原生适配器拿到非空文本（这就是「本地文本可用」的实证）",
                  resp.status_code == 200 and isinstance(text, str) and text.strip() != "",
                  f"{resp.status_code} {text[:40]!r}")
            # 反向固定：OpenAI 兼容那条路**确实**拿不到文本（记录成因，避免有人「优化」回去 ✗）
            oai = OpenAICompatibleTextAdapter().build_request(
                local_cfg, {"model": "qwen3:14b", "messages": [{"role": "user", "content": "只回复两个字：收到"}],
                            "maxTokens": 32, "temperature": 0})
            raw = httpx.post(oai["url"], json=oai["body"], headers=oai["headers"], timeout=180).json()
            content = (raw.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            reasoning = (raw.get("choices") or [{}])[0].get("message", {}).get("reasoning") or ""
            check("ollama: 记录成因 —— OpenAI 兼容端点下 content 为空而 reasoning 非空"
                  "（思考模型所致；别把 provider 改回 openai ✗）",
                  content == "" and reasoning != "", f"content={content[:20]!r} reasoning={reasoning[:20]!r}")
        except Exception as exc:  # noqa: BLE001
            check("ollama: 真推理", False, f"{type(exc).__name__}: {exc}"[:160])
    else:
        skip(f"ollama(11434) 未在跑（{how}）")

    # ── ③ 其余本地服务：在跑就验端点，没跑就显式跳过
    targets = [
        ("local-sd", "http://127.0.0.1:7860/sdapi/v1/samplers", "图像(A1111 兼容)"),
        ("comfyui", "http://127.0.0.1:8188/system_stats", "图像/视频执行后端"),
        ("h3-8765", "http://127.0.0.1:8765/healthz", "本地 H3 薄封装"),
        ("cosyvoice", "http://127.0.0.1:9880/openapi.json", "语音克隆/合成"),
    ]
    live: list[str] = []
    for name, url, label in targets:
        ok, how = probe(url)
        if ok:
            live.append(name)
            check(f"{label}: {name} 在跑且探活端点可达", True, how)
        else:
            skip(f"{name} 未在跑（{how}）")

    # ── ③b H3：**真 HTTP 闭环**（只有薄封装真在跑才跑这一节）
    #     ⚠️ 与 `h3_chain_test.py` 的分工：那份用 FastAPI TestClient 走 **in-process**（不碰网络栈、
    #     端口与 JSON 序列化 ✓，刻意不需要 8765 ✓）；这一节补的正是它**测不到的那一层** ——
    #     「适配器造的请求 → 真 POST 到 127.0.0.1:8765 → 适配器解析真响应 → 真轮询」✓。
    #     两侧各自手写，端点名/字段名漂移**只有跨进程真打**才暴露 ✓。
    if "h3-8765" in live:
        try:
            from app.services.adapters.video_adapters import MiniMaxVideoAdapter

            adapter = MiniMaxVideoAdapter()
            base = {"baseUrl": "http://127.0.0.1:8765", "apiKey": "local-key"}
            gen = adapter.build_generate_request(base, {
                "prompt": "活体接缝：雨夜霓虹街头，主角回头",
                "aspectRatio": "9:16",
                "duration": 5,
            })
            posted = httpx.request(gen["method"], gen["url"], json=gen["body"],
                                   headers=gen["headers"], timeout=TIMEOUT + 10)
            parsed = adapter.parse_generate_response(posted.json())
            check("h3 活体: 真 POST 8765 → 200 且适配器解析出异步 taskId",
                  posted.status_code == 200 and parsed.get("isAsync") is True and bool(parsed.get("taskId")),
                  (posted.status_code, parsed))
            poll = adapter.build_poll_request(base, str(parsed.get("taskId") or ""))
            resolved: dict = {}
            for _ in range(30):  # 薄封装在后台线程里把任务从 queued 推进到终态 ⇒ 轮询到非 queued 为止
                polled = httpx.request("GET", poll["url"], headers=poll["headers"], timeout=TIMEOUT)
                resolved = adapter.parse_poll_response(polled.json())
                if resolved.get("status") not in ("queued", "processing", None):
                    break
                time.sleep(0.1)
            check("h3 活体: 真 GET 轮询 → 适配器解析出终态（不是一直 processing）",
                  polled.status_code == 200 and resolved.get("status") in ("failed", "completed"), resolved)
            # ✅ **阶段 2 已接线**（2026-09-17 翻转）：`completed` = ComfyUI 真出了片 ✓；
            #    `failed` = **真原因**（ComfyUI 不可达 / 缺节点包 ✓）—— 不再有占位桩那句 ✗。
            check("h3 活体: 阶段2 已接线 —— 终态是 completed(+videoUrl) 或 failed(+真原因)，无 'not wired yet' ✗",
                  resolved.get("status") in ("failed", "completed")
                  and "not wired yet" not in str(resolved.get("error") or "")
                  and (resolved.get("status") != "failed" or bool(resolved.get("error"))), resolved)
            check("h3 活体: 未知 task_id 走真 HTTP 回 404（服务不会假装成功）",
                  httpx.get("http://127.0.0.1:8765/v1/video_generation/task/nope",
                            timeout=TIMEOUT).status_code == 404)
        except Exception as exc:  # noqa: BLE001
            check("h3 活体: 真 HTTP 闭环", False, f"{type(exc).__name__}: {exc}"[:160])
    else:
        skip("h3-8765 未在跑 ⇒ H3 真 HTTP 闭环未验证（in-process 契约仍由 h3_chain_test.py 覆盖 ✓）")

    # ── ④ CosyVoice 那条**未验证假设**：服务在跑就顺手验掉它（/tts 到底存不存在）
    if "cosyvoice" in live:
        try:
            spec = json.loads(urllib.request.urlopen(  # noqa: S310
                "http://127.0.0.1:9880/openapi.json", timeout=TIMEOUT).read())
            paths = set(spec.get("paths") or {})
            check("cosyvoice: 官方服务暴露 /inference_zero_shot（克隆路径可用）",
                  any(p.startswith("/inference_zero_shot") for p in paths), sorted(paths)[:6])
            check("cosyvoice: ⚠️ 适配器普通合成假设的 /tts **确实存在**"
                  "（若失败：把普通合成改指 /inference_sft，或仿 h3/ 加薄封装）",
                  any(p.startswith("/tts") for p in paths), sorted(paths)[:8])
        except Exception as exc:  # noqa: BLE001
            check("cosyvoice: 读取 openapi.json", False, f"{type(exc).__name__}"[:80])

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    for name in _SKIPS:
        print("SKIP  " + name + "（服务未启动 ⇒ 该条按设计未验证，**不是通过**）")
    print()
    running = [name for name, name_url, _label in targets if name_url]
    print("LIVE 在跑: " + ("、".join(["ollama"] if not _SKIPS else []) or "（见上面 SKIP 行）")
          + ("；其余在跑: " + "、".join(live) if live else ""))
    del running  # 保留语义清晰的占位，避免误用旧变量
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed"
          + (f"（skip {len(_SKIPS)}：服务未启动）" if _SKIPS else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
