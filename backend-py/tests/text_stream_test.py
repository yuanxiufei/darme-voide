"""``text_stream`` 流式文本自检 ✓（2026-11-01 新增，P0-1）。

判据分四层 ✓（越往下越贵 ✓，所以每一层都钉住一种「只有真流式才会错」的形状 ✗）：

1. **协议层**：SSE 帧 / 两家 ``parse_stream_line``（Ollama ndjson、OpenAI SSE）✓
2. **构造层**：``build_stream_request`` 与非流式**只差该差的几格** ✓（端点、``think``、``num_ctx`` 不许被改回 ✗）
3. **增量层**：自研引擎的「半个字符不发」+「文字级停用词只发命中点之前」✓
4. **编排层**：``stream_text`` 的事件顺序 / 思考与正文分栏 / 空流不许当成功 ✓

跑法：``.venv\\Scripts\\python.exe tests\\text_stream_test.py``（退出码 0 = 全绿 ✓）
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="textstream_"))
sys.stdout.reconfigure(encoding="utf-8")  # noqa: F821（cp936 控制台下首行 ✓ 会 UnicodeEncodeError ⇒ 0 判据就退 1 ✗）

_RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: Any = "") -> bool:
    _RESULTS.append((bool(ok), name, "" if ok else str(detail)[:400]))
    return bool(ok)


def summary(label: str) -> int:
    failed = [item for item in _RESULTS if not item[0]]
    for ok, name, detail in failed:
        print(f"FAIL  {name}  {detail}")
    # ⚠️ 收尾行必须是 ``SUMMARY: n/m passed`` 这个**字面量** ✓（``run_all`` 只认它 ✓；
    # 写成 ``SUMMARY[text_stream]: …`` 会让本套在全量表里**项数恒为 0** ✗，见 ``structure_test`` ㊈ ✓）
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    print(f"  （{label}）")
    return 1 if failed else 0


# ══════════════════════ 1. 协议层

def case_sse_frame() -> None:
    from app.routers.text import _compact, _sse_frame

    frame = _sse_frame("content", _compact({"text": "你\n好"}))
    check("SSE 帧以 event: 开头", frame.startswith("event: content\n"), frame[:20])
    check("SSE 帧以空行收尾（否则浏览器不 flush ✗）", frame.endswith("\n\n"), repr(frame[-4:]))
    check("SSE data 是紧凑 JSON（无空格分隔符 ✓）", '{"text":"你\\n好"}' in frame, frame)
    check("紧凑 JSON 里中文不转义（否则前端拿到 \\uXXXX ✓✗）", "你" in frame, frame)


def case_ollama_parse() -> None:
    from app.services.adapters.registry import get_text_adapter

    adapter = get_text_adapter("ollama")
    ev = adapter.parse_stream_line(
        '{"message":{"role":"assistant","content":"你"},"done":false}')
    check("ollama 分块：content 取到", ev["content"] == "你" and not ev["done"], ev)
    check("ollama 分块：thinking 不被塞进 content（分栏 ✓）", ev["thinking"] == "", ev)

    ev = adapter.parse_stream_line(
        '{"message":{"role":"assistant","content":"","thinking":"先想一下"},"done":false}')
    check("ollama 分块：thinking 单独成栏", ev["thinking"] == "先想一下" and ev["content"] == "", ev)

    ev = adapter.parse_stream_line(
        '{"done":true,"prompt_eval_count":33,"eval_count":12}')
    check("ollama 收尾块：done=True", ev["done"] is True, ev)
    check("ollama 收尾块：usage 归一成驼峰（与 agent 域同口径 ✓）",
          ev["usage"] == {"promptTokens": 33, "completionTokens": 12}, ev["usage"])

    ev = adapter.parse_stream_line('{"error":"model requires more system memory"}')
    check("ollama 错误块：error 透出（不许当成空分块吞掉 ✗）",
          ev["error"] == "model requires more system memory", ev)

    ev = adapter.parse_stream_line("")
    check("ollama 空行：空事件且不报错", ev == {"thinking": "", "content": "", "done": False,
                                               "usage": None, "error": ""}, ev)

    ev = adapter.parse_stream_line("{坏 json")
    check("ollama 坏行：报错而不是抛异常（一行坏不该终止整条流 ✓）",
          bool(ev["error"]) and not ev["done"], ev)


def case_openai_parse() -> None:
    from app.services.adapters.registry import get_text_adapter

    adapter = get_text_adapter("openai")
    ev = adapter.parse_stream_line('data: {"choices":[{"delta":{"content":"你"}}]}')
    check("openai 分块：剥 data: 前缀后取 delta.content", ev["content"] == "你", ev)

    ev = adapter.parse_stream_line(
        'data: {"choices":[{"delta":{"reasoning_content":"推理中"}}]}')
    check("openai 分块：reasoning_content 归 thinking（别当正文 ✗）",
          ev["thinking"] == "推理中" and ev["content"] == "", ev)

    ev = adapter.parse_stream_line(
        'data: {"choices":[{"delta":{"reasoning":"另一家的字段名"}}]}')
    check("openai 分块：reasoning 也认（字段名各家不一 ✓）", ev["thinking"] == "另一家的字段名", ev)

    ev = adapter.parse_stream_line("data: [DONE]")
    check("openai 终块：[DONE] ⇒ done", ev["done"] is True, ev)

    ev = adapter.parse_stream_line(": ping")
    check("openai 心跳注释行：空事件不报错", ev["done"] is False and not ev["error"], ev)

    ev = adapter.parse_stream_line('data: {"error":{"message":"rate limited"}}')
    check("openai 错误对象：取 message", ev["error"] == "rate limited", ev)

    ev = adapter.parse_stream_line(
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":5,"completion_tokens":7}}')
    check("openai 收尾：finish_reason ⇒ done", ev["done"] is True, ev)
    check("openai 收尾：usage 归一", ev["usage"] == {"promptTokens": 5, "completionTokens": 7}, ev)


# ══════════════════════ 2. 构造层

def case_build_stream_request() -> None:
    from app.services.adapters.registry import get_text_adapter

    config = {"provider": "ollama", "baseUrl": "http://127.0.0.1:11434", "settings": {}}
    params = {"model": "qwen3:14b", "messages": [{"role": "user", "content": "hi"}], "stop": ["\n\n"]}

    plain = get_text_adapter("ollama").build_request(config, params)
    stream = get_text_adapter("ollama").build_stream_request(config, params)
    check("ollama 流式端点仍走 /api/chat（不许漂到 OpenAI 兼容口 ✗）",
          stream["url"].endswith("/api/chat"), stream["url"])
    check("ollama 流式：think=False 保留（思考内容会灌满正文 ✗）",
          stream["body"].get("think") is False, stream["body"].get("think"))
    check("ollama 流式：stream=True", stream["body"].get("stream") is True, stream["body"].get("stream"))
    check("ollama 流式：stop 落进 options.stop",
          stream["body"]["options"].get("stop") == ["\n\n"], stream["body"]["options"])
    # 口径：两侧同时剔除 stream / options —— 非流式 body 里带 stream:False，流式带 stream:True，
    # 这是**唯一**应有的差别；其余字段逐字相同才说明"流式没有偷偷改别的参数"。
    check("ollama 流式：除 stream/stop 外与非流式逐字相同",
          {k: v for k, v in stream["body"].items() if k not in ("stream", "options")}
          == {k: v for k, v in plain["body"].items() if k not in ("stream", "options")},
          (plain["body"], stream["body"]))

    oai_stream = get_text_adapter("openai").build_stream_request(
        {"provider": "openai", "baseUrl": "http://127.0.0.1:11434/v1", "settings": {}, "apiKey": "k"}, params)
    check("openai 流式：stream=True + stream_options.include_usage",
          oai_stream["body"].get("stream") is True
          and (oai_stream["body"].get("stream_options") or {}).get("include_usage") is True,
          oai_stream["body"])
    check("openai 流式：stop 落到顶层 stop", oai_stream["body"].get("stop") == ["\n\n"], oai_stream["body"].get("stop"))


# ══════════════════════ 3. 增量层（自研引擎）

class _StubHub:
    """词表替身 ✓：id → 一段字符串（含**跨 token 才成形**的多字节字符 ✓）。"""

    def __init__(self, table: dict[int, str]) -> None:
        self.table = table

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self.table.get(int(i), "") for i in ids)


def _engine_backend(table: dict[int, str], script: list[int]) -> Any:
    from app.services.engine import llm as llm_mod
    from app.services.engine import llm_backend as backend_mod

    backend = backend_mod.LlmBackend(llm_mod.LlmConfig(
        vocab_size=8, hidden=8, depth=1, heads=1, kv_heads=1, head_dim=8, ffn=16, eos_id=None))
    backend._model = object()
    backend._hub = _StubHub(table)
    backend._device = "cpu"

    def fake_generate(model, ids, *, on_token=None, **kwargs):
        for token_id in script:
            if on_token is not None:
                on_token(token_id)
        return None

    saved_generate, saved_device = llm_mod.generate_llm, llm_mod.model_device
    llm_mod.generate_llm = fake_generate
    llm_mod.model_device = lambda _model: "cpu"
    return backend, (saved_generate, saved_device)


def _restore(saved: tuple[Any, Any]) -> None:
    from app.services.engine import llm as llm_mod
    llm_mod.generate_llm, llm_mod.model_device = saved


def case_engine_incremental() -> None:
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        print("SKIP  引擎增量判据（本环境无 torch ⇒ 造不出 ids 张量，**不是通过**）")
        return

    # id 1→"你"，2→""（半字节 ⇒ 解码不变 ⇒ 这一轮不该发），3→"好"：验证"前缀不稳就不发 ✓"
    table = {1: "你", 2: "", 3: "好"}
    backend, saved = _engine_backend(table, [1, 2, 3])
    events: list[tuple[str, dict[str, Any]]] = []
    try:
        text = backend.generate_stream("hi", max_new_tokens=3,
                                       on_event=lambda name, data: events.append((name, data)))
    finally:
        _restore(saved)
    deltas = [data.get("text") for name, data in events if name == "delta"]
    check("引擎流式：返回值是整段", text == "你好", text)
    check("引擎流式：增量拼起来 == 整段（多一字少一字都算错 ✗）", "".join(deltas) == "你好", deltas)
    check("引擎流式：空增量不发事件（别让前端收到空帧 ✗）", all(d for d in deltas), deltas)

    # 停用词剪裁：脚本吐 "一"、"二"、"STOP"、"三" ⇒ 只许发出 "一二"
    table2 = {1: "一", 2: "二", 3: "STOP", 4: "三"}
    backend, saved = _engine_backend(table2, [1, 2, 3, 4])
    events2: list[tuple[str, dict[str, Any]]] = []
    try:
        text = backend.generate_stream("hi", max_new_tokens=4, stop=["STOP"],
                                       on_event=lambda name, data: events2.append((name, data)))
    finally:
        _restore(saved)
    joined = "".join(data.get("text") or "" for name, data in events2 if name == "delta")
    check("引擎流式：停用词命中即止（后面那 token 不许漏出去 ✗）", "STOP" not in joined and "三" not in joined, joined)
    check("引擎流式：返回值已剪掉停用词", text == "一二", text)


def case_engine_abort() -> None:
    """``abort`` 事件 ✓：客户端断开 ⇒ 采样循环**当步**跳出 ✓（不是跑完 max_new_tokens 才停 ✗）。

    ⚠️ 这是「取消任务 ≠ 停止推理」那个坑的真判据 ✗✗：``asyncio`` 侧 ``task.cancel()`` 只能停掉
    await ✓，线程里的同步采样循环照跑 ✗ ⇒ 必须由 Event 在**循环内部**看一眼 ✓。
    """
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        print("SKIP  引擎 abort 判据（本环境无 torch ⇒ 造不出 ids 张量，**不是通过**）")
        return

    import threading

    # ① 一开始就置位 ⇒ 一个 token 都不许吐（连 new_ids 都不许进 ✓）
    table = {1: "一", 2: "二", 3: "三"}
    backend, saved = _engine_backend(table, [1, 2, 3])
    events: list[tuple[str, dict[str, Any]]] = []
    abort = threading.Event()
    abort.set()
    try:
        text = backend.generate_stream("hi", max_new_tokens=3, abort=abort,
                                       on_event=lambda n, d: events.append((n, d)))
    finally:
        _restore(saved)
    check("引擎 abort：已置位 ⇒ 一个增量都不发", events == [], events)
    check("引擎 abort：返回空串（不许补发任何字 ✗）", text == "", text)

    # ② 流到一半置位 ⇒ 只发出置位前那一段，之后立刻停 ✓
    backend, saved = _engine_backend(table, [1, 2, 3])
    events2: list[tuple[str, dict[str, Any]]] = []
    abort2 = threading.Event()

    def _on_event(name: str, data: dict[str, Any]) -> None:
        events2.append((name, data))
        abort2.set()          # 第一个增量到手就"客户端断开" ✓

    try:
        text2 = backend.generate_stream("hi", max_new_tokens=3, abort=abort2, on_event=_on_event)
    finally:
        _restore(saved)
    deltas = [d.get("text") for n, d in events2 if n == "delta"]
    check("引擎 abort：中途置位 ⇒ 只发出置位前那段（后面不许再补 ✗）", deltas == ["一"], deltas)
    check("引擎 abort：中途置位的返回值就是已生成那段", text2 == "一", text2)



# ══════════════════════ 4. 编排层

def case_stream_text_engine() -> None:
    from app.services import text_generation as gen_mod
    from app.services import text_stream as stream_mod

    class _Backend:
        def generate_stream(self, prompt, *, on_event=None, **kwargs):
            for chunk in ("拆", "成", "两段"):
                on_event("delta", {"text": chunk})
            return "拆成两段"

    saved_cfg = stream_mod.get_text_config
    saved_backend = gen_mod._get_engine_backend
    stream_mod.get_text_config = lambda _conn: {
        "provider": "engine", "settings": {"ggufPath": "stub.gguf", "model": "tiny-gguf"}}
    gen_mod._get_engine_backend = lambda *a, **k: _Backend()  # noqa: ARG005
    events: list[tuple[str, dict[str, Any]]] = []
    try:
        result = asyncio.run(stream_mod.stream_text(None, "hi", {"maxTokens": 8},
                                                    lambda n, d: events.append((n, d))))
    finally:
        stream_mod.get_text_config = saved_cfg
        gen_mod._get_engine_backend = saved_backend

    names = [name for name, _ in events]
    check("编排：以 meta 开头", names[0] == "meta", names[:3])
    check("编排：以 done 收尾（前端靠它关 loading ✓）", names[-1] == "done", names[-3:])
    check("编排：正文分成 3 个 content 事件（不是整段一次 ✗）",
          names.count("content") == 3, names)
    check("编排：done 携带整段文本与模型名",
          events[-1][1].get("text") == "拆成两段" and events[-1][1].get("model") == "tiny-gguf",
          events[-1][1])
    check("编排：首 token 延迟已记录（没记录就没法验证优化 ✗）",
          isinstance(result["firstTokenMs"], int) and result["firstTokenMs"] >= 0, result)
    check("编排：streamed=True（引擎路是真流式 ✓）", result["streamed"] is True, result)

    class _Empty:
        def generate_stream(self, prompt, *, on_event=None, **kwargs):
            return ""

    stream_mod.get_text_config = lambda _conn: {
        "provider": "engine", "settings": {"ggufPath": "stub.gguf"}}
    gen_mod._get_engine_backend = lambda *a, **k: _Empty()  # noqa: ARG005
    raised = ""
    try:
        asyncio.run(stream_mod.stream_text(None, "hi", {}, lambda n, d: None))
    except Exception as err:  # noqa: BLE001
        raised = str(err)
    finally:
        stream_mod.get_text_config = saved_cfg
        gen_mod._get_engine_backend = saved_backend
    check("编排：空输出**抛错**而不是静默返回空串（这就是「拿不到文本」那个老坑 ✗）",
          "为空" in raised, raised)

    # 老/替身后端**不收** abort ⇒ 不许因为"多送了一个 kwarg"把正常流式弄崩 ✗✗
    class _OldStyle:
        # 故意照**加 abort 之前**的 LlmBackend 签名写：有 system/stop/on_event ✓，没有 abort ✗、也没有 **kwargs ✗
        def generate_stream(self, prompt, *, system=None, temperature=1.0, max_new_tokens=256,
                            top_p=1.0, top_k=0, stop=(), on_event=None):
            on_event("delta", {"text": "老"})
            return "老"

    stream_mod.get_text_config = lambda _conn: {
        "provider": "engine", "settings": {"ggufPath": "stub.gguf"}}
    gen_mod._get_engine_backend = lambda *a, **k: _OldStyle()  # noqa: ARG005
    old_result: dict[str, Any] = {}
    old_err = ""
    try:
        old_result = asyncio.run(stream_mod.stream_text(None, "hi", {}, lambda n, d: None))
    except Exception as err:  # noqa: BLE001
        old_err = f"{type(err).__name__}: {err}"
    finally:
        stream_mod.get_text_config = saved_cfg
        gen_mod._get_engine_backend = saved_backend
    check("编排：后端不收 abort ⇒ **不送**这个 kwarg（不能因为取消功能把流式弄崩 ✗✗）",
          not old_err and (old_result or {}).get("text") == "老", old_err or old_result)


def case_cancel_sets_abort() -> None:
    """取消任务 ⇒ 必须把 ``abort`` 事件置位 ✓（否则线程把 max_new_tokens 烧完 ✗✗）。

    这里用「假后端一直等到 abort 置位才返回」来判 ✓：只在 ``asyncio`` 侧 cancel 而没置位的话，
    这个假后端会一直跑到轮询上限 ⇒ 判据落红 ✓（正是真机上「客户端走了 CPU 还在满转」的复现 ✓）。
    """
    from app.services import text_generation as gen_mod
    from app.services import text_stream as stream_mod

    holder: dict[str, Any] = {"abort": None, "stopped": False, "why": ""}

    class _Blocking:
        def generate_stream(self, prompt, *, on_event=None, abort=None, **kwargs):
            holder["abort"] = abort
            for _ in range(3000):            # 最多 ~3 秒（步进 1ms ✓）
                if abort is not None and abort.is_set():
                    holder["stopped"] = True
                    return ""
                time.sleep(0.001)
            holder["why"] = "abort 一直没被置位 ⇒ 采样循环跑满上限"
            return ""

    async def _scenario():
        task = asyncio.create_task(stream_mod.stream_text(None, "hi", {}, lambda n, d: None))
        await asyncio.sleep(0.05)            # 让它先进到 to_thread ✓
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        for _ in range(200):                 # 等线程看到事件 ✓（最多 ~2 秒）
            if holder["stopped"]:
                break
            await asyncio.sleep(0.01)

    saved_cfg = stream_mod.get_text_config
    saved_backend = gen_mod._get_engine_backend
    stream_mod.get_text_config = lambda _conn: {
        "provider": "engine", "settings": {"ggufPath": "stub.gguf"}}
    gen_mod._get_engine_backend = lambda *a, **k: _Blocking()  # noqa: ARG005
    try:
        asyncio.run(_scenario())
    finally:
        stream_mod.get_text_config = saved_cfg
        gen_mod._get_engine_backend = saved_backend

    check("编排：取消任务时把 abort 事件传给了引擎（没有它就只能跑满 token ✗✗）",
          holder["abort"] is not None, holder)
    check("编排：取消后引擎侧真的停下来了（不是「cancel 完线程照跑」✗）",
          holder["stopped"], holder["why"] or holder)


class _FakeResponse:
    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


def _fake_client(lines: list[str]):
    class _Client:
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, content=None, headers=None):  # noqa: ARG002
            return _FakeStream(_FakeResponse(lines))

    return _Client


def case_stream_text_ollama() -> None:
    from app.services import text_stream as stream_mod

    lines = [
        json.dumps({"message": {"role": "assistant", "content": "", "thinking": "想一想"},
                    "done": False}, ensure_ascii=False),
        json.dumps({"message": {"role": "assistant", "content": "收到"}, "done": False},
                   ensure_ascii=False),
        json.dumps({"message": {"role": "assistant", "content": "。"}, "done": False},
                   ensure_ascii=False),
        json.dumps({"done": True, "prompt_eval_count": 9, "eval_count": 2}, ensure_ascii=False),
    ]
    saved_cfg, saved_client = stream_mod.get_text_config, stream_mod.httpx.AsyncClient
    stream_mod.get_text_config = lambda _conn: {
        "provider": "ollama", "baseUrl": "http://127.0.0.1:11434",
        "models": ["qwen3:14b"], "settings": {}}
    stream_mod.httpx.AsyncClient = _fake_client(lines)
    events: list[tuple[str, dict[str, Any]]] = []
    try:
        result = asyncio.run(stream_mod.stream_text(None, "只回复收到", {},
                                                    lambda n, d: events.append((n, d))))
    finally:
        stream_mod.get_text_config = saved_cfg
        stream_mod.httpx.AsyncClient = saved_client

    thinking = "".join(d.get("text") or "" for n, d in events if n == "thinking")
    content = "".join(d.get("text") or "" for n, d in events if n == "content")
    check("ollama 流式：思考进 thinking 通道（不进正文 ✗✗）", thinking == "想一想", thinking)
    check("ollama 流式：正文只含回答", content == "收到。", content)
    check("ollama 流式：done 汇总 usage（流式也要能计费 ✓）",
          (result.get("usage") or {}).get("completionTokens") == 2, result.get("usage"))
    check("ollama 流式：streamed=True", result["streamed"] is True, result["streamed"])

    # 空流（只有 done）⇒ 必须报错，不许当成功
    saved_cfg, saved_client = stream_mod.get_text_config, stream_mod.httpx.AsyncClient
    stream_mod.get_text_config = lambda _conn: {
        "provider": "ollama", "baseUrl": "http://127.0.0.1:11434",
        "models": ["qwen3:14b"], "settings": {}}
    stream_mod.httpx.AsyncClient = _fake_client([json.dumps({"done": True})])
    raised = ""
    try:
        asyncio.run(stream_mod.stream_text(None, "hi", {}, lambda n, d: None))
    except Exception as err:  # noqa: BLE001
        raised = str(err)
    finally:
        stream_mod.get_text_config = saved_cfg
        stream_mod.httpx.AsyncClient = saved_client
    check("ollama 流式：空正文 ⇒ 抛错（否则前端显示「成功了但没字」✗）", "为空" in raised, raised)


def case_router_dispatches() -> None:
    """路由登记（不是起服务 ✓）：两个端点都在 ``/api/v1/text`` 下 ✓，且 SSE 端点声明了流式响应 ✓。

    ⚠️ 别用 ``app.routes`` 判登记：FastAPI 0.162 起 ``include_router`` 是**惰性**的 ——
    ``app.routes`` 里只有 ``_IncludedRouter`` 占位（``path`` 为 None），子路由要到请求时才展开。
    本仓既有口径统一走 ``app.openapi()["paths"]``（见 ``smoke_test`` / ``route_parity_test``）。
    """
    from app.main import app

    paths = app.openapi()["paths"]
    check("路由：POST /api/v1/text/stream 已登记",
          "post" in (paths.get("/api/v1/text/stream") or {}), sorted(paths)[:5])
    check("路由：POST /api/v1/text/generate 已登记",
          "post" in (paths.get("/api/v1/text/generate") or {}), sorted(paths)[:5])

    from fastapi.responses import StreamingResponse
    from fastapi.routing import APIRoute

    from app.routers.text import router as text_router

    stream_route = next((r for r in text_router.routes
                         if isinstance(r, APIRoute) and r.path == "/api/v1/text/stream"), None)
    # ⚠️ ``route.response_class`` 在 FastAPI 0.162 里默认是 ``DefaultPlaceholder``（惰性默认 ✓），
    # 比字符串会永远不等 ⇒ 判据分两半：显式声明的那半（StreamingResponse）＋ openapi 里的 media type ✓
    content = (((paths.get("/api/v1/text/stream") or {}).get("post") or {})
               .get("responses", {}).get("200", {}).get("content") or {})
    check("路由：stream 端点响应声明为 SSE",
          stream_route is not None
          and getattr(stream_route, "response_class", None) is StreamingResponse
          and "text/event-stream" in content,
          (getattr(stream_route, "response_class", None), sorted(content)))


def _frames(raw: str) -> list[tuple[str, Any]]:
    """把 SSE 原文切成 ``[(event, data)]`` ✓（注释帧 ``: ping`` 直接跳过 ✓）。"""
    out: list[tuple[str, Any]] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        name, payload = "", ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                payload += line[len("data: "):]
        if name:
            out.append((name, json.loads(payload) if payload else None))
    return out


def case_route_stream_e2e() -> None:
    """HTTP 层（真起 TestClient ✓）：队列中继 / 帧格式 / 收尾 / 失败也要落到 ``error`` 帧 ✓。

    ⚠️ 只打桩 ``stream_text``（**服务层**）✗ —— 路由自己的那段 ``asyncio.Queue`` + ``call_soon_threadsafe``
    + ``_sse_frame`` 是**本轮新写的** ✓，不跑一遍就等于没测 ✗（真缺陷正是在这一层：工作线程直接
    ``queue.put_nowait`` 会不唤醒事件循环 ⇒ 前端「卡住」✗）。
    """
    from fastapi.testclient import TestClient

    import app.routers.text as router_mod
    from app.main import app

    async def fake_ok(_conn, prompt, options, emit):
        emit("meta", {"model": "stub", "streamed": True})
        emit("thinking", {"text": "想"})
        emit("content", {"text": "你"})
        emit("content", {"text": "好"})
        emit("usage", {"completionTokens": 2})
        emit("done", {"text": "你好", "thinking": "想", "usage": {"completionTokens": 2}})
        return {"text": "你好"}

    async def fake_fail(_conn, prompt, options, emit):
        raise RuntimeError("模型没装")

    async def fake_fail_after_error(_conn, prompt, options, emit):
        # 真实路径：``stream_text`` 抛错前**已经** ``_push("error")`` 过一次 ✓
        emit("error", {"message": "显存不足"})
        raise RuntimeError("显存不足")

    saved = router_mod.stream_text
    try:
        router_mod.stream_text = fake_ok
        with TestClient(app) as client:
            with client.stream("POST", "/api/v1/text/stream", json={"prompt": "hi"}) as resp:
                status = resp.status_code
                ctype = resp.headers.get("content-type", "")
                raw = "".join(resp.iter_text())
        frames = _frames(raw)
        check("HTTP 流式：200 + `text/event-stream`（不许被缓冲成一次性 JSON ✗）",
              status == 200 and ctype.startswith("text/event-stream"), (status, ctype))
        check("HTTP 流式：事件顺序 meta→thinking→content→usage→done（思考**排在正文前** ✓）",
              [n for n, _ in frames] == ["meta", "thinking", "content", "content", "usage", "done"],
              [n for n, _ in frames])
        check("HTTP 流式：正文按帧拼回原样（增量不许丢字 ✗）",
              "".join(str(d.get("text") or "") for n, d in frames if n == "content") == "你好",
              [d for n, d in frames if n == "content"])
        check("HTTP 流式：收尾帧带全文与用量（前端不必自己攒 ✓）",
              next((d for n, d in frames if n == "done") or {}).get("usage") == {"completionTokens": 2},
              next((d for n, d in frames if n == "done"), None))

        router_mod.stream_text = fake_fail
        with TestClient(app) as client:
            with client.stream("POST", "/api/v1/text/stream", json={"prompt": "hi"}) as resp:
                raw2 = "".join(resp.iter_text())
        frames2 = _frames(raw2)
        check("HTTP 流式：推理炸了 ⇒ **最后**是 error 帧（状态码早发出去了，改不了 200 ✗）",
              bool(frames2) and frames2[-1][0] == "error"
              and "模型没装" in str((frames2[-1][1] or {}).get("message")),
              frames2[-1] if frames2 else frames2)

        router_mod.stream_text = fake_fail_after_error
        with TestClient(app) as client:
            with client.stream("POST", "/api/v1/text/stream", json={"prompt": "hi"}) as resp:
                raw3 = "".join(resp.iter_text())
        frames3 = _frames(raw3)
        errors3 = [d for n, d in frames3 if n == "error"]
        check("HTTP 流式：服务层已发 error ⇒ 路由**不再补发**（同一句报错上屏两次 ✗✗）",
              len(errors3) == 1 and "显存不足" in str((errors3[0] or {}).get("message")), errors3)
    finally:
        router_mod.stream_text = saved


def main() -> int:
    case_sse_frame()
    case_ollama_parse()
    case_openai_parse()
    case_build_stream_request()
    case_engine_incremental()
    case_engine_abort()
    case_stream_text_engine()
    case_cancel_sets_abort()
    case_stream_text_ollama()
    case_router_dispatches()
    case_route_stream_e2e()
    return summary("text_stream")


if __name__ == "__main__":
    sys.exit(main())
