"""自检：**自研引擎当 agent 的模型**（``provider="engine"`` ✓ 2026-11-07 补 ✓）。

为什么非要有这一套 ✗：``text_generation`` 早就支持 ``provider="engine"`` ✓，但那是**一问一答** ✓；
agent 要的是「调工具 → 回灌结果 → 再想一步」✓✗。缺这一跳时，本仓自己下载的 GGUF
**只能当聊天框**：``/tune``、``/evaluation/run`` 全都打不上去 ✓✗，而报错长得像「没有模型」✓✗。

判据（都是「看着通了、其实协议没真跑」的形状 ✓✗）：

1. ``parse_engine_tool_calls``：**没有块 ⇒ 空** ✓ / 一块 / 两块 / 参数是嵌套对象**不切坏** ✓ /
   参数是 JSON 字符串也吃 ✓；
2. 块在但 JSON 坏、或缺 ``name`` ⇒ **抛** ✗✗（静默丢弃 ⇒ 模型以为调用成功、干等结果 ✓✗）；
3. 骨架没闭合标签（生成被截断）⇒ **不算调用** ✓（不是抛 ✓）；
4. ``_normalize_messages`` **不许**把 ``tool.name`` / ``assistant.tool_calls`` 压掉 ✗✗
   （压掉 ⇒ 模型看不见自己调过工具、反复调同一个 ✓✗，且不报错 ✓✗）；
5. ``_guard_context`` 超窗**当场拒** ✗；权重没声明窗口 ⇒ 不检查 ✓ 不猜 ✗；
6. 权重**没有 chat 模板** ⇒ ``generate_messages`` 当面拒 ✗（不拼裸文本冒充 ✓）；
7. ``_run_engine_tool_loop`` 端到端（stub 后端 ✓）：第一轮出 ``<tool_call>`` ⇒ 工具**真被执行** ⇒
   第二轮的消息里**真的有** ``role="tool"`` 那条 ⇒ 收敛 ✓；
8. ``default_generate`` 的 ``provider="engine"`` **必须在本函数里分派** ✗✗
   （走 ``get_text_adapter`` ⇒ ``UnknownProviderError`` ⇒ 本仓模型永远用不上 ✓✗）；
9. 缺 ``settings.ggufPath`` ⇒ **明确报缺** ✓（不静默用某个默认模型 ✗）。

运行::

    ./.venv/Scripts/python.exe tests/engine_agent_loop_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="agentloop_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ⚠️ cp936 控制台下 ``✓`` 直接 UnicodeEncodeError ⇒ 0 条判据就退 1 ✓✗（单独跑才露头 ✗）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:      # noqa: BLE001 —— 老版本/被重定向过就算了 ✓
        pass

from app.agent import runtime  # noqa: E402
from app.agent.tool import Tool, ToolRegistry  # noqa: E402
from app.services import text_generation  # noqa: E402
from app.services.engine import chat_template, llm, llm_backend  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


CONFIG = llm.LlmConfig(vocab_size=32, hidden=16, depth=1, heads=2, kv_heads=1,
                       head_dim=8, ffn=32, eos_id=31)


class _StubHub:
    """stub 词表 ✓（记录调用 ✓ —— 不读真词表 ✗）。"""

    special_tokens = {"bos_token": "<s>", "eos_token": "</s>"}

    def __init__(self) -> None:
        self.encoded: list[list[int]] = []

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        self.encoded.append(list(text.encode("utf-8")))
        return [b % CONFIG.vocab_size for b in text.encode("utf-8")] or [0]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return f"DECODED({len(ids)})"


def raises(exc: type[BaseException], func, *args, **kwargs) -> BaseException | None:
    """调用 ⇒ 返回那个异常 ✓；没抛 ⇒ ``None`` ✓（判据里好用 ✓）。"""
    try:
        func(*args, **kwargs)
    except exc as err:            # noqa: BLE001 —— 就要那个类型 ✓
        return err
    except BaseException as err:  # noqa: BLE001
        return err
    return None


# ===================================================================== ① 解析
def case_parse() -> None:
    parse = runtime.parse_engine_tool_calls

    check("①a 没有 <tool_call> 块 ⇒ 空（= 这轮没调工具，不是错误）✓",
          parse("就是一段普通回复，没有工具") == [])

    one = parse('<tool_call>\n{"name": "save_script", "arguments": {"text": "hi"}}\n</tool_call>'
                "\n写完收工")
    check("①b 单块 ⇒ 名字与参数都解出来 ✓",
          one == [{"function": {"name": "save_script", "arguments": {"text": "hi"}}}], one)

    nested = parse('<tool_call>{"name":"x","arguments":{"a":{"b":1},"c":[1,2]}}</tool_call>')
    check("①c 参数是**嵌套对象/数组**也不许被切坏 ✓（靠闭合标签定界，不靠花括号配对）",
          nested == [{"function": {"name": "x", "arguments": {"a": {"b": 1}, "c": [1, 2]}}}], nested)

    as_str = parse('<tool_call>{"name":"x","arguments":"{\\"a\\": 2}"}</tool_call>')
    check("①d 参数是 **JSON 字符串**（Ollama 那套写法）也要吃得下 ✓",
          as_str == [{"function": {"name": "x", "arguments": {"a": 2}}}], as_str)

    two = parse('<tool_call>{"name":"a","arguments":{}}</tool_call>'
                '<tool_call>{"name":"b","arguments":{}}</tool_call>')
    check("①e 一轮写多个块 ⇒ 都收下 ✓（顺序不许乱）",
          [c["function"]["name"] for c in two] == ["a", "b"], two)


def case_parse_bare() -> None:
    """⭐ 没写标签、**整段**就是一串调用对象 ⇒ 认 ✓（2026-11-07 真机撞出来的那口 ✓✗）。

    权重自带的 chat 模板明写「调用写成 ``<tool_call>...</tool_call>``」✓，本仓下载的
    ``Qwen3-4B-Instruct-2507`` 却只吐 ``{"name": "save_note", "arguments": {...}}`` ✗ ——
    按旧判据这轮算「没调工具」⇒ 工具不执行、``tool_results`` 空 ⇒ 探针 ``PROBE_NO_TOOL`` ✗✗。
    退路必须**极窄** ✓：整段只有这些对象 + 键只有那四个 + 工具名全真注册过 ✓，缺一条就退回「没调」✗。
    ⚠️ 而且必须是「一串」不是「一个」✗✗ —— 见 ①m：extractor 一口气吐了四个读工具 ✓。
    """
    parse = runtime.parse_engine_tool_calls
    names = {"save_note", "read_script"}

    bare = parse('{"name": "save_note", "arguments": {"text": "第一集"}}', tool_names=names)
    check("①f ⭐ 整段一个裸调用对象 + 名字真注册过 ⇒ **认**（真机小模型漏标签的唯一退路）✓",
          bare == [{"function": {"name": "save_note", "arguments": {"text": "第一集"}}}], bare)

    fenced = parse('```json\n{"name": "save_note", "arguments": {"text": "x"}}\n```',
                   tool_names=names)
    check("①g 外面包一层 ```json 围栏也认 ✓（仍是整段唯一对象 ✓）",
          [c["function"]["name"] for c in fenced] == ["save_note"], fenced)

    prose = parse('我来保存：{"name": "save_note", "arguments": {}} 好了', tool_names=names)
    check("①h ⚠️ 夹在散文里的同形状对象 ⇒ **不认**（那是举例/回显，认了就是凭空造一次副作用 ✗✗）",
          prose == [], prose)

    unknown = parse('{"name": "rm_rf_root", "arguments": {}}', tool_names=names)
    check("①i ⚠️ 名字**没注册过** ⇒ 当散文 ✓（不认、也**不抛** ✗：一次幻觉不该毁掉整轮）",
          unknown == [], unknown)

    extra = parse('{"name": "save_note", "arguments": {}, "note": "多一个键"}', tool_names=names)
    check("①j 多出别的键 ⇒ 不是工具调用 ✓（键集是封闭的，认了就什么都能冒充调用 ✗）",
          extra == [], extra)

    strict = parse('{"name": "save_note", "arguments": {}}')
    check("①k ⚠️ 不给 ``tool_names`` ⇒ **不认**裸对象 ✗（解析器自己核不了「这名字是不是真工具」，"
          "没清单就宁可当散文 ✓）", strict == [], strict)

    trailing = parse('{"name": "save_note", "arguments": {}} 谢谢', tool_names=names)
    check("①l 对象后面还跟着话 ⇒ 不算「整段一个调用」✓", trailing == [], trailing)

    # ⚠️⚠️ ①m 起是**同日第二次真跑**才露头的那口 ✗✗：extractor 那跑模型一口气吐**四个**裸对象
    # （读剧本 + 读既有角色/场景/道具），旧写法只认「整段唯一对象」⇒ 一个都不认 ⇒ 读工具全没执行
    # ⇒ 模型永远走不到 save_* ⇒ ``/tune`` 四跑恒 0 分（**看起来像模型不行**，其实是解析层整串丢了 ✗）。
    real = ('{"name": "read_script", "arguments": {}}\n\n\n'
            '{"name": "save_note", "arguments": {"text": "a"}}\n\n'
            '{"name": "read_script", "arguments": {"episodeId": 11}}')
    many = parse(real, tool_names=names)
    check("①m ⭐⭐ 空行隔开的**一串**裸对象 ⇒ 全认、**顺序照原文** ✓（真机原文就长这样 ✓）",
          [c["function"]["name"] for c in many] == ["read_script", "save_note", "read_script"]
          and many[1]["function"]["arguments"] == {"text": "a"}, many)

    fenced_many = parse('```json\n{"name": "save_note", "arguments": {}}\n'
                        '{"name": "read_script", "arguments": {}}\n```', tool_names=names)
    check("①n 一串裸对象外面套 ```json 围栏也认 ✓（围栏只是壳 ✓）",
          [c["function"]["name"] for c in fenced_many] == ["save_note", "read_script"], fenced_many)

    mixed = parse('{"name": "save_note", "arguments": {}}\n\n'
                  '{"name": "rm_rf_root", "arguments": {}}', tool_names=names)
    check("①o ⚠️ 一串里**混着编造的名字** ⇒ **整串不认** ✗（只跑认得的几个＝替模型裁剪它没说过的调用 ✗✗）",
          mixed == [], mixed)

    broken = parse('{"name": "save_note", "arguments": {}}\n\n{"name": "read_script", oops}',
                   tool_names=names)
    check("①p 一串里有一个 JSON 坏 ⇒ 当散文 ✓（既不认、也不抛 ✗ —— 抛会把没标签的散文也变成致命错 ✗）",
          broken == [], broken)

    flood = "\n".join('{"name": "save_note", "arguments": {}}'
                      for _ in range(runtime._MAX_BARE_CALLS + 1))
    check("①q 超过上限的一长串 ⇒ 当散文 ✓（上百个基本是在刷格式 ✓✗，认了就是一次真跑上百个副作用 ✗✗）",
          parse(flood, tool_names=names) == [], len(flood))


def case_parse_rejects() -> None:
    parse = runtime.parse_engine_tool_calls

    bad = raises(ValueError, parse, '<tool_call>{"name": "x", "arguments": {</tool_call>')
    check("②a 块里 JSON 坏 ⇒ **抛**（静默丢弃＝这次调用凭空消失，模型还干等结果）✗✗",
          isinstance(bad, ValueError), bad)

    no_name = raises(ValueError, parse, '<tool_call>{"arguments": {"a": 1}}</tool_call>')
    check("②b 块里缺 name ⇒ **抛**（不然「调谁」全靠猜）✗✗",
          isinstance(no_name, ValueError), no_name)

    truncated = parse('<tool_call>{"name": "x", "arguments": {"a": 1}')
    check("③ 只有开标签、没有 </tool_call>（生成被截断）⇒ 不算调用 ✓（不是抛 ✓）",
          truncated == [], truncated)


# ===================================================================== ④ 消息字段
def case_message_fields() -> None:
    rows = chat_template._normalize_messages([
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "调了", "tool_calls": [{"function": {"name": "x"}}]},
        {"role": "tool", "tool_name": "x", "content": "结果"},
    ])
    check("④a assistant 的 tool_calls 原样留着 ✓（压掉 ⇒ 模型看不见自己调过什么）",
          rows[1].get("tool_calls") == [{"function": {"name": "x"}}], rows[1])
    check("④b tool 消息的 name 也留着 ✓（键名 Ollama 叫 tool_name 也认 ✓）",
          rows[2].get("name") == "x", rows[2])

    source = [{"role": "assistant", "content": "调了", "tool_calls": [{"function": {"name": "x"}}]}]
    rows = chat_template._normalize_messages(source)
    rows[0]["tool_calls"].append({"function": {"name": "y"}})
    check("④c 传出去的是**副本** ✓（改布局不许反向污染调用方那份历史）",
          len(source[0]["tool_calls"]) == 1, source)


# ===================================================================== ⑤⑥ 守门
def case_guards() -> None:
    backend = llm_backend.LlmBackend(CONFIG, gguf_path="nowhere.gguf")
    backend._model = object()
    backend._hub = _StubHub()

    backend._context_length = 100
    over = raises(llm.LlmError, backend._guard_context, 90, 20)
    check("⑤a 提示词 90 + 预留 20 > 窗口 100 ⇒ **当场拒**（不交给采样半路越界/静默截断）✗✗",
          isinstance(over, llm.LlmError), over)
    check("⑤b 恰好塞得下（90 + 10 = 100）⇒ 不拦 ✓",
          raises(llm.LlmError, backend._guard_context, 90, 10) is None)
    backend._context_length = 0
    check("⑤c 权重没声明窗口 ⇒ 不检查 ✓（不拿常数当窗口 ✗）",
          raises(llm.LlmError, backend._guard_context, 10 ** 6, 10 ** 6) is None)

    check("⑤d 窗口从元数据按**后缀**认 ✓（qwen3./llama. 前缀都行 ✓）",
          llm_backend._context_length_of({"qwen3.context_length": 32768}) == 32768
          and llm_backend._context_length_of({"llama.context_length": 8192}) == 8192)
    check("⑤e 0 / 负数 / bool / 无点前缀 一律当「没声明」✓（不猜 ✗，bool 是 int 子类要先挡 ✓）",
          [llm_backend._context_length_of(x) for x in (
              {"qwen3.context_length": 0}, {"qwen3.context_length": -5},
              {"qwen3.context_length": True}, {"context_length": 4096})] == [0, 0, 0, 0])

    backend._chat_template = None
    backend._chat_template_error = ""
    no_tpl = raises(chat_template.ChatTemplateError, backend.generate_messages,
                    [{"role": "user", "content": "hi"}])
    check("⑥ 权重没有 chat 模板 ⇒ 多轮生成**当面拒** ✗（不拼裸文本冒充模板 ✓）",
          isinstance(no_tpl, chat_template.ChatTemplateError), no_tpl)


# ===================================================================== ⑦ 端到端
class _StubBackend:
    """stub 引擎 ✓：第一轮吐工具调用、第二轮收工 ✓（不真跑前向 ✗）。"""

    name = "stub-engine"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def generate_messages(self, messages, *, tools=None, **kwargs) -> str:
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


def _registry(calls: list[dict]) -> ToolRegistry:
    def execute(arguments: dict) -> str:
        calls.append(arguments)
        return "已保存 ✓"

    return ToolRegistry({"save_script": Tool(
        id="save_script", description="保存剧本",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=execute,
    )})


def run_loop(stub: _StubBackend, *, config: dict | None = None, calls: list | None = None,
             max_steps: int = 3) -> dict:
    calls = [] if calls is None else calls
    original = text_generation.get_engine_backend
    text_generation.get_engine_backend = lambda *a, **k: stub   # type: ignore[assignment]
    try:
        return asyncio.run(runtime._run_engine_tool_loop(
            model="ignored", config=config or {"provider": "engine",
                                               "settings": {"ggufPath": "w.gguf"}},
            instructions="你是编剧", tools=_registry(calls), message="写一集",
            max_steps=max_steps, model_settings={"temperature": 0.3, "maxOutputTokens": 64},
        ))
    finally:
        text_generation.get_engine_backend = original   # type: ignore[assignment]


def case_loop() -> None:
    calls: list[dict] = []
    stub = _StubBackend([
        '<tool_call>{"name": "save_script", "arguments": {"text": "第一集"}}</tool_call>',
        "已经保存好了 ✓",
    ])
    result = run_loop(stub, calls=calls)

    check("⑦a 模型写的 <tool_call> ⇒ 工具**真被执行**（不是只解析就算）✓",
          calls == [{"text": "第一集"}], calls)
    check("⑦b 第二轮 messages 里**真的有** role=tool 那条 ✓（回灌断了 ⇒ 模型只能瞎猜结果）",
          any(m.get("role") == "tool" and m.get("name") == "save_script"
              for m in stub.seen[1]), stub.seen[1])
    check("⑦c assistant 那轮的 tool_calls 也回灌了 ✓（模型要知道自己调过什么）",
          any(m.get("role") == "assistant" and m.get("tool_calls")
              for m in stub.seen[1]), stub.seen[1])
    check("⑦d 最终文本＝模型收工那句 ✓（不是把工具调用块当正文）",
          result["text"] == "已经保存好了 ✓", result["text"])
    check("⑦e 记账：1 次调用 1 条结果 ✓（优化/评测按这个判「有没有干活」✓）",
          len(result["tool_calls"]) == 1 and len(result["tool_results"]) == 1, result)

    bad = _StubBackend(['<tool_call>{"name": "save_script", "arguments": {</tool_call>', "收工"])
    broken = raises(ValueError, run_loop, bad)
    check("⑦f 模型写坏参数 ⇒ 整轮**抛** ✗（不静默跳过 ⇒ 不许假装保存成功 ✓✗）",
          isinstance(broken, ValueError), broken)

    # ⭐ 真机形态（2026-11-07）：本仓下载的 Qwen3-4B 那次**标签没写**、正文就是一个调用对象 ✓✗
    calls_bare: list[dict] = []
    result_bare = run_loop(_StubBackend([
        '{"name": "save_script", "arguments": {"text": "第二集"}}',
        "好了 ✓",
    ]), calls=calls_bare)
    check("⑦g ⭐ 模型漏写 <tool_call> 标签、整段只有一个调用对象 ⇒ 工具**照样真执行** ✓"
          "（旧判据在这里算「这轮没调工具」⇒ 一条结果都没有 ✗✗）",
          calls_bare == [{"text": "第二集"}] and len(result_bare["tool_results"]) == 1,
          (calls_bare, result_bare["tool_results"]))

    missing = raises(ValueError, run_loop, _StubBackend(["收工"]),
                     config={"provider": "engine", "settings": {}})
    check("⑨ 缺 settings.ggufPath ⇒ 明确报缺 ✓（不静默拿某个默认模型顶上 ✗）",
          isinstance(missing, ValueError) and "ggufPath" in str(missing), missing)


# ===================================================================== ⑧ 分派
def case_dispatch() -> None:
    seen: list[dict] = []

    async def fake_loop(**kwargs) -> dict:
        seen.append(kwargs)
        return {"text": "走的是引擎 ✓", "thinking": "", "tool_calls": [],
                "tool_results": [], "steps": []}

    original = runtime._run_engine_tool_loop
    runtime._run_engine_tool_loop = fake_loop   # type: ignore[assignment]
    try:
        result = asyncio.run(runtime.default_generate(
            config={"provider": "engine", "settings": {"ggufPath": "w.gguf"}},
            model="ignored", instructions="i", tools=ToolRegistry({}), message="m",
            max_steps=1, model_settings={},
        ))
    finally:
        runtime._run_engine_tool_loop = original   # type: ignore[assignment]

    check("⑧ provider=engine **在 default_generate 里就分派** ✓"
          "（走 get_text_adapter ⇒ UnknownProviderError ⇒ 本仓模型永远用不上 ✗✗）",
          result.get("text") == "走的是引擎 ✓" and len(seen) == 1, result)

    unknown = raises(Exception, runtime.get_text_adapter, "engine")
    check("⑧b 反面：registry 里**确实没有** engine ⇒ 那一行分派是必需的（不是冗余）✓",
          unknown is not None, unknown)


def main() -> int:
    case_parse()
    case_parse_bare()
    case_parse_rejects()
    case_message_fields()
    case_guards()
    case_loop()
    case_dispatch()

    failed = [name for name, ok, _ in _RESULTS if not ok]
    for name, ok, detail in _RESULTS:
        print(f"{'✓' if ok else '✗'} {name}")
        if not ok:
            print(f"    {detail!r}")
    print(f"\nSUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
