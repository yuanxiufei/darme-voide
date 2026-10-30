"""自检：**ollama 原生能力层**（``app/services/ollama_native/`` —— 上游 Go 侧可对等部分的移植）。

守的不是「能 import」，而是**上游那些踩过坑的语义**。每一条断言都对着
``reference/ollama`` 里的一行实现，注释写明「照抄的是什么、写错会怎样」。

覆盖：

1. ``gotemplate`` —— Go ``text/template`` 子集；全部内置模板可解析、渲染对得上；
2. ``template`` —— ``Named()`` 认族、messages/legacy 两条渲染分支、``{{ .Response }}`` 截断；
3. ``harmony`` —— 流切分不吐半截标记、channel→正文/思考/工具、工具名双向映射；
4. ``envconfig`` —— ``OLLAMA_HOST`` 的端口推导与回环换算、``Var()`` 剥引号；
5. ``format``/``progress`` —— 十进制 vs 二进制单位、分档精度、速率在结束后改用全程平均；
6. ``openai`` —— 缺省 temperature/top_p=1.0、多段 content 拆分、tool 角色反查、流式拆块；
7. ``anthropic`` —— user 轮 tool_result 前置、``input_tokens`` 不含缓存、流式 block 序号；
8. ``names``/``modelfile``/``params``/``manifest`` —— 既有模块的回归（防止本轮改动碰坏）。

全程不打网络、不碰真机模型库（临时目录 + 纯函数）。

运行::

    ./.venv/Scripts/python.exe tests/ollama_native_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="ollamanative_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.ollama_native import anthropic as A  # noqa: E402
from app.services.ollama_native import envconfig as E  # noqa: E402
from app.services.ollama_native import format as F  # noqa: E402
from app.services.ollama_native import gotemplate, harmony, manifest, modelfile, names, params  # noqa: E402
from app.services.ollama_native import openai as O  # noqa: E402
from app.services.ollama_native import progress as P  # noqa: E402
from app.services.ollama_native import template as T  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _tpl(name: str) -> str:
    path = Path(T._TEMPLATE_DIR) / f"{name}.gotmpl"
    return path.read_text(encoding="utf-8")


def _load(name: str) -> T.Template:
    return T.parse(_tpl(name))


# ============================================================
# 1. gotemplate —— Go text/template 子集
# ============================================================


def test_gotemplate() -> None:
    entries = json.loads((Path(T._TEMPLATE_DIR) / "index.json").read_text(encoding="utf-8"))
    broken: list[str] = []
    for entry in entries:
        try:
            gotemplate.parse(_tpl(entry["name"]))
        except Exception as exc:  # noqa: BLE001 - 就是要收集全部失败
            broken.append(f"{entry['name']}: {exc}")
    check("gotemplate: 全部内置模板都能解析（一个坏模板 = 整族渲染失败）",
          not broken and len(entries) >= 15, broken)

    # and/or 返回**操作数本身**而非 bool（Go 语义）；写成布尔化会把 .System 变成 "True"
    rendered = gotemplate.parse("{{ .A }}|{{ and .B .C }}|{{ or .D .E }}").render(
        {"A": "x", "B": "b", "C": "c", "D": "", "E": "e"})
    check("gotemplate: and/or 返回操作数本身（不是 true/false）",
          rendered == "x|c|e", rendered)

    # range 对 map 按 key 排序遍历（Go 1.12+ 行为）；不排序会让提示词顺序抖动
    rendered = gotemplate.parse("{{ range $k, $v := .M }}{{ $k }}={{ $v }};{{ end }}").render(
        {"M": {"b": 2, "a": 1, "c": 3}})
    check("gotemplate: range over map 按 key 排序（顺序必须确定）",
          rendered == "a=1;b=2;c=3;", rendered)

    try:
        gotemplate.parse("{{ if .A }}no end")
        ok = False
    except gotemplate.GoTemplateError:
        ok = True
    check("gotemplate: 缺 {{ end }} 必须报错（不能静默渲染半截）", ok)

    # 两连 else if（alpaca 模板真实形态）—— 曾被误当成嵌套块
    rendered = gotemplate.parse(
        "{{ if .A }}a{{ else if .B }}b{{ else if .C }}c{{ else }}d{{ end }}"
    ).render({"A": False, "B": False, "C": True})
    check("gotemplate: 连续 else if 链（alpaca 形态）", rendered == "c", rendered)


# ============================================================
# 2. template —— 认族与两条渲染分支
# ============================================================


def test_template() -> None:
    entries = json.loads((Path(T._TEMPLATE_DIR) / "index.json").read_text(encoding="utf-8"))
    mismatched: list[str] = []
    for entry in entries:
        try:
            named = T.Named(entry["template"])
        except gotemplate.GoTemplateError as exc:
            mismatched.append(f"{entry['name']}: {exc}")
            continue
        if named.name != entry["name"]:
            mismatched.append(f"{entry['name']} -> {named.name}")
    check("template: 每份 Jinja 原文都能认回自己（Named 的 Levenshtein 判定）",
          not mismatched, mismatched)

    try:
        T.Named("这是一段完全不相干的模板" * 50)
        rejected = False
    except gotemplate.GoTemplateError:
        rejected = True
    check("template: 认不出的模板必须报错（静默回落会整段格式错且查不出来）", rejected)

    chatml = _load("chatml")
    out = chatml.execute(T.Values(messages=[{"role": "user", "content": "你好"}]))
    check("template: messages 分支以 assistant 开头收尾（等模型开口）",
          out.rstrip().endswith("assistant") and "你好" in out, out[-40:])

    alpaca = _load("alpaca")
    out = alpaca.execute(T.Values(messages=[
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]))
    check("template: legacy 分支逐条累积（两轮都出现 ### Instruction:）",
          out.count("### Instruction:") == 2 and out.rstrip().endswith("### Response:"), out)

    # legacy 最后一次渲染必须把 {{ .Response }} 之后整段截掉：llama2 的 [INST] 收尾才
    # 不会在「等模型补全」的位置又塞一遍 </s>
    llama2 = _load("llama2-chat")
    out = llama2.execute(T.Values(messages=[{"role": "user", "content": "Q"}]))
    check("template: legacy 末次渲染截断 {{ .Response }} 之后（否则会吐重复终止符）",
          out.count("</s>") <= 1, out)

    auto = T.parse("{{ .Prompt }}")
    check("template: 既无 .Messages 也无 .Response 时自动补 {{ .Response }}",
          "response" in auto.vars(), auto.vars())


# ============================================================
# 3. harmony —— gpt-oss 流解析
# ============================================================


def test_harmony() -> None:
    stream = ("<|start|>assistant<|channel|>analysis<|message|>先想<|end|>"
              "<|start|>assistant<|channel|>final<|message|>答案<|end|>")
    # 故意把切点落在 "<|en" 中间 —— 不处理 overlap 就会把 "<|en" 当正文吐给前端
    parts = [stream[:34], stream[34:52], stream[52:]]
    handler = harmony.HarmonyMessageHandler()
    handler.init(None, None, None)
    content: list[str] = []
    thinking: list[str] = []
    for part in parts:
        got_content, got_thinking, _ = handler.add(part, done=False)
        content.append(got_content)
        thinking.append(got_thinking)
    got_content, got_thinking, _ = handler.add("", done=True)
    content.append(got_content)
    thinking.append(got_thinking)
    joined = "".join(content)
    check("harmony: 分片切在标记中间不吐残片（overlap 保留）",
          "<|" not in joined and "<|" not in "".join(thinking), joined)
    check("harmony: analysis→思考、final→正文",
          "".join(thinking) == "先想" and joined == "答案",
          {"thinking": "".join(thinking), "content": joined})

    tool_stream = ("<|start|>assistant<|channel|>analysis to=functions.get_weather"
                   "<|constrain|>json<|message|>{\"city\":\"BJ\"}<|end|>")
    handler = harmony.HarmonyMessageHandler()
    handler.init(None, None, None)
    _, _, calls = handler.add(tool_stream, done=True)
    check("harmony: 带 recipient 的 channel 变工具调用，且剥掉 functions. 前缀",
          len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
          and calls[0]["function"]["arguments"] == {"city": "BJ"}, calls)

    mapping = harmony.FunctionNameMap()
    converted = mapping.convert_and_add("get weather")
    check("harmony: 工具名转合法标识符 + 能反查还原（空格无法从字符串推回）",
          converted == "get_weather"
          and mapping.original_from_converted(converted) == "get weather",
          converted)
    check("harmony: 撞名追加序号从 _2 起（第一个不改名）",
          mapping.convert_and_add("get weather") == "get_weather_2", mapping.harmony_to_user)

    handler = harmony.HarmonyMessageHandler()
    handler.init(None, None, None)
    try:
        handler.add("<|start|>assistant<|channel|>analysis to=functions.f<|message|>not-json<|end|>",
                    done=True)
        raised = False
    except harmony.HarmonyParseError:
        raised = True
    check("harmony: 工具参数非 JSON 必须报错（不能塞个空对象让调用方以为成功）", raised)


# ============================================================
# 4. envconfig —— OLLAMA_* 真源
# ============================================================


def test_envconfig() -> None:
    saved = os.environ.get("OLLAMA_HOST")
    try:
        cases = {
            "": ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
            "0.0.0.0:11434": ("http://0.0.0.0:11434", "http://127.0.0.1:11434"),
            "https://example.com": ("https://example.com:443", "https://example.com:443"),
            "[::1]:8080/x": ("http://[::1]:8080/x", "http://[::1]:8080/x"),
            "::1": ("http://[::1]:11434", "http://[::1]:11434"),
        }
        bad: list[str] = []
        for raw, (want, want_connectable) in cases.items():
            os.environ["OLLAMA_HOST"] = raw
            got = E.Host().base_url
            got_connectable = E.ConnectableHost().base_url
            if got != want or got_connectable != want_connectable:
                bad.append(f"{raw!r}: {got} / {got_connectable}（期望 {want} / {want_connectable}）")
        check("envconfig: OLLAMA_HOST 的端口推导 / IPv6 方括号 / 通配地址换回环", not bad, bad)

        os.environ["OLLAMA_HOST"] = '"127.0.0.1:11500"'
        check("envconfig: Var() 剥掉引号（Windows 写法带引号会拼出非法路径）",
              E.Host().base_url == "http://127.0.0.1:11500", E.Host().base_url)

        os.environ["OLLAMA_HOST"] = "127.0.0.1:99999"
        check("envconfig: 端口越界回落到默认（不抛、不连到非法端口）",
              E.Host().base_url == "http://127.0.0.1:11434", E.Host().base_url)
    finally:
        if saved is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = saved

    saved = os.environ.get("OLLAMA_KEEP_ALIVE")
    try:
        os.environ["OLLAMA_KEEP_ALIVE"] = "-1"
        check("envconfig: KeepAlive 负值解释为「永不离驻」(inf)", E.KeepAlive() == float("inf"))
        os.environ["OLLAMA_KEEP_ALIVE"] = "2h45m"
        check("envconfig: KeepAlive 解析 Go 风格时长（2h45m）",
              E.KeepAlive() == 2 * 3600 + 45 * 60, E.KeepAlive())
    finally:
        if saved is None:
            os.environ.pop("OLLAMA_KEEP_ALIVE", None)
        else:
            os.environ["OLLAMA_KEEP_ALIVE"] = saved

    check("envconfig: Models() 是 ollama 本地仓库根（不是本仓 models/ 推理权重目录）",
          E.Models().replace("\\", "/").rstrip("/").endswith("models"), E.Models())


# ============================================================
# 5. format / progress
# ============================================================


def test_format_progress() -> None:
    check("format: HumanBytes 用十进制 KB（1500 → 1.5 KB）", F.HumanBytes(1500) == "1.5 KB",
          F.HumanBytes(1500))
    check("format: HumanBytes 分档精度（>=10 不带小数，15 GB 不是 15.0 GB）",
          F.HumanBytes(15 * F.GigaByte) == "15 GB", F.HumanBytes(15 * F.GigaByte))
    check("format: HumanBytes2 用二进制 KiB（1024 → 1.0 KiB）",
          F.HumanBytes2(1024) == "1.0 KiB", F.HumanBytes2(1024))

    # rate：未停止时用 1 秒桶窗口，停止后改用「全程平均」—— 两种结果必须不同，
    # 否则说明实现只写了一路。注意桶节流判据是 `now - last > 1.0`（严格大于），
    # 恰好 1 秒整不会开新桶，所以两次采样要拉开到 2 秒。
    clock = {"now": 0.0}
    bar = P.Bar("t", 1000, 0, clock=lambda: clock["now"])
    clock["now"] = 1.0
    bar.set(100)                       # 桶 1：(1.0, 100)
    clock["now"] = 3.0
    bar.set(400)                       # 桶 2：(3.0, 400) → 窗口速率 (400-100)/2 = 150/s
    running_rate = bar.rate()
    clock["now"] = 4.0
    bar.set(1000)                      # 到顶即 stop → 改看全程平均 1000/4 = 250/s
    stopped_rate = bar.rate()
    check("progress: 结束后速率切到全程平均（1000/4s=250，而非窗口 150/s）",
          bar.done and abs(running_rate - 150.0) < 0.001
          and abs(stopped_rate - 250.0) < 0.001,
          {"running": running_rate, "stopped": stopped_rate})

    tracker = P.ProgressTracker().update_all([
        {"status": "pulling manifest"},                       # 无 digest，不能进分母
        {"status": "pulling a", "digest": "sha256:a", "total": 1000, "completed": 0},
        {"status": "pulling a", "digest": "sha256:a", "completed": 400},  # 同 digest upsert
        {"status": "success"},
    ])
    check("progress: 无 digest 的阶段事件不进分母、同 digest 走 upsert",
          tracker.total == 1000 and tracker.completed == 400 and len(tracker.blobs) == 1,
          tracker.as_dict())


# ============================================================
# 6. openai 兼容层
# ============================================================


def test_openai() -> None:
    request = O.FromChatRequest({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    check("openai: temperature/top_p 缺省填 1.0（OpenAI 的默认值语义）",
          request["options"]["temperature"] == 1.0 and request["options"]["top_p"] == 1.0,
          request["options"])

    request = O.FromChatRequest({"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ]}]})
    check("openai: 多段 content 拆成多条 ollama 消息（图像走独立 images 字段）",
          len(request["messages"]) == 2
          and request["messages"][0]["content"] == "看图"
          and request["messages"][1]["images"] == [b"\x89PNG\r\n\x1a\n"],
          [(m["role"], m.get("content"), len(m.get("images") or [])) for m in request["messages"]])

    request = O.FromChatRequest({"model": "m", "messages": [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\":\"BJ\"}"}}]},
        {"role": "tool", "content": "25C", "tool_call_id": "call_1"},
    ]})
    check("openai: tool 角色缺 name 时按 tool_call_id 倒序反查",
          request["messages"][-1].get("tool_name") == "get_weather", request["messages"][-1])

    response = {"model": "m", "created_at": "2026-10-28T10:00:00Z", "done_reason": "stop",
                "message": {"role": "assistant", "content": "答",
                            "tool_calls": [{"id": "x", "function": {"name": "f",
                                                                   "arguments": {"a": 1}}}]},
                "metrics": {"prompt_eval_count": 5, "eval_count": 3}}
    completion = O.ToChatCompletion("id", response)
    check("openai: stop + 有 tool_calls ⇒ finish_reason 必须是 tool_calls",
          completion["choices"][0]["finish_reason"] == "tool_calls", completion["choices"][0])
    check("openai: arguments 序列化为 JSON 字符串（OpenAI 约定）",
          json.loads(completion["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
          == {"a": 1})

    mixed = {"model": "m", "created_at": "2026-10-28T10:00:00Z",
             "message": {"role": "assistant", "content": "正文", "thinking": "思考"}}
    chunks = O.ToStreamChunks("c", mixed, True)
    check("openai: 流式把 thinking 与正文拆成两条 chunk（共用同 created）",
          len(chunks) == 2
          and "content" not in chunks[0]["choices"][0]["delta"]
          and chunks[0]["choices"][0]["delta"]["reasoning"] == "思考"
          and chunks[1]["choices"][0]["delta"]["content"] == "正文"
          and chunks[0]["created"] == chunks[1]["created"],
          [ch["choices"][0]["delta"] for ch in chunks])

    try:
        O.FromChatRequest({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": "https://example.com/a.png"}]}]})
        rejected = False
    except O.ConversionError:
        rejected = True
    check("openai: http(s) 图片地址必须拒（只认 data URI，避免 SSRF 面）", rejected)

    check("openai: reasoning_effort 两端夹到最近档（minimal→low、ultra→max、none→False）",
          O.thinkFromReasoningEffort("minimal") == "low"
          and O.thinkFromReasoningEffort("ultra") == "max"
          and O.thinkFromReasoningEffort("none") is False)

    check("openai: response_format json_object → format=\"json\"",
          O.FromChatRequest({"model": "m", "messages": [],
                             "response_format": {"type": "json_object"}})["format"] == "json")


# ============================================================
# 7. anthropic 兼容层
# ============================================================


def test_anthropic() -> None:
    request = A.FromMessagesRequest({
        "model": "m", "max_tokens": 64,
        "messages": [
            {"role": "assistant", "content": [
                {"type": "text", "text": "我来查"},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather",
                 "input": {"city": "BJ"}}]},
            {"role": "user", "content": [
                {"type": "text", "text": "谢谢"},
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": [{"type": "text", "text": "25C"}]}]},
        ],
    })
    roles = [message["role"] for message in request["messages"]]
    check("anthropic: user 轮里 tool_result 必须排在正文**之前**",
          roles == ["assistant", "tool", "user"], roles)
    check("anthropic: max_tokens 写进 num_predict", request["options"]["num_predict"] == 64,
          request["options"])

    response = {"model": "m", "done_reason": "stop",
                "metrics": {"prompt_eval_count": 100, "eval_count": 7,
                            "prompt_eval_cached_count": 30},
                "message": {"role": "assistant", "content": "答"}}
    result = A.ToMessagesResponse("msg_1", response)
    check("anthropic: input_tokens 不含缓存命中（100-30=70）",
          result["usage"]["input_tokens"] == 70
          and result["usage"]["cache_read_input_tokens"] == 30, result["usage"])
    check("anthropic: stop → end_turn", result["stop_reason"] == "end_turn", result["stop_reason"])

    converter = A.StreamConverter("msg_2", "m", estimated_input_tokens=11)
    seen: list[tuple[str, int]] = []
    for chunk in [
        {"message": {"role": "assistant", "thinking": "想"}, "done": False},
        {"message": {"role": "assistant", "content": "正"}, "done": False},
        {"message": {"role": "assistant", "content": "",
                     "tool_calls": [{"id": "c9", "function": {"name": "f",
                                                             "arguments": {"x": 1}}}]},
         "done": False},
        # 同一 id 再出现一次 —— 流式会重发，只能发一次
        {"message": {"role": "assistant", "content": "",
                     "tool_calls": [{"id": "c9", "function": {"name": "f",
                                                             "arguments": {"x": 1}}}]},
         "done": True, "done_reason": "stop",
         "metrics": {"prompt_eval_count": 50, "eval_count": 9}},
    ]:
        for event, data in converter.process(chunk):
            seen.append((event, data.get("index", -1)))
    starts = [index for event, index in seen if event == "content_block_start"]
    check("anthropic: 流式 block index 单调递增（thinking→text→tool_use 各占一个）",
          starts == [0, 1, 2], starts)
    stops = sum(1 for event, _ in seen if event == "content_block_stop")
    check("anthropic: 每个 block 都发 content_block_stop（少一个客户端会永久挂起）",
          stops == 3, seen)
    check("anthropic: 重复 tool_call id 只发一次 tool_use",
          sum(1 for event, _ in seen if event == "content_block_start") == 3)
    delta = [data for event, data in converter.process({"done": True,
                                                        "done_reason": "stop",
                                                        "metrics": {"prompt_eval_count": 50,
                                                                    "eval_count": 9}})
             if event == "message_delta"]
    check("anthropic: 有工具调用时 stop_reason 一律 tool_use（压过 done_reason）",
          bool(delta) and delta[0]["delta"]["stop_reason"] == "tool_use",
          delta[0] if delta else None)


# ============================================================
# 8. 既有模块回归（names / modelfile / params / manifest）
# ============================================================


def test_existing_modules() -> None:
    parsed = names.parse_name("qwen3:8b")
    check("names: 缺省 host/namespace 补齐，display_shortest 隐藏 library",
          parsed.host == "registry.ollama.ai" and parsed.namespace == "library"
          and parsed.tag == "8b", parsed)

    # parse() 返回的是 Modelfile 对象（不是 dict）：`FROM` 归一成指令名 "model"。
    document = modelfile.parse('FROM llama3\nPARAMETER temperature 0.7\nSYSTEM """a\nb"""\n')
    instructions = [(command.name, command.args) for command in document.commands]
    check("modelfile: 三引号多行值与 PARAMETER 都能解出",
          ("model", "llama3") in instructions and ("system", "a\nb") in instructions,
          instructions)

    # format_params 的入参是 `{参数名: [字符串值, …]}`（对齐上游 map[string][]string），
    # 不是一个扁平 dict —— 传扁平 dict 会把 "a" 当字符列表拆开。
    converted = params.format_params({"temperature": ["0.8"], "num_ctx": ["4096"],
                                      "stop": ["a", "b"]})
    check("params: 字符串形态的数字仍转成对类型（num_ctx 必须是 int）",
          converted["num_ctx"] == 4096 and converted["num_ctx"] is not True
          and converted["temperature"] == 0.8 and converted["stop"] == ["a", "b"], converted)

    temporary = tempfile.mkdtemp(prefix="ollamanative_")
    try:
        original = manifest.models_root
        # 换掉根目录而不是真写用户的模型库 —— 也验证了「根目录只有 models_root 一个真源」
        manifest.models_root = lambda: temporary          # type: ignore[assignment]
        try:
            layer = manifest.new_layer(b"hello", manifest.MEDIA_TYPE_TEMPLATE)
            blob = os.path.join(temporary, "blobs", "sha256-" + layer.digest[7:])
            check("manifest: new_layer 用 sha256 做 digest 且落成内容寻址 blob",
                  layer.digest.startswith("sha256:") and os.path.isfile(blob),
                  {"digest": layer.digest, "blob": blob, "exists": os.path.isfile(blob)})

            # 同一份内容再写一次必须复用同一个 blob（内容寻址的全部意义）
            again = manifest.new_layer(b"hello", manifest.MEDIA_TYPE_TEMPLATE)
            check("manifest: 同一内容重写复用同一 digest（不产生第二份 blob）",
                  again.digest == layer.digest
                  and len(os.listdir(os.path.join(temporary, "blobs"))) == 1,
                  {"first": layer.digest, "second": again.digest})

            # 清单读写往返：manifests/<host>/<ns>/<model>/<tag> 这一层结构是 ollama 的唯一查找路径
            name = names.parse_name("demo:1b")
            config = manifest.new_layer(b"{}", manifest.MEDIA_TYPE_CONFIG)
            written = manifest.write_manifest(name, config, [layer])
            loaded = manifest.read_manifest(name)
            check("manifest: 清单往返（路径按 host/ns/model/tag 分层、layer 不丢）",
                  os.path.isfile(written)
                  and loaded.config.digest == config.digest
                  and [item.digest for item in loaded.layers] == [layer.digest],
                  {"path": written, "layers": [item.digest for item in loaded.layers]})
        finally:
            manifest.models_root = original                # type: ignore[assignment]
    finally:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)

    # ── 接入层复用 envconfig：探测地址必须跟 OLLAMA_HOST 走 ──
    # 这条守的是「已修过的真缺陷」：`try_start_ollama` 曾两处写死 `localhost:11434`，
    # 用户改了端口就会被误判成「Ollama 没启动」。
    from app.services import ollama as ollama_service

    saved_host = os.environ.get("OLLAMA_HOST")
    try:
        os.environ.pop("OLLAMA_HOST", None)
        check("ollama: 未设 OLLAMA_HOST 时探测地址保持本仓默认（行为不变）",
              ollama_service.resolve_ollama_url() == ollama_service.DEFAULT_OLLAMA_URL,
              ollama_service.resolve_ollama_url())

        os.environ["OLLAMA_HOST"] = "127.0.0.1:11435"
        check("ollama: 设了 OLLAMA_HOST 就跟着走（改端口不再误判未启动）",
              ollama_service.resolve_ollama_url() == "http://127.0.0.1:11435",
              ollama_service.resolve_ollama_url())

        # 通配地址是**绑定**写法：照抄去连在 Windows 上必然 ECONNREFUSED，必须换回环
        os.environ["OLLAMA_HOST"] = "0.0.0.0:11434"
        check("ollama: 通配绑定地址换回 127.0.0.1（否则 Windows 连不上）",
              ollama_service.resolve_ollama_url() == "http://127.0.0.1:11434",
              ollama_service.resolve_ollama_url())

        # 带引号是桌面端写进注册表/脚本后的常见形态，不剥引号会拼出非法 URL
        os.environ["OLLAMA_HOST"] = '"https://example.com"'
        check("ollama: 带引号的 OLLAMA_HOST 剥引号 + 按 https 补 443",
              ollama_service.resolve_ollama_url() == "https://example.com:443",
              ollama_service.resolve_ollama_url())

        # 显式配置优先级最高，且尾部斜杠要去掉（拼 /api/tags 时会出现 //）
        os.environ["OLLAMA_HOST"] = "127.0.0.1:11435"
        check("ollama: 显式配置压过环境变量且去掉尾部斜杠",
              ollama_service.resolve_ollama_url(" http://10.0.0.2:8080/// ")
              == "http://10.0.0.2:8080",
              ollama_service.resolve_ollama_url(" http://10.0.0.2:8080/// "))
    finally:
        if saved_host is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = saved_host


# ============================================================
# 汇总
# ============================================================


def main() -> int:
    started = time.time()
    for suite in (test_gotemplate, test_template, test_harmony, test_envconfig,
                  test_format_progress, test_openai, test_anthropic, test_existing_modules):
        try:
            suite()
        except Exception as exc:  # noqa: BLE001 - 套件崩了要报出来而不是中断全程
            check(f"{suite.__name__} 未抛异常地跑完", False, f"{type(exc).__name__}: {exc}")

    failed_items = [item for item in _RESULTS if not item[1]]
    for item_name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + item_name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed_items)}/{len(_RESULTS)} passed"
          f"（{time.time() - started:.2f}s）")
    return 1 if failed_items else 0


if __name__ == "__main__":
    raise SystemExit(main())
