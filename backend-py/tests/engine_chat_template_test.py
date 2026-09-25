"""自检：对话骨架（``engine/chat_template.py`` ✓ + 接进 ``llm_backend`` ✓ 2026-09-25 起 ✓）。

验什么 ✗：``generate`` 此前是**裸拼接**（``system + "\\n\\n" + prompt``）✗ —— 对「骨架决定格式」的模型
（``<|im_start|>`` 系 ✓），提示词**没落进对话骨架** ✓✗。本套钉住四件事：**骨架从权重元数据来** ✓、
**逐字可核** ✓、**没有就如实报** ✓、**坏了就拒** ✗✗（**不静默回落裸拼接** ✗）。

判据：
1. ⭐⭐ **「没有模板」与「模板坏了」是两回事** ✗✗：没有 ⇒ ``None`` ✓；坏了 ⇒ **抛** ✗（不是 None ✗）；
2. ⭐⭐ **产出逐字可核** ✓：ChatML 骨架逐字相等 ✓（含 assistant 前缀 ✓ / 不给 system 就不编造 ✓）；
3. ⭐⭐ **本层不内置骨架** ✗✗：模块级没有模板常量 ✓、空元数据拿不到 ✓；
4. ⭐ **渲染失败 ⇒ 拒** ✗（未定义变量 / 空 messages / 形状不对 / 缺 role / ``raise_exception`` ✓）；
5. ⭐ **接线**：有骨架 ⇒ encode 收到骨架文本 ✓；没有 ⇒ **老行为一字未动** ✓；坏了 ⇒ generate 拒 ✗；
   ``describe`` 如实报 ✓。

⚠️ 本套**不读真 GGUF** ✗（造元数据字典 ✓ —— 真装载走 ``gguf_to_llm`` 那套已验 ✓）。

运行::

    ./.venv/Scripts/python.exe tests/engine_chat_template_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="chattmpl_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import chat_template  # noqa: E402
from app.services.engine import llm as llm_mod  # noqa: E402
from app.services.engine import llm_backend  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


# ChatML 骨架 ✓（**照抄权重元数据里的形态** ✓ —— 本仓**不内置它** ✗✗，只在测试里当样本 ✓）
QWEN_LIKE = ("{% for message in messages %}"
             "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
             "{% endfor %}"
             "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
EXPECTED = ("<|im_start|>system\n你是分镜师<|im_end|>\n"
            "<|im_start|>user\n画个猫<|im_end|>\n"
            "<|im_start|>assistant\n")
CONFIG = llm_mod.LlmConfig(vocab_size=32, hidden=16, depth=1, heads=2, kv_heads=1,
                           head_dim=8, ffn=32, eos_id=31)


def _raises(fn, *, needle: str = "") -> bool:
    try:
        fn()
    except chat_template.ChatTemplateError as err:
        return needle in str(err) if needle else True
    except Exception:  # noqa: BLE001
        return False
    return False


def _from(source: str):
    return chat_template.ChatTemplate.from_metadata({"tokenizer.chat_template": source})


def case_absent_vs_broken() -> None:
    check("① ⭐⭐ 元数据**没有**模板 ⇒ None ✓（不是空串、不是默认骨架 ✗）",
          chat_template.ChatTemplate.from_metadata({}) is None
          and chat_template.ChatTemplate.from_metadata(None) is None
          and chat_template.ChatTemplate.from_metadata({"tokenizer.chat_template": "   "}) is None)
    try:
        _from("{% for message in messages %}{{ message['role'] }}")   # 没闭合 ✗
        broken: object = None
    except chat_template.ChatTemplateError as err:
        broken = err
    check("①′ ⭐⭐ 模板**在但编译不过** ⇒ 抛 ✗（**不是** None ✗✗ —— 「坏了」≠「没有」✓）",
          broken is not None and "存在但不可用" in str(broken), broken)
    check("①″ ⭐ 裸键写法也认 ✓（``chat_template`` ✓）",
          chat_template.ChatTemplate.from_metadata({"chat_template": QWEN_LIKE}) is not None)


def case_render_exact() -> None:
    template = _from(QWEN_LIKE)
    text = template.render(chat_template.build_messages("画个猫", "你是分镜师"))
    check("② ⭐⭐ 骨架**逐字相等** ✓（system+user+assistant 前缀 ✓）", text == EXPECTED, text)
    check("②′ ⭐ 没给 system ⇒ **不编造** system 段 ✓（骨架与参考实现一致 ✓）",
          "<|im_start|>system" not in template.render(chat_template.build_messages("画个猫")))
    check("②″ ⭐ add_generation_prompt=False ⇒ 不追加 assistant 前缀 ✓",
          not template.render(chat_template.build_messages("画个猫"),
                              add_generation_prompt=False).endswith("<|im_start|>assistant\n"))
    check("②‴ ⭐ 来源如实记录 ✓（origin ✓）", template.origin == "tokenizer.chat_template", template.origin)


def case_no_builtin() -> None:
    constants = [getattr(chat_template, name) for name in dir(chat_template) if name.isupper()]
    check("③ ⭐⭐ 本层**不内置骨架** ✗✗（模块级常量里没有模板串 ✓ —— 骨架只能从元数据来 ✓）",
          all(not isinstance(item, str) or "im_start" not in item for item in constants), constants)
    check("③′ ⭐ 空元数据拿不到骨架 ✓（没有「默认模板」兜底 ✗）",
          chat_template.ChatTemplate.from_metadata({}) is None
          and not hasattr(chat_template.ChatTemplate, "default"))


def case_render_refuses() -> None:
    undefined = _from("{{ messages[0]['content'] }}{{ tools }}")     # tools 没给 ✗
    check("④ ⭐ 引用**没给**的变量 ⇒ 抛 ✗（且点明不静默回落 ✓）",
          _raises(lambda: undefined.render([{"role": "user", "content": "x"}]),
                  needle="不静默回落裸拼接"))
    check("④′ ⭐ 空 messages ⇒ 拒 ✗", _raises(lambda: undefined.render([])))
    check("④″ ⭐ 形状不对（传字符串 / 缺 role）⇒ 拒 ✗（不替调用方猜角色 ✓）",
          _raises(lambda: undefined.render("画个猫"))
          and _raises(lambda: undefined.render([{"content": "x"}])))
    rejecting = _from("{% if messages[0]['role'] != 'user' %}"
                      "{{ raise_exception('只收 user ✓') }}{% endif %}")
    check("④‴ ⭐ 模板主动拒（``raise_exception`` ✓）⇒ 抛且带原文 ✓",
          _raises(lambda: rejecting.render([{"role": "system", "content": "x"}]),
                  needle="只收 user ✓"))


def case_backend_wiring() -> None:
    class _StubHub:
        def __init__(self) -> None:
            self.encoded: list[str] = []

        def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
            self.encoded.append(text)
            return [1, 2, 3]

        def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
            return "OUT"

    original = llm_mod.generate_llm
    llm_mod.generate_llm = lambda model, ids, **kwargs: ids   # stub ✓ 只回显 ⇒ 不跑真推理 ✓
    try:
        backend = llm_backend.LlmBackend(CONFIG)
        hub = _StubHub()
        backend._model, backend._hub = object(), hub
        backend._chat_template = _from(QWEN_LIKE)
        backend._chat_template_note = "stub ✓"
        backend.generate("画个猫", system="你是分镜师")
        check("⑤ ⭐ 有骨架 ⇒ encode 收到的是**骨架文本** ✓（不是裸拼接 ✗）",
              hub.encoded[-1] == EXPECTED, hub.encoded[-1])

        blank = llm_backend.LlmBackend(CONFIG)
        check("⑤′ ⭐ 无骨架 ⇒ **老行为一字未动** ✓（裸拼接 ✓）",
              blank._build_prompt_text("画个猫", "你是分镜师") == "你是分镜师\n\n画个猫"
              and blank._build_prompt_text("画个猫") == "画个猫")
        blank._read_chat_template()          # 没给权重路径 ⇒ 读不出元数据 ✓
        check("⑤″ ⭐ 读不出元数据 ⇒ 按**没有模板**处理 ✓（不假装有骨架 ✗）+ note 非空 ✓",
              blank._chat_template is None and bool(blank._chat_template_note), blank._chat_template_note)

        broken = llm_backend.LlmBackend(CONFIG)
        broken._model, broken._hub = object(), _StubHub()
        broken._chat_template_error = "模板坏了 ✓"
        check("⑤‴ ⭐⭐ 模板坏了 ⇒ generate **当场拒** ✗（**不静默回落裸拼接** ✗✗）",
              _raises(lambda: broken.generate("x"), needle="模板坏了"))

        described = backend.describe()["chatTemplate"]
        check("⑤⁗ ⭐ describe 如实报骨架 ✓（present/origin/note 三件 ✓）",
              described["present"] is True and described["origin"] == "tokenizer.chat_template"
              and bool(described["note"]), described)
    finally:
        llm_mod.generate_llm = original


def main() -> int:
    case_absent_vs_broken()
    case_render_exact()
    case_no_builtin()
    case_render_refuses()
    case_backend_wiring()
    failures = [item for item in _RESULTS if not item[1]]
    for name, passed, detail in _RESULTS:
        print(("PASS  " if passed else "FAIL  ") + name + ("" if passed else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
