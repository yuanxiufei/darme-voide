"""S5 工具层自检：超长截断 + 剧本改写工具 + 语料检索工具。

三块**错了不报错、只是结果不对**的逻辑：

1. **滑窗截断**：保留前 70% + 后 30% + 中间显式省略提示（超长输入直接注入会触发
   context length 超限，是 fatal 不可重试的）；边界字符必须精确落在 16800/7200；
2. **剧本工具链**：``content || scriptContent`` 取值链、三条各不相同的报错文案、
   保存后**必须重算剧本指纹**（否则下游会拿过期指纹判断"剧本没变"）；
3. **语料检索**：语料缺失时**优雅降级**（available=false + 提示，不抛错）；
   CJK 2-gram 分词；打分权重 5/3/2/2/1；top 夹到 [1,5]；excerpt 截断 400 且压平空白。

运行::

    ./.venv/Scripts/python.exe tests/agent_script_corpus_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="agentscript_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import episodes  # noqa: E402
from app.services import text_slice  # noqa: E402
from app.agent.tools import corpus_tools as ct  # noqa: E402
from app.agent.tools.script_tools import create_script_tools  # noqa: E402
from app.services.prompt_blocks import SCREENPLAY_FORMAT_RULES  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _call(tools: dict, tool_id: str, arguments: dict | None = None):
    return asyncio.run(tools[tool_id].run(arguments))


_FINGERPRINT_CALLS: list[int] = []


def main() -> int:  # noqa: C901
    # ================= 滑窗截断 =================
    short = text_slice.slice_long_text("短文本")
    check("截断: 未超限原样返回（truncated=False，kept=total）",
          short == {"text": "短文本", "truncated": False, "total_chars": 3, "kept_chars": 3}, short)
    check("截断: None -> 空串且 total=0",
          text_slice.slice_long_text(None)["text"] == ""
          and text_slice.slice_long_text(None)["total_chars"] == 0)

    content = "".join(str(i % 10) for i in range(30000))
    sliced = text_slice.slice_long_text(content)
    head, tail = int(24000 * 0.7), 24000 - int(24000 * 0.7)
    check("截断: 头 16800 / 尾 7200 的边界字符精确",
          sliced["text"][:head] == content[:head]
          and sliced["text"].endswith(content[-tail:])
          and head == 16800 and tail == 7200,
          (head, tail))
    check("截断: 中间省略提示带**具体省略字数**（6000 = 30000-24000）",
          f"……（中间 {30000 - 24000} 字因内容超长已省略"
          in sliced["text"] and "请基于上文开头与下文结尾的关键情节继续）……" in sliced["text"],
          sliced["text"][16800:16920])
    check("截断: 标记 truncated/kept_chars", sliced["truncated"] is True
          and sliced["kept_chars"] == 24000 and sliced["total_chars"] == 30000)
    check("截断: 自定义上限也走同一条 70/30 规则",
          text_slice.slice_long_text("abcdefghij", 4)["text"].startswith("ab")
          and text_slice.slice_long_text("abcdefghij", 4)["text"].endswith("ij"))

    # ================= 剧本工具 =================
    client = TestClient(app)
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 剧本工具"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]

    # 指纹刷新：替换成记录器（真身会重算并写 episodes.script_hash）
    import app.agent.tools.script_tools as st  # noqa: E402

    original_refresh = st.refresh_episode_script_hash

    def _fake_refresh(conn, ep_id):
        _FINGERPRINT_CALLS.append(ep_id)
        return "hash"

    st.refresh_episode_script_hash = _fake_refresh  # type: ignore[assignment]
    tools = create_script_tools(episode_id)
    check("工具: 剧本工具集 3 个，键即 id",
          set(tools) == {"read_episode_script", "rewrite_to_screenplay", "save_script"}
          and all(k == t.id for k, t in tools.items()), sorted(tools))

    missing = create_script_tools(999999)
    check("剧本: 剧集不存在 -> `Episode not found (id=999999)`",
          _call(missing, "read_episode_script")["error"] == "Episode not found (id=999999)",
          _call(missing, "read_episode_script"))
    check("剧本: 无原文 -> `Episode has no content (id=N)`",
          _call(tools, "read_episode_script")["error"] == f"Episode has no content (id={episode_id})",
          _call(tools, "read_episode_script"))
    check("剧本: 改写工具的两条报错文案**不同**（不带 id / 带 has no content to rewrite）",
          _call(missing, "rewrite_to_screenplay")["error"] == "Episode not found"
          and _call(tools, "rewrite_to_screenplay")["error"] == "Episode has no content to rewrite",
          (_call(missing, "rewrite_to_screenplay"), _call(tools, "rewrite_to_screenplay")))

    # 只写 script_content（验 content || scriptContent 取值链）
    with engine.begin() as conn:
        conn.execute(episodes.update().where(episodes.c.id == episode_id)
                     .values(script_content="原文内容"))
    read = _call(tools, "read_episode_script")
    check("剧本: content 为空时回落到 script_content",
          read["content"] == "原文内容" and read["word_count"] == 4
          and read["episode_id"] == episode_id and read["truncated"] is False,
          read)

    rewritten = _call(tools, "rewrite_to_screenplay", {"instructions": "保持第一人称"})
    check(
        "改写: instruction 里注入了 **prompt_blocks 的规范** + 自定义指令 + 原始内容",
        rewritten["instruction"].startswith("请将以下内容改写为格式化剧本。")
        and SCREENPLAY_FORMAT_RULES in rewritten["instruction"]
        and "保持第一人称" in rewritten["instruction"]
        and rewritten["instruction"].endswith("【原始内容】\n原文内容"),
        rewritten["instruction"][:60],
    )
    check("改写: 不传 instructions 时该行为空（不留 'undefined'）",
          "请将以下内容改写为格式化剧本。\n\n" + SCREENPLAY_FORMAT_RULES + "\n\n\n\n【原始内容】"
          in _call(tools, "rewrite_to_screenplay")["instruction"],
          _call(tools, "rewrite_to_screenplay")["instruction"][-20:])

    _FINGERPRINT_CALLS.clear()
    saved = _call(tools, "save_script", {"content": "格式化后的剧本"})
    with engine.begin() as conn:
        row = conn.execute(select(episodes).where(episodes.c.id == episode_id)).first()
    check("保存: 落库 script_content 并返回 word_count",
          row.script_content == "格式化后的剧本" and saved == {"message": "Script saved", "word_count": 7},
          saved)
    check("保存: **重算了剧本指纹**（否则下游以为剧本没变）",
          _FINGERPRINT_CALLS == [episode_id], _FINGERPRINT_CALLS)
    st.refresh_episode_script_hash = original_refresh  # type: ignore[assignment]

    # ================= 语料检索 =================
    check(
        "分词: 拉丁词取 ≥2 字符 + CJK 2-gram + 单字 CJK 段保留，且去重",
        ct.tokenize("雨夜 neon 12") == ["neon", "12", "雨夜"],
        ct.tokenize("雨夜 neon 12"),
    )
    check("分词: 中文长串切 2-gram（'雨夜霓虹' -> 雨夜/夜霓/霓虹）",
          ct.tokenize("雨夜霓虹") == ["雨夜", "夜霓", "霓虹"], ct.tokenize("雨夜霓虹"))
    check("分词: 空串 -> 空列表", ct.tokenize("") == [] and ct.tokenize("   ") == [])

    # 语料缺失 -> 优雅降级
    original_jsonl, original_cache = ct.CORPUS_JSONL, ct._cache
    ct.CORPUS_JSONL = Path(tempfile.mkdtemp()) / "missing.jsonl"  # type: ignore[assignment]
    ct._cache = False  # type: ignore[assignment]
    degraded = _call(ct.create_corpus_tools(), "search_reference_prompts", {"query": "雨夜"})
    check(
        "语料缺失: available=false + 安装指引（**不抛错**，否则 Agent 直接跑不动）",
        degraded["available"] is False and degraded["results"] == []
        and "data/prompt-corpus/ 已 gitignore" in degraded["hint"]
        and "fetch_raw.py" in degraded["hint"],
        degraded["hint"][:40],
    )

    # 造一份临时语料验打分
    corpus_dir = Path(tempfile.mkdtemp())
    corpus_file = corpus_dir / "prompts.jsonl"
    rows = [
        {"id": "a", "source": "s", "license": "l", "lang": "zh", "title": "雨夜霓虹街道",
         "summary_zh": "雨夜街头", "prompt": "  a  very\n\n  long   prompt  ", "category": "短剧",
         "tags": ["霓虹", "手持"], "mode": "t2v", "duration": "5s", "aspect": "16:9"},
        {"id": "b", "source": "s", "license": "l", "lang": "zh", "title": "无关标题",
         "summary_zh": "", "prompt": "nothing", "category": "x", "tags": [], "mode": "t2v",
         "duration": "5s", "aspect": "16:9"},
        {"id": "c", "source": "s", "license": "l", "lang": "zh", "title": "",
         "summary_zh": "", "prompt": "", "category": "", "tags": [], "mode": "t2v",
         "duration": "5s", "aspect": "16:9"},
    ]
    corpus_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                           encoding="utf-8")
    ct.CORPUS_JSONL = corpus_file  # type: ignore[assignment]
    ct._cache = False  # type: ignore[assignment]
    found = _call(ct.create_corpus_tools(), "search_reference_prompts", {"query": "雨夜霓虹"})
    check(
        "检索: 命中的排前、未命中的不进结果（c 行全空 -> 0 分被剔除）",
        found["available"] is True and found["total_matched"] == 1
        and found["results"][0]["title"] == "雨夜霓虹街道",
        (found["total_matched"], [r["id"] if "id" in r else r["title"] for r in found["results"]]),
    )
    check(
        "检索: 打分口径 —— **各字段都加权**（title 5x3 + tags 3 + summary 2 = 20）",
        found["results"][0]["score"] == 20 and found["results"][0]["coverage"] == 1.0,
        (found["results"][0]["score"], found["results"][0]["coverage"]),
    )
    check("检索: 返回字段齐全（含 note 合规提示）",
          set(found["results"][0]) == {
              "score", "coverage", "source", "license", "lang", "title", "category",
              "tags", "mode", "duration", "aspect", "prompt_excerpt"}
          and "不要搬运原文" in found["note"],
          sorted(found["results"][0]))
    # 注意：`\s+ -> 单空格` **不 trim 首尾**（原 TS 如此）⇒ 首尾各留一个空格
    check("检索: excerpt 压平空白（多空格/换行 -> 单空格，但不 trim）",
          found["results"][0]["prompt_excerpt"] == " a very long prompt ", 
          found["results"][0]["prompt_excerpt"])
    check("检索: 空检索词 -> available=true 但 results 空 + hint",
          _call(ct.create_corpus_tools(), "search_reference_prompts", {"query": ""}) ==
          {"available": True, "results": [], "hint": "检索词为空。"})
    check("检索: top 夹到 [1,5]（传 99 -> 5，传 0 -> 1）",
          len(_call(ct.create_corpus_tools(), "search_reference_prompts",
                    {"query": "雨夜", "top": 99})["results"]) <= 5
          and len(_call(ct.create_corpus_tools(), "search_reference_prompts",
                        {"query": "雨夜", "top": 0})["results"]) >= 1)
    ct.CORPUS_JSONL, ct._cache = original_jsonl, original_cache  # type: ignore[assignment]

    # ================= 汇总 =================
    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    # ⚠️ Windows 中文控制台是 **GBK**（py3.14 及以前不会自动 UTF-8 ✗）⇒ 不加这行，裸跑第一条 `print` 就 `UnicodeEncodeError` 崩 ✗（守口径见 `check_cli_encoding.py` ✓）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
