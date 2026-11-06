"""提示词模板编译层的守卫（2026-11-03 立，随「comfyui-gateway 提示词线迁入」一起落）。

为什么单独立这套：这四份模板的错法**全都不报错**✗ —— 参数填错格、参考图没接上、
``输入模式`` 停在「文本」，跑起来照样出提示词，只是参考图白传、任务类型不对（"跑通了、结果不对" ✗✗）。
所以判据只钉四处会静默走样的地方：

1. **按上下文算的两格** ✓：H3 的 ``#3.输入模式``、生图的 ``#1.任务模式`` 必须跟着"有没有参考图"走
   （写死成常量 ⇒ 参考图白传 ✗）；
2. **参考图接线** ✓：第 i 张 ⇒ ``LoadImage`` 节点 ``9101+i`` ⇒ ``图片`` / ``图片2`` …（接错格 ⇒ 图不进模型 ✗）；
3. **反复编译不叠加** ✗：H3 每次往 ``系统提示词`` 追加语言指令 + 输出锁，若模板被缓存起来
   再追加，第二次就叠两段、第三次三段（提示词越跑越长，且没人看得出来 ✓✗）；
4. **该拒就拒** ✓：拼错的键、非法的任务类型、超长提示词、超 8 张参考图 —— 静默忽略/截断
   都会让人以为"改了"（实际没改 ✗）。

本套**完全离线** ✓：不连 ComfyUI、不碰显卡、不开端口（模板就在 ``configs/comfyui/prompts`` ✓）。

运行::

    ./.venv/Scripts/python.exe tests/comfyui_prompts_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

BACKEND_PY = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROXY_TO_NODE", "0")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(BACKEND_PY))

from app.services import comfyui_prompts as prompts  # noqa: E402
from app.services.comfyui_catalog import is_api_format  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _err(fn) -> str:
    """跑一段必然失败的动作，返回错误原文 ✓（没抛就返回空串 ⇒ 判据自己会红 ✓）。"""
    try:
        fn()
    except prompts.PromptTemplateError as err:
        return str(err)
    except Exception as err:  # noqa: BLE001 - 抛了别的类型同样算红 ✓
        return f"[{type(err).__name__}] {err}"
    return ""


# ─────────────────── ① 目录：四份模板都在、参数都落到真格上 ───────────────────

def case_catalog() -> None:
    data = prompts.catalog()
    rows = {row["id"]: row for row in data["templates"]}

    check("① 目录列出四份模板",
          sorted(rows) == ["h3-enhance", "image-caption", "image-caption-florence", "image-enhance"],
          sorted(rows))
    check("① 四份模板文件都在盘上", all(row["exists"] for row in rows.values()),
          {key: row["exists"] for key, row in rows.items()})
    check("① 每份都报出节点数与所需节点类",
          all(row.get("nodeCount") and row.get("requiredNodes") for row in rows.values()),
          {key: (row.get("nodeCount"), row.get("requiredNodes")) for key, row in rows.items()})
    check("① 每份模板都带人话标题与用途",
          all(row["title"] and row["purpose"] for row in rows.values()),
          [row["title"] for row in rows.values()])

    # 参数表必须逐条写清「落到哪个节点哪一格」✓ —— 缺一格就是"填了没生效"的源头 ✗
    slots = {(p["node"], p["input"]) for row in rows.values() for p in row["params"]}
    check("① 参数都声明了落点（节点 + 输入格名）",
          all(p["node"] and p["input"] for row in rows.values() for p in row["params"]) and len(slots) >= 6,
          sorted(slots))

    # 行为键：h3 / 生图吃参考图；florence 版**明确不吃** language（传了要拒 ✓）
    check("① H3 与生图增强吃参考图，两份反推都不吃",
          (rows["h3-enhance"]["images"] is not None and rows["image-enhance"]["images"] is not None
           and rows["image-caption"]["images"] is None and rows["image-caption-florence"]["images"] is None),
          {key: row["images"] for key, row in rows.items()})
    check("① florence 版明确声明不吃 language / instruction",
          rows["image-caption-florence"]["extras"] == [] and "不吃" in rows["image-caption-florence"]["notice"],
          rows["image-caption-florence"]["extras"])

    skill = data["h3Skill"]
    check("② H3 提示词 skill 文本资产齐备（三份都非空）",
          skill["available"] and all(size and size > 0 for size in skill["files"].values()),
          skill["files"])


# ─────────────────── ③ H3：默认值 + 两格按上下文算 + 逐字枚举 ───────────────────

def case_h3_defaults() -> None:
    result = prompts.compile_template("h3-enhance", {"prompt": "雨夜的便利店，霓虹在积水里晃"})
    graph = result["workflow"]
    node2, node3 = graph["2"]["inputs"], graph["3"]["inputs"]

    check("③ 原始提示词落到 #2.输入提示词", node2["输入提示词"] == "雨夜的便利店，霓虹在积水里晃", node2["输入提示词"])
    check("③ 任务类型缺省＝文生视频(T2VA)（逐字带括号 ✓）",
          node2["任务类型"] == "文生视频(T2VA)", node2["任务类型"])
    check("③ 时长缺省＝5 秒", node2["视频时长"] == 5, node2["视频时长"])
    check("③ 无参考图 ⇒ #3.输入模式＝文本",
          node3["输入模式"] == "文本", node3["输入模式"])
    check("③ 无参考图 ⇒ 不凭空造 LoadImage 节点",
          not [node for node in graph if node.startswith("91")], sorted(graph))
    check("③ 系统提示词末尾带输出锁（不锁会把「思考过程」当交付 ✗）",
          node3["系统提示词"].endswith(prompts._H3_OUTPUT_LOCK), node3["系统提示词"][-40:])
    check("③ 默认中文 ⇒ 追加中文语言指令",
          "字段正文全部使用中文" in node3["系统提示词"], node3["系统提示词"][:40])
    check("③ 编译结果本身是 API 格式（可直塞 POST /runs ✓）", is_api_format(graph))
    check("③ 报出所需节点类（判「上游装了没」用 ✓）",
          "TE_H3_Prompt_Enhancer" in result["requiredNodes"], result["requiredNodes"])


def case_h3_images() -> None:
    result = prompts.compile_template(
        "h3-enhance",
        {"prompt": "她在雨里回头", "mode": "首帧图生视频(I2VA)", "seconds": 6,
         "images": ["start.png", {"name": "end.png", "subfolder": "refs"}], "language": "en"},
    )
    graph = result["workflow"]
    node3 = graph["3"]["inputs"]

    check("③ 有参考图 ⇒ #3.输入模式＝图片", node3["输入模式"] == "图片", node3["输入模式"])
    check("③ 第 1 张接在 #3.图片 上（节点 9101）",
          node3["图片"] == ["9101", 0] and graph["9101"]["inputs"]["image"] == "start.png",
          node3.get("图片"))
    check("③ 第 2 张接在 #3.图片2 上，且 subfolder 拼成 refs/end.png",
          node3["图片2"] == ["9102", 0] and graph["9102"]["inputs"]["image"] == "refs/end.png",
          graph.get("9102"))
    check("③ 参考图节点是 LoadImage（不是别的加载器）",
          (graph["9101"]["class_type"], graph["9102"]["class_type"]) == ("LoadImage", "LoadImage"),
          graph["9101"]["class_type"])
    check("③ 任务类型逐字保留（首帧图生视频(I2VA)）",
          graph["2"]["inputs"]["任务类型"] == "首帧图生视频(I2VA)", graph["2"]["inputs"]["任务类型"])
    check("③ language=en ⇒ 追英文指令、不追中文指令",
          "Write every field body in English" in node3["系统提示词"]
          and "字段正文全部使用中文" not in node3["系统提示词"], node3["系统提示词"][:60])

    # ④ 反复编译不叠加（模板若被缓存，这里会 2 段、3 段地长 ✗）
    first = prompts.compile_template("h3-enhance", {"prompt": "x"})["workflow"]["3"]["inputs"]["系统提示词"]
    second = prompts.compile_template("h3-enhance", {"prompt": "x"})["workflow"]["3"]["inputs"]["系统提示词"]
    check("④ 连续编译两次，系统提示词一字不差（不越叠越长 ✓）", first == second, (len(first), len(second)))
    check("④ 输出锁只出现一次", second.count(prompts._H3_OUTPUT_LOCK) == 1,
          second.count(prompts._H3_OUTPUT_LOCK))


# ─────────────────── ⑤⑥ 该夹紧的夹紧、该拒的拒 ───────────────────

def case_h3_bounds() -> None:
    over = prompts.compile_template("h3-enhance", {"prompt": "x", "seconds": 30})
    check("⑤ 时长 30 ⇒ 夹到 15", over["workflow"]["2"]["inputs"]["视频时长"] == 15,
          over["workflow"]["2"]["inputs"]["视频时长"])
    check("⑤ 夹紧留一条 warning（静默改写数值同样是骗人 ✗）",
          any("15" in item for item in over["warnings"]), over["warnings"])

    under = prompts.compile_template("h3-enhance", {"prompt": "x", "seconds": 0})
    check("⑤ 时长 0 ⇒ 夹到 1", under["workflow"]["2"]["inputs"]["视频时长"] == 1,
          under["workflow"]["2"]["inputs"]["视频时长"])
    check("⑤ 字符串数字也认（前端表单只送字符串 ✓）",
          prompts.compile_template("h3-enhance", {"prompt": "x", "seconds": "7"})["workflow"]["2"]["inputs"]["视频时长"] == 7)

    check("⑤ 时长 abc ⇒ 报错（不静默用默认值 ✗）",
          "要整数" in _err(lambda: prompts.compile_template("h3-enhance", {"prompt": "x", "seconds": "abc"})))

    check("⑥ 缺原始提示词 ⇒ 报必填",
          "必填" in _err(lambda: prompts.compile_template("h3-enhance", {})))
    check("⑥ 提示词 8001 字 ⇒ 报超长（不静默截断 ✗）",
          "超过上限 8000" in _err(lambda: prompts.compile_template("h3-enhance", {"prompt": "字" * 8001})))
    bad_mode = _err(lambda: prompts.compile_template("h3-enhance", {"prompt": "x", "mode": "图生视频"}))
    check("⑥ 非法的任务类型 ⇒ 报错并列出可选值（别名不算 ✓）",
          "只支持" in bad_mode and "首帧图生视频(I2VA)" in bad_mode, bad_mode[:80])
    check("⑥ 拼错的键 ⇒ 报错并列出可用键（静默忽略⇒按默认值跑 ✗）",
          "不认识这些键" in _err(lambda: prompts.compile_template("h3-enhance", {"prompt": "x", "secods": 10})),
          _err(lambda: prompts.compile_template("h3-enhance", {"prompt": "x", "secods": 10}))[:60])
    check("⑥ 参考图 9 张 ⇒ 报错（不静默丢图 ✗）",
          "最多 8 张" in _err(lambda: prompts.compile_template("h3-enhance", {"prompt": "x", "images": [f"{i}.png" for i in range(9)]})))
    check("⑥ 不存在的模板 ⇒ unknown_template（不回落到第一份 ✗）",
          "没有提示词模板" in _err(lambda: prompts.compile_template("h3-enhance2", {})),
          _err(lambda: prompts.compile_template("h3-enhance2", {}))[:60])


# ─────────────────── ⑦ 生图增强：任务模式 / 增强方式 / 卸载 都钉住 ───────────────────

def case_image_enhance() -> None:
    text_only = prompts.compile_template("image-enhance", {"prompt": "一只橘猫趴在窗台"})
    node1 = text_only["workflow"]["1"]["inputs"]

    check("⑦ 无参考图 ⇒ 任务模式＝文生图", node1["任务模式"] == "文生图", node1["任务模式"])
    check("⑦ 输出语言缺省＝中文", node1["输出语言"] == "中文", node1["输出语言"])
    check("⑦ 增强方式每次钉回「本地官方PE」（模板可被改成云端 ✗）",
          node1["增强方式"] == "本地官方PE", node1["增强方式"])
    check("⑦ 生成后自动卸载模型＝True（布尔，不是字符串 ✓）",
          node1["生成后自动卸载模型"] is True, node1["生成后自动卸载模型"])

    with_image = prompts.compile_template(
        "image-enhance", {"prompt": "把猫换成狗", "images": ["cat.png"], "language": "en"})
    inputs = with_image["workflow"]["1"]["inputs"]
    check("⑦ 有参考图 ⇒ 任务模式＝图生图", inputs["任务模式"] == "图生图", inputs["任务模式"])
    check("⑦ 参考图接在 #1.图片 上（节点 9101）",
          inputs["图片"] == ["9101", 0] and with_image["workflow"]["9101"]["inputs"]["image"] == "cat.png",
          inputs.get("图片"))
    check("⑦ language=en ⇒ 输出语言＝英文", inputs["输出语言"] == "英文", inputs["输出语言"])
    check("⑦ 生图增强不吃 instruction（传了要拒，不静默忽略 ✓）",
          "不认识这些键" in _err(lambda: prompts.compile_template("image-enhance", {"prompt": "x", "instruction": "y"})))


# ─────────────────── ⑧ 反推：两份图各自的默认 system prompt ───────────────────

def case_caption() -> None:
    zh = prompts.compile_template("image-caption", {"image": "shot.png"})
    node1, node3 = zh["workflow"]["1"]["inputs"], zh["workflow"]["3"]["inputs"]

    check("⑧ 图名落到 #1.image", node1["image"] == "shot.png", node1["image"])
    check("⑧ 缺省中文 system prompt",
          node3["system_prompt"] == prompts._CAPTION_SYSTEM["zh"], node3["system_prompt"][:30])
    check("⑧ user prompt 跟着语言走（中文）",
          node3["custom_prompt"] == prompts._CAPTION_USER["zh"], node3["custom_prompt"])
    check("⑧ 模板里 images 仍连在 #1 上（反推要真的看图 ✓）",
          node3.get("images") == ["1", 0], node3.get("images"))

    en = prompts.compile_template("image-caption", {"image": {"name": "a.png", "subfolder": "shots"}, "language": "en"})
    check("⑧ language=en ⇒ 英文 system prompt",
          en["workflow"]["3"]["inputs"]["system_prompt"] == prompts._CAPTION_SYSTEM["en"],
          en["workflow"]["3"]["inputs"]["system_prompt"][:40])
    check("⑧ subfolder 拼成 shots/a.png",
          en["workflow"]["1"]["inputs"]["image"] == "shots/a.png", en["workflow"]["1"]["inputs"]["image"])

    custom = prompts.compile_template("image-caption", {"image": "a.png", "instruction": "只描述服装"})
    check("⑧ instruction 给了就整段顶掉 system prompt",
          custom["workflow"]["3"]["inputs"]["system_prompt"] == "只描述服装",
          custom["workflow"]["3"]["inputs"]["system_prompt"])

    check("⑧ 缺图片 ⇒ 报必填", "必填" in _err(lambda: prompts.compile_template("image-caption", {})))

    florence = prompts.compile_template("image-caption-florence", {"image": "p.png"})
    check("⑧ florence 版用的是 Florence2 节点",
          "Florence2Run" in florence["requiredNodes"], florence["requiredNodes"])
    check("⑧ florence 版不吃 language（传了要拒 ✓ 免得以为换成了英文 ✗）",
          "不认识这些键" in _err(lambda: prompts.compile_template("image-caption-florence", {"image": "p.png", "language": "en"})))
    check("⑧ 两份反推的图名落点一致（#1.image）",
          florence["workflow"]["1"]["inputs"]["image"] == "p.png",
          florence["workflow"]["1"]["inputs"]["image"])


# ─────────────────── ⑨ 模板被改坏：要报，且不拖垮整个目录 ───────────────────

def case_template_drift() -> None:
    source = prompts.prompt_path("h3-enhance")
    graph = json.loads(source.read_text(encoding="utf-8"))
    broken = {("9" if key == "2" else key): value for key, value in graph.items()}

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "h3-prompt-enhance.json"
        target.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
        with patch.object(prompts, "PROMPT_DIR", Path(tmp)):
            message = _err(lambda: prompts.compile_template("h3-enhance", {"prompt": "x"}))
            data = prompts.catalog()
            rows = {row["id"]: row for row in data["templates"]}

    check("⑨ 节点被改名 ⇒ 报「没有节点 2」（不假装填成功 ✗）",
          "没有节点 2" in message, message[:80])
    check("⑨ 单份坏掉不拖垮目录：四份仍在列", len(rows) == 4, sorted(rows))
    check("⑨ 坏的那份在自己那行报 error（表单别假装能填 ✓）", bool(rows["h3-enhance"].get("error")),
          rows["h3-enhance"].get("error"))
    check("⑨ 这份坏法的 error 点名了参数指向的节点（不是一句「坏 JSON」✓）",
          "节点 2" in str(rows["h3-enhance"].get("error")), rows["h3-enhance"].get("error"))
    check("⑨ 这个临时目录里另三份文件确实不在 ⇒ 如实报 exists=False（不假装可用 ✓）",
          all(row["exists"] is False for key, row in rows.items() if key != "h3-enhance"),
          {key: row["exists"] for key, row in rows.items()})


def main() -> None:
    case_catalog()
    case_h3_defaults()
    case_h3_images()
    case_h3_bounds()
    case_image_enhance()
    case_caption()
    case_template_drift()

    failures = 0
    for name, ok, detail in _RESULTS:
        if ok:
            print(f"  ✓ {name}")
        else:
            failures += 1
            print(f"  ✗ {name}  -> {detail}")
    print(f"\nSUMMARY: {len(_RESULTS) - failures}/{len(_RESULTS)} passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
