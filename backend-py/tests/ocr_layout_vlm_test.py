"""版面分栏重排 + 本地 VLM 升级通道的单测（不加载 rapidocr、不联网、不碰真语料）。

为什么这两块必须单钉
--------------------
* **分栏重排**只对双栏页改变结果、对单栏页必须是恒等变换。一旦它对单栏页动了一个字，
  全库 17 万页里 97% 会被这个新功能拖累 —— 所以「单栏恒等」要当成主用例钉死，
  而不是顺手写一句 assert。
* **VLM 决策**（``decide``）是纯策略：两遍结果取谁、什么时候降级为待复核。它错了不会
  报错，只会悄悄把一份漏行的文本写进库，或者把好页全标成待复核（人工量爆炸）。
  所以每条规则都配一个反证。

运行::

    ./.venv/Scripts/python.exe tests/ocr_layout_vlm_test.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_PY))

from app.scripts.corpus import ocr_index as oi  # noqa: E402
from app.scripts.corpus import ocr_layout as ol  # noqa: E402
from app.scripts.corpus import ocr_vlm as vlm  # noqa: E402

CHECKS = 0
FAILED = 0


def check(label: str, ok: bool, detail=None) -> None:
    global CHECKS, FAILED
    CHECKS += 1
    if ok:
        print(f"  ✓ {label}")
        return
    FAILED += 1
    print(f"  ✗ {label}" + (f"  ⇒ {detail}" if detail is not None else ""))


def line(text: str, x0: float, y0: float, x1: float, y1: float):
    """造一个 rapidocr 形状的行：``[四点框, 文本, 分]``。"""
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, 0.99]


def two_column_page(left_lines: int = 5, right_lines: int = 5, width: float = 1000.0):
    """左右两栏、**按 y 交替**的输入（正是 rapidocr 的产出顺序）。"""
    rows = []
    for index in range(max(left_lines, right_lines)):
        y = 100 + index * 40
        if index < left_lines:
            rows.append(line(f"L{index + 1}", 100, y, 460, y + 28))
        if index < right_lines:
            rows.append(line(f"R{index + 1}", 540, y, 900, y + 28))
    return rows, width


def texts(rows) -> list[str]:
    return [ol.text_of(row) for row in rows]


def case_columns():
    print("\n[版面] 双栏重排")
    rows, width = two_column_page()
    report = ol.analyze(rows, width)
    order = [rows[i] for i in report["order"]] if report["order"] else rows
    got = texts(order)
    expected = ["L1", "L2", "L3", "L4", "L5", "R1", "R2", "R3", "R4", "R5"]
    check("左右不交替：左栏整体在前", got == expected, got)
    check("栏间有硬断点", report["breaks"] == {5}, report["breaks"])
    check("识别出 2 栏", report["columns"] == 2, report["columns"])

    #: 反证：不重排时必然是交替的 —— 说明上面那条不是因为输入本来就对
    raw = texts(rows)
    check("原始输入确实是交替的（否则用例无效）", raw[:4] == ["L1", "R1", "L2", "R2"], raw[:4])

    reversed_page = list(reversed(rows))
    check("乱序输入同样能还原",
          texts(ol.order_lines(reversed_page, width)) == expected,
          texts(ol.order_lines(reversed_page, width)))


def case_single_column_identity():
    print("\n[版面] 单栏必须是恒等变换")
    rows = [line(f"第{index}行正文内容在这里", 100, 100 + index * 36, 880, 128 + index * 36)
            for index in range(1, 11)]
    check("单栏返回原对象（一次拷贝都不做）", ol.order_lines(rows, 1000) is rows)
    check("analyze 报 order=None", ol.analyze(rows, 1000)["order"] is None)

    short = rows[:4]
    check("行数不足 6 不做判断", ol.order_lines(short, 1000) is short)

    table = [line(f"表格第{index}行", 60, 80 + index * 30, 940, 105 + index * 30)
             for index in range(1, 9)]
    check("满页宽行（表格）不动", ol.order_lines(table, 1000) is table)


def case_spanning_title():
    print("\n[版面] 通栏标题不能毁掉分栏检测")
    rows, width = two_column_page()
    title = line("第三章　雨夜", 100, 40, 900, 76)
    page = [title] + rows
    report = ol.analyze(page, width)
    order = [page[i] for i in report["order"]] if report["order"] else page
    got = texts(order)
    check("通栏标题仍能识别出分栏", report["order"] is not None, "分栏整体失效＝标题覆盖了中缝")
    check("标题在最前", got[0] == "第三章　雨夜", got[:3])
    check("标题之后左栏完整且在前", got[1:6] == ["L1", "L2", "L3", "L4", "L5"], got[1:7])
    check("再之后才是右栏", got[6:] == ["R1", "R2", "R3", "R4", "R5"], got[6:])


def case_stray_page_number():
    print("\n[版面] 页码/落款孤行不能被当成第二栏")
    rows = [line(f"正文第{index}行", 100, 100 + index * 36, 640, 128 + index * 36)
            for index in range(1, 9)]
    rows.append(line("12", 860, 700, 900, 728))
    report = ol.analyze(rows, 1000)
    check("不重排（右侧只有 1 行 ⇒ 不足成栏）", report["order"] is None,
          f"columns={report['columns']}")


def case_broken_boxes():
    print("\n[版面] 坏框不能拖垮整本书")
    rows = [line("正常", 100, 100, 400, 130), [None, "坏框", 0.5],
            line("正常二", 100, 160, 400, 190)] + [
                line(f"补{index}", 500, 100 + index * 36, 900, 128 + index * 36)
                for index in range(1, 6)]
    try:
        result = ol.order_lines(rows, 1000)
        ok = result is rows
        detail = "有界定异常时应放弃重排"
    except Exception as exc:  # noqa: BLE001 —— 这里就是要证明它不抛
        ok, detail = False, f"抛了异常：{type(exc).__name__}: {exc}"
    check("坏框 → 放弃重排且不抛异常", ok, detail)


def case_loopback():
    print("\n[VLM] 只许环回")
    allowed = ["http://127.0.0.1:8118/v1", "http://localhost:8118",
               "http://[::1]:8118/v1", "http://127.0.1.5:9000/ocr"]
    blocked = ["https://api.openai.com/v1", "http://192.168.1.5:8118",
               "http://8.8.8.8/v1", "file:///tmp/x", "127.0.0.1:8118", ""]
    check("环回地址放行", all(vlm.is_loopback(url) for url in allowed),
          [url for url in allowed if not vlm.is_loopback(url)])
    check("外部地址拒绝", not any(vlm.is_loopback(url) for url in blocked),
          [url for url in blocked if vlm.is_loopback(url)])
    config = vlm.VlmConfig(endpoint="http://192.168.1.5:8118/v1", max_pages=6)
    check("配了外部地址 ⇒ 整条通道不可用", not config.usable, config.describe())


def case_divergence():
    print("\n[VLM] 分歧度")
    base = "他站在门口没有动，风从走廊尽头灌进来。"
    check("完全相同 = 0", vlm.divergence(base, base) == 0.0)
    check("完全不同 > 0.9", vlm.divergence("甲乙丙丁", "子丑寅卯") > 0.9)
    check("空白差异不影响判定",
          vlm.divergence(base, base.replace("，", "， ")) < 0.05)
    leaked = base.replace("风", "")
    value = vlm.divergence(base, leaked)
    check("漏一字 → 分歧小但非零", 0 < value < 0.2, value)


def case_decide():
    print("\n[VLM] 两遍取谁")
    good = "他推开门，走廊里空无一人，只有尽头的窗户在响。" * 4
    other = "他推开门，走廊里空荡荡的，只有尽头的窗子在响。" * 4

    result = vlm.decide(good, "ok", "", "reject")
    check("VLM 没内容 ⇒ 用本地", result["source"] == "local" and result["text"] == good, result)

    result = vlm.decide("|||||||" * 20, "reject", good, "ok")
    check("本地 reject + VLM ok ⇒ 采用 VLM", result["source"] == "vlm", result)
    check("并留下升级说明", "VLM 升级" in result["note"], result["note"])

    result = vlm.decide(good, "ok", "|||||||" * 20, "reject")
    check("本地 ok + VLM 更差 ⇒ 不采信 VLM", result["source"] == "local", result)

    result = vlm.decide(good, "ok", good, "ok")
    check("两遍一致 ⇒ 用本地", result["source"] == "local" and result["note"] == "", result)

    result = vlm.decide(good, "ok", other, "ok")
    check("两遍都 ok 但文本不同 ⇒ 降级为待复核", result["verdict"] == "suspect", result)
    check("并说明分歧", "分歧" in result["note"], result["note"])
    check("仍然用本地文本（不采信可能被改写的版本）", result["text"] == good)

    result = vlm.decide(good, "suspect", good[:20], "ok")
    check("VLM 篇幅明显缩水 ⇒ 判漏行、不采用", result["source"] == "local", result)
    check("且写明原因", "偏短" in result["note"], result["note"])

    result = vlm.decide("短", "suspect", "更短", "suspect")
    check("都非 ok ⇒ 保持本地", result["source"] == "local", result)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def case_client_payload(monkeypatch_target=None):
    print("\n[VLM] 请求形状与熔断")
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "  转录结果  "}}]})

    original = vlm.urllib.request.urlopen
    vlm.urllib.request.urlopen = fake_urlopen
    try:
        class _Image:
            shape = (10, 10, 3)

            def __array__(self):
                import numpy as np
                return np.zeros((10, 10, 3), dtype="uint8")

        client = vlm.VlmClient(vlm.VlmConfig(endpoint="http://127.0.0.1:8118/v1",
                                             model="PaddleOCR-VL-0.9B", max_pages=3))
        import numpy as np
        text = client.recognize(np.zeros((10, 10, 3), dtype="uint8"))
        check("返回内容已去掉首尾空白", text == "转录结果", repr(text))
        check("打到环回端点的 chat/completions",
              captured["url"] == "http://127.0.0.1:8118/v1/chat/completions", captured["url"])
        check("temperature=0（不要采样，否则同一页两次结果不同）",
              captured["body"]["temperature"] == 0)
        prompt = json.dumps(captured["body"], ensure_ascii=False)
        check("prompt 明令禁止改写/翻译/纠错",
              "不要翻译" in prompt and "不要改写" in prompt and "不要纠正错别字" in prompt)
        check("图片以 data URL 内联传输", "data:image/png;base64," in prompt)

        def boom(request, timeout=None):
            raise OSError("connection refused")

        vlm.urllib.request.urlopen = boom
        for _ in range(3):
            try:
                client.recognize(np.zeros((10, 10, 3), dtype="uint8"))
            except RuntimeError:
                pass
        check("连续失败 3 次后熔断（不再每页白等超时）", not client.available)
    finally:
        vlm.urllib.request.urlopen = original


def case_escalator_off():
    print("\n[VLM] 未配置时零行为变化")
    escalator = vlm.Escalator(vlm.VlmConfig())
    check("未配置端点 ⇒ enabled=False", not escalator.enabled)
    check("consider 原样返回、不留痕", escalator.consider(1, None, "原文") == ("原文", None))
    check("预算为 0", escalator._budget == 0)


class _FakeImage:
    """只提供 ``shape``：分栏重排只用宽度，不读像素。"""

    def __init__(self, width: int = 1000, height: int = 1400) -> None:
        self.shape = (height, width, 3)


def case_page_text():
    print("\n[接线] _page_text 把三层串起来")
    rows, width = two_column_page()
    image = _FakeImage(width)

    def fake_ocr(_image):
        return rows, 0.1

    original_layout = oi._LAYOUT_ENABLED
    original_escalator = oi._ESCALATOR
    try:
        oi._ESCALATOR = None
        oi._LAYOUT_ENABLED = True
        text, note = oi._page_text(image, fake_ocr, page_no=1)
        flat = text.replace("\n", "")
        check("双栏页拼出的文本是「左栏全部 + 右栏全部」",
              flat == "L1L2L3L4L5R1R2R3R4R5", repr(flat))
        check("栏间确实插了换行", "\n" in text, repr(text))
        check("未配 VLM ⇒ 不留痕", note is None)

        oi._LAYOUT_ENABLED = False
        legacy, _ = oi._page_text(image, fake_ocr, page_no=1)
        check("关掉重排 ⇒ 回到按 y 的旧口径（对照基线）",
              legacy == "L1R1L2R2L3R3L4R4L5R5", repr(legacy))
    finally:
        oi._LAYOUT_ENABLED = original_layout
        oi._ESCALATOR = original_escalator


def case_audit_notes():
    print("\n[接线] 升级痕迹进审计表，且不污染剔除统计")
    good = "他站在门口没有动，风从走廊尽头灌进来，把纸页吹得哗哗响。" * 6
    bad = "||||||||||" * 30
    packed = oi._page_chunks(
        [(1, good), (2, bad), (3, good)],
        {1: "VLM 升级：本地 reject → VLM ok（分歧 18%）",
         2: "VLM 结果偏短（20/180 字，疑漏行），未采用"},
    )
    verdicts = [(row["page"], row["verdict"]) for row in packed["audit"]]
    check("救回的页记 escalated（不混进 reject）", (1, "escalated") in verdicts, verdicts)
    check("仍坏的页 verdict 保持不变", (2, "reject") in verdicts, verdicts)
    check("剔除计数只算真坏页", packed["rejected"] == 1, packed["rejected"])
    rejected_row = next(row for row in packed["audit"] if row["page"] == 2)
    check("坏页的 reasons 带上 VLM 说明",
          rejected_row["reasons"].startswith("VLM 结果偏短"), rejected_row["reasons"])
    check("好页照常产块", len(packed["chunks"]) >= 1, packed["chunks"])


def case_blank_page():
    print("\n[VLM] 空白页识别（实测纯白页会吐出「100%」）")
    import numpy as np
    from PIL import Image

    white = np.full((900, 700, 3), 255, dtype="uint8")
    check("纯白页判为空白", vlm.is_blank_page(white) is True)
    check("PIL 图同样能判", vlm.is_blank_page(Image.new("RGB", (100, 100), "white")) is True)

    inked = np.full((900, 700, 3), 255, dtype="uint8")
    for row in range(10):
        inked[100 + row * 60:120 + row * 60, 80:600] = 30
    check("有墨迹的页不会被误判成空白（误判＝静默丢正文）",
          vlm.is_blank_page(inked) is False)

    noisy = np.full((900, 700, 3), 255, dtype="uint8")
    noisy[::97, ::89] = 120
    check("带稀疏噪点的扫描空白页仍判空白（噪点不是墨迹）",
          vlm.is_blank_page(noisy) is True)
    check("拿不到像素时不做判定（宁可多试一次）", vlm.is_blank_page(object()) is False)


def case_escalator_blank():
    print("\n[VLM] 空白页短路：不烧预算、不给幻觉留缝")
    import numpy as np

    from app.scripts.corpus import text_clean

    calls = []

    class _Spy:
        available = True

        def recognize(self, image):
            calls.append(1)
            return "100%"

    escalator = vlm.Escalator(vlm.VlmConfig(endpoint="http://127.0.0.1:8118/v1", max_pages=3))
    escalator.client = _Spy()
    escalator.reset()

    #: 真实形态：空白页上本地引擎常读到零星噪点（破折号、间隔号、引号），这时
    #: judge_page 会判 reject ⇒ 若没有空白页短路，就会把这些噪点送进 VLM 换来幻觉。
    junk = "· · ·\n- ·"
    check("前提：这串噪点会被判可疑（否则测不到短路）",
          text_clean.judge_page(junk)["verdict"] != "ok", text_clean.judge_page(junk))
    text, note = escalator.consider(1, np.full((900, 700, 3), 255, dtype="uint8"), junk)
    check("空白页保留本地结果、不留痕", (text, note) == (junk, None), (text, note))
    check("空白页一次都没调模型", calls == [], calls)
    check("空白页没吃掉升级预算", escalator._budget == 3, escalator._budget)

    #: 另一层保护（既有行为）：完全空文本由 judge_page 直接判 ok，本就不升级
    check("空文本被判「空白页」不升级（原有保护仍在）",
          text_clean.judge_page("")["verdict"] == "ok")

    inked = np.full((900, 700, 3), 255, dtype="uint8")
    for row in range(10):
        inked[100 + row * 60:120 + row * 60, 80:600] = 30
    escalator.consider(2, inked, junk)
    check("有墨迹的可疑页照旧升级（别把短路写成一律跳过）", calls == [1], calls)


def case_strip_tokens():
    print("\n[VLM] 输出清洗：布局标记必须洗掉")
    from app.scripts.corpus import ocr_vlm_local as local

    raw = "<|LOC_12|>第一章 风起\n<|line_3|>他说：走吧。\n<|end_of_sentence|>"
    got = local.strip_layout_tokens(raw)
    check("洗掉全部 <|...|> 标记", "<|" not in got and "|>" not in got, repr(got))
    check("正文一字不改（只损失标记）", got == "第一章 风起\n他说：走吧。", repr(got))
    check("空输入返回空", local.strip_layout_tokens("") == "")
    check("连续空行收敛成一个", local.strip_layout_tokens("a\n\n\n\nb") == "a\n\nb")


def case_local_config():
    print("\n[VLM] 后端选择：local / http")
    keys = ("OCR_VLM_BACKEND", "OCR_VLM_MODEL_PATH", "OCR_VLM_ENDPOINT")
    saved = {key: os.environ.get(key) for key in keys}
    fake = BACKEND_PY / "tmp" / "_vlm_cfg_probe"
    try:
        os.environ["OCR_VLM_BACKEND"] = "local"
        os.environ.pop("OCR_VLM_ENDPOINT", None)
        os.environ["OCR_VLM_MODEL_PATH"] = str(BACKEND_PY / "tmp" / "_definitely_absent")
        config = vlm.VlmConfig.from_env()
        check("backend=local 被识别", config.local is True)
        check("权重不在盘上 ⇒ 整条通道不存在（不靠端点）", config.usable is False)
        check("describe 直接指出去下载", "ocr_vlm_fetch.py" in config.describe(), config.describe())

        fake.mkdir(parents=True, exist_ok=True)
        (fake / "config.json").write_text("{}", encoding="utf-8")
        os.environ["OCR_VLM_MODEL_PATH"] = str(fake)
        check("权重在盘上 ⇒ local 通道可用", vlm.VlmConfig.from_env().usable is True)

        os.environ["OCR_VLM_BACKEND"] = "inproc"
        check("写错的后端名退回 http（不静默变成第三态）",
              vlm.VlmConfig.from_env().backend == "http")
    finally:
        shutil.rmtree(fake, ignore_errors=True)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def case_local_engine():
    print("\n[VLM] 进程内后端：懒加载与显存门槛")
    from app.scripts.corpus import ocr_vlm_local as local

    engine = local.LocalEngine(model_path=BACKEND_PY / "tmp" / "_vlm_absent")
    check("构造不加载权重（否则 8 个 worker 各占一份显存）",
          engine.loaded is False and engine._model is None)
    try:
        engine.recognize(None)
        check("权重缺失时抛错", False)
    except RuntimeError as exc:
        check("权重缺失的报错指出去下载", "ocr_vlm_fetch.py" in str(exc), str(exc))
    check("失败被记成状态，不反复重试", engine.error is not None)

    fake = BACKEND_PY / "tmp" / "_vlm_gate_probe"
    fake.mkdir(parents=True, exist_ok=True)
    (fake / "config.json").write_text("{}", encoding="utf-8")
    original_free = local.free_gib
    try:
        local.free_gib = lambda: 0.5
        gated = local.LocalEngine(model_path=fake, min_free_gib=3.0)
        try:
            gated.recognize(None)
            check("显存不足时拒绝加载", False)
        except RuntimeError as exc:
            check("显存不足时拒绝加载并写明原因", "显存不足" in str(exc), str(exc))
        check("拒绝之后确实没加载权重", gated.loaded is False)
    finally:
        local.free_gib = original_free
        shutil.rmtree(fake, ignore_errors=True)


def case_fetch_inspect():
    print("\n[VLM] 权重体检（跑 OCR 前就能问「齐没齐」）")
    from app.scripts.corpus import ocr_vlm_fetch as fetch

    empty = fetch.inspect(BACKEND_PY / "tmp" / "_vlm_absent")
    check("空目录 ⇒ 未就绪", empty["ready"] is False and empty["files"] == 0, empty)
    check("报出缺哪些必需文件", "config.json" in empty["missing"], empty["missing"])

    fake = BACKEND_PY / "tmp" / "_vlm_fetch_probe"
    fake.mkdir(parents=True, exist_ok=True)
    try:
        for name in fetch.REQUIRED_FILES:
            (fake / name).write_text("{}", encoding="utf-8")
        partial = fetch.inspect(fake)
        check("只有配置没有权重 ⇒ 仍算未就绪",
              partial["ready"] is False and "*.safetensors" in partial["missing"], partial["missing"])
        (fake / "model.safetensors").write_bytes(b"x" * 1024)
        ready = fetch.inspect(fake)
        check("三份配置 + 权重 ⇒ 就绪", ready["ready"] is True, ready)
        check("统计出字节数（大页权重看得出下没下完）", ready["bytes"] > 0, ready["bytes"])
        check("列出权重文件名", ready["weights"] == ["model.safetensors"], ready["weights"])
    finally:
        shutil.rmtree(fake, ignore_errors=True)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    case_columns()
    case_single_column_identity()
    case_spanning_title()
    case_stray_page_number()
    case_broken_boxes()
    case_loopback()
    case_divergence()
    case_decide()
    case_client_payload()
    case_escalator_off()
    case_page_text()
    case_audit_notes()
    case_blank_page()
    case_escalator_blank()
    case_strip_tokens()
    case_local_config()
    case_local_engine()
    case_fetch_inspect()
    print(f"\nSUMMARY: {CHECKS - FAILED}/{CHECKS} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
