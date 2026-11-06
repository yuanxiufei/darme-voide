"""Music3 语义塔的**装载对账** ＋ H3 双流路由闸门 ✓（2026-11-02）。

这一套盯两件「不报错但结果全错」的事 ✗✗：

① **键对账**：``te_music3.build_tower()`` 在 meta 上建的壳，``state_dict`` 必须与盘上
   那份 ``pruned_int8_convrot`` 的**参数键逐一对齐** ✓（327/327 ✓）——
   少一个键 ⇒ 装载器报「缺键」但如果壳里**多**一个没人用的参数，装载会**静默留零** ✓✗
   （零权重层不报错、只让整首歌变成噪声 ✓✗）；同理 ``comfy_quant``/``weight_scale``
   这两类**配对键**不算参数，混进来就会让对账永远红 ✗。
② **int8 计划全命中**：160 层 ``int8_tensorwise + convrot(256)`` ✓，计划点名的层若有一个
   对不上模块路径 ⇒ 那层按 bf16 物化 ✓ ⇒ 8.6 GB 变 17 GB ⇒ 装载期 OOM ✓✗（而报错只说
   「显存不足」，指不到「一层没换壳」✓✗）。

再加一组 **H3 路由闸门**判据 ✓：``stage=h3`` 缺 ``audioLatentMode`` / ``temporalCompression``
或显式 ``attachAudioVae=false`` ⇒ **提交时就 400** ✓✗（不给这两格 ⇒ 装载 20 秒之后
采样第 0 步才炸，报错还读起来像「主干坏了」✗）。
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")   # cp936 控制台下首行 ✓ 直接 UnicodeEncodeError ✓

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASSED = 0
FAILED = 0


def check(name: str, ok: bool, detail: object = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}　{detail}")


def summary() -> int:
    print(f"SUMMARY: {PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


TE_PATH = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\text_encoders"
               r"\minimax_music3_text_encoder_pruned_int8_convrot.safetensors")


def case_key_reconciliation() -> None:
    from app.services.engine import safetensors as st
    from app.services.engine import te_music3 as tm
    from app.services.engine import torch_backend as tb

    if not TE_PATH.exists():
        check("⑳ᴷ TE 权重在盘上（不在 ⇒ 这组对账跳过，别报假红 ✗）", True, "权重不在盘上 ⇒ 跳过")
        return
    info = st.inspect(TE_PATH)
    tower = tm.build_tower()                     # meta 壳 ✓ 一块实存不占 ✓
    own = set(tower.state_dict().keys())
    file_keys = set(info.tensors)
    non_param = {k for k in file_keys
                 if k == "tokenizer_json" or k.endswith(".comfy_quant")}
    scaled = {k for k in file_keys if k.endswith(".weight_scale")}
    expected = file_keys - non_param - scaled
    check("⑳ᴷ 参数键**逐一对齐**（327/327 ✓ 多一个参数 ⇒ 装载静默留零 ✓✗）",
          own == expected, (sorted(expected - own)[:3], sorted(own - expected)[:3]))
    check("⑳ᴷ 配对键不混进参数账（`comfy_quant` / `weight_scale` 不算 ✗）",
          not (non_param | scaled) & own, sorted((non_param | scaled) & own)[:3])

    plan = tb.int8_scale_plan(info)
    names = dict(tower.named_modules())
    unhit = [name for name in plan if name not in names]
    check("⑳ᴷ int8 计划 **160 层全命中**（一个不中 ⇒ 该层按 bf16 物化 ⇒ 8.6 GB 变 17 GB ✓✗）",
          len(plan) == 160 and not unhit, (len(plan), unhit[:3]))

    # ⚠️ 反向（不许见谁都喊 ✗）：计划只该点名真带 scale 的那些层
    check("⑳ᴷ 计划宽 = 盘中 `weight_scale` 键数（不许凭空多 ✗）",
          len(plan) == len(scaled), (len(plan), len(scaled)))


def case_tokenizer() -> None:
    """词表内嵌 ✓ 特殊 token id 逐个核 ✗（换一个数 ⇒ 提示词模板全错且不报错 ✓✗）。"""
    from app.services.engine import te_music3 as tm

    if not TE_PATH.exists():
        check("⑳ᴷ 词表核验（权重不在盘上 ⇒ 跳过 ✓）", True, "跳过")
        return
    # 只读 tokenizer_json 那一格（不装那 9.2 GB ✓）
    from tokenizers import Tokenizer

    from app.services.engine import safetensors as st
    raw = st.read_tensor_bytes(TE_PATH, "tokenizer_json", max_bytes=16 * 1024 * 1024)
    hf = Tokenizer.from_str(raw.decode("utf-8"))

    class _Compat:
        """`validate_tokenizer` 吃的是 comfy 风格（`.token_id()` + `encode()` 直接给 id 列表 ✓）✗，
        而 HF 的 `tokenizers.Tokenizer` 这两样都不是 ✓ —— 薄薄包一层，别改被测模块 ✗。"""

        def token_id(self, token: str):
            return hf.token_to_id(token)

        def encode(self, text: str, **kwargs):  # noqa: ARG002
            return hf.encode(text, add_special_tokens=False).ids

    problems = tm.validate_tokenizer(_Compat())
    check("⑳ᴷ 词表通过 `validate_tokenizer` ✓（特殊 token id 全对 ✓ —— 对不上 ⇒ 整条提示"
          "被切普通文本、歌照样出但与提示无关 ✓✗）", not problems, problems[:3])
    ids = hf.encode(tm.build_prompt("pop", "[start]\nla"), add_special_tokens=False).ids
    check("⑳ᴷ 提示词能编且首 token 是 `<|im_start|>`(151644 ✓)",
          bool(ids) and ids[0] == tm.SPECIAL_TOKEN_IDS["<|im_start|>"], ids[:4])


def case_h3_gate() -> None:
    """H3 双流的两格口径 + 音频 VAE：缺任一项 ⇒ **提交时就 400** ✗✗。"""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    base = {"stage": "h3", "prompt": "x", "megapixels": 0.1, "ratio": "1:1",
            "seconds": 1.4, "seed": 1, "wait": False}
    r1 = client.post("/api/v1/engine/generate", json=dict(base))
    check("⑳ᴷ h3 缺 audioLatentMode ⇒ 400 且点名要补的键 ✓",
          r1.status_code == 400 and "audioLatentMode" in r1.text, r1.status_code)
    r2 = client.post("/api/v1/engine/generate", json={**base, "audioLatentMode": "round"})
    check("⑳ᴷ h3 缺 temporalCompression ⇒ 400 点名第二格 ✓",
          r2.status_code == 400 and "temporalCompression" in r2.text, r2.status_code)
    r3 = client.post("/api/v1/engine/generate",
                     json={**base, "audioLatentMode": "round",
                           "temporalCompression": 4, "attachAudioVae": False})
    check("⑳ᴷ h3 显式关掉 attachAudioVae ⇒ 400（H3 没有单流前向 ✓✗）",
          r3.status_code == 400 and "音频 VAE" in r3.text, r3.status_code)
    r4 = client.post("/api/v1/engine/generate", json={**base, "dryRun": True})
    check("⑳ᴷ dryRun 不受闸门拦 ✓（干跑不装载 ⇒ 不需要口径 ✓）",
          r4.status_code == 200, r4.status_code)


def main() -> int:
    case_key_reconciliation()
    case_tokenizer()
    case_h3_gate()
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
