"""S2 第 3+4 块 + 视觉图谱自检。

覆盖三块**直接决定生成结果**的逻辑：

1. **视觉图谱**（``visual_graph.py``）：中文术语 → 英文电影术语的匹配规则（精确优先 / 包含兜底 /
   单字不兜底）；41 个节点的 ``zh`` 别名要与 DB 实际取值对得上，漂移守卫另做逐条比对；
2. **分镜 prompt 构建**（``build_storyboard_image_prompt`` / ``build_storyboard_video_prompt``）：
   分句符、末尾拼装的**逗号与空格位置**（差一个字符就是另一段 prompt）；
3. **DB 读取辅助**：镜头级服装变体优先、参考图三源合并、参考音频**最多 3 条**且要转绝对 URL。

运行::

    ./.venv/Scripts/python.exe tests/prompt_storyboard_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="s2prompt_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import (  # noqa: E402
    characters,
    prop_templates,
    scenes,
    storyboard_characters,
    storyboard_props,
    storyboards,
)
from app.core.response import now  # noqa: E402
from app.services import prompt_utils as pu  # noqa: E402
from app.services import visual_graph as vg  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def main() -> int:  # noqa: C901
    client = TestClient(app)

    # ================= 视觉图谱：匹配规则 =================
    check(
        "图谱: 41 个节点 / 四类齐全",
        sum(len(nodes) for nodes in vg.VISUAL_GRAPH.values()) == 41
        and set(vg.VISUAL_GRAPH) == {"shot_size", "composition", "movement", "lighting"},
        {k: len(v) for k, v in vg.VISUAL_GRAPH.items()},
    )
    check("图谱: 精确匹配", vg.resolve_visual_term("近景") == "close-up")
    check("图谱: 别名也能命中（近景镜头）", vg.resolve_visual_term("近景镜头") == "close-up")
    check(
        "图谱: 包含匹配兜底（快速横移 -> 横移 -> lateral tracking）",
        vg.resolve_visual_term("快速横移") == "lateral tracking",
        vg.resolve_visual_term("快速横移"),
    )
    check(
        "图谱: 单字术语**不参与**包含兜底（避免误命中）",
        vg.resolve_visual_term("他站在那里摇了一会儿") is None
        and vg.resolve_visual_term("摇镜") == "pan",
        vg.resolve_visual_term("他站在那里摇了一会儿"),
    )
    check(
        "图谱: 未命中返回 None（调用方保留原文）",
        vg.resolve_visual_term("随便什么") is None
        and vg.resolve_visual_term("") is None
        and vg.resolve_visual_term(None) is None,
    )
    check(
        "图谱: resolveVisualTerms 只在**类别相符**时输出英文",
        vg.resolve_visual_terms("movement", ["推镜", "近景", "自定义运镜", None])
        == ["dolly in", "近景", "自定义运镜"],
        vg.resolve_visual_terms("movement", ["推镜", "近景", "自定义运镜", None]),
    )
    check(
        "图谱: listVisualTerms 给出该类全部英文术语",
        vg.list_visual_terms("shot_size")[0] == "extreme wide shot" and len(vg.list_visual_terms("lighting")) == 11,
    )
    check(
        "图谱: getVisualGraph 带类别只返回该类",
        set(vg.get_visual_graph("movement")) == {"movement"} and len(vg.get_visual_graph()) == 4,
    )
    check(
        "图谱: 引导文本按风格拼推荐项（悬疑 -> 低光/硬光）",
        "本剧推荐灯光：低光 / 硬光 / 侧光 / 顶光。" in vg.build_visual_graph_guidance("悬疑", None),
    )
    check(
        "图谱: 风格可叠加（悬疑 + 都市 两组提示都在）",
        "框架构图" in vg.build_visual_graph_guidance("悬疑都市", None)
        and "自然光" in vg.build_visual_graph_guidance("悬疑都市", None),
    )
    check(
        "图谱: 无风格匹配时不出推荐行，但固定骨架仍在",
        "本剧推荐景别" not in vg.build_visual_graph_guidance("完全无关", None)
        and "5. 一致性与安全区" in vg.build_visual_graph_guidance(None, None),
    )

    # ================= build_storyboard_image_prompt =================
    check(
        "分镜图: 底线 —— 什么信息都没有时回退 cinematic shot + 画风后缀",
        pu.build_storyboard_image_prompt({})
        == f"cinematic shot{pu.build_storyboard_art_style_suffix(None)}",
        pu.build_storyboard_image_prompt({}),
    )
    prompt = pu.build_storyboard_image_prompt({
        "sceneDescription": "雨夜街道", "location": "客栈", "characterDescription": "林昭",
        "shotType": "近景", "cameraAngle": "俯拍", "storyboardDescription": "视觉描述", "description": "叙事描述",
    })
    check(
        "分镜图: sceneDescription 优先于 location；storyboardDescription 优先于 description",
        prompt.startswith("Scene: 雨夜街道. Characters: 林昭. Camera: close-up, 俯拍. 视觉描述")
        and "客栈" not in prompt and "叙事描述" not in prompt,
        prompt[:110],
    )
    check(
        "分镜图: 景别/机位走图谱翻译，未命中则保留原文（中文混入英文 prompt 是已知代价）",
        "Camera: close-up, 俯拍" in pu.build_storyboard_image_prompt({"shotType": "近景", "cameraAngle": "俯拍"}),
    )
    check(
        "分镜图: 只有机位时也拼 Camera 段",
        pu.build_storyboard_image_prompt({"cameraAngle": "俯拍"}).startswith("Camera: 俯拍"),
    )
    check(
        "分镜图: 段间用 '. ' 连接、末尾挂画风后缀",
        pu.build_storyboard_image_prompt({"location": "客栈", "description": "对视"}).endswith(
            pu.build_storyboard_art_style_suffix(None)
        ),
    )

    # ================= build_storyboard_video_prompt =================
    video = pu.build_storyboard_video_prompt({
        "characterAppearances": ["林昭: 青衫", "黑衣人: 黑袍"],
        "scenePrompt": "客栈", "storyboardDescription": "对峙", "action": "拔剑", "movement": "推镜",
    })
    check(
        "分镜视频: 角色段用 '; ' 连接 + 强调一致性文案",
        video.startswith("Characters (maintain strict visual consistency): 林昭: 青衫; 黑衣人: 黑袍. Setting: 客栈. 对峙. Action: 拔剑. "
                          "Camera movement: dolly in"),
        video[:150],
    )
    check(
        "分镜视频: 声音策略原文（无 BGM，后期混音）",
        "Sound: production audio only, no background music in frame (music is mixed in post-production)." in video,
    )
    check(
        "分镜视频: 无画风时退回历史默认视觉层（VIDEO + MASTER 两段）",
        video.endswith(f"{pu.VISUAL_STYLE_VIDEO}, {pu.VISUAL_STYLE_MASTER}")
        and "cinematic motion" in video,
        video[-80:],
    )
    check(
        "分镜视频: 命中画风时用画风层（**去掉逗号前缀**，改用空格衔接）",
        "  " not in pu.build_storyboard_video_prompt({"description": "x", "dramaStyle": "ink-wash"})
        and pu.build_storyboard_video_prompt({"description": "x", "dramaStyle": "ink-wash"}).count(",")
        >= 1,
    )
    # ⚠️ 声音策略前面的分隔符是 **`, `**（各段 join 后的收尾逗号），不是句点 —— 第一版断言写错了
    ui_prompt = pu.build_storyboard_video_prompt({"description": "她看着手机屏幕"})
    check(
        "分镜视频: 命中屏幕元素才注入 UI 留白规则（紧跟在声音策略的句点之后、带前置逗号）",
        f", {pu._SOUND_POLICY}., {pu.UI_OVERLAY_RULE} " in ui_prompt,
        ui_prompt[:130],
    )
    check(
        "分镜视频: 不含屏幕元素时不注入 UI 规则（句点后直接接画风层）",
        f", {pu._SOUND_POLICY}. " in pu.build_storyboard_video_prompt({"description": "两人对视"}),
        pu.build_storyboard_video_prompt({"description": "两人对视"}),
    )
    check(
        "分镜视频: 环境音标记前置『空格』且只取 trim 后的值",
        pu.build_storyboard_video_prompt({"description": "x", "backgroundAudio": "  远处犬吠  "})
        .startswith(f"x, {pu._SOUND_POLICY}. [background_audio] 远处犬吠 "),
        pu.build_storyboard_video_prompt({"description": "x", "backgroundAudio": "  远处犬吠  "})[:90],
    )
    check(
        "分镜视频: backgroundAudio 纯空白不注入",
        "[background_audio]" not in pu.build_storyboard_video_prompt({"description": "x", "backgroundAudio": "   "}),
    )
    check(
        "分镜视频: storyboardDescription 优先于 description",
        pu.build_storyboard_video_prompt({"storyboardDescription": "A", "description": "B"}).startswith("A")
        and "B" not in pu.build_storyboard_video_prompt({"storyboardDescription": "A", "description": "B"}),
    )

    # ================= DB 读取辅助 =================
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 分镜上下文"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]
    sb_id = client.post("/api/v1/storyboards", json={"episode_id": episode_id, "title": "镜头"}).json()["data"]["id"]
    scene_id = client.post("/api/v1/scenes", json={
        "drama_id": drama_id, "location": "客栈", "time": "夜晚", "prompt": "木质结构",
    }).json()["data"]["id"]

    def insert_character(name: str, **extra) -> int:
        values: dict[str, object] = {"drama_id": drama_id, "name": name}
        for column in ("created_at", "updated_at"):
            if column in characters.c:
                values[column] = now()
        values.update(extra)
        with engine.begin() as conn:
            return int(conn.execute(characters.insert().values(**values)).lastrowid)

    char_with_variation = insert_character(
        "林昭", clothing="青衫", appearance="短发侠客",
        image_url="static/images/main.png",
        variations=json.dumps([{"name": "红衣", "imageUrl": "static/images/red.png"}]),
        voice_sample_url="static/audio/linzhao.mp3",
    )
    char_plain = insert_character("黑衣人", image_url="static/images/black.png")
    char_no_image = insert_character("路人")
    with engine.begin() as conn:
        conn.execute(storyboard_characters.insert().values(
            storyboard_id=sb_id, character_id=char_with_variation, costume="红衣"))
        conn.execute(storyboard_characters.insert().values(
            storyboard_id=sb_id, character_id=char_plain))
    # 场景关联（分镜的 scene_id 用 API 也可，但直接写更稳）
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id).values(scene_id=scene_id))
        conn.execute(scenes.update().where(scenes.c.id == scene_id).values(image_url="static/images/scene.png"))

    with engine.begin() as conn:
        appearances = pu.get_storyboard_character_appearances(conn, sb_id)
        scene_desc = pu.get_storyboard_scene_description(conn, sb_id)
        char_urls = pu.get_storyboard_character_image_urls(conn, sb_id)
        ref_images = pu.get_storyboard_reference_images(conn, sb_id)
        audio_urls = pu.get_storyboard_reference_audio_urls(conn, sb_id)
    check(
        "上下文: 角色外观列表走 buildCharacterAppearanceText（用 ': ' 拼接）",
        len(appearances) == 2 and appearances[0].startswith("林昭:") and "黑衣人" in appearances[1],
        appearances,
    )
    check(
        "上下文: 镜头级服装变体注入外观（wearing 红衣 优先于 clothing 青衫）",
        "wearing 红衣" in appearances[0] and "青衫" not in appearances[0],
        appearances[0],
    )
    # ⚠️ 该场景有 `prompt`（木质结构）⇒ `build_scene_image_prompt` 的 **prompt 分支优先**，
    #    location+time 分支不会生效（那条分支由第 2 块的用例单独覆盖）。
    check(
        "上下文: 场景描述走 buildSceneImagePrompt，且有 prompt 时 prompt 优先",
        scene_desc is not None and scene_desc.startswith("木质结构, "), scene_desc,
    )
    check(
        "上下文: 角色图 —— 有变体且变体有立绘时用变体图",
        char_urls[0] == "static/images/red.png" and char_urls[1] == "static/images/black.png",
        char_urls,
    )
    check(
        "上下文: 无立绘的角色不进列表（char_no_image 已加入但无图）",
        len(char_urls) == 2, char_urls,
    )
    check(
        "上下文: 参考图 = 角色图 + 场景图",
        ref_images == ["static/images/red.png", "static/images/black.png", "static/images/scene.png"],
        ref_images,
    )
    check(
        "上下文: 参考音频转**绝对 URL**（PUBLIC_BASE_URL 默认 5789）",
        audio_urls == ["http://localhost:5789/static/audio/linzhao.mp3"],
        audio_urls,
    )
    os.environ["PUBLIC_BASE_URL"] = "http://localhost:5790/"
    check(
        "上下文: PUBLIC_BASE_URL 可覆盖且尾部斜杠被剥（不产生 //static）",
        pu.to_public_media_url("static/audio/a.mp3") == "http://localhost:5790/static/audio/a.mp3",
        pu.to_public_media_url("static/audio/a.mp3"),
    )
    check(
        "上下文: 已是 http/data 的路径原样返回",
        pu.to_public_media_url("https://cdn.test/a.mp3") == "https://cdn.test/a.mp3"
        and pu.to_public_media_url("data:audio/mp3;base64,AA") == "data:audio/mp3;base64,AA",
    )
    os.environ.pop("PUBLIC_BASE_URL", None)

    # 物品参考图（连续性状态机 v3）
    prop_id = client.post("/api/v1/props", json={
        "drama_id": drama_id, "name": "玉佩", "category": "信物",
    }).json()["data"]["id"]
    with engine.begin() as conn:
        conn.execute(prop_templates.update().where(prop_templates.c.id == prop_id).values(
            image_url="static/images/prop.png"))
        conn.execute(storyboard_props.insert().values(storyboard_id=sb_id, prop_id=prop_id))
        refs_with_prop = pu.get_storyboard_reference_images(conn, sb_id)
    check(
        "上下文: 参考图加上本镜物品设定图（排在角色与场景之后）",
        refs_with_prop[-1] == "static/images/prop.png" and len(refs_with_prop) == 4,
        refs_with_prop,
    )

    # 参考音频最多 3 条。
    # ⚠️ 注意**不要再把 `engine.begin()` 嵌进另一个 `engine.begin()`**：SQLite 只有一个写者，
    #    外层事务持着写锁时内层新连接会直接 `database is locked`（本用例第一版就这么挂的）。
    for index in range(4):
        cid = insert_character(f"群演{index}", voice_sample_url=f"static/audio/v{index}.mp3")
        with engine.begin() as conn:
            conn.execute(storyboard_characters.insert().values(storyboard_id=sb_id, character_id=cid))
    with engine.begin() as conn:
        capped = pu.get_storyboard_reference_audio_urls(conn, sb_id)
    check(
        "上下文: 参考音频**最多 3 条**（H3 Ref2VA 上限）",
        len(capped) == 3 and all(u.startswith("http://localhost:5789/static/audio/") for u in capped),
        capped,
    )

    # 无关联时的空值
    other_sb = client.post("/api/v1/storyboards", json={
        "episode_id": episode_id, "title": "空镜",
    }).json()["data"]["id"]
    with engine.begin() as conn:
        check(
            "上下文: 无关联角色/场景时返回空列表与 None",
            pu.get_storyboard_character_appearances(conn, other_sb) == []
            and pu.get_storyboard_scene_description(conn, other_sb) is None
            and pu.get_storyboard_reference_images(conn, other_sb) == []
            and pu.get_storyboard_reference_audio_urls(conn, other_sb) == [],
        )

    client.delete(f"/api/v1/dramas/{drama_id}")

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
