"""视频生成链路自检（``app/services/video_generation.py`` + S2 第 5 块）。

重点覆盖三条**只在这一层存在**的逻辑：

1. **prompt 洗白**：剥结构化标签 → 注入逐镜禁止变化 → 注入 ``[background_audio]``（幂等）；
2. **两处「空数组待遇不同」**：``referenceImageUrls: []`` 落 ``"[]"``（truthy 判据），
   而 ``referenceAudioUrls: []`` 落 ``NULL``（判的是 ``.length``）—— 极易写反；
3. **同步路径不回写分镜**：``handleVideoComplete`` 的 storyboardId 形参在同步路径是
   ``undefined`` ⇒ 分镜行不更新，但版本留档/QC 走「形参 ?? 记录值」仍然生效。

运行::

    ./.venv/Scripts/python.exe tests/video_generation_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("DATA_ROOT", tempfile.mkdtemp(prefix="vidgen_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.db import engine  # noqa: E402
from app.main import app  # noqa: E402
from app.core.models import (  # noqa: E402
    api_usage,
    asset_versions,
    storyboard_characters,
    storyboards,
    video_generations,
)
from app.core.response import now  # noqa: E402
from app.services import prompt_utils as pu  # noqa: E402
from app.services import video_generation as vg  # noqa: E402
from app.services import video_probe  # noqa: E402

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def run(coro):
    return asyncio.run(coro)


def _noop_spawn(coro):
    coro.close()


def main() -> int:  # noqa: C901
    client = TestClient(app)
    vg._spawn = _noop_spawn  # type: ignore[assignment]

    # ================= strip_video_prompt_tags =================
    check(
        "strip: 保留标签内内容、去掉标签本身",
        pu.strip_video_prompt_tags("<location>客栈</location>两人对视") == "客栈两人对视",
        pu.strip_video_prompt_tags("<location>客栈</location>两人对视"),
    )
    check(
        "strip: <n> / <n/> / <n /> 都换成换行",
        pu.strip_video_prompt_tags("A<n>B<n/>C<n />D") == "A\nB\nC\nD",
        pu.strip_video_prompt_tags("A<n>B<n/>C<n />D"),
    )
    check(
        "strip: 标签大小写不敏感",
        pu.strip_video_prompt_tags("<LOCATION>x</Location><ROLE>y</ROLE>") == "xy",
        pu.strip_video_prompt_tags("<LOCATION>x</Location><ROLE>y</ROLE>"),
    )
    check(
        "strip: 压缩行内空白 + 去换行两侧空白（顺序敏感）",
        pu.strip_video_prompt_tags("A    B  \n  C") == "A B\nC",
        repr(pu.strip_video_prompt_tags("A    B  \n  C")),
    )
    check(
        "strip: 空串原样（不做 strip）",
        pu.strip_video_prompt_tags("") == "",
    )
    check(
        "strip: 不碰没有任何标签的普通提示词（除空白规整）",
        pu.strip_video_prompt_tags("两人在客栈对视") == "两人在客栈对视",
    )
    check(
        "strip: <voice> 也剥掉但保留内容",
        pu.strip_video_prompt_tags("<voice>低沉</voice>") == "低沉",
    )

    # ================= validate_dialogue_character_consistency =================
    drama_id = client.post("/api/v1/dramas", json={"title": "SMOKE 视频父剧"}).json()["data"]["id"]
    episode_id = client.get(f"/api/v1/dramas/{drama_id}").json()["data"]["episodes"][0]["id"]
    sb_id = client.post("/api/v1/storyboards", json={"episode_id": episode_id, "title": "镜头"}).json()["data"]["id"]

    from app.core.models import characters  # noqa: PLC0415

    def insert_character(name: str, deleted: bool = False) -> int:
        values = {"drama_id": drama_id, "name": name}
        if "created_at" in characters.c:
            values["created_at"] = now()
        if "updated_at" in characters.c:
            values["updated_at"] = now()
        if deleted and "deleted_at" in characters.c:
            values["deleted_at"] = now()
        with engine.begin() as conn:
            return int(conn.execute(characters.insert().values(**values)).lastrowid)

    char_a = insert_character("林昭")
    char_b = insert_character("黑衣人")
    insert_character("已删角色", deleted=True)
    with engine.begin() as conn:
        conn.execute(storyboard_characters.insert().values(
            storyboard_id=sb_id, character_id=char_a))
        conn.execute(storyboard_characters.insert().values(
            storyboard_id=sb_id, character_id=char_b))

    # ⚠️ 抽取正则要求**台词 ≥ 8 字**（`[^:：\n]{8,}`）—— 台词太短的行压根抽不出说话人，
    #    所以这里的样例台词必须够长，否则测的是别的东西。
    with engine.begin() as conn:
        ok = pu.validate_dialogue_character_consistency(
            conn, "林昭：我这就过去看看情况\n黑衣人：不必了，你且在这里等着", sb_id)
    check(
        "对话: 全部命中 -> mismatches 空 + matchCount 2 + allMatch",
        ok == {"mismatches": [], "matchCount": 2, "allMatch": True}, ok,
    )
    with engine.begin() as conn:
        bad = pu.validate_dialogue_character_consistency(
            conn, "林昭：我这就过去看看情况\n张三：不必了，你且在这里等着", sb_id)
    check(
        "对话: 未关联的角色进 mismatches",
        bad["mismatches"] == ["张三"] and bad["matchCount"] == 1 and bad["allMatch"] is False, bad,
    )
    with engine.begin() as conn:
        todo = pu.validate_dialogue_character_consistency(
            conn, "林昭：我这就过去看看情况\n黑衣人：不必了", sb_id)
    check(
        "对话: 台词不足 8 字的行**抽不出说话人**（正则 `{8,}` 的门槛）",
        todo["matchCount"] == 1 and todo["mismatches"] == [], todo,
    )
    with engine.begin() as conn:
        narrator = pu.validate_dialogue_character_consistency(
            conn, "旁白：很久很久以前有一个传说\n林昭：我这就过去看看情况", sb_id)
    check(
        "对话: 旁白/画外音/narrator 跳过（不计 mismatch 也不计 match）",
        narrator == {"mismatches": [], "matchCount": 1, "allMatch": True}, narrator,
    )
    with engine.begin() as conn:
        with_narrator_en = pu.validate_dialogue_character_consistency(
            conn, "Narrator：a long long story begins here", sb_id)
    check("对话: narrator 大小写不敏感", with_narrator_en["matchCount"] == 0, with_narrator_en)
    with engine.begin() as conn:
        short = pu.validate_dialogue_character_consistency(conn, "林昭：嗯", sb_id)
    check(
        "对话: 台词短于 8 字不构成抽取 -> 回退 parse_dialogue_for_tts（说话人仍是林昭）",
        short["matchCount"] == 1 and short["mismatches"] == [], short,
    )
    with engine.begin() as conn:
        empty = pu.validate_dialogue_character_consistency(conn, "", sb_id)
    check("对话: 空台词 -> 直接放行", empty == {"mismatches": [], "matchCount": 0, "allMatch": True}, empty)
    with engine.begin() as conn:
        no_link = pu.validate_dialogue_character_consistency(
            conn, "林昭：我这就过去看看情况", 999999)
    check("对话: 分镜无关联角色 -> 全部算 mismatch", no_link["mismatches"] == ["林昭"], no_link)
    with engine.begin() as conn:
        deleted_name = pu.validate_dialogue_character_consistency(
            conn, "已删角色：我这就过去看看情况", sb_id)
    check("对话: 已软删角色不算命中", deleted_name["mismatches"] == ["已删角色"], deleted_name)

    # ================= probe_video_duration =================
    check(
        "probe: 文件不存在/无 ffprobe -> 0（与 TS 的 resolve(0) 一致，不抛错）",
        run(video_probe.probe_video_duration("static/videos/nope.mp4")) == 0,
    )
    _rel = video_probe._absolute_path("static/videos/a.mp4")
    check(
        "probe: 相对路径拼到 storageRoot 且**剥掉 static/ 前缀**（不出现两级 static）",
        Path(_rel).name == "a.mp4" and Path(_rel).parent.name == "videos"
        and _rel.count("static") == 1,
        _rel,
    )
    check(
        "probe: 绝对路径不被改写",
        video_probe._absolute_path("/abs/a.mp4") == "/abs/a.mp4",
    )

    # ================= 配置 + 入队 =================
    client.post("/api/v1/ai-configs", json={
        "service_type": "video", "provider": "volcengine", "base_url": "https://ark.example.com",
        "api_key": "k", "model": ["doubao-seedance-1-5-pro-251215"], "is_active": True,
    })

    # 给分镜配上「逐镜禁止变化」与环境音
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id).values(
            constraints="服装不得变化", sound_effect="远处犬吠"))

    base = {
        "storyboardId": sb_id,
        "dramaId": drama_id,
        "prompt": "<location>客栈</location><n>0-3秒 两人对视<n>3-6秒 转身",
        "force": True,
    }
    with engine.begin() as conn:
        video_id = run(vg.generate_video(conn, base))
        row = conn.execute(select(video_generations).where(video_generations.c.id == video_id)).first()
    check("enqueue: 返回自增 id 且 processing", isinstance(video_id, int) and row.status == "processing")
    check(
        "enqueue: prompt 已剥标签（换行分段）+ 注入禁止变化 + 注入 [background_audio]",
        row.prompt
        == "客栈\n0-3秒 两人对视\n3-6秒 转身 -- 逐镜禁止变化（画面中以下元素必须始终保持不变，"
           "不得增减或改变）: 服装不得变化 [background_audio] 远处犬吠",
        repr(row.prompt),
    )
    check(
        "enqueue: 默认 reference_mode='none' / duration=5 / aspect_ratio='16:9'",
        row.reference_mode == "none" and row.duration == 5 and row.aspect_ratio == "16:9",
        (row.reference_mode, row.duration, row.aspect_ratio),
    )
    check("enqueue: model/provider 来自配置", row.provider == "volcengine" and row.model.startswith("doubao"), (row.provider, row.model))

    # [background_audio] 幂等
    with engine.begin() as conn:
        already = run(vg.generate_video(conn, {**base, "prompt": "画面 [background_audio] 已有标记"}))
        r2 = conn.execute(select(video_generations).where(video_generations.c.id == already)).first()
    check(
        "enqueue: prompt 已含 [background_audio] 则**不重复注入**（幂等）",
        r2.prompt.count("[background_audio]") == 1, repr(r2.prompt),
    )

    # 空数组待遇不同：图片参考图是 truthy 判据、参考音频是 length 判据
    with engine.begin() as conn:
        mixed = run(vg.generate_video(conn, {
            **base, "referenceImageUrls": [], "referenceAudioUrls": [],
        }))
        r3 = conn.execute(select(video_generations).where(video_generations.c.id == mixed)).first()
    check(
        "enqueue: referenceImageUrls=[] -> 落 \"[]\"（truthy 判据：空数组是真值）",
        r3.reference_image_urls == "[]", repr(r3.reference_image_urls),
    )
    check(
        "enqueue: referenceAudioUrls=[] -> 落 NULL（判的是 .length）—— 两者**不同**",
        r3.reference_audio_urls is None, repr(r3.reference_audio_urls),
    )
    with engine.begin() as conn:
        audio = run(vg.generate_video(conn, {**base, "referenceAudioUrls": ["a.mp3"]}))
        r4 = conn.execute(select(video_generations).where(video_generations.c.id == audio)).first()
    check("enqueue: 非空参考音频落紧凑 JSON", r4.reference_audio_urls == '["a.mp3"]', repr(r4.reference_audio_urls))

    # sceneType 空串 -> null；referenceMode 空串 -> 'none'
    with engine.begin() as conn:
        blank = run(vg.generate_video(conn, {**base, "sceneType": "", "referenceMode": ""}))
        r5 = conn.execute(select(video_generations).where(video_generations.c.id == blank)).first()
    check(
        "enqueue: sceneType 空串 -> null、referenceMode 空串 -> 'none'（都是 truthy 回落）",
        r5.scene_type is None and r5.reference_mode == "none", (r5.scene_type, r5.reference_mode),
    )
    with engine.begin() as conn:
        route = run(vg.generate_video(conn, {**base, "route": "ref2va", "routeReason": "对话镜头"}))
        r6 = conn.execute(select(video_generations).where(video_generations.c.id == route)).first()
    check("enqueue: route / routeReason 落库", r6.route == "ref2va" and r6.route_reason == "对话镜头")

    # take 预算
    check("enqueue: 提交即消耗一次 take", _take_count(sb_id) >= 5, _take_count(sb_id))

    # ================= 参考图归一化 =================
    check("ref: 空/坏 JSON -> []",
          run(vg._normalize_video_reference_urls(None)) == []
          and run(vg._normalize_video_reference_urls("bad")) == [])
    check(
        "ref: data URL 与 http URL 原样保留、去重去空",
        run(vg._normalize_video_reference_urls(
            '["data:image/png;base64,AA", "data:image/png;base64,AA", "", "https://x.com/a.png"]'))
        == ["data:image/png;base64,AA", "https://x.com/a.png"],
        run(vg._normalize_video_reference_urls(
            '["data:image/png;base64,AA", "data:image/png;base64,AA", "", "https://x.com/a.png"]')),
    )
    check("ref: 单个空值 -> None", run(vg._normalize_video_reference_url("")) is None)
    check(
        "ref: 读不出来的 static/ 文件 -> None（会被外层丢掉）",
        run(vg._normalize_video_reference_url("static/images/definitely-missing.png")) is None,
    )

    # ================= 完成收尾 =================
    async def _fake_download(url, sub_dir):
        return f"static/{sub_dir}/fake.mp4"

    vg.download_file = _fake_download  # type: ignore[assignment]

    qc_calls: list[tuple[object, int]] = []
    vg._run_qc_after_video_complete = lambda sb, vid: qc_calls.append((sb, vid))  # type: ignore[assignment]

    # ① 轮询路径：传了 storyboardId -> 会更新分镜行。
    #    轮询路径传的 duration 是 **None**（异步提供商不返回时长）⇒ 走 ffprobe 探测分支，
    #    这里把探测打桩成 42 秒，验证「探测值会被回填」。
    async def _fake_probe(local_path):
        return 42

    vg.probe_video_duration = _fake_probe  # type: ignore[assignment]

    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id).values(
            video_url=None, duration=None))
        poll_id = run(vg.generate_video(conn, base))
    run(vg._handle_video_complete(poll_id, "https://cdn.test/v.mp4", None, sb_id))
    with engine.begin() as conn:
        vrow = conn.execute(select(video_generations).where(video_generations.c.id == poll_id)).first()
        sb_row = conn.execute(select(storyboards).where(storyboards.c.id == sb_id)).first()
        ver = conn.execute(select(asset_versions).where(
            asset_versions.c.asset_type == "storyboard", asset_versions.c.asset_id == sb_id)).all()
    check(
        "complete(轮询): 记录 completed + videoUrl + localPath + completedAt",
        vrow.status == "completed" and vrow.video_url == "https://cdn.test/v.mp4"
        and vrow.local_path == "static/videos/fake.mp4" and vrow.completed_at,
        (vrow.status, vrow.video_url, vrow.local_path, vrow.completed_at),
    )
    check(
        "complete(轮询): 分镜 video_url 被回写，且**探测到的时长（42s）被回填**",
        sb_row.video_url == "static/videos/fake.mp4" and sb_row.duration == 42,
        (sb_row.video_url, sb_row.duration),
    )
    check(
        "complete: 资产留档 mediaType=video（frame_type 为空）",
        any(v.media_type == "video" for v in ver), [(v.media_type, v.frame_type) for v in ver],
    )
    check("complete: 触发了 QC 钩子（当前为未迁占位）", len(qc_calls) == 1 and qc_calls[0][1] == poll_id, qc_calls)

    # ② 同步路径：**不传** storyboardId -> 分镜行不更新，但版本/QC 仍生效
    vg._run_qc_after_video_complete = lambda sb, vid: qc_calls.append((sb, vid))  # type: ignore[assignment]
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id).values(
            video_url=None, duration=None))
        sync_id = run(vg.generate_video(conn, base))
    qc_calls.clear()
    run(vg._handle_video_complete(sync_id, "https://cdn.test/s.mp4", 7))
    with engine.begin() as conn:
        srow = conn.execute(select(video_generations).where(video_generations.c.id == sync_id)).first()
        sb_sync = conn.execute(select(storyboards).where(storyboards.c.id == sb_id)).first()
        ver_sync = conn.execute(select(asset_versions).where(
            asset_versions.c.asset_type == "storyboard",
            asset_versions.c.asset_id == sb_id,
            asset_versions.c.generation_id == sync_id)).all()
    check(
        "complete(同步): 分镜行**不**被更新（storyboardId 形参为 None）—— 保真，别「修好」",
        sb_sync.video_url is None and sb_sync.duration is None,
        (sb_sync.video_url, sb_sync.duration),
    )
    check(
        "complete(同步): 但**版本留档仍然生效**（走「形参 ?? 记录 storyboardId」）",
        len(ver_sync) == 1, len(ver_sync),
    )
    check("complete(同步): QC 钩子也仍然生效", len(qc_calls) == 1 and qc_calls[0][0] == sb_id, qc_calls)
    check("complete(同步): 记录 completed", srow.status == "completed" and srow.local_path == "static/videos/fake.mp4")

    # 探测也拿不到（=0）时**不写 duration 键**，旧值保留
    async def _zero_probe(local_path):
        return 0

    vg.probe_video_duration = _zero_probe  # type: ignore[assignment]

    # ③ duration 为 None 且 ffprobe 探不到 -> 分镜**不写 duration 键**（保留旧值）
    with engine.begin() as conn:
        conn.execute(storyboards.update().where(storyboards.c.id == sb_id).values(
            video_url=None, duration=9))
        keep_id = run(vg.generate_video(conn, base))
    run(vg._handle_video_complete(keep_id, "https://cdn.test/k.mp4", None, sb_id))
    with engine.begin() as conn:
        sb_keep = conn.execute(select(storyboards).where(storyboards.c.id == sb_id)).first()
    check(
        "complete: 拿不到时长时**不写** duration 键（`?? undefined` 丢键），旧值保留",
        sb_keep.video_url == "static/videos/fake.mp4" and sb_keep.duration == 9,
        (sb_keep.video_url, sb_keep.duration),
    )

    # ④ meta 里的 duration 取的是**记录里的 duration**（不是本次传入/探测到的那个）
    with engine.begin() as conn:
        rows = conn.execute(select(asset_versions).where(
            asset_versions.c.generation_id == sync_id)).all()
    check(
        "complete: meta 用**记录里的** duration（`gen?.duration`，默认 5）而非本次入参 7",
        any('"duration":5' in (r.meta or "") for r in rows), [r.meta for r in rows],
    )
    check("complete: submitted 用量已收口", _submitted_count() == 0, _submitted_count())

    # ================= Vidu：提交后不轮询 =================
    client.post("/api/v1/ai-configs", json={
        "service_type": "video", "provider": "vidu", "base_url": "https://vidu.example.com",
        "api_key": "k", "model": ["viduq3-turbo"], "is_active": True,
    })
    from app.services.adapters.registry import get_video_adapter  # noqa: PLC0415

    check(
        "vidu: 轮询请求是伪协议 URL、解析恒为 processing",
        get_video_adapter("vidu").build_poll_request({}, "t")["url"] == "vidu://no-polling-endpoint"
        and get_video_adapter("vidu").parse_poll_response({})["status"] == "processing",
    )

    # ================= 崩溃恢复 =================
    with engine.begin() as conn:
        expired = run(vg.generate_video(conn, base))
        conn.execute(video_generations.update().where(video_generations.c.id == expired).values(
            task_id="t1", updated_at="2020-01-01T00:00:00.000Z"))
        no_task = run(vg.generate_video(conn, base))
        unknown = run(vg.generate_video(conn, base))
        conn.execute(video_generations.update().where(video_generations.c.id == unknown).values(
            provider="nope", task_id="t2", updated_at=now()))
        vidu_wait = run(vg.generate_video(conn, base))
        conn.execute(video_generations.update().where(video_generations.c.id == vidu_wait).values(
            provider="vidu", task_id="t3", updated_at=now()))
    vg.recover_video_tasks_on_startup()
    with engine.begin() as conn:
        states = {
            r[0]: (r[1], r[2])
            for r in conn.execute(select(video_generations.c.id, video_generations.c.status,
                                         video_generations.c.error_msg).where(
                video_generations.c.id.in_([expired, no_task, unknown, vidu_wait]))).all()
        }
    check("recover: 超时 -> failed", states[expired][0] == "failed" and "expired" in (states[expired][1] or ""), states[expired])
    check("recover: 无 taskId -> failed",
          states[no_task][0] == "failed" and "before task submit" in (states[no_task][1] or ""), states[no_task])
    check("recover: 未知 provider -> failed",
          states[unknown][0] == "failed" and "Unknown video provider" in (states[unknown][1] or ""), states[unknown])
    check(
        "recover: **vidu 保持 processing**（Webhook 型，外部回调仍会到达）",
        states[vidu_wait][0] == "processing", states[vidu_wait],
    )

    # ================= 汇总 =================
    client.delete(f"/api/v1/dramas/{drama_id}")

    failed = [item for item in _RESULTS if not item[1]]
    for name, ok, detail in _RESULTS:
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed")
    return 1 if failed else 0


def _take_count(storyboard_id: int) -> int:
    with engine.begin() as conn:
        row = conn.execute(
            select(storyboards.c.take_count).where(storyboards.c.id == storyboard_id)
        ).first()
    return row[0]


def _submitted_count() -> int:
    with engine.begin() as conn:
        return len(conn.execute(select(api_usage).where(api_usage.c.status == "submitted")).all())


if __name__ == "__main__":
    raise SystemExit(main())
