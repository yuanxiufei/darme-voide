"""SEEDVR2 **修复支路**自检 ✓：plan → 17 通道条件 → denoise → decode → 落盘 ✓。

这条支路是本仓第一个「**输入是视频、输出也是视频**」的族 ✓（``h3`` 那条是"文字出视频" ✓），
所以判据集中在三处最容易被写错、而且**不报错**的地方 ✗✗：

① **plan 的帧数必须是 ``4n+1`` 且以源视频为准** ✓ —— 按「秒数 × 帧率」猜出来的帧数
   会在采样第 1 步「少一个潜帧」✓，报错指向形状 ✓、指不到"帧数算错"✓；
② **条件是 ``16 潜 ＋ 1 mask = 17``** ✓，主干拼成 **33** ✓ —— mask 通道数写成 3 只会在
   ``vid_in`` 的 Conv3d 上炸 ✓（报「期望 33、实际 19」✓），指不到"mask 写多了"✓；
③ **落盘回执不许把真 VAE 解码的成品说成"噪声"** ✓✗ —— 这是接这条线**当场踩到**的假证 ✓
   （`write` 的 5 维分支曾把 `synthetic=True` 与"未训练 VAE"写死 ✓✗）。

真 VAE 权重不在盘上时后两节 ``skip`` ✓（plan 那节照跑 ✓：它只读 ffprobe ✓）。
"""

import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")   # cp936 控制台下首行 ✓ 直接 UnicodeEncodeError ✓

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.engine import media, pipeline as pl, torch_backend as tb

#: 真权重（501 MB ✓）。⚠️ 与 `engine_seedvr2_vae_test` 同一份 ✓ —— 不在就 skip ✓。
REAL = Path("D:/Comfy-Desktop/ComfyUI-Shared/models/vae/SEEDVR2/ema_vae_fp16.safetensors")

#: 源素材：11 帧 ⇒ ``4n+1`` 向下吸到 **9**（截尾 2 ✓）；48×32 都是 16 的整数倍 ✓
#: （故意不用 24 高：那会被就近吸到 32 ✓ —— 吸附本身另有判据 ✓，这里只想测主干 ✓）。
SOURCE_W, SOURCE_H, SOURCE_FRAMES, FPS = 48, 32, 11, 8
EXPECT_FRAMES = 9
EXPECT_LATENT_FRAMES = 3

PASSED = 0
FAILED = 0
SKIPPED = 0


def check(name: str, ok: bool, detail: object = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}　{detail}")


def skip(name: str, why: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  ⊘ {name}：{why}")


def _err(fn) -> str:
    """跑一下，把异常压成一行 ✓（判据只看**说了什么** ✓，不看类型名 ✓）。"""
    try:
        fn()
    except BaseException as exc:      # noqa: BLE001 —— SystemExit/KeyboardInterrupt 也压 ✓
        return f"{type(exc).__name__}: {exc}"
    return ""


def _make_source(where: Path) -> Path:
    """造一段**真 mp4** ✓（ffmpeg ✓）—— 只有真文件才骗得过 ffprobe ✓。"""
    torch.manual_seed(7)
    frames = torch.rand(1, 3, SOURCE_FRAMES, SOURCE_H, SOURCE_W) * 2 - 1
    target = where / "source.mp4"
    media.write_video(frames, target, fps=FPS)
    return target


def _request(source=None, **kwargs) -> pl.GenerationRequest:
    params = {
        "stage": pl.SEEDVR2_STAGE,
        #: ⚠️ **空提示词合法** ✓（这一族没有文本塔 ✓）—— 判据 ⑥ 专门守这条 ✓。
        "prompt": "",
        "seconds": 1.0,
        "fps": FPS,
        "steps": 1,
        "seed": 424242,
        "outputs_dir": tempfile.mkdtemp(prefix="repair_"),
        "guidance": pl.guidance_mod.GuidanceConfig(),
        "conditioning": pl.conditioning_mod.ConditioningConfig(),
    }
    params.update(kwargs)
    if source is not None:
        params["reference_videos"] = [str(source)]
    return pl.GenerationRequest(**params)


def case_plan(where: Path) -> None:
    """① plan 层：帧数 ``4n+1``、尺寸以源为准、空提示词合法 ✓。"""
    print("① plan：修复族的帧数/画布口径")
    source = _make_source(where)
    request = _request(source)
    plan = pl.build_plan(request)
    check("① 空提示词在修复族合法（不抛）✓", plan.frames == EXPECT_FRAMES, plan.frames)
    check("① 11 帧 ⇒ 修 9 帧（4n+1 向下吸）",
          plan.frames == EXPECT_FRAMES, plan.frames)
    check("① 潜帧数 = (9−1)/4+1 = 3", plan.latent_frames == EXPECT_LATENT_FRAMES,
          plan.latent_frames)
    check("① 画布以源视频为准（48×32）",
          (plan.width, plan.height) == (SOURCE_W, SOURCE_H), (plan.width, plan.height))
    check("① 帧率取源视频的 8", plan.fps == FPS, plan.fps)
    check("① 默认步数 = 官方 1 步", plan.steps == pl.SEEDVR2_OFFICIAL_STEPS, plan.steps)

    #: ⚠️ 这条是"**拒跑 > 静默退化成文生视频**"的判据 ✓：没有源视频就没有条件 ✓。
    msg = _err(lambda: pl.build_plan(_request()))
    check("① 没有源视频 ⇒ 拒并点名 referenceVideos",
          "referenceVideos" in msg or "reference_videos" in msg, msg)
    msg = _err(lambda: pl.build_plan(_request(**{"reference_videos": [
        str(source), str(source)]})))
    check("① 两条源视频 ⇒ 拒（不静默只取第一路）", "一条" in msg, msg)

    steps_plan = pl.build_plan(_request(source, steps=4))
    check("① 非官方步数照跑但出声（留痕不拦）",
          steps_plan.steps == 4 and any("步" in str(item) for item in steps_plan.warnings),
          (steps_plan.steps, steps_plan.warnings[:2]))


def case_condition(where: Path) -> None:
    """② 条件层：源视频 → ``16 潜 ＋ 1 mask`` ✓；③ 条件在 denoise 里**不会**被当成文本 ✓。"""
    print("② 条件：源视频 → 17 通道（＋ denoise 的 conditioning 位）")
    if not REAL.exists():
        skip("② 条件 / ③ denoise", f"权重未就位：{REAL}")
        return
    source = _make_source(where)
    request = _request(source)
    plan = pl.build_plan(request)

    backend = tb.TorchBackend(device="cpu")
    backend.attach_seedvr2_vae(str(REAL), dtype=torch.float32)
    #: 模拟"装配过 SEEDVR2 主干" ✓ —— 真 3B 权重不进单测 ✓（`SEEDVR2_REAL=1` 那条在
    #: `engine_seedvr2_test` 里 ✓）；这里只验**接线**，不验画质 ✓。
    backend._ditFamily = "seedvr2"
    backend.bind_plan(plan)

    condition = backend.encode_text(request)
    positive = condition["positive"]
    report = condition["repair"]
    check("② positive 是**张量**（不是文本状态）", hasattr(positive, "shape"),
          type(positive).__name__)
    check("② 条件 = (1, 17, 3, 4, 6)",
          list(positive.shape) == [1, 17, EXPECT_LATENT_FRAMES, SOURCE_H // 8, SOURCE_W // 8],
          list(getattr(positive, "shape", ())))
    check("② mask 通道全 1（整段重绘）",
          bool((positive[:, 16:] == 1).all()), positive[:, 16:].unique())
    check("② 回执说清 16＋1＝17、主干 33",
          report["channels"] == {"degraded": 16, "mask": 1, "vidIn": 33}, report["channels"])
    check("② 回执记截尾 2 帧（11 → 9）", report["droppedFrames"] == 2,
          report.get("droppedFrames"))
    check("② 真条件登记在同一处（denoise 取得到）",
          getattr(backend, "_repairCondition", None) is positive, type(
              getattr(backend, "_repairCondition", None)).__name__)

    class _FakeDiT(torch.nn.Module):
        """只记参数、不算数值 ✓ —— 判据是"喂到了哪个位置"✓，不是"算得对不对"✓。"""

        def __init__(self) -> None:
            super().__init__()
            self.seen = {}

        def forward(self, latents, sigma, context=None, conditioning=None):
            self.seen = {"views": tuple(latents.shape), "context": context,
                         "conditioning": conditioning,
                         "in_channels": latents.shape[1] + conditioning.shape[1]}
            return torch.zeros_like(latents)

    fake = _FakeDiT()
    backend._model = fake
    latents = torch.randn(1, 16, EXPECT_LATENT_FRAMES, SOURCE_H // 8, SOURCE_W // 8)
    backend.denoise(latents, 0.9, positive, request)
    check("③ 条件进的是 `conditioning` 位（不是 context）",
          fake.seen.get("conditioning") is not None and fake.seen.get("context") is None,
          {k: type(v).__name__ for k, v in fake.seen.items()})
    check("③ 主干入维 = 16 ＋ 17 = 33",
          fake.seen.get("in_channels") == 33, fake.seen.get("in_channels"))

    #: pipeline 在某些路径上会把条件替成 `None` ✓ —— 后端必须**自己兜住** ✓
    #: （兜不住的症状：`_resolve_context` 报「要 (B,L,D)」，读起来像提示词坏了 ✓✗）。
    backend.denoise(latents, 0.9, None, request)
    check("③ condition=None 时回落到已登记的条件",
          fake.seen.get("in_channels") == 33, fake.seen.get("in_channels"))

    backend._repairCondition = None
    msg = _err(lambda: backend.denoise(latents, 0.9, None, request))
    check("③ 条件全无 ⇒ 拒并说清「不能凭空生成」",
          "condit" in msg.lower() or "referenceVideos" in msg, msg)


def case_write(where: Path) -> None:
    """④ 落盘：`synthetic` 由 `decode` 说了算 ✓（不许在这里替它表态 ✗）。"""
    print("④ 落盘：真 VAE 产物不叫「噪声」")
    request = _request()
    plan = pl.GenerationPlan(fps=FPS)
    backend = tb.TorchBackend(device="cpu")
    frames = torch.rand(1, 3, EXPECT_FRAMES, SOURCE_H, SOURCE_W) * 2 - 1
    real = backend.write({"frames": frames, "synthetic": False,
                          "vaeKind": "seedvr2-real"}, plan, request)
    check("④ 真产物 ⇒ synthetic=False 且落真 mp4",
          real["synthetic"] is False and real.get("realFile") is True
          and "真 VAE 解码" in real["note"], (real["synthetic"], real["note"]))
    noise = backend.write({"frames": frames, "synthetic": True,
                           "vaeKind": "reference-untrained"}, plan, request)
    check("④ 仍保留旧口径（参考 VAE ⇒ 明说噪声）",
          noise["synthetic"] is True and "未训练" in noise["note"], noise["note"])
    #: ⚠️ 反向：**别只信调用方递来的 ``synthetic``** ✗ —— 给了 ``synthetic=False`` 却没有
    #: ``vaeKind``（＝没挂 VAE），回执必须如实说「占位解码」✓✗。只按标记表态会让
    #: 「拿一组合成帧冒充真成品」静默通过 ✓（这正是这节的旧写法，已被这条反证挡掉 ✓）。
    bare = backend.write({"frames": frames, "synthetic": False}, plan, request)
    check("④ 没有 vaeKind ⇒ 回执如实说占位解码（不替调用方担保真画面）",
          "占位解码" in bare["note"] and "真 VAE 解码" not in bare["note"], bare["note"])


def main() -> int:
    print("=" * 78)
    print("SEEDVR2 修复支路自检（plan / 17 通道条件 / denoise / 落盘）")
    print("=" * 78)
    with tempfile.TemporaryDirectory(prefix="seedvr2_repair_") as where:
        root = Path(where)
        case_plan(root)
        case_condition(root)
        case_write(root)
    print("-" * 78)
    #: ⚠️ 收尾行**必须是** `SUMMARY: n/m passed` ✓✗ —— `run_all` 总表按这一行取项数 ✓，
    #: 写成「通过 X / 失败 Y」时总表里这套的项数**恒为 0** ✓（`structure_test` ㊈ 会红 ✓）。
    print(f"通过 {PASSED} / 失败 {FAILED} / 跳过 {SKIPPED}")
    print(f"SUMMARY: {PASSED}/{PASSED + FAILED} passed"
          + (f"（跳过 {SKIPPED}）" if SKIPPED else ""))
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
