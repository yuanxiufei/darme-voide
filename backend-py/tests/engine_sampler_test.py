"""采样器自检 ✓（2026-10-31：SA-Solver / dpmpp_2m / lms / euler_ancestral 落户 ✓）。

**为什么单独一份** ✗：采样器是**纯数学** ✓（不装权重、不碰 torch ✓ —— 本模块张量无关 ✓），
所以判据不能是"跑通就行"✗，而要用**有解析解的 ODE** 当标尺 ✓：

    令 ``denoised = x − σ·c`` ⇒ 方向 ``d = (x − denoised)/σ = c``（**常数** ✓）
    ⇒ ODE 解析解 ``x(σ) = x(σ₀) + c·(σ − σ₀)`` ✓（末位 σ=0 ⇒ ``x_end = x₀ − c·σ₀`` ✓）

常数是 0 次多项式 ⇒ **任何一阶以上的正确实现都应精确命中**（误差 ≈ 机器精度 ✓）。
这一条能同时抓出：系数符号错、λ 口径拿错、积分权重错、末步没走 denoised 分支 ✓✗
—— 全是"不报错只是图变糊"的那类 ✗✗（写松了就永远发现不了 ✓）。

另一条横切判据：``noise_sampler=None`` ⇒ **必须严格等于 ODE 解** ✓✗
（参考实现的随机项按"有噪声"推导系数 ⇒ 若"τ 照算但不加噪声"就会偏离 ✓✗，这条专抓它 ✓）。
"""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: ⚠️ cp936 控制台下 ``✓`` 会 ``UnicodeEncodeError`` ⇒ 判据还没跑就崩在第一条打印上 ✗
#: （``run_all`` 已重配 stdout，只有单独跑才露头 ⇒ 每个 ``__main__`` 测试都自带这一句 ✓）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from app.services.engine import sampler as sm  # noqa: E402
from app.services.engine import schedules  # noqa: E402

_FAILS: list[str] = []
_PASSED = 0

#: 起点（σ₀ **不得为 1.0** —— flow 口径下会被 `_offset_first_sigma` 改成 0.9999 ✓，
#: 那条偏移另有专测 ③，这里要让解析解可比 ✓）
SIGMAS = [0.99, 0.9, 0.7, 0.5, 0.3, 0.15, 0.05, 0.0]
START = 3.0
SLOPE = -0.8            # ⇒ d 恒为 SLOPE（常数方向 ✓）
EXPECT = START + SLOPE * (SIGMAS[-1] - SIGMAS[0])


def check(label: str, ok: bool, detail: object = "") -> None:
    global _PASSED
    if ok:
        _PASSED += 1
        print(f"PASS  {label}")
        return
    _FAILS.append(label)
    print(f"FAIL  {label}  ->  {detail!r}")


def _model(x: float, sigma: float) -> float:
    """``denoised = x − σ·c`` ⇒ ``d = c`` 常数 ✓。"""
    return x - float(sigma) * SLOPE


def _run(name: str, **options: object) -> tuple[float, int]:
    out, steps = sm.sample(_model, START, list(SIGMAS), name, **options)
    return float(out), int(steps)


def main() -> int:
    # ⓪ 积分与线性代数（高阶系数的地基 ✓ —— 表贴错则 dpmpp/lms/sa 全长歪 ✓）
    exact = [abs(sm._integrate(lambda t: t ** p, 0.0, 1.0) - 1.0 / (p + 1))  # noqa: SLF001
             for p in range(0, 16)]
    check("⓪ 8 点高斯积分对 p=0..15 的 x^p **全部精确**（≤15 次多项式无误差 ✓）",
          max(exact) < 1e-14, max(exact))

    solved = sm._solve_linear([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])  # noqa: SLF001
    check("⓪′ `_solve_linear` 解得 (1, 3) ✓（部分主元 ⇒ 需要换行也对 ✓）",
          max(abs(a - b) for a, b in zip(solved, (1.0, 3.0))) < 1e-12, solved)

    swapped = sm._solve_linear([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0])  # noqa: SLF001
    check("⓪″ 首列全零的矩阵 ⇒ 换行后仍解出 (3, 2) ✓（不换行就会除零 ✗）",
          max(abs(a - b) for a, b in zip(swapped, (3.0, 2.0))) < 1e-12, swapped)

    # ① λ 两式 + σ₀=1 的偏移
    lambdas = sm._half_log_snr([0.8, 0.5], "flow")  # noqa: SLF001
    check("① flow 口径 λ = log((1−σ)/σ) ✓",
          abs(lambdas[0] - math.log(0.2 / 0.8)) < 1e-12
          and abs(lambdas[1] - math.log(0.5 / 0.5)) < 1e-12, lambdas)
    discrete = sm._half_log_snr([0.8, 0.5], "snr")  # noqa: SLF001
    check("①′ snr 口径 λ = −log σ ✓（**两支不许互换** ✗：拿错 ⇒ 高阶系数全错 ✓✗）",
          abs(discrete[0] + math.log(0.8)) < 1e-12 and abs(discrete[1] + math.log(0.5)) < 1e-12,
          discrete)
    try:
        sm._half_log_snr([0.5], "nope")  # noqa: SLF001
        bad_kind = ""
    except ValueError as err:
        bad_kind = str(err)
    check("①″ 口径名写错 ⇒ 当场 ValueError（不静默回落成 flow ✗）",
          "可用" in bad_kind, bad_kind)

    offset = sm._offset_first_sigma([1.0, 0.5, 0.0], "flow")  # noqa: SLF001
    check("③ flow 且 σ₀=1.0 ⇒ 首 σ 压成 1−1e-4 ✓（否则 λ₀=−inf ⇒ 首步炸 ✗✗）",
          abs(offset[0] - (1.0 - 1e-4)) < 1e-15 and offset[1] == 0.5, offset)
    kept = sm._offset_first_sigma([14.6, 1.0, 0.0], "snr")  # noqa: SLF001
    check("③′ snr 口径**不动**首 σ ✓（−log σ 本来就有限，乱动反而改轨迹 ✗）",
          kept[0] == 14.6, kept)

    # ② 祖先步与 LMS 权重
    down, up = sm._ancestral_step(1.0, 0.6, 1.0)  # noqa: SLF001
    check("② 祖先步能量守恒：down² + up² = σ_to² ✓",
          abs(down ** 2 + up ** 2 - 0.6 ** 2) < 1e-12, (down, up))
    none = sm._ancestral_step(1.0, 0.6, 0.0)  # noqa: SLF001
    check("②′ η=0 ⇒ (σ_to, 0) 无回注 ✓", none == (0.6, 0.0), none)

    coeff_sum = sum(sm._linear_multistep_coeff(4, SIGMAS, 4, j) for j in range(4))  # noqa: SLF001
    check("②″ LMS 权重之和 = 步长 Δσ ✓（Lagrange 基和为 1 ⇒ 常数方向精确的必要条件 ✓；"
          "σ **递减** ⇒ Δσ 是负的 ✗，第一版把符号写反了 ✓）",
          abs(coeff_sum - (SIGMAS[5] - SIGMAS[4])) < 1e-12, coeff_sum)

    # ④ ⭐ 七个采样器全部命中解析解
    for name in ("euler", "heun", "multistep", "dpmpp_2m", "lms",
                 "euler_ancestral", "sa_solver"):
        value, steps = _run(name)
        check(f"④ ⭐ `{name}` 在常数方向 ODE 上命中解析解 {EXPECT:.6f} ✓（步数 {steps}）",
              abs(value - EXPECT) < 1e-9, (round(value, 12), round(EXPECT, 12)))

    discrete_end, _ = _run("euler_ancestral", parameterization="discrete")
    check("④′ `euler_ancestral` 的离散口径同样命中解析解 ✓（两支都算对 ✓）",
          abs(discrete_end - EXPECT) < 1e-9, discrete_end)
    try:
        _run("euler_ancestral", parameterization="nope")
        bad_param = ""
    except ValueError as err:
        bad_param = str(err)
    check("④″ 参数化写错 ⇒ 当场 ValueError ✓（不静默按 flow 跑 ✗）",
          "可用" in bad_param, bad_param)

    # ⑤ ⭐ 随机口径：无噪声 ⇒ 严格等于 ODE 解；有噪声 ⇒ 真被扰动
    silence, _ = _run("sa_solver", noise_sampler=None)
    check("⑤ ⭐ `sa_solver` 无 `noise_sampler` ⇒ **严格等于** ODE 解 ✓✗"
          "（反例：τ 照算却不加噪声 ⇒ 系数按随机推导、实际没加 ⇒ 数学不自洽 ✗）",
          abs(silence - EXPECT) < 1e-9, silence)

    forced, _ = _run("sa_solver", eta=0.0, noise_sampler=lambda a, b: 0.5)
    check("⑤′ η=0 ⇒ 即使给了噪声也不注入 ✓（结果仍等于 ODE 解 ✓）",
          abs(forced - EXPECT) < 1e-9, forced)

    noisy, _ = _run("sa_solver", noise_sampler=lambda a, b: 0.5)
    check("⑤″ 给了 `noise_sampler` ⇒ 结果**确实被扰动** ✓（证明随机项真接上了 ✓）",
          abs(noisy - EXPECT) > 1e-6, noisy)

    ancestral_ode, _ = _run("euler_ancestral", eta=1.0)
    check("⑤‴ `euler_ancestral` 无噪声 ⇒ 退化成确定性 euler ✓（不是「随机但没加噪声」✗）",
          abs(ancestral_ode - EXPECT) < 1e-9, ancestral_ode)

    muted, _ = _run("sa_solver", s_noise=0.0, noise_sampler=lambda a, b: 0.5)
    check("⑤‴′ `s_noise=0` ⇒ 同样不注入 ✓（参考实现只门控噪声、**系数仍按 τ 算** ⇒ 不自洽 ✗）",
          abs(muted - EXPECT) < 1e-9, muted)

    shared = [name for name, fn in (("sa_solver", sm.sample_sa_solver),
                                    ("euler_ancestral", sm.sample_euler_ancestral))
              if "_stochastic_strength(" in inspect.getsource(fn)]
    check("⑤⁗ ⭐ 两个随机采样器**都**走 `_stochastic_strength` ✓✗"
          "（首版只有 sa_solver 走了 ⇒ euler_ancestral 按 η=1 缩短步长却不注入噪声 ⇒ 两头错 ✓✗）",
          len(shared) == 2, shared)

    # ⑥ 只认自己的参数：静默吞掉 ⇒ "设了 eta 却没生效"的假绿 ✗✗
    try:
        sm.sample(_model, START, list(SIGMAS), "euler", eta=0.5)
        swallowed = ""
    except TypeError as err:
        swallowed = str(err)
    check("⑥ ⭐ 把 `eta` 交给不认识它的 `euler` ⇒ TypeError ✓✗"
          "（静默吞参数正是「设了却没生效」的来源 ✓）", "eta" in swallowed, swallowed)

    check("⑥′ `STOCHASTIC` 与 `SAMPLERS` 不脱节 ✓（前者不许点名不存在的采样器 ✗）",
          sm.STOCHASTIC <= set(sm.SAMPLERS), sorted(sm.STOCHASTIC - set(sm.SAMPLERS)))
    check("⑥″ 四个新采样器都在册 ✓（漏登记 ⇒ `sample()` 里报「未知采样器」✗）",
          {"dpmpp_2m", "lms", "euler_ancestral", "sa_solver"} <= set(sm.SAMPLERS),
          sorted(sm.SAMPLERS))

    # ⑦ 输入校验仍在
    for label, bad in (("序列反了", [0.0, 0.5, 1.0]), ("末位不是 0", [0.5, 0.0, 0.2])):
        try:
            sm.sample(_model, START, bad, "dpmpp_2m")
            rejected = ""
        except ValueError as err:
            rejected = str(err)
        check(f"⑦ `{label}` ⇒ 当场 ValueError ✓（静默出噪声是最难查的 ✗）", bool(rejected), rejected)

    # ⑧ 少步/极端调度不炸（真实用途：8 步、单步、σ 跨度极小）
    for label, sigmas in (("8 步", [0.99, 0.86, 0.7, 0.5, 0.34, 0.2, 0.09, 0.02, 0.0]),
                          ("单步", [0.99, 0.0]),
                          ("紧跨度", [0.05, 0.04, 0.03, 0.02, 0.01, 0.0])):
        ok = True
        detail: list[object] = []
        for name in ("dpmpp_2m", "lms", "sa_solver", "euler_ancestral"):
            try:
                value, _ = float(sm.sample(_model, START, list(sigmas), name)[0]), 0
                want = START + SLOPE * (sigmas[-1] - sigmas[0])
                detail.append((name, round(abs(value - want), 12)))
                ok = ok and abs(value - want) < 1e-8
            except BaseException as err:  # noqa: BLE001
                ok = False
                detail.append((name, f"{type(err).__name__}: {err}"))
        check(f"⑧ {label} 调度下四个新采样器都不炸且命中解析解 ✓", ok, detail)

    # ═══════════════════════════════════════════════════════════════════════════
    # ⑨ `res_multistep` 与参考实现的**逐位对拍**（2026-11-01 ✓）
    #
    # 为什么不能只靠 ④ 的解析解 ✗✗：常数方向的模型上 ``x0_i ≡ x0_{i−1}`` ✓ ⇒
    # ``b₁ + b₂ = φ₁`` ✓，``ratio``（＝"上一拍的 σ_down 在哪" ✓）**整个被消掉** ✓ ——
    # 于是"``ratio`` 取错""``h`` 符号取反"这类错**照样命中解析解** ✓✗（本轮真栽在
    # ``e^{−h}`` 写成 ``e^{+h}`` 上 ✓，④ 抓不到 ✓，是下面这份对拍抓到的 ✓）。
    # ═══════════════════════════════════════════════════════════════════════════
    import math as _math

    def _phi1(value: float) -> float:
        # ⚠️ `φ₁(t) = (eᵗ − 1)/t` ✓（参考 `phi1_fn` ✓）—— 写成 `(1 − e^{−t})/t` 是**镜像** ✓✗：
        #    两边都喂 `−h` 时它算的是 `φ₁(+h)` ✓，与实现差一个符号位、而形状看着都对 ✓。
        return _math.expm1(value) / value if value else 1.0

    def _phi2(value: float) -> float:
        return (_phi1(value) - 1.0) / value if value else 0.5

    def _reference_res_multistep(model, x0, sigmas):
        """**逐行转写** ``comfy/k_diffusion/sampling.py::sample_res_multistep``（eta=0 ✓）。"""
        old_denoised = None
        old_sigma_down = 0.0
        x = x0
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            denoised = model(x, sigma)
            sigma_down = sigma_next                     # get_ancestral_step(eta=0) ✓
            if sigma_down == 0 or old_denoised is None:
                d = (x - denoised) / sigma
                x = x + d * (sigma_down - sigma)
            else:
                t = -_math.log(sigma)
                t_next = -_math.log(sigma_down)
                t_old = -_math.log(old_sigma_down)
                h = t_next - t
                c2 = (-_math.log(sigmas[i - 1]) - t_old) / h
                b1 = _phi1(-h) - _phi2(-h) / c2
                b2 = _phi2(-h) / c2
                x = _math.exp(-h) * x + h * (b1 * denoised + b2 * old_denoised)
            old_denoised, old_sigma_down = denoised, sigma_down
        return x

    def _nonlinear(x, sigma):
        # 刻意**非线性** ✓：``x0`` 随 σ 变 ✓ ⇒ ``ratio``/``b₂`` 有没有接对才看得出来 ✓。
        return x * 0.82 - 0.35 * float(sigma) ** 1.5

    grid = [1.0, 0.93, 0.81, 0.62, 0.44, 0.27, 0.13, 0.05, 0.0]   # ← 递减 ✓、末位 0 ✓
    got, _steps = sm.sample(_nonlinear, 2.5, grid, "res_multistep")
    want = _reference_res_multistep(_nonlinear, 2.5, list(grid))
    check("⑨ `res_multistep` 与参考实现逐位对拍 ✓（非线性模型 ✓ —— 常数模型下 `ratio` 会被消掉 ✗）",
          abs(float(got) - want) < 1e-12, (float(got), want))

    # 反证：把指数符号改回去（`e^{+h}` ✓）**必须**被同一条判据抓到 ✓
    broken = 2.5
    previous = previous_down = None
    for index in range(len(grid) - 1):
        sigma, sigma_next = grid[index], grid[index + 1]
        denoised = _nonlinear(broken, sigma)
        if sigma_next == 0 or previous is None:
            broken = broken + (broken - denoised) / sigma * (sigma_next - sigma)
        else:
            h = _math.log(sigma) - _math.log(sigma_next)
            ratio = (_math.log(previous_down) - _math.log(grid[index - 1])) / h
            broken = (_math.exp(h) * broken
                      + h * ((_phi1(-h) - _phi2(-h) / ratio) * denoised
                             + (_phi2(-h) / ratio) * previous))
        previous, previous_down = denoised, sigma_next
    check("⑨′ 反证：指数写成 `e^{+h}` 会被 ⑨ 抓到 ✓（否则那条判据是假绿 ✓✗）",
          abs(broken - want) > 0.05, (broken, want))

    # ═══════════════════════════════════════════════════════════════════════════
    # ⑩ `lcm` 的语义 = **每步直接取 denoised** ✓，再按 σ_next 回噪 ✓（不是 ODE 求解器 ✗）
    # ═══════════════════════════════════════════════════════════════════════════
    lcm_grid = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    # 模型给**恒定的 denoised=1.0** ✓ ⇒ 贪心档每一步都直接落到 1.0 ✓，
    # 而任何 ODE 求解器会把它当成"方向 d=(x−1)/σ"去积分 ✓ ⇒ 结局绝不是 1.0 ✓（两者可分辨 ✓）。
    const = lambda x, s: 1.0                                   # noqa: E731
    greedy, _steps = sm.sample(const, START, lcm_grid, "lcm")
    check("⑩ `lcm` 不给噪声 ⇒ 结局**恰好等于 denoised** ✓（逐步贪心 ✓、不是欧拉 ✓）",
          abs(float(greedy) - 1.0) < 1e-12, float(greedy))

    # 反证：同一条 σ 表喂 `euler` ⇒ **中途**状态整体不同 ✓（末步都收敛到 denoised ✓，
    # 所以判据只能看中途 ✓ —— 只看结局会得出"两者一样"的假结论 ✓✗）
    euler_seen: list[float] = []
    sm.sample(const, START, lcm_grid, "euler",
              callback=lambda index, sigma, value: euler_seen.append(float(value)))
    check("⑩″ 同一模型同一条 σ 表 ⇒ `euler` 的**中途**状态与 ⑩ 差得很远 ✓（⑩ 不是恒真判据 ✓）",
          max(abs(a - b) for a, b in zip(euler_seen, [1.0] * len(euler_seen))) > 0.1,
          euler_seen)

    seen: list[float] = []
    sm.sample(const, START, lcm_grid, "lcm", noise_sampler=lambda a, b: 2.0,
              callback=lambda index, sigma, value: seen.append(float(value)))
    # flow 支回噪式 `x = denoised·(1−σ_next) + n·σ_next` ✓（参考 `ModelSamplingFlux.noise_scaling` ✓）
    want_states = [1.0 * (1.0 - s) + 2.0 * s for s in lcm_grid[1:]]
    check("⑩′ `lcm` + 常量噪声 2.0 ⇒ 每拍状态 = `1−σ_next + 2σ_next` ✓（末步 σ=0 ⇒ 回到 1.0 ✓）",
          len(seen) == len(want_states)
          and max(abs(a - b) for a, b in zip(seen, want_states)) < 1e-12,
          (seen, want_states))

    # ═══════════════════════════════════════════════════════════════════════════
    # ⑪ `er_sde`：确定性 + 阶数（步数翻倍 ⇒ 误差显著下降 ✓）
    # ═══════════════════════════════════════════════════════════════════════════
    def _flow_grid(steps):
        return schedules.flow_sigmas_for_steps(steps, shift=3.0)

    def _soft(x, sigma):
        return _math.tanh(0.6 * x) + 0.2 * float(sigma)

    twice = [sm.sample(_soft, 1.0, _flow_grid(12), "er_sde")[0] for _ in range(2)]
    check("⑪ `er_sde` 不给噪声 ⇒ 两次结果**逐位相同** ✓（确定性 ✓ —— 否则就是偷偷加了噪 ✓✗）",
          twice[0] == twice[1], twice)

    with_noise = sm.sample(_soft, 1.0, _flow_grid(12), "er_sde",
                                noise_sampler=lambda sigma, sigma_next: 0.05)[0]
    check("⑪′ `er_sde` + 噪声 ⇒ 结局**改变** ✓（登记在 STOCHASTIC 里就必须真会加噪 ✓）",
          float(with_noise) != float(twice[0]), (float(with_noise), float(twice[0])))

    fine = sm.sample(_soft, 1.0, _flow_grid(16384), "euler")[0]      # ← 细网格当真值 ✓
    err_coarse = abs(float(sm.sample(_soft, 1.0, _flow_grid(8), "er_sde")[0]) - float(fine))
    err_fine = abs(float(sm.sample(_soft, 1.0, _flow_grid(64), "er_sde")[0]) - float(fine))
    check("⑪″ 步数 8 ⇒ 64：误差**至少减半** ✓（阶数没接对时这条会红 ✓）",
          err_fine < err_coarse * 0.5, (err_coarse, err_fine))

    # ═══════════════════════════════════════════════════════════════════════════
    # ⑫ 注册表：三档新采样器都进了 `SAMPLERS`，随机档只在 `STOCHASTIC` 里 ✓
    # ═══════════════════════════════════════════════════════════════════════════
    for name in ("res_multistep", "er_sde", "lcm"):
        check(f"⑫ `{name}` 已注册进 `SAMPLERS` ✓（工作流参数名可直接透传 ✓）",
              name in sm.SAMPLERS, sorted(sm.SAMPLERS))
    check("⑫′ `res_multistep` **不是**随机档 ✓（eta=0 / s_noise=1 的默认档不接噪声 ✓）；"
          "`er_sde` / `lcm` **是** ✓（它们的更新式里本来就有随机项 ✓）",
          "res_multistep" not in sm.STOCHASTIC
          and {"er_sde", "lcm"} <= set(sm.STOCHASTIC), sorted(sm.STOCHASTIC))
    check("⑫″ 三档都是**每步一次**模型调用 ✓（双流才套得上 ✓）",
          all(sm.model_calls_per_step(name) == 1.0
              for name in ("res_multistep", "er_sde", "lcm")), None)

    # ═══════════════════════════════════════════════════════════════════════════
    # ⑬ 双流的逐拍更新式**不许另抄一份** ✓（2026-11-01 抄出来的那份把 `c₂` 符号写反了 ✓✗）
    #
    # 双流每步只有"一拍" ✓、调不了整个 `sample_res_multistep` ✓，所以最容易的写法就是
    # 把公式再抄一遍 ✓ —— 抄的代价是一次**不报错**的符号错 ✓（`b₂` 变号、结局差一倍 ✓）。
    # 这里钉住"只有一份"：源码里必须出现 `multistep_step(` ✓，且不出现 φ 的本地实现 ✓。
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        from app.services.engine import h3_form  # noqa: PLC0415 —— 要 torch 才拿得到这个属性 ✓
        dual_src = inspect.getsource(h3_form.sample_dual_stream)
    except Exception as problem:                                    # noqa: BLE001
        dual_src = ""
        check("⑬ 取不到 `sample_dual_stream` 源码 ⇒ 跳过结构判据 ✓", True, problem)
    if dual_src:
        check("⑬ ⭐ 双流调的是**同一个** `multistep_step` ✓（另抄一份 ⇒ 符号错没人拦 ✗✗）",
              "multistep_step(" in dual_src, dual_src[:0])
        check("⑬′ 双流源码里**没有**本地 φ/指数实现 ✓（`expm1`、`math.exp(-h)` 都不该出现 ✗）",
              "expm1" not in dual_src and "exp(-h)" not in dual_src, None)

    total = _PASSED + len(_FAILS)
    print(f"\nSUMMARY: {_PASSED}/{total} passed")
    for label in _FAILS:
        print(f"  FAILED: {label}")
    return 0 if not _FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
