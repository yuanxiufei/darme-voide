# 兼容性矩阵（E0.2）

> 口径：**只写本机实测值**，不写下限、不写"应该支持" ✓✗。过期会被 `tests/engine_env_docs_test.py` 判红 ✓。
>
> 采集日 2026-10-30 · 采集机：单卡 RTX A5000（22.49 GiB 可用 / 23028 MiB 标称，ECC 开启）
> 复现：`python -m app.services.engine doctor --json`（能力 / 清单段）＋ `python -m pip check` ＋ 本文件 §一 的直采命令

## 一、运行时

| 维度 | 实测 | 本仓要求 | 结论 |
|---|---|---|---|
| Python | CPython 3.12.10 | ≥ 3.11（`pyproject` 未钉死） | 可用 ✓ |
| torch | 2.14.1+cu130 | ≥ 2.6（`safetensors`/`torch.load` 的新 `weights_only` 语义） | 可用 ✓ |
| torch 编入的 CUDA | 13.0 | 与驱动匹配即可 | 可用 ✓ |
| `torch.cuda.is_available()` | `True` | — | **GPU 路径通** ✓（早先那版「只能走 CPU 轮子」的记录**已作废**，见 §六） |
| cuDNN | 9.24.0（`torch.backends.cudnn.version() == 92400`） | — | 可用 ✓ |
| 驱动 | 610.88 | ≥ CUDA 13.0 所需 | 可用 ✓ |
| GPU 算力 | sm_86（Ampere） | — | 见 §四：**有 fp8 语义、无 fp8 硬件加速** |

## 二、Python 依赖（实测版本）

| 包 | 实测 | 用途 | 装不上会怎样 |
|---|---|---|---|
| torch | 2.14.1+cu130 | 推理 / 反量化 | `E_ENGINE_UNAVAILABLE`（`backend.missingRequired` 非空） |
| numpy | 2.5.4 | 张量搬运 / 音频 | 同上 |
| safetensors | 0.8.0 | 权重直读（头部即 `comfy_quant` 真源） | 同上 |
| pillow | 13.0.0 | 图像编解码 | 同上 |
| transformers | 5.21.0 | 可选（TE 分词/配置） | 只是**可选缺失**，不拦装载 ✓ |
| fastapi / uvicorn / SQLAlchemy / PyYAML / httpx / python-multipart | 见 `THIRD_PARTY_NOTICES.md` §二 | 服务层 | 本仓不拦（引擎可独立跑） |

许可一栏不在这里重复 —— 真源是 `THIRD_PARTY_NOTICES.md`，两份文件由同一套判据盯着。

## 三、量化族 × 支持结论

真源是 `app/services/engine/quant.py`（`capabilities().quant` 由它出），**不是**这张表；表只是把结论翻成人话。

| 族 / 格式 | 结论 | 依据 |
|---|---|---|
| per-tensor / per-row / per-col / elementwise 布局 | **支持** ✓ | `quant.plan_dequant` 能给出反量化计划 |
| `convrot`（带 Hadamard 反旋转的 per-row） | **支持** ✓ | `quant.convrot_group_of` ＋ `build_hadamard` ＋ 激活侧旋转搬运；两个 19.53 GiB 主 DiT 实测就是这个族 |
| 4 bit **打包**（`w4a4` / `linear_dtype: int4`，每字节两个有符号 nibble） | **支持** ✓ | `quant.int4_stems_of` ＋ `unpack_int4`（低半字节=偶数列、nibble ≥ 8 减 16）＋ 逻辑列 ×2 后再判 `convrot`/布局；盘上 8 个 int4 文件实测全 `per-row+int4pack+convrot*`。⚠️ 打包形状与 per-row int8 **逐字相同** ⇒ 只有文件自述（`comfy_quant` / `_quantization_metadata`）能识破，判据**绝不许按形状猜** |
| `nvfp4`（4 bit E2M1 打包 ＋ e4m3 块 scale[16] ＋ f32 二级 scale，AWQ 变体） | **支持** ✓ | `quant_nvfp4` 逐层还原（E2M1 高半字节在前、cuBLAS 128×4 tiled 块 scale、AWQ 折进权重列）；与 `reference` 的实现 `torch.equal` 逐位对拍；清单里 `te_nvfp4` 属这一族 |
| `mxfp4` | **不支持** ✗ | `capabilities().quant.unsupportedFormats` 明确列出（E8M0 共享指数，解码表与 nvfp4 不同）⇒ 走 `E_UNSUPPORTED_QUANT` |
| 块/分组量化（**上表已列各族以外**，如带 `qzeros` / `g_idx` 的 GPTQ 式 `group_size` 变体） | **不支持** ✗ | 一律拒绝，**不猜**（不按 per-row 糊过去、也不猜 group size）。⚠️ 别与上一行的 4 bit **打包**混淆：`convrot_w4a4` 里的 `quant_group_size` 是 Hadamard **旋转分组**，不是量化分组 ⇒ 那一族**支持** |

E0.4 已裁定并落地（`docs/adr/0002-quant-families.md`，**已接受 · A 路**）：`nvfp4` 从「未实现」转为**已实现** ✓。
⚠️ 但「能还原权重」不等于「H3 能跑」✗：本仓 32B Qwen3-VL 文本编码器**尚无模型骨架** ⇒ 还原出的权重暂时没有
挂载点（见该 ADR §5.4）。

## 四、GPU 能力边界

| 事实 | 数值 | 影响 |
|---|---|---|
| 算力级别 | sm_86 | fp8 只能 emulated（`supports_fp8_compute` 要 sm_89+）⇒ fp8 权重"装得下"不等于"更快" |
| 可用显存 | 23028 MiB 标称 / 22.49 GiB 实测可用 | h3 主 DiT 19.53 GiB ⇒ 720p 必须 `int8_offload`；装载期 OOM 的提示见 `registry.VRAM_HINT`。⚠️ **整阶段装不下**：`readiness("h3").vram` 实测估算 49.44 GiB（权重精确 39.55 GiB）> 容量 24 GiB，`fits=false`、`headroom −25.44 GiB` ⇒ 单卡 720p 端到端**跑不起来**，这不是轮子问题（见 `docs/adr/0001-torch-wheel.md` §3.2） |
| 单卡 | 1 | 多卡只有接口，无性能承诺（`capabilities().limits`） |

## 五、`pip check`

```
No broken requirements found.
```

**无冲突** ✓（这是 E0.2 的硬判据之一）。⚠️ 但「`pip check` 干净」只说明依赖声明自洽，
**不说明 GPU 能用** —— 后者看 §一 的 `cuda.is_available()`。

## 六、两份落盘事实怎么维护

| 文件 | 内容 | 谁生成 |
|---|---|---|
| `docs/env/hardware-report.json` | `facade.capabilities()` 原样（能力 / 阶段 / 清单 / 量化 / 边界 / `knownGaps`） | `python -m app.services.engine doctor --json` 的载荷直接落盘 |
| `docs/env/runtime-versions.json` | python / torch / CUDA / cuDNN / 驱动 / GPU / 依赖版本 / `pipCheck` | `doctor --json` 的 python 与清单段 ＋ 本机 torch 直采 |

⚠️ 这两份文件**没有读者**（代码里 0 处引用，见判据 ④），所以它们最容易悄悄过期 ✓✗ ——
实测就抓到过一次：旧 `hardware-report.json` 是 `schemaVersion: 1`，还把 `transformers` 记成
「可选缺失」、把 `convrot` 记成「未实现」、`modelsDir` 还指向旧的 `data/models` ✓✗。
现在由 `tests/engine_env_docs_test.py` 逐项对账：**改了代码忘了刷文件 ⇒ 红** ✓。
