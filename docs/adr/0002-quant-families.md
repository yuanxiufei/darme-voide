# ADR 0002：量化族的取舍（`nvfp4` / `mxfp4`）

- **状态**：**已接受 · A 路已落地**（2026-10-30 草案 ⇒ 同日创作者裁定 **A（实现解码）** 并实现完毕 ✓）
- **决策者**：创作者（方案 owner）＋ 引擎 owner 提供评估
- **对应 WBS**：E0.4「量化族裁定」
- **为什么现在必须定**：这不是「多支持一种格式」的加分项 —— 它是**当前唯一挡住 H3 视频全链路**的东西（见 §1 实测）。

> **落地结论（2026-10-30）**：A 已实现 —— 新模块 `app/services/engine/quant_nvfp4.py`（解码表 · 128×4 swizzled
> 块 scale · f32 二级 scale · AWQ 折列），判据 `tests/engine_quant_nvfp4_test.py` **35 条**（与
> `reference/lora/musubi-tuner/.../nvfp4_utils.py` `torch.equal` 逐位对拍 ＋ 跑 `comfy.float::to_blocked` 往返 ＋
> 真 350 层权重实测与错解反证 ＋ **`weights.load_module_weights` 端到端**：合成文件与真权重两层都是 `torch.equal`）；
> `quant` 拆出 `SUPPORTED_FORMATS`（nvfp4）/ `UNSUPPORTED_FORMATS`（仅 mxfp4）；
> `loader.dequant_plan_of()` 成为**唯一口径**（`plan_component` 与 `facade.inspect_weights` 都走它）。
> **仍未落**：把还原出的权重挂进某个模型的前向（本仓 32B Qwen3-VL 尚无骨架）—— 见 §5。

---

## 1. 现状（实测，不是推测）

`facade.readiness("h3")` ⇒ `ready=False`，`missingRequired` 只有一条：`te_nvfp4`。

也就是说：**DiT（19.53 GiB）与两个 VAE 都就绪，唯独文本编码器装不进来** ⇒ `Engine.load("h3")` 在装配阶段中止 ⇒
所有「生成 / 性能基线 / 60 分钟稳定性 / 端到端验收」都**无从谈起** ✓✗。

盘上那份 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`（14.61 GiB，`D:\Comfy-Desktop\ComfyUI-Shared\models\text_encoders`）
**已在盘、已校验**（`present=True`、`verified=True`），缺的不是文件，是**解码能力** ✓。

### 1.1 文件规格（用 `safe_open` 直读头部 + `comfy_quant` 真字节）

| 事实 | 实测值 |
|---|---|
| 张量总数 | 2054 |
| `comfy_quant` 小张量 | 351 个 ⇒ **`nvfp4` 350 个** ＋ **`int8_tensorwise` 1 个**（`embed_tokens`） |
| 量化层权重 | `dtype=U8`，如 `mlp.down_proj.weight` = `[5120, 12800]` ⇒ 逻辑 `[5120, 25600]`（**两值/字节**） |
| 块 scale | `weight_scale` = `F8_E4M3`，如 `[5120, 1600]` ⇒ `25600/1600 = **16**`（group size 16） |
| 二级 scale | `weight_scale_2` = `F32` **标量 `[]`**（per-tensor） |
| AWQ 平滑因子 | `pre_quant_scale` = `BF16 [25600]`（**激活侧**缩放：运行时 `x' = x · s` ✓；装载期可等价地折进**权重列** `W'[:,k] = W[:,k]·s[k]` ✓ —— 依据 `Σₖ(xₖ·sₖ)·Wₖ = Σₖxₖ·(Wₖ·sₖ)` ✓，与 `reference` 的 `x = x * pre_quant_scale` 同义 ✓） |
| 不量化的部分 | `norm` 一类仍是 `BF16` |

⇒ 这是**教科书式 nvfp4（AWQ 变体）**：4 bit E2M1 权重 · group 16 · e4m3 块 scale · f32 二级 scale · AWQ 预缩放。

### 1.2 清单里**没有**第二选择

`configs/models.json` 16 条里，32B 文本编码器**只有 nvfp4 这一版**：

| key | 文件 | 是否必需 |
|---|---|---|
| `te_nvfp4` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`（14.61 GiB） | **必需** |
| `te_qwen3_4b` | `qwen_3_4b.safetensors`（7.6 GiB） | 可选 |

⚠️ `te_qwen3_4b` **不是**替代品 ✗：它是**另一个模型**（4B），H3 训练时用的是 32B Qwen3-VL ⇒ 换它等于换条件编码器，
文本理解与指令遵循都会掉（按「不许为制作便利降质量」的口径，这条**不能**当作省事方案 ✓✗）。

---

## 2. 三条路

### A. 实现 nvfp4 解码（保持 4 bit 存储 · 逐层还原）
**做法**：新增 `fp4(E2M1)` 解码表 + 按 group 16 上采样 `weight_scale`、乘 `weight_scale_2`，再按 AWQ 语义乘
`pre_quant_scale`；**不整份反量化**（32B ⇒ bf16 是 ~64 GiB，22.49 GiB 的卡装不下 ✗），而是**用完就放**：
单层最大 `down_proj` 是 5120×25600×2 B ≈ **250 MiB**，逐层安全 ✓。

**不确定性已被消掉**（这是推荐它的关键）：本机 `comfy-studio/ComfyUI/comfy/` 里有**现成实现可照抄** ✓ ——

| 依据 | 位置 |
|---|---|
| fp4 = E2M1、最大值 6.0 | `comfy/float.py:160` `F4_E2M1_MAX = 6.0` |
| 量化/打包与舍入 | `comfy/float.py:159` `stochastic_round_quantize_nvfp4_block`、`:176`、`:196` |
| 二级 scale 的算法 | `comfy/float.py:168`（`amax / (F8_E4M3_MAX · F4_E2M1_MAX)`）、`quant_ops.py:160` |
| 装载期布局与 scale 配对 | `comfy/ops.py:1218-1222`（`quant_format == "nvfp4"` ⇒ 缺 scales 直接报错）、`:1518`（`weight_scale_2` 语义） |
| 本机**没有** nvfp4 硬件 | `comfy/model_management.py:2035` `supports_nvfp4_compute`（sm_89+）；本机 sm_86 ⇒ 只能 emulated ✓ |

⇒ **不是靠猜打包顺序** ✓：解码表与半字节顺序照 `float.py` 的量化实现反推，且有 `stochastic_round_quantize_nvfp4_block` 的
“量化→反量化”自洽性可做单元验证（同一实现往返误差应 ≤ 半步长 ✓）。

**成本**：约 1–2 天（解码表 + group 上采样 + AWQ 缩放 + 与 `quant.plan_dequant` 的族谱对接 + 判据）。
**风险**：中等偏低（有参考实现）；真正的难点是**逐层还原的挂载点**（要接管 `Linear.forward` 前的权重还原，
不能把 64 GiB 全解出来 ✓），属工程而非算法风险。
**质量**：与上游一致（就是同一个文件的同一套数值）✓。

### B. 换权重：找 32B Qwen3-VL 的 fp8/bf16 版
**做法**：下载同模型 fp8（约 32 GiB）替 `te_nvfp4`，改清单即可 —— **零解码工作量** ✓。
**成本**：带宽/磁盘（共享池 789 GB，够 ✓）＋ 清单改一条 ＋ 一次装载验证。
**风险**：fp8 版**得存在**且与 H3 的期望**逐层对齐**（同 revision ✓）；不同来源的 fp8 可能带不同 scale 语义。
**质量**：与 nvfp4 版**不完全等价**（fp8 e4m3 精度更高 ⇒ 更好或持平，不会更差 ✓）。
**卡点**：这条**要联网 + 要创作者点头下载 32 GiB** ✓✗ —— 我这边不能自行决定下载。

### C. 明确不支持，H3 视频阶段在本仓停用
**做法**：`readiness` 保持 `ready=False`，文档写明「H3 需回到 ComfyUI 跑」，本仓只做编排。
**成本**：0。
**代价**：E5.2 性能基线、E3.5 稳定性、端到端验收**全部作废**（它们都要真跑 H3）⇒ 整个「引擎独立」的目标落空 ✗。

---

## 3. 推荐

**A 为主，B 作兜底**：

1. 先按 A 实现（有参考实现、不猜、质量与上游一致、不需要下载）；
2. 若逐层还原的工程量比预期大（例如 transformers 的 `Linear` 挂载点太绕），**临时**用 B 顶上先让链路通，
   A 作为长期解 —— 但 B 要创作者批准下载。

**不同意的做法**（写下来是为了以后别有人这么干 ✗）：
- ✗ 把 U8 的 4 bit 打包当 fp8 解释（数值全错且**不报错**）—— 这正是 2026-10-30 修掉的那个误导
  （`plan.quantScheme` 曾把 nvfp4 报成 `fp8 + scales`，见 `tests/engine_inventory_test.py` 用例 ⑤）；
- ✗ 把 `te_qwen3_4b` 当替代品（换模型 ⇒ 掉质量）；
- ✗ 静默降级成「跳过 TE」或「只跑 DiT」（H3 没有文本条件就出不了片 ✗）。

---

## 4. 请创作者裁定（三问）—— 已裁定

1. **走 A 吗？** ⇒ **是**（2026-10-30：「`.reference` 可以参考这里的代码，继续实现」）✓
2. 若 A 的工程量超预期，**允许下载 32B fp8 文本编码器（约 32 GiB）走 B 吗？** ⇒ 未启用（A 已够，不需要下载 ✓）
3. **C 的代价**你是否接受？ ⇒ 不适用（没走 C ✓）

---

## 5. 落地记录（2026-10-30）

### 5.1 做了什么

| 件 | 位置 | 判据 |
|---|---|---|
| 解码器 | `app/services/engine/quant_nvfp4.py` | `unpack_e2m1` / `from_blocked` / `dequantize` / `fold_pre_quant_scale` / `dequantize_state` |
| 逐层体检 | 同上 `layer_plan()` / `plan()` | 全部是**等式**（形状 × 元素数），无一处猜测 |
| 族表 | `quant.SUPPORTED_FORMATS` ＋ `quant.supported_format()` | 与 `UNSUPPORTED_FORMATS`（仅 `mxfp4`）成对 |
| 唯一口径 | `loader.dequant_plan_of()` | `plan_component` 与 `facade.inspect_weights` 都走它 |
| **装载接线** | `weights.load_module_weights()` | `nvfp4_layers_of()`（逐层认族，与体检同一入口）＋ 打包形状放行（列数**恰好**一半 ✓）＋ 写前逐层预检（不合格整份中止 ✓）＋ float32 里还原再降 dtype |
| 判据 | `tests/engine_quant_nvfp4_test.py`（35 条） | 规格手推 ＋ 参考实现逐位对拍 ＋ `comfy.float::to_blocked` 往返 ＋ 真权重实测 ＋ 端到端装载 |

### 5.2 三条独立真源（为什么这次不是「看着像」

1. **公开规格**：E2M1 值表 `0,.5,1,1.5,2,3,4,6` ＋ 符号；一字节两值、**高半字节存元素 0**；
   块 scale 每 16 元素一个；cuBLAS 128×4 tiled 布局。
2. **参考实现**：`reference/lora/musubi-tuner/src/musubi_tuner/modules/nvfp4_utils.py`（抬头写明已对该 32B 文本
   编码器做过 BF16 逐位核对）—— 用 `ast` 抽其函数原文执行，`torch.equal` 对拍，含三种 padding 情形。
3. **写文件那一方**：`d:/code/voide/comfy-studio/ComfyUI/comfy/float.py::to_blocked` —— 直接用它的输出做
   `from_blocked` 往返（含真实 `(2560,160)` 形状），证明布局不是"我们跟参考实现一起错"。

**反证**（比正例更能说明判据有效）：真权重上故意不解 swizzle ⇒ 相对差 > 0.1；把 LUT 两半字节顺序对调 ⇒
判据 21/26（五项变红）；把 `dequantize` 的 `from_blocked` 换成「按行主序直接截断」⇒ 31/35（②×3 与
「不还原 swizzle 就该不一样」那条反证同时变红）；三次复原后 35/35。

### 5.3 顺手堵掉的两个真隐患

1. `dequantize` 对块 scale **元素数**原本只依赖 `from_blocked` 的 reshape：**少**一截会崩（好查 ✓），
   **多**一截会静默错位（最坏 ✓✗）⇒ 加了前置等式校验，报「实际 N 个 ✗ ↦ 应当是 M 个 ✓」。
2. `precheck_dequant` 的输入原来喂的是**整份文件**的键 ⇒ 从 350 层里只装 1 层时，另外 349 层的 scale
   会把这次装载拦下来，且报错指向一个**根本不会被写**的键 ✓✗ ⇒ 现在只喂目标模块自己的键。

### 5.4 仍未落（明确记下来，别当成已完成）

- **前向挂载点**：本仓文本编码器是玩具实现，32B Qwen3-VL 尚无骨架 ⇒ 现在能到「逐层还原出正确 fp32 权重」
  并**逐位装进 `nn.Linear`**（`weights.load_module_weights` 端到端已验 ✓），把权重喂进真实模型的前向
  还没有对象。**不为「看起来接通」写无人调用的假接线** ✓✗。
- 逐层反量化**不做整份还原**：32B 全展开成 bf16 ≈ 64 GiB、fp32 ≈ 128 GiB ⇒ 只能逐层「用完就放」，
  这条与骨架一起落地（`quant_nvfp4.dequantize_state` 的说明里写了同一句）。
