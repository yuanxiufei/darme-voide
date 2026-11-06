# 待下载 / 待补齐清单（2026-11-03 核对）

从 `ComfyUI-Local-Infinite-Canvas` 迁完之后，**本仓能跑的两条路**各自还缺哪些文件：

* **自研引擎**（`backend-py/app/services/engine/`，走 safetensors 直接装）—— 只认装配参数，不认文件名。
* **ComfyUI 工作流**（`configs/comfyui/workflows/` 的 21 份）—— 走 ComfyUI，**认死文件名**。

⚠️ 本清单**按"文件名在共享模型池里能不能找到"判**，不是按目录判。踩过的坑：
`krea2_turbo_int8_convrot` 落在 `diffusion_models/Krea2/`、`ema_vae_fp16` 落在 `vae/SEEDVR2/`、
`z_image_turbo_bf16` 落在 `unet/` —— 按 target 原路径判会得出"缺 64 件 / 118 GB"的假结论 ✗。

---

## A. 自研引擎侧（2 件，唯一真缺口）

| 文件（常见发布名） | 落点 | 体积 | 用途 |
|---|---|---|---|
| `t5xxl_fp16.safetensors` 或 `t5xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | ~9.8 / ~4.9 GB | Flux 的**宽条件塔**（`dit_build` 里 flux 分支按 4096 宽装配） |
| `clip_l.safetensors` | `text_encoders/` | ~0.25 GB | Flux 的 **pooled 向量塔**（768 宽） |

**为什么确认是缺**：全池搜 `t5xxl*` / `clip_l*` / `clip_g*` / `t5-xxl*` 命中 **0**；
`text_encoders/` 现有 9 份里没有这两座塔；`dit_flux.py` 模块头写明"两座塔都没有 ⇒ 条件由外部注入"，
所以 flux 目前只能到**结构验证**（`attach_flux_conditioning` 喂向量），**出不了语义图** ✗。

⚠️ 引擎侧**不认文件名**（TE 是装配时传的），所以上面写的是社区常见发布名 —— 只要宽度对
（4096 / 768）且结构是 T5-XXL + CLIP-L，改配置即可，不必强求同名。

---

## B. ComfyUI 工作流侧（13 件待下载）

按 `configs/comfyui/workflows/*.json` 里 loader 节点写死的文件名逐份比对得出的，**改不了名**（改了要同步改 21 份工作流）。
合计约 **47.7 GB**。

### B1 有明确下载源（12 件）

| 文件 | 落点 | 体积 | 堵住的工作流 | 源 |
|---|---|---|---|---|
| `moss_image_sft_bf16.safetensors` | `diffusion_models/` | 9.47 GB | `image-moss-sft-50step` / `-8step` | modelscope `Comfy-Org/MOSS-Image` |
| `moss_image_sft_int8_convrot.safetensors` | `diffusion_models/` | 5.04 GB | `image-moss-sft-int8-50step` / `-8step` | 同上 |
| `qwen3.5_9b_int8_convrot.safetensors` | `text_encoders/` | 9.08 GB | 上面四个 moss 全要它 | 同上 |
| `flux2-vae.safetensors` | `vae/` | 0.31 GB | 上面四个 moss 全要它 | 同上 |
| `krea2_turbo_fp8_scaled.safetensors` | `diffusion_models/` | 12.24 GB | `image-krea2-turbo` / `image-krea2-four-view` | modelscope `Comfy-Org/Krea-2` |
| `qwen_image_2.1_turbo_int8_convrot.safetensors` | `diffusion_models/` | 6.76 GB | `image-qwen21-*` 各档 | modelscope `Comfy-Org/Qwen-Image-2.1` |
| `Qwen21-图像扩展qwen-image-2.1-outpaint-v2.safetensors` | `loras/Qwen-功能2.1/` | 0.15 GB | `image-qwen21-outpaint` | modelscope `ausboss/Qwen-Image-2.1-Outpaint-LoRA` |
| `MOSS-Image-SFT-8-steps.safetensors` | `loras/Moss_image/` | 1.11 GB | `image-moss-sft-8step*` | modelscope `lodestones/MOSS-Image-SFT-8-steps-dmd` |
| `minimax_h3_latent_upscaler_3d_fp16.safetensors` | `latent_upscale_models/` | 0.64 GB | h3 三个工作流 | modelscope `LBH-123-AI/Minimax_h3_latent_Upscaler` |
| `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | `loras/` | 1.82 GB | `video-h3-ref2va-drama` | modelscope `Comfy-Org/MiniMax-H3` |
| `h3_character_swap_v1.5.safetensors` | `loras/` | 0.14 GB | `video-h3-character-swap-vdn` | modelscope `akatz-ai/MiniMax-H3-Character-Swap-LoRA` |
| `RealESRGAN_x2plus.pth` | `upscale_models/` | 0.06 GB | `image-esrgan-upscale` / `video-esrgan_2x` | modelscope 数据集镜像 `shensanshu/...-Models` |

⚠️ `krea2_turbo_fp8_scaled` 缺，但盘上有 **int8 那一档**（`diffusion_models/Krea2/`）。
两者是**不同权重** ✗ —— 不能靠改文件名顶替（fp8 与 int8-convrot 的数值不同，引擎侧实测产出也不同）。

### B2 清单登记"随包"、但源项目里其实没有（1 件）

| 文件 | 落点 | 体积 | 堵住的工作流 |
|---|---|---|---|
| `Krea2-功能-角色四视图.safetensors` | `loras/` | 0.85 GB | `image-krea2-four-view` |

源 `resources/index.json` 把它标成 `bundled`（随包），但 `resources/loras/` 下**没有这个文件** ✗
⇒ 清单与实际不符，需要另外找源（HF / 魔搭上 Krea-2 四视图 LoRA 的转档）。

### B3 已在源里、本轮已直接迁入（1 件 ✓）

| 文件 | 来源 | 落点 | 结果 |
|---|---|---|---|
| `taeh3.safetensors` | 源 `resources/vae_approx/`（9.3 MB，随包） | `models/vae_approx/` | **已拷入** ✓（目标目录原本是空的） |

---

## C. 明确**不需要**下载的（源清单里有，本仓用不上）

源的 `resources/index.json` 共 73 条，除上面 15 件外，其余 57 条**对本仓没有价值**，别照着清单全下（77.8 GB）：

| 类别 | 条数 | 为什么不下载 |
|---|---|---|
| `custom_nodes/*` | 17 | 本仓自研引擎**不跑 ComfyUI 节点**；ComfyUI 路径的节点已单独装过 |
| `LLM/*`（llama.cpp / Florence / SageAttention） | 5 | 那是源项目 prompt 优化器的本地推理依赖；本仓 prompt 打磨走自有 provider |
| `vdn/*`（stage-dmd 等） | 8 | H3 的 VDN 分阶段蒸馏，只有源项目的 ComfyUI 节点用 |
| `SEEDVR2/runtime/*` | 5 | ComfyUI 节点的 python 运行时包；本仓 seedvr2 走自研引擎（已端到端跑通） |
| ONNX / TensorRT 加速件 | 4 | 本仓走 torch，不用 ONNX/TRT 后端 |

---

## 复现方法

```powershell
# 1. 取源清单
$idx = 'd:/code/voide/anli/ComfyUI-Local-Infinite-Canvas/comfyui-gateway/resources/index.json'

# 2. 从工作流 loader 节点反推"本仓到底要哪些文件名"（API 格式：顶层就是节点表）
cd d:/code/voide/darme-voide/configs/comfyui/workflows
#    遍历每个 json 的 <nid>.inputs 里以 .safetensors/.pth 结尾的字符串

# 3. 拿文件名去共享池**全树**比对（不要按 target 目录比对，见文首的坑）
#    池根：D:\Comfy-Desktop\ComfyUI-Shared\models
```

下载落点一律是 `D:\Comfy-Desktop\ComfyUI-Shared\models\<target 相对路径>`。

⚠️ 源项目的下载器（`comfyui-gateway/src/resources.js`，2.5k 行，含分片/断点/校验）**不迁** ✗ ——
那是分发层，本仓按上表手工下即可；真要批量，直接用 `sources[]` 里的 URL 逐个拉，sha256 在清单里也有。
