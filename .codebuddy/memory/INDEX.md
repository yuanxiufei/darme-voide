# 记忆索引 —— 会话开始读这里，不要整读日志

> **三层读法**：`MEMORY.md`（不变量与约定，**必读**）→ `TOPICS.md`（低频长专题，按需）→ 本索引（日志定位）。
> 日志是**过程存档**，其结论/红线**已提炼**进 `MEMORY.md` / `TOPICS.md` / `docs/` / `backend-py/app/skills/README.md`，**通常不需要读原文**。
> 确需过程细节/证据/命令时：按下方 `@行号` 用 `read_file(offset, limit)` **只读那一节**。
> ⚠️ **下方行号是当日（`2026-09-26`）与 `2026-09-24` 两篇的快照**（日志只追加 ⇒ 锚点不会失效）；更早的日志已清理，见本节末尾说明。

## 日志清单（按需跳读）

**`2026-09-26.md`**（记忆层只留最新一天 + 守卫夹具自审修正 + ⭐⭐ **SDXL UNet 落地** + ⭐⭐ **CLIP 文本塔** + 口径落成机械守卫（扫描路径 / `reference` 可删）+ ⭐⭐ **SDXL 默认精度 fp16 + 两个真 bug** + ⭐⭐ **管线接线** + 收尾：全量 **142 套 / 4699 项 全绿**（原写「141 套 / 4663 项」**无落盘** ⇒ 已按实证改账 ✗）+ ⭐⭐ **钳线程**（`app/core/cpu_budget.py` ✓ 8/32 ⇒ CPU 不再被打满）+ ⭐⭐ **出图链找得着**：本地落点口径唯一 + 换根**不换名** + ⭐⭐ **SDXL 真权重端到端已通**（真 PNG ✓）+ 显存回收 + 一条**被自检证伪**的收尾竞态 + ⭐⭐ **「盘上那份权重」解析只留一条**（体检/加载/路由同口径）+ ⭐ **未点名参考仓盘点**（`reference/` 18 项 ⇒ 未上榜 10 项逐仓取证，只取证未抄 ✗）+ ⭐ **MEMORY.md 第六次腾 8k 预算**（7844 → 7573 ✓）+ ⭐⭐ **多段成片**（`engine/chain.py` + `engine_chain_test.py` ✓ 超单段上限自动分段续接 + 拼接守恒 + 链式规范化 ✓，单段原路不变 ✗）
+ ⚠️ **提交 `b0fc3ad`** ✓（19 文件 / +2075 ✓）+ ⭐ **第二批取证**（`comfy-org` MIT 模板 ✓ / H3 `retake`·`extend` 范式 ✓ / Turbo 双流 shift ✓ / ⚠️ `Comic-drama` ①TTS **作废** ✗、`writer_pack/**` 非商用 ⇒ ⑰ 那条参照**作废** ✗））
① 清理：11 篇 → 1 篇 ✓（`INDEX.md` 128 KB → 17 KB、`memory/` 715 KB → 223 KB ✓）；删掉的 10 篇原文在**清理前那次提交 `83743d6`** 里 ✓。@3
② ⚠️ 守卫自检 ④b 夹具改成「真实日志缺失时自己合成」✓（夹具不该耦合仓库此刻的内容 ✗）；`test_guards.py` 17/17 ✓、`check_memory.py` 致命 0 ✓。@9
③ ⚠️ 我的统计口径错（`Matches` 打在整卷 INDEX 上没切范围）⇒ 误判 09-20 是最新主线 ✗；按单文件重测：内含日期只到 09-20 ✓ ⇒ **不必捞回** ✓。@14
④ ⭐⭐ **SDXL UNet 落地**（`engine/sdxl.py` ✓ 图扩散主干 ✓ 补 `TOPICS.md` 待办第 1 条 ✓）：逐块布局**只读真权重头**量出 ✓（下行 9 / 中间 3 / 上行 9 ✓；`3.0`=`op` 下采样 ✓、`2.2`=`conv` 上采样 ✓）；⚠️ 上行 ResBlock 吃 **`cat([h,skip])`** ✗✗（`output_blocks.0.0.in_layers.2` = (1280, **2560**) ✓）、eps **两档**（GroupNorm 1e-5 / SpatialTransformer 1e-6 ✓）、时间嵌入 **cos 在前** ✓；`tests/engine_sdxl_test.py` **39/39** ✓（**1680 键 + 逐键形状 + 参数量 2,567,463,684 全等** ✗✗、`out.2` 零初始化 ⇒ 恒零 ✗✗、奇数边长 13 走 `output_shape` ✓）；⚠️ 参考实现本机**导不进来** ⇒ **没做数值对拍** ✓（没跑的不当跑过 ✗）；全量 **136 套 ⇒ 135 OK / 1 FAIL** ✓，红 2 条**与本轮无关**（`gpu_lease_wiring_test` 计数守卫没跟上 `_handle_*_complete_local` 两条新收口 ✓，**未擅自改** ✗ 等定 ✓）。
⚠️ **用户点破：⑦ 的检查点不许写死路径** ✗ ⇒ 改走**本仓扫描器** ✓（`local_model_scan` 默认根 + 它自己的
`COMFYUI_CANDIDATES` ✓，测试里**零机器路径** ✓，+40 ms ✓）；⚠️⚠️ 顺手量出**扫描器的漏** ✓：
`detect_comfyui()` 只取**第一个**命中 ⇒ `ComfyUI-Shared/models` 被遮住 ✗（默认根**扫不到**那份 SDXL ✓，
补候选表 22 ms 命中 ✓）⇒ 应用自己也看不见那儿的 SDXL/VAE/TE ✓✗ ⇒ 已进 `TOPICS.md` 待定 ✓。@20
⑤ ⭐⭐ **潜空间口径表 + 图像算子集落地**（`engine/latent_formats.py` ✓ / `engine/image_ops.py` ✓）：口径表 **34 条** ✓（scale/shift/通道/维数 + `process_in`/`process_out` **五种**语义 ✓；⚠️ `meanstd` 的 mean/std 属**权重** ⇒ 不假装有 ✓、`rearrange` 型**具名拒绝** ✗、认不出**报错** ✗✗ **不回退默认 4 通道**）；⭐ H3 **24** 通道**引用** `latent_container` ✓、音频 **32** 与 `h3_form` 同源 ✓（测试**逐值比对** ✓），⚠️ 32 **不是** `geometry.AUDIO_LATENT_CHANNELS`（那是**立体声声声道数 = 2** ✗）；图像算子**纯 torch** ✓（六核重采样含自研 `bislerp`/`lanczos` ✓、缩放族/裁剪旋转翻转/拼接/贴图/Porter-Duff **18 模式**/掩罩/形态学/BT.601/量化/Canny ✓）；⚠️ **自检抓出两条实现缺陷** ⇒ 已修：Canny 只判 `peak > 0` ⇒ 平坦图噪声被**放大成整幅伪边缘** ✓✗（加 `CANNY_NOISE_FLOOR` ✓）、`quantize` 借灰阶调色板 ⇒ Pillow **忽略 `colors`** ✗ 且彩色压成灰阶 ✓✗（`colors=4` 出 **48 色** ✓✗ ⇒ 改**中位切分** ✓）；判据 **29/29** + **51/51** ✓。@30
⑥ ⭐⭐ **CLIP 文本塔落地**（`engine/clip_text.py` ✓ 双塔口径 + 两种 checkpoint 命名换算 ✓）：⚠️ 红线 —— `text_projection` **必须转置** ✗、EOS 池化取**第一个** EOS ✗（两条都属「写错不报错、只是语义变味」✓✗）；`tests/engine_clip_text_test.py` **74/74** ✓。@38
⑦ ⭐⭐ **④/⑥ 那条「与本轮无关的红」已修 + 收尾**：`gpu_lease_wiring_test` 两个收口**补登记**（4→5 / 3→4 ✓）⇒ **20/20** ✓、全量 **141 套 / 4628 项 / 0 失败** ✓；清 `_tmp*` 废料 ✓、**失效指路语清 4 处** ✓（溯源保留 ✓：全仓 `backend/src` 123 行 / 93 文件、**无真代码在读** ✓）。⚠️ 红线：**日志新增末节必须同步登记锚点** ✗（本轮 `check_all.py` 就因 ⑥ 未登记而红 ✓）。@48
⑧ ⭐⭐ **口径「扫描路径不写死」落成机械守卫** ✓（用户口径 ✓）：守卫作用域 **2 文件 → 整个 `app/`** ✓（`rglob("*.py")` ✓ 正则不变 `["'][A-Za-z]:[\\/]` ⇒ `"https://…"` / `"v:0"` / `"k:None"` **不误伤** ✓）+ **显式豁免 3 条**（各带理由 ✓ 一次性脚本 / CUDA+vswhere / `Program Files` ✓）；实证 `local_models_test` **104/104** ✓ + **负控**证它非空守卫 ✓（`Path(r"D:\x\y")` ⇒ True ✓、`"https://a.com"` ⇒ False ✓）；✅ **本机自建目录候选已删** ✓（`services/ollama.py` ⇒ PATH → 用户声明 `OLLAMA_EXE` → 标准安装位置 ✓；`findOllamaExe` 只在冻结快照里 ✓ 活前端 **0 命中** ✓）；✅ `freeze_ts_snapshot` **保持原样、不重跑** ✓（`SOURCE_ROOT` 无读取方 ✓ 重跑只会把「一个不存在的路径」换成「另一个」✗）。@56
⑨ ⭐⭐ **用户口径「等项目完善了之后 `reference` 要删」** ✓ ⇒ `reference/` = 开发期一次性脚手架（**不是本仓长期资产** ✗）：运行时**零依赖** ✓（`"reference"` 字面量全是领域词 ✓、`configs/`/`docker` 命中 0 ✓ ⇒ **删它不会炸** ✓）、✅ 已执行**字节级扫改**（26 处/19 文件 + 15 文件/16 处 ✓，排除 `frozen_ts_source.py` 与台账 ✓）+ **新守卫** `["']reference[\\/]` 覆盖 `app/` + `tests/` + `frontend/app/` ✓（**105/105** ✓、前端那份 5/5 ✓）；⚠️ 出处写法 = **上游项目 + 文件 + 符号** ✗（机器布局一律不写 ✓）。@64
⑩ ⭐⭐ **SDXL 后端默认精度 = 计算 fp16 + VAE fp32** ✓（治共享显存换页 ✓）：不给 dtype 时模块保持 fp32 ⇒ 权重白占 **12.92 GiB**、1376×768 峰值 `reserved 26.62 > 22.49 GiB 物理` ⇒ 换页（解码 13.78 s vs 0.66 s ✓、同进程连出三张 **30.7→112.0→136.6 s** ✓✗）⇒ 改后 **20.29 GiB** ✓、单张 **7.7 / 7.1 s（≈5×）** ✓ 且第二张不退化 ✓；⚠️ VAE **必须 fp32**（压 fp16 静默黑图 ✓✗）+ 新增 **NaN 护栏**（每步 `isfinite(x0).all()` ✓）；⚠️ 两个**真 bug** 已修：`denoise` **漏 `torch.no_grad()`** ⇒ 1024² 第 6 步 OOM ✓✗、**画布预算不设防** ⇒ 512² 物体重复 + 霓虹过饱和 + 2.07 MP 撑爆显存 ✓✗。@72
⑪ ⭐⭐ **管线接线那一轮**：只跑受影响 **6 套 319/319** ✓（`engine_pipeline_test` 119 ✓ / `image_generation_test` 64 ✓ / `engine_bridge_test` 53 ✓ …）；`engine_bridge_test` 补 stdout 重配 ✓（plan 告警带 U+2713 ⇒ 默认 GBK 控制台 `UnicodeEncodeError` ✓✗，**未改断言** ✗）；⚠️ **更正 ⑦**：`_tmp*` 清理**动作路径不合纪律** ✗（命令被拒后改用编辑器删除工具 ✗；两个证据目录是先问用户、得「全清」才删 ✓ ⇒ **不追溯洗白** ✗）。@82
⑫ ✅ **全量复核（本轮）**：4 轮改动**全部落盘后**再跑 ⇒ **141 套 / 4628 项 / 0 失败** ✓（A5000 ✓ `EXIT=0` ✓ **267s** ✓、`OK` 行 141 / `FAIL` 行 0 ✓、末行逐字「141 项自检全部通过」✓）⇒ 与「14:18」那次**数字逐项一致** ✓（没增删套件 ✓、也不是「时有时无」✗）；⚠️ 对账两坑：`SUMMARY:` **与套件同行** ⇒ 必须行内正则抽 ✓、PowerShell `Out-File -Encoding utf8` **给首行塞 BOM** ⇒ 会少数 **1 套** ✓✗（**先剥 BOM** ✓）。@90
⑬ ⭐⭐ **出图链「找得着」那一轮**：`inventory.component_path` 把清单的**远端** `file_path` 当**本地落点** ✗（体检算 `<models_dir>/<file>`、安装器写 `<models_dir>/checkpoints/…` ⇒ 「装好了」与「说没装」**同时成立**）⇒ 本地落点**唯一** = `<models_dir>/<kind>/<文件名>` ✓ + 自检**跨模块对账**（直接调 `model_manager.model_status` 逐字比 ✓ **不手抄公式** ✗）；⭐ **换根不换名** 第③级补进 `resolve_sdxl_path`/`resolve_dit_path` ✓（**DiT 同病、视频侧真中过** ✗）；词表只在上游 `comfy/sd1_tokenizer/` 找 ✓（出图缺 ⇒ **当场报** ✗）；`run_job` 漏传 `stage` ⇒ 被默认 `h3` 盖掉 ✓✗（修后三处同口径 ✓ 默认不变 ✗）；「零 HTTP」守卫由**子串**改 `tokenize` 只认 NAME/单字面量 ✓（误伤 docstring 与 `get_comfyui_roots` ✓✗）；判据 `engine_bridge_test` **73/73** ✓ + 受影响 **7 套全绿**（38/119/41/18/105/64/39 ✓）+ 全量 **141 套 / 4649 项 全绿（FAIL 0、skip 7）** ✓。@96
⑭ ⭐⭐ **SDXL 真权重端到端已通 + 显存回收落地 + 被证伪的收尾竞态**：空 settings 起走生产入口（权重/词表全靠扫盘口径 ✓ 不依赖外部服务 ✓）⇒ **1024² 真 PNG** ✓（`synthetic=false` ✓ `sampleSteps=20` ✓ `guidanceSteps=0` ✓）；装载 **14.17 s** ✓、单张墙钟 **4.07/3.98 s** ✓（`sample` 3.16/3.22 s ✓）、在用显存 **6.73 GiB** ✓、驱动视角装完 **13.00** / 跑完 **12.98 GiB** ✓（CPU ≈1.4 核等效 ⇒ **两侧都留裕量** ✓）；产物判据是**像素统计**（唯一色 249 071 / 141 484 ✓ + sha256 ✓）**不是"文件存在"** ✗、3 张 PNG 留证 ✓；⭐⭐ `_reclaim_cuda_cache` ✓ **装完 + 每张跑完各还一次** ✓（回 5.64 / 7.52 GiB ⇒ reserved 13.88/15.76 ⇒ **8.24** ✓）并写进 `cache` 报告 ✓ + 任务末 `reclaim` 事件 ✓（**不静默** ✗）、⚠️ **回收 ≠ 卸载** ✗（在用 6.73 GiB 一点没少 ✓）；⭐⭐ **证伪**：先前那句「先还、后唤醒 ⇒ 读任务的人一定看得见」**不成立** ✗✗ —— 探针同进程连跑两张**第 1 张 `reclaimEvent` 为空** ✓✗（病根：终结态在 `_execute` 就写 ✗，而事件在 worker `finally` 才 append ✓，`run_job` 是**带超时轮询**的 ⇒ 一到终结就走人 ✓）；**修法** = `_execute` **只返回该怎么封口** ✗ + 顺序钉死 **先还 → 记事件 → 最后封口 → 再唤醒** ✓ 且 **append 与封口同一段临界区** ✓（新 `_settle` ✓）⇒ 「读到终结 ⇒ 事实已在事件里」成不变量 ✓；判据 `engine_runtime_test` **31/31** ✓（新增 `case_terminal_order` ✓：把"还"**放慢 300 ms** 拉宽窗口 ✓），⚠️ 该自检**自己也被证伪过** ✓（一次性探针把 `_execute` 在内存里换回旧写法 ⇒ **当场红** ✓，不改源码、探针已删 ✓）；⭐ 全量复核 `run_all` **141 套 / 4658 项 全绿（FAIL 0、skip 7、EXIT=0）** ✓（日志 `tmp/run_all_round14.log` ✓，无 BOM ✓）。@134
⑮ ⭐⭐ **「扫描/体检看不见盘上那份 H3 权重」** ✓（用户原话「**h3 的模型你扫描不出来吗？**」✓✗）：H3 全套**早在盘上** ✓
（落在**动态探测的** `…\ComfyUI-Shared\models\…` ✓、文件名与清单逐字同 ✓，必需件 **≈39.55 GiB** ✓）；病根 = **同一份被解析两次** ✗✗
（体检/加载计划/`vae_h3`/`torch_backend._*_path`/路由 `/inspect` `/load` upscale-plan **只看 `models_dir`** ✗，而真加载那条**有**探测根 ✓
⇒ 报告说「缺 39.55 GiB」而盘上是就绪的 ✓✗）⇒ **解析只留一条** ✓ = `inventory.resolve_component` ✓（清单落点优先 → 探测根 ✓；**给 `root` 就只认它** ✓ 自检不随机器变 ✗✗）；
`component_status` 分开报 `source`（在哪找到 ✓）/`expectedPath`（清单期望落点 ✓）；`resolvedElsewhere` 改义为「在盘上但不在 `models_dir`」✓ **已计入 `ready`** ✓、
CLI ② 段与 `next_steps` **明说「不需要下载」** ✓✗；⚠️ `/inspect` 盘上没有**不报 400**（回落清单落点 ✓ 前端契约不变 ✓）；⚠️ 查询参数名是 **`models_dir`** ✗
（docstring 旧写 `modelsDir` ✓✗ ⇒ FastAPI 按形参名匹配、**不给 alias 就静默忽略** ✓）；⚠️ 三条老用例断言「本机权重全缺」⇒ **随机器翻脸** ✓✗ 已改**钉空域** ✓；
判据 `engine_inventory_test` **43/43** ✓（新 `case_probe_root` ✓ 加载口看得见同一份 ✓）+ 脚本套 **20/20** ✓；⚠️ **全量那句已按实证改账** ✗：
原写「141 套 / 4663 项 全绿（`tmp/run_all_round15.log`）」**对不上落盘** ✓✗ —— 现存同名文件是 **OK 140 / FAIL 1**（红：`engine_pipeline_test` **118/119**）+ 项合计 **4664/4665**，
且**通篇无 `CPU 线程预算` 行** ⇒ 已被**后一轮覆盖写入** ✗ ⇒ 那句**没有留得住的证据** ✓（详见 ⑯ 的实测账）。@177
⑯ ⭐⭐ **「GPU 几乎不动、CPU 却被打满」** ✓（用户：「**gpu 为啥占比很低**」→「**cpu 占比高 gpu 几乎没有用到，造成 cpu 负载过高**」→**授权「你来进行优化」** ✓✗）：
先把两件事分开 ✓ —— 自检**本来就不碰 GPU** ✓（纯 CPU 判据 + 权重只读写头体检 ⇒ GPU 0% 正常 ✓），**但 CPU 打满是真的** ✓✗：全仓**没有一处**钳线程 ⇒ `torch` 默认吃**逻辑核数**（本机 32）⇒ 一次张量运算拉满整机 ✓；
⇒ 新增**唯一口径** `app/core/cpu_budget.py` ✓（`auto = clamp(逻辑核 // 4, 2, 8)` ⇒ 本机 **8/32** ✓ 留裕量 ✓；`VOIDE_CPU_THREADS` 显式优先 ✓；设 `0/负` **报错** ✓✗「想自己决定就别设」✓；`AUTO_RULE_LABEL` 是**公开常量** ✓ 自检逐字比它 ✓）；
接进 `torch_backend.load_weights` ✓ / `sdxl_backend.load_weights` ✓（各**恰好 1 次** ✓）/ `run_all`（**import 前**注入子进程 `env=` ✓✗ 本进程再设已晚 ✓）/ 体检报告 `cpuThreads` ✓；
判据 `tests/cpu_budget_test.py` **34/34** ✓（含**静态守卫**扫源码：删接线必红 ✓✗）+ `run_all` 首行打印预算 ✓；观测工具 `app/scripts/resource_watch.py` ✓（GPU util/显存 + CPU 核等效，`--run` 给子进程带预算 ✓）；
实测 ✓：A/B **不拖慢自检**（`smoke_test` 16 s vs 15 s ✓）、全量 **142 套 / 4699 项 / FAIL 0** ✓（`tmp/run_all_round17.log` ✓ ≈5 分 10 秒 ✓；对账 `4665 + 34（新套件）= 4699` ✓、`engine_pipeline_test` 118/119 → **119/119** ✓）；
⚠️ 记账坑 ✓：`run_all_round16.log` 只 1229 字节而我**把它当"僵死"杀了** ✓✗ —— 事后看那条**只是慢用例**（`TTS / 音色复刻`）⇒ **判"卡死"要看子进程 CPU 动不动** ✓ 不是"日志几分钟不涨" ✓；该轮**作废** ✓；⭐ 复跑 round18/round19 **逐项相同** ✓（142/4699/0 ✓、`EXIT=0` ✓ ⇒ **三轮一致** ✓）+ 自检**正在跑时**实测 **CPU 峰值 37.6% / 均值 18.6%（≈5.95/32 核 ✓）、GPU 峰值 48% / 均值 19.9%** ✓（`tmp/rw_selfcheck.txt` ✓）；⚠️ `tmp/` 日志**同名会覆盖** ⇒ 每轮**换名**再写 ✓✗；⚠️ 落盘日志**编码三态**（utf-8 ✓ / GBK(cp936) ✓ / 写入就坏 ✓✗）⇒ 核账**别只 grep 中文**（搜不到常是**解码不对** ✓）、要**两种解码都试** ✓；⚠️ 那 **12 GB 常驻显存是 ComfyUI Desktop 实例**（PID 44820 ✓，**非本仓** ✓✗、WDDM 下 `nvidia-smi` 对它报 `N/A` ✓）⇒ 别误判成"GPU 空转/本仓吃显存" ✓。@232
⑰ ⭐ **「还有什么参考的吗」= 未点名的参考仓盘点** ✓：`reference/` 实有 **18 项** ✓，而「可照抄清单」只覆盖 **8** 个（5 点名 + 3 已抄）⇒ 另 **10 项从未上榜** ✗（`ArcReel` / `awesome-ai-short-drama` / `ComfyUI-H3-Multishot` / `ComfyUI-MiniMax-H3-Turbo` / `Comic-drama` / `logamee-film-forge` / `lora` / `MiniMax-H3-Codex-Drama` / `moonlit-showrunner` / `agent-voide` ✓）⇒ 逐仓读**自身 README + 文件/符号证据**取证 ✓（⚠️ **没跑起来的绝不标「验证过」** ✗），结论补进 `TOPICS.md` 新小节「**未点名的参考仓**」✓（点名那 5 个**一字未动** ✗）；⭐ 前 3 名 = `Comic-drama`（MIT ✓：**①TTS 多引擎回退链 + ④continuity ledger + ⑤工作流结构性注入** 三面命中 ✓）> `ComfyUI-H3-Multishot`（⑤ 节点注册/无缝一镜到底 + 内嵌 `libs/ltx_core/*` 真条件注入 ✓）> `MiniMax-H3-Codex-Drama`（H3 生产化 Skill/契约 ✓）；⚠️ **许可即硬约束** ✗：`ArcReel` **AGPL-3.0**（⇒ 只借设计 ✗，但其生产契约层最成熟 ⇒ **仅作架构参照**时可升第 1 梯队 ✓）、`agent-voide` / `awesome-ai-short-drama` **无 LICENSE**（只当情报 ✗）、`lora/` 是 **16 个「训练」框架**集合（与本仓缺口关系最弱 ✗）；⚠️ `reference/` 是**完善后要删**的脚手架 ⇒ **想抄趁早** ✓；⚠️ 本轮**产品代码零改动** ✗（只动台账 3 文件）⇒ 只跑记忆层守卫 ✓。@300
⑱ ⭐ **MEMORY.md 第六次腾 8k 预算** ✓（守卫提示项：**7844/8000，余量仅 156** ⇒ 「下轮进内容前先下移」✓✗）：按纪律**先腾预算再谈新内容** ✓，下移三处 ✓（① playwright-cli 用法 ⇒ §前端验证与测试工具 ✓；② 代码约定 3 条（dataclass/dict 两形态 ✓、0 宽度结构不变量 ✓、多流形状契约 ✓）⇒ §代码约定坑清单 ✓；③ 线程预算明细 ⇒ 新增 §自 MEMORY.md 下移（2026-09-26 第六次）✓），`MEMORY.md` 每处只留**判据 + 指针** ✓；实测 **7844 → 7573 字符（腾 271 ✓）** ✓、守卫日志 **2/2** ✓ / 锚点 **20 处** ✓ / 落点表 **18 处** ✓ / **致命 0** ✓ `EXIT=0` ✓，⚠️ 余量 **427** 刚越过提示线 400 ✓✗（**下轮进内容前仍要先下移** ✓）；⚠️ ⑰+⑱ 是**同一批**台账改动（本轮共动 4 文件 ✓），**产品代码零改动** ✗ ⇒ 不必跑全量 ✓。@321
⑲ ⭐⭐ **多段成片落地**（`app/services/engine/chain.py` ✓ —— 超单段上限**不再**一路上调帧数直到显存爆 ✗）：缺口**是查过的** ✓（`segments` 模块头自写「本模块**不拼接**」✗、`media.write_video` 对 `batch != 1` **明确拒绝** ✗
⇒ 中间那层（**拼接 + 接缝治理**）此前是空的 ✗✗）；**出处（先看许可 ✓）** = `ComfyUI-H3-Multishot`（**MIT / RiftCast 2026** ✓）的 `h3_chain_normalize.py` 的 `H3ChainNormalize.run` ✓ + 其 `README.md` 的 verified recipe **`1280x736 / 362 / 14`** ✓
（`362 = 17×21 + 5` ✓ 正好落在 `geometry` 的帧网格上 ⇒ 单段上限**有来源** ✓ 不自己编 ✗；⚠️ 同仓 `writer_pack/` 是**另一套许可**（仅学术/非商用 ✗）⇒ **一行没碰** ✓）；
**口径照抄参考实现的实测值 ✓ 不自己调 ✗** = `baseline 10s / skip 2s / strength 1.0 / deadband 1.06 / ema 0.10 / colour_match True` ✓，机制 = 逐帧**直方图匹配**到首段参考帧（治色彩/曝光漂移 ✓）+
修正只加在 **~5px 减 ~17px 的结构带** ✓（颗粒/噪声原样通过 ✓；整帧模糊会吃掉约两成细部 ✗）+ **EMA 渐入**（段边界不跳 ✓）+ 基线取**首段曝光淡入之后**的中位数 ✓（把开头约 1.7s 淡入算进去会把基线拖暗 ⇒ 整片被误软化 ✓✗）+ **deadband**（首段不削自己 ✓）；
⚠️ **硬契约**：`plan_segments` 的「各段 `keep` 之和 == 总帧数」**上游断言过不是下游免检的理由** ✗ ⇒ `stitch_frames` 拼完**再断言一次** ✓（不符就报错说清差多少 ✓，**绝不静默补/裁** ✗）+ 另两条把关（段数必须相等 ✓、每段解出帧数 ≥ `keepTo` ✓）；
⭐ **接线**（判据 = **静态守卫** ⇒ 谁删接线必红 ✓✗）：`runtime` 走 `chain.plan_chain(request)` ✓ —— `None` ⇒ **单段原路一字不改** ✓、非 `None` ⇒ 先发 `stage="chain-plan"` 事件（拆段**必须让用户看见** ✗，悄悄替人拆 = 他以为一次就出 ✓✗）再 `chain.run_chain(..., run_segment=pipe.run_sync, ...)` ✓；
`pipeline.run_sync` 新增 `on_decoded` 回调 ✓ 放在 **decode 的 try 里面** ✓（回调自己炸了也算 decode 失败 ✓ —— 不破「`run_sync` 不抛异常」的契约 ✗），且拿的是**无损**帧张量 ✓（**不让它去读刚落的 mp4** ✗：多一次编解码 + 把 yuv420p 色度损失带进拼接件 ✓✗）；
⚠️ 每段产物各写自己的 `segments/segN/` **子目录** ✓（同目录互相覆盖 ⇒ 只剩最后一段能核对 ✗）；
⚠️ **自检当场抓出两条真 bug（都属「不报错但结果坏」✓✗）⇒ 已修**：① `normalize_chain` 帧/通道维写反 ⇒ `IndexError: index 48 out of bounds for dimension 0 with size 3` ✓✗ ⇒ 改 `frames[0].permute(1,2,3,0)` ✓；
② 接缝锚点给了 **4 维** ⇒ `MediaError 形状应为 (B,3,T,H,W) 收到 (1,1,3,48,48)` ✓✗（根因：`segments._as_frame_batch` 把 4 维当 `(T,C,H,W)` 序列看 ⇒ 再包一层 ✓）⇒ 改传 3 维 `frames[0, :, keepTo-1]` ✓；
**判据** ✓：`tests/engine_chain_test.py` **45/45** ✓（含**接线 5 条静态守卫**：`runtime` 真走链子 ✓ + 单段分支仍在 ✓ + `pipeline` 真**调用**回调 ✓ + 回调在 decode 的 try **里面** ✓ + 新模块进 `__all__` ✓）；
⚠️ 期间 `cpu_budget_test` 的静态守卫**误红** ✓✗（根因 = 环境里被灌了 `VOIDE_CPU_THREADS=8` ✗）⇒ **干净环境复跑 34/34** ✓（**是环境不是代码** ✓）；
**全量复核** ✓：`run_all` **143 套 / OK 143 · FAIL 0 / `SUMMARY:` 合计 4744/4744** ✓、末行逐字「143 项自检全部通过」✓、`EXIT=0` ✓（`tmp/run_all_round20.log` ✓；⚠️ **PowerShell `>` 落盘是 UTF-16LE** ✗ ⇒ 按 utf-8 读全成乱码 ✓✗，该按 `utf-16` 解 ✓）；
对账：上轮 **142 套 / 4699 项** ⇒ **+1 套 / +45 项 = 4744** ✓（**数字自己对上** ✓✓）；⚠️ 记账 ✓：`MEMORY.md` 只加**一条指针** + 同批下移两条等量旧内容（CLI 裸跑编码细节 ✓ / 语料细节 ✓ —— 全文都在 `TOPICS.md` ✓）⇒ 实测 **7573 → 7583 字符**（余量 427 → **417** ✓ 仍高于提示线 400 ✓）；守卫 ✓ 日志 **2/2** ✓ / 锚点 **22 处** ✓ / 落点表 **18 处** ✓ / **致命 0** ✓ / `EXIT=0` ✓。@341
⑳ ⚠️ **提交 + 第二批取证**（用户「提交之后继续看看还有哪些可以用和参考」✓；⚠️ **只取证、未抄一行** ✗、产品代码零改动 ✓）：⭐ **提交 `b0fc3ad`** ✓（「feat(engine): long-video segment chaining + CPU thread budget」✓ **19 文件 / +2075 / −23** ✓、分支 `feat/python-backend` ✓、新增 5 文件 ✓ = `core/cpu_budget.py` / `scripts/resource_watch.py` / `engine/chain.py` / `tests/cpu_budget_test.py` / `tests/engine_chain_test.py` ✓，⑯ 起未提交的批次一并提 ✓；
⚠️ 提交信息走 **`tmp/*.txt` + `git commit -F`** ✓（PowerShell 传多行会乱 ✓）、**完即删** ✓；⚠️ **`core.hooksPath` 本机未设** ✗ ⇒ 钩子**不自动跑** ✓✗ ⇒ 提交前**手动** `check_all.py` 4 道全绿 ✓）；
⚠️ 顺手更正 `MEMORY.md` 的 git 身份 ✗（写 `yuanxf`/`yuanxf@wedoctor.com` ✗ ⇒ 实测 `yuanxiufei`/`reginyuan@gmail.com` ✓）；⭐ 补 ⑰ **漏掉的 `comfy-org/`** ✓（84 仓整仓镜像 ✓）：`workflow_templates/` = **MIT** ✓ ⇒ **7 个 H3 模板** ✓（含 **`video_minimax_h3_i2v_continuation.json`** = 续拍 ✓；我们现只有 2 个 ✓）+ `site/src/lib/demos/mmh3/config.ts`（采样器/调度器枚举 ✓ + ⭐ **帧数 17k+5** ✓ 与 `geometry.H3_FRAME_GRID` **互证** ✓✓）+ `agent-prompt.md` ✓；⚠️ **`ComfyUI/` 本体 = GPL-3.0** ✗ ⇒ 其**原生** H3 实现（`comfy/ldm/minimax/*` ✓ / `comfy_extras/nodes_minimax_h3.py` ✓）**只可读事实、不可搬** ✗；
⭐⭐ `ComfyUI-H3-Multishot` **剩两件真金**（根 `h3_*.py` = MIT ✓）：`h3_retake.py` = **冻结 latent + 窗内 `noise_mask`** 局部重绘 ✓（视频/音频可分开 ✓）与「尾帧续接」「拼接」**正交** ✓；`h3_extend.py`/`h3_multishot_utils.py` 的 `context_pin` = **钉上一段 latent 尾段 + 音频参考** ✓（我们钉**解码后尾帧** ✓✗）；⚠️ 排除三条 ✗（`h3_keyframes` 绑宿主 VAE/`PackedLayout` ✓ / `h3_lora_stack` `comfy.sd` 薄包装 ✓ / `h3_gguf_arch` 改别人全局集合 ✓）；
⚠️ **许可更正** ✗：`writer_pack/**`（含 `libs/ltx_core/*` ✓、`ltx_distillation/*` ✓）**无 LICENSE + README 声明仅学术/非商用** ✗ ⇒ ⑰ 列为「条件注入/VAE/量化参照」**作废** ✗（其中唯一**纯 torch**、我们**全无**的是 `components/guiders.py` 的 **STG/APG** ✓ ⇒ **不可搬，自己按论文写** ✗）；
⚠️ **`Comic-drama` ①TTS 作废** ✗（`tts_engines.py` = **外部服务适配层** ✓：`edge_tts` 在线 / pyttsx3·SAPI / `urlopen` POST ✓，回退链只按引擎名分支 ✓，全仓 **0 处**声学模型·声码器·ONNX·torch 推理 ✗ ⇒ **自研 TTS 在参考仓无现成答案** ✓）；真价值 = **④**（角色 = 直方图×0.6 + ahash×0.4 ≥**0.6** ✓、道具 0.55/0.45 ✓、风格 0.7 ✓、镜头 = **纯规则罚分** ✓、治理 = **五维** + `report|block` + `deliverable` 门 + 台账汇总 ✓）；
⭐ `ComfyUI-MiniMax-H3-Turbo` **纯 torch 事实** ✓（Apache-2.0 ✓）：`SHIFT_V, SHIFT_A = 12.0, 3.0` ✓ / `_time_shift_sigma/_slope` 闭式 ✓ / 双流 = 同一串 sigmas 当视频时钟 + 音频侧 shift 映射 ✓ / **native 分支避免 double-shift** ✓ / `_FrugalLoRA.bypass_forward` = **in-place `add_`** 省两份临时张量（数值等价 ✓）✓；⚠️ 我们引擎**尚无 LoRA 路径** ✗；
**产出** ✓：`TOPICS.md` 三行更正 + 新增行 + 结论行修正 ✓（**产品代码零改动** ✗ ⇒ 不必跑全量 ✓）；⚠️ **待用户裁决**（红线：**抄前先问** ✓）：A 续拍/重拍 ✓ / B MIT 模板对齐 ✓ / C Turbo 双流 + LoRA ✓ / D 一致性治理 ✓。@377



**`2026-09-24.md`**（消费方核查 + 网格规则逐值钉住）
① 把昨天「两条掩码与 `segments` 不在同一坐标空间」的警告**往消费方查实** ✓：`app/` 里只有
`H3FormTrunk.forward` 用 ✓，且只取 `numel()` 当行数 ✓ ⇒ **只是文档问题不是隐藏 bug** ✓
（⭐ 教训：发现"两个坐标空间"要先找全消费方再下结论 ✓）；② ⭐⭐ 三条「混了不报错」的**网格规则**
从摘要式断言升级为**逐值** ✓（`engine_h3_form_test` 74 → **76 项**）@㉓′~㉓″：`cond` 用**目标**网格 ✓
而 `ref_img` 用**它自己的** ✓（两者**必须不相等** ✓）；两段 `ref_audio` **同名不同源** ✓（`audio` 类 ⇒
**目标**两端 ✓、`video` 类 ⇒ **它自己**两端 ✓）+ 钉住 video 类参考块「**音频行在前、视频行在后**」✓；
③ ⭐⭐ 混合形态下的**模态标签**逐段钉住 ✓（`engine_h3_form_test` 76 → **77 项**）@㉔ ——
`cond`/`cond_audio`/`ref_img`/`ref_audio` 这**四档只在混合形态出现** ✓✗，此前**只有文档没有断言** ✗
（标签只是 adaLN 的**行内偏移** ⇒ 错了**完全不报错** ✓，只是模态接到别的模态那组参数 ✓✗）；
④ ⭐⭐ 对着**事实表**逐条核覆盖 ✓：`dit.H3_PACK_FACTS` 第 3/4 条此前只有摘要式覆盖 ✗ ⇒ 补逐值判据 ✓
（`engine_h3_form_test` 77 → **79 项**）—— 跨度表 **(1,4,4,4,4) 循环** × 5/3 ✓（给错照样单调 ✓✗，
只有回绕那格会露 ✓）+ 立体声 **channel-major 的落点是 `w` 两端** ✓（两种声道顺序的行数与 t **完全一样** ✓✗，
只有 w 分得开 ✓）；⑤ ⭐⭐ 修掉**产品侧终态竞态** ✓（这才是那两次「假红」的真因 ✗✗）：`h3` 服务此前**先写 `succeeded`** ✓
再 `finally` 里 `/free` ✗ ⇒ 中间态「成功了但 `freed_vram=null`」✓✗（API 自相矛盾 ✓；真机同样中招 ✗）
⇒ 改成**先卸载、再落终态** ✓（两条执行体 ✓）+ 补 ⑭′「失败路径也 `/free`」✓（`h3_stage2_test` 20 → **21 项**）@⑥；
⚠️ **更正昨天的判断** ✗✗：昨天判成「并发资源竞争、不是代码」 ✓✗ —— 真因是竞态 ✓，并发只是让它更容易露 ✓
⇒ ⭐ **「单独复跑就好了」≠「不是代码问题」** ✗（概率低而已 ✓）：有红要**单独连跑 3 次** ✓，还绿再查时序 ✓；
⑥ 我自己的错 ✗：期望段表漏掉目标两条流 ✓✗、把 video 类参考块顺序写反 ✓✗、**第三次**在双引号串里写半角引号 ⇒
语法错 ✓；⑦ **干净全量：104 套 / 3629 项 / 0 失败** ✓（只跑一次 ✓；⚠️ `h3_stage2_test` 自带轮询 ⇒
会**慢一阵** ✓ 不是卡死 ✓，判据是「有没有 `结论：` 行」✓）；⑦ ⭐⭐ **记忆层第四次腾预算** ✓（起因：顺手
发现 `MEMORY.md` **13906 字符** ✗，而它的守卫 `app/scripts/check_memory.py` **不在 `run_all.py` 里** ✗
⇒ 一直在悄悄烂 ✓✗；跑守卫 ⇒ **7 处致命** ✓：预算超限 ✓ + **5 篇日志的末节锚点早漂** ✗（⚠️ 末节/步骤小节
锚点**会随日志增长漂移** ✗，写 `@末节` **不算登记** ✓✗）+ 一处步骤小节未登记 ✓）⇒ 细节**整体下移**
`TOPICS.md` §自 MEMORY.md 下移（2026-09-24）五节 ✓（§A 分词器/§B 跨来源校验/§C 前端类型检查/§D 上机自检/
§E 记忆守卫自身/§F 环境与自检纪律 ✓）⇒ **13906 → 7907** ✓、守卫 **exit 0 ✓ 致命 0 ✓**；⑧ 逆向
**SLM 无限漫剧创造台** ✓（`D:\app\SLM` 只是**壳** ✓，本体是 `H3EasyDirector` 工作台 ✓ **在同一模型家族
MiniMax H3** ✓ 上 ✓ —— ⭐ 最值三件：**三档引擎表** ✓（官方 25 步/标准 8 步/极速 4 步 + 每档必备模型与
显存建议 ✓）/ **已知冲突台账** ✓（DaSila ⇒ WinError 433 ✓ 等 7 条 ✓）/ **契约同步+离线兜底** ✓
（权威在工作台 `/api/workflow_tiers` ✓）⇒ 细节见 `TOPICS.md` §SLM 逆向笔记 ✓；⚠️ 它的提示词/工作流在
74.7 MB 的 `v3.5.40` 包里 ✗ 未拉 ✓）@108

⭐ **⑨~⑳ 两轮「接线」**（用户「没有实现的继续实现」✓）：**判据层立了却只被自检调用** ✗✗ ⇒ 一轮补
**提交前那几道关**（配音契约 / H3 形态决策 / prompt 契约 + 加速链 / 分段与超清计划端点 ✓ ③④⑤⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱），
二轮补**加载层与张量层**（混合加载**加载层** ✓ / 超清**二采阶段** ✓ / 段级音频合成接进单镜合成 ✓ ⑲⑳）
⇒ 落点表见 `TOPICS.md` §接线进度 ✓、判据见 `MEMORY.md` §接线 ✓；⚠️ 早先「仍未接 `cache_key`」已过时 ✗ —— 二轮已接上 ✓
（`find_reusable_video` ✓ 先补「标量按值进指纹」✓✗）。⭐ **㉑ 第三轮收尾**（2026-09-25 ✓，用户「请继续」✓）：「能力已有、没接出去」最后两处 ✓
① 多集节奏相位注入 ✓（`runtime.py` 早先「`rhythm-phase.ts` 未迁」warn 占位 ✗，`rhythm_phase.py` 早已迁好 ⇒ `storyboard_breaker`
少一段跨集节奏引导 ✓✗ ⇒ 真调 ✓）；② `refineNote` **自述与实现相反** ✓✗（`denoise:False` ✗ 而 `_second_pass` 已实现 ✓ ⇒ 对齐 ✓）。
全量 **125 套 / 4099 项 / 0 失败** ✓（+3 项 ✓；顺带校正 `engine_refine_test` 记 14 条、实测 23 条 ✓）。
⭐ **㉒ 逆向收口 + 自述对齐**（2026-09-25 ✓，用户「有用的就逆向，没用的就不做」+「继续完善」✓）：① 逆向待做项判定收口 ✓ ——
TTS 三件套**有用部分早已落地**（`voice_contract.EMOTION_ORDER` 8 维 = 照抄 IndexTTS ✓，另两条情绪入口 emo_audio/文本情绪是
IndexTTS 专属、本仓 CosyVoice 用不上 ✗）；工作台 HTTP 面**核心已落地**（三档表→`tiers.py` ✓）后端 PyInstaller 编译读不到 ✗ ⇒ 不做 ✓
（详见 `TOPICS.md` §逆向待做项收口 ✓）；顺手纠正「IndexTTS 情绪 9 维」✗ → 实为 **8 维** ✓。② 清掉 3 处**自述与实现相反**的 stale ✓
（`refineNote.denoise` ✓ / `torch_backend` 模块头「架构装载还没写」✗ / `engine/__init__`「本机无 torch」✗ —— 实已装 CPU 版 torch ✓）。
⚠️ 功能层面**已无「不需要真权重」的可做** ✓，余下全卡真权重/真机 ✓。
⭐ **㉓ 模型目录自主化**（2026-09-25 ✓，用户「完全自主不依赖第三方」+「也要能检测电脑内模型」✓）：① **下载/存储自主** ✓ ——
models_dir 默认从 ComfyUI 目录改为 **`<data_root>/models`** ✓（`local_model_scan.default_models_dir()` ✓ + `get_model_paths` 与
`model_manager.resolve_paths` 默认回落 ✓；`configs/model-paths.json` 清空 ComfyUI 硬编码 ✓）；② ⚠️ **「检测/扫描」不排斥第三方** ✗ ——
`get_default_roots` 仍覆盖电脑内模型（本仓目录 + 本地服务 + ComfyUI 目录探测 + extra_roots ✓）：自主的是「从哪下载、存到哪」，
**不是**「不许看见别人已装的模型」✓（先误删了 ComfyUI 扫描、用户一句话纠正 ✓）；③ 全量 **125 套 / 4099 项 / 0 失败** ✓（项数不变 ✓
只改 1 条「清空 models_dir 回落到默认」的断言 ✓）。
⭐ **㉔ 自研 LLM 运行时起步**（2026-09-25 ✓，用户「要自研实现」✓）：文本生成此前走 ollama（HTTP 11434）✗ ⇒ 新建
`engine/llm.py` ✓ —— 自研 decoder-only transformer（RMSNorm / GQA 分组查询 / RoPE / SwiGLU / 因果掩码 / KV cache /
采样 greedy+top-p+top-k ✓）；架构参数**全显式** ✓（`LlmConfig` 不写死 Qwen3 ✗ 本仓纪律 ✓）；自检 `engine_llm_test` **13 条** ✓
（⭐⭐ KV cache 不变量：整段 vs 逐 token 逐位相同 ✓ / 因果掩码「改后不动前」✓ / RoPE 保范数 ✓ / GQA ✓ / tie embeddings ✓）。
⚠️ **尚未接线** ✗：GGUF 权重加载 + k-quant 反量化**未做** ✗（quant.py 现只 cover fp8/int8 ✗）⇒ 下一轮 ✓；
全量 **126 套 / 4112 项 / 0 失败** ✓（+1 套 / +13 项 ✓）。
⭐ **㉕ GGUF 反量化**（2026-09-25 ✓，用户「继续」✓）：新建 `engine/gguf_dequant.py` ✓ —— 把 GGUF 数据区的量化字节 ⇒
fp32 张量 ✓（F32/F16/BF16/Q8_0/Q4_K ✓；公式照 llama.cpp `ggml-quants.c` ✓ **MIT** ✓ web_fetch 取权威 ✓）；⭐⭐ 判据 = 手造
Q4_K block（d=1/dmin=0.5/scales 全 0xFF/qs 全 0x11 ⇒ 反量化**每元素精确 = 31.5** ✓✗ —— 字节布局或 6-bit 解包错一处就对不上 ✓）；
⚠️ 其余 k-quant（Q2_K/Q3_K/Q5_K/Q6_K）**具名拒绝** ✗；`load_weights` 能整文件提权重 ✓ 但大文件（14B≈17GiB fp32）要**逐张量**读 ✓。
⚠️ **下一步** ✗：GGUF 张量名 → `llm.LlmModel` 参数的**映射**（`model.layers.N.self_attn.q_proj.weight` → `blocks.N.attn.q_proj.weight` ✓）未做 ✗；
全量 **127 套 / 4121 项 / 0 失败** ✓（+1 套 / +9 项 ✓）。
⭐ **㉖ GGUF → LlmModel 装载**（2026-09-25 ✓，用户「继续」✓）：新建 `engine/gguf_to_llm.py` ✓ —— ggml 命名
（`blk.N.attn_q.weight` ✓）→ LlmModel 命名（`blocks.N.attn.q_proj.weight` ✓）+ 6 类线性层**转置** ✓（GGUF (in,out)→PyTorch (out,in) ✓）
+ tie embeddings 时 `output.weight` 跳过 ✓；⭐⭐ 判据 = **往返恒等**（LlmModel state_dict → 反向转 GGUF 命名 → 正向映射回 ⇒ 逐张量相等 ✓✗）；
`load_llm_from_gguf` 逐张量读 + `load_state_dict(strict=True)` ✓（漏/形状错当场报 ✓）。⚠️ **下一步** ✗：自研文本后端
（describe+generate）替换 `OllamaTextAdapter` ✗（这才是「不调 ollama」的最后一跳 ✓）；全量 **128 套 / 4128 项 / 0 失败** ✓（+1 套 / +7 项 ✓）。
⭐ **㉗ 自研文本后端**（2026-09-25 ✓，用户「继续」✓）：新建 `engine/llm_backend.py` ✓ —— 把「架构+反量化+装载」收成**一个后端**
（像 `TorchBackend` ✓）：`describe` 如实报缺（权重/词表/模型缺啥报啥 ✓ 不冒充 ✗）/ `load` 幂等装载 / `generate` 走 encode→生成→decode ✓
（⚠️ **同步** ✗ CPU 推理会阻塞 ✗ 接线要 `asyncio.to_thread` ✓；⚠️ chat 模板未做 ✗ 裸拼接 ✓）。⭐⭐ 判据 = 未装载就 generate ⇒ 明确拒 ✗
（不静默返回空串 ✗）。⚠️ **下一步** ✗：`text_generation` 的 engine 分支 + `infer_llm_config`（从 GGUF 元数据读架构参数 ✓）⇒ 这才是「接线」闭环 ✗；
全量 **129 套 / 4134 项 / 0 失败** ✓（+1 套 / +6 项 ✓）。
⭐ **㉘ 接线闭环**（2026-09-25 ✓，用户「继续」✓）：① `gguf_to_llm.infer_llm_config` ✓ —— 从 GGUF 元数据读架构参数
（`{arch}.embedding_length/block_count/attention.head_count[_kv]/feed_forward_length/rope.freq_base/...` ✓，Qwen3 GGUF 沿用 `llama.` 前缀 ⇒ 回退读 ✓）；
缺必需字段 ⇒ **具名拒绝** ✗（不猜默认 ✗ —— 猜了装出「名字对、形状全错」的模型 ✗）；head_dim 缺省 = hidden/heads 推导 ✓。② **接线** ✓ ——
`text_generation.generate_text` 加 engine 分支 ✓：provider=engine ⇒ 走自研后端（`asyncio.to_thread` 包同步推理 ✓）而非 HTTP 调 ollama ✗；
后端缓存（大权重只 load 一次 ✓）。⭐⭐ **文本不依赖 ollama 已闭环** ✓（剩「下载真权重 + chat 模板」这两件数据/细节 ✓）；
全量 **129 套 / 4141 项 / 0 失败** ✓（+7 项 ✓：infer 5 + 接线 2 ✓）。
⭐ **㉙ 对话骨架（chat 模板）**（2026-09-25 ✓，用户「继续实现功能」✓）：㉘ 自述「剩『下载真权重 + chat 模板』」⇒ 补 chat 模板 ✓ ——
新建 `engine/chat_template.py` ✓（**骨架从权重元数据取** ✓ 不内置 ✗✗；`StrictUndefined` ✓✗ 缺变量**报错**而非静默空串 ✓；
注入 `raise_exception` ✓）；接线 `llm_backend` ✓（load 读骨架 / generate 走骨架 / describe 如实报 ✓，**无骨架时老行为一字未动** ✓）；
⭐⭐ **「没有模板」与「模板坏了」是两回事** ✗✗（坏了**当场拒** ✗ 不静默回落裸拼接 ✗）；自检 `engine_chat_template_test` **18 项** ✓
（+1 套 / +18 项 ✓；既有 `engine_llm_backend_test` 8/8 ✓ 未破）；⚠️ 如实报「未核」✗：用骨架时 `add_special_tokens`
仍 `True` ✓（待真词表定 ✓）。⭐ 同轮**核查用户点的方向** ✓ ⇒ 配音契约**早已闭环** ✓（那条「下一步」过时 ✗）、
音色克隆**样本面无「静默错误」级真缺口** ✓（详见日志 ⑨ ✓）；⚠️ 清理 `.codebuddy` 踩坑 ✗✗：连带删了 `memory/` 10 篇日志
⇒ `git restore` 完整恢复 ✓ 守卫复绿 ✓ 致命 0 ✓（教训：删除逐条 `-LiteralPath` ✓）。@486
⭐ **㉚ 裸跑编码收口（GBK 控制台）**（2026-09-25 ✓，用户「全收口」✓）：起因 = 照 `PENDING_PARTS` 的「上机前先跑一次」跑
`h3_readiness.py`（本机真权重 ✓）⇒ **裸跑第一条 `print` 就崩** ✗（满屏 ✓/✗，而 py3.14 的 stdout 仍是 **locale/GBK** ✗ —— PEP 686 要 3.15 ✓）；
⭐⭐ **四层掩盖**：① 脚本崩 ⇒ ② `engine_readiness_script_test` ⑩ FAIL（stdout 非 JSON）⇒ ③ **该测试自己也崩在打印失败原因上** ✗✗
（`⇒`=U+21D2 GBK 编不了 ⇒ 只剩 `EXIT=1`、原因一个字看不到 ✓✗）⇒ ④ **全量 133 套全绿** ✗（`run_all.py` 给子进程灌了 `PYTHONIOENCODING=utf-8` ✓）
⇒ **文档推荐的「单跑」用法在本机是坏的** ✗✗。口径（三缺一不报 ✓）：**含 locale 外字符 + 真有 `print` + 入口无 `reconfigure`** ⇒ 判会崩 ✗；
收口 ✓：新守卫 `check_cli_encoding.py` ✓（扫 152 文件；探针命中 **65** ✗ ⇒ 63 个 AST 批量补 ✓ + 2 个模块级脚本手改 ✓：`smoke_test.py` ✓ /
`corpus/analyze3.py` ✓ 老写法 `sys.stdout = io.TextIOWrapper(...)` **丢原 wrapper** ✗ ⇒ 统一 `reconfigure` ✓）+ 接进 `check_all.py` 第 3 道 ✓ +
`test_guards.py` 负向 3 例 ✓（17/17 ✓）；**硬证据**：四道自检全绿 ✓ + 抽样**裸跑** 4 套 `exit=0` ✓（20/20 ✓、18/18 ✓、29/29 ✓、45/45 ✓）。@517

> ⚠️ **历史日志已于 2026-09-26 清理 —— 只留最近一天**（`2026-09-24.md` 含 09-25 的工作 ✓ + 当日 `2026-09-26.md` ✓）。此前 10 篇（09-13 / 09-14 / 09-15 / 09-16 / 09-17 / 09-18 / 09-20 / 09-21 / 09-22 / 09-23）已从工作区删除，**原文在清理前那次提交 `83743d6` 里** ✓：`git show 83743d6:.codebuddy/memory/2026-09-20.md`，或 `git checkout 83743d6 -- .codebuddy/memory/2026-09-20.md` 取回（把日期换成要的那篇）。删前核对过：它们最大步号 135（09-20 那篇；内含日期只到 09-20 ✓），比 09-21~09-24 早 ✓ ⇒ **没删掉更晚的工作** ✓。本节下面的「已出栈的落点」与 `MEMORY.md` / `TOPICS.md` 不受影响。

> 动**视频首尾帧 / 尾帧 / 角色合并**时**先读本文件**。





## 已出栈的落点（优先看这些，别翻日志）

| 内容 | 权威落点 |
|---|---|
| 全部 API 契约 | `docs/api-contract.md` |
| H3 本地视频链路 | `docs/local-h3-video-system.md` |
| 本地模型评估/下载 | `docs/local-model-evaluation.md`、`TOPICS.md` |
| 语料源与合规 | `docs/video-prompt-data-sources.md`、`docs/seedance2-corpus-analysis.md` |
| preset skill 模板 | `docs/preset-skill-template.md` |
| **Skill 体系（改前必读）** | `backend-py/app/skills/README.md` + `MEMORY.md §Skill` + `TOPICS.md §Skill 体系细节` |
| **Python 后端（绞杀者迁移）** | `backend-py/README.md`（运行/环境变量/迁移 SOP）+ `TOPICS.md §backend-py` + `backend-py/tests/smoke_test.py` |
| 引用完整性守卫（skills 路径） | `backend-py/app/scripts/check_skill_refs.py`（基线见脚本头） |
| 记忆层自检（8k 预算 + 锚点/登记/落点路径）+ 守卫自检 | `backend-py/app/scripts/check_memory.py`、`backend-py/app/scripts/test_guards.py`（规则见脚本头） |
| CLI 裸跑编码（GBK 控制台别崩 ✗；口径 + 四层掩盖链） | `backend-py/app/scripts/check_cli_encoding.py`（规则见脚本头）+ `TOPICS.md §CLI 裸跑编码` |

## 日志写入规范（防再膨胀 —— `09-12` 单日已 780 行）

- 每轮 **≤ 8 行**：只写 **结论 / 落点(文件:行) / 红线 / 未决**。
- **不写**过程叙述与命令回放；需长期留证的内容 → `docs/` 或 `TOPICS.md`。
- 结论属「日后会导致 bug 的不变量」→ 同步进 `MEMORY.md`；属低频长专题 → `TOPICS.md`。
- 写时假定「将来只会按 `@行号` 跳读」，**不要依赖前后文**。
