# OCR 并行（GPU）吞吐不升反降：瓶颈定位方案

对象：`backend-py/app/scripts/corpus/ocr_index.py`（多进程 + 进程内页线程 + CUDA EP）。

## 0. 先决条件

停掉在跑的批次。剩余显存 1240 MiB 时任何 A/B 都不成立（分配器抖动/随时 OOM）。
每条配置 3 次取中位，第 1 次当 warm-up 丢弃（含建会话与 CUDA 预热）。

## 1. 现状证据（分级）

| 结论 | 证据 | 强度 |
| --- | --- | --- |
| GPU 已接近饱和：util 92–100%，显存 21498/23028 MiB | `nvidia-smi` 连续采样 | 实测 |
| 单进程 2 页线程有效：40 页 0.184 s/页 vs 老口径 0.504 ⇒ 2.74× | 受控 A/B | 实测 |
| 整批实测：5168 页 / 599.4 s = **8.62 页/秒**（0.116 s/页），4 进程 × 2 页线程 | 批次收尾行 | 实测 |
| 但每进程页耗时 0.464 s/页 vs 单进程 0.184 ⇒ 4 倍进程仅换 1.59× 吞吐，并行效率 0.2 | 收尾行 + 单进程 A/B | 实测 |
| 批内采样：CPU 平均 32%（峰 38%）、GPU 平均 87%（峰 100%）、显存最低空闲 1.1 GiB；程序自判「GPU 已吃满、瓶颈在算力」 | `_ocr_reburn.log:27-28` | 实测 |
| 主因：GPU 已饱和，多出来的 in-flight 只排队不增吞吐（H1） | 上两条合看 | 已确认 |

⚠️ 上一批 20 份混了图册（块数远小于页数、可能触发 VLM 升级）与文本页，逐份 s/页 不可横向比；只有聚合吞吐与采样值可比。故仍须固定语料再谈拐点。

### 0.1 依赖核查（回答「是不是装坏了」）

| 事实 | 证据 | 判定 |
| --- | --- | --- |
| CUDA EP 真的加载且排第一 | 抬头 `引擎实报：CUDAExecutionProvider,CPUExecutionProvider` | 不是静默回落 CPU |
| 可用 provider 含 TensorRT/CUDA | `ort 1.31.1`，`get_available_providers()` | 装的是 GPU 版 |
| **`onnxruntime` 与 `onnxruntime-gpu` 同版本并存（各 1.31.1）** | `pip list` | **已收敛为单一依赖**：两者往同一个 `onnxruntime/` 目录叠文件（RECORD 624 vs 626，差的就是 cuda/tensorrt 两个 provider dll），已摘掉 CPU 版元数据、保留包文件。见 0.4 |
| `opencv-python` 与 `opencv-python-headless` 并存（5.0.0.93） | `pip list` | 同类隐患，非本问题成因 |
| `rapidocr 3.10.0` + `rapidocr-onnxruntime 1.4.4` 并存 | `pip list` | 有意为之（两档引擎），正常 |

### 0.2 受控实测（同一批 4 份 / 30 页 ×4 = 120 页，150 dpi，独立 sidecar 副本）

| 配置 | 吞吐 | s/页 | GPU 平均/峰值 | CPU 平均/峰值 | 显存最低空闲 |
| --- | --- | --- | --- | --- | --- |
| E2 `cpu` 4 进程 ×（6 识别 + 1 渲染） | 2.05 页/秒 | 0.49 | 7% / 12% | **81% / 89%** | 20.9 GiB |
| E3 `cuda` 1 进程 × 1 页线程 | **3.21 页/秒** | 0.31 | 35% / 46% | 15% / 17% | 17.6 GiB |
| E7 `cuda` 4 进程 × 2 页线程 | **9.68 页/秒** | 0.10 | 90% / 96% | 37% / 39% | 7.8 GiB |
| E8 = E7 + `rec_batch=24` | 9.60 页/秒 | 0.10 | 78% / 94% | 36% / 40% | 6.8 GiB |

块数逐份对拍：E3 与 E7 **完全一致**（17 / 27+剔除2 / 7 / 21）⇒ 并行没有拿质量换速度。
E7 每进程页耗时 0.35–0.38 s/页（E3 为 0.31）⇒ 争用轻微，不是「4 倍进程只换 1.59×」。

**结论修正（原假设 H1 部分成立、部分被否）**

1. 并行是有效的：8 in-flight 相对单流 **3.02×**，瓶颈已从「延迟」转成「GPU 算力」（90%）。
2. 单流只有 35% ⇒ 真正的浪费在**流水线太浅**：一页一个请求，det/rec 之间是主机串行段。
3. `rec_batch` 16→24 无收益（9.60 vs 9.68，噪声内）⇒ 单页内批已饱，要再压 kernel 数只能**跨页合批**。
4. 显存不是瓶颈，但 `GPU_SESSION_MIB=4200` **是准的，不要下调**：E7「最低空闲 7.8 GiB」是**剩余**不是**占用**，反推 4 会话共占 ~14.8 GiB ≈ 3.7 GiB/会话，与 4200 同一量级；且 `ocr_index.py:1461` 记录了第 5 会话顶到 22489 MiB、每页 0.44→0.88 s 的实测边界。该常量参与 `gpu_session_cap()` 决策（`workers()` 里按实时空闲显存算会话数），下调会在卡被部分占用时误开过多会话。
5. **「CPU 利用率从 81% 掉到 37%」是收益的形态，不是故障**：同一语料 E2（纯 CPU）2.05 页/秒，E7（GPU 并行）9.68 页/秒 = **4.72×**。
   CPU 档把 CPU 当算力用（每进程 6 条 ORT 线程，故 81%）；GPU 档算子全在卡上，`worker_threads()` 只给 2 条（故 37%），省下的 CPU 已经转化成吞吐。
6. 因此不要再「把 CPU 填满」：E7 时 GPU 已 90%，CPU 侧加线程只会和渲染抢 GIL。若确实想让 CPU 也忙起来，唯一有意义的做法是把渲染/切块/落库真正并行化（当前 5 条渲染线程已不是瓶颈）。

### 0.3 拐点曲线与合批可行性

| in-flight | 配置 | 吞吐 | 每档增幅 | 相对单流 | 并行效率 | GPU 平均/峰值 | CPU 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 进程 × 1 页线程 (E3) | 3.21 页/秒 | — | 1.00× | 1.00 | 35% / 46% | 15% |
| 2 | 1 进程 × 2 页线程 (E5b) | 4.84 页/秒 | 1.51× | 1.51× | 0.75 | 40% / 46% | 20% |
| 4 | 2 进程 × 2 页线程 (E5) | 7.45 页/秒 | 1.54× | 2.32× | 0.58 | 59% / 70% | 24% |
| 8 | 4 进程 × 2 页线程 (E7) | 9.68 页/秒 | 1.30× | 3.02× | 0.38 | 90% / 96% | 37% |

- 每档增幅 1.51→1.54→1.30，**到 8 才开始衰减**；并行效率 0.75→0.58→0.38 一路掉，但绝对吞吐仍是单流的 3 倍 ⇒ 拐点卡在「GPU 90%」，不是「进程太多」。
- 决定性证据：吞吐几乎与 GPU 占用成正比 —— 每 1% GPU 对应 0.121 / 0.126 / 0.108 页/秒（in-flight 2/4/8）。**没有可观测的并行开销**，唯一限制就是卡被吃满。故 8 in-flight 是保留配置，不必再往上加。
- E5c（6 in-flight）按此线性关系推算约 8.8 页/秒、GPU ~78%，落在 4 与 8 之间，不改变结论，未单独测。

**跨页合批值不值得做：不值得。** E7 时 GPU 已 90%，把主机串行段全部抹平最多把利用率推到 100%，吞吐上限 = 9.68 × 100/90 = **10.76 页/秒，仅 +11%**；而实现上必须绕开 RapidOCR 高层 API（`_ModernEngine.__call__` 是单图调用，`ocr_index.py:1762`），自己拆 det/rec 并管理跨页坐标回填，改动量与 11% 上限不成比例。
另注：上一批全量实测 8.62 页/秒，比受控 9.68 低 11%，正好等于「非 OCR 开销 + 被其他进程抢卡」的量级 ⇒ **大批并没有变慢**。

### 0.4 依赖收敛：一个 `onnxruntime-gpu` 同时服务 CPU / GPU 两档

结论：**可以共用一份依赖，且不影响效率。** `onnxruntime-gpu` 是超集，`get_available_providers()` 实测返回
`['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`，CPU 档走的正是其中的 CPU EP。
本机已有直接证据：E2 纯 CPU 基线（2.05 页/秒、CPU 81%）就是在 GPU 版生效的状态下跑出来的。

做了什么：摘掉 `site-packages/onnxruntime-1.31.1.dist-info`（已备份至 `_bench/onnx_backup/`），**包文件一个不删**。
为什么不用 `pip uninstall onnxruntime`：它的 RECORD 覆盖了 GPU 版除 2 个 provider dll 之外的全部文件，单独卸会把包目录删空，
而 `pip cache list` 显示无本地缓存，恢复只能联网下约 250 MB。
为什么不用「两个都卸再装回」：`pymupdf` 与 `rapidocr-onnxruntime` 都硬依赖名为 `onnxruntime` 的发行版，
装回去之后任何一次 `pip install` 仍会重新拉 CPU 轮子 —— 卸载换不来持久性，只多一次不可逆风险。

验证（两档端到端各跑一份 2 页）：

| 档 | 抬头 | 耗时 |
| --- | --- | --- |
| `OCR_DEVICE=cuda` | `引擎实报：CUDAExecutionProvider,CPUExecutionProvider` | 0.2 s/页 |
| `OCR_DEVICE=cpu` | `引擎实报：设备=cpu（未探测 CUDA provider）` | 0.6 s/页（2 页小样，含建会话） |

已知代价：`pip check` 现在会报两条
「`pymupdf` / `rapidocr-onnxruntime` requires onnxruntime, which is not installed」——这是元数据口径不一致，运行期无影响。

**复发时的修复命令（不要卸载，只覆盖文件）**：

```
pip install --force-reinstall --no-deps onnxruntime-gpu==1.31.1
```

复发判据：抬头不再是 `CUDAExecutionProvider,...`，或批处理返回码 3（`ocr_index.py:1962` 的 fail-fast 会当场中止，不会静默跑 CPU）。

**已知遗留（不改，仅记录）**

- `opencv-python` 与 `opencv-python-headless` 同为 5.0.0.93、同提供 `cv2/`：与 onnxruntime 同类问题，同样不能靠卸载收敛（会删空 `cv2/` 目录）。若哪天真要收敛，用 `pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93`。
- `pip check` 会常驻两条 `requires onnxruntime, which is not installed`（`pymupdf` / `rapidocr-onnxruntime`）。运行期无影响。

### 0.5 强杀批处理会留孤儿 worker（实测泄漏 8.7 GiB 显存）

`ocr_index` 的 worker 由 `multiprocessing.spawn` 起，子进程命令行是 `-c "from multiprocessing.spawn import spawn_main..."`，**不含 `ocr_index`**。父进程被 `Stop-Process` 硬杀后，子进程不会退出，继续持有 CUDA 会话。

实测：2026-11-04 17:28 硬杀一次运行后，3 个孤儿 worker 仍占 **8.7 GiB**（显卡空闲从 1438 MiB 掉到 10172 MiB 被占用），足以让后续 `gpu_session_cap()` 把并发从 4 压到 2，或触发批中「有别的任务在抢卡」告警 —— 这正是当时「显存最低空闲仅 1.1 GiB」的来源之一，而不是外部程序。

清理必须按 spawn 关键字匹配，不能按脚本名：

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*multiprocessing.spawn*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

注意别误杀服务端：本机常驻 `python -m uvicorn app.main:app --port 5790` 也是 `python.exe`，其命令行不含 `multiprocessing.spawn`，用上面的过滤条件不会命中。正常退出（跑完整批）不会留孤儿，只有硬杀才会。

## 2. 刻度尺（先钉死，再谈对比）

| 维度 | 值 | 位置 |
| --- | --- | --- |
| 页图 | `DEFAULT_DPI=150` ⇒ ≈944×1335×3 ≈ 3.8 MB/页 | `ocr_index.py:207` |
| 切块 | `CHUNK_CHARS=800`、`MIN_PAGE_CHARS=12` | `:205,213` |
| 批大小 | `Rec.rec_batch_num=_rec_batch()`（`OCR_REC_BATCH`），det 无批 | `_new_engine()` |
| 模型 | PP-OCRv6 small（det+rec+cls），ONNX CUDA EP | `_ModernEngine` |
| 并行 | `GPU_MAX_WORKERS=4`、`GPU_PAGE_THREADS=2`、`GPU_ORT_THREADS=2`(intra)、inter=1 | 常量段 |
| 预算 | `GPU_SESSION_MIB=4200`/会话、`GPU_HEADROOM_MIB=1024`、`MAX_RENDER_THREADS=6`、`GPU_THREADS_PER_WORKER=7` | 常量段 |

三个派生量必须进每次实验记录：
`T_ocr = procs × page_threads × intra_op`（OCR 线程上限）；`B_page ≈ W×H×3`；`S = procs`（页线程共享同一 session，**不额外吃显存**）。

## 3. 实验矩阵

语料：3 份文本 + 1 份图册，共约 120 页，用 `--limit`/`--pages` 截齐。

| ID | 设备 | workers | page_threads | intra | in-flight | 目的 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | cpu | 1 | 1 | 16 | 1 | CPU 串行基线 |
| E2 | cpu | 4 | 1 | 4 | 4 | CPU 并行基线（应得的收益） |
| E3 | cuda | 1 | 1 | 2 | 1 | 单流延迟 L1 与上限吞吐 |
| E4 | cuda | 1 | 2 | 2 | 2 | 拐点 |
| E5 | cuda | 2 | 2 | 2 | 4 | 拐点 |
| E6 | cuda | 4 | 1 | 2 | 4 | 旧口径 |
| E7 | cuda | 4 | 2 | 2 | 8 | 新口径（回归嫌疑） |
| E8 | cuda | 4 | 2 | 2 | 8 | E7 + `OCR_REC_BATCH=24` |

命令模板（变量名先跑 `Select-String -Pattern "OCR_DEVICE|OCR_PAGE_THREADS|OCR_REC_BATCH"` 核对）：

```powershell
cd d:/code/voide/darme-voide/backend-py
$env:OCR_DEVICE="cuda"; $env:OCR_WORKERS="4"; $env:OCR_PAGE_THREADS="2"; $env:OCR_REC_BATCH="16"
.venv/Scripts/python.exe -X utf8 -m app.scripts.corpus.ocr_index --limit 4 --pages 30 --dpi 150 --workers 4
```

每次实验前把 sidecar 库复制到临时目录：既不污染真库，才能逐份对拍块数。

## 4. 采集

```powershell
nvidia-smi dmon -s um -d 1 -o DT > _bench\E7_gpu.csv
typeperf "\Process(python*)\% Processor Time" "\Process(python*)\Working Set - Private" "\Process(python*)\IO Read Bytes/sec" "\System\Processor Queue Length" -sc 120 -f CSV -o _bench\E7_host.csv
```

- GPU：看序列是否锯齿。锯齿＝内核间空洞（主机同步）；贴顶＝真饱和。同时看 `clk` 是否掉 base（撞功耗墙会伪装成「并行无效」）。
- CPU：`% Processor Time` ÷ (100×核数)，上限就是 `T_ocr`。≈上限一半 ⇒ 线程在 `cudaStreamSynchronize` 等，不是算不动。
- `Queue Length` 持续 >2×核数 ⇒ 线程超卖（`THREADS_PER_WORKER` 与 `page_threads` 叠加）。
- 渲染与 OCR 同进程：要分离就同页数下 `OCR_DEVICE=cpu` vs `cuda` 相减，渲染成本相同。

二阶（E3/E7 差异仍无解释时才开）：`nsys profile -t cuda --stats=true`，只读三个数——memcpy+sync 占比、单内核均长（det/rec <2 ms 且内核数/页数十 ⇒ launch 主导，加并发无用）、GPU 空闲间隙占比。

## 5. 判读矩阵

| 观测 | 结论 | 该动的旋钮 |
| --- | --- | --- |
| util ≥90% 且吞吐随 in-flight 平坦/下降 | H1 GPU 饱和 | 停加并发，改加批或换档 |
| util <40%、吞吐不涨、CPU 也低 | H2 主机侧串行段 | 整段落库，减少 Run 次数 |
| 每进程 CPU% << `intra/核数` 且 Queue 高 | H3 线程超卖 | 两个放大旋钮只留一个 |
| free <1.5 GiB 且 dmon 锯齿回收 | H4 显存抖动 | `GPU_MAX_WORKERS→2` 或 `gpu_mem_limit` |
| memcpy+sync >10% 总时长 | H5 传输/同步 | 合并每页 Run，别加进程 |
| HtoD ≈ 页数×3.8 MB、占总带宽 <1% | 传输不是瓶颈（排除项） | 不优化拷贝 |
| 块/秒 随并发下降 | 并行掉了正确性 | 判失败，先修再谈吞吐 |

统一口径：`S(N)=页/秒(N)/页/秒(E3)`；`E(N)=S(N)/N`。
`E(4)<0.5` 且 util>90% ⇒ 饱和到顶；`E(4)<0.5` 且 util<60% ⇒ 主机侧或有锁；`E(8)<E(4)` ⇒ 超拐点，回退。

## 6. 建议与验收阈值

1. 先定拐点 N\*（E3→E4→E5→E7），把 `workers × page_threads` 钉在 N\*。验收：`S(N*)≥1.5×` 且 `E(N*)≥0.55`。
2. 加批优先于加并发：`rec_batch_num` 16→24（老档 6→16 实测 3.8×、字数不变）。验收：E8 比 E7 提升 ≥1.25×，`块/秒` 不低于 E7 的 95%。
3. 会话数收敛到显存预算：`GPU_SESSION_MIB` 是每会话值，4 会话+arena=21.5 GiB 是实测。验收：全程 free ≥2 GiB，无台阶式下跌。
4. 线程预算单一化：`T_ocr` 不得超过物理核的一半，渲染线程单独给。验收：`页/秒` 提升且 `Queue Length` 回落到核数以内。
5. 若 E8 仍不足：把逐页 OCR 改成「收集 N 页的 det 结果 → 一次 rec 批」，即真正压住 kernel 数。验收：nsys 中每页内核数下降 ≥3×。

结论成立的最低门槛：同一语料、同页数、同 DPI、GPU 独占、warm-up 已丢弃、`块/秒` 不倒退、3 次中位数的极差 <10%。
