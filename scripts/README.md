# scripts/ — AI/GPU 工具链

本目录是 Drama Studio 的「AI/GPU 计算层」脚本，与 TS 后端（`backend/`）通过 subprocess 解耦。
全部 Python 脚本仅依赖 **Python 3.8+ 标准库**（零第三方 pip 依赖）。

## 工具清单

| 文件 | 语言 | 职责 |
| --- | --- | --- |
| `model_manager.py` | Python | 通用本地模型安装/管理（多类模型 + comfyui/ollama/git/manual 四种安装方式） |
| `sd_h3_pipeline.py` | Python | sd.cpp × MiniMax-H3 Turbo GGUF 一键编排（doctor → download → build → probe） |
| `sd_h3_compat_probe.py` | Python | GGUF 与 sd.cpp 兼容性判定（零依赖读元数据，可选 `--live` 真机 load） |
| `h3_install.py` | Python | 旧 H3 安装 CLI 兼容 shim（数据已并入 `model_manager.py`） |
| `migrate_models.ps1` | PowerShell | 一次性：模型硬链接迁移到 ComfyUI Desktop 共享库 |

## 用法

### model_manager.py（核心入口）
```powershell
python scripts/model_manager.py list [--category X] [--runtime Y] [--missing]
python scripts/model_manager.py download [--key ...] [--category ...] [--required] [--all] [--force]
python scripts/model_manager.py remove --key ...
python scripts/model_manager.py doctor
python scripts/model_manager.py install-nodes [--only ...]
python scripts/model_manager.py add-model --key ... --name ... --category ... --runtime ... [...]
python scripts/model_manager.py remove-model --key ...
```
数据源：`configs/models.json`（模型清单）+ `configs/model-paths.json`（路径配置）。
路径优先级：命令行参数 > 环境变量 > `model-paths.json` > 默认探测。

### sd_h3_pipeline.py（路线 A 编排）
```powershell
python scripts/sd_h3_pipeline.py                 # 顺序执行 doctor→download→build→probe
python scripts/sd_h3_pipeline.py doctor          # 单步：环境体检
python scripts/sd_h3_pipeline.py download        # 单步：下载 Q4_0 Turbo GGUF（断点续传）
python scripts/sd_h3_pipeline.py build           # 单步：编译 sd.cpp（cmake -DSD_CUDA=ON）
python scripts/sd_h3_pipeline.py probe [--live]  # 单步：兼容性判定
```
编译依赖：Git、CMake 3.x+、Visual Studio 2019/2022（C++ 桌面开发）、CUDA Toolkit（nvcc）。
> 注：本机当前无 CUDA Toolkit（nvcc），`build` 只能产出 CPU 版，详见 `docs/local-h3-video-system.md`。

### sd_h3_compat_probe.py（兼容性判定）
```powershell
python scripts/sd_h3_compat_probe.py --gguf <path.gguf>          # 静态判定
python scripts/sd_h3_compat_probe.py --gguf <path.gguf> --live   # 真机 load
python scripts/sd_h3_compat_probe.py --list                      # 支持矩阵
```

### h3_install.py（兼容 shim）
保留旧 CLI（`doctor` / `download` / `download-model` / `install-nodes`）与库接口，
数据一律从 `models.json` 读取。新能力请改用 `model_manager.py`。

### migrate_models.ps1（一次性迁移）
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/migrate_models.ps1
```

## 依赖关系
```
sd_h3_pipeline.py ──┐
h3_install.py      ──┼──> model_manager.py（复用 download_url / check_bin / load_catalog）
                    │
sd_h3_compat_probe.py（独立，零依赖）
```
