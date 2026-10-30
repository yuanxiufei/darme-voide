# 演练：HTTP 停掉之后，引擎还能不能给结论（E1.4）

> 对应《引擎独立性实施方案》E1.4，判据来源：LiteInfer WBS §4 G1「HTTP 服务停止不影响独立 SDK 的基本功能」。
> **判据只有一句**：后端进程不在时，`python -m app.services.engine` 四条命令仍能各自给出**正确退出码与结论**。

---

## 1. 为什么要单独演练

「引擎不 import fastapi」只是**静态**事实（AST 能证明）。但**静态干净 ≠ 运行时能跑**：只要有任何一个模块在 `import` 期去读配置、连数据库、等 HTTP，命令行就会在启动那一刻挂掉——而这条路径平时被后端全程覆盖着，永远不会有人发现。

所以 E1.4 不看代码，只看**进程**：先杀掉后端，再跑命令。

---

## 2. 怎么做（PowerShell，全程约 20 秒）

```powershell
# ① 停掉后端（按端口找，不猜进程名）
cd d:/code/voide/darme-voide
$p = (netstat -ano | Select-String ':5790' | Select-String 'LISTENING' |
      ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -First 1)
if ($p) { Stop-Process -Id $p -Force }
Test-NetConnection -ComputerName localhost -Port 5790 -InformationLevel Quiet   # 必须 False

# ② 在无 HTTP 的状态下跑四条命令，逐个取退出码
cd backend-py
$env:PYTHONIOENCODING='utf-8'
foreach ($sub in @('doctor','plan','readiness','status')) {
  $null = .\.venv\Scripts\python.exe -X utf8 -m app.services.engine $sub 2>&1
  "$sub=$LASTEXITCODE"
}
```

> 用 `$null = ...` 而不是接管道：接了管道后 `$LASTEXITCODE` 会被后续 cmdlet 覆盖，读出来的数字不可信。

---

## 3. 判定标准

| 命令 | 允许的退出码 | 说明 |
|---|---|---|
| `doctor` | `0` | 能力声明 + 环境事实，永远能答（缺包也是 0，缺什么写在输出里） |
| `plan` | `0` 或 `1` | `1` = 就绪为假（缺权重/量化族未实现），**不是**「命令坏了」 |
| `readiness` | `0` 或 `1` | 同上 |
| `status` | `0` | 只读运行时状态，不需要装载 |

**不通过的样子**（任一出现即判失败）：

- 任何一条出现 Python traceback（哪怕后面还打了结论）；
- 退出码是 `2`（参数错）或 `3`（内部错误）以外的意外值 —— 这两个码是**真故障**；
- `plan`/`readiness` 报「连接被拒绝」「无法连接 5790」之类：说明有代码偷偷在走 HTTP。

---

## 4. 本轮实测记录（2026-10-29）

- 后端：**已停**（`Test-NetConnection -Port 5790` ⇒ `False`）；
- 环境：`backend-py\.venv`，`torch 2.15.0+cpu`；

| 命令 | 退出码 | 关键结论 |
|---|---|---|
| `doctor` | `0` | 1420 字节输出，含 `apiVersion` / 缺什么 / 未验收项 / 既知缺口 |
| `plan --stage h3` | `1` | `阶段 h3 就绪=False`；`dit_fl2va_int8` 19.53 GiB sequential int8，并指出 `convrot` 族未实现 |
| `readiness --stage h3` | `1` | 3 条阻断（`convrot`×2、`nvfp4`×1）＋ 4 条未查（文件不在盘上）＋ 显存估 49.44 GiB / 容量 24 GiB ⇒ 溢出 |
| `status` | `0` | `阶段 None 运行中=None / 已装: 无`（无进程内会话，正常） |

原始输出留档：`tmp/e14-{doctor,plan,readiness,status}.txt`。

结论：**通过**。四条命令全部在无 HTTP 的进程里给出结论，且退出码与语义一致。

---

## 5. 顺带得到的两个事实（记录下来，别下次再查）

1. **引擎默认权重根是仓内 `data/models`**，不是桌面共享池（`D:\Comfy-Desktop\ComfyUI-Shared\models`）。所以同一份 stage 在 CLI 里可能报「缺文件」，而 ComfyUI 那边明明有 —— 这不是 bug，是根目录不同。要按共享池算，显式传 `--root`。
2. **显存估算会溢出**（49.44 GiB vs 24 GiB）是**权重总量**口径：`readiness` 报的是「必需权重合计」，不是「同时驻留」。真正的驻留视图在 `plan` 的 `residency.peakResidentGiB`（分块/分层装载后是否装得下，以它为准）。
