# `logs/` — 运行日志暂存区

**一行版**：这里的每个文件都是「某次手动起服务 / 跑全量时 `>` 出来的屏幕输出」，
**全是可丢弃的**；要留证据就贴进 `docs/`，别指望本目录。

## 为什么单独一个目录

2026-11-07 整理前，这些日志直接摊在仓库根上（`backend.out.log` 605 KB 那种），
与 `README.md`、`Dockerfile`、`docker-compose.yml` 混在一起，一眼分不清哪个是本仓的
真文件、哪个是刚跑出来的烟。现在根上只留源码与配置，日志一律进本目录。

## 逐文件

| 文件 | 谁写的 | 内容 | 能不能删 |
|---|---|---|---|
| `backend.out.log` | 手动起后端时 `> backend-py/… 2>&1`（历史上摊在根上） | uvicorn 标准输出：启动横幅、每个请求一行、引擎装配回执 | ✅ 随时 |
| `backend.err.log` | 同上，`2>` 的那一半 | 异常栈、警告（常为空） | ✅ 随时 |
| `desk.out.log` | 手动起桌面壳 / Vite dev 时重定向 | Vite 编译、tauri dev 输出 | ✅ 随时 |
| `desk.err.log` | 同上，`2>` 的那一半 | 同上（常为空） | ✅ 随时 |
| `_run_all.log` | `python tests/run_all.py 2>&1 > _run_all.log` | 全套自检的逐套 `OK/FAIL` 行与末尾 `结论` 汇总 | ✅ 随时（结论该抄进 PR/变更记录，不该只躺在日志里） |

**没有任何代码按文件名打开它们**（2026-11-07 全仓搜过：`backend.out`/`desk.out`/`_run_all.log`
零引用，唯一的 `backend.err.log` 出现在 `backend-py/tests/structure_test.py:666`，那是临时沙箱里
造出来的假文件，与这里无关）⇒ 改名、移动、删除都不会打断任何流程。

## 与 `backend-py/tmp/` 的分工（别开第三处）

两处都是「一次跑的输出」，区别只在**谁写**：

| | 本目录 | `backend-py/tmp/` |
|---|---|---|
| 谁写 | **人**手动起服务（uvicorn / `tauri dev`）时在**仓库根**重定向出来 | **`tests/run_all.py` 自己**写的账（`run_all_*.log`）＋ 一次性探针 `.py` / `.cmd` |
| 在哪 | 仓库根外侧，与 `README.md` 同级 | 后端包内部 |
| 忽略 | 靠 `.gitignore` 的 `*.log` 整类命中 | 另有 `backend-py/tmp/` **专属**条目（`output_layout_test` ⑨′ 守着，防止通用规则一改就混进提交） |
| 进安装包 | 不进（`Dockerfile` 按路径 `COPY`，`.dockerignore` 的 `*.log` 也挡） | 不进（`desktop/scripts/sync-backend.mjs` 的 `EXCLUDE_DIRS` 排除） |

判据：**跟测试脚本一起跑的落 `backend-py/tmp/`，人在根上手敲的重定向落本目录**。两边都只许
`.log` 与脚本；媒体产物一律进 `<dataRoot>/experiments/<实验名>/`（`output_layout_test` ⑨ 守着 ——
落在 `tmp/` 会随 `tmp/` 一起被当垃圾清）。

**第三处已经冒充过一次**：`backend-py/logs/`（后端包内部）在 2026-11-08 10:54–11:34 攒了
**73 个** `_chk_*.log` / `_erc*.log` / `_r3_*.log` 这类「单套复跑 / 反向验证」的屏幕输出。
按上面判据它们该落 `backend-py/tmp/`，当日已整体移过去（`tmp/` 现在纯 `.log`、⑨ 仍绿）并删掉那个目录。
教训是：`*.log` 的忽略规则只保证它们**不入库**，一点也挡不住「又多一个落点」——
所以这里把话说死：**`backend-py/logs/` 不许存在**，见到就按本判据搬进 `tmp/`。

## 与 `.gitignore` 的关系

- `.gitignore` 的 `*.log` 已经整类挡住 ⇒ 本目录里的 `.log` **永不入库**；
- 本文件（`README.md`）不在该模式内 ⇒ 会入库，作用是把「这里能删」这件事写死给下一个人（与
  `data/README.md`、`models/README.md` 同一套做法）。

## 想留证据怎么办

日志不是证据。真跑验收要留的（图片、mp4、引擎回执）分别归：

- 引擎**成品** ⇒ `<dataRoot>/static/engine/`（要发 URL、随数据根备份）；**实验产物** ⇒
  `<dataRoot>/experiments/<实验名>/`（不上线、不发 URL）—— 两者都不在代码目录里（`backend-py/output/` 已退役）；
- 数据面结果 ⇒ `data/`（`static/` 图、`traces/` 轨迹）；
- 需要**入库**的结论（性能数字、基准对比）⇒ 写进 `docs/`，日志只作附注。

## 清理

```powershell
# 全清（保留本文件）
Get-ChildItem logs -Filter *.log | Remove-Item
```
