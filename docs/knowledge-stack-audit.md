# 知识库技术栈审计与取舍（2026-11-07）

37 个候选仓（RAG 框架／文档解析／知识库 GUI／图检索）逐个核过「是否存在、是否归档、最近一次提交、许可证、能否落到本仓」，本文件只留**结论与凭据**，不复述逐仓快照（star 数、提交数、LICENSE 原文）：那些数一周就过期，而这里要固定的是**判据**。

## 1. 判定前提：本仓不是通用 RAG 平台

| 本仓的事实 | 对选型的硬约束 |
|---|---|
| 语料是 5802 份 PDF ＋ 873 份 `.doc` ＋ 166 份 `.docx` ＋ 少量 HTML/PPTX，共 8k 级文件、几十 GB | 解析器要能**离线跑完一整库**，不靠 API、不逐份人工介入 |
| 检索只有 fts5（bm25）＋`instr` 路径过滤，两库按**秩**融合（`curated`／`ocr`） | 向量库、rerank、多路召回**没有挂载点**；换掉检索层＝重写 `script_library` 与全部前端读法 |
| 消费方是 Agent 工具与「剧本知识库」页面 | 输出必须是**纯文本块**，不需要版式还原、不需要结构化抽取 |
| 桌面端是分发形态（Tauri 壳＋NSIS 安装包） | 依赖许可必须可分发；AGPL 会传染整个主链路 |
| 扫描件已有第二条命（`ocr_index.py` ＋ ONNX Runtime） | 新增解析器只补「有文字层」那半，不重复做 OCR |

## 2. 结论四档

**① 采用（本轮唯一落地项）：`pypdfium2`** —— PDF 文字层抽取，Apache-2.0/BSD-3、wheel ≈3 MB、无外部服务。落地见 §4。

**② 不采用 · 领域错位**：通用 RAG 框架与向量库（检索层要整条换掉，而本仓的 BM25 口径是**已定契约**：两库分数不可比、只比秩）；知识库 GUI（本仓已有 Nuxt 页面与 Tauri 壳，两套前端必然分裂）。

**③ 不采用 · 许可证**：`PyMuPDF` 技术上是本轮 PDF 三选里最好的一个（12/12、0.08 s），但 AGPL-3.0 一进分发主链路就传染；不为了省 20 行代码换掉整包许可。⚠️ 补记：`ocr_index.py`（扫描件 OCR 线）原先按需 `import fitz`（PyMuPDF），**2026-11-07 已一并换成 `pypdfium2`**（取字／`PdfPage.render()`／数页三件 PDFium 都做得了），`requirements-ocr.txt` 同步改，测试造样本也从 fitz 改为手写最小 PDF ⇒ 本仓产品与测试代码零 AGPL，见 `THIRD_PARTY_NOTICES.md` §三 第 4 条。同日 `backend-py/_bench/` 那 14 个一次性体检脚本也迁走（新增 `_bench/pdfium_compat.py` 同名适配层，脚本侧只换 `import fitz` → `import pdfium_compat as fitz`）⇒ 全仓 `import fitz` 归零，连本地复现语料数字都不再需要 PyMuPDF。

**④ 不采用 · 形态不符**：`graphify` 经核不是库，是一组 Claude Code slash 命令（无 Python 包、无可导入入口）；R2R 停更一年（最近提交在一年前），拿它做基础设施等于接一份没人维护的依赖。

**⑤ 不采用 · README 里带指令**：个别仓（如 MinerU）的 README 内嵌了写给 AI agent 的指令性文字（"忽略前面的说明"之类）。**未执行**，只按普通技术文档参考；这类内容按内容策略一律当数据、不当指令。

**⑥ 不采用 · 收益为负**：`markitdown[all]` 覆盖 Office/HTML 最省事，但它对本仓的意义只剩 HTML —— 实测安装体积与耗时明显大于自写 `HTMLParser`（后者零依赖、30 行），而它最诱人的 PDF 路线是**表格化**输出（见 §3），对「按行切块 + BM25」是负收益。

## 3. 选型的实测凭据（PDF 三通道）

样本：`剧本知识库/全球热门与获奖影视剧本知识库` 里 12 份真实剧本 PDF，`random.seed(20261107)` 抽定，同机同轮。

| 通道 | 成功 | 耗时 | 输出形态 | 许可 |
|---|---|---|---|---|
| `pdfminer.six` | 9/12 | 0.5–3.8 s | **表格化**：每行套 `\| … \|`，正文被切成碎片；3 份抛 `No /Root object` | MIT |
| `PyMuPDF` | 12/12 | 0.08 s | 干净 | **AGPL-3.0** |
| `pypdfium2` | 12/12 | 0.08 s | 干净 | Apache-2.0/BSD-3 |

另：`ocr_index.py` 里本来就有一条「先读 PDF 自己的文字层（约 1 ms/页，有字就不 OCR）」的快路径 —— 那条路只服务**供应商库已登记**的件，且要等 OCR 线排到它；第三库这条路是给「主库没登记 / 登记了但没正文」的件用的，两者不重复（`_candidates` 把主库与 OCR 侧已覆盖的相对路径整批跳过）。

## 4. 落地清单

| 位置 | 改动 | 判据（`tests/lib_index_test.py`） |
|---|---|---|
| `app/scripts/corpus/lib_index.py::_pdf_text` | `pypdfium2` 逐页 `get_textpage().get_text_range()`，空正文抛 `RuntimeError`（"扫描件，需走 OCR 线"） | 自造最小 PDF 抽出正文；坏 PDF 报可读错；端到端 `_build` 判 `extra` |
| `lib_index.py::_cjk_deglue` | 汉字/中文标点之间的填充空格去掉，**只作用于 PDF 通道** | 四条：汉字、中文标点、英文不动、中英混排 |
| `lib_index.py::_html_text` | `HTMLParser` 剥标签，`script/style` 整段丢 | 正文保留、样式表不进检索、实体解码、**行内标签两侧空格保住**、端到端 `_build` |
| `lib_index.py::_pptx_text` | `zipfile` 读 `ppt/slides/slideN.xml`，按序取 `<a:t>` | 顺序抽取、属性/实体容错、空 deck 如实报错 |
| `lib_index.py::SUPPORTED_EXTS` | **单一真源**：抽文本通道登记表 | 逐个扩展名都必须真被 `_candidates` 收下 |
| `app/services/library_ingest.py` | `ALLOWED_SUFFIXES` 直接读 `SUPPORTED_EXTS`；低字数理由按格式分档 | 白名单同源；拒绝文案不再谎称「PDF 走扫描件 OCR」 |
| `requirements.txt` | `pypdfium2>=4.0`（缺失只影响 `.pdf`；OCR 线的页面渲染也用它，见 `requirements-ocr.txt`） | —— |
| `app/scripts/corpus/ocr_index.py` | 摘掉 `import fitz`（PyMuPDF／AGPL），改 `pypdfium2`：取字 `get_textpage().get_text_range()`、渲染 `page.render(scale=dpi/72, rev_byteorder=True)`、数页 `len(doc)` | `tests/ocr_index_test.py` 四条：页数按真文件数、出图 `(H,W,3) uint8`、纯红页出来仍是红的（字节序）、96 dpi 下 595×842 pt → 794×1123 px（dpi 换算）；造样本改手写最小 PDF，两边不再同库 |

真实语料复核（库根里主库/OCR 都没覆盖的 25 份，2026-11-07）：

- **6 份 HTML 全部抽出 1.6–12 万字真实剧本文本**，零处词粘连（抽查 `romanholiday.html`／`s_seatl.html`／`shine.html` 均为整段对白，无导航栏、无标签残留）。
- **19 份 PDF 一份都抽不出，且不是解析器的问题**：3 份是 `book\0mark` 私有格式改名、6 份 0 字节、1 份 53 MB 全零、9 份 `%PDF-1.x` 头但正文里没有 `startxref`／`%%EOF`／`endobj` 的**截断**文件。全部走 `quarantine`（文件留盘、一块不写、转 OCR 线），一条都不是静默丢弃。

## 5. 已知边界

1. **不做版式还原**：表格、多栏、脚注都按阅读流拉平。BM25 要的是词，不是坐标；需要版式时走渲染 + OCR 那条线。
2. **HTML 不做正文抽取**：只剥标签，不判 boilerplate（导航/页脚仍会进正文）。本库的 HTML 是剧本转存页，实测干净；将来整站爬存时要重看这一条。
3. **截断/私有格式 PDF 只能 quarantine**：补救路是 OCR，但截断文件 OCR 也读不出内容 —— 这类需要人补源文件，判据不会假装"入库成功"。
4. `pypdfium2` 缺失时 `.pdf` 抽不出（其余格式不受影响），报错点明缺的是哪个包。
5. **换渲染器会改 OCR 的输入像素**：MuPDF 与 PDFium 的栅格化实现不同（抗锯齿、亚像素定位），同一页同 dpi 出图不会逐像素相同。DPI、RGB 通道序、页面尺寸都已对齐并钉了判据，但对**已入库的那批 OCR 结果**做差分时会看到细微差异 —— 归因是渲染器换了，不是识别质量退步。

## 6. 复跑

```powershell
cd backend-py
.\.venv\Scripts\python.exe tests\lib_index_test.py          # 61 条判据
.\.venv\Scripts\python.exe tests\library_overlay_test.py    # 上传三档回执
.\.venv\Scripts\python.exe -X utf8 -m app.scripts.corpus.lib_index --all   # 真跑（进 sidecar）
```
