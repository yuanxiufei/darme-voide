# playbook —— 方法原料库（项目数据层 · 入库可维护）

短剧/编剧方法的第一手原料（招式库、造型库、特效库、分镜规范、技法资料库、情绪实例等）。

**定位（2026-11-04 创作者裁定）**：它是**本项目的可维护资产**，不是外部只读快照 ——
本仓可以改它、修它、按需重写它，改动走 code review。它住在 `data/` 这个项目数据层里，
与 `drama.db`／`static/` 那些机器数据的区别只有一条：**它入库、可维护、会被人改**。

两层分工，别混：

| 层 | 放什么 | 落点 |
|---|---|---|
| 原料层（本目录） | **资料本身**：招式、造型、特效、分镜规范的原话与结构 | `corpus/playbook/` |
| 判据层 | **怎么用**：归纳成可判读的规则、反例、自检项 | `backend-py/app/skills/*/references/` |

唯一的硬边界：来源与许可不明（见下「来源与版本」），所以**对外输出不许搬运原文句子**
（`THIRD_PARTY_NOTICES.md` §六）。这条不因「可修改」而解除 —— 改它是把别人的资料改成本项目的
写法，不是把别人的句子搬进产品。

## 来源与版本

| 项 | 值 |
|---|---|
| 上游 | `AI STUDIO洋子 V3.3.1`（热修版课程包） |
| 获取方式 | 课程包随盘到货，**无 LICENSE 文件** |
| 入仓日期 | 2026-11-04 |
| 规模 | 138 份 Markdown / 2348 小节 / 约 620 K 字符 / 1.46 MiB |
| 旧位置 | `参考/` → `方法语料/AI STUDIO洋子-V3.3.1-热修版/`（两处都在 `.gitignore` 里） |
| 许可 | **来源与许可均不明** ⇒ 只作写法与结构参考，**输出不许搬运原文句子**（见 `THIRD_PARTY_NOTICES.md` §六） |

为什么搬进仓库：前两处位置都被 gitignore，后果是新克隆的仓库里知识库页面与
`search_script_library(source="playbook")` **一起是空的**（`available=false`）——
资料只存在于作者机器上，等于系统里没有。

## 结构

```
corpus/playbook/
  README.md              ← 本文件（元数据，不进检索索引）
  film-expert-bridge/    ← 目录名＝skill 名（PLAYBOOK_DIRS 登记项），被索引
    combat/              ← 打斗素材（77 份：连招专项 26 + 徒手技法 12 + 战场场景库 11
                            + 技能机制库 11 + 兵器技法 10 + 案例与角色 4 + 根下系统文档 3）
    ultimate/            ← 大招素材（11 份，原料）
    general/             ← 通用素材与绘制规范（39 份：11 族 38 份原料 + 1 份入口索引）
    modules/             ← 7 份**规约摘要**（不是原料，见下）
  film-cinedance/        ← 4 份**规约文档**（不是原料，见下）
```

**2026-11-04 扁平化**：原路径是 `packs/film-expert-bridge/references/library/combat/战场场景库/x.md`
（6 层目录，从仓库根算 9 段），其中 `packs/`、`references/`、`library/` 三层对检索毫无贡献，
只让引用变长；`references/` 还跟判据层 `app/skills/*/references/` 撞名，最容易被认错。
现在包名直接在 playbook 根下，最深 3 层（`film-expert-bridge/combat/战场场景库/x.md`）。
文件一个没动、内容一字未改，只是换了位置。

**2026-11-04 二次分层（combat 内部）**：扁平化之后 `combat/` 根下还剩 55 份平铺，
问题不是「文件多」，而是**三套编号系列挤在同一平面**——`07-32` 连招专项、`00-10` 技能机制、
`17-27` 战场场景，编号互相撞（`07`／`08`／`17` 各有两份以上），且「机制原理」与「具体怎么打」
之间存在正向配对引用却看不出配对关系。故按**文件性质**而非编号重分成 6 族，根下只留系统文档：

| 族 | 份数 | 装什么 |
|---|---|---|
| `连招专项/` | 26 | `07-32` 编号的成套连招（机制的具体打法侧） |
| `徒手技法/` | 12 | `徒手技法资料库_*` |
| `战场场景库/` | 11 | `17-27` 编号的战场拓扑 |
| `技能机制库/` | 11 | `00-10` 编号的机制原理（连招的上游） |
| `兵器技法/` | 10 | `兵器技法资料库_*` |
| `案例与角色/` | 4 | 精品词库／精品案例提炼／七角色武学绑定V2／Web 工作台对接 |
| 根下 3 份 | 3 | `打戏系统交接包.md`／`打戏系统资料索引.md`／`打戏生成自检清单.md`（系统级入口） |

配套改动只做一件事：把「引用」改成带族前缀的相对路径（索引 33 处、交接包 2 处、
`技能机制库 → 连招专项` 12 处跨目录 `../`）。**正文一字未改**，份数仍 77、全库当时 137（现 138，多出的是
`general/通用资料索引.md`）。

### 两类东西混居（2026-11-04 检查记录）

两个技能目录合计 126 份是上游课程包的**原料**；另 11 份不是，它们性质接近
`app/skills/*/references/` 里的判据摘要：

- `film-expert-bridge/modules/` 7 份：`shot_grammar.md`／`beat_rhythm.md`／
  `director.md`／`master_cases.md`／`reference-art.md`／`combat.md`／`ultimate.md`
- `film-cinedance/` 4 份：`shot-planning.md`／`continuity-compiler.md`／
  `seedance-25-prompt-format.md`／`visual-continuity-audit.md`

**裁定：留在原处不搬。** 搬进 `app/skills/` 会让 `playbook_skills()` 的 `skill=` 过滤维度
失去意义（`film-cinedance` 会空），且份数、小节数这套账全部要重算。但它们"能被检索
到的判据摘要"这个身份本身有用——所以调用方要知道：**从这两处检索到的是判据措辞，
不是可借鉴的原料写法**。检查过程与拒收理由见 `docs/playbook-audit.md`。

**为什么目录名就是技能名**：`skill=` 的取值直接取路径第一段，白名单由 `PLAYBOOK_DIRS`
显式登记，不是「扫根下所有目录」——后者省事，但会把以后丢进根目录的别家资料一起卷进索引。
新增一棵来源必须登记；漏登记的 md 不会静默消失，`/report` 的 `unindexedFiles` 会如实报出来。

历史上这两层叫过 `skills/`（上游原名）与 `packs/`（2026-11-04 为避与
`backend-py/app/skills/` 撞名而改），同日扁平化时一并删掉。旧写法
`skill="packs/film-acting"`、`skill="skills/xxx"` 仍被 `_skill_arg()` 归一成技能名，
已有调用方不受影响。**`path=` 一侧没有兼容**：检索结果现在回的是新形状，手抄旧路径
（`packs/…`、`…/references/library/…`）会被判越界，`_playbook_target()` 直接返回 `None`。

## 前后端怎么用

| 使用方 | 入口 | 说明 |
|---|---|---|
| 后端服务 | `app/services/script_library.py` | `PLAYBOOK_ROOT` 指向本目录；按 `#` 小节切块、超 1500 字切窗、进程内建命中表（毫秒级）；`SCRIPT_PLAYBOOK_ROOT` 可指向外部副本 |
| HTTP API | `GET /api/v1/knowledge/search?source=playbook` | 六个目录维度只筛 methods，`skill` 只筛 playbook |
| HTTP API | `GET /api/v1/knowledge/report` | `playbook` 段给份数/小节/技能分布/未索引数 |
| HTTP API | `GET /api/v1/knowledge/document?path=film-expert-bridge/...` | **翻全文**（playbook 的 md 直接读，不经 sqlite） |
| HTTP API | `GET /api/v1/knowledge/facets` | `playbookSkills` 给可用技能名，别让调用方瞎猜 |
| 前端 | 知识库页 `/knowledge` | 检索结果标「方法」来源，点开在侧栏读全文 |
| Agent | 工具 `search_script_library` / `list_script_library` | `source="playbook"`；`path_contains` 收窄到子目录 |
| Skill | `app/skills/*/SKILL.md` | 只装归纳出的判据；要原文才回来检索这里 |

## 维护

**动手前先分清改的是哪一层**：改「资料本身」在本目录改；改「怎么用」的规则去
`backend-py/app/skills/*/references/`，别把判据写进原料库。

| 动作 | 手续 |
|---|---|
| 加文件 | 放进对应来源目录的对应族（如 `film-expert-bridge/combat/连招专项/`，族划分见上表），下次检索自动纳入（缓存按 `(路径, mtime)` 判重）；**新增一棵来源**要先进 `PLAYBOOK_DIRS` 白名单，否则不索引；同时改本 README「规模」行——那条有测试钉着 |
| 改正文 | 可以改：错字、过期口径、明显错分类、把文艺自造名换成通用术语。改完先查 `app/skills/*/references/` 是否引用了同一内容，引用到就一起改 |
| 删文件 | 可直接删；`/report` 的 `unindexedFiles` 会如实报还有几份没进索引，不会静默 |
| 整包替换 | 换新版本课程包前先读 `docs/playbook-audit.md`：那里记了哪些内容已判定为「别家角色资产／世界观的文艺自造名」而不收 |

其余四条不变：

- **不要**在本目录放 pdf/图片/docx：只有 `*.md` 会被索引，其他格式静默不计（不报错，也不参与检索）。
- 根目录直属的 `*.md`（如本文件）视为元数据，**不计入** `unindexedFiles`。
- 改完跑一次 `python -X utf8 tests/script_library_test.py`（README 的份数/小节数被它钉着）。
- 这里有 138 份资料，改一份别顺手重排全目录 —— 检索命中表按路径缓存，大搬家会让所有引用一起失效。真要重排就像 2026-11-04 那样**一次做完**：改白名单 → 移动文件 → 同步全部引用 → 跑守卫，别分批挪一半。
