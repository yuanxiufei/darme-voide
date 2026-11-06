# 多设备响应式适配（Web 端 + 桌面端）

> 状态：**已落地并实测**（两端各 20 页，320 / 414 / 768 / 1024 / 1440 / 1920 六档视口，横向滚动全为 0）
> 更新时间：2026-11-04
> 相关：`docs/web-desktop-split.md`（两端工程拆分）、两端各一份的 `scripts/check-responsive.mjs`（静态守卫）

---

## 1. 适配范围（设备与视口）

| 档位 | 视口宽度 | 代表设备 | 布局形态 |
|---|---|---|---|
| mobile | ≤ 480px | iPhone SE 320 / 375、Android 360 / 390 / 414 | 单列、页头纵向、顶栏抽屉、弹窗全宽 |
| mobile-wide | 481–767px | 手机横屏、折叠屏展开、小平板竖屏 | 单列 + 双列统计卡 |
| tablet | 768–1023px | iPad 768 / 834 竖屏、1024 横屏 | 侧栏收窄或变顶部横条，主区保留双栏 |
| laptop | 1024–1439px | 1280 / 1366 / 1440 | 完整桌面布局，留白由 48px 收到 36px |
| desktop | ≥ 1440px | 1920 / 2560 | 完整布局，≥1920 内容宽封顶 1680px 居中 |

**两端差异**（如实）：桌面壳 `desktop/src-tauri/tauri.conf.json` 写死 `minWidth: 1024`、`minHeight: 680`，
所以桌面端业务上只会用到 1024 以上；1024 以下同样适配，是为了放开窗口限制、或在 WebView 里嵌移动端时不必重做。
Web 端（Nuxt，浏览器直开）覆盖 320–2560 全区间。

---

## 2. 唯一真源：响应式基础层

两端的 `assets/studio.css` 末尾各有一段 **「响应式基础层」**（Web `frontend/app/assets/studio.css`、
桌面 `desktop/frontend/src/assets/studio.css`），**除顶部 Google Fonts 的 `@import` 外两份文件逐字相同**。
这一段是全仓断点与留白的唯一真源，分 7 节：

| 节 | 内容 | 关键规则 |
|---|---|---|
| 0 | 变量 | `--shell-pad-x/y`（48/36/22/14）、`--tap-min: 34px` |
| 1 | 全局防溢出 | `img/video/canvas{max-width:100%}`、`pre{overflow-x:auto}`、`.mono,.path,.uuid{overflow-wrap:anywhere}`、`.content>*,.page>*{min-width:0}` |
| 2 | 页面容器与页头 | ≤1023 与 ≤767 两档收 `--shell-pad-*`、页头转纵向；`.lib-page` 一并收（它不走全局 `.page`） |
| 3 | 弹窗 | 4 种遮罩 + 7 种面板（`.modal`/`.dialog`/`.ao-panel`/`.confirm-panel`/`.ob-panel` …）在 ≤767 一律 `width:100%`、`overflow-y:auto`、按钮换行并 `flex:1 1 130px`；高度上限写 `calc(100vh - N)` + `calc(100dvh - N)` 双行，支持 `dvh` 的浏览器取后者（移动端地址栏伸缩时不裁底部按钮），不支持的退回 `vh` |
| 4 | 表单与统计网格 | `.field-row/.form-grid/.grid-2/3/4` 降单列；`.token-stats/.cost-stats` 767 两列、420 单列 |
| 5 | 触控 | `pointer:coarse` 下 `.btn` ≥34px、`.btn-icon` 34×34、`.nav-link` 38px、取消 `.btn/.card:hover` 位移 |
| 6 | 超宽屏 | ≥1920 `.page { max-width: 1680px; margin: auto }` |
| 7 | 减少动效 | `prefers-reduced-motion: reduce` 下动画/过渡压到 0.01ms |

原则：**只做「收窄 → 折行 → 单列」，不隐藏任何业务信息**。
窄屏下每个按钮、每段正文、每张统计卡都仍在页面上，只是换了排法。

---

## 3. 布局层：顶栏在 ≤1023 变抽屉

`layouts/default.vue`（两端同一改法）：

- header 内新增汉堡按钮 `.nav-toggle`（三个 `<span>`，展开变 X），`aria-expanded` 跟随状态；
- `≤1023` 时 `.header-nav` 变 `position: fixed; right: 0; transform: translateX(100%)`，宽度 `min(84vw, 320px)`，展开加 `.open` 滑入；
- 遮罩 `.nav-scrim` 只在展开时渲染；
- 层级：`.nav-toggle` 130 > `.header-nav` 120 > `.nav-scrim` 110（保证滑动过程中汉堡始终可点，遮罩不盖住抽屉）；
- 三条关闭路径：点遮罩、按 Esc（`window` keydown）、路由切换（`watch(route.path)`）；
- `≤1023` 时 header 高 54px，隐藏 `.brand-text` 与 `.film-strip`（纯装饰），保留品牌标与汉堡。

实测（390px，触屏仿真）：抽屉闭合时 `left=396`（视口外），展开后 `left=70 / width=320`，
`aria-expanded` 由 `false` → `true` → 点遮罩关闭 → 再开按 Esc 关闭，全部符合预期。

---

## 4. 页面级适配清单

`≤1023` 与 `≤767` 两档在页面自己的 `<style scoped>` 里补规则；表里只列该页**特有**的处理
（通用处理由第 2 节的全局层兜底）。

| 页面 | ≤1023 | ≤767（及 ≤480） |
|---|---|---|
| `index.vue` 项目列表 | 网格 `minmax(min(280px,100%),1fr)` | `.stats-bar` 收间距；`.dialog-head` 转纵向、`.dialog-foot` 按钮换行 |
| `library/index.vue` 资源库 | 沿用 900 档（`.lib-hub` 单列、`.recent-grid` 两列）；`≤1023` 收 `.lib-page` 留白 | `.lib-page` 16/14px、标题 19px、`.recent-grid` 单列 |
| `library/{characters,scenes,props,costumes,weapons}.vue` | 靠全局 `.lib-page` 收留白 | 标题 19px、卡片内边距 12px、`.card-top` 换行、分页间距收敛；卡片网格靠 `min()` 兜底自然降列；触屏下 `.action-btn/.btn-mini/.pager button` ≥34px |
| `drama/[id]/index.vue` 剧集总览 | 已有 860 档 | `.dialog-title` 20px、`.dialog-head` 转纵向、`.dialog-foot .btn` 满宽 |
| `drama/[id]/episode/[n].vue` 制作台 | 沿用页面自有 1240 / 860 档 | `.studio-topbar` 转纵向、`.param-panel` 宽 `min(320px, 100vw-24px)`、`.btn` 抬到 34px、`.bgm-filename` 截断；≤480 `.studio` padding 6px；触屏下 8px 步骤点用伪元素外扩热区到 26px 并拉开点距 |
| `drama/[id]/character/[id].vue` 角色详情 | 已有 760 档；`.variation-grid` 3 列→2 列、留白 `20px` | `.variation-grid` 保持 2 列（≤480 仍 2 列）；`.split-row` 换行且 `.btn-split` 允许折字；`.gen-toolbar`/`.gen-anchorbar` 换行、模型选择器与「一键重做」各占满一行 |
| `drama/[id]/scene/[id].vue` 场景详情 | 已有 800 档；`.detail-body` 由 `320px 1fr` 降单列、`.preview-col` 取消 sticky | `.detail-header` 换行并收留白、`.form-section` 16px、`.header-actions` 换行 |
| `drama/[id]/prompts.vue` 提示词 | 页留白 22/24px | 页头转纵向、搜索框满宽、卡片间距 10px；卡片网格改 `minmax(min(320px,100%),1fr)`，320px 下自然降单列 |
| `settings.vue` 设置 | `.settings-layout` 转列、`.settings-nav` 变顶部横向滚动条（隐藏滚动条、隐藏分组标题） | `.history-stats` 两列、`.art-style-grid` 单列 |
| `models.vue` 模型 | 侧栏 156px | 整页转列并改为整页滚动（`.settings-scroll` 取消内部滚动）、`.models-side` 变顶部横条、`.token-stats` 两列、`.preset-grid` 单列、`.config-row/.config-line` 换行 |
| `skills.vue` 技能 | `.skills-layout` 转列、`.skills-agent-list`（200px）变顶部横条、隐藏 `.skills-agent-title` | 同左，条目 34px 高 |
| `knowledge.vue` 知识库 | `.kb-body` 由三栏降 `184px | 1fr` | 单列；`.kb-side` 变横条；`.kb-drawer` 满宽 |
| `canvas.vue` 画布 | — | `.cv-ai` 满宽、`.cv-hint` 从左下移到左上并限制宽度 |
| `visual-graph.vue` 视觉图谱 | `.vg-search` 200px | 搜索框满宽、`.vg-grid` 单列、`.vg-section-head` 转纵向；触屏 `.vg-en` chip 抬到 32px |
| `style-profiles.vue` 风格预设 | 网格 220px | 单列、`.setup-panel-head` 转纵向 |
| `agents.vue` 智能体 | `.settings-scroll` 留白 24/22px | 标题 19px、卡片头/体/脚内边距 12px、`.agent-card-foot` 换行且按钮 34px、`.skill-bind-usage` 允许折字、`.field-row` 单列 |

组件层：

- `components/Onboarding.vue`：`.ob-skip/.ob-link` 原高 17px → 窄屏与触屏抬到 34px；`.ob-dot` 伪元素外扩热区、点距 14px；`.ob-panel` 限制 `max-height` 并可滚动；`.ob-footer` 允许换行。
- `components/EpisodePreflight.vue`：≤767 弹层变底部抽屉（`.pf-panel` 只留上圆角、`max-height` 用 `calc(100dvh - N)` 双行），长清单在体内滚动。
- `components/PresetPipeline.vue`：同上抽屉化；`.pp-foot` 按钮 `flex: 1 1 130px` 换行、`.pp-counts` 独占一行；卡面 chip 用 `overflow-wrap: anywhere`，320px 下 `hScroll = 0`（实测）。
- `components/canvas/CanvasZoomControls.vue`：触屏下 `.zc-btn` 26px → 34px。
- 画布工具条 `.cv-bar .btn-sm` 触屏抬到 34px（组件内在自己 scoped 里处理，样式跨不过组件边界）。

---

## 5. 有意保留的横向滚动区

以下是**设计如此**的 `overflow-x: auto` 滚动条，不是溢出；320px 下可用（手指横滑），
静态守卫与实测都把它们排除在「溢出」之外：

| 位置 | 内容 |
|---|---|
| `models.vue` `.side-nav` | 模型侧栏 9 项，窄屏变横条 |
| `skills.vue` `.skills-agent-list` | 智能体列表 |
| `knowledge.vue` `.kb-side` | 知识库分组 |
| `episode/[n].vue` `.stage-subnav` | 阶段子导航（68px × N 项） |
| `pre` | 代码/路径块 |

---

## 6. 验收标准（可机器判读）

每条都能用脚本判定，不靠肉眼：

1. **无横向滚动**：每个页面 × 每个视口，`document.documentElement.scrollWidth - clientWidth === 0`；
2. **无越界元素**：除「横向滚动容器内的子元素」与「收缩状态的抽屉」，没有元素 `getBoundingClientRect().right > clientWidth`；
3. **断点真源一致**：`npm run check:responsive` 比对两端 `studio.css` 的响应式基础层，逐字相同；
4. **无硬宽度网格**：不出现 `repeat(auto-fill|auto-fit, minmax(<Npx>, 1fr))` 缺 `min()` 兜底（320px 会被撑破）；
5. **每页都有窄屏规则**：`pages/**/*.vue` 至少一条 `@media (max-width: ...)`；
6. **触控热区**：`pointer: coarse` 下 `.btn` ≥34px、`.nav-link` ≥38px、`.btn-icon` 34×34、抽屉可开可关且 `aria-expanded` 正确；
7. **不隐藏业务信息**：窄屏只允许「收窄 / 折行 / 单列 / 内部滚动」，不允许 `display: none` 掉业务按钮或字段。
   仅两类例外，且都是纯标签而非信息：顶栏装饰 `.film-strip`、`.brand-text`；设置页导航的分组小标题 `.nav-group-label`
   （导航项本身一个不少，只是不再显示「模型 / 平台」这层分组字）。

---

## 7. 实测记录（2026-11-04）

工具：Chrome Headless + CDP（`Runtime.evaluate` 读 `scrollWidth/clientWidth` 与逐元素 `getBoundingClientRect`），
触控档额外开 `Emulation.setTouchEmulationEnabled` + `mobile: true`。

**Web 端**（`http://localhost:3013`，20 个页面）：

| 视口 | 覆盖页面 | 结果 |
|---|---|---|
| 320 | 20/20 | 横向滚动 0，无越界元素 |
| 414 | 20/20 | 同上 |
| 768 | 20/20 | 同上 |
| 1024 | 20/20 | 同上 |
| 1440 | 16/16（省略 library 的 4 个同类子页） | 同上 |
| 1920 | 16/16 | 同上 |

**桌面端**（`http://localhost:3014`，hash 路由，20 个页面 —— 2026-11-04 补齐 `knowledge.vue`，
与 Web 端页面清单现已 1:1 对齐）：

| 视口 | 覆盖页面 | 结果 |
|---|---|---|
| 320 | 20/20 | 横向滚动 0 |
| 768 | 20/20 | 同上 |
| 414 / 1024 / 1440 / 1920 | 7 个核心页（首页/资源库/剧集/制作台/角色/模型/设置） | 同上 |

**断点边界专项**（断点跳变处最容易露缝，除上表主打宽度外，另在**边界值本身**各测一遍）：

| 边界 | Web 端 | 桌面端 |
|---|---|---|
| 480（`≤480` 命中 / 481 不命中） | 8 个核心页 ok | 6 个核心页 ok |
| 767（`≤767` 命中 / 768 不命中） | 15 页 ok | 15 页 ok |
| 1023（抽屉导航命中 / 1024 还原横排） | 15 页 ok | 15 页 ok |
| 1439（`≥1440` 未命中 / 1440 命中） | 8 个核心页 ok | 6 个核心页 ok |

**触控档**（390px，`pointer: coarse` 生效）：`.nav-toggle` 34×34、`.nav-link` 47px 高、`.btn` 34px、
`.ob-skip` 34px、8px 步骤点 `::after` 热区存在（26px）、抽屉开合与 Esc 关闭均正常。

**构建**：Web `vue-tsc --noEmit` 0 错误；桌面 `npm run build` 通过（`dist-frontend` 产物 1.37s）。

---

## 8. 守卫与回归

```bash
# 两端一起扫：断点真源一致性 + 每页有窄屏规则 + 无硬宽度网格
cd frontend && npm run check:responsive
# 等价（桌面端工程里）
cd desktop/frontend && npm run check:responsive
```

脚本 `scripts/check-responsive.mjs`（两端各一份，逐字相同）的判定已做过反向验证：
把某页 `minmax(320px, 1fr)` 写回去 → 报「缺 min() 兜底」；在桌面端 `studio.css` 末尾加一行 → 报「两端响应式基础层不一致」；
复原后退出 0。

改样式的固定流程：**先改 Web 端 → 把 `<style` 之后的整段同步到桌面端同名文件 → 跑守卫 → 跑实测矩阵**。
两端风格化差异只有 `studio.css` 顶部的字体 `@import`（桌面端为 CSP 改成系统字体栈），响应式层必须逐字相同。

---

## 9. 已知边界

1. **桌面壳窗口限制**：`minWidth: 1024`，所以 1024 以下只是「样式层就绪」，产品上还不能把窗口拉到手机宽度；
2. **未做真机 Safari 回归**：触控档用 CDP 的触摸仿真 + `mobile` 设备度量。弹窗高度已写 `100dvh` 渐进增强兜住地址栏伸缩，
   但真机上的 `dvh` 重排时机（地址栏收缩过程中）没有实测样本；
3. **图片未做 `srcset`**：素材是本地单档（2048×1536 角色板等），只做 `max-width:100%` 等比缩放，不做多分辨率取图；
4. **表格类内容**（如成本明细）在 320px 下依赖 `pre/table` 的横向滚动，不强行折行；
5. **超宽屏只封顶不放大**：≥1920 时 `.page` 封顶 1680px，不额外增列（避免把信息密度堆到无法扫读）；
6. **两端页面清单已对齐**（2026-11-04）：桌面端补齐 `knowledge.vue` + `/knowledge` 路由 + 导航项，
   两端各 20 页；此前"知识库只在 Web 端"的差异已消除。
