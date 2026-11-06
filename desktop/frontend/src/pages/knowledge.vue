<template>
  <div class="kb-page">
    <header class="kb-head">
      <div class="kb-head-main">
        <div class="setup-kicker">Knowledge Base</div>
        <h2 class="kb-title">知识库</h2>
        <p class="kb-desc">
          剧本／编剧方法／影视研究期刊 —— 可检索、可编辑、可删除、可上传。登记 {{ totals?.documents ?? '—' }} 份，
          其中 {{ totals?.withText ?? '—' }} 份带文字层；扫描件由本仓 OCR 逐页补进 sidecar，
          补出来的片段在结果里标 <span class="kb-pill kb-pill-ocr">OCR</span>。
          正文改动落在<b>覆盖层</b>（供应商/OCR 的原库 sqlite 一个字不动，重跑索引也冲不掉）；
          删除先进回收站可还原，只有自己上传的件才能彻底删。
          随仓库自带的方法原料库（<span class="kb-pill kb-pill-ok">方法</span>，{{ playbook?.files ?? '—' }} 份）
          直接可搜可读，不依赖剧本库是否安装。
        </p>
      </div>
      <div class="kb-stats">
        <div class="kb-stat">
          <b>{{ totals?.documents ?? '—' }}</b><span>登记</span>
        </div>
        <div class="kb-stat kb-stat-ok">
          <b>{{ totals?.withText ?? '—' }}</b><span>有正文</span>
        </div>
        <div class="kb-stat kb-stat-ocr">
          <b>{{ totals?.ocr ?? '—' }}</b><span>OCR 已补</span>
        </div>
        <div class="kb-stat kb-stat-warn">
          <b>{{ totals?.pending ?? '—' }}</b><span>待补</span>
        </div>
        <div class="kb-stat kb-stat-ok" :title="playbookNote">
          <b>{{ playbook?.files ?? '—' }}</b><span>方法</span>
        </div>
        <button class="btn btn-ghost" :disabled="loadingTree" @click="reload">
          <RefreshCw :size="14" :class="{ 'animate-spin': loadingTree }" /> 刷新
        </button>
      </div>
    </header>

    <!-- 剧本库没装时**不挡整页**：方法库是独立源，照样能搜能读。此前这里是 v-if/v-else，
         两个源一起被藏掉 —— 没装供应商库的人连自带的方法库都用不了。 -->
    <div v-if="tree && !tree.available" class="kb-empty">
      <p>剧本库未安装：{{ tree.hint }}</p>
      <p class="kb-hint">把供应商的 <code>curated-screenplay-methods.sqlite</code> 放回
        <code>剧本知识库/…/99-索引与检索规则/</code> 即可（只读打开，编辑与删除都落覆盖层，不会动这个文件）。<template
          v-if="playbook?.available">方法库不受影响（{{ playbook.files ?? 0 }} 份 /
          {{ playbook.sections ?? 0 }} 节，可直接搜）。</template></p>
    </div>

    <div class="kb-body">
      <aside class="kb-side">
        <div class="kb-side-head">资料目录</div>
        <button
          class="kb-group"
          :class="{ active: !activeGroup }"
          @click="pickGroup(undefined)"
        >
          <span class="kb-group-name">全部</span>
          <span class="kb-group-num">{{ totals?.documents ?? 0 }}</span>
        </button>
        <button
          v-for="g in tree?.groups ?? []"
          :key="g.path"
          class="kb-group"
          :class="{ active: activeGroup === g.path }"
          :title="g.path"
          @click="pickGroup(g.path)"
        >
          <span class="kb-group-name">{{ g.name }}</span>
          <span class="kb-group-num">{{ g.documents }}</span>
          <span class="kb-group-badges">
            <i v-if="g.ocr" class="kb-pill kb-pill-ocr">OCR {{ g.ocr }}</i>
            <i v-if="g.ocrPartial" class="kb-pill kb-pill-warn">OCR 部分 {{ g.ocrPartial }}</i>
            <i v-if="g.pending" class="kb-pill kb-pill-warn">待补 {{ g.pending }}</i>
          </span>
        </button>
      </aside>

      <main class="kb-main">
        <div class="kb-toolbar">
          <div class="kb-search">
            <Search :size="15" class="kb-search-icon" />
            <input
              v-model="q"
              class="kb-search-input"
              placeholder="搜正文：三幕结构 / 激励事件 / 序列节拍 …"
              @keyup.enter="doSearch"
            >
            <button v-if="q" class="kb-search-clear" @click="clearSearch">
              <X :size="13" />
            </button>
          </div>
          <button class="btn btn-primary" :disabled="searching || !q.trim()" @click="doSearch">
            <Loader2 v-if="searching" :size="14" class="animate-spin" />
            <span v-else>检索</span>
          </button>
          <button class="btn btn-ghost kb-upload-btn" @click="openUpload">
            <Upload :size="14" />
            <span>上传语料</span>
          </button>
        </div>

        <!-- 筛选维度**由 /facets 下发**（值 + 条数），页面不写死任何国别/类型清单：
             目录 CSV 一换，下拉跟着变；猜词猜出来的条件会静默筛出 0 条。 -->
        <div v-if="facetRows.length || playbookSkills.length" class="kb-filters">
          <select
            v-for="row in facetRows"
            :key="row.key"
            v-model="filters[row.key]"
            class="kb-filter"
            :class="{ on: !!filters[row.key] }"
            :title="row.label"
          >
            <option value="">{{ row.label }}</option>
            <option v-for="opt in row.options" :key="opt.value" :value="opt.value">
              {{ opt.value }}（{{ opt.count }}）
            </option>
          </select>
          <select
            v-if="playbookSkills.length"
            v-model="skillFilter"
            class="kb-filter"
            :class="{ on: !!skillFilter }"
            title="技能目录：只筛方法库（剧本库没有这个维度）"
          >
            <option value="">技能（方法库）</option>
            <option v-for="s in playbookSkills" :key="s.name" :value="s.name">
              {{ s.name }}（{{ s.sections }} 节）
            </option>
          </select>
          <label class="kb-check" :title="partialHint">
            <input v-model="includePartial" type="checkbox">
            含 OCR 半成品
          </label>
          <label v-if="activeGroup" class="kb-check" title="默认搜全库；勾上只在这个目录里搜">
            <input v-model="searchInGroup" type="checkbox">
            只搜当前目录
          </label>
          <button v-if="filtersActive" class="kb-filter-clear" @click="resetFilters">清空条件</button>
          <span class="kb-filters-hint">改完条件要点「检索」才生效</span>
        </div>

        <div class="kb-tabs">
          <button
            v-for="tab in TABS"
            :key="tab.id"
            class="kb-tab"
            :class="{ active: only === tab.id && view === 'browse' }"
            @click="pickOnly(tab.id)"
          >
            {{ tab.label }}
            <span v-if="tabBadge(tab.id)" class="kb-tab-badge" :class="{ 'is-quiet': tab.id === 'buried' }">
              {{ tabBadge(tab.id) }}
            </span>
          </button>
          <!-- `ranking` 必须露在脸上：bm25 / 短词 like / 字符 n-gram 是三种**不可比**的排序，
               只看条数会得错误结论（同一条查询换个词数就从 bm25 掉到 like）。 -->
          <span v-if="view === 'search'" class="kb-tabs-note">
            检索结果 · {{ hits.length }} 条<template v-if="scopeNote"> · {{ scopeNote }}</template><template v-if="rankNote"> · {{ rankNote }}</template>
          </span>
        </div>

        <div v-if="view === 'search'" class="kb-list">
          <div v-if="searching" class="kb-loading">检索中…</div>
          <div v-else-if="!hits.length" class="kb-empty-inline">
            没有命中。{{ sourcesNote }}
          </div>
          <!-- 条件改过没重搜：结果与下面所有回执（ranking/filters/覆盖率）都是上一次的，
               必须当场说清，否则人会拿旧数字给新条件下结论 -->
          <div v-if="filtersDirty" class="kb-warn-inline">
            条件已改，下面还是上一次检索的结果 —— 点「检索」重新查。
          </div>
          <div v-if="filtersWarning" class="kb-warn-inline">{{ filtersWarning }}</div>
          <div v-if="methodsEmptyNote" class="kb-warn-inline">{{ methodsEmptyNote }}</div>
          <article
            v-for="(hit, i) in hits"
            :key="`${hit.path}-${hit.page}-${i}`"
            class="kb-hit"
            @click="openDoc(hit.path)"
          >
            <div class="kb-hit-head">
              <span class="kb-hit-title">{{ hit.title }}</span>
              <span v-if="hit.source === 'playbook'" class="kb-pill kb-pill-ok">方法</span>
              <span v-else-if="hit.from === 'ocr'" class="kb-pill kb-pill-ocr">OCR</span>
              <span v-else class="kb-pill">抽取</span>
              <!-- 半成品必须逐条标覆盖率：`chunkPages` 远小于 `pages` 时，
                   这条引用只代表那几页，不代表全书 —— 不加这个数就是骗人。
                   注意 `hit.coverage` 在方法库命中上是**数字**（词覆盖率），不是这个对象。 -->
              <span
                v-if="ocrCov(hit)"
                class="kb-pill kb-pill-warn"
                :title="`这份共 ${ocrCov(hit)?.pages} 页，只有 ${ocrCov(hit)?.chunkPages} 页有正文（OCR 没跑完）`"
              >OCR {{ ocrCov(hit)?.chunkPages }}/{{ ocrCov(hit)?.pages }} 页</span>
              <!-- 方法库的 `coverage` 是排序主键（命中查询词的比例），不显示等于藏起排序理由 -->
              <span
                v-else-if="termCov(hit) !== null"
                class="kb-pill"
                title="命中了多少查询词（0–1）。方法库按它排序，不是相似度"
              >词覆盖 {{ Math.round((termCov(hit) ?? 0) * 100) }}%</span>
              <span v-if="hit.page" class="kb-hit-page">第 {{ hit.page }} 页</span>
            </div>
            <p class="kb-hit-excerpt">{{ hit.excerpt }}</p>
            <div class="kb-hit-path">{{ hit.path }}</div>
          </article>
        </div>

        <div v-else class="kb-list">
          <!-- 文件名筛选与「正文检索」是两条路：检索走 /search（bm25），这里走 /documents 的
               `q`（路径模糊匹配）。分开摆，是因为它们的口径不同，混在一个框里会让人以为
               「搜不到正文就是不在这」—— 而正文没抽出来的件照样该按文件名找到。 -->
          <div class="kb-browse-bar">
            <input
              v-model="docQ"
              class="kb-name-input"
              :placeholder="`按文件名筛选（回车生效${docQuery ? `，当前：${docQuery}` : ''}）`"
              @keyup.enter="applyDocQ"
            >
            <button class="btn btn-ghost" @click="applyDocQ">筛选</button>
            <button v-if="docQuery" class="kb-filter-clear" @click="clearDocQ">清空</button>
            <span class="kb-filters-hint">
              共 {{ docs?.total ?? 0 }} 份<template v-if="docs?.total">，本页 {{ docs.items.length }} 份</template>
            </span>
          </div>
          <div v-if="loadingDocs" class="kb-loading">读取中…</div>
          <div v-else-if="!docs?.items.length" class="kb-empty-inline">
            {{ docQuery ? `没有文件名含「${docQuery}」的文档。` : '这个目录下没有符合条件的文档。' }}
          </div>
          <div
            v-for="doc in docs?.items ?? []"
            :key="doc.path"
            class="kb-doc"
            :class="{ 'kb-doc-buried': doc.buried }"
            role="button"
            tabindex="0"
            @click="openDoc(doc.path)"
            @keydown.enter.self="openDoc(doc.path)"
            @keydown.space.self.prevent="openDoc(doc.path)"
          >
            <FileText :size="14" class="kb-doc-icon" />
            <span class="kb-doc-name">{{ doc.name }}</span>
            <span v-if="doc.buried" class="kb-pill kb-pill-bad">回收站</span>
            <span v-if="doc.extra" class="kb-pill">补录</span>
            <span v-if="doc.edited" class="kb-pill kb-pill-warn">已修改</span>
            <span v-if="doc.ocr" class="kb-pill kb-pill-ocr">OCR</span>
            <span v-else-if="doc.partial" class="kb-pill kb-pill-warn">OCR 部分</span>
            <span v-else-if="doc.withText" class="kb-pill kb-pill-ok">有正文</span>
            <span v-else-if="!doc.buried" class="kb-pill kb-pill-warn">待补</span>
            <span class="kb-doc-meta">{{ doc.pages || '—' }} 页 · {{ fmtSize(doc.size) }}</span>
            <!-- 行内只放**一个**动作：删除/还原。编辑必须先看到正文，那属于抽屉里的事。 -->
            <span class="kb-doc-act" @click.stop>
              <button
                v-if="doc.buried"
                class="kb-mini"
                title="从回收站还原"
                @click="restoreRow(doc)"
              >还原</button>
              <button
                v-else
                class="kb-mini kb-mini-danger"
                title="移入回收站（可还原）"
                @click="deleteRow(doc)"
              >删除</button>
            </span>
          </div>
          <!-- 「加载更多」换成**真分页**：一次 200 条时用户既不知道自己在第几页，
               也没法回到上一批（翻过头只能重来）。 -->
          <div v-if="docs && docs.total > PAGE_SIZE" class="kb-pager">
            <button
              class="btn btn-ghost"
              :disabled="loadingDocs || docs.offset <= 0"
              @click="gotoPage(docs.offset - PAGE_SIZE)"
            >上一页</button>
            <span class="kb-pager-info">
              第 {{ pageNo }} / {{ pageCount }} 页 · {{ docs.offset + 1 }}–{{ docs.offset + docs.items.length }} / {{ docs.total }}
            </span>
            <button
              class="btn btn-ghost"
              :disabled="loadingDocs || docs.offset + docs.items.length >= docs.total"
              @click="gotoPage(docs.offset + PAGE_SIZE)"
            >下一页</button>
          </div>
        </div>
      </main>

      <aside v-if="opened" class="kb-drawer">
        <div class="kb-drawer-head">
          <div>
            <div v-if="editingTitle" class="kb-title-edit">
              <input
                v-model="titleDraft"
                class="kb-name-input"
                placeholder="标题"
                @keyup.enter="saveTitle"
                @keyup.esc="editingTitle = false"
              >
              <button class="kb-mini" :disabled="busy" @click="saveTitle">保存</button>
              <button class="kb-mini" :disabled="busy" @click="editingTitle = false">取消</button>
            </div>
            <div v-else class="kb-drawer-title">
              {{ opened.document?.title || '文档' }}
              <button v-if="canEdit" class="kb-mini" @click="startTitleEdit">改标题</button>
            </div>
            <div class="kb-drawer-sub">
              <span v-if="opened.origin === 'playbook'" class="kb-pill kb-pill-ok">方法正文</span>
              <span v-else-if="opened.origin === 'ocr'" class="kb-pill kb-pill-ocr">OCR 正文</span>
              <span v-else class="kb-pill">供应商抽取</span>
              <span>{{ opened.total }} {{ opened.origin === 'playbook' ? '节' : '块' }}</span>
              <span v-if="opened.document?.source === 'extra'" class="kb-pill">本仓补录</span>
              <span v-if="opened.document?.edited || state?.edits" class="kb-pill kb-pill-warn">
                已修改{{ state?.edits ? ` ${state.edits} 块` : '' }}
              </span>
              <span v-if="opened.document?.buried" class="kb-pill kb-pill-bad">回收站</span>
              <span v-if="opened.document?.onDisk === false" class="kb-pill kb-pill-warn">盘上找不到</span>
            </div>
          </div>
          <button class="kb-drawer-close" @click="closeDoc"><X :size="15" /></button>
        </div>
        <!-- 动作条只画**当前真能做的事**：`hardDeletable` 只对「本仓上传且文件在盘上」成立，
             回收站里的件只留「还原」。画满按钮=让人点了才报错。 -->
        <div v-if="opened.origin !== 'playbook'" class="kb-drawer-tools">
          <template v-if="opened.document?.buried">
            <button class="btn btn-ghost" :disabled="busy" @click="restoreDoc">还原到知识库</button>
            <span class="kb-tools-hint">在回收站里：检索与清单都不会出现它</span>
          </template>
          <template v-else>
            <button class="btn btn-ghost kb-danger" :disabled="busy" @click="deleteDoc(false)">移入回收站</button>
            <button
              v-if="state?.hardDeletable"
              class="btn btn-ghost kb-danger"
              :disabled="busy"
              @click="deleteDoc(true)"
            >彻底删除</button>
            <span class="kb-tools-hint">
              {{ state?.hardDeletable
                ? '彻底删除会连盘上的文件一起删（只有自己上传的件可以）'
                : '删除只是移入回收站，随时还原；原库文件不动' }}
            </span>
          </template>
        </div>
        <!-- 正文与场次是两份数据：正文来自抽取/OCR，场次来自 A 侧派生索引（可能没跑过）。
             分两个页签，是为了**不把「没解析过」显示成「这部没有场次」。** -->
        <div class="kb-drawer-tabs">
          <button class="kb-tab" :class="{ active: drawerTab === 'text' }" @click="drawerTab = 'text'">
            正文 <span class="kb-tab-num">{{ opened.total }}</span>
          </button>
          <button class="kb-tab" :class="{ active: drawerTab === 'scenes' }" @click="drawerTab = 'scenes'">
            场次 <span class="kb-tab-num">{{ scenesLoading ? '…' : (scenes?.total ?? '—') }}</span>
          </button>
          <span v-if="scenesNote" class="kb-drawer-tabs-note">{{ scenesNote }}</span>
        </div>
        <div v-if="drawerTab === 'scenes'" class="kb-drawer-body">
          <div v-if="scenesLoading && !scenes" class="kb-loading">读场次…</div>
          <div v-else-if="!scenes" class="kb-empty-inline">场次数据未取到。</div>
          <!-- 三种回执分开说：索引没建 / 这份没解析过 / 解析过但认不出边界 -->
          <div v-else-if="!scenes.available || !scenes.document" class="kb-empty-inline">
            {{ scenes.hint || '这份还没有场次数据。' }}
          </div>
          <template v-else>
            <div class="kb-scene-meta">
              <span
                class="kb-pill"
                :class="scenes.document.status === 'ok' ? 'kb-pill-ok' : 'kb-pill-warn'"
              >{{ scenes.document.status === 'ok' ? '已切出场次' : '无场次边界' }}</span>
              <span :title="scenes.document.strategy ? `解析路线 ${scenes.document.strategy}` : ''">
                策略 {{ strategyLabel(scenes.document.strategy) }}
              </span>
              <span>{{ scenes.total }} 场</span>
              <span v-if="scenes.document.confidence != null">置信 {{ scenes.document.confidence.toFixed(2) }}</span>
            </div>
            <!-- 有场次也要交代：解析器可能补了**生产编号**、或把前言丢在首场之外 -->
            <p v-if="scenes.document.hint" class="kb-scene-hint">{{ scenes.document.hint }}</p>
            <p v-if="!scenes.scenes.length" class="kb-empty-inline">
              {{ scenes.document.hint || scenes.hint || '这份没切出场次（整篇记一场或判不出边界）。' }}
            </p>
            <article
              v-for="(sc, i) in scenes.scenes"
              :key="`${sc.sceneNo ?? 'seq'}-${sc.pageFrom ?? i}-${i}`"
              class="kb-scene"
            >
              <div class="kb-scene-head">
                <span class="kb-scene-no">{{ sc.sceneNo ?? '—' }}</span>
                <!-- 原文场次头：``1-1a日外焉州-康郡`` 的**集号**只在这里（sceneNo 只有一个数、
                     集号每集重置）⇒ 不显示它，用户就无法判断这是第几集第几场 -->
                <span v-if="sc.heading && showsSceneHeading" class="kb-scene-heading" :title="`原文场次头：${sc.heading}`">
                  {{ sc.heading }}
                </span>
                <span class="kb-scene-tags">
                  <i v-if="sc.intExt" class="kb-pill">{{ sc.intExt }}</i>
                  <i v-if="sc.location" class="kb-pill">{{ sc.location }}</i>
                  <i v-if="sc.time" class="kb-pill">{{ sc.time }}</i>
                </span>
                <span v-if="sc.pageFrom" class="kb-scene-page">
                  P{{ sc.pageFrom }}<template v-if="sc.pageTo && sc.pageTo !== sc.pageFrom">–{{ sc.pageTo }}</template>
                </span>
              </div>
              <div v-if="sc.cast.length" class="kb-scene-cast">{{ sc.cast.join('、') }}</div>
              <p v-if="sc.opening" class="kb-scene-open">{{ sc.opening }}</p>
            </article>
            <button
              v-if="scenes.page * scenes.limit < scenes.total"
              class="btn btn-ghost kb-more"
              :disabled="scenesLoading"
              @click="loadMoreScenes"
            >
              加载更多场次（已显示 {{ scenes.scenes.length }} / {{ scenes.total }}）
            </button>
          </template>
        </div>
        <div v-else class="kb-drawer-body">
          <!-- 原库重建过（块变少/变多）⇒ 老编辑的块号对不上正文。**必须当场说**：
               不说的话用户看见「已修改」标记，却找不到自己改的那句。 -->
          <div v-if="opened.staleEdits?.length" class="kb-warn-inline">
            这份的原库被重建过：块 {{ opened.staleEdits.join('、') }} 的修改已经对不上正文，
            不会再叠加显示（改动仍留在覆盖层里，需要在这几块上重新保存一次）。
          </div>
          <div v-if="!opened.chunks.length" class="kb-empty-inline">
            {{ opened.hint || '这份还没有正文（待 OCR）。' }}
          </div>
          <article
            v-for="chunk in opened.chunks"
            :key="chunk.chunkNo"
            class="kb-chunk"
            :class="{ 'kb-chunk-edited': chunk.edited }"
          >
            <div class="kb-chunk-head">
              <span class="kb-chunk-no">
                <template v-if="chunk.page == null">
                  第 {{ chunk.chunkNo }} 节<template v-if="chunk.heading"> · {{ chunk.heading }}</template>
                </template>
                <template v-else>第 {{ chunk.page }} 页 · 块 {{ chunk.chunkNo }}</template>
              </span>
              <span v-if="chunk.edited" class="kb-pill kb-pill-warn">已修改</span>
              <span class="kb-chunk-acts">
                <button v-if="canEdit && editing?.chunkNo !== chunk.chunkNo" class="kb-mini" @click="startEdit(chunk)">
                  编辑
                </button>
                <button
                  v-if="canEdit && chunk.edited"
                  class="kb-mini"
                  :disabled="busy"
                  title="丢掉这块的改动，回到原库正文"
                  @click="undoEdit(chunk)"
                >撤销改动</button>
              </span>
            </div>
            <template v-if="editing && editing.chunkNo === chunk.chunkNo">
              <textarea v-model="editing.text" class="kb-editor" rows="8" spellcheck="false" />
              <input
                v-model="editing.note"
                class="kb-editor-note"
                placeholder="改动备注（可选：为什么改、改自哪一版，写给以后的自己）"
              >
              <div class="kb-editor-acts">
                <button class="btn btn-primary" :disabled="busy || !editing.text.trim()" @click="saveEdit">
                  <Loader2 v-if="busy" :size="13" class="animate-spin" />
                  <span v-else>保存</span>
                </button>
                <button class="btn btn-ghost" :disabled="busy" @click="editing = null">取消</button>
                <span class="kb-editor-hint">保存只落覆盖层，原库文件一个字不改</span>
              </div>
            </template>
            <template v-else>
              <p class="kb-chunk-text">{{ chunk.text }}</p>
              <details v-if="chunk.edited && chunk.original" class="kb-orig">
                <summary>看原库原文</summary>
                <p class="kb-chunk-text">{{ chunk.original }}</p>
              </details>
              <p v-if="chunk.note" class="kb-chunk-note">备注：{{ chunk.note }}</p>
            </template>
          </article>
          <button
            v-if="opened.offset + opened.chunks.length < opened.total"
            class="btn btn-ghost kb-more"
            :disabled="loadingDoc"
            @click="loadMoreChunks"
          >
            加载更多片段
          </button>
        </div>
      </aside>

      <!-- 上传面：拖拽＝选择同一件事，两种入口都要有（触屏没有拖拽）。 -->
      <div v-if="uploadOpen" class="kb-modal-mask" @click.self="closeUpload">
        <div class="kb-modal" role="dialog" aria-modal="true" aria-label="上传语料">
          <div class="kb-modal-head">
            <div>
              <div class="kb-modal-title">上传语料到知识库</div>
              <div class="kb-modal-sub">
                落到 <code>{{ limits?.dir || '00-本仓补录' }}</code>；
                收 {{ limits?.label || '正在读取可接受类型…' }}；
                单份 ≤ {{ limits ? fmtSize(limits.maxBytes) : '—' }}，一次最多 {{ limits?.maxFiles ?? '—' }} 份
              </div>
            </div>
            <button class="kb-drawer-close" title="关闭" @click="closeUpload"><X :size="15" /></button>
          </div>
          <div class="kb-modal-body">
            <label
              class="kb-drop"
              :class="{ 'is-over': dragging, 'is-busy': uploading }"
              @dragover.prevent="dragging = true"
              @dragleave.prevent="dragging = false"
              @drop.prevent="onDrop"
            >
              <input
                type="file"
                multiple
                class="kb-drop-input"
                :accept="acceptAttr"
                :disabled="uploading"
                @change="onPick"
              >
              <Upload :size="22" />
              <b>{{ dragging ? '松手即可加入' : '把文件拖到这里，或点击选择' }}</b>
              <span>{{ limits?.label || '…' }}（上传后立刻可搜）</span>
            </label>

            <ul v-if="queue.length" class="kb-queue">
              <li v-for="(item, i) in queue" :key="item.key" class="kb-queue-item" :class="{ 'is-bad': item.problem }">
                <span class="kb-queue-name">{{ item.file.name }}</span>
                <span class="kb-queue-size">{{ fmtSize(item.file.size) }}</span>
                <span v-if="item.problem" class="kb-queue-problem">{{ item.problem }}</span>
                <button v-else class="kb-mini" :disabled="uploading" @click="queue.splice(i, 1)">移除</button>
              </li>
            </ul>

            <div v-if="uploading" class="kb-progress">
              <div class="kb-progress-bar"><i :style="{ width: `${uploadPct}%` }" /></div>
              <span>{{ uploadPct < 100 ? `上传中 ${uploadPct}%` : '服务器抽正文中…（大件要几十秒）' }}</span>
            </div>

            <div v-if="uploadResults.length" class="kb-results">
              <div class="kb-results-head">
                本次结果：写入 {{ uploadCount.ingested }} · 待 OCR {{ uploadCount.quarantined }} ·
                未接受 {{ uploadCount.rejected }}
              </div>
              <div
                v-for="(r, i) in uploadResults"
                :key="`${r.name}-${i}`"
                class="kb-result"
                :class="`is-${r.status}`"
              >
                <b class="kb-result-name">{{ r.name }}</b>
                <span class="kb-pill" :class="STATUS_CLASS[r.status]">{{ STATUS_LABEL[r.status] }}</span>
                <span class="kb-result-note">
                  <template v-if="r.status === 'ingested'">
                    {{ r.chunks }} 块 · {{ r.chars }} 字 · {{ fmtSize(r.size) }} · {{ (r.seconds ?? 0).toFixed(1) }}s
                  </template>
                  <template v-else>{{ r.reason || '' }}</template>
                </span>
              </div>
              <!-- 「待 OCR」与「未接受」的差别**只在盘上有没有文件**，必须说清，
                   否则用户会对着一条 quarantined 反复重传（文件其实早就进去了）。 -->
              <p v-if="uploadCount.quarantined" class="kb-results-hint">
                待 OCR 的件<b>已经在盘上</b>（清单里能看到、可打开），只是抽不出正文；走 OCR 后会被自动补进来。
              </p>
            </div>
          </div>
          <div class="kb-modal-foot">
            <span class="kb-filters-hint">{{ queueHint }}</span>
            <button class="btn btn-ghost" :disabled="uploading" @click="closeUpload">
              {{ uploadResults.length ? '完成' : '取消' }}
            </button>
            <button class="btn btn-primary" :disabled="!canUpload" @click="doUpload">
              <Loader2 v-if="uploading" :size="13" class="animate-spin" />
              <span v-else>开始上传</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { Search, X, RefreshCw, Loader2, FileText, Upload } from 'lucide-vue-next'
import { toast } from 'vue-sonner'
import {
  knowledgeAPI,
  type KnowledgeTree,
  type KnowledgeDocs,
  type KnowledgeHit,
  type KnowledgeFacets,
  type KnowledgeScenes,
  type KnowledgeDoc,
  type KnowledgeDocState,
  type KnowledgeOverlay,
  type KnowledgeUploadResult,
} from '~/composables/useApi'
import { useConfirm } from '~/composables/useConfirm'

const { confirm } = useConfirm()

const TABS = [
  { id: 'all', label: '全部' },
  { id: 'withText', label: '有正文' },
  { id: 'ocr', label: 'OCR 已补' },
  { id: 'partial', label: 'OCR 部分' },
  { id: 'pending', label: '待补' },
  //: 后两档是**覆盖层**（不在任何一份 sqlite 里）：改过的、进了回收站的。
  //: 没有这两个页签，用户改完/删完就再也找不到自己做过的那些事。
  { id: 'edited', label: '已修改' },
  { id: 'buried', label: '回收站' },
] as const

const PAGE_SIZE = 30
const PAGE = PAGE_SIZE

//: 六个目录维度 → 中文标签。**键名必须与后端 `_FILTER_KEYS` 一致**（docType 是 camelCase）
const FACET_LABELS: Array<[string, string]> = [
  ['category', '一级分类'],
  ['country', '国家地区'],
  ['genre', '类型'],
  ['docType', '资料类型'],
  ['permission', '权限'],
  ['festival', '电影节'],
]

//: 三种排序**不可比**，标签写全，别只写 bm25
const RANK_LABELS: Record<string, string> = {
  bm25: 'bm25',
  like: '短词精确',
  'like-or': '短词任一命中',
  gram: '字符 n-gram',
}

const tree = ref<KnowledgeTree | null>(null)
const docs = ref<KnowledgeDocs | null>(null)
const hits = ref<KnowledgeHit[]>([])
const sources = ref<Record<string, any>>({})
const view = ref<'browse' | 'search'>('browse')
const activeGroup = ref<string | undefined>(undefined)
const only = ref<string>('all')
const q = ref('')
const loadingTree = ref(false)
const loadingDocs = ref(false)
const searching = ref(false)
const opened = ref<any>(null)
/** 抽屉当前打开的相对路径：`/document` 回执里的 path 已归一，但场次必须按打开时那份查 */
const openedPath = ref('')
const loadingDoc = ref(false)
const playbook = ref<any>(null)

const facets = ref<KnowledgeFacets | null>(null)
const filters = reactive<Record<string, string>>({
  category: '', country: '', genre: '', docType: '', permission: '', festival: '',
})
const skillFilter = ref('')
const includePartial = ref(false)
/** 检索是否只限左侧选中的目录。**默认全库**：侧栏点目录是「浏览」动作，
    顺手点一下就悄悄把检索范围缩到某个子目录，是用户完全无法察觉的窄化。 */
const searchInGroup = ref(false)
/** 上次真正发出去的那组条件（JSON 串）。与当前条件不等 ⇒ 屏幕上是旧结果 */
const searchedSig = ref('')

const scenes = ref<KnowledgeScenes | null>(null)
const scenesLoading = ref(false)
const drawerTab = ref<'text' | 'scenes'>('text')

/* ================= 写能力（覆盖层） ================= */
/** 文件名筛选：与「正文检索」是两条路（一个走 /documents?q=，一个走 /search） */
const docQ = ref('')
const docQuery = ref('')
/** 覆盖层盘点：已改/回收站/最近上传 + 上传口径（前端校验的真源） */
const overlay = ref<KnowledgeOverlay | null>(null)
/** 当前打开的这份「能做什么」（可编辑/可删/可彻底删/在回收站） */
const state = ref<KnowledgeDocState | null>(null)
const editing = ref<{ chunkNo: number; text: string; note: string } | null>(null)
const editingTitle = ref(false)
const titleDraft = ref('')
/** 任何写操作在途 —— 一件一件来，避免「连点两下删除」这种自相矛盾的请求 */
const busy = ref(false)

const canEdit = computed(() => state.value?.editable === true)
const pageNo = computed(() =>
  docs.value ? Math.floor(docs.value.offset / PAGE_SIZE) + 1 : 1)
const pageCount = computed(() =>
  docs.value ? Math.max(1, Math.ceil(docs.value.total / PAGE_SIZE)) : 1)

async function loadOverlay() {
  try {
    overlay.value = await knowledgeAPI.overlay()
  } catch {
    //: 盘点失败不该挡住浏览/检索：徽标只是提示，不是判据
    overlay.value = null
  }
}

/** 重新读当前打开的这份（改完/撤销完/还原完都要重读，别自己在本地拼状态） */
async function reloadOpen() {
  if (!openedPath.value) return
  const limit = Math.max(20, opened.value?.chunks?.length ?? 20)
  try {
    opened.value = await knowledgeAPI.document(openedPath.value, 0, limit)
  } catch (err: any) {
    toast.error(err.message || '重新读取正文失败')
    return
  }
  try {
    state.value = await knowledgeAPI.state(openedPath.value)
  } catch {
    state.value = null
  }
}

function startEdit(chunk: any) {
  editing.value = { chunkNo: chunk.chunkNo, text: chunk.text ?? '', note: chunk.note ?? '' }
}

async function saveEdit() {
  if (!editing.value || !openedPath.value) return
  busy.value = true
  try {
    const res = await knowledgeAPI.saveChunk(
      openedPath.value, editing.value.chunkNo, editing.value.text, editing.value.note)
    if (!res?.ok) {
      //: 服务端的 `reason` 是给人看的（块号越界、块已不存在…），别用「保存失败」盖掉它
      toast.error(res?.reason || '保存失败')
      return
    }
    editing.value = null
    toast.success('已保存（改动落在覆盖层，原库文件未改动）')
    await reloadOpen()
    await loadOverlay()
    await refreshLists()
  } catch (err: any) {
    toast.error(err.message || '保存失败')
  } finally {
    busy.value = false
  }
}

async function undoEdit(chunk: any) {
  if (!openedPath.value) return
  const ok = await confirm({
    title: '撤销这块改动',
    message: '这块会回到原库正文，你改的那一版不会被保留。',
  })
  if (!ok) return
  busy.value = true
  try {
    const res: any = await knowledgeAPI.undoChunk(openedPath.value, chunk.chunkNo)
    if (res && res.ok === false) {
      toast.error(res.reason || '撤销失败')
      return
    }
    if (res && res.dropped === false) toast.info('这块本来就没有改动')
    else toast.success('已回到原库正文')
    await reloadOpen()
    await loadOverlay()
    await refreshLists()
  } catch (err: any) {
    toast.error(err.message || '撤销失败')
  } finally {
    busy.value = false
  }
}

function startTitleEdit() {
  titleDraft.value = opened.value?.document?.title || ''
  editingTitle.value = true
}

async function saveTitle() {
  if (!openedPath.value) return
  const title = titleDraft.value.trim()
  if (!title) {
    toast.error('标题不能为空')
    return
  }
  busy.value = true
  try {
    const res = await knowledgeAPI.saveChunk(openedPath.value, -1, title)
    if (!res?.ok) {
      toast.error(res?.reason || '标题保存失败')
      return
    }
    editingTitle.value = false
    await reloadOpen()
    await loadOverlay()
    await refreshLists()
    toast.success('标题已更新')
  } catch (err: any) {
    toast.error(err.message || '标题保存失败')
  } finally {
    busy.value = false
  }
}

/** 删/还原之后，清单、盘点和检索都得重算 —— 只改一处就会出现「删了但计数没变」 */
async function refreshLists() {
  await loadOverlay()
  if (view.value === 'browse') await loadDocs(docs.value?.offset ?? 0)
  else if (q.value.trim() && searchedSig.value === filterSig()) await doSearch()
  void loadTree()
}

async function buryPath(path: string, hard: boolean) {
  const res: any = await knowledgeAPI.bury(path, hard)
  if (res && res.ok === false) {
    //: 「库里原有的件不能彻底删」这类拒绝理由必须原样透出来，否则用户只会反复点
    toast.error(res.reason || '删除失败')
    return false
  }
  toast.success(hard ? '已彻底删除（文件与索引记录一起删掉）' : '已移入回收站，可随时还原')
  return true
}

async function deleteRow(doc: KnowledgeDoc) {
  const ok = await confirm({
    title: '移入回收站',
    message: `「${doc.name}」会从清单和检索里消失，随时可以还原；原库文件不会被删。`,
  })
  if (!ok) return
  busy.value = true
  try {
    if (await buryPath(doc.path, false)) {
      if (openedPath.value === doc.path) closeDoc()
      await refreshLists()
    }
  } catch (err: any) {
    toast.error(err.message || '删除失败')
  } finally {
    busy.value = false
  }
}

async function restoreRow(doc: KnowledgeDoc) {
  busy.value = true
  try {
    await knowledgeAPI.restore(doc.path)
    toast.success('已还原到知识库')
    await refreshLists()
  } catch (err: any) {
    toast.error(err.message || '还原失败')
  } finally {
    busy.value = false
  }
}

async function deleteDoc(hard: boolean) {
  if (!openedPath.value) return
  const name = opened.value?.document?.name || openedPath.value
  const ok = await confirm({
    title: hard ? '彻底删除' : '移入回收站',
    message: hard
      ? `「${name}」会连盘上的文件一起删掉，不可还原。只对本仓上传的件有效。`
      : `「${name}」会从清单和检索里消失，随时可以还原；原库文件不会被删。`,
    danger: hard,
  })
  if (!ok) return
  busy.value = true
  try {
    if (await buryPath(openedPath.value, hard)) {
      closeDoc()
      await refreshLists()
    }
  } catch (err: any) {
    toast.error(err.message || '删除失败')
  } finally {
    busy.value = false
  }
}

async function restoreDoc() {
  if (!openedPath.value) return
  busy.value = true
  try {
    await knowledgeAPI.restore(openedPath.value)
    toast.success('已还原到知识库')
    await reloadOpen()
    await refreshLists()
  } catch (err: any) {
    toast.error(err.message || '还原失败')
  } finally {
    busy.value = false
  }
}

function closeDoc() {
  opened.value = null
  openedPath.value = ''
  state.value = null
  editing.value = null
  editingTitle.value = false
  scenes.value = null
}

/* ================= 上传（补录库） ================= */
interface QueueItem { key: string; file: File; problem: string }

const uploadOpen = ref(false)
const queue = ref<QueueItem[]>([])
const uploading = ref(false)
const uploadPct = ref(0)
const uploadResults = ref<KnowledgeUploadResult[]>([])
const dragging = ref(false)

const limits = computed(() => overlay.value?.upload ?? null)
/** `accept` 由后端下发（扩展名列表），前端拼成 `<input accept>` 那种串 */
const acceptAttr = computed(() =>
  (limits.value?.accept ?? []).map((suffix) => (suffix.startsWith('.') ? suffix : `.${suffix}`)).join(','))
const uploadCount = computed(() => ({
  ingested: uploadResults.value.filter((r) => r.status === 'ingested').length,
  quarantined: uploadResults.value.filter((r) => r.status === 'quarantined').length,
  rejected: uploadResults.value.filter((r) => r.status === 'rejected').length,
}))
const canUpload = computed(() =>
  !uploading.value && queue.value.some((item) => !item.problem))
const queueHint = computed(() => {
  if (uploading.value) return '上传中…'
  const ok = queue.value.filter((item) => !item.problem).length
  if (!queue.value.length) return `还没有选文件（一次最多 ${limits.value?.maxFiles ?? '—'} 份）`
  const bad = queue.value.length - ok
  return bad ? `可上传 ${ok} 份，${bad} 份有问题（见上方原因）` : `可上传 ${ok} 份`
})

const STATUS_LABEL: Record<string, string> = {
  ingested: '已入库',
  quarantined: '待 OCR',
  rejected: '未接受',
}
const STATUS_CLASS: Record<string, string> = {
  ingested: 'kb-pill-ok',
  quarantined: 'kb-pill-warn',
  rejected: 'kb-pill-bad',
}

/** 提交前校验与后端**同一口径**（类型/大小都由 `/overlay` 下发），只是把话说在前头 */
function problemOf(file: File): string {
  const lim = limits.value
  if (!lim) return '还没读到上传口径，请稍后重试'
  const dot = file.name.lastIndexOf('.')
  const suffix = dot >= 0 ? file.name.slice(dot).toLowerCase() : ''
  if (!suffix || !lim.accept.includes(suffix)) return `类型不支持（只收 ${lim.label}）`
  if (!file.size) return '空文件（0 字节）'
  if (file.size > lim.maxBytes) return `太大（${fmtSize(file.size)}，上限 ${fmtSize(lim.maxBytes)}）`
  return ''
}

function addFiles(files: File[]) {
  const lim = limits.value
  if (!lim || !files.length) return
  const seen = new Set(queue.value.map((item) => `${item.file.name}:${item.file.size}:${item.file.lastModified}`))
  let room = lim.maxFiles - queue.value.length
  let skipped = 0
  for (const file of files) {
    const key = `${file.name}:${file.size}:${file.lastModified}`
    if (seen.has(key)) continue
    if (room <= 0) { skipped += 1; continue }
    seen.add(key)
    room -= 1
    queue.value.push({ key, file, problem: problemOf(file) })
  }
  if (skipped) toast.error(`一次最多 ${lim.maxFiles} 份，多出来的 ${skipped} 份没有加入`)
}

function onPick(ev: Event) {
  const input = ev.target as HTMLInputElement
  addFiles(Array.from(input.files ?? []))
  //: 清空 input，否则「同一个文件再选一次」不会触发 change
  input.value = ''
}

function onDrop(ev: DragEvent) {
  dragging.value = false
  if (uploading.value) return
  addFiles(Array.from(ev.dataTransfer?.files ?? []))
}

async function openUpload() {
  uploadOpen.value = true
  uploadResults.value = []
  uploadPct.value = 0
  if (!overlay.value) await loadOverlay()
}

function closeUpload() {
  if (uploading.value) return
  uploadOpen.value = false
  queue.value = []
  uploadResults.value = []
  uploadPct.value = 0
}

async function doUpload() {
  const files = queue.value.filter((item) => !item.problem).map((item) => item.file)
  if (!files.length) return
  uploading.value = true
  uploadPct.value = 0
  uploadResults.value = []
  try {
    const res = await knowledgeAPI.upload(files, (pct) => { uploadPct.value = pct })
    uploadResults.value = res.results ?? []
    if (res.ingested) toast.success(`已入库 ${res.ingested} 份，现在就能搜到`)
    if (res.quarantined) toast.info(`${res.quarantined} 份文件已在盘上，但抽不出正文（待 OCR）`)
    if (res.rejected) toast.error(`${res.rejected} 份未被接受，原因逐条见列表`)
    //: 传成功的从队列里拿走，失败/被拒的留着让人看清是哪一份 —— 一键清空会抹掉证据
    const done = new Set(uploadResults.value.filter((r) => r.status !== 'rejected').map((r) => r.name))
    queue.value = queue.value.filter((item) => !done.has(item.file.name))
    await loadOverlay()
    await refreshLists()
  } catch (err: any) {
    toast.error(err.message || '上传失败')
  } finally {
    uploading.value = false
    uploadPct.value = 0
  }
}

const totals = computed(() => tree.value?.totals ?? null)

//: 后端的 `playbookSkills` 是 `{技能名: 小节数}` —— 转成数组再渲染，别对对象直接 v-for
const playbookSkills = computed(() =>
  Object.entries(facets.value?.playbookSkills ?? {}).map(([name, sections]) => ({ name, sections })))

/** 只列**真有多个取值**的维度：单选一个值的下拉没有意义，只是噪音 */
const facetRows = computed(() => {
  const dims = facets.value?.dimensions ?? {}
  return FACET_LABELS
    .map(([key, label]) => ({
      key,
      label,
      options: Object.entries(dims[key] ?? {})
        .map(([value, count]) => ({ value, count }))
        .sort((a, b) => b.count - a.count || a.value.localeCompare(b.value)),
    }))
    .filter((row) => row.options.length > 1)
})

const filtersActive = computed(() =>
  Object.values(filters).some(Boolean) || !!skillFilter.value || includePartial.value
  || (searchInGroup.value && !!activeGroup.value))

/** 当前条件快照。只用来判「屏幕上的结果是不是当前条件的」，键序固定故可直接比串 */
function filterSig(): string {
  return JSON.stringify({
    f: { ...filters },
    skill: skillFilter.value,
    partial: includePartial.value,
    group: searchInGroup.value ? (activeGroup.value ?? '') : '',
  })
}

/** 条件改过但还没重搜 ⇒ 结果与回执全都是旧的，不该拿它下判断。
    检索进行中不算「脏」：此刻屏上的东西正要被替换，再提示「点检索」纯属噪音。 */
const filtersDirty = computed(() =>
  view.value === 'search' && !searching.value && searchedSig.value !== filterSig())

/** 显式勾了「只搜当前目录」才缩范围 —— 范围必须写在结果行首，否则「怎么少这么多」无从解释 */
const scopeNote = computed(() =>
  (searchInGroup.value && activeGroup.value ? `限「${activeGroup.value}」` : ''))

const rankNote = computed(() => {
  const bits: string[] = []
  const m = sources.value?.methods
  const p = sources.value?.playbook
  if (m?.ranking) bits.push(`剧本库 ${RANK_LABELS[m.ranking] ?? m.ranking}`)
  if (p?.ranking) bits.push(`方法库 ${RANK_LABELS[p.ranking] ?? p.ranking}`)
  return bits.join(' / ')
})

/** 目录维度**只筛剧本库**：条件给了却没生效时必须说出来，不能让人以为筛过了。
    ⚠️ 判据不能用 `sources.methods.filtersApplied`：那个位在「只给了 skill」时也会被置 false
    （意思是「skill 不适用于剧本库」），与「目录条件没生效」是两件事；两个条件同时给时
    它会恒定 false ⇒ 假警报。真凭据是 `filters.applied`。 */
const filtersWarning = computed(() => {
  if (filtersDirty.value || searching.value) return ''
  const f = sources.value?.methods?.filters
  if (!f || f.applied !== false) return ''
  if (!Object.values(filters).some(Boolean)) return ''
  return f.hint
    || '目录条件没生效：剧本库这一源没被过滤（目录 CSV 缺失或不认这些值），结果里混着别的作品。'
})

/** 目录侧筛到 0 条、但方法库照常出结果 —— 必须说明「下面这些不是筛出来的」，
    否则人看到「筛了国家却冒出一堆方法库段落」只会以为筛选坏了。 */
const methodsEmptyNote = computed(() => {
  if (filtersDirty.value || searching.value) return ''
  const m = sources.value?.methods
  if (!m || m.filters?.applied !== true) return ''
  if ((m.matched ?? 0) > 0 || !hits.value.length) return ''
  const pb = hits.value.filter((h) => h.source === 'playbook').length
  if (!pb) return ''
  return `剧本库在这些目录条件下 0 条；下面 ${pb} 条是方法库命中（方法库没有国家/类型/奖项维度，不受目录条件影响）。`
})

/* `coverage` 这个字段名在两条源上是**两种东西**，不区分就会把数字渲染成
   「OCR undefined/undefined 页」：
     · 剧本库命中 → `{ pages, chunkPages }`（有正文页数 / 总页数，只在含半成品时给）
     · 方法库命中 → `0–1 的数字`（命中了多少查询词，方法库就按它排序） */
function ocrCov(hit: KnowledgeHit): { pages: number; chunkPages: number } | null {
  const c = hit.coverage
  return c && typeof c === 'object' && typeof c.pages === 'number' ? c : null
}
function termCov(hit: KnowledgeHit): number | null {
  return typeof hit.coverage === 'number' ? hit.coverage : null
}

/* 解析路线必须讲人话：`ep-scene`、`place-time` 这类名字是给代码看的，
   直接渲染等于让用户去猜。原串保留在 title 里，便于报缺陷时对号。 */
const STRATEGY_LABEL: Record<string, string> = {
  'md-heading': 'Markdown 标题',
  'cn-numbered': '中文章节号',
  'ep-scene': '集-场号',
  'hash-scene': 'S# 场号',
  'inline': '行内场次头',
  'slug': '文件名（一份一场）',
  'place-time': '地点+时段',
  'none': '未认出边界',
}
function strategyLabel(strategy?: string | null): string {
  if (!strategy) return '—'
  return STRATEGY_LABEL[strategy] ?? strategy
}

/* 只在「原文标记提供结构化字段推不出来的信息」时才贴原文场次头：
   `ep-scene` 的**集号**（schema 没有 episode 列，集号只活在原串里）与 `hash-scene` 的
   `S#N`（补的生产编号可能与它对不上）。其余路线 pills 已能完整还原，再贴一遍是噪音。 */
const showsSceneHeading = computed(() =>
  ['ep-scene', 'hash-scene'].includes(scenes.value?.document?.strategy ?? ''))

const partialHint = 'OCR 半成品＝只扫了前几页。默认不进检索；勾上会把它们带出来，并在每条上标出「有正文页数 / 总页数」。'

/** 页签上的小字：**只**标「解析过但没认出边界」这类需要解释的情况；
    不可用时正文里已写清原因（索引没建 / 没解析过 / 方法库无场次），不必在这儿缩写。 */
const scenesNote = computed(() => {
  const sc = scenes.value
  if (!sc || !sc.available) return ''
  return sc.document?.status === 'ok' ? '' : '认不出场次边界'
})

const sourcesNote = computed(() => {
  const bits: string[] = []
  const parts = sources.value?.methods?.parts
  if (parts) {
    const labels: Record<string, string> = { ocr: '本仓 OCR', curated: '供应商库' }
    const detail = Object.entries(parts)
      .map(([name, val]: [string, any]) => `${labels[name] ?? name} ${val?.matched ?? 0} 条`)
      .join(' / ')
    if (detail) bits.push(`剧本库：${detail}`)
  }
  //: 方法库与剧本库是两个源，命中数要分开说 —— 合成一个数就分不清「一共 300 段」
  //: 和「这个源就 3 段」。
  const pb = sources.value?.playbook
  if (pb?.available) bits.push(`方法库：${pb.matched ?? pb.count ?? 0} 条`)
  return bits.length ? `${bits.join('；')}。` : ''
})

const playbookNote = computed(() => {
  const pb = playbook.value
  if (!pb) return '方法库未读取'
  if (pb.available === false) return pb.hint || '方法库不可用'
  const skills = Object.keys(pb.skills ?? {})
  return `方法库 ${pb.files ?? 0} 份 / ${pb.sections ?? 0} 节；技能：${skills.join('、') || '（空）'}`
})

/** 人话体积：原来的实现只有 MB 一档，**每份 txt 都显示「0 MB」**，等于没显示 */
function fmtSize(bytes: number) {
  if (!bytes) return '0 B'
  if (bytes < 1024) return `${bytes} B`
  const kb = bytes / 1024
  if (kb < 1024) return `${kb.toFixed(0)} KB`
  const mb = kb / 1024
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(1)} MB`
}

async function loadTree() {
  loadingTree.value = true
  try {
    //: 方法库与剧本库是两个独立源：方法库缺了不该挡住剧本库，反之亦然 ⇒ 分开取，
    //: 方法库这一支失败只落到 ref 上（页头显示「方法 0」而不是整页报错）。
    void knowledgeAPI.report()
      .then((res) => { playbook.value = res?.playbook ?? null })
      .catch(() => { playbook.value = null })
    void loadFacets()
    void loadOverlay()
    tree.value = await knowledgeAPI.tree(2)
    if (!activeGroup.value) await loadDocs(0)
  } catch (err: any) {
    toast.error(err.message || '知识库盘点失败')
  } finally {
    loadingTree.value = false
  }
}

/** 取一页清单。`offset` 是真偏移（服务端算 `total`/`offset`），换页＝重新取一页，不追加。 */
async function loadDocs(offset = 0) {
  view.value = 'browse'
  loadingDocs.value = true
  try {
    docs.value = await knowledgeAPI.documents({
      category: activeGroup.value,
      only: only.value,
      q: docQuery.value || undefined,
      offset,
      limit: PAGE_SIZE,
    })
  } catch (err: any) {
    toast.error(err.message || '目录读取失败')
  } finally {
    loadingDocs.value = false
  }
}

function gotoPage(offset: number) {
  const max = Math.max(0, (docs.value?.total ?? 0) - 1)
  void loadDocs(Math.min(Math.max(0, offset), max - (max % PAGE_SIZE)))
}

function applyDocQ() {
  docQuery.value = docQ.value.trim()
  docs.value = null
  void loadDocs(0)
}

function clearDocQ() {
  docQ.value = ''
  applyDocQ()
}

function pickGroup(path?: string) {
  activeGroup.value = path
  //: 回到「全部」时把「只搜当前目录」一并取消：勾选框本身会隐藏，留着 true 就是看不见的残留状态
  if (!path) searchInGroup.value = false
  q.value = ''
  hits.value = []
  docs.value = null
  void loadDocs(0)
}

function pickOnly(id: string) {
  only.value = id
  q.value = ''
  hits.value = []
  docs.value = null
  void loadDocs(0)
}

/** 页签上的小徽标：只标「下一档有多少」这种**用户需要预判点击后果**的数字 */
function tabBadge(id: string): number | string {
  if (id === 'pending') return totals.value?.pending ?? 0
  if (id === 'edited') return overlay.value?.edits ?? 0
  if (id === 'buried') return overlay.value?.tombstones ?? 0
  return 0
}

async function doSearch() {
  const keyword = q.value.trim()
  if (!keyword) return
  searching.value = true
  view.value = 'search'
  const sig = filterSig()
  try {
    const res = await knowledgeAPI.search({
      q: keyword,
      limit: 10,
      //: 只有显式勾了「只搜当前目录」才缩范围；默认全库
      path: searchInGroup.value ? activeGroup.value : undefined,
      country: filters.country || undefined,
      genre: filters.genre || undefined,
      category: filters.category || undefined,
      docType: filters.docType || undefined,
      permission: filters.permission || undefined,
      festival: filters.festival || undefined,
      skill: skillFilter.value || undefined,
      includePartial: includePartial.value,
    })
    hits.value = res.results ?? []
    sources.value = res.sources ?? {}
    searchedSig.value = sig
  } catch (err: any) {
    toast.error(err.message || '检索失败')
    hits.value = []
    //: 失败时把旧回执一并清掉：留着它，页面就会拿上一次的 ranking/filters 去解释这一次的空气
    sources.value = {}
    searchedSig.value = sig
  } finally {
    searching.value = false
  }
}

async function loadFacets() {
  try {
    facets.value = await knowledgeAPI.facets()
  } catch {
    //: 目录 CSV 缺了就没有维度可筛 —— 检索本身照旧可用，不当整页错误
    facets.value = null
  }
}

function resetFilters() {
  for (const key of Object.keys(filters)) filters[key] = ''
  skillFilter.value = ''
  includePartial.value = false
  searchInGroup.value = false
}

function clearSearch() {
  q.value = ''
  hits.value = []
  view.value = 'browse'
  if (!docs.value) void loadDocs()
}

async function openDoc(path: string) {
  loadingDoc.value = true
  openedPath.value = path
  scenes.value = null
  state.value = null
  editing.value = null
  editingTitle.value = false
  drawerTab.value = 'text'
  try {
    opened.value = await knowledgeAPI.document(path, 0, 20)
  } catch (err: any) {
    toast.error(err.message || '打开文档失败')
  } finally {
    loadingDoc.value = false
  }
  //: 「按钮该不该亮」单独问一次：正文回执不区分「这份能不能彻底删」，
  //: 靠前端猜会出现「点了彻底删除才发现只能进回收站」。
  if (opened.value?.origin !== 'playbook') {
    try {
      state.value = await knowledgeAPI.state(path)
    } catch {
      state.value = null
    }
  }
  if (opened.value?.origin === 'playbook') {
    //: 方法库是 md 小节，场次索引只扫剧本库 ⇒ 服务端会回 `available=false` + 「这份还没解析过
    //: （属按需生成）」——**这句对方法库是假的**（永远不会去解析它），所以这一侧直接自己交代，不发请求。
    scenes.value = {
      available: false,
      hint: '方法库是方法小节，没有场次表（场次只针对剧本库的剧本正文）。',
      document: null, scenes: [], total: 0, page: 1, limit: 200,
    }
  } else {
    //: 场次是独立派生数据，**不挡正文**：读不到就只是页签上少个数，正文照看
    void loadScenes(path, 1)
  }
}

async function loadScenes(path: string, page: number) {
  if (!path) return
  scenesLoading.value = true
  try {
    const res = await knowledgeAPI.scenes(path, page, 200)
    scenes.value = page > 1 && scenes.value
      ? { ...res, scenes: [...scenes.value.scenes, ...res.scenes] }
      : res
  } catch (err: any) {
    //: 三种回执里再补一种「读失败」，仍然走 available=false + hint，不伪装成「这部 0 场」
    scenes.value = {
      available: false,
      hint: err?.message ? `场次读取失败：${err.message}` : '场次读取失败（不影响正文）。',
      document: null,
      scenes: [],
      total: 0,
      page: 1,
      limit: 200,
    }
  } finally {
    scenesLoading.value = false
  }
}

function loadMoreScenes() {
  if (scenes.value) void loadScenes(openedPath.value, scenes.value.page + 1)
}

async function loadMoreChunks() {
  if (!opened.value) return
  loadingDoc.value = true
  try {
    const offset = opened.value.offset + opened.value.chunks.length
    const next = await knowledgeAPI.document(opened.value.document?.path ?? '', offset, 20)
    opened.value = { ...next, chunks: [...opened.value.chunks, ...next.chunks] }
  } catch (err: any) {
    toast.error(err.message || '读取片段失败')
  } finally {
    loadingDoc.value = false
  }
}

function reload() {
  void loadTree()
}

onMounted(loadTree)
</script>

<style scoped>
.kb-page { display: flex; flex-direction: column; height: 100%; min-height: 0; }

.kb-head {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 24px;
  padding: 28px 40px 18px; border-bottom: 1px solid var(--border);
}
.kb-title { font-family: var(--font-display); font-size: 22px; font-weight: 700; }
.kb-desc { margin-top: 6px; max-width: 720px; font-size: 12.5px; line-height: 1.7; color: var(--text-2); }
.kb-stats { display: flex; align-items: center; gap: 18px; flex-shrink: 0; }
.kb-stat { display: flex; flex-direction: column; align-items: flex-end; }
.kb-stat b { font-size: 17px; font-weight: 700; color: var(--text-0); font-variant-numeric: tabular-nums; }
.kb-stat span { font-size: 10.5px; color: var(--text-3); }
.kb-stat-ok b { color: #34d399; }
.kb-stat-ocr b { color: #60a5fa; }
.kb-stat-warn b { color: #fbbf24; }

.kb-body { flex: 1; min-height: 0; display: grid; grid-template-columns: 232px minmax(0, 1fr); }

.kb-side {
  border-right: 1px solid var(--border); padding: 12px 10px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 2px;
}
.kb-side-head {
  font-size: 10px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--text-3); padding: 4px 8px 8px;
}
.kb-group {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
  padding: 8px 9px; border-radius: 8px; border: 1px solid transparent;
  background: none; cursor: pointer; text-align: left; color: var(--text-1);
}
.kb-group:hover { background: var(--bg-surface); }
.kb-group.active { background: var(--bg-surface); border-color: var(--accent); color: var(--text-0); }
.kb-group-name { flex: 1; font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kb-group-num { font-size: 11px; color: var(--text-3); font-variant-numeric: tabular-nums; }
.kb-group-badges { flex-basis: 100%; display: flex; gap: 5px; padding-left: 1px; }

.kb-main { min-width: 0; display: flex; flex-direction: column; }
.kb-toolbar { display: flex; flex-wrap: wrap; gap: 10px; padding: 16px 24px 0; }
.kb-upload-btn { flex-shrink: 0; }
.kb-search {
  flex: 1; display: flex; align-items: center; gap: 8px; padding: 0 12px;
  border: 1px solid var(--border); border-radius: 10px; background: var(--bg-surface);
}
.kb-search-icon { color: var(--text-3); flex-shrink: 0; }
.kb-search-input {
  flex: 1; height: 38px; border: none; background: none; outline: none;
  color: var(--text-0); font-size: 13px;
}
.kb-search-clear { border: none; background: none; color: var(--text-3); cursor: pointer; display: flex; }

.kb-filters {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 24px 0;
}
.kb-filter {
  height: 30px; max-width: 200px; padding: 0 8px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-surface);
  color: var(--text-1); font-size: 11.5px; cursor: pointer;
}
.kb-filter.on { border-color: var(--accent); color: var(--text-0); }
.kb-check {
  display: flex; align-items: center; gap: 5px; font-size: 11.5px; color: var(--text-2);
  cursor: pointer; user-select: none;
}
.kb-check input { accent-color: var(--accent); cursor: pointer; }
.kb-filter-clear {
  border: none; background: none; color: var(--text-3); font-size: 11.5px;
  cursor: pointer; text-decoration: underline; text-underline-offset: 3px;
}
.kb-filter-clear:hover { color: var(--text-1); }
.kb-filters-hint { font-size: 10.5px; color: var(--text-3); opacity: 0.8; }
.kb-warn-inline {
  margin: 2px 0 10px; padding: 7px 11px; border-radius: 8px; font-size: 11.5px; line-height: 1.6;
  background: rgba(251, 191, 36, 0.1); color: #fbbf24;
  border: 1px solid rgba(251, 191, 36, 0.28);
}

.kb-tabs { display: flex; align-items: center; gap: 4px; padding: 14px 24px 6px; }
.kb-tab {
  display: flex; align-items: center; gap: 6px; padding: 5px 11px; border-radius: 7px;
  border: 1px solid transparent; background: none; cursor: pointer;
  font-size: 12px; color: var(--text-2);
}
.kb-tab:hover { color: var(--text-0); }
.kb-tab.active { background: var(--bg-surface); border-color: var(--border); color: var(--text-0); }
.kb-tab-badge {
  font-size: 10px; padding: 1px 5px; border-radius: 999px;
  background: rgba(251, 191, 36, 0.16); color: #fbbf24;
}
.kb-tabs-note { margin-left: auto; font-size: 11px; color: var(--text-3); }

.kb-list { flex: 1; min-height: 0; overflow-y: auto; padding: 8px 24px 28px; }

.kb-doc {
  width: 100%; display: flex; align-items: center; gap: 9px;
  padding: 10px 12px; margin-bottom: 4px; border-radius: 9px;
  border: 1px solid transparent; background: none; cursor: pointer; text-align: left;
  color: var(--text-1);
}
.kb-doc:hover { background: var(--bg-surface); border-color: var(--border); }
.kb-doc-icon { color: var(--text-3); flex-shrink: 0; }
.kb-doc-name { flex: 1; font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kb-doc-meta { font-size: 11px; color: var(--text-3); flex-shrink: 0; }

.kb-hit {
  padding: 12px 14px; margin-bottom: 8px; border-radius: 10px; cursor: pointer;
  border: 1px solid var(--border); background: var(--bg-surface);
}
.kb-hit:hover { border-color: var(--accent); }
.kb-hit-head { display: flex; align-items: center; gap: 8px; }
.kb-hit-title { flex: 1; font-size: 13px; font-weight: 600; color: var(--text-0); }
.kb-hit-page { font-size: 11px; color: var(--text-3); }
.kb-hit-excerpt { margin: 7px 0 0; font-size: 12px; line-height: 1.65; color: var(--text-1); }
.kb-hit-path { margin-top: 6px; font-size: 10.5px; color: var(--text-3); word-break: break-all; }

.kb-pill {
  display: inline-block; padding: 1px 6px; border-radius: 999px;
  font-size: 10px; font-style: normal; background: rgba(148, 163, 184, 0.16); color: var(--text-2);
}
.kb-pill-ocr { background: rgba(96, 165, 250, 0.16); color: #60a5fa; }
.kb-pill-ok { background: rgba(52, 211, 153, 0.16); color: #34d399; }
.kb-pill-warn { background: rgba(251, 191, 36, 0.16); color: #fbbf24; }
.kb-pill-bad { background: rgba(248, 113, 113, 0.16); color: #f87171; }

/* ============================ 写能力（列表行 / 抽屉 / 弹层） ============================ */
.kb-mini {
  padding: 2px 7px; border-radius: 6px; font-size: 10.5px; cursor: pointer;
  border: 1px solid var(--border); background: none; color: var(--text-2);
  white-space: nowrap;
}
.kb-mini:hover:not(:disabled) { border-color: var(--accent); color: var(--text-0); }
.kb-mini:disabled { opacity: 0.45; cursor: not-allowed; }
.kb-mini-danger:hover:not(:disabled) { border-color: #f87171; color: #f87171; }

.kb-browse-bar {
  display: flex; align-items: center; gap: 8px; padding: 2px 0 12px;
  border-bottom: 1px solid var(--border); margin-bottom: 10px;
}
.kb-name-input {
  flex: 1; min-width: 0; height: 30px; padding: 0 9px;
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-surface); color: var(--text-0); font-size: 12px; outline: none;
}
.kb-name-input:focus { border-color: var(--accent); }

.kb-doc { cursor: pointer; }
.kb-doc:focus-visible { outline: 1px solid var(--accent); outline-offset: -1px; }
.kb-doc-buried .kb-doc-name { color: var(--text-3); text-decoration: line-through; }
.kb-doc-act { flex-shrink: 0; display: flex; }
/* 默认半隐：不抢正文的视觉重量；悬停/聚焦到这一行才显形（触屏无悬停，故不做 display:none） */
.kb-doc-act .kb-mini { opacity: 0.55; transition: opacity 0.12s; }
.kb-doc:hover .kb-doc-act .kb-mini,
.kb-doc:focus-within .kb-doc-act .kb-mini { opacity: 1; }

.kb-pager {
  display: flex; align-items: center; justify-content: flex-end; gap: 10px;
  padding: 12px 0 4px;
}
.kb-pager-info { font-size: 11px; color: var(--text-3); font-variant-numeric: tabular-nums; }

.kb-title-edit { display: flex; align-items: center; gap: 6px; }

.kb-drawer-tools {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  padding: 10px 18px; border-bottom: 1px solid var(--border);
}
.kb-danger:hover:not(:disabled) { border-color: #f87171; color: #f87171; }
.kb-tools-hint { flex-basis: 100%; font-size: 10.5px; line-height: 1.6; color: var(--text-3); }

.kb-chunk-head { display: flex; align-items: baseline; gap: 8px; }
.kb-chunk-no { flex: 1; min-width: 0; }
.kb-chunk-acts { display: flex; gap: 5px; flex-shrink: 0; }
.kb-chunk-edited { border-left: 2px solid rgba(251, 191, 36, 0.55); padding-left: 10px; }
.kb-chunk-note { margin: 6px 0 0; font-size: 10.5px; line-height: 1.6; color: var(--text-3); }
.kb-orig { margin-top: 6px; }
.kb-orig summary {
  font-size: 10.5px; color: var(--text-3); cursor: pointer;
  text-decoration: underline; text-underline-offset: 3px;
}
.kb-orig .kb-chunk-text { margin-top: 6px; color: var(--text-3); }

.kb-editor {
  width: 100%; padding: 9px 11px; border-radius: 9px; resize: vertical;
  border: 1px solid var(--border); background: var(--bg-app);
  color: var(--text-0); font-size: 12.5px; line-height: 1.75;
  font-family: inherit; outline: none;
}
.kb-editor:focus { border-color: var(--accent); }
.kb-editor-note {
  width: 100%; margin-top: 6px; height: 30px; padding: 0 9px;
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-app); color: var(--text-1); font-size: 11.5px; outline: none;
}
.kb-editor-acts { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 8px; }
.kb-editor-hint { font-size: 10.5px; color: var(--text-3); }

.kb-modal-mask {
  position: fixed; inset: 0; z-index: 60; display: flex; align-items: center; justify-content: center;
  padding: 24px; background: rgba(0, 0, 0, 0.55);
}
.kb-modal {
  width: min(560px, 100%); max-height: calc(100vh - 48px); max-height: calc(100dvh - 48px);
  display: flex; flex-direction: column; border-radius: 14px;
  background: var(--bg-surface); border: 1px solid var(--border);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.42);
}
.kb-modal-head {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
  padding: 16px 18px; border-bottom: 1px solid var(--border);
}
.kb-modal-title { font-size: 14px; font-weight: 600; color: var(--text-0); }
.kb-modal-sub {
  margin-top: 6px; font-size: 11px; line-height: 1.7; color: var(--text-3);
  overflow-wrap: anywhere;
}
.kb-modal-sub code {
  padding: 0 4px; border-radius: 4px; background: rgba(148, 163, 184, 0.14); font-size: 10.5px;
}
.kb-modal-body { flex: 1; min-height: 0; overflow-y: auto; padding: 16px 18px; }
.kb-modal-foot {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  padding: 12px 18px; border-top: 1px solid var(--border);
}
.kb-modal-foot .kb-filters-hint { flex: 1; min-width: 0; }

.kb-drop {
  position: relative; display: flex; flex-direction: column; align-items: center; gap: 6px;
  padding: 26px 16px; border-radius: 12px; cursor: pointer; text-align: center;
  border: 1px dashed var(--border); background: var(--bg-app); color: var(--text-2);
}
.kb-drop.is-over { border-color: var(--accent); background: rgba(56, 189, 248, 0.06); }
.kb-drop.is-busy { opacity: 0.6; cursor: progress; }
.kb-drop b { font-size: 12.5px; color: var(--text-0); font-weight: 600; }
.kb-drop span { font-size: 11px; color: var(--text-3); }
/* 输入框铺满整块而不是 display:none：后者在部分浏览器里会让 label 点击失效 */
.kb-drop-input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }

.kb-queue { list-style: none; margin: 12px 0 0; padding: 0; display: flex; flex-direction: column; gap: 5px; }
.kb-queue-item {
  display: flex; align-items: center; gap: 8px; padding: 7px 9px; border-radius: 8px;
  border: 1px solid var(--border); font-size: 11.5px;
}
.kb-queue-item.is-bad { border-color: rgba(248, 113, 113, 0.4); }
.kb-queue-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-1); }
.kb-queue-size { flex-shrink: 0; font-size: 10.5px; color: var(--text-3); font-variant-numeric: tabular-nums; }
.kb-queue-problem { flex-shrink: 0; font-size: 10.5px; color: #f87171; }

.kb-progress { display: flex; align-items: center; gap: 10px; margin-top: 12px; font-size: 11px; color: var(--text-2); }
.kb-progress-bar {
  flex: 1; height: 6px; border-radius: 999px; overflow: hidden;
  background: rgba(148, 163, 184, 0.18);
}
.kb-progress-bar i { display: block; height: 100%; background: var(--accent); transition: width 0.2s; }

.kb-results { margin-top: 14px; }
.kb-results-head { font-size: 11px; color: var(--text-2); margin-bottom: 8px; }
.kb-result {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  padding: 7px 9px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 5px;
}
.kb-result.is-rejected { border-color: rgba(248, 113, 113, 0.4); }
.kb-result-name { flex: 1; min-width: 0; font-size: 11.5px; font-weight: 600; color: var(--text-0); overflow-wrap: anywhere; }
.kb-result-note { flex-basis: 100%; font-size: 10.5px; line-height: 1.6; color: var(--text-3); }
.kb-results-hint { margin: 8px 0 0; font-size: 10.5px; line-height: 1.7; color: var(--text-3); }

.kb-drawer {
  position: fixed; top: 0; right: 0; bottom: 0; width: 460px; max-width: 46vw; z-index: 30;
  display: flex; flex-direction: column;
  background: var(--bg-surface); border-left: 1px solid var(--border);
  box-shadow: -20px 0 40px rgba(0, 0, 0, 0.28);
}
.kb-drawer-head {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
  padding: 16px 18px; border-bottom: 1px solid var(--border);
}
.kb-drawer-title { font-size: 13.5px; font-weight: 600; color: var(--text-0); word-break: break-all; }
.kb-drawer-sub { display: flex; align-items: center; gap: 8px; margin-top: 6px; font-size: 11px; color: var(--text-3); }
.kb-drawer-close { border: none; background: none; color: var(--text-3); cursor: pointer; display: flex; }
.kb-drawer-tabs {
  display: flex; align-items: center; gap: 4px; padding: 8px 14px;
  border-bottom: 1px solid var(--border);
}
.kb-tab-num { font-size: 10px; color: var(--text-3); font-variant-numeric: tabular-nums; }
.kb-drawer-tabs-note { margin-left: auto; font-size: 10.5px; color: #fbbf24; }
.kb-drawer-body { flex: 1; min-height: 0; overflow-y: auto; padding: 14px 18px 28px; }

.kb-scene-meta {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  font-size: 11px; color: var(--text-3); margin-bottom: 12px;
}
.kb-scene-hint {
  margin: 0 0 10px; font-size: 11px; line-height: 1.65; color: var(--text-3);
}
.kb-scene {
  padding: 10px 12px; margin-bottom: 8px; border-radius: 9px;
  border: 1px solid var(--border); background: var(--bg-surface);
}
.kb-scene-head { display: flex; align-items: center; gap: 8px; }
.kb-scene-no {
  min-width: 24px; font-size: 12px; font-weight: 700; color: var(--text-0);
  font-variant-numeric: tabular-nums;
}
/* 原文场次头（含集号）：窄屏要能折行，别把右侧标签挤出卡片 */
.kb-scene-heading {
  min-width: 0; max-width: 46%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-size: 10.5px; color: var(--text-3);
  font-variant-numeric: tabular-nums;
}
.kb-scene-tags { flex: 1; display: flex; flex-wrap: wrap; gap: 5px; }
.kb-scene-page { font-size: 10.5px; color: var(--text-3); flex-shrink: 0; }
.kb-scene-cast { margin-top: 6px; font-size: 11px; color: var(--text-2); }
.kb-scene-open {
  margin: 6px 0 0; font-size: 11.5px; line-height: 1.7; color: var(--text-1);
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.kb-chunk { margin-bottom: 12px; }
.kb-chunk-head { font-size: 10.5px; color: var(--text-3); margin-bottom: 4px; }
.kb-chunk-text {
  margin: 0; font-size: 12.5px; line-height: 1.75; color: var(--text-1);
  white-space: pre-wrap; word-break: break-word;
}

.kb-loading, .kb-empty-inline { padding: 18px 4px; font-size: 12.5px; color: var(--text-3); }
.kb-more { width: 100%; margin-top: 10px; }
.kb-empty { padding: 40px; font-size: 13px; color: var(--text-2); }
.kb-hint { margin-top: 8px; font-size: 12px; color: var(--text-3); }

/* ==========================================================================
 平板 / 手机：232px 目录树在 768 以下会把结果区压到 400px 出头，
 改为「顶部横向目录条 + 结果占满」；右侧抽屉在手机上铺满整屏。
 ========================================================================== */
@media (max-width: 1023px) {
.kb-head { padding: 20px 22px 14px; gap: 16px; }
.kb-body { grid-template-columns: 184px minmax(0, 1fr); }
.kb-stats { gap: 12px; }
}

@media (max-width: 767px) {
.kb-head { flex-direction: column; align-items: stretch; gap: 12px; padding: 16px 14px 12px; }
.kb-title { font-size: 18px; }
.kb-desc { font-size: 12px; }
.kb-stats { flex-wrap: wrap; gap: 12px; }
.kb-stat { align-items: flex-start; }
.kb-stat b { font-size: 15px; }
.kb-body { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr); }
.kb-side {
  flex-direction: row; align-items: center; gap: 6px;
  border-right: none; border-bottom: 1px solid var(--border);
  padding: 8px 10px; overflow-x: auto; overflow-y: hidden;
  flex-shrink: 0;
}
.kb-side::-webkit-scrollbar { display: none; }
.kb-side-head { display: none; }
.kb-group { flex-shrink: 0; padding: 7px 10px; white-space: nowrap; }
.kb-group-name { flex: 0 0 auto; max-width: 44vw; }
.kb-group-badges { flex-basis: auto; padding-left: 0; }
.kb-main { min-height: 0; }
.kb-drawer { width: 100%; max-width: 100%; }

/* 手机：弹层铺满可用高度，按钮均分整行 —— 底部「开始上传」不许被地址栏挤出去 */
.kb-modal-mask { padding: 12px; align-items: flex-end; }
.kb-modal { max-height: calc(100vh - 24px); max-height: calc(100dvh - 24px); }
.kb-modal-foot .btn { flex: 1 1 120px; }
.kb-browse-bar { flex-wrap: wrap; }
.kb-name-input { flex: 1 1 100%; }
.kb-pager { justify-content: space-between; flex-wrap: wrap; }
.kb-doc-act .kb-mini { opacity: 1; }
}
</style>
