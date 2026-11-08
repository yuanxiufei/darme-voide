<template>
  <div class="kb-page">
    <div class="kb-head">
      <div>
        <div class="kl-kicker">Knowledge Loop</div>
        <h1 class="kb-title">知识闭环</h1>
        <p class="kb-desc">
          本仓自己跑出来的经验：跑次事实 → 蒸馏提案 → 人审采纳 → 注入 Agent 与导出训练。
          与「知识库」页分工明确 —— 那边是仓外剧本资料的读写，这边是本仓跑次沉淀的经验账。
        </p>
      </div>
      <div class="kb-stats">
        <div class="kb-stat"><b>{{ report.runsTotal ?? 0 }}</b><span>跑次事实</span></div>
        <div class="kb-stat"><b>{{ ledger.total ?? 0 }}</b><span>台账条目</span></div>
        <div class="kb-stat kb-stat-ok"><b>{{ countOf('accepted') }}</b><span>已采纳</span></div>
        <div class="kb-stat kb-stat-warn"><b>{{ countOf('proposed') }}</b><span>待审</span></div>
      </div>
    </div>

    <div class="kb-body">
      <aside class="kb-side">
        <div class="kb-side-head">闭环</div>
        <button
          v-for="item in navItems"
          :key="item.id"
          :class="['kb-group', { active: view === item.id }]"
          @click="view = item.id"
        >
          <span class="kb-group-name">{{ item.label }}</span>
          <span v-if="item.badge" class="kb-group-num">{{ item.badge }}</span>
        </button>
        <p class="kl-nav-note">
          采纳与调参都会改变后续出图行为，所以每一步都要凭据（run_id / 盘上文件）。
          凭据核不过时后端整条拒，页面原样透出那句病因。
        </p>
      </aside>

      <section class="kb-main">
        <!-- 概览 -->
        <div v-if="view === 'overview'" class="kb-list">
          <div class="kb-toolbar">
            <span class="kl-section-title">事实层</span>
            <button class="btn btn-ghost kl-tool-btn" :disabled="busy.rebuild" @click="rebuildRuns">
              {{ busy.rebuild ? '重建中…' : '从 traces 重建索引' }}
            </button>
            <button class="btn btn-ghost kl-tool-btn" :disabled="loading.report" @click="loadReport(true)">重算候选</button>
          </div>
          <div class="kl-cards">
            <div class="kl-card">
              <div class="kl-card-head">
                <span>跑次能力分布</span>
                <span class="kb-pill">{{ report.runs?.groupBy || 'stage' }}</span>
              </div>
              <div v-if="report.runs && report.runs.available === false" class="kb-warn-inline">
                {{ report.runs.hint || '索引库不可用' }}
              </div>
              <table v-else-if="runGroups.length" class="kl-table">
                <thead><tr><th>阶段</th><th>条数</th><th>失败率</th><th>平均分</th><th>P50</th><th>P95</th></tr></thead>
                <tbody>
                  <tr v-for="g in runGroups" :key="g.key">
                    <td class="kl-mono">{{ g.key }}</td>
                    <td>{{ g.total }}</td>
                    <td :class="{ 'kl-bad': (g.failRate ?? 0) > 0.2 }">{{ g.failRate == null ? '—' : pct(g.failRate) }}</td>
                    <td>{{ g.avgScore == null ? '—' : g.avgScore.toFixed(3) }}</td>
                    <td>{{ fmtMs(g.p50ElapsedMs) }}</td>
                    <td>{{ fmtMs(g.p95ElapsedMs) }}</td>
                  </tr>
                </tbody>
              </table>
              <p v-else class="kb-empty-inline">还没有跑次事实。跑一次画布或 Agent 之后这里会有数据。</p>
            </div>

            <div class="kl-card">
              <div class="kl-card-head">
                <span>台账分布</span>
                <span class="kb-pill">{{ ledger.total ?? 0 }} 条</span>
              </div>
              <div v-if="ledger.available === false" class="kb-warn-inline">{{ ledger.hint || '台账不可用' }}</div>
              <template v-else>
                <div v-for="row in kindStatusRows" :key="row.kind" class="kl-kv">
                  <span class="kl-kv-key">{{ kindLabel(row.kind) }}</span>
                  <span class="kl-kv-val">
                    <em v-for="s in row.statuses" :key="s.status" :class="['kb-pill', statusPillClass(s.status)]">
                      {{ statusLabel(s.status) }} {{ s.count }}
                    </em>
                  </span>
                </div>
                <p v-if="!kindStatusRows.length" class="kb-empty-inline">台账是空的。</p>
              </template>
            </div>

            <div class="kl-card">
              <div class="kl-card-head">
                <span>定时复盘</span>
                <span :class="['kb-pill', sched.enabled ? 'kb-pill-ok' : 'kb-pill-warn']">
                  {{ sched.enabled ? '已启用' : '未启用' }}
                </span>
              </div>
              <div class="kl-kv"><span class="kl-kv-key">上次运行</span><span>{{ fmtTime(sched.lastRunAt) }}</span></div>
              <div class="kl-kv"><span class="kl-kv-key">下次运行</span><span>{{ fmtTime(sched.nextRunAt) }}</span></div>
              <div class="kl-kv">
                <span class="kl-kv-key">计划</span>
                <span class="kl-mono">
                  每天 {{ pad(sched.schedule?.hour) }}:{{ pad(sched.schedule?.minute) }}
                  · limit {{ sched.schedule?.limit ?? '—' }}
                </span>
              </div>
              <p class="kl-note">定时器是进程级后台任务，这里只读。「未启用」与「启用了但失败」是两件事。</p>
              <div class="kl-card-acts">
                <button class="btn btn-ghost kl-tool-btn" :disabled="busy.distill" @click="distill(true)">
                  {{ busy.distill ? '跑…' : '预览一轮' }}
                </button>
                <button class="btn btn-primary kl-tool-btn" :disabled="busy.distill" @click="distill(false)">真跑一轮</button>
                <button class="btn btn-ghost kl-tool-btn" :disabled="busy.lift" @click="refreshLift">重算 lift</button>
              </div>
            </div>
          </div>

          <div class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>下一批蒸馏候选</span>
              <span class="kb-pill">{{ candidates.length }} 条</span>
            </div>
            <p v-if="!candidates.length" class="kb-empty-inline">
              没有达到阈值的统计信号 —— 样本不足与确实无异常都不该被显示成「没有问题」。
            </p>
            <div v-for="c in candidates" :key="c.id || c.key || c.pattern || 'cand'" class="kl-cand">
              <div class="kl-cand-head">
                <span class="kb-pill">{{ c.pattern || c.rule || c.kind || 'rule' }}</span>
                <span class="kl-mono">{{ candScope(c) }}<template v-if="c.field"> · {{ c.field }}</template></span>
                <span class="kl-cand-samples">样本 {{ candSamples(c) }}</span>
              </div>
              <p class="kl-cand-why">{{ candWhy(c) }}</p>
              <div v-if="c.evidence && c.evidence.length" class="kl-evidence">
                <span class="kl-evidence-label">凭据</span>
                <code v-for="ev in c.evidence" :key="ev" class="kl-code">{{ ev }}</code>
              </div>
            </div>
          </div>

          <pre v-if="lastAction" class="kl-out">{{ lastAction }}</pre>
        </div>

        <!-- 优化史 -->
        <div v-else-if="view === 'optimizations'" class="kb-list">
          <div class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>优化史（每一轮到底提升了多少）</span>
              <span class="kb-pill">{{ optSummary.rounds ?? 0 }} 轮</span>
            </div>
            <p class="kl-note">
              一版一行、追加不覆盖：<code class="kl-code">agent_configs</code> 是覆盖式的 ——
              上一次生效的提示词与它的分数当场就没了。所以「判决」（分数严格更高）与「生效」
              （真写成默认值）是<b>两个</b>布尔：合成一个就会出现「页面说已采纳、而默认提示词还是旧的」。
              分数只记前→后与 Δ，提示词正文只留指纹（全文在 <code class="kl-code">benchmarks/history/</code> 里）。
              参数调优（同一份提示词只换 <code class="kl-code">temperature</code>）也进这张表，
              目标写成「生成参数（温度）」—— 那种行的指纹与上一版<b>一样是正常的</b>（提示词没动），
              分数差就是那一格旋钮的功劳；涨了但没超过抖动带的记「涨了，但没超过抖动带」。
            </p>
            <div v-if="optData.available === false" class="kb-warn-inline">
              {{ optData.reason || '优化史读不出来' }}（不拿空表充数）
            </div>
            <div v-else class="kl-note">
              跑过 {{ optSummary.rounds ?? 0 }} 轮 · 分数更高 {{ optSummary.accepted ?? 0 }} 次 · 真落库
              {{ optSummary.persisted ?? 0 }} 次 · 能算 Δ 的 {{ optSummary.scored ?? 0 }} 条 · 最好一轮
              <b>{{ optSummary.bestDelta == null ? '—（还没有能成对比较的一轮）' : signed(optSummary.bestDelta) }}</b>
              <template v-if="optSummary.bestVersion != null">（v{{ optSummary.bestVersion }}）</template>
              · 最后一条 {{ fmtTime(optSummary.lastAt) }}
            </div>
            <!-- 筛选与「刷新」不论成败都要在：懒加载只认「第一次切进来」，
                 失败后切走再切回不会重试（`loaded[v]` 已置真）⇒ 不留这条路就是死胡同 -->
            <div class="kl-inline-form">
              <input v-model="optForm.agentType" class="kb-name-input" placeholder="agentType（可空）">
              <input v-model="optForm.caseId" class="kb-name-input" placeholder="caseId（可空）">
              <label class="kl-inline-num">条数 <input v-model.number="optForm.limit" type="number" min="1" max="500"></label>
              <button class="btn btn-ghost kl-tool-btn" :disabled="busy.optimizations" @click="loadOptimizations">
                {{ busy.optimizations ? '读取中…' : '刷新' }}
              </button>
            </div>

            <table v-if="optData.available !== false && optItems.length" class="kl-table">
              <thead>
                <tr>
                  <th>版本</th><th>父版</th><th>目标</th><th>分数（前 → 后）</th><th>Δ</th>
                  <th>判决</th><th>落库</th><th>原因</th><th>模型</th><th>提示词指纹</th><th>时间</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="r in optItems" :key="r.id">
                  <td class="kl-mono">v{{ r.version }}</td>
                  <td class="kl-mono">{{ r.parentVersion == null ? '—' : `v${r.parentVersion}` }}</td>
                  <td>{{ r.targetLabel || r.target }}</td>
                  <td class="kl-mono">{{ scorePair(r) }}</td>
                  <td class="kl-mono" :class="deltaClass(r.delta)">{{ signed(r.delta) }}</td>
                  <td>
                    <span :class="['kb-pill', r.accepted ? 'kb-pill-ok' : '']">
                      {{ r.accepted ? '采纳' : '未采纳' }}
                    </span>
                  </td>
                  <td>
                    <span :class="['kb-pill', r.persisted ? 'kb-pill-ok' : 'kb-pill-warn']">
                      {{ r.persisted ? '已是默认' : '未落库' }}
                    </span>
                  </td>
                  <td class="kl-wrap">{{ r.reasonLabel || r.reason || '—' }}</td>
                  <td class="kl-mono kl-wrap">{{ r.runtimeModel || '—' }}</td>
                  <td class="kl-mono" :title="r.promptSha || ''">{{ r.promptSha || '—' }}</td>
                  <td class="kl-mono">{{ fmtTime(r.createdAt) }}</td>
                </tr>
              </tbody>
            </table>
            <p v-else-if="optData.available !== false" class="kb-empty-inline">
              表是空的 —— 优化器还没跑过一轮（表空 ≠ 读不出来，读不出来上面那行会红着脸说）。
            </p>
          </div>
        </div>

        <!-- 台账 -->
        <div v-else-if="view === 'entries'" class="kb-list">
          <div class="kb-toolbar">
            <select v-model="filter.status" class="kb-filter on" @change="loadEntries(0)">
              <option value="">全部状态</option>
              <option v-for="s in STATUSES" :key="s" :value="s">{{ statusLabel(s) }}</option>
            </select>
            <select v-model="filter.kind" class="kb-filter" @change="loadEntries(0)">
              <option value="">全部类型</option>
              <option v-for="k in kindOptions" :key="k.kind" :value="k.kind">{{ k.label }}</option>
            </select>
            <input v-model="filter.scope" class="kb-name-input kl-scope-input" placeholder="范围：global / stage:image" @keyup.enter="loadEntries(0)">
            <input v-model="filter.tags" class="kb-name-input kl-scope-input" placeholder="标签（多标签＝同时具备）" @keyup.enter="loadEntries(0)">
            <button class="btn btn-ghost kl-tool-btn" @click="loadEntries(0)">筛选</button>
            <button class="btn btn-primary kl-tool-btn" @click="openCreate">新建条目</button>
            <label class="kl-check">
              <input v-model="filter.includeDeleted" type="checkbox" @change="loadEntries(0)">
              <span>回收站</span>
            </label>
            <button class="btn btn-ghost kl-tool-btn" @click="toggleAudit()">{{ auditOpen ? '收起审计' : '操作审计' }}</button>
          </div>

          <!--
            标签 facet：数字与「点进去的条数」同口径（后端同一套词边界匹配）。
            点标签 = 切换 AND 条件，不是替换 —— 点「节奏」再点「过曝」是"同时具备"。
          -->
          <div v-if="taxonomy?.tags?.length" class="kl-facet-row">
            <button
              v-for="f in taxonomy.tags.slice(0, 24)" :key="f.tag"
              :class="['kb-pill', 'kl-facet', { on: activeTags.includes(f.tag) }]"
              @click="toggleTag(f.tag)"
            >{{ f.tag }} <b>{{ f.count }}</b></button>
          </div>

          <div class="kl-list-meta">
            <span>共 {{ entryList.total }} 条</span>
            <span v-if="entryList.available === false" class="kl-bad">{{ entryList.hint }}</span>
            <span v-else-if="filter.includeDeleted" class="kl-bad">回收站视图：这里的条目默认不出现在列表与检索里。</span>
            <span v-else class="kl-note-inline">不筛状态时提案与已采纳混在一起 —— 请显式选一个状态再读。</span>
          </div>

          <!-- 谁在做：`role` 是**声明式**的（本仓没有登录态），写进审计行 -->
          <div class="kb-toolbar kl-actor-row">
            <input v-model="actor.operator" class="kb-name-input kl-scope-input" placeholder="操作者（进审计）">
            <select v-model="actor.role" class="kb-filter">
              <option value="">角色未声明</option>
              <option v-for="r in ROLES" :key="r" :value="r">{{ r }}</option>
            </select>
            <span class="kl-note-inline">
              角色是自己声明的，只保证「审计里的角色与动作不自相矛盾」，不防冒充。
            </span>
          </div>

          <div v-if="auditOpen" class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>操作审计（软删/恢复/硬删都在这里留痕，硬删后账仍在）</span>
              <button class="kb-mini" @click="loadAudit()">刷新</button>
            </div>
            <p v-if="auditList?.available === false" class="kl-note">{{ auditList.hint }}</p>
            <p v-else-if="!auditList?.items?.length" class="kb-empty">还没有审计记录</p>
            <div v-for="a in auditList?.items || []" :key="a.id" class="kl-kv">
              <span class="kl-kv-key">{{ fmtTime(a.atNs) }}</span>
              <span class="kl-mono">{{ a.action }}</span>
              <span class="kl-mono">{{ a.entryId }}</span>
              <span>{{ a.operator }}<template v-if="a.role"> / {{ a.role }}</template></span>
              <span class="kl-muted">{{ a.detail }}</span>
            </div>
          </div>

          <p v-if="!entryList.items.length && !loading.entries" class="kb-empty">
            {{ entryList.available === false ? '台账不可用' : '这个筛选下没有条目' }}
          </p>
          <article v-for="e in entryList.items" :key="e.id" class="kl-entry" @click="openEntry(e.id, filter.includeDeleted)">
            <div class="kl-entry-head">
              <span :class="['kb-pill', statusPillClass(e.status)]">{{ statusLabel(e.status) }}</span>
              <span class="kb-pill">{{ kindLabel(e.kind) }}</span>
              <span v-if="e.deleted" class="kb-pill kl-bad">已删除 · {{ e.deletedBy || '未知' }}</span>
              <span class="kl-entry-title">{{ e.title }}</span>
              <span class="kl-mono kl-muted">v{{ e.version }}</span>
            </div>
            <p class="kl-entry-body">{{ e.body }}</p>
            <div class="kl-entry-foot">
              <span class="kl-mono">{{ e.scope }}</span>
              <span>佐证 {{ e.support }} · 反证 {{ e.contra }}</span>
              <span :class="{ 'kl-bad': (e.lift ?? 0) < 0 }">
                lift {{ e.lift == null ? '样本不足' : e.lift.toFixed(3) }}
              </span>
              <span class="kl-muted">{{ fmtTime(e.updated_ns) }}</span>
              <span class="kl-entry-acts" @click.stop>
                <button v-for="t in e.tagList || []" :key="t" class="kb-pill kl-facet" @click="toggleTag(t)">{{ t }}</button>
                <template v-if="e.deleted">
                  <button class="kb-mini" @click="restoreEntry(e)">恢复</button>
                  <button class="kb-mini kl-bad" @click="removeEntry(e, true)">彻底删除</button>
                </template>
                <template v-else>
                  <button class="kb-mini" @click="openEdit(e)">编辑</button>
                  <button class="kb-mini" @click="openVersions(e)">版本</button>
                  <button class="kb-mini kl-bad" @click="removeEntry(e, false)">删除</button>
                </template>
                <button v-for="next in nextStatuses(e.status)" :key="next" class="kb-mini" @click="changeStatus(e, next)">
                  {{ statusLabel(next) }}
                </button>
              </span>
            </div>
          </article>

          <div v-if="entryList.total > pageSize" class="kb-pager">
            <span class="kb-pager-info">{{ entryOffset + 1 }}–{{ Math.min(entryOffset + pageSize, entryList.total) }} / {{ entryList.total }}</span>
            <button class="btn btn-ghost kl-tool-btn" :disabled="entryOffset === 0" @click="loadEntries(entryOffset - pageSize)">上一页</button>
            <button class="btn btn-ghost kl-tool-btn" :disabled="entryOffset + pageSize >= entryList.total" @click="loadEntries(entryOffset + pageSize)">下一页</button>
          </div>
        </div>

        <!-- 检索测试 -->
        <div v-else-if="view === 'retrieval'" class="kl-test">
          <div class="kl-test-form">
            <label class="kl-field">
              <span class="kl-field-label">查询</span>
              <textarea v-model="searchForm.q" class="kl-textarea" rows="4" placeholder="例如：人脸崩坏 / 夜景噪点 / 3 秒钩子" @keydown.ctrl.enter="runSearch" />
            </label>
            <div class="kl-field-row">
              <label class="kl-field">
                <span class="kl-field-label">类型</span>
                <select v-model="searchForm.kind" class="kb-filter on">
                  <option value="">不限</option>
                  <option v-for="k in KINDS" :key="k" :value="k">{{ kindLabel(k) }}</option>
                </select>
              </label>
              <label class="kl-field">
                <span class="kl-field-label">状态</span>
                <select v-model="searchForm.status" class="kb-filter on">
                  <option value="">不限</option>
                  <option v-for="s in STATUSES" :key="s" :value="s">{{ statusLabel(s) }}</option>
                </select>
              </label>
            </div>
            <div class="kl-field-row">
              <label class="kl-field">
                <span class="kl-field-label">范围</span>
                <input v-model="searchForm.scope" class="kb-name-input" placeholder="global / stage:image">
              </label>
              <label class="kl-field kl-field-narrow">
                <span class="kl-field-label">条数</span>
                <input v-model.number="searchForm.limit" type="number" min="1" max="50" class="kb-name-input">
              </label>
            </div>
            <button class="btn btn-primary kl-run" :disabled="loading.search" @click="runSearch">
              {{ loading.search ? '检索中…' : '检索' }}
            </button>
            <p class="kl-note">
              ≥3 字走全文索引（bm25），1–2 字回落模糊匹配（like），空查询按最近更新倒序（recent）。
              三种排序的分数不可比，排名方式会原样显示在结果上方。
            </p>
          </div>

          <div class="kl-test-result">
            <div v-if="loading.search" class="kb-empty">检索中…</div>
            <template v-else-if="searched">
              <div class="kl-result-head">
                <span>命中 {{ searchResult.total }} 条</span>
                <span class="kb-pill">ranking: {{ searchResult.ranking }}</span>
                <span v-if="searchResult.available === false" class="kl-bad">{{ searchResult.hint }}</span>
                <span v-else-if="searchResult.hint" class="kl-muted">{{ searchResult.hint }}</span>
              </div>
              <p v-if="!searchResult.items.length" class="kb-empty">这条查询没有命中 —— 是「查过了没有」，不是「没查」。</p>
              <article v-for="(e, i) in searchResult.items" :key="e.id" class="kl-entry" @click="openEntry(e.id)">
                <div class="kl-entry-head">
                  <span class="kl-rank">{{ i + 1 }}</span>
                  <span :class="['kb-pill', statusPillClass(e.status)]">{{ statusLabel(e.status) }}</span>
                  <span class="kl-entry-title">{{ e.title }}</span>
                </div>
                <p class="kl-entry-body">{{ e.body }}</p>
                <div class="kl-entry-foot">
                  <span class="kl-mono">{{ e.kind }} · {{ e.scope }}</span>
                  <span>{{ fmtTime(e.updated_ns) }}</span>
                </div>
              </article>
            </template>
            <p v-else class="kb-empty">输入查询词后点「检索」。这里不会预填任何结果。</p>
          </div>
        </div>

        <!-- 投喂与数据集 -->
        <div v-else-if="view === 'feed'" class="kb-list">
          <div class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>L1 注入预览（system prompt 的 Knowledge 段）</span>
              <span v-if="feed" class="kb-pill">{{ feed.entries?.length ?? 0 }} 条 / {{ feed.used ?? 0 }} 字</span>
              <span v-if="feed && feed.targeted" class="kb-pill">其中 {{ feed.targeted }} 条定向</span>
            </div>
            <div class="kl-inline-form">
              <input v-model="feedForm.agentType" class="kb-name-input" placeholder="agentType（可空）">
              <label class="kl-inline-num">剧 <input v-model.number="feedForm.dramaId" type="number" min="1" step="1"></label>
              <label class="kl-inline-num">集 <input v-model.number="feedForm.episodeId" type="number" min="1" step="1"></label>
              <label class="kl-inline-num">预算 <input v-model.number="feedForm.budgetChars" type="number" min="0" max="8000" step="100"></label>
              <label class="kl-inline-num">条数 <input v-model.number="feedForm.limit" type="number" min="1" max="30"></label>
              <button class="btn btn-ghost kl-tool-btn" :disabled="busy.feed" @click="loadFeed">编译预览</button>
            </div>
            <p class="kl-note">
              填上「剧」「集」才看得到本集/本剧那几条 —— 不填就只剩全局成法，
              生产侧（`assemble_full_instructions`）则是带着集号跑的。
            </p>
            <div v-if="feed && feed.available === false" class="kb-warn-inline">{{ feed.hint || '台账不可用' }}</div>
            <template v-else-if="feed">
              <pre v-if="feed.text" class="kl-out">{{ feed.text }}</pre>
              <div v-else class="kb-warn-inline">{{ feed.hint || '（空段：这条链上没有够格的知识）' }}</div>
              <p class="kl-note">
                作用域：{{ (feed.scopes ?? []).join(' · ') || '（未限定）' }}
                <span v-if="feed.blocked" class="kl-bad">
                  另有 {{ feed.blocked }} 条知识存在、但作用域不符没进这一段。
                </span>
              </p>
              <p class="kl-note">
                真正被写进去的 id 会记 usage —— 没有这一步 lift 永远算不出来。
                <span v-if="feed.truncated" class="kl-bad">本次因预算被截断。</span>
              </p>
            </template>
            <p v-else class="kb-empty-inline">还没编译过。点「编译预览」看这段长什么样。</p>
          </div>

          <div class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>L3 导出数据集</span>
              <span class="kb-pill">{{ datasets.length }} 份</span>
            </div>
            <div class="kl-inline-form">
              <select v-model="exportForm.target" class="kb-filter on">
                <option value="knowledge">已采纳知识</option>
                <option value="runs">高分跑次</option>
              </select>
              <input v-model.number="exportForm.limit" type="number" min="1" max="5000" class="kb-name-input kl-mini-input">
              <button class="btn btn-ghost kl-tool-btn" :disabled="busy.export" @click="runExport">导出</button>
              <button class="btn btn-ghost kl-tool-btn" :disabled="busy.datasets" @click="loadDatasets">刷新列表</button>
            </div>
            <table v-if="datasets.length" class="kl-table">
              <thead><tr><th>数据集</th><th>类型</th><th>样本</th><th>生成时间</th><th>文件</th><th /></tr></thead>
              <tbody>
                <tr v-for="d in datasets" :key="d.ref">
                  <td class="kl-mono">{{ d.name }}</td>
                  <td>{{ d.dataset || '—' }}</td>
                  <td>{{ d.count ?? '—' }}</td>
                  <td>{{ fmtTime(d.generatedNs) }}</td>
                  <td>
                    <span :class="['kb-pill', d.itemsExists ? 'kb-pill-ok' : 'kb-pill-bad']">
                      {{ d.itemsExists ? '样本在' : '样本已缺失' }}
                    </span>
                  </td>
                  <td><button class="kb-mini" @click="resolveDataset(d.ref)">核一遍</button></td>
                </tr>
              </tbody>
            </table>
            <p v-else class="kb-empty-inline">{{ datasetsAvailable ? '还没有导出过数据集。' : '数据集根目录不存在，导出后会创建。' }}</p>
          </div>

          <pre v-if="lastAction" class="kl-out">{{ lastAction }}</pre>
        </div>

        <!-- 调参与训练 -->
        <div v-else class="kb-list">
          <div class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>参数覆盖层</span>
              <span class="kb-pill">{{ overrideCount }} 条覆盖 / {{ tuning.fields?.length ?? 0 }} 个可调项</span>
            </div>
            <div v-if="tuning.problems?.length" class="kb-warn-inline">{{ tuning.problems.join('；') }}</div>
            <table v-if="tuning.fields?.length" class="kl-table">
              <thead><tr><th>阶段</th><th>字段</th><th>默认</th><th>当前</th><th>来源</th><th>依据</th><th /></tr></thead>
              <tbody>
                <tr v-for="f in tuning.fields" :key="`${f.stage}.${f.field}`">
                  <td class="kl-mono">{{ f.stage }}</td>
                  <td class="kl-mono">{{ f.field }}</td>
                  <td class="kl-mono">{{ f.default }}</td>
                  <td class="kl-mono" :class="{ 'kl-good': String(f.value) !== String(f.default) }">{{ f.value }}</td>
                  <td><span :class="['kb-pill', f.source === 'default' ? '' : 'kb-pill-warn']">{{ f.source }}</span></td>
                  <td class="kl-wrap">{{ f.basis || '—' }}</td>
                  <td>
                    <button v-if="f.source !== 'default'" class="kb-mini kb-mini-danger" @click="clearTuningRow(f)">清除</button>
                  </td>
                </tr>
              </tbody>
            </table>
            <p v-else class="kb-empty-inline">{{ tuning.exists === false ? '还没有覆盖文件（全部走默认值）。' : '没有可调项。' }}</p>
          </div>

          <div class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>调参候选（只读）</span>
              <span class="kb-pill">样本门槛 {{ suggest.minSamples ?? suggestForm.minSamples }} 次</span>
            </div>
            <div class="kl-inline-form">
              <label class="kl-inline-num">门槛 <input v-model.number="suggestForm.minSamples" type="number" min="1" max="5000"></label>
              <button class="btn btn-ghost kl-tool-btn" :disabled="busy.suggest" @click="loadSuggest">重算候选</button>
            </div>
            <p v-if="suggest.available === false" class="kb-warn-inline">索引库不可用：{{ suggest.db }}</p>
            <div v-if="suggest.notEvaluated?.length" class="kl-note">
              算不了的维度（没检查 ≠ 检查过没问题）：
              <div v-for="n in suggest.notEvaluated" :key="n.rule" class="kl-dim">
                <code class="kl-code">{{ n.rule }}</code>
                <span class="kl-dim-why">{{ n.why }}</span>
              </div>
            </div>
            <div v-for="c in suggest.candidates || []" :key="c.id" class="kl-cand">
              <div class="kl-cand-head">
                <span class="kb-pill kb-pill-warn">{{ c.rule }}</span>
                <span class="kl-mono">{{ c.stage }} · {{ c.field }}</span>
                <span class="kl-cand-samples">{{ c.current }} → {{ c.proposed }}（样本 {{ c.samples }}）</span>
                <span v-if="c.qcGate" class="kb-pill" :class="c.qcGate.verdict === 'pass' ? 'kb-pill-ok' : 'kb-pill-warn'">
                  质检{{ c.qcGate.verdict === 'pass' ? '过了这道关' : '没数据、判不了' }}
                </span>
              </div>
              <p class="kl-cand-why">{{ c.why }}</p>
              <div class="kl-card-acts">
                <span class="kb-pill">{{ c.canApply ? '可应用' : '需人工核实' }}</span>
                <button class="kb-mini" @click="prefillTuning(c)">按这条写覆盖</button>
              </div>
            </div>
            <p v-if="!(suggest.candidates || []).length && suggest.available !== false" class="kb-empty-inline">
              没有候选。样本量到了再看，别拿少量跑次改参数。
            </p>
          </div>

          <div class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>质检命中率与人工复核</span>
              <span class="kb-pill">复核门槛 {{ qcRate.minReviews ?? 10 }} 条</span>
            </div>
            <p class="kl-note">
              命中率的分母是「人看过成片之后的结论」，不是触发次数 —— 阈值报了一次黑场，
              是真黑还是片头淡入误杀，机器判不出来。下面每一条都能点「判对 / 误报 / 漏报」。
            </p>
            <p v-if="qcRate.available === false" class="kb-warn-inline">
              复核表读不出来：{{ qcRate.reason }}（这里不会编一个 0 条来充数）
            </p>
            <div class="kl-note">
              质检记录 {{ qcRate.records?.total ?? 0 }} 条 · 已复核 {{ qcRate.records?.reviewed ?? 0 }}
              · 待复核 {{ qcRate.records?.unreviewed ?? 0 }}
              <template v-if="qcRate.notes?.length">；{{ qcRate.notes.join('；') }}</template>
            </div>
            <table v-if="(qcRate.codes || []).length" class="kl-table">
              <thead>
                <tr>
                  <th>类别</th><th>触发</th><th>复核</th><th>判对</th><th>误报</th><th>漏报</th>
                  <th>命中率</th><th>漏报率</th><th>结论</th><th>对应阈值</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="c in qcRate.codes" :key="c.code">
                  <td class="kl-mono" :title="c.label">{{ c.code }}</td>
                  <td>{{ c.triggeredRecords }}</td>
                  <td>{{ c.reviewed }}</td>
                  <td>{{ c.hit }}</td>
                  <td>{{ c.falseAlarm }}</td>
                  <td>{{ c.missed }}</td>
                  <td class="kl-mono">{{ c.hitRate == null ? '—' : Math.round(c.hitRate * 100) + '%' }}</td>
                  <td class="kl-mono">{{ c.missRate == null ? '—' : Math.round(c.missRate * 100) + '%' }}</td>
                  <td>
                    <span :class="['kb-pill', QC_VERDICT_CLASS[c.verdict] || '']">
                      {{ QC_VERDICT_LABEL[c.verdict] || c.verdict }}
                    </span>
                  </td>
                  <td class="kl-mono">
                    {{ c.threshold ? c.threshold.field : '无阈值（环境问题）' }}
                    <span v-if="c.threshold && c.threshold.aux" class="kl-note">（辅 {{ c.threshold.aux }}）</span>
                  </td>
                </tr>
              </tbody>
            </table>
            <p v-else class="kb-empty-inline">还没有触发过带 code 的质检缺陷。</p>

            <div class="kl-cand-head kl-qc-head">
              <span>待复核（{{ qcPending.unreviewed ?? 0 }} 条没判完）</span>
              <button class="btn btn-ghost kl-tool-btn" @click="loadQc">刷新</button>
            </div>
            <div v-for="item in qcPending.items" :key="item.checkId" class="kl-cand">
              <div class="kl-cand-head">
                <span class="kl-mono">QC #{{ item.checkId }}</span>
                <span class="kb-pill">{{ item.status }}</span>
                <span v-if="item.storyboardId" class="kl-mono">分镜 {{ item.storyboardId }}</span>
                <span class="kl-cand-samples">{{ fmtTime(item.createdAt) }}</span>
                <span :class="['kb-pill', item.reviewed ? 'kb-pill-ok' : 'kb-pill-warn']">
                  {{ item.reviewed ? '已判完' : '待判' }}
                </span>
              </div>
              <div v-for="issue in item.issues" :key="issue.code" class="kl-qc-row">
                <span class="kl-mono">{{ issue.code }}</span>
                <span class="kl-note kl-wrap">{{ issue.message || issue.label }}</span>
                <span v-if="issue.review" class="kb-pill">{{ issue.review.verdictLabel }}</span>
                <input
                  v-model="qcNotes[qcKey(item.checkId, issue.code)]"
                  class="kb-name-input kl-qc-note"
                  placeholder="备注（可选）"
                >
                <button
                  v-for="v in VERDICT_ORDER"
                  :key="v"
                  class="kb-mini"
                  :disabled="busyQc[qcKey(item.checkId, issue.code)]"
                  @click="judge(item, issue, v)"
                >
                  {{ VERDICT_SHORT[v] }}
                </button>
                <button
                  v-if="issue.review"
                  class="kb-mini kb-mini-danger"
                  :disabled="busyQc[qcKey(item.checkId, issue.code)]"
                  @click="undoJudge(item, issue)"
                >
                  撤销
                </button>
              </div>
            </div>
            <p v-if="!(qcPending.items || []).length" class="kb-empty-inline">
              最近没有需要复核的质检记录（跑一次技术质检后这里会出现）。
            </p>
          </div>

          <div class="kl-card kl-card-wide">
            <div class="kl-card-head"><span>写一条覆盖（必须给依据与凭据）</span></div>
            <div class="kl-inline-form">
              <input v-model="tuneForm.stage" class="kb-name-input" placeholder="stage，如 image">
              <input v-model="tuneForm.field" class="kb-name-input" placeholder="field，如 steps">
              <input v-model="tuneForm.value" class="kb-name-input" placeholder="值，如 24 / 4.5 / true">
            </div>
            <textarea v-model="tuneForm.basis" class="kl-textarea" rows="2" placeholder="为什么改（写清症状与判断）" />
            <textarea v-model="tuneForm.evidence" class="kl-textarea" rows="2" placeholder="凭据，一行一个：真实 run_id 或盘上文件路径" />
            <div class="kl-card-acts">
              <button class="btn btn-primary kl-tool-btn" :disabled="busy.tuning" @click="submitTuning">写入</button>
              <span class="kl-note">凭据核不过时整条会被拒 —— 这是设计，不是 bug。</span>
            </div>
          </div>

          <div class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>训练任务与产物</span>
              <span class="kb-pill">产物不会自动生效</span>
            </div>
            <div v-if="training.problems?.length" class="kb-warn-inline">{{ training.problems.join('；') }}</div>
            <table v-if="training.jobs?.length" class="kl-table">
              <thead><tr><th>任务</th><th>模型键</th><th>阶段</th><th>状态</th><th>创建</th><th>来源</th><th /></tr></thead>
              <tbody>
                <tr v-for="j in training.jobs" :key="j.taskId">
                  <td class="kl-mono">{{ j.taskId }}</td>
                  <td class="kl-mono">{{ j.modelKey }}</td>
                  <td>{{ j.stage || '—' }}</td>
                  <td><span :class="['kb-pill', j.status === 'done' ? 'kb-pill-ok' : j.status === 'failed' ? 'kb-pill-bad' : '']">{{ j.status }}</span></td>
                  <td>{{ fmtTime(j.createdAt) }}</td>
                  <td><span class="kb-pill">{{ j.restored ? '盘上登记' : '本进程' }}</span></td>
                  <td><button class="kb-mini" @click="scanTraining(j.taskId)">重扫产物</button></td>
                </tr>
              </tbody>
            </table>
            <p v-else class="kb-empty-inline">还没有训练任务。</p>

            <table v-if="training.models?.length" class="kl-table kl-mt">
              <thead><tr><th>产物</th><th>类型</th><th>状态</th><th>文件</th><th>依据</th><th /></tr></thead>
              <tbody>
                <tr v-for="m in training.models" :key="m.modelId">
                  <td class="kl-mono">
                    {{ m.modelKey }}<span v-if="m.isDefault" class="kb-pill kb-pill-ok kl-ml">默认</span>
                  </td>
                  <td>{{ m.kind }}</td>
                  <td><span :class="['kb-pill', m.status === 'accepted' ? 'kb-pill-ok' : '']">{{ m.status }}</span></td>
                  <td>
                    <span :class="['kb-pill', m.exists ? 'kb-pill-ok' : 'kb-pill-bad']">
                      {{ m.exists ? fmtBytes(m.sizeBytes) : '不在盘上' }}
                    </span>
                  </td>
                  <td class="kl-wrap">{{ m.basis || '—' }}</td>
                  <td class="kl-entry-acts">
                    <button v-if="!m.isDefault" class="kb-mini" :disabled="!m.exists" @click="openPromote(m)">设为默认</button>
                    <button v-else class="kb-mini kb-mini-danger" @click="demote(m)">撤销</button>
                  </td>
                </tr>
              </tbody>
            </table>
            <p v-else class="kb-empty-inline">还没有训练产物。扫到 0 个不等于出错。</p>
          </div>

          <pre v-if="lastAction" class="kl-out">{{ lastAction }}</pre>
        </div>

        <!-- 成片评测（缺口 8 的另一半：此前评测面全在"产出层"，没有任何判据看**成片**） -->
        <div v-if="view === 'film'" class="kb-list">
          <div class="kb-toolbar">
            <span class="kl-section-title">成片评测</span>
            <span class="kl-note">
              读数不重量：只读已发生的分镜 / 成片 / 技术质检 / 人工复核，
              <b>不重跑 ffmpeg、不调模型</b>（重跑会与质检页的口径打架）
            </span>
          </div>

          <div class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>选一集来评</span>
              <span class="kb-pill">判不了 ≠ 通过，也 ≠ 0 分</span>
            </div>
            <div class="kl-inline-form">
              <select v-model.number="filmDramaId" class="kb-filter on" @change="loadFilmEpisodes">
                <option :value="0">选择剧组</option>
                <option v-for="d in filmDramas" :key="d.id" :value="d.id">{{ d.title }}</option>
              </select>
              <select v-model.number="filmEpisodeId" class="kb-filter on">
                <option :value="0">选择集</option>
                <option v-for="e in filmEpisodes" :key="e.id" :value="e.id">
                  第 {{ e.episode_number }} 集{{ e.title ? ` · ${e.title}` : '' }}
                </option>
              </select>
              <button
                class="btn btn-primary kl-tool-btn"
                :disabled="busy.film || !filmEpisodeId"
                @click="runFilmEval"
              >
                {{ busy.film ? '评测中…' : '评测这一集的成片' }}
              </button>
            </div>
            <p v-if="!filmReport" class="kb-empty-inline">
              还没评过。四维等权：出片覆盖 / 技术规格 / 缺陷面 / 人工复核 ——
              哪一维**一条样本都没有**就整维不参与总分（这里会如实写「判不了」，不折成 0 分也不折成满分）。
            </p>
          </div>

          <div v-if="filmReport" class="kl-card kl-card-wide">
            <div class="kl-card-head">
              <span>第 {{ filmReport.episodeId }} 集 · 成片四维</span>
              <span :class="['kb-pill', verdictPill(filmReport.verdict)]">{{ verdictLabel(filmReport.verdict) }}</span>
            </div>

            <div class="kb-stats kl-mt">
              <div class="kb-stat">
                <b>{{ filmReport.score == null ? '判不了' : filmReport.score.toFixed(1) }}</b>
                <span>总分{{ filmReport.score == null ? '（一维都判不了）' : ` / 线 ${filmReport.passScore}` }}</span>
              </div>
              <div class="kb-stat">
                <b>{{ filmReport.ready }} / {{ filmReport.total }}</b>
                <span>已出片 / 计划镜</span>
              </div>
              <div class="kb-stat">
                <b>{{ filmReport.dimensions.filter(d => d.judged).length }} / 4</b>
                <span>判得了的维度</span>
              </div>
            </div>

            <table class="kl-table">
              <thead>
                <tr><th>维度</th><th>得分</th><th>样本</th><th>说明</th></tr>
              </thead>
              <tbody>
                <tr v-for="d in filmReport.dimensions" :key="d.name">
                  <td>{{ d.label }}<span class="kl-mono kl-ml">{{ d.name }}</span></td>
                  <td>
                    <span :class="['kb-pill', d.judged ? (d.score! >= filmReport.passScore ? 'kb-pill-ok' : 'kb-pill-warn') : '']">
                      {{ d.judged ? d.score!.toFixed(1) : '判不了' }}
                    </span>
                  </td>
                  <td>{{ d.samples }}</td>
                  <td class="kl-wrap">{{ d.notes.length ? d.notes.join('；') : '—' }}</td>
                </tr>
              </tbody>
            </table>

            <div v-if="filmReport.missing.length" class="kb-warn-inline kl-mt">
              这几维没有样本，因此没有参与总分：{{ filmReport.missing.map(missingLabel).join('、') }}
            </div>

            <div class="kl-kv kl-mt">
              <span class="kl-kv-key">人工复核</span>
              <span>
                命中 {{ filmReport.review.hit }} · 误报 {{ filmReport.review.falseAlarm }} ·
                漏报 {{ filmReport.review.missed }}（漏报没有 issue 行可挂，不进分母）· 命中率
                {{ filmReport.review.hitRate == null ? '判不了（还没人复核过）' : pct(filmReport.review.hitRate) }}
              </span>
            </div>

            <table v-if="filmReport.defectCodes.length" class="kl-table kl-mt">
              <thead><tr><th>缺陷码</th><th>命中镜数</th></tr></thead>
              <tbody>
                <tr v-for="c in filmReport.defectCodes" :key="c.code">
                  <td class="kl-mono">{{ c.code }}</td>
                  <td>{{ c.count }}</td>
                </tr>
              </tbody>
            </table>

            <p v-if="filmReport.notes.length" class="kl-note kl-mt">
              {{ filmReport.notes.join('；') }}
            </p>
            <p class="kl-note">
              {{ filmReport.recorded ? '已落一条 eval 跑次（同一集重评是覆盖，不是并列）。' : '这条没有落库（记账失败或本次不记）。' }}
            </p>
          </div>

          <pre v-if="lastAction" class="kl-out">{{ lastAction }}</pre>
        </div>
      </section>
    </div>

    <!-- 条目详情 -->
    <div v-if="opened" class="kl-mask" @click.self="opened = null">
      <div class="kl-drawer">
        <div class="kl-drawer-head">
          <span :class="['kb-pill', statusPillClass(opened.status)]">{{ statusLabel(opened.status) }}</span>
          <span class="kb-pill">{{ kindLabel(opened.kind) }}</span>
          <button class="kb-mini" @click="opened = null">关闭</button>
        </div>
        <h3 class="kl-drawer-title">{{ opened.title }}</h3>
        <p class="kl-drawer-body">{{ opened.body }}</p>
        <div class="kl-kv"><span class="kl-kv-key">范围</span><span class="kl-mono">{{ opened.scope }}</span></div>
        <div class="kl-kv"><span class="kl-kv-key">置信度</span><span>{{ opened.confidence }}</span></div>
        <div class="kl-kv"><span class="kl-kv-key">佐证/反证</span><span>{{ opened.support }} / {{ opened.contra }}</span></div>
        <div class="kl-kv">
          <span class="kl-kv-key">lift</span>
          <span>{{ opened.lift == null ? '样本不足（不是 0）' : opened.lift.toFixed(3) }}</span>
        </div>
        <div v-if="opened.source_ref" class="kl-kv">
          <span class="kl-kv-key">来源</span><span class="kl-mono">{{ opened.source_kind }} · {{ opened.source_ref }}</span>
        </div>
        <div v-if="opened.usageSummary" class="kl-kv">
          <span class="kl-kv-key">被注入</span>
          <span>{{ opened.usageSummary.settled }} 次结算 · 成功率
            {{ opened.usageSummary.okRate == null ? '样本不足' : pct(opened.usageSummary.okRate) }}</span>
        </div>
        <div v-if="opened.evidence?.length" class="kl-evidence">
          <span class="kl-evidence-label">凭据</span>
          <code v-for="ev in opened.evidence" :key="ev" class="kl-code">{{ ev }}</code>
        </div>
        <div v-if="opened.usage?.length" class="kl-evidence">
          <span class="kl-evidence-label">注入记录</span>
          <code v-for="u in opened.usage.slice(0, 12)" :key="u.usedNs" class="kl-code">
            {{ u.runId || '（无 run）' }} · {{ u.outcome || '未结算' }}
          </code>
        </div>
        <div v-if="opened.tagList?.length" class="kl-evidence">
          <span class="kl-evidence-label">标签</span>
          <button v-for="t in opened.tagList" :key="t" class="kb-pill kl-facet" @click="toggleTag(t)">{{ t }}</button>
        </div>
        <div v-if="opened.links?.length" class="kl-evidence">
          <span class="kl-evidence-label">挂载</span>
          <code v-for="l in opened.links" :key="`${l.refKind}:${l.refId}`" class="kl-code">{{ l.refKind }}:{{ l.refId }}</code>
        </div>
        <div v-if="related?.items?.length" class="kl-evidence">
          <span class="kl-evidence-label">相关知识</span>
          <button v-for="item in related.items" :key="`${item.via}:${item.entry.id}`"
                  class="kb-pill kl-facet" :title="item.entry.snippet"
                  @click="openEntry(item.entry.id, item.entry.deleted)">
            {{ item.via === 'shared' ? `共用 ${item.refKind}:${item.refId}` : `相似 ${(item.score ?? 0).toFixed(2)}` }}
            · {{ item.entry.title }}
          </button>
        </div>
        <div v-else-if="related?.hint" class="kl-evidence">
          <span class="kl-evidence-label">相关知识</span>
          <span class="kl-code">{{ related.hint }}</span>
        </div>
        <div class="kl-kv">
          <span class="kl-kv-key">版本</span>
          <span>v{{ opened.version }}<template v-if="opened.deleted"> · 已软删（{{ opened.deletedBy || '未知' }}）</template></span>
        </div>
        <div class="kl-drawer-acts">
          <template v-if="opened.deleted">
            <button class="btn btn-ghost kl-tool-btn" @click="restoreEntry(opened)">恢复</button>
            <button class="btn btn-ghost kl-tool-btn kl-bad" @click="removeEntry(opened, true)">彻底删除</button>
          </template>
          <template v-else>
            <button class="btn btn-ghost kl-tool-btn" @click="openEdit(opened)">编辑</button>
            <button class="btn btn-ghost kl-tool-btn" @click="openVersions(opened)">版本记录</button>
            <button class="btn btn-ghost kl-tool-btn" @click="toggleAudit(opened.id)">这条的审计</button>
            <button class="btn btn-ghost kl-tool-btn kl-bad" @click="removeEntry(opened, false)">删除</button>
          </template>
          <button v-for="next in nextStatuses(opened.status)" :key="next" class="btn btn-ghost kl-tool-btn" @click="changeStatus(opened, next)">
            {{ statusLabel(next) }}
          </button>
        </div>
      </div>
    </div>

    <!-- 新建条目 -->
    <div v-if="createOpen" class="kl-mask" @click.self="createOpen = false">
      <div class="kl-drawer">
        <div class="kl-drawer-head"><span class="kl-drawer-title-sm">新建条目</span><button class="kb-mini" @click="createOpen = false">关闭</button></div>
        <div class="kl-field-row">
          <label class="kl-field">
            <span class="kl-field-label">类型</span>
            <select v-model="createForm.kind" class="kb-filter on">
              <option v-for="k in KINDS" :key="k" :value="k">{{ kindLabel(k) }}</option>
            </select>
          </label>
          <label class="kl-field">
            <span class="kl-field-label">范围</span>
            <input v-model="createForm.scope" class="kb-name-input" placeholder="global">
          </label>
        </div>
        <label class="kl-field">
          <span class="kl-field-label">标题</span>
          <input v-model="createForm.title" class="kb-name-input" placeholder="一句话说清这条经验">
        </label>
        <label class="kl-field">
          <span class="kl-field-label">正文</span>
          <textarea v-model="createForm.body" class="kl-textarea" rows="5" placeholder="怎么做 / 为什么 / 失效边界" />
        </label>
        <label class="kl-field">
          <span class="kl-field-label">凭据（一行一个 run_id 或路径）</span>
          <textarea v-model="createForm.evidence" class="kl-textarea" rows="3" />
        </label>
        <p class="kl-note">
          同 (类型, 范围, 标题) 重复提交是**叠加**（support +1 并覆盖正文），不是替换式编辑；
          要改错请用条目上的「编辑」（PATCH，版本 +1、佐证数不动）。
        </p>
        <div class="kl-drawer-acts">
          <button class="btn btn-primary kl-tool-btn" :disabled="busy.create" @click="submitCreate">提交</button>
        </div>
      </div>
    </div>

    <!-- 编辑条目（PATCH：只改传了的字段；带 expectedVersion 做乐观锁） -->
    <div v-if="editTarget" class="kl-mask" @click.self="editTarget = null">
      <div class="kl-drawer">
        <div class="kl-drawer-head">
          <span class="kl-drawer-title-sm">编辑 · {{ editTarget.title }}</span>
          <button class="kb-mini" @click="editTarget = null">关闭</button>
        </div>
        <p class="kl-note">
          这是**改错**，不是「又一份证据」：版本 +1、佐证数不变。改前的样子会留成快照，
          随时能看／能回滚。
        </p>
        <label class="kl-field">
          <span class="kl-field-label">标题</span>
          <input v-model="editForm.title" class="kb-name-input">
        </label>
        <label class="kl-field">
          <span class="kl-field-label">正文</span>
          <textarea v-model="editForm.body" class="kl-textarea" rows="6" />
        </label>
        <label class="kl-field">
          <span class="kl-field-label">标签（逗号分隔，清空即删掉全部标签）</span>
          <input v-model="editForm.tags" class="kb-name-input" placeholder="节奏, 过曝">
        </label>
        <label class="kl-field">
          <span class="kl-field-label">凭据（一行一个；留空＝不动）</span>
          <textarea v-model="editForm.evidence" class="kl-textarea" rows="2" />
        </label>
        <label class="kl-field">
          <span class="kl-field-label">改动理由（进快照与审计）</span>
          <input v-model="editForm.reason" class="kb-name-input" placeholder="为什么改">
        </label>
        <p class="kl-note">
          乐观锁：本次基于 v{{ editForm.expectedVersion }}。若别人先改过，后端会回「版本冲突」
          并保留对方正文（不会静默覆盖）—— 此时重新打开这条再改。
        </p>
        <div class="kl-drawer-acts">
          <button class="btn btn-primary kl-tool-btn" :disabled="busy.edit" @click="submitEdit">保存</button>
        </div>
      </div>
    </div>

    <!-- 版本记录（列表不含正文，点开某一版才拉全文） -->
    <div v-if="versionsTarget" class="kl-mask" @click.self="versionsTarget = null">
      <div class="kl-drawer">
        <div class="kl-drawer-head">
          <span class="kl-drawer-title-sm">版本 · {{ versionsTarget.title }}</span>
          <span class="kb-pill">当前 v{{ versionsData?.current ?? versionsTarget.version }}</span>
          <button class="kb-mini" @click="versionsTarget = null">关闭</button>
        </div>
        <p v-if="versionsData?.available === false" class="kl-note">{{ versionsData.hint }}</p>
        <p v-else-if="!versionsData?.items?.length" class="kb-empty">还没改过（只有 v1）</p>
        <div v-for="v in versionsData?.items || []" :key="v.version" class="kl-version-row">
          <div class="kl-kv">
            <span class="kl-kv-key">v{{ v.version }}</span>
            <span>{{ v.title }}</span>
            <span class="kl-muted">{{ v.bodyLen }} 字 · {{ fmtTime(v.createdNs) }}</span>
            <span class="kl-mono">{{ v.editor || '未知' }}<template v-if="v.reason"> · {{ v.reason }}</template></span>
          </div>
          <div class="kl-entry-acts">
            <button class="kb-mini" @click="peekVersion(v.version)">看全文</button>
            <button class="kb-mini" @click="revertTo(v.version)">回滚到这一版</button>
          </div>
        </div>
        <div v-if="peeked" class="kl-evidence">
          <span class="kl-evidence-label">v{{ peeked.version }} 全文<template v-if="peeked.isCurrent">（当前值）</template></span>
          <p class="kl-drawer-body">{{ peeked.body }}</p>
        </div>
        <p class="kl-note">
          回滚**是一次普通编辑**：版本 +1、后续版本不会被删（所以还能再滚回来）。
        </p>
      </div>
    </div>

    <!-- 设为默认档 -->
    <div v-if="promoteTarget" class="kl-mask" @click.self="promoteTarget = null">
      <div class="kl-drawer">
        <div class="kl-drawer-head">
          <span class="kl-drawer-title-sm">设为默认：{{ promoteTarget.modelKey }}</span>
          <button class="kb-mini" @click="promoteTarget = null">关闭</button>
        </div>
        <p class="kl-note">这是唯一能改变出图行为的入口，所以要写依据与凭据。</p>
        <label class="kl-field">
          <span class="kl-field-label">依据</span>
          <textarea v-model="promoteForm.basis" class="kl-textarea" rows="3" placeholder="为什么这套权重更好" />
        </label>
        <label class="kl-field">
          <span class="kl-field-label">凭据（一行一个）</span>
          <textarea v-model="promoteForm.evidence" class="kl-textarea" rows="3" />
        </label>
        <div class="kl-drawer-acts">
          <button class="btn btn-primary kl-tool-btn" :disabled="busy.promote" @click="submitPromote">确认设为默认</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { toast } from 'vue-sonner'
import { useConfirm } from '~/composables/useConfirm'
import {
  dramaAPI,
  knowledgeLoopAPI,
  type FilmEvalReport,
  type KnowledgeAuditList,
  type KnowledgeCandidate,
  type KnowledgeEntry,
  type KnowledgeEntryDetail,
  type KnowledgeEntryList,
  type KnowledgeEntryVersion,
  type KnowledgeEntryVersionDetail,
  type KnowledgeLedgerStats,
  type KnowledgeLoopSearch,
  type KnowledgeRelated,
  type KnowledgeTaxonomy,
  type KnowledgeTunable,
} from '~/composables/useApi'

/** 后端 `knowledge_ledger.KINDS` 的同一份词表 —— 前端只说人话，不做二次归类 */
const KINDS = ['fact', 'work', 'recipe', 'lesson', 'failure', 'threshold', 'prompt_patch']
const KIND_LABEL: Record<string, string> = {
  fact: '事实',
  work: '作品偏好',
  recipe: '做法',
  lesson: '经验',
  failure: '失败',
  threshold: '阈值',
  prompt_patch: '提示词补丁',
}
const STATUSES = ['proposed', 'accepted', 'deprecated', 'rejected']
const STATUS_LABEL: Record<string, string> = {
  proposed: '待审',
  accepted: '已采纳',
  deprecated: '已废弃',
  rejected: '已驳回',
}
/**
 * 状态迁移白名单照抄 `knowledge_ledger.set_status`：后端只有这四条边，
 * 前端多给一个按钮就会收到 400 —— 按钮由这张表长出来，不手写。
 */
const TRANSITIONS: Record<string, string[]> = {
  proposed: ['accepted', 'rejected'],
  accepted: ['deprecated', 'rejected'],
  deprecated: ['accepted', 'rejected'],
  rejected: ['proposed'],
}

type ViewId = 'overview' | 'optimizations' | 'entries' | 'retrieval' | 'feed' | 'engine' | 'film'
const view = ref<ViewId>('overview')
const lastAction = ref('')
const busy = reactive({
  rebuild: false, distill: false, lift: false, feed: false, optimizations: false,
  export: false, datasets: false, tuning: false, create: false, suggest: false, promote: false,
  film: false, edit: false,
})
const loading = reactive({ report: true, entries: false, search: false })

const report = ref<any>({ runsTotal: 0, runs: { groupBy: 'stage', groups: [], total: 0, available: true } })
const sched = ref<any>({ enabled: false, schedule: {} })
const candidates = ref<KnowledgeCandidate[]>([])
const ledger = computed<KnowledgeLedgerStats>(() => report.value.ledger || { total: 0, available: true })
const runGroups = computed<any[]>(() => report.value.runs?.groups || [])

const navItems = computed<Array<{ id: ViewId; label: string; badge?: number; note?: string }>>(() => [
  { id: 'overview', label: '概览', note: '跑次事实 · 蒸馏候选' },
  { id: 'optimizations', label: '优化史', badge: optSummary.value.rounds || 0, note: '一版一行 · 判决≠生效' },
  { id: 'entries', label: '经验台账', badge: ledger.value.total || 0, note: '人审状态迁移' },
  { id: 'retrieval', label: '检索测试', note: 'bm25 / like 不可比' },
  { id: 'feed', label: '投喂与数据集', note: 'L1 预览 · L3 导出' },
  { id: 'engine', label: '调参与训练', note: '覆盖层 · 冒烟产物' },
  { id: 'film', label: '成片评测', note: '四维等权 · 判不了不进总分' },
])

const kindStatusRows = computed(() => {
  const rows = ledger.value.byKindStatus || []
  const map = new Map<string, Array<{ status: string; count: number }>>()
  for (const r of rows) {
    if (!map.has(r.kind)) map.set(r.kind, [])
    map.get(r.kind)!.push({ status: r.status, count: r.count })
  }
  return [...map.entries()].map(([kind, statuses]) => ({ kind, statuses }))
})

function countOf(status: string) {
  return (ledger.value.byKindStatus || [])
    .filter((r) => r.status === status)
    .reduce((n, r) => n + r.count, 0)
}

// ── 概览 ────────────────────────────────────────────────────────────────
async function loadReport(includeCandidates = false) {
  loading.report = true
  try {
    const res = await knowledgeLoopAPI.report(includeCandidates)
    report.value = res || {}
    if (includeCandidates) {
      candidates.value = (res as any)?.candidates || []
      toast.success(`候选 ${candidates.value.length} 条`)
    }
  } catch (err: any) {
    toast.error(err?.message || '闭环总览读取失败')
  } finally {
    loading.report = false
  }
}
async function loadScheduler() {
  try {
    sched.value = (await knowledgeLoopAPI.scheduler()) || {}
  } catch (err: any) {
    toast.error(err?.message || '调度状态读取失败')
  }
}
async function rebuildRuns() {
  busy.rebuild = true
  try {
    lastAction.value = dump(await knowledgeLoopAPI.rebuildRuns({}))
    toast.success('索引已重建')
    await loadReport()
  } catch (err: any) {
    toast.error(err?.message || '重建失败')
  } finally {
    busy.rebuild = false
  }
}
async function distill(dryRun: boolean) {
  busy.distill = true
  try {
    lastAction.value = dump(await knowledgeLoopAPI.distill({ dryRun, limit: 20 }))
    toast.success(dryRun ? '预览完成（未落盘）' : '已写入提案')
    if (!dryRun) await loadReport()
  } catch (err: any) {
    toast.error(err?.message || '蒸馏失败')
  } finally {
    busy.distill = false
  }
}
async function refreshLift() {
  busy.lift = true
  try {
    lastAction.value = dump(await knowledgeLoopAPI.refreshLift({}))
    toast.success('lift 已重算')
    await loadReport()
  } catch (err: any) {
    toast.error(err?.message || 'lift 重算失败')
  } finally {
    busy.lift = false
  }
}

// ── 优化史：回答「这次改动到底提升了多少」 ───────────────────────────────
//
// 后端这张表**只读不写**（写入只发生在 `agent/evaluation/optimizer.py` 内部）：HTTP 手写一行
// 「提升了多少」就是编的，也没法核。前端只负责把三个数**分开**摆出来 —— `rounds`（试了几轮）、
// `accepted`（分数严格更高）、`persisted`（真写成默认值），合成一个「成功率」会糊掉关键差别。
const optData = ref<any>({ available: true, items: [], summary: {} })
const optForm = reactive({ agentType: '', caseId: '', limit: 100 })
const optItems = computed<any[]>(() => optData.value.items || [])
const optSummary = computed<any>(() => optData.value.summary || {
  rounds: 0, accepted: 0, persisted: 0, scored: 0, bestDelta: null, bestVersion: null, lastAt: null,
})

async function loadOptimizations() {
  busy.optimizations = true
  try {
    const res = await knowledgeLoopAPI.optimizations({
      agentType: optForm.agentType.trim() || undefined,
      caseId: optForm.caseId.trim() || undefined,
      limit: optForm.limit,
    })
    optData.value = res || { available: false, reason: '后端没有返回内容', items: [], summary: {} }
  } catch (err: any) {
    // 读失败也把 available 打 false：不能留着上一批数据装作「还是这些」
    optData.value = { available: false, reason: err?.message || '优化史读取失败', items: [], summary: {} }
    toast.error(err?.message || '优化史读取失败')
  } finally {
    busy.optimizations = false
  }
}
/** Δ 的人话：`null` 就是「算不了」（首版没有对手），绝不显示成 0 —— 那是编数 */
function signed(v: any) {
  if (v == null || !Number.isFinite(Number(v))) return '—'
  const n = Number(v)
  return `${n > 0 ? '+' : ''}${n.toFixed(3)}`
}
function deltaClass(v: any) {
  if (v == null || !Number.isFinite(Number(v))) return ''
  return Number(v) > 0 ? 'kl-good' : Number(v) < 0 ? 'kl-bad' : ''
}
/** 前 → 后：任一端为空写「—」，不补 0（补 0 会被读成「原本是 0 分」） */
function scorePair(r: any) {
  const f = (v: any) => (v == null || !Number.isFinite(Number(v)) ? '—' : Number(v).toFixed(3))
  return `${f(r.scoreBefore)} → ${f(r.scoreAfter)}`
}

// ── 台账 ────────────────────────────────────────────────────────────────
const pageSize = 20
const entryOffset = ref(0)
/**
 * `tags` 是**多标签 AND**；`includeDeleted` 是回收站视图。
 * 这两参数必须真的发出去（后端收下就生效）—— 传了不生效等于给用户一个假筛选。
 */
const filter = reactive({ kind: '', status: 'proposed', scope: '', tags: '', includeDeleted: false })
const entryList = ref<KnowledgeEntryList>({
  total: 0, items: [], limit: pageSize, offset: 0, available: true,
} as KnowledgeEntryList)
const opened = ref<KnowledgeEntryDetail | null>(null)

/** 操作者/角色：`role` 是声明式的（本仓没有登录态），只用于让审计行自洽 */
const ROLES = ['viewer', 'editor', 'owner']
const actor = reactive({ operator: '', role: '' })

const activeTags = computed(() => filter.tags.split(/[,，\s]+/).filter(Boolean))
const taxonomy = ref<KnowledgeTaxonomy | null>(null)

/**
 * 类型下拉的选项：优先用 `/taxonomy` 下发的 kind 闭集＋中文名＋计数。
 * 页面里那份 `kindLabel` 只作**离线兜底** —— 后端新增一个 kind 时，
 * 拿 taxonomy 的版本会自动多出一项，写死的那份会静默漏掉。
 */
const kindOptions = computed(() => (taxonomy.value?.kinds?.length
  ? taxonomy.value.kinds.map(k => ({ kind: k.kind, label: `${k.label} · ${k.count}` }))
  : KINDS.map(k => ({ kind: k, label: kindLabel(k) }))))

async function loadTaxonomy() {
  try {
    taxonomy.value = await knowledgeLoopAPI.taxonomy({ includeDeleted: filter.includeDeleted })
  } catch {
    //: 分类导航是锦上添花：读不到就不渲染 facet，不弹错（列表本身还能用）
    taxonomy.value = null
  }
}

/** 点标签 = 切换 AND 条件（不是替换），与后端「同时具备」语义一致 */
function toggleTag(tag: string) {
  const set = new Set(activeTags.value)
  if (set.has(tag)) set.delete(tag)
  else set.add(tag)
  filter.tags = [...set].join(',')
  filter.includeDeleted = false
  view.value = 'entries'
  loadEntries(0)
  loadTaxonomy()
}

async function loadEntries(offset = entryOffset.value) {
  loading.entries = true
  entryOffset.value = Math.max(0, offset)
  try {
    entryList.value = await knowledgeLoopAPI.entries({
      kind: filter.kind || undefined,
      status: filter.status || undefined,
      scope: filter.scope.trim() || undefined,
      tags: filter.tags.trim() || undefined,
      includeDeleted: filter.includeDeleted || undefined,
      limit: pageSize,
      offset: entryOffset.value,
    })
  } catch (err: any) {
    toast.error(err?.message || '清单读取失败')
  } finally {
    loading.entries = false
  }
  loadTaxonomy()
}
/**
 * 邻居（`/entries/{id}/related`）。两档 `via` 的**证据强度不同**，界面上一眼分得出来：
 * `shared` 是事实（共用同一 stage/shot），`similar` 只是正文像 —— 把字面像当"同一件事"
 * 去合并，会把这本台账最值钱的分歧抹掉，所以两档不许混成一条分数显示。
 */
const related = ref<KnowledgeRelated | null>(null)

async function openEntry(id: string, includeDeleted = false) {
  opened.value = null
  related.value = null
  try {
    opened.value = await knowledgeLoopAPI.entry(id, includeDeleted)
  } catch (err: any) {
    toast.error(err?.message || '条目详情读取失败')
    return
  }
  // 邻居是**附加信息**：读失败不该把已经拿到的详情一起丢掉
  try {
    related.value = await knowledgeLoopAPI.relatedEntries(id, { includeDeleted })
  } catch (err: any) {
    related.value = {
      id, found: true, deleted: false, available: false, links: [],
      items: [], counts: { shared: 0, similar: 0 },
      hint: err?.message || '邻居读取失败',
    }
  }
}

// ── 编辑 / 删除 / 恢复 / 版本 ───────────────────────────────────────────

const editTarget = ref<KnowledgeEntry | KnowledgeEntryDetail | null>(null)
const editForm = reactive({
  title: '', body: '', tags: '', evidence: '', reason: '', expectedVersion: 1, confidence: 0,
})

function openEdit(entry: KnowledgeEntry | KnowledgeEntryDetail) {
  editTarget.value = entry
  editForm.title = entry.title
  editForm.body = entry.body
  editForm.tags = (entry as any).tags || ''
  editForm.evidence = ''
  editForm.reason = ''
  editForm.expectedVersion = entry.version || 1
}

async function submitEdit() {
  const target = editTarget.value
  if (!target) return
  if (!editForm.title.trim() || !editForm.body.trim()) {
    toast.error('标题与正文都不能为空（要废弃请改状态或删除）')
    return
  }
  busy.edit = true
  try {
    const res = await knowledgeLoopAPI.updateEntry(target.id, {
      title: editForm.title.trim(),
      body: editForm.body.trim(),
      tags: editForm.tags,
      evidence: lines(editForm.evidence).length ? lines(editForm.evidence) : undefined,
      expectedVersion: editForm.expectedVersion,
      reason: editForm.reason.trim() || undefined,
      operator: actor.operator.trim() || undefined,
      role: actor.role || undefined,
    })
    if (res?.ok === false) {
      //: `conflict` ⇒ 别人先改了；此时**正文没有被覆盖**，必须重新读取再改
      toast.error(res.why || (res.code === 'conflict' ? '版本冲突：别人先改过' : '编辑被拒'))
      return
    }
    editTarget.value = null
    toast.success(`已保存（v${res.version ?? '?'}）`)
    await loadEntries()
    if (opened.value?.id === target.id) await openEntry(target.id)
  } catch (err: any) {
    toast.error(err?.message || '保存失败')
  } finally {
    busy.edit = false
  }
}

/**
 * 删除。默认软删（可恢复）；`hard=true` 是物理删除，必须**先软删**且角色 owner。
 * 前端不替后端做判断：直接把硬删按钮摆在回收站里，让 409 的 `why` 说话。
 */
async function removeEntry(entry: KnowledgeEntry | KnowledgeEntryDetail, hard: boolean) {
  const { confirm } = useConfirm()
  const ok = await confirm({
    title: hard ? '彻底删除' : '删除条目',
    message: hard
      ? `「${entry.title}」将被物理删除，回收站里也找不回来（审计账保留）。确认？`
      : `「${entry.title}」进回收站，可随时恢复。确认？`,
    danger: true,
    confirmText: hard ? '彻底删除' : '删除',
  })
  if (!ok) return
  try {
    const res = await knowledgeLoopAPI.deleteEntry(entry.id, {
      hard,
      operator: actor.operator.trim() || undefined,
      role: actor.role || undefined,
    })
    if (res?.ok === false) {
      toast.error(res.why || '删除被拒')
      return
    }
    opened.value = null
    toast.success(hard ? '已彻底删除（审计保留）' : '已进回收站')
    await loadEntries()
    await loadReport()
    if (auditOpen.value) await loadAudit()
  } catch (err: any) {
    toast.error(err?.message || '删除失败')
  }
}

async function restoreEntry(entry: KnowledgeEntry | KnowledgeEntryDetail) {
  try {
    const res = await knowledgeLoopAPI.restoreEntry(entry.id, {
      operator: actor.operator.trim() || undefined,
      role: actor.role || undefined,
    })
    if (res?.ok === false) {
      toast.error(res.why || '恢复被拒')
      return
    }
    toast.success(res?.changed === false ? '它本来就没删' : '已恢复')
    opened.value = null
    await loadEntries()
    await loadReport()
  } catch (err: any) {
    toast.error(err?.message || '恢复失败')
  }
}

const versionsTarget = ref<KnowledgeEntry | KnowledgeEntryDetail | null>(null)
const versionsData = ref<{ available: boolean; current: number; items: KnowledgeEntryVersion[]; hint?: string } | null>(null)
const peeked = ref<KnowledgeEntryVersionDetail | null>(null)

async function openVersions(entry: KnowledgeEntry | KnowledgeEntryDetail) {
  versionsTarget.value = entry
  peeked.value = null
  try {
    versionsData.value = await knowledgeLoopAPI.entryVersions(entry.id)
  } catch (err: any) {
    toast.error(err?.message || '版本读取失败')
    versionsData.value = null
  }
}

async function peekVersion(version: number) {
  if (!versionsTarget.value) return
  try {
    peeked.value = await knowledgeLoopAPI.entryVersion(versionsTarget.value.id, version)
  } catch (err: any) {
    toast.error(err?.message || '这一版读不出来')
  }
}

/** 回滚＝一次普通编辑（version+1，后续版本仍在）；不是「清空后面的历史」 */
async function revertTo(version: number) {
  if (!versionsTarget.value) return
  const { confirm } = useConfirm()
  const ok = await confirm({
    title: `回滚到 v${version}`,
    message: '回滚会生成一个新版本，而不是删掉后面几版 —— 之后还能再滚回来。确认？',
    confirmText: '回滚',
  })
  if (!ok) return
  try {
    const res = await knowledgeLoopAPI.revertEntry(versionsTarget.value.id, {
      version,
      reason: `回滚到 v${version}`,
      operator: actor.operator.trim() || undefined,
      role: actor.role || undefined,
    })
    if (res?.ok === false) {
      toast.error(res.why || '回滚被拒')
      return
    }
    toast.success(`已回滚（现 v${res.version ?? '?'}）`)
    await openVersions(versionsTarget.value)
    await loadEntries()
    if (opened.value) await openEntry(opened.value.id)
  } catch (err: any) {
    toast.error(err?.message || '回滚失败')
  }
}

// ── 审计 ────────────────────────────────────────────────────────────────
const auditOpen = ref(false)
const auditList = ref<KnowledgeAuditList | null>(null)

async function loadAudit(entryId?: string) {
  try {
    auditList.value = await knowledgeLoopAPI.audit({ entryId, limit: 50 })
    auditOpen.value = true
  } catch (err: any) {
    toast.error(err?.message || '审计读取失败')
  }
}

function toggleAudit(entryId?: string) {
  if (auditOpen.value && !entryId) {
    auditOpen.value = false
    return
  }
  loadAudit(entryId)
}
function nextStatuses(status: string) { return TRANSITIONS[status] || [] }

async function changeStatus(entry: Pick<KnowledgeEntry, 'id' | 'status' | 'title'>, next: string) {
  const { confirm } = useConfirm()
  if (next === 'deprecated' || next === 'rejected') {
    const ok = await confirm({
      title: `确认${STATUS_LABEL[next]}`,
      message: `「${entry.title}」将不再注入 Agent。确认？`,
      danger: true,
      confirmText: STATUS_LABEL[next],
    })
    if (!ok) return
  }
  try {
    await knowledgeLoopAPI.setEntryStatus(entry.id, next)
    toast.success(`已${STATUS_LABEL[next]}`)
    if (opened.value?.id === entry.id) opened.value = { ...opened.value, status: next }
    await loadEntries()
    await loadReport()
  } catch (err: any) {
    toast.error(err?.message || '状态迁移被拒')
  }
}

const createOpen = ref(false)
const createForm = reactive({ kind: 'lesson', scope: 'global', title: '', body: '', evidence: '' })
function openCreate() {
  createForm.title = ''
  createForm.body = ''
  createForm.evidence = ''
  createOpen.value = true
}
async function submitCreate() {
  if (!createForm.title.trim() || !createForm.body.trim()) {
    toast.error('标题与正文都不能为空')
    return
  }
  busy.create = true
  try {
    const res = await knowledgeLoopAPI.createEntry({
      kind: createForm.kind,
      scope: createForm.scope.trim() || 'global',
      title: createForm.title.trim(),
      body: createForm.body.trim(),
      evidence: lines(createForm.evidence),
    })
    if (res?.ok === false) {
      toast.error(res.why || '条目被拒')
      return
    }
    createOpen.value = false
    toast.success(res?.created === false ? '已叠加到既有条目（support +1）' : '已写入')
    await loadEntries(0)
    await loadReport()
  } catch (err: any) {
    toast.error(err?.message || '写入失败')
  } finally {
    busy.create = false
  }
}

// ── 检索测试 ─────────────────────────────────────────────────────────────
const searchForm = reactive({ q: '', kind: '', status: '', scope: '', limit: 10 })
const searchResult = ref<KnowledgeLoopSearch>({ query: '', ranking: 'none', total: 0, items: [], limit: 10, available: true })
const searched = ref(false)

async function runSearch() {
  if (!searchForm.q.trim()) {
    toast.error('先写查询词')
    return
  }
  loading.search = true
  try {
    searchResult.value = await knowledgeLoopAPI.search({
      q: searchForm.q.trim(),
      kind: searchForm.kind || undefined,
      status: searchForm.status || undefined,
      scope: searchForm.scope.trim() || undefined,
      limit: searchForm.limit || 10,
    })
    searched.value = true
  } catch (err: any) {
    toast.error(err?.message || '检索失败')
  } finally {
    loading.search = false
  }
}

// ── 投喂与数据集 ─────────────────────────────────────────────────────────
const feedForm = reactive({
  agentType: '', episodeId: '' as number | '', dramaId: '' as number | '',
  budgetChars: 600, limit: 8,
})
const feed = ref<any>(null)
const exportForm = reactive({ target: 'knowledge' as 'knowledge' | 'runs', limit: 500 })
const datasets = ref<any[]>([])
const datasetsAvailable = ref(true)

async function loadFeed() {
  busy.feed = true
  try {
    feed.value = await knowledgeLoopAPI.feedPreview({
      agentType: feedForm.agentType.trim() || undefined,
      episodeId: Number(feedForm.episodeId) || undefined,
      dramaId: Number(feedForm.dramaId) || undefined,
      budgetChars: feedForm.budgetChars,
      limit: feedForm.limit,
    })
  } catch (err: any) {
    toast.error(err?.message || '编译预览失败')
  } finally {
    busy.feed = false
  }
}
async function loadDatasets() {
  busy.datasets = true
  try {
    const res = await knowledgeLoopAPI.datasets()
    datasets.value = res?.items || []
    datasetsAvailable.value = res?.available !== false
  } catch (err: any) {
    toast.error(err?.message || '数据集列表读取失败')
  } finally {
    busy.datasets = false
  }
}
async function runExport() {
  busy.export = true
  try {
    lastAction.value = dump(await knowledgeLoopAPI.feedExport({ target: exportForm.target, limit: exportForm.limit }))
    toast.success('已导出')
    await loadDatasets()
  } catch (err: any) {
    toast.error(err?.message || '导出失败')
  } finally {
    busy.export = false
  }
}
async function resolveDataset(ref: string) {
  try {
    lastAction.value = dump(await knowledgeLoopAPI.datasetResolve(ref))
  } catch (err: any) {
    toast.error(err?.message || '核对失败')
  }
}

// ── 调参与训练 ───────────────────────────────────────────────────────────
const tuning = ref<{ path?: string; exists?: boolean; fields?: KnowledgeTunable[]; problems?: string[] }>({})
const overrideCount = computed(() => (tuning.value.fields || []).filter((f) => f.source !== 'default').length)
const suggestForm = reactive({ minSamples: 30 })
const suggest = ref<any>({ candidates: [], notEvaluated: [], minSamples: 30, available: true })
const tuneForm = reactive({ stage: '', field: '', value: '', basis: '', evidence: '' })

async function loadTuning() {
  try {
    tuning.value = await knowledgeLoopAPI.tuning()
  } catch (err: any) {
    toast.error(err?.message || '覆盖层读取失败')
  }
}
async function loadSuggest() {
  busy.suggest = true
  try {
    suggest.value = await knowledgeLoopAPI.tuningSuggest(suggestForm.minSamples)
  } catch (err: any) {
    toast.error(err?.message || '候选计算失败')
  } finally {
    busy.suggest = false
  }
}
function prefillTuning(c: KnowledgeCandidate) {
  tuneForm.stage = String(c.stage || '')
  tuneForm.field = String(c.field || '')
  tuneForm.value = c.proposed == null ? '' : String(c.proposed)
  tuneForm.basis = c.why || ''
  tuneForm.evidence = (c.evidence || []).join('\n')
}
async function submitTuning() {
  if (!tuneForm.stage.trim() || !tuneForm.field.trim()) {
    toast.error('stage 与 field 都要填')
    return
  }
  const evidence = lines(tuneForm.evidence)
  if (!tuneForm.basis.trim() || !evidence.length) {
    toast.error('依据与凭据都不能空 —— 后端会整条拒')
    return
  }
  busy.tuning = true
  try {
    lastAction.value = dump(await knowledgeLoopAPI.setTuning({
      stage: tuneForm.stage.trim(),
      field: tuneForm.field.trim(),
      value: parseValue(tuneForm.value),
      basis: tuneForm.basis.trim(),
      evidence,
    }))
    toast.success('已写入覆盖')
    await loadTuning()
  } catch (err: any) {
    toast.error(err?.message || '覆盖被拒')
  } finally {
    busy.tuning = false
  }
}
async function clearTuningRow(f: KnowledgeTunable) {
  const { confirm } = useConfirm()
  const ok = await confirm({ message: `清除 ${f.stage}.${f.field} 的覆盖？`, danger: true })
  if (!ok) return
  try {
    await knowledgeLoopAPI.clearTuning({ stage: f.stage, field: f.field })
    toast.success('已清除')
    await loadTuning()
  } catch (err: any) {
    toast.error(err?.message || '清除失败')
  }
}

// ── 质检阈值命中率与人工复核 ─────────────────────────────────────────────
//
// 这一块补的是 `suggest.notEvaluated` 里那条 `qc-threshold-hit-rate`：光有「触发频次」算不出
// 命中率（那是分子），分母必须是**人看过成片之后的结论**。所以这里有个写入口 ——
// 与整页其它「只读候选」的部分不同，但写进去的只是「我看到了什么」，不是参数改动。
const qcRate = ref<any>({ available: true, codes: [], records: {}, notes: [], minReviews: 10 })
const qcPending = ref<any>({ items: [], unreviewed: 0, verdicts: {}, minReviews: 10 })
/** `checkId:code` → 备注（可选，人写给人看的一句话） */
const qcNotes = reactive<Record<string, string>>({})
const busyQc = reactive<Record<string, boolean>>({})

const VERDICT_ORDER = ['hit', 'false_alarm', 'missed'] as const
const VERDICT_SHORT: Record<string, string> = { hit: '判对', false_alarm: '误报', missed: '漏报' }
/** 统计结论的人话（`insufficient` 不是"没问题"，是"还看不出来"） */
const QC_VERDICT_LABEL: Record<string, string> = {
  insufficient: '样本不足',
  false_alarm_heavy: '误报偏多',
  miss_heavy: '漏报偏多',
  ok: '在容忍范围',
}
const QC_VERDICT_CLASS: Record<string, string> = {
  insufficient: '',
  false_alarm_heavy: 'kb-pill-warn',
  miss_heavy: 'kb-pill-warn',
  ok: 'kb-pill-ok',
}

function qcKey(checkId: number, code: string) { return `${checkId}:${code}` }

async function loadQc() {
  try {
    const [rate, pend] = await Promise.all([
      knowledgeLoopAPI.qcHitRate(),
      knowledgeLoopAPI.qcPending(20),
    ])
    qcRate.value = rate
    qcPending.value = pend
  } catch (err: any) {
    toast.error(err?.message || '质检复核读取失败')
  }
}
/** 判一条：`hit` 判对 / `false_alarm` 误报 / `missed` 该判没判。同 `(checkId, code)` 是改判 */
async function judge(item: any, issue: any, verdict: 'hit' | 'false_alarm' | 'missed') {
  const key = qcKey(item.checkId, issue.code)
  busyQc[key] = true
  try {
    lastAction.value = dump(await knowledgeLoopAPI.qcReview({
      checkId: item.checkId,
      code: issue.code,
      verdict,
      note: qcNotes[key] || '',
      storyboardId: item.storyboardId ?? undefined,
      videoGenerationId: item.videoGenerationId ?? undefined,
    }))
    toast.success('已记录复核')
    await loadQc()
    // 命中率变了 ⇒ 候选跟着变（不然要人自己再点一次「重算」）
    await loadSuggest()
  } catch (err: any) {
    toast.error(err?.message || '复核被拒')
  } finally {
    busyQc[key] = false
  }
}
async function undoJudge(item: any, issue: any) {
  const reviewId = issue.review?.id
  if (!reviewId) return
  const key = qcKey(item.checkId, issue.code)
  busyQc[key] = true
  try {
    await knowledgeLoopAPI.deleteQcReview(reviewId)
    toast.success('已撤销')
    await loadQc()
    await loadSuggest()
  } catch (err: any) {
    toast.error(err?.message || '撤销失败')
  } finally {
    busyQc[key] = false
  }
}

const training = ref<any>({ jobs: [], models: [], problems: [] })
const promoteTarget = ref<any>(null)
const promoteForm = reactive({ basis: '', evidence: '' })

async function loadTraining() {
  try {
    training.value = (await knowledgeLoopAPI.training({ limit: 20 })) || { jobs: [], models: [], problems: [] }
  } catch (err: any) {
    toast.error(err?.message || '训练登记读取失败')
  }
}
async function scanTraining(taskId: string) {
  try {
    lastAction.value = dump(await knowledgeLoopAPI.trainingScan(taskId))
    await loadTraining()
  } catch (err: any) {
    toast.error(err?.message || '重扫失败')
  }
}
function openPromote(m: any) {
  promoteTarget.value = m
  promoteForm.basis = ''
  promoteForm.evidence = ''
}
async function submitPromote() {
  if (!promoteTarget.value) return
  const evidence = lines(promoteForm.evidence)
  if (!promoteForm.basis.trim() || !evidence.length) {
    toast.error('依据与凭据都不能空')
    return
  }
  busy.promote = true
  try {
    lastAction.value = dump(await knowledgeLoopAPI.promoteModel(promoteTarget.value.modelId, {
      basis: promoteForm.basis.trim(),
      evidence,
    }))
    promoteTarget.value = null
    toast.success('已设为默认档')
    await loadTraining()
  } catch (err: any) {
    toast.error(err?.message || '设为默认被拒')
  } finally {
    busy.promote = false
  }
}
async function demote(m: any) {
  const { confirm } = useConfirm()
  const ok = await confirm({ message: `撤销 ${m.modelKey} 的默认档？`, danger: true })
  if (!ok) return
  try {
    lastAction.value = dump(await knowledgeLoopAPI.demoteModel(m.modelId))
    toast.success('已撤销')
    await loadTraining()
  } catch (err: any) {
    toast.error(err?.message || '撤销失败')
  }
}

// ── 成片评测（缺口 8：评"成片好坏"）──────────────────────────────────────
//
// 两级选择：剧组 → 集。**不做"只有一部剧就默认选中"**：本仓可以有多部剧，
// 默认选中会让创作者在自己没看清评的是哪一集时就把结论记进闭环。
const filmDramaId = ref(0)
const filmEpisodeId = ref(0)
const filmDramas = ref<Array<{ id: number; title: string }>>([])
const filmEpisodes = ref<Array<{ id: number; episode_number: number; title?: string }>>([])
const filmReport = ref<FilmEvalReport | null>(null)

//: `verdict` 只表达"够不够格交片"。`partial` **不是**"及格"也不是"不及格" ——
//: 它的意思是"有几维根本没数据"，所以单独给一个中性色，别混进通过/不通过。
const VERDICT_LABEL: Record<string, string> = {
  empty: '这集还没分镜',
  partial: '有维度判不了',
  blocked: '低于交片线',
  ready: '够格交片',
}
const DIM_LABEL: Record<string, string> = {
  coverage: '出片覆盖', spec: '技术规格', defects: '缺陷面', review: '人工复核',
}

async function loadFilmDramas() {
  if (filmDramas.value.length) return
  try {
    const res = await dramaAPI.list()
    filmDramas.value = (res.items || []).map((d: any) => ({ id: d.id, title: d.title }))
  } catch (err: any) {
    toast.error(err?.message || '剧组列表读取失败')
  }
}

async function loadFilmEpisodes() {
  filmEpisodeId.value = 0
  filmReport.value = null
  filmEpisodes.value = []
  if (!filmDramaId.value) return
  try {
    const drama = await dramaAPI.get(filmDramaId.value)
    filmEpisodes.value = (drama.episodes || []).map((e: any) => ({
      id: e.id, episode_number: e.episode_number, title: e.title,
    }))
  } catch (err: any) {
    toast.error(err?.message || '剧集列表读取失败')
  }
}

async function runFilmEval() {
  if (!filmEpisodeId.value) return
  busy.film = true
  try {
    filmReport.value = await knowledgeLoopAPI.filmEval(filmEpisodeId.value, true)
    lastAction.value = dump(filmReport.value)
    // 三种"不是结论"的结论各有各的说法，别一律 toast 成功 —— 那样创作者会以为"评过了就是好的"。
    if (filmReport.value.verdict === 'empty') toast.info('这集还没分镜：暂无成片可评（不是 0 分）')
    else if (filmReport.value.missing.length) toast.info(`有 ${filmReport.value.missing.length} 维没有样本，未参与总分`)
    else toast.success('评测完成')
  } catch (err: any) {
    toast.error(err?.message || '评测失败')
  } finally {
    busy.film = false
  }
}

// ── 懒加载：切到哪块才拉哪块，首屏只打两次 ───────────────────────────────
const loaded = reactive<Record<string, boolean>>({ overview: true })
watch(view, (v) => {
  if (loaded[v]) return
  loaded[v] = true
  if (v === 'optimizations') loadOptimizations()
  if (v === 'entries') loadEntries(0)
  //: facet 与列表同口径（同一套标签匹配），所以在进入台账页时一起拉
  if (v === 'entries') loadTaxonomy()
  if (v === 'film') loadFilmDramas()
  if (v === 'feed') { loadFeed(); loadDatasets() }
  if (v === 'engine') { loadTuning(); loadSuggest(); loadQc(); loadTraining() }
})
onMounted(async () => {
  await Promise.all([loadReport(false), loadScheduler()])
})

// ── 小工具 ──────────────────────────────────────────────────────────────
function kindLabel(k: string) { return KIND_LABEL[k] || k }
function statusLabel(s: string) { return STATUS_LABEL[s] || s }
function statusPillClass(s: string) {
  if (s === 'accepted') return 'kb-pill-ok'
  if (s === 'proposed') return 'kb-pill-warn'
  if (s === 'rejected') return 'kb-pill-bad'
  return ''
}
/**
 * 候选的「作用面」：两族候选给的字段不同 —— 蒸馏候选把阶段/镜头塞在 `stats` 里，
 * 调参候选直接给 `stage`。这里只做取值，不做猜测性归因。
 */
function candScope(c: KnowledgeCandidate): string {
  if (c.stage) return String(c.stage)
  const s = c.stats || {}
  return String(s.stage || s.shotId || c.key || '全局')
}
/** 候选的样本量：`samples` / `total` / `runs` 三个键谁有取谁，都没有就显示「—」 */
function candSamples(c: KnowledgeCandidate): string {
  if (typeof c.samples === 'number') return String(c.samples)
  const s = c.stats || {}
  for (const k of ['total', 'runs', 'samples']) {
    if (typeof s[k] === 'number') return String(s[k])
  }
  return '—'
}
/**
 * 候选的归因文字。**蒸馏候选没有 `why`，就不替它编** ——
 * 只用 `stats` 里真有的数字拼一句可核的，最后把「凭什么」指回下面那排 run_id。
 */
function candWhy(c: KnowledgeCandidate): string {
  if (c.why) return c.why
  const s = c.stats || {}
  const bits: string[] = []
  if (s.failRate != null) bits.push(`失败率 ${pct(Number(s.failRate))}`)
  if (s.failed != null && s.total != null) bits.push(`${s.failed}/${s.total} 条失败`)
  if (s.p10 != null && s.p90 != null) bits.push(`分数 p10 ${s.p10} / p90 ${s.p90}（跨度 ${s.gap ?? '—'}）`)
  if (s.p50Ms != null && s.p95Ms != null) bits.push(`耗时 p50 ${fmtMs(s.p50Ms)} / p95 ${fmtMs(s.p95Ms)}（${s.ratio ?? '—'}×）`)
  if (s.runs != null) bits.push(`同镜重跑 ${s.runs} 次`)
  const top = Array.isArray(s.topErrors) && s.topErrors.length
    ? `主要错误：${s.topErrors
      .map((e: any) => (Array.isArray(e) ? `${e[0]}×${e[1]}` : String(e)))
      .join(' / ')}`
    : ''
  const head = bits.length ? bits.join('，') : '后端没给文字归因（这条只有统计信号）'
  return top ? `${head}。${top}` : `${head}，凭据见下，逐条能在跑次索引里核`
}
/**
 * 时间：后端三种写法混用 —— 纳秒（`*_ns`）、毫秒（`*At`）、**ISO 字符串**
 * （`agent_optimizations.created_at` 落库就是 `2026-11-07T02:36:34.545Z`）。
 * 只认数字会让整列静默显示成「—」（真踩过：表里时间明明有值，页面上全是破折号）。
 * 空 / 0 / 解析不了才显示「—」，绝不显示 1970。
 */
function fmtTime(v: any) {
  if (v == null || v === 0 || v === '') return '—'
  if (typeof v === 'string') {
    const d = new Date(v)
    return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString('zh-CN', { hour12: false })
  }
  const n = Number(v)
  if (!Number.isFinite(n) || n <= 0) return '—'
  const ms = n > 1e15 ? n / 1e6 : n
  const d = new Date(ms)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleString('zh-CN', { hour12: false })
}
function fmtMs(v: any) {
  if (v == null || !Number.isFinite(Number(v))) return '—'
  const n = Number(v)
  return n >= 1000 ? `${(n / 1000).toFixed(2)}s` : `${Math.round(n)}ms`
}
function fmtBytes(v: any) {
  const n = Number(v)
  if (!Number.isFinite(n) || n <= 0) return '0 B'
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${n} B`
}
function pct(v: number) { return `${(v * 100).toFixed(1)}%` }
function verdictLabel(v: string) { return VERDICT_LABEL[v] || v }
/** `partial` 给中性色：它的意思是"有几维没数据"，既不是通过也不是不通过 */
function verdictPill(v: string) {
  if (v === 'ready') return 'kb-pill-ok'
  if (v === 'blocked') return 'kb-pill-bad'
  if (v === 'partial') return 'kb-pill-warn'
  return ''
}
function missingLabel(name: string) { return DIM_LABEL[name] || name }
function pad(v: any) { return String(v ?? 0).padStart(2, '0') }
function lines(text: string) {
  return text.split('\n').map((s) => s.trim()).filter(Boolean)
}
/** 值口先当数字，再试布尔，最后退回字符串 —— 与后端 `_coerce` 同序 */
function parseValue(raw: string): any {
  const s = raw.trim()
  if (!s) return ''
  if (s === 'true') return true
  if (s === 'false') return false
  const n = Number(s)
  if (Number.isFinite(n) && s !== '') return n
  return s
}
/** 回执原样透出（截断 3000 字）：宁可难读，也不要把「写了几条/跳了几条」揉成一句话后失真 */
function dump(res: any) {
  try {
    const t = JSON.stringify(res, null, 2)
    return t.length > 3000 ? `${t.slice(0, 3000)}\n…（截断）` : t
  } catch {
    return String(res)
  }
}
</script>

<style scoped>
/* 与「知识库」页共用同一套视觉词汇（kb-*）：两页相邻、同一分区，样式不能各长一套。
   这里是**作用域内的同名副本** —— scoped 样式不跨组件，所以必须自己写一遍。 */
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
  display: flex; align-items: center; gap: 6px; width: 100%;
  padding: 8px 9px; border-radius: 8px; border: 1px solid transparent;
  background: none; cursor: pointer; text-align: left; color: var(--text-1);
}
.kb-group:hover { background: var(--bg-surface); }
.kb-group.active { background: var(--bg-surface); border-color: var(--accent); color: var(--text-0); }
.kb-group-name { flex: 1; font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kb-group-num { font-size: 11px; color: var(--text-3); font-variant-numeric: tabular-nums; }
.kb-main { min-width: 0; min-height: 0; display: flex; flex-direction: column; }
.kb-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 16px 24px 0; }
.kb-filter {
  height: 30px; max-width: 200px; padding: 0 8px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-surface);
  color: var(--text-1); font-size: 11.5px; cursor: pointer;
}
.kb-filter.on { border-color: var(--accent); color: var(--text-0); }
.kb-name-input {
  flex: 1; min-width: 0; height: 30px; padding: 0 9px;
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-surface); color: var(--text-0); font-size: 12px; outline: none;
}
.kb-name-input:focus { border-color: var(--accent); }
.kb-list { flex: 1; min-height: 0; overflow-y: auto; padding: 8px 24px 28px; }
.kb-pill {
  display: inline-block; padding: 1px 6px; border-radius: 999px;
  font-size: 10px; font-style: normal; background: rgba(148, 163, 184, 0.16); color: var(--text-2);
}
.kb-pill-ok { background: rgba(52, 211, 153, 0.16); color: #34d399; }
.kb-pill-warn { background: rgba(251, 191, 36, 0.16); color: #fbbf24; }
.kb-pill-bad { background: rgba(248, 113, 113, 0.16); color: #f87171; }
.kb-mini {
  padding: 2px 7px; border-radius: 6px; font-size: 10.5px; cursor: pointer;
  border: 1px solid var(--border); background: none; color: var(--text-2); white-space: nowrap;
}
.kb-mini:hover:not(:disabled) { border-color: var(--accent); color: var(--text-0); }
.kb-mini:disabled { opacity: 0.45; cursor: not-allowed; }
.kb-mini-danger:hover:not(:disabled) { border-color: #f87171; color: #f87171; }
.kb-warn-inline {
  margin: 2px 0 10px; padding: 7px 11px; border-radius: 8px; font-size: 11.5px; line-height: 1.6;
  background: rgba(251, 191, 36, 0.1); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.28);
}
.kb-empty { padding: 40px; font-size: 13px; color: var(--text-2); }
.kb-empty-inline { padding: 18px 4px; font-size: 12.5px; color: var(--text-3); }
.kb-pager {
  display: flex; align-items: center; justify-content: flex-end; gap: 10px; padding: 12px 0 4px;
}
.kb-pager-info { font-size: 11px; color: var(--text-3); font-variant-numeric: tabular-nums; }

/* ── 本节特有 ──────────────────────────────────────────────────────────── */
.kl-kicker {
  font-size: 10px; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 6px;
}
.kl-nav-note {
  margin: 12px 6px 0; font-size: 10.5px; line-height: 1.7; color: var(--text-3);
}
.kl-section-title { font-size: 12px; font-weight: 600; color: var(--text-0); margin-right: auto; }
.kl-tool-btn { padding: 5px 11px; font-size: 11.5px; }
.kl-scope-input { max-width: 260px; flex: 0 1 260px; }
.kl-mini-input { max-width: 92px; flex: 0 0 92px; }

.kl-cards {
  display: grid; gap: 12px; padding: 10px 0 4px;
  grid-template-columns: repeat(auto-fit, minmax(min(320px, 100%), 1fr));
}
.kl-card {
  padding: 12px 14px; border-radius: 12px;
  border: 1px solid var(--border); background: var(--bg-surface);
}
.kl-card-wide { margin-top: 12px; }
.kl-card-head {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  font-size: 12px; font-weight: 600; color: var(--text-0); margin-bottom: 10px;
}
.kl-card-acts { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 10px; }

.kl-table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
.kl-table th {
  text-align: left; font-weight: 600; font-size: 10.5px; color: var(--text-3);
  padding: 4px 8px 6px; border-bottom: 1px solid var(--border); white-space: nowrap;
}
.kl-table td { padding: 6px 8px; border-bottom: 1px solid rgba(148, 163, 184, 0.1); color: var(--text-1); }
.kl-mt { margin-top: 14px; }
.kl-mono { font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace); font-size: 11px; }
.kl-wrap { max-width: 260px; overflow-wrap: anywhere; }
.kl-bad { color: #f87171; }
.kl-good { color: #34d399; }
.kl-muted { color: var(--text-3); }
.kl-ml { margin-left: 6px; }
.kl-note { margin: 8px 0 0; font-size: 10.5px; line-height: 1.7; color: var(--text-3); }
/* 「算不了的维度」逐条带原因 —— 只摆 id 的话，`qc-threshold-hit-rate` 被读成"有问题"，
   而它其实是"复核条数还没到门槛"，原因必须看得见才不至于误判 */
.kl-dim { display: flex; align-items: baseline; gap: 8px; padding: 2px 0; }
.kl-dim-why { flex: 1 1 auto; min-width: 0; color: var(--text-3); }
.kl-note-inline { font-size: 10.5px; color: var(--text-3); }
.kl-kv { display: flex; align-items: baseline; gap: 10px; padding: 3px 0; font-size: 11.5px; color: var(--text-1); }
.kl-kv-key { flex: 0 0 74px; color: var(--text-3); font-size: 10.5px; }
.kl-kv-val { display: flex; flex-wrap: wrap; gap: 5px; }

.kl-cand {
  padding: 9px 11px; margin-bottom: 8px; border-radius: 9px;
  border: 1px solid var(--border); background: var(--bg-app, rgba(148, 163, 184, 0.04));
}
.kl-cand-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 11px; color: var(--text-2); }
.kl-cand-samples { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--text-3); }
.kl-cand-why { margin: 6px 0 0; font-size: 11.5px; line-height: 1.65; color: var(--text-1); }

/* 人工复核：一行一条 (code, 结论)，备注与四个判定按钮同一行；窄屏自然折行 */
.kl-qc-head { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); }
.kl-qc-row {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
  padding: 6px 0; border-top: 1px solid rgba(148, 163, 184, 0.1);
}
.kl-qc-row:first-of-type { border-top: none; }
.kl-qc-row .kl-note { margin: 0; }
.kl-qc-note { flex: 1 1 140px; min-width: 0; height: 26px; font-size: 11.5px; }
.kl-evidence { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-top: 7px; }
.kl-evidence-label { font-size: 10px; color: var(--text-3); }
.kl-code {
  padding: 1px 5px; border-radius: 5px; font-size: 10px;
  background: rgba(148, 163, 184, 0.14); color: var(--text-2); overflow-wrap: anywhere;
}
.kl-out {
  margin: 12px 0 0; padding: 10px 12px; border-radius: 10px; max-height: 260px; overflow: auto;
  border: 1px solid var(--border); background: var(--bg-app, rgba(148, 163, 184, 0.05));
  font-size: 10.5px; line-height: 1.6; color: var(--text-1); white-space: pre-wrap; word-break: break-word;
}

.kl-list-meta {
  display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
  padding: 12px 0 6px; font-size: 11px; color: var(--text-2);
}
.kl-entry {
  padding: 11px 13px; margin-bottom: 8px; border-radius: 10px; cursor: pointer;
  border: 1px solid var(--border); background: var(--bg-surface);
}
.kl-entry:hover { border-color: var(--accent); }
.kl-entry-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.kl-entry-title { flex: 1; min-width: 0; font-size: 13px; font-weight: 600; color: var(--text-0); }
.kl-entry-body {
  margin: 7px 0 0; font-size: 12px; line-height: 1.7; color: var(--text-1);
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.kl-entry-foot {
  display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
  margin-top: 8px; font-size: 10.5px; color: var(--text-3);
}
.kl-entry-acts { margin-left: auto; display: flex; flex-wrap: wrap; gap: 5px; }
/*: facet / 回收站 / 版本 / 审计 —— 都是台账管理面的辅助控件 */
.kl-check {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11.5px; color: var(--text-2); cursor: pointer; white-space: nowrap;
}
.kl-facet-row { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0 2px; }
.kl-facet {
  border: 1px solid var(--line-1); background: transparent; color: var(--text-2);
  cursor: pointer; font: inherit;
}
.kl-facet.on { border-color: var(--accent); color: var(--accent); }
.kl-facet b { font-weight: 600; opacity: 0.7; margin-left: 2px; }
.kl-actor-row { margin-top: 6px; align-items: center; }
.kl-version-row {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 0; border-bottom: 1px dashed var(--line-1);
}
.kl-version-row:last-of-type { border-bottom: none; }
.kl-rank {
  min-width: 20px; font-size: 11px; font-weight: 700; color: var(--text-3);
  font-variant-numeric: tabular-nums;
}

.kl-test { flex: 1; min-height: 0; display: grid; grid-template-columns: 340px minmax(0, 1fr); gap: 0; }
.kl-test-form {
  border-right: 1px solid var(--border); padding: 16px 18px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 10px;
}
.kl-test-result { min-width: 0; overflow-y: auto; padding: 16px 20px 28px; }
.kl-result-head {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  font-size: 11.5px; color: var(--text-2); padding-bottom: 10px;
}
.kl-field { display: flex; flex-direction: column; gap: 5px; }
.kl-field-row { display: flex; gap: 8px; }
.kl-field-row .kl-field { flex: 1; min-width: 0; }
.kl-field-narrow { flex: 0 0 84px; }
.kl-field-label { font-size: 10.5px; color: var(--text-3); }
.kl-textarea {
  width: 100%; padding: 8px 10px; border-radius: 9px; resize: vertical;
  border: 1px solid var(--border); background: var(--bg-surface);
  color: var(--text-0); font-size: 12px; line-height: 1.7; font-family: inherit; outline: none;
}
.kl-textarea:focus { border-color: var(--accent); }
.kl-run { margin-top: 2px; }
.kl-inline-form { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 10px; }
.kl-inline-num { display: flex; align-items: center; gap: 6px; font-size: 10.5px; color: var(--text-3); }
.kl-inline-num input {
  width: 84px; height: 30px; padding: 0 8px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg-surface);
  color: var(--text-0); font-size: 12px; outline: none;
}

.kl-mask {
  position: fixed; inset: 0; z-index: 60; display: flex; justify-content: flex-end;
  background: rgba(0, 0, 0, 0.5);
}
.kl-drawer {
  width: min(480px, 100%); max-width: 100%; height: 100%; overflow-y: auto;
  display: flex; flex-direction: column; gap: 9px; padding: 16px 18px 28px;
  background: var(--bg-surface); border-left: 1px solid var(--border);
  box-shadow: -20px 0 40px rgba(0, 0, 0, 0.28);
}
.kl-drawer-head { display: flex; align-items: center; gap: 8px; }
.kl-drawer-head .kb-mini:last-child { margin-left: auto; }
.kl-drawer-title { margin: 2px 0 0; font-size: 14px; font-weight: 600; color: var(--text-0); }
.kl-drawer-title-sm { font-size: 13px; font-weight: 600; color: var(--text-0); flex: 1; }
.kl-drawer-body { margin: 0; font-size: 12.5px; line-height: 1.75; color: var(--text-1); white-space: pre-wrap; }
.kl-drawer-acts { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }

@media (max-width: 1023px) {
  .kb-head { padding: 20px 22px 14px; gap: 16px; }
  .kb-body { grid-template-columns: 184px minmax(0, 1fr); }
  .kl-test { grid-template-columns: 300px minmax(0, 1fr); }
}

@media (max-width: 767px) {
  .kb-head { flex-direction: column; align-items: stretch; gap: 12px; padding: 16px 14px 12px; }
  .kb-title { font-size: 18px; }
  .kb-stats { flex-wrap: wrap; gap: 12px; }
  .kb-stat { align-items: flex-start; }
  .kb-body { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr); }
  .kb-side {
    flex-direction: row; align-items: center; gap: 6px;
    border-right: none; border-bottom: 1px solid var(--border);
    padding: 8px 10px; overflow-x: auto; overflow-y: hidden; flex-shrink: 0;
  }
  .kb-side-head, .kl-nav-note { display: none; }
  .kb-group { flex-shrink: 0; width: auto; padding: 7px 10px; white-space: nowrap; }
  .kb-group-name { flex: 0 0 auto; max-width: 44vw; }
  .kb-toolbar { padding: 12px 14px 0; }
  .kb-list { padding: 6px 14px 24px; }
  .kb-name-input { flex: 1 1 100%; }
  .kb-pager { justify-content: space-between; flex-wrap: wrap; }
  /* 检索测试：左右并排会各自剩不到 200px，改成上下两段 */
  .kl-test { grid-template-columns: 1fr; grid-template-rows: auto minmax(0, 1fr); }
  .kl-test-form { border-right: none; border-bottom: 1px solid var(--border); }
  .kl-test-result { padding: 14px; }
  .kl-drawer { width: 100%; }
  .kl-wrap { max-width: none; }
}
</style>
