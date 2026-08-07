<template>
  <div class="lib-page">
    <div class="breadcrumb">
      <NuxtLink to="/library" class="bc-link">资源库</NuxtLink>
      <span class="bc-sep">/</span>
      <span class="bc-current">服装库</span>
    </div>

    <header class="page-header">
      <div>
        <h2>服装库</h2>
        <p>按风格、部位、材质维度分类，支持快速检索与预览</p>
      </div>
      <button class="btn-primary" @click="openCreate">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        新建服装
      </button>
    </header>

    <div class="toolbar">
      <div class="search-box">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text-3)" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input v-model="search" @input="onSearch" placeholder="搜索服装名称、外观、材质、配色..." />
        <button v-if="search" class="clear-btn" @click="search=''; onSearch()">×</button>
      </div>
      <div class="filter-row">
        <select v-model="filterCategory" @change="onFilter" class="filter-select">
          <option value="">全部分类</option>
          <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
        </select>
        <select v-model="filterStyle" @change="onFilter" class="filter-select">
          <option value="">全部风格</option>
          <option v-for="s in filterOpts.styles" :key="s" :value="s">{{ s }}</option>
        </select>
        <select v-model="filterBodyPart" @change="onFilter" class="filter-select">
          <option value="">全部部位</option>
          <option v-for="b in filterOpts.bodyParts" :key="b" :value="b">{{ b }}</option>
        </select>
        <div class="tag-chips">
          <button v-for="t in allTags.slice(0,15)" :key="t" class="tag-chip" :class="{ active: selectedTags.includes(t) }" @click="toggleTag(t)">{{ t }}</button>
        </div>
        <button v-if="selectedItems.length" class="btn-danger-outline" @click="batchDelete">删除 ({{ selectedItems.length }})</button>
      </div>
    </div>

    <div class="card-grid" v-if="items.length">
      <div
        v-for="item in items" :key="item.id"
        class="lib-card" :class="{ selected: selectedItems.includes(item.id) }"
        @click="toggleSelect(item.id)"
      >
        <div class="card-check"><input type="checkbox" :checked="selectedItems.includes(item.id)" @click.stop /></div>
        <div class="costume-swatch" :style="item.colorScheme ? { background: item.colorScheme.includes('#') ? item.colorScheme : `linear-gradient(135deg, ${item.colorScheme})` } : {}">
          <span v-if="!item.colorScheme && !item.imageUrl">👘</span>
        </div>
        <div class="card-body">
          <div class="card-top">
            <h4>{{ item.name }}</h4>
            <div class="costume-badges">
              <span class="costume-style-badge" v-if="item.style">{{ item.style }}</span>
              <span class="card-cat">{{ item.category }}</span>
            </div>
          </div>
          <p class="card-desc">{{ truncate(item.appearance || item.description, 80) }}</p>
          <div class="costume-attrs">
            <span v-if="item.bodyPart">部位: {{ item.bodyPart }}</span>
            <span v-if="item.material">材质: {{ item.material }}</span>
            <span v-if="item.season">季节: {{ item.season }}</span>
          </div>
          <div class="card-meta">
            <div class="card-tags">
              <span v-for="t in (item.tags || []).slice(0,3)" :key="t" class="mini-tag">{{ t }}</span>
            </div>
            <div class="card-actions" @click.stop>
              <button class="action-btn" @click="openEdit(item)">编辑</button>
              <button class="action-btn danger" @click="deleteItem(item)">删除</button>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="empty" v-else-if="!loading"><p>暂无服装模板</p><button class="btn-secondary" @click="openCreate">创建第一个服装</button></div>

    <div class="pager" v-if="totalPages > 1">
      <button :disabled="page <= 1" @click="page--; load()">上一页</button>
      <span>第 {{ page }} / {{ totalPages }} 页 (共 {{ total }} 条)</span>
      <button :disabled="page >= totalPages" @click="page++; load()">下一页</button>
    </div>

    <!-- 表单模态框 -->
    <div class="modal-overlay" v-if="showForm" @click.self="showForm = false">
      <div class="modal">
        <h3>{{ editingItem ? '编辑服装' : '新建服装' }}</h3>
        <div class="form-grid">
          <label>名称 <input v-model="form.name" placeholder="服装名称" /></label>
          <label>分类 <input v-model="form.category" placeholder="如: 日常、战斗" /></label>
          <label>风格 <select v-model="form.style">
            <option value="">不限</option><option v-for="s in filterOpts.styles" :key="s" :value="s">{{ s }}</option>
          </select></label>
          <label>部位 <select v-model="form.bodyPart">
            <option value="">不限</option><option v-for="b in filterOpts.bodyParts" :key="b" :value="b">{{ b }}</option>
          </select></label>
        </div>
        <label>描述 <textarea v-model="form.description" rows="2" /></label>
        <label>外观描述 <textarea v-model="form.appearance" rows="2" placeholder="服装外观详细描述" /></label>
        <div class="form-grid">
          <label>材质 <input v-model="form.material" placeholder="如: 丝绸" /></label>
          <label>配色 <input v-model="form.colorScheme" placeholder="如: #ff0000,#220000 (渐变色)" /></label>
          <label>季节 <select v-model="form.season">
            <option value="">不限</option><option v-for="s in filterOpts.seasons" :key="s" :value="s">{{ s }}</option>
          </select></label>
        </div>
        <label>图片URL <input v-model="form.imageUrl" placeholder="https://..." /></label>
        <label>标签 <input v-model="tagInput" @keydown.enter.prevent="addTag" placeholder="输入标签后按回车" />
          <div class="chip-row"><span v-for="(t,i) in form.tags" :key="i" class="chip">{{ t }} <button @click="form.tags.splice(i,1)">×</button></span></div>
        </label>
        <div class="modal-actions">
          <button class="btn-secondary" @click="showForm = false">取消</button>
          <button class="btn-primary" @click="save" :disabled="saving">{{ saving ? '保存中...' : '保存' }}</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { costumeLibraryAPI } from '~/composables/useApi'

const search = ref('')
const filterCategory = ref('')
const filterStyle = ref('')
const filterBodyPart = ref('')
const selectedTags = ref([] as string[])
const page = ref(1)
const pageSize = 20

const items = ref([] as any[])
const total = ref(0)
const totalPages = computed(() => Math.ceil(total.value / pageSize))
const loading = ref(false)

const categories = ref([] as string[])
const allTags = ref([] as string[])
const filterOpts = reactive({ styles: [] as string[], bodyParts: [] as string[], seasons: [] as string[] })

const selectedItems = ref([] as number[])
function toggleSelect(id: number) {
  const idx = selectedItems.value.indexOf(id)
  if (idx >= 0) selectedItems.value.splice(idx, 1)
  else selectedItems.value.push(id)
}

const showForm = ref(false)
const editingItem = ref(null as any)
const saving = ref(false)
const form = reactive({
  name: '', category: '', description: '', style: '', bodyPart: '',
  material: '', colorScheme: '', season: '', appearance: '', imageUrl: '',
  tags: [] as string[],
})
const tagInput = ref('')
function addTag() { const t = tagInput.value.trim(); if (t && !form.tags.includes(t)) form.tags.push(t); tagInput.value = '' }

async function load() {
  loading.value = true
  try {
    const res = await costumeLibraryAPI.list({
      page: page.value, pageSize,
      search: search.value || undefined,
      category: filterCategory.value || undefined,
      style: filterStyle.value || undefined,
      bodyPart: filterBodyPart.value || undefined,
      tags: selectedTags.value.join(',') || undefined,
    })
    items.value = res.items || []
    total.value = res.total || 0
  } finally { loading.value = false }
}

async function loadMeta() {
  try {
    const [cats, tags, opts] = await Promise.all([
      costumeLibraryAPI.categories(), costumeLibraryAPI.tags(), costumeLibraryAPI.filterOptions(),
    ])
    categories.value = cats || []
    allTags.value = tags || []
    if (opts) {
      filterOpts.styles = opts.styles || []
      filterOpts.bodyParts = opts.bodyParts || []
      filterOpts.seasons = opts.seasons || []
    }
  } catch { /* silent */ }
}

function onSearch() { page.value = 1; load() }
function onFilter() { page.value = 1; load() }
function toggleTag(t: string) {
  const i = selectedTags.value.indexOf(t)
  if (i >= 0) { selectedTags.value.splice(i, 1); onFilter() }
  else { selectedTags.value.push(t); onFilter() }
}

function openCreate() {
  editingItem.value = null
  Object.assign(form, { name: '', category: '', description: '', style: '', bodyPart: '', material: '', colorScheme: '', season: '', appearance: '', imageUrl: '', tags: [] })
  showForm.value = true
}

function openEdit(item: any) {
  editingItem.value = item
  Object.assign(form, {
    name: item.name, category: item.category, description: item.description,
    style: item.style, bodyPart: item.bodyPart, material: item.material,
    colorScheme: item.colorScheme, season: item.season, appearance: item.appearance,
    imageUrl: item.imageUrl, tags: Array.isArray(item.tags) ? [...item.tags] : [],
  })
  showForm.value = true
}

async function save() {
  if (!form.name) return
  saving.value = true
  try {
    const payload = { ...form, referenceImages: editingItem.value?.referenceImages || null }
    if (editingItem.value) await costumeLibraryAPI.update(editingItem.value.id, payload)
    else await costumeLibraryAPI.create(payload)
    showForm.value = false
    await load(); await loadMeta()
  } finally { saving.value = false }
}

async function deleteItem(item: any) {
  if (!confirm(`确认删除服装 "${item.name}"？`)) return
  await costumeLibraryAPI.del(item.id)
  selectedItems.value = selectedItems.value.filter(id => id !== item.id)
  await load(); await loadMeta()
}

async function batchDelete() {
  if (!confirm(`确认删除 ${selectedItems.value.length} 个服装？`)) return
  await costumeLibraryAPI.batchDelete(selectedItems.value)
  selectedItems.value = []; await load(); await loadMeta()
}

onMounted(() => { load(); loadMeta() })
function truncate(s: string, n: number) { return s && s.length > n ? s.slice(0, n) + '...' : s }
</script>

<style scoped>
.lib-page { height: 100%; overflow-y: auto; padding: 24px 40px 40px; max-width: 1200px; margin: 0 auto; width: 100%; box-sizing: border-box; }
.breadcrumb { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-2); margin-bottom: 16px; }
.bc-link { color: var(--accent-text); text-decoration: none; }
.bc-sep { color: var(--text-3); }
.bc-current { color: var(--text-0); font-weight: 500; }
.page-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 24px; }
.page-header h2 { font-family: var(--font-display); font-size: 22px; font-weight: 700; color: var(--text-0); }
.page-header p { font-size: 13px; color: var(--text-2); margin-top: 4px; }
.btn-primary { display: flex; align-items: center; gap: 6px; padding: 9px 18px; background: var(--accent); color: #fff; border: none; border-radius: var(--radius); font-size: 13px; font-weight: 600; cursor: pointer; }
.btn-primary:hover { opacity: 0.85; }
.btn-primary:disabled { opacity: 0.5; cursor: default; }
.btn-secondary { padding: 9px 18px; background: var(--bg-2); color: var(--text-1); border: 1px solid var(--border); border-radius: var(--radius); font-size: 13px; cursor: pointer; }
.btn-danger-outline { padding: 7px 14px; background: transparent; color: #ef4444; border: 1px solid #ef4444; border-radius: var(--radius); font-size: 12px; cursor: pointer; }
.toolbar { margin-bottom: 20px; }
.search-box { display: flex; align-items: center; gap: 8px; padding: 8px 14px; background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 12px; }
.search-box input { flex: 1; background: none; border: none; outline: none; font-size: 13px; color: var(--text-0); }
.clear-btn { background: none; border: none; color: var(--text-3); cursor: pointer; font-size: 16px; }
.filter-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.filter-select { padding: 7px 12px; background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--radius); font-size: 13px; color: var(--text-0); outline: none; }
.tag-chips { display: flex; gap: 6px; flex-wrap: wrap; flex: 1; }
.tag-chip { padding: 4px 10px; font-size: 12px; border-radius: 12px; background: var(--bg-2); color: var(--text-2); border: 1px solid var(--border); cursor: pointer; }
.tag-chip.active { background: var(--accent-bg); color: var(--accent-text); border-color: rgba(76,125,255,0.3); }
.card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }
.lib-card { display: flex; gap: 14px; padding: 16px; background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--radius); cursor: pointer; transition: all 0.15s; }
.lib-card:hover { border-color: var(--accent); }
.lib-card.selected { border-color: var(--accent); background: var(--accent-bg); }
.card-check { flex-shrink: 0; padding-top: 2px; }
.costume-swatch {
  width: 56px; height: 56px; flex-shrink: 0; border-radius: var(--radius);
  background: var(--bg-2); display: flex; align-items: center; justify-content: center;
  font-size: 24px; border: 1px solid var(--border);
}
.card-body { flex: 1; min-width: 0; }
.card-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
.card-top h4 { font-size: 15px; font-weight: 600; color: var(--text-0); }
.costume-badges { display: flex; gap: 6px; }
.costume-style-badge { font-size: 10px; padding: 1px 6px; background: rgba(236,72,153,0.1); color: #f472b6; border-radius: 6px; font-weight: 500; }
.card-cat { font-size: 11px; padding: 1px 7px; background: var(--bg-2); border-radius: 8px; color: var(--text-2); }
.card-desc { font-size: 12px; color: var(--text-2); line-height: 1.5; margin-bottom: 6px; }
.costume-attrs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; font-size: 11px; color: var(--text-3); }
.card-meta { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.card-tags { display: flex; gap: 4px; flex-wrap: wrap; }
.mini-tag { font-size: 10px; padding: 1px 6px; background: var(--bg-2); border-radius: 6px; color: var(--text-3); }
.card-actions { display: flex; gap: 4px; flex-shrink: 0; }
.action-btn { padding: 3px 8px; font-size: 11px; border-radius: 4px; background: var(--bg-2); color: var(--text-2); border: 1px solid var(--border); cursor: pointer; }
.action-btn:hover { color: var(--accent-text); border-color: var(--accent); }
.action-btn.danger:hover { color: #ef4444; border-color: #ef4444; }
.pager { display: flex; align-items: center; justify-content: center; gap: 16px; margin-top: 24px; font-size: 13px; color: var(--text-2); }
.pager button { padding: 6px 14px; background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--radius); cursor: pointer; color: var(--text-1); font-size: 13px; }
.pager button:disabled { opacity: 0.4; cursor: default; }
.modal-overlay { position: fixed; inset: 0; z-index: 100; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; }
.modal { background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px; width: 640px; max-height: 80vh; overflow-y: auto; }
.modal h3 { font-size: 17px; font-weight: 600; color: var(--text-0); margin-bottom: 20px; }
.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.modal label { display: block; font-size: 12px; color: var(--text-2); margin-bottom: 10px; font-weight: 500; }
.modal input, .modal textarea, .modal select { display: block; width: 100%; margin-top: 4px; padding: 8px 10px; background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--radius); font-size: 13px; color: var(--text-0); outline: none; box-sizing: border-box; font-family: inherit; }
.modal textarea { resize: vertical; }
.modal input:focus, .modal textarea:focus, .modal select:focus { border-color: var(--accent); }
.chip-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
.chip { font-size: 12px; padding: 2px 8px; background: var(--accent-bg); color: var(--accent-text); border-radius: 10px; display: flex; align-items: center; gap: 4px; }
.chip button { background: none; border: none; color: var(--accent-text); cursor: pointer; font-size: 14px; padding: 0; }
.modal-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 20px; }
.empty { text-align: center; padding: 60px 20px; color: var(--text-3); }
.empty p { margin-bottom: 16px; font-size: 14px; }
</style>
