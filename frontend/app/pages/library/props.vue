<template>
  <div class="lib-page">
    <div class="breadcrumb">
      <NuxtLink to="/library" class="bc-link">资源库</NuxtLink>
      <span class="bc-sep">/</span>
      <span class="bc-current">物品库</span>
    </div>

    <header class="page-header">
      <div>
        <h2>物品库</h2>
        <p>道具、信物、线索、法器资产库，设定图自动注入分镜生成参考</p>
      </div>
      <button class="btn-primary" @click="openCreate">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        新建物品
      </button>
    </header>

    <div class="toolbar">
      <div class="search-box">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text-3)" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input v-model="search" placeholder="搜索物品名称、外观、线索..." />
        <button v-if="search" class="clear-btn" @click="search=''">×</button>
      </div>
      <div class="filter-row">
        <select v-model="filterCategory" class="filter-select">
          <option value="">全部类别</option>
          <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
        </select>
        <span class="filter-count" v-if="filtered.length">共 {{ filtered.length }} 件</span>
      </div>
    </div>

    <div class="card-grid" v-if="filtered.length">
      <div v-for="item in filtered" :key="item.id" class="lib-card">
        <div class="prop-thumb" :class="{ empty: !item.imageUrl }" @click="generateImage(item)">
          <img v-if="item.imageUrl" :src="item.imageUrl" :alt="item.name" />
          <span v-else class="thumb-placeholder">{{ propIcons[item.category] || '📦' }}</span>
          <div class="thumb-gen" v-if="!item.imageUrl">生成设定图</div>
        </div>
        <div class="card-body">
          <div class="card-top">
            <h4>{{ item.name }}</h4>
            <div class="prop-badges">
              <span class="card-cat">{{ item.category }}</span>
              <span v-if="item.keyClue" class="clue-badge" :title="`关键线索：${item.keyClue}`">线索</span>
            </div>
          </div>
          <p class="card-desc">{{ truncate(item.description || item.appearance || '暂无描述', 80) }}</p>
          <div class="prop-meta">
            <span v-if="item.holder">持有: {{ item.holder }}</span>
            <span v-if="item.sizeHint">尺寸: {{ item.sizeHint }}</span>
          </div>
          <div class="card-actions" @click.stop>
            <button class="action-btn" @click="generateImage(item)" :disabled="item._generating">设定图</button>
            <button class="action-btn" @click="openEdit(item)">编辑</button>
            <button class="action-btn danger" @click="deleteItem(item)">删除</button>
          </div>
        </div>
      </div>
    </div>
    <div class="empty" v-else-if="!loading"><p>暂无物品模板</p><button class="btn-secondary" @click="openCreate">创建第一个物品</button></div>

    <!-- 表单模态框 -->
    <div class="modal-overlay" v-if="showForm" @click.self="showForm = false">
      <div class="modal">
        <h3>{{ editingItem ? '编辑物品' : '新建物品' }}</h3>
        <div class="form-grid">
          <label>名称 <input v-model="form.name" placeholder="如: 青铜玉坠" /></label>
          <label>类别 <select v-model="form.category">
            <option value="道具">道具</option>
            <option value="信物">信物</option>
            <option value="线索">线索</option>
            <option value="法器">法器</option>
            <option value="其他">其他</option>
          </select></label>
        </div>
        <label>描述 <textarea v-model="form.description" rows="2" placeholder="物品作用与剧情意义" /></label>
        <label>外观描述 <textarea v-model="form.appearance" rows="2" placeholder="物品外观详细描述（用于设定图生成）" /></label>
        <div class="form-grid">
          <label>尺寸提示 <input v-model="form.size_hint" placeholder="如: 掌心大小" /></label>
          <label>持有者 <input v-model="form.holder" placeholder="如: 林晚" /></label>
        </div>
        <label>关键线索 <input v-model="form.key_clue" placeholder="如: 玉坠内侧刻有家族徽记" /></label>
        <label>自定义正向提示词 <textarea v-model="form.custom_prompt" rows="2" placeholder="可选，覆盖默认设定图提示词" /></label>
        <label>反向提示词 <textarea v-model="form.negative_prompt" rows="2" placeholder="如: text, watermark, multiple objects" /></label>
        <div class="modal-actions">
          <button class="btn-secondary" @click="showForm = false">取消</button>
          <button class="btn-primary" @click="save" :disabled="saving">{{ saving ? '保存中...' : '保存' }}</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { toast } from 'vue-sonner'
import { propsLibraryAPI } from '~/composables/useApi'
import { useConfirm } from '~/composables/useConfirm'

const { confirm } = useConfirm()
const route = useRoute()

const propIcons: Record<string, string> = { '道具': '📦', '信物': '🔖', '线索': '🔍', '法器': '🪔', '其他': '📌' }

const search = ref('')
const filterCategory = ref('')
const items = ref([] as any[])
const loading = ref(false)
const dramaId = computed(() => Number(route.query.drama_id) || 0)

const filtered = computed(() => {
  let list = items.value
  if (filterCategory.value) list = list.filter((i) => i.category === filterCategory.value)
  if (search.value.trim()) {
    const q = search.value.trim().toLowerCase()
    list = list.filter((i) =>
      [i.name, i.description, i.appearance, i.holder, i.keyClue].some((v) => v && String(v).toLowerCase().includes(q))
    )
  }
  return list
})

const categories = computed(() => {
  const set = new Set<string>(['道具', '信物', '线索', '法器', '其他'])
  items.value.forEach((i) => i.category && set.add(i.category))
  return [...set]
})

async function load() {
  loading.value = true
  try {
    const res = await propsLibraryAPI.list(dramaId.value ? { drama_id: dramaId.value } : undefined)
    items.value = Array.isArray(res) ? res : res?.items || []
  } catch (e: any) {
    toast.error(e?.message || '加载物品列表失败')
  } finally { loading.value = false }
}

const showForm = ref(false)
const editingItem = ref(null as any)
const saving = ref(false)
const form = reactive({
  name: '', category: '道具', description: '', appearance: '',
  size_hint: '', holder: '', key_clue: '', custom_prompt: '', negative_prompt: '',
})

function openCreate() {
  editingItem.value = null
  Object.assign(form, { name: '', category: '道具', description: '', appearance: '', size_hint: '', holder: '', key_clue: '', custom_prompt: '', negative_prompt: '' })
  showForm.value = true
}

function openEdit(item: any) {
  editingItem.value = item
  Object.assign(form, {
    name: item.name, category: item.category, description: item.description, appearance: item.appearance,
    size_hint: item.sizeHint || '', holder: item.holder || '', key_clue: item.keyClue || '',
    custom_prompt: item.customPrompt || '', negative_prompt: item.negativePrompt || '',
  })
  showForm.value = true
}

async function save() {
  if (!form.name) return
  saving.value = true
  try {
    const payload = {
      ...form,
      drama_id: editingItem.value?.dramaId || dramaId.value || 1,
    }
    if (editingItem.value) await propsLibraryAPI.update(editingItem.value.id, payload)
    else await propsLibraryAPI.create(payload)
    showForm.value = false
    await load()
  } catch (e: any) {
    toast.error(e?.message || '保存失败')
  } finally { saving.value = false }
}

async function deleteItem(item: any) {
  if (!(await confirm({ message: `确认删除物品 "${item.name}"？`, danger: true }))) return
  try {
    await propsLibraryAPI.del(item.id)
    await load()
  } catch (e: any) {
    toast.error(e?.message || '删除失败')
  }
}

async function generateImage(item: any) {
  if (item._generating) return
  item._generating = true
  try {
    toast.loading(`正在生成「${item.name}」设定图...`, { id: `prop-img-${item.id}` })
    await propsLibraryAPI.generateImage(item.id)
    toast.success('设定图生成任务已提交，生成完成后自动刷新', { id: `prop-img-${item.id}` })
    // 轮询刷新直到拿到图
    for (let i = 0; i < 12; i++) {
      await new Promise((r) => setTimeout(r, 5000))
      const res = await propsLibraryAPI.list(dramaId.value ? { drama_id: dramaId.value } : undefined)
      const arr = Array.isArray(res) ? res : res?.items || []
      const updated = arr.find((p: any) => p.id === item.id)
      if (updated?.imageUrl) { items.value = arr; break }
    }
  } catch (e: any) {
    toast.error(e?.message || '设定图生成失败', { id: `prop-img-${item.id}` })
  } finally { item._generating = false }
}

onMounted(() => load())
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
.toolbar { margin-bottom: 20px; }
.search-box { display: flex; align-items: center; gap: 8px; padding: 8px 14px; background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 12px; }
.search-box input { flex: 1; background: none; border: none; outline: none; font-size: 13px; color: var(--text-0); }
.clear-btn { background: none; border: none; color: var(--text-3); cursor: pointer; font-size: 16px; }
.filter-row { display: flex; align-items: center; gap: 10px; }
.filter-select { padding: 7px 12px; background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--radius); font-size: 13px; color: var(--text-0); outline: none; }
.filter-count { font-size: 12px; color: var(--text-3); }
.card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(320px, 100%), 1fr)); gap: 12px; }
.lib-card { display: flex; gap: 14px; padding: 16px; background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--radius); transition: all 0.15s; }
.lib-card:hover { border-color: var(--accent); }
.prop-thumb {
  position: relative; width: 92px; height: 92px; flex-shrink: 0; border-radius: var(--radius);
  background: var(--bg-2); border: 1px solid var(--border); overflow: hidden;
  display: flex; align-items: center; justify-content: center; cursor: pointer;
}
.prop-thumb img { width: 100%; height: 100%; object-fit: cover; }
.prop-thumb.empty:hover .thumb-gen { opacity: 1; }
.thumb-placeholder { font-size: 34px; opacity: 0.7; }
.thumb-gen {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  background: rgba(0,0,0,0.6); color: #fff; font-size: 12px; font-weight: 600;
  opacity: 0; transition: opacity 0.15s; pointer-events: none;
}
.card-body { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.card-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
.card-top h4 { font-size: 15px; font-weight: 600; color: var(--text-0); }
.prop-badges { display: flex; gap: 6px; }
.card-cat { font-size: 11px; padding: 1px 7px; background: var(--bg-2); border-radius: 8px; color: var(--text-2); }
.clue-badge { font-size: 10px; padding: 1px 6px; background: rgba(245,158,11,0.12); color: #fbbf24; border-radius: 6px; font-weight: 600; }
.card-desc { font-size: 12px; color: var(--text-2); line-height: 1.5; margin-bottom: 6px; flex: 1; }
.prop-meta { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; font-size: 11px; color: var(--text-3); }
.card-actions { display: flex; gap: 4px; flex-shrink: 0; }
.action-btn { padding: 5px 10px; font-size: 12px; border-radius: 6px; background: var(--bg-2); color: var(--text-2); border: 1px solid var(--border); cursor: pointer; }
.action-btn:hover { color: var(--accent-text); border-color: var(--accent); }
.action-btn.danger:hover { color: #ef4444; border-color: #ef4444; }
.action-btn:disabled { opacity: 0.5; cursor: default; }
.modal-overlay { position: fixed; inset: 0; z-index: 100; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; }
.modal { background: var(--bg-1); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px; width: 640px; max-height: 80vh; overflow-y: auto; }
.modal h3 { font-size: 17px; font-weight: 600; color: var(--text-0); margin-bottom: 20px; }
.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.modal label { display: block; font-size: 12px; color: var(--text-2); margin-bottom: 10px; font-weight: 500; }
.modal input, .modal textarea, .modal select { display: block; width: 100%; margin-top: 4px; padding: 8px 10px; background: var(--bg-2); border: 1px solid var(--border); border-radius: var(--radius); font-size: 13px; color: var(--text-0); outline: none; box-sizing: border-box; font-family: inherit; }
.modal textarea { resize: vertical; }
.modal input:focus, .modal textarea:focus, .modal select:focus { border-color: var(--accent); }
.modal-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 20px; }
.empty { text-align: center; padding: 60px 20px; color: var(--text-3); }
.empty p { margin-bottom: 16px; font-size: 14px; }
</style>
