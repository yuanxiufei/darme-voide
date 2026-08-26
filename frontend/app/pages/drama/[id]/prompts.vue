<script setup lang="ts">
import { ref, computed } from 'vue'
import { toast } from 'vue-sonner'
import { Search } from 'lucide-vue-next'
import { dramaAPI, characterAPI, sceneAPI, storyboardAPI } from '~/composables/useApi'
import PromptCard from '~/components/PromptCard.vue'

const route = useRoute()
const router = useRouter()
const dramaId = computed(() => Number(route.params.id))
const drama = ref<any>(null)

const characters = ref<any[]>([])
const scenes = ref<any[]>([])
const episodes = ref<any[]>([])
const storyboards = ref<any[]>([])

const query = ref('')
const filter = ref<'all' | 'character' | 'scene' | 'image' | 'video'>('all')
const loading = ref(false)

const filters = [
  { key: 'all', label: '全部' },
  { key: 'character', label: '角色' },
  { key: 'scene', label: '场景' },
  { key: 'image', label: '分镜图' },
  { key: 'video', label: '分镜视频' },
] as const

const q = computed(() => query.value.trim().toLowerCase())
function matchName(...parts: (string | null | undefined)[]) {
  if (!q.value) return true
  return parts.some(p => (p || '').toLowerCase().includes(q.value))
}

function countFor(key: string) {
  if (key === 'character') return characters.value.length
  if (key === 'scene') return scenes.value.length
  if (key === 'image') return storyboards.value.length
  if (key === 'video') return storyboards.value.length
  return characters.value.length + scenes.value.length + storyboards.value.length * 2
}

const visibleCharacters = computed(() =>
  characters.value.filter(c =>
    (filter.value === 'all' || filter.value === 'character') &&
    matchName(c.name, c.customPrompt),
  ),
)

const visibleScenes = computed(() =>
  scenes.value.filter(s =>
    (filter.value === 'all' || filter.value === 'scene') &&
    matchName(s.location, s.customPrompt, s.prompt),
  ),
)

const visibleStoryboards = computed(() =>
  storyboards.value.filter(sb => {
    const showImage = filter.value === 'all' || filter.value === 'image'
    const showVideo = filter.value === 'all' || filter.value === 'video'
    const imgHit = showImage && matchName(sb.title, sb.customImagePrompt, sb.imagePrompt)
    const vidHit = showVideo && matchName(sb.title, sb.customVideoPrompt, sb.videoPrompt)
    return imgHit || vidHit
  }),
)

const visibleEpisodes = computed(() =>
  episodes.value.filter(ep => visibleStoryboards.value.some(sb => sb.episodeId === ep.id)),
)

function storyboardsOf(epId: number) {
  return visibleStoryboards.value.filter(sb => sb.episodeId === epId)
}

function showImageCard() { return filter.value === 'all' || filter.value === 'image' }
function showVideoCard() { return filter.value === 'all' || filter.value === 'video' }

const hasAny = computed(() =>
  visibleCharacters.value.length > 0 ||
  visibleScenes.value.length > 0 ||
  visibleEpisodes.value.length > 0,
)

async function load() {
  loading.value = true
  try {
    const [d, p] = await Promise.all([
      dramaAPI.get(dramaId.value),
      dramaAPI.prompts(dramaId.value),
    ])
    drama.value = d
    characters.value = p.characters || []
    scenes.value = p.scenes || []
    episodes.value = p.episodes || []
    storyboards.value = p.storyboards || []
  } catch (e: any) {
    toast.error(e?.message || '加载提示词失败')
  } finally {
    loading.value = false
  }
}

function back() {
  const state = window.history.state
  if (state && state.back) {
    router.back()
  } else {
    navigateTo(`/drama/${dramaId.value}`)
  }
}

// ===== 保存（写回 custom 字段；custom 优先于自动生成 prompt）=====
async function saveCharacterPrompt(ch: any, v: string) {
  try {
    await characterAPI.update(ch.id, { customPrompt: v })
    ch.customPrompt = v
    toast.success(`已保存「${ch.name}」提示词`)
  } catch (e: any) {
    toast.error(e?.message || '保存失败')
    throw e
  }
}

async function saveScenePrompt(sc: any, v: string) {
  try {
    await sceneAPI.update(sc.id, { customPrompt: v })
    sc.customPrompt = v
    toast.success(`已保存「${sc.location}」提示词`)
  } catch (e: any) {
    toast.error(e?.message || '保存失败')
    throw e
  }
}

async function saveStoryboardImagePrompt(sb: any, v: string) {
  try {
    await storyboardAPI.update(sb.id, { custom_image_prompt: v })
    sb.customImagePrompt = v
    toast.success('已保存分镜图片提示词')
  } catch (e: any) {
    toast.error(e?.message || '保存失败')
    throw e
  }
}

async function saveStoryboardVideoPrompt(sb: any, v: string) {
  try {
    await storyboardAPI.update(sb.id, { custom_video_prompt: v })
    sb.customVideoPrompt = v
    toast.success('已保存分镜视频提示词')
  } catch (e: any) {
    toast.error(e?.message || '保存失败')
    throw e
  }
}

onMounted(load)
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div class="head-left">
        <button class="back-btn" @click="back">‹</button>
        <div class="head-info">
          <h1 class="page-title">提示词管理</h1>
          <div class="page-meta">{{ drama?.title || '统一查看与编辑' }} · 编辑将写入「自定义提示词」并覆盖自动生成</div>
        </div>
      </div>
    </div>

    <!-- 搜索 + 筛选 -->
    <div class="toolbar">
      <div class="search-box">
        <Search :size="14" />
        <input v-model="query" class="input" placeholder="搜索名称或提示词内容..." />
      </div>
      <div class="filters">
        <button
          v-for="f in filters"
          :key="f.key"
          class="filter-pill"
          :class="{ active: filter === f.key }"
          @click="filter = f.key"
        >
          {{ f.label }}
          <span class="pill-count">{{ countFor(f.key) }}</span>
        </button>
      </div>
    </div>

    <div v-if="loading" class="empty">加载中…</div>

    <div v-else-if="!hasAny" class="empty">没有匹配的提示词</div>

    <template v-else>
      <!-- 角色 -->
      <section v-if="visibleCharacters.length" class="block">
        <h2 class="block-title">角色立绘 <span class="block-count">{{ visibleCharacters.length }}</span></h2>
        <div class="cards">
          <PromptCard
            v-for="c in visibleCharacters"
            :key="'c' + c.id"
            :name="c.name"
            :tag="c.role || '角色'"
            :value="c.customPrompt"
            placeholder="未设置自定义提示词，生成时按角色外貌自动拼装"
            :on-save="(v: string) => saveCharacterPrompt(c, v)"
          />
        </div>
      </section>

      <!-- 场景 -->
      <section v-if="visibleScenes.length" class="block">
        <h2 class="block-title">场景 <span class="block-count">{{ visibleScenes.length }}</span></h2>
        <div class="cards">
          <PromptCard
            v-for="s in visibleScenes"
            :key="'s' + s.id"
            :name="s.location"
            :tag="'场景'"
            :value="s.customPrompt || s.prompt"
            placeholder="未设置自定义提示词，生成时按场景描述自动拼装"
            :on-save="(v: string) => saveScenePrompt(s, v)"
          />
        </div>
      </section>

      <!-- 分镜（按集分组） -->
      <section v-if="visibleEpisodes.length" class="block">
        <h2 class="block-title">分镜 <span class="block-count">{{ visibleStoryboards.length }}</span></h2>
        <div v-for="ep in visibleEpisodes" :key="'ep' + ep.id" class="ep-group">
          <div class="ep-title">第 {{ ep.episodeNumber }} 集 · {{ ep.title }}</div>
          <div class="cards">
            <template v-for="sb in storyboardsOf(ep.id)" :key="'sb' + sb.id">
              <PromptCard
                v-if="showImageCard()"
                :name="`#${sb.storyboardNumber} ${sb.title || '未命名镜头'}`"
                :tag="'分镜图'"
                :value="sb.customImagePrompt || sb.imagePrompt"
                placeholder="未设置自定义提示词，生成时按分镜描述自动拼装"
                :on-save="(v: string) => saveStoryboardImagePrompt(sb, v)"
              />
              <PromptCard
                v-if="showVideoCard()"
                :name="`#${sb.storyboardNumber} ${sb.title || '未命名镜头'}`"
                :tag="'分镜视频'"
                :value="sb.customVideoPrompt || sb.videoPrompt"
                placeholder="未设置自定义提示词，生成时按分镜动作自动拼装"
                :on-save="(v: string) => saveStoryboardVideoPrompt(sb, v)"
              />
            </template>
          </div>
        </div>
      </section>
    </template>
  </div>
</template>

<style scoped>
.page {
  padding: 28px 48px 40px;
  overflow-y: auto;
  height: 100%;
  animation: fadeUp 0.35s var(--ease-out) both;
}
.page-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 24px;
  gap: 20px;
}
.head-left { display: flex; align-items: flex-start; gap: 12px; }
.head-info { display: flex; flex-direction: column; gap: 8px; }
.back-btn {
  display: flex; align-items: center; gap: 6px;
  padding: 7px 12px; font-size: 13px; font-weight:500;
  border: 1px solid var(--border); border-radius: var(--radius);
  background: var(--bg-0); color: var(--text-2);
  cursor: pointer; transition: all 0.18s var(--ease-out);
  box-shadow: var(--shadow-xs);
}
.back-btn:hover { background: var(--bg-hover); border-color: var(--border-strong); color: var(--text-0); }
.page-title {
  font-family: var(--font-display);
  font-size: 26px; font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.2;
}
.page-meta { font-size: 12px; color: var(--text-2); }
.toolbar {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.search-box {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0 12px;
  color: var(--text-3);
  flex: 1;
  min-width: 220px;
  max-width: 420px;
}
.search-box .input {
  border: none;
  background: transparent;
  padding: 9px 0;
  flex: 1;
}
.filters {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.filter-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 12px;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-2);
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 999px;
  cursor: pointer;
  transition: all 0.15s;
}
.filter-pill:hover { border-color: var(--accent); color: var(--text-1); }
.filter-pill.active {
  background: var(--accent-bg);
  color: var(--accent-text);
  border-color: var(--accent);
}
.pill-count {
  font-size: 11px;
  opacity: 0.7;
}
.block { margin-bottom: 28px; }
.block-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  font-weight: 600;
  color: var(--text-1);
  margin: 0 0 14px;
}
.block-count {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-3);
  background: var(--bg-2);
  border-radius: 999px;
  padding: 2px 10px;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 12px;
}
.ep-group { margin-bottom: 20px; }
.ep-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-2);
  margin-bottom: 10px;
  padding-left: 10px;
  border-left: 3px solid var(--accent);
}
.empty {
  padding: 60px 0;
  text-align: center;
  color: var(--text-3);
  font-size: 14px;
}
</style>
