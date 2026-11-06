/**
 * 画布节点的「上次用过什么」记忆（上游 `node-settings-memory.ts` 的**适配改造** ✓，非照搬 ✗）。
 *
 * ## 与源的区别（为什么不能直接搬）
 *
 * 源项目的节点带一整套生成参数 metadata（`model` / `size` / `quality` / `seed` / `fps` …），
 * 所以它记的是「参数快照」。本仓画布节点只有 `text` / `url` / `w` / `h`，
 * **没有生成参数** ⇒ 照搬那 24 个字段的记忆表会是一张**永远空的表**（死代码 ✗）。
 *
 * 适配后记两样**真的会被复用**的东西：
 *
 * 1. ⭐ **尺寸**（所有类型）：把一张图片卡拉大后，下一张图片卡就该是这个尺寸 ✓ ——
 *    否则每加一张都要重新拉一次，批量摆素材时极其难受 ✗。
 * 2. ⭐ **配置卡的正文**（只有 `config`）：配置卡装的是 `key=value` 参数，
 *    新建时就该带上一次的参数作起手 ✓（这就是源项目「rememberNodeSettings」的真实用途 ✓）。
 *    文本卡**不记** ✓ —— 那是用户写的剧本/台词，继承上一次会把新内容**直接覆盖**掉 ✗。
 *
 * ⚠️ 记忆是**会话级内存**（源同口径 ✓）：刷新页面就重置 ✓。
 * 不写 localStorage 的理由：尺寸/参数属于「这一轮怎么摆」的临时偏好，
 * 落盘会让用户**下次打开还在用上周的尺寸**且找不到哪里能改 ✗。
 */
import type { CanvasNodeType } from '../composables/useCanvas'

export interface RememberedSettings {
  w?: number
  h?: number
  /** 仅 `config` 会记 ✓ */
  text?: string
}

const memory = new Map<CanvasNodeType, RememberedSettings>()

/** 记一次「这类节点上次长什么样」✓（尺寸缺失/非法就不记那一项 ✓） */
export function rememberNodeSettings(type: CanvasNodeType, node: { w?: number; h?: number; text?: string }) {
    const next: RememberedSettings = { ...(memory.get(type) || {}) }
    if (Number.isFinite(node.w) && (node.w as number) > 0) next.w = Math.round(node.w as number)
    if (Number.isFinite(node.h) && (node.h as number) > 0) next.h = Math.round(node.h as number)
    // 只有配置卡继承正文 ✓（文本卡是创作内容，继承＝覆盖用户的稿子 ✗）
    if (type === 'config' && typeof node.text === 'string') next.text = node.text
    if (Object.keys(next).length) memory.set(type, next)
}

/** 取这类节点上次的设置 ✓（拿不到 ⇒ `undefined`，由调用方回默认 ✓） */
export function recallNodeSettings(type: CanvasNodeType): RememberedSettings | undefined {
    const remembered = memory.get(type)
    return remembered ? { ...remembered } : undefined
}

export function clearNodeSettingsMemory() {
    memory.clear()
}
