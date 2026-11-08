/**
 * 画布节点的「谁引用了谁」（移植自上游 `lib/canvas/canvas-resource-references.ts`，MIT；自研重写 ✓）。
 *
 * ## 为什么这块值得搬
 *
 * 画布上「连线 + 分组」本身就是一份**素材装配说明**：图片卡连到配置卡，
 * 就是把这张图当参考图喂给那次生成 ✓。源项目把这件事收成了纯数据层（不依赖 React、不依赖生成链路），
 * 所以它能先落地、等生成链路接上直接复用 ✓ —— 这就是本次搬它的理由。
 *
 * ## 三条口径（与源一致）
 *
 * 1. **引用来源两条**：显式连线（`edges`）+ 隐式同组（`groupId`）✓。
 *    只认连线会漏掉"我把三张图和一个配置卡放进一个分组"这种最常见的摆法 ✗。
 * 2. **分组要展开成成员** ✓：连线指向分组框时，实际引用的是框里的成员 ✗（框本身没有素材）。
 * 3. **编号按类别独立计数** ✓：`参考图 1 / 参考图 2 / 参考视频 1` ——
 *    统一编号会让「参考图 3」其实是第 1 张图，提示词里对不上 ✗。
 */

import type { CanvasEdge, CanvasNode } from '../composables/useCanvas'

export type ResourceKind = 'image' | 'video' | 'text' | 'config'

export interface ResourceReference {
  nodeId: string
  kind: ResourceKind
  title: string
  /** 可直接写进提示词的称呼 ✓，如 `参考图 1` */
  label: string
}

const KIND_LABEL: Record<ResourceKind, string> = {
  image: '参考图',
  video: '参考视频',
  text: '参考文本',
  config: '参考配置',
}

/** 分组框不是素材 ✓（它只是容器） */
export function resourceKindOf(node: CanvasNode | undefined | null): ResourceKind | null {
  if (!node) return null
  if (node.type === 'group') return null
  if (node.type === 'image') return node.url ? 'image' : null
  if (node.type === 'video') return node.url ? 'video' : null
  if (node.type === 'text') return node.text?.trim() ? 'text' : null
  return 'config'
}

/** 分组框 → 成员（按画布顺序，稳定 ✓） */
export function groupMembers(group: CanvasNode, nodes: CanvasNode[]): CanvasNode[] {
  if (group.type !== 'group') return []
  return nodes.filter((node) => node.groupId === group.id)
}

/**
 * 某个节点的**上游输入**：直接连它的 + 与它同组的（都做分组展开、去重、保序 ✓）。
 * 自己不会出现在结果里 ✓。
 */
export function upstreamNodes(nodeId: string, nodes: CanvasNode[], edges: CanvasEdge[]): CanvasNode[] {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const self = byId.get(nodeId)
  const ordered: CanvasNode[] = []
  const seen = new Set<string>([nodeId])

  const push = (node: CanvasNode | undefined) => {
    if (!node || seen.has(node.id)) return
    seen.add(node.id)
    if (node.type === 'group') {
      for (const member of groupMembers(node, nodes)) push(member)
      return
    }
    ordered.push(node)
  }

  // ① 显式连线（按连线建立顺序 ✓）
  for (const edge of edges) {
    if (edge.to !== nodeId) continue
    push(byId.get(edge.from))
  }
  // ② 隐式同组 ✓（同一分组里的其它卡就是这批素材的一部分）
  if (self?.groupId) {
    for (const member of nodes) {
      if (member.groupId === self.groupId) push(member)
    }
  }
  return ordered
}

/**
 * 上游素材 → 带编号的引用清单 ✓。
 *
 * 编号**按类别各自从 1 开始** ✓；没有内容的节点（空文本 / 没 url 的卡片）跳过 ✓ ——
 * 否则会出现「参考图 1」指向一张还没放图的空卡 ✗。
 */
export function buildResourceReferences(nodeId: string, nodes: CanvasNode[], edges: CanvasEdge[]): ResourceReference[] {
  const counts: Record<ResourceKind, number> = { image: 0, video: 0, text: 0, config: 0 }
  const result: ResourceReference[] = []
  for (const node of upstreamNodes(nodeId, nodes, edges)) {
    const kind = resourceKindOf(node)
    if (!kind) continue
    counts[kind] += 1
    result.push({
      nodeId: node.id,
      kind,
      title: node.title,
      label: `${KIND_LABEL[kind]} ${counts[kind]}`,
    })
  }
  return result
}

export interface ReferenceInputs {
  /** 生成接口要的参考图 URL ✓（连线优先、去重、保序 ✓） */
  images: string[]
  /**
   * 上游**文本** ✓（`text` 卡正文，按连线顺序用空行连接）。
   *
   * 只给插件节点用（HTML 的 `{{input}}`、便利贴的「来自上游」）—— 生成接口不吃它，
   * 所以它和 `images` 分开放，别混进请求里 ✗。
   */
  text?: string
  /**
   * 这张卡的**真正提示词** ✓ —— 生成请求该送的那句话。
   *
   * 配置卡正文不是人话而是 `@[node:text-xxxx]` 引用串（`canvas_ops._generation_flow` 就是这么
   * 回给前端的：提示词只存一份，改文本卡配置卡跟着变 ✓）。**引用不解析就直接提交**，
   * 引擎拿到的会是一串 `@[node:text-9f8e7d6c5a]` ✗ —— 出的图与用户看到的字毫无关系。
   * 所以这里统一解析成正文（见 :func:`resolvePromptText`），卡片只管取用 ✓。
   */
  prompt?: string
}

/** 配置卡正文里的节点引用标记 ✓（与后端 `canvas_ops.REFERENCE_PATTERN` 同形） */
export const REFERENCE_TOKEN = /@\[node:([^\]]+)\]/g

/**
 * 引用串 → 人话 ✓（`@[node:text-x]` 换成那张文本卡的正文）。
 *
 * `missing` 如实返回找不到的 id —— 静默删掉会让用户以为提示词就这几句 ✗。
 */
export function resolvePromptText(text: string, nodes: CanvasNode[]): { text: string; missing: string[] } {
  if (!text.includes('@[node:')) return { text: text.trim(), missing: [] }
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const missing: string[] = []
  const resolved = text.replace(REFERENCE_TOKEN, (_all, id: string) => {
    const target = byId.get(id)
    const body = (target?.text || '').trim()
    if (!target) missing.push(id)
    return body
  })
  // 引用都是空卡时会剩下成片空行 ⇒ 归一，别把一串 `\n\n\n` 当提示词送出去 ✗
  const cleaned = resolved
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .join('\n')
    .trim()
  return { text: cleaned, missing }
}

/**
 * 上游素材 → **生成接口的输入清单** ✓（与 :func:`buildResourceReferences` 同一套遍历 ✓）。
 *
 * 图片进 `images`（只有图片有落点：注册表里 ``video-h3-ref2va`` 虽然声明了
 * ``videos`` / ``audios``，但 ``apply_bindings`` 只为 ``from: "image"`` 建 LoadImage ✗）。
 * 文本进 `text`（**不**进生成请求）—— 插件节点的 `{{input}}` 吃它 ✓。
 */
export function collectReferenceInputs(
  nodeId: string,
  nodes: CanvasNode[],
  edges: CanvasEdge[],
): ReferenceInputs {
  const images: string[] = []
  const texts: string[] = []
  const upstream = upstreamNodes(nodeId, nodes, edges)
  for (const node of upstream) {
    if (node.type === 'image' && node.url && !images.includes(node.url)) images.push(node.url)
    if (node.type === 'text' && node.text?.trim()) texts.push(node.text.trim())
  }
  const inputs: ReferenceInputs = texts.length ? { images, text: texts.join('\n\n') } : { images }

  /*
   * 提示词口径（与源项目的一处语义修正）：
   * 源把 prompt 存进 config 节点的 `metadata.prompt`，正文另存一份 ⇒ 两处会漂 ✗。
   * 本仓正文就是**唯一真源**：是人话直接送；是 `@[node:…]` 引用就先解析 ✓。
   * 都不成立（配置卡还没写）时才回落到「上游文本拼接」——
   * 那是用户「连了一张文本卡过来」的意图 ✓，但优先级必须低于正文 ✗。
   */
  const self = nodes.find((node) => node.id === nodeId)
  const own = (self?.text || '').trim()
  if (own) {
    const resolved = resolvePromptText(own, nodes)
    if (resolved.text) inputs.prompt = resolved.text
  }
  if (!inputs.prompt && inputs.text) inputs.prompt = inputs.text
  return inputs
}

/** 给人看的一行摘要 ✓（画布右键「查看引用」直接显示） */
export function describeReferences(nodeId: string, nodes: CanvasNode[], edges: CanvasEdge[]): string[] {
  const references = buildResourceReferences(nodeId, nodes, edges)
  if (!references.length) return ['没有引用任何素材（连线或放进同一个分组）']
  return references.map((item) => `${item.label} · ${item.title}`)
}
