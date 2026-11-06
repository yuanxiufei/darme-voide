/**
 * 画布几何：分组收纳 + 连线归一 + 聚焦视口。
 *
 * 移植自上游 `ComfyUI-Local-Infinite-Canvas`（MIT）的
 * `infinite-canvas/web/src/lib/canvas/canvas-node-geometry.ts` ⇒ 只搬**几何口径**，
 * 代码按本仓的 `CanvasNode`（`x/y/w/h`，不是 `position/width`）重写 ✓。
 *
 * ## 五处口径（都是「能做出来但手感不对」的地方）
 *
 * 1. ⭐ **成组要 ≥2 个成员**：一个人不成组；且**已在同一个组**的不能再成组
 *    （否则会出现「组套组」✗）。
 * 2. ⭐ **分组框上下留白不一样** ✓：上 52、下/左右 24 —— 上面要露出分组的**标题栏**，
 *    按 24 会把标题压掉 ✗。
 * 3. ⭐ **判定「拖进了哪个组」看**中心点** ✓：不是看有没有重叠 —— 重叠判定会让
 *    擦边而过的节点被误收进组 ✗。而且**倒着找**（后加的组在上面，先命中）✓。
 * 4. ⭐ **收纳时整体平移，不是逐个摆位** ✓：算一次 `dx/dy` 施加给所有成员，
 *    成员之间的相对位置**保持不变** ✓（逐个摆会散架 ✗）；装不下时才贴边。
 * 5. ⭐ **连线三不许** ✓：不许指向**分组**（分组不是执行节点）、
 *    两个**配置卡**之间不连（配置是参数源，不是数据流）、不许自连。
 * 6. ⭐ **聚焦时节点最多占视口 60%** ✓，缩放钳在 `0.05–1`（不放大 ✗ ——
 *    聚焦是为了「看见全貌」）；下方有浮层时按 **3:1** 让位，否则居中 ✓。
 */
import type { CanvasNode } from '../composables/useCanvas'

/** 分组框内边距：左右/下 24 ✓ */
export const GROUP_WRAP_PADDING = 24
/** 分组框**上**留白 —— 要露出标题栏，比左右大 ✓ */
export const GROUP_WRAP_TOP_PADDING = 52

export interface Bounds {
  left: number
  top: number
  right: number
  bottom: number
}

export function nodeBounds(nodes: CanvasNode[]): Bounds {
  return nodes.reduce<Bounds>(
    (acc, node) => ({
      left: Math.min(acc.left, node.x),
      top: Math.min(acc.top, node.y),
      right: Math.max(acc.right, node.x + node.w),
      bottom: Math.max(acc.bottom, node.y + node.h),
    }),
    { left: Infinity, top: Infinity, right: -Infinity, bottom: -Infinity },
  )
}

/** 用**中心点**判定包含 ✓（擦边不算 ✗） */
export function containsCenter(group: CanvasNode, node: CanvasNode): boolean {
  const x = node.x + node.w / 2
  const y = node.y + node.h / 2
  return x >= group.x && x <= group.x + group.w && y >= group.y && y <= group.y + group.h
}

/** 拖动的节点落进了哪个分组（**倒序**找，后加的组在上 ✓） */
export function findGroupDropTarget(movedIds: string[], nodes: CanvasNode[]): CanvasNode | null {
  const moved = new Set(movedIds)
  if (nodes.some((node) => moved.has(node.id) && node.type === 'group')) return null
  const moving = nodes.filter((node) => moved.has(node.id) && node.type !== 'group')
  if (!moving.length) return null
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    const group = nodes[index]
    if (group.type !== 'group' || moved.has(group.id)) continue
    if (moving.some((node) => containsCenter(group, node))) return group
  }
  return null
}

/** 把成员**整体平移**进分组 ✓（相对位置不变 ✓），并打上 `groupId` */
export function snapNodesIntoGroup(movedIds: string[], nodes: CanvasNode[], group: CanvasNode): CanvasNode[] {
  const moved = new Set(movedIds)
  const moving = nodes.filter((node) => moved.has(node.id) && node.type !== 'group')
  if (!moving.length) return nodes
  const pad = GROUP_WRAP_PADDING
  const bounds = nodeBounds(moving)
  const left = group.x + pad
  const top = group.y + pad
  const right = group.x + group.w - pad
  const bottom = group.y + group.h - pad
  const dx =
    bounds.right - bounds.left > right - left
      ? left - bounds.left
      : bounds.left < left
        ? left - bounds.left
        : bounds.right > right
          ? right - bounds.right
          : 0
  const dy =
    bounds.bottom - bounds.top > bottom - top
      ? top - bounds.top
      : bounds.top < top
        ? top - bounds.top
        : bounds.bottom > bottom
          ? bottom - bounds.bottom
          : 0
  return nodes.map((node) =>
    moved.has(node.id) && node.type !== 'group'
      ? { ...node, x: node.x + dx, y: node.y + dy, groupId: group.id }
      : node,
  )
}

/** 按成员算分组框（**上留白更大** ✓） */
export function getGroupWrapRect(members: CanvasNode[]) {
  const bounds = nodeBounds(members)
  return {
    x: bounds.left - GROUP_WRAP_PADDING,
    y: bounds.top - GROUP_WRAP_TOP_PADDING,
    w: bounds.right - bounds.left + GROUP_WRAP_PADDING * 2,
    h: bounds.bottom - bounds.top + GROUP_WRAP_TOP_PADDING + GROUP_WRAP_PADDING,
  }
}

export function canGroupSelectedNodes(selectedIds: string[], nodes: CanvasNode[]): boolean {
  const selected = new Set(selectedIds)
  const members = nodes.filter((node) => node.type !== 'group' && selected.has(node.id))
  if (members.length < 2) return false
  const groupId = members[0].groupId
  return !groupId || members.some((node) => node.groupId !== groupId)
}

export function canUngroupSelectedNodes(selectedIds: string[], nodes: CanvasNode[]): boolean {
  const selected = new Set(selectedIds)
  return nodes.some((node) => selected.has(node.id) && (node.type === 'group' || Boolean(node.groupId)))
}

/** 连线归一：不许自连 / 不许指向分组 / 两个配置卡之间不连 ✓ */
export function normalizeConnection(fromId: string, toId: string, nodes: CanvasNode[]): { from: string; to: string } | null {
  const first = nodes.find((node) => node.id === fromId)
  const second = nodes.find((node) => node.id === toId)
  if (!first || !second || first.id === second.id) return null
  if (second.type === 'group') return null
  if (first.type === 'config' && second.type === 'config') return null
  return { from: first.id, to: second.id }
}

export interface FocusFrame {
  /** 节点上/下方被浮层占掉的屏幕空间 */
  above?: number
  below?: number
  /** 底部固定栏高度（不参与可用区） */
  bottomInset?: number
}

/**
 * 聚焦某节点的视口：节点占视口 **≤60%**，缩放钳 `0.05–1`（**只缩小不放大** ✓）；
 * 下方有浮层时按 **3:1** 让位，否则上下居中 ✓。
 */
export function focusViewportForNode(
  node: Pick<CanvasNode, 'x' | 'y' | 'w' | 'h'>,
  size: { width: number; height: number },
  frame: FocusFrame = {},
): { x: number; y: number; scale: number } {
  const above = frame.above ?? 0
  const below = frame.below ?? 0
  const usableBottom = Math.max(0, size.height - (frame.bottomInset ?? 0))
  const centerX = node.x + node.w / 2
  const centerY = node.y + node.h / 2
  const verticalRoom = usableBottom * 0.92 - above - below
  const k = Math.min(
    Math.max(Math.min((size.width * 0.6) / node.w, (size.height * 0.6) / node.h, verticalRoom / node.h), 0.05),
    1,
  )
  const groupHeight = above + node.h * k + below
  const slack = usableBottom - groupHeight
  const groupTop = slack * (below > 0 && slack > 0 ? 0.75 : 0.5)
  return {
    x: size.width / 2 - centerX * k,
    y: groupTop + groupHeight / 2 - centerY * k - (below - above) / 2,
    scale: k,
  }
}
