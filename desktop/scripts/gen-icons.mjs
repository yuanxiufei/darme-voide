#!/usr/bin/env node
/**
 * 生成 Tauri 需要的全套图标（PNG / ICO / ICNS），**不依赖任何 npm 包**。
 *
 * 为什么手写：图标是打包的硬前置（缺 `icon.ico` 时 Windows 打包直接失败），而
 * `tauri icon` 要额外装 sharp/CLI；本项目对依赖体积敏感 ⇒ 用 Node 内置 `zlib` 编 PNG，
 * 用 PNG-in-ICO / PNG-in-ICNS 容器打包（Vista+ / macOS 10.7+ 均支持）。
 *
 * 图标是**程序化绘制**的（深色圆角底 + 渐变环 + 中心点），不存任何二进制素材，
 * 因此仓库里不会出现不可 review 的大 blob。
 */
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { deflateSync } from 'node:zlib'

const here = dirname(fileURLToPath(import.meta.url))
const outDir = resolve(here, '..', 'src-tauri', 'icons')

/* ---------- PNG 编码 ---------- */

const CRC_TABLE = (() => {
  const table = new Int32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c
  }
  return table
})()

function crc32(buf) {
  let c = -1
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8)
  return (c ^ -1) >>> 0
}

function chunk(type, data) {
  const len = Buffer.alloc(4)
  len.writeUInt32BE(data.length, 0)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(body), 0)
  return Buffer.concat([len, body, crc])
}

/** RGBA 像素（Uint8Array，长度 w*h*4）→ PNG Buffer */
function encodePng(size, rgba) {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(size, 0)
  ihdr.writeUInt32BE(size, 4)
  ihdr.writeUInt8(8, 8) // bit depth
  ihdr.writeUInt8(6, 9) // color type: RGBA
  ihdr.writeUInt8(0, 10)
  ihdr.writeUInt8(0, 11)
  ihdr.writeUInt8(0, 12)

  const stride = size * 4
  const raw = Buffer.alloc((stride + 1) * size)
  for (let y = 0; y < size; y++) {
    raw[y * (stride + 1)] = 0 // filter: none
    rgba.copy(raw, y * (stride + 1) + 1, y * stride, y * stride + stride)
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

/* ---------- 绘制 ---------- */

const BG = [0x14, 0x18, 0x1f]
const RING_A = [0x4f, 0x8c, 0xff]
const RING_B = [0xa0, 0x6b, 0xff]

function draw(size) {
  const buf = Buffer.alloc(size * size * 4)
  const r = size * 0.2 // 圆角半径
  const cx = size / 2
  const cy = size / 2
  const ringOuter = size * 0.35
  const ringInner = size * 0.235
  const dot = size * 0.085

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const px = x + 0.5
      const py = y + 0.5
      let [cr, cg, cb] = BG
      let alpha = 255

      // 圆角矩形遮罩：四角落在圆角方框外就透明
      const qx = Math.max(r - px, px - (size - r), 0)
      const qy = Math.max(r - py, py - (size - r), 0)
      if (Math.hypot(qx, qy) > r) alpha = 0

      const d = Math.hypot(px - cx, py - cy)
      if (d <= ringOuter && d >= ringInner) {
        const t = (px / size + py / size) / 2
        cr = Math.round(RING_A[0] + (RING_B[0] - RING_A[0]) * t)
        cg = Math.round(RING_A[1] + (RING_B[1] - RING_A[1]) * t)
        cb = Math.round(RING_A[2] + (RING_B[2] - RING_A[2]) * t)
      } else if (d < dot) {
        cr = 0xff
        cg = 0xff
        cb = 0xff
      }

      const o = (y * size + x) * 4
      buf[o] = cr
      buf[o + 1] = cg
      buf[o + 2] = cb
      buf[o + 3] = alpha
    }
  }
  return buf
}

/* ---------- ICO / ICNS 容器 ---------- */

function encodeIco(items) {
  const header = Buffer.alloc(6)
  header.writeUInt16LE(0, 0)
  header.writeUInt16LE(1, 2)
  header.writeUInt16LE(items.length, 4)

  let offset = 6 + 16 * items.length
  const entries = []
  for (const { size, data } of items) {
    const entry = Buffer.alloc(16)
    entry.writeUInt8(size >= 256 ? 0 : size, 0)
    entry.writeUInt8(size >= 256 ? 0 : size, 1)
    entry.writeUInt8(0, 2)
    entry.writeUInt8(0, 3)
    entry.writeUInt16LE(1, 4) // planes
    entry.writeUInt16LE(32, 6) // bit count
    entry.writeUInt32LE(data.length, 8)
    entry.writeUInt32LE(offset, 12)
    offset += data.length
    entries.push(entry)
  }
  return Buffer.concat([header, ...entries, ...items.map((i) => i.data)])
}

function encodeIcns(items) {
  const parts = []
  for (const { type, data } of items) {
    const head = Buffer.alloc(8)
    head.write(type, 0, 4, 'ascii')
    head.writeUInt32BE(data.length + 8, 4)
    parts.push(head, data)
  }
  const body = Buffer.concat(parts)
  const head = Buffer.alloc(8)
  head.write('icns', 0, 4, 'ascii')
  head.writeUInt32BE(body.length + 8, 4)
  return Buffer.concat([head, body])
}

/* ---------- 落盘 ---------- */

await mkdir(outDir, { recursive: true })

const sizes = [32, 64, 128, 256, 512]
const pngs = new Map()
for (const size of sizes) pngs.set(size, encodePng(size, draw(size)))

const files = [
  ['32x32.png', pngs.get(32)],
  ['128x128.png', pngs.get(128)],
  ['128x128@2x.png', pngs.get(256)],
  ['icon.png', pngs.get(512)],
  ['icon.ico', encodeIco([
    { size: 32, data: pngs.get(32) },
    { size: 64, data: pngs.get(64) },
    { size: 128, data: pngs.get(128) },
    { size: 256, data: pngs.get(256) },
  ])],
  ['icon.icns', encodeIcns([
    { type: 'ic07', data: pngs.get(128) },
    { type: 'ic08', data: pngs.get(256) },
    { type: 'ic09', data: pngs.get(512) },
  ])],
]

for (const [name, data] of files) {
  await writeFile(join(outDir, name), data)
  console.log(`[icons] ${name.padEnd(16)} ${(data.length / 1024).toFixed(1)} KB`)
}
console.log(`[icons] 输出目录：${outDir}`)
