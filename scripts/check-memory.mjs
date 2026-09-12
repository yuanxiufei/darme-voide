#!/usr/bin/env node
/**
 * 记忆层自检：MEMORY.md 体积预算 + INDEX.md 锚点有效性
 *
 * 为什么需要
 *   `.codebuddy/memory/` 的两条不变量全靠人工维护，已反复漂移（第十五轮治理截断、
 *   第十四/十七轮修索引），故补机器校验：
 *     ① MEMORY.md 受注入长度限制（实测 9.4k 即被截断，且**断在半句**——尾部
 *        `## 协作与提交` 最先丢）⇒ **8k 字符是硬预算**，不是建议。
 *     ② INDEX.md 的 `@行号` 是日志跳读入口，日志**只追加** ⇒ 锚点必须永远落在同一
 *        小节的首行；一旦有人重排/改写日志，锚点会静默指错位置。
 *
 * 检查项
 *   致命  ① 三层文件（MEMORY.md / TOPICS.md / INDEX.md）存在
 *         ② MEMORY.md 字符数 ≤ BUDGET（默认 8000）
 *         ③ INDEX.md 每个 `@N`：所属日志存在、N 在范围内、第 N 行是「小节首行」
 *         ④ 每篇日志的**最后一节**都有登记锚点（防「写了日志忘登记索引」→ 新内容不可达）
 *         ⑤ 磁盘上每篇日志都已在 INDEX.md 登记（防「新的一天建了日志忘登记」→ 整篇不可跳读）
 *         ⑥ 「已出栈的落点」表里的路径引用必须存在，且 `文件 §小节` 的**小节名**要在该
 *            文件内真的出现（该小节承诺「优先看这些，别翻日志」，断链 = 读者找不到落点；
 *            只扫本小节，日志摘要里的历史文件名是叙述、不算引用）
 *   提示  ⑦ MEMORY.md 逼近预算（余量 < 400）
 *
 * 刻意不校验 INDEX.md 的「N 行 / M 轮」：索引自述其为快照，且历史取值口径不一致
 *   （实测 6 篇日志恒差 1）⇒ 当检查项只会产生噪声；被保证的不变量是**锚点**。
 * ④ 只查**末节**、不查全量小节：索引对历史日志本就是**摘要式**（只登记重点小节），
 *   但「当天最后一个新小节」漏登记 = 刚写的结论在索引里完全不可达，是真实回归。
 * ⑤⑥ 与 `scripts/check-skill-refs.mjs` 同源动机：**索引是引用层，只增不改**。断链必须
 *   当场报——否则读者按索引找不到权威落点，只能退回整读日志（索引的全部价值）。
 *
 * 约定
 *   - 「小节首行」= `#`~`####` 标题行，或 `- 【…】` 条目行（全仓现存两种写法，
 *     见 `2026-09-03.md` 与其余日志）。新增日志请沿用其一，否则锚点无法校验。
 *   - `@58/66` 视为**两个**锚点（58 与 66 都要过）；`@51~64` 是范围，**只校验 51**。
 *   - 锚点只在 `## 日志清单` 小节内解析，尾部「写入规范」等段落里的 `@N` 不算锚点。
 *
 * 用法
 *   node scripts/check-memory.mjs               退出码 1 = 存在致命项
 *   MEMORY_DIR=<目录> node scripts/check-memory.mjs   校验该目录而非真实记忆层
 *     ↑ 仅供 `scripts/test-guards.mjs` 在临时副本上造故障用；仓库根相对路径仍按真实
 *       ROOT 解析，故 ⑥ 在副本上依然是**实战口径**。
 */
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const MEM = process.env.MEMORY_DIR
  ? resolve(process.env.MEMORY_DIR)
  : join(ROOT, '.codebuddy', 'memory');
const BUDGET = 8000;
const WARN_AT = BUDGET - 400;
const SECTION_RE = /^(#{1,4}\s|-\s*【)/;

const fatal = [];
const info = [];

/** 行数口径与 PowerShell `Get-Content` 一致：末尾换行不额外算一行 */
const linesOf = (s) => {
  const a = s.split(/\r?\n/);
  if (a.length && a[a.length - 1] === '') a.pop();
  return a;
};
const readLines = (p) => linesOf(readFileSync(p, 'utf8'));

// ① 三层文件必须齐（读法链：MEMORY → TOPICS → INDEX）
for (const f of ['MEMORY.md', 'TOPICS.md', 'INDEX.md']) {
  if (!existsSync(join(MEM, f))) fatal.push(`缺少三层文件  .codebuddy/memory/${f}`);
}

const memPath = join(MEM, 'MEMORY.md');
let memLen = 0;
if (existsSync(memPath)) {
  memLen = readFileSync(memPath, 'utf8').length;
  // ② 预算为致命：一旦超限，注入会从尾部截断，`## 协作与提交` 最先丢失
  if (memLen > BUDGET) {
    fatal.push(
      `MEMORY.md ${memLen} 字符 > 预算 ${BUDGET} ⇒ 注入必被截断（尾部「协作与提交」最先丢），请把细节下移 TOPICS.md`
    );
  } else if (memLen > WARN_AT) {
    info.push(`MEMORY.md ${memLen}/${BUDGET}，余量仅 ${BUDGET - memLen} ⇒ 下轮进内容前先下移 TOPICS.md`);
  }
}

// ③ INDEX.md 锚点：按日志分块，块首 `**`YYYY-MM-DD.md`**（…）` 决定后续 @N 归属
const idxPath = join(MEM, 'INDEX.md');
let logs = 0;
let anchors = 0;
/** 日志名 → { lines, nums }：nums 是该日志已登记的锚点集合（供 ④ 用） */
const byLog = new Map();
if (existsSync(idxPath)) {
  let cur = null;
  let curLines = null;
  for (const line of readLines(idxPath)) {
    const marker = line.match(/\*\*`(\d{4}-\d{2}-\d{2}\.md)`\*\*/);
    if (marker) {
      const p = join(MEM, marker[1]);
      if (!existsSync(p)) {
        fatal.push(`INDEX.md 引用的日志不存在  ${marker[1]}`);
        cur = null;
        continue;
      }
      cur = marker[1];
      curLines = readLines(p);
      logs++;
      if (!byLog.has(cur)) byLog.set(cur, { lines: curLines, nums: new Set() });
      continue;
    }
    // 锚点只在「## 日志清单」小节内解析（尾部「写入规范」等段落里的 @N 不算锚点）
    if (/^##\s/.test(line) && !/^##\s*日志清单/.test(line)) {
      cur = null;
      continue;
    }
    if (!cur) continue;
    // `@N` 或 `@N/M/...`（斜杠链视为多个锚点；`~` 是范围，只取起点）
    for (const m of line.matchAll(/@(\d+(?:\s*\/\s*\d+)*)/g)) {
      for (const num of m[1].match(/\d+/g)) {
        const n = Number(num);
        anchors++;
        byLog.get(cur)?.nums.add(n);
        if (n < 1 || n > curLines.length) {
          fatal.push(`锚点越界  ${cur} @${n}（该日志仅 ${curLines.length} 行）`);
        } else if (!SECTION_RE.test(curLines[n - 1])) {
          fatal.push(`锚点未落在小节首行  ${cur} @${n}  → ${curLines[n - 1].slice(0, 40)}`);
        }
      }
    }
  }
}

// ④ 每篇日志的末节必须有锚点：日志只追加 ⇒ 忘登记 = 刚写的结论在索引里完全不可达
for (const [name, { lines, nums }] of byLog) {
  let last = 0;
  lines.forEach((l, i) => {
    if (SECTION_RE.test(l)) last = i + 1;
  });
  if (last && !nums.has(last)) {
    fatal.push(
      `日志末节未登记锚点  ${name} @${last}（共 ${lines.length} 行，末节从第 ${last} 行起）`
    );
  }
}

// ⑤ 磁盘上的每日日志必须都已登记进 INDEX.md（新的一天最容易漏；漏了则整篇不可跳读）
let diskLogs = 0;
if (existsSync(idxPath)) {
  for (const f of readdirSync(MEM)) {
    if (!/^\d{4}-\d{2}-\d{2}\.md$/.test(f)) continue;
    diskLogs++;
    if (!byLog.has(f)) {
      fatal.push(`日志未登记进 INDEX.md  ${f}（整篇小节都无法按 @行号 跳读）`);
    }
  }
}

// ⑥ 「已出栈的落点」表引用的路径必须存在（表本身承诺「优先看这些，别翻日志」）
//    ⚠️ 只扫该小节：日志清单里的摘要文字会提到历史文件名（如 `models.vue`），那些是
//    **叙述**不是引用，扫全文必误报（实测首版即被 `models.vue` 撞出致命 1 处）。
let refs = 0;
let secRefs = 0;
if (existsSync(idxPath)) {
  let inRefTable = false;
  for (const line of readLines(idxPath)) {
    if (/^##\s/.test(line)) inRefTable = /^##\s*已出栈的落点/.test(line);
    if (!inRefTable) continue;
    for (const m of line.matchAll(/`([^`]+)`/g)) {
      // `TOPICS.md §小节` → 拆成「文件」+「小节名」，**两段都要落地**
      const [filePart, ...secRest] = m[1].split('§');
      const sec = secRest.join('§').trim();
      const raw = filePart.trim();
      if (!raw || /[<>*…"'()|]/.test(raw) || raw.startsWith('http')) continue;
      const fromRoot = raw.includes('/');
      // 裸文件名只认三层记忆文件与日期日志，其余（`models.vue` 之类）是叙述
      if (!fromRoot && !/^(MEMORY|TOPICS|INDEX)\.md$|^\d{4}-\d{2}-\d{2}\.md$/.test(raw)) continue;
      const target = join(fromRoot ? ROOT : MEM, raw);
      refs++;
      if (!existsSync(target)) {
        fatal.push(`落点表引用的路径不存在  ${raw}（相对${fromRoot ? '仓库根' : 'memory 目录'}）`);
      } else if (sec) {
        secRefs++;
        if (!readFileSync(target, 'utf8').includes(sec)) {
          fatal.push(`落点表的小节指针落空  ${raw} §${sec}（该文件内查不到此小节名）`);
        }
      }
    }
  }
}

// 输出：与 scripts/check-skill-refs.mjs 保持同款格式
for (const f of fatal) console.log(`✗ ${f}`);
for (const i of info) console.log(`! ${i}`);
console.log(
  `扫描 .codebuddy/memory ｜ 日志 ${logs} 篇已登记 / ${diskLogs} 篇在盘 ｜ ` +
    `MEMORY.md ${memLen}/${BUDGET} 字符 ｜ 校验锚点 ${anchors} 处、落点表路径 ${refs} 处` +
    `（其中 §小节指针 ${secRefs} 处）｜ 致命 ${fatal.length} 处`
);
process.exit(fatal.length ? 1 : 0);
