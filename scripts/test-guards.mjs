#!/usr/bin/env node
/**
 * 守卫自检：把「人工负向实证」固化成可重跑用例
 *
 * 为什么需要
 *   `check-memory.mjs` 的 ①~⑥ 每一项都是**手工造故障 + 手工还原**验证的（首版 ⑥ 正是
 *   这样抓出了假警：扫全文撞上日志摘要里的 `models.vue`）。但手测证据是一次性的，而
 *   「守卫已失效」这件事**不会自己暴露**：校验逻辑被改坏、正则被放宽、白名单被删，
 *   基线都会**照样是绿的**（因为真实仓库本来就合规）⇒ **不报错 ≠ 还能报错**。
 *
 * 原理
 *   把真实 `.codebuddy/memory/` 拷进临时目录 → 用 `MEMORY_DIR` 把守卫指向副本 →
 *   造故障 → 断言「退出码 1 + 命中预期文案」。真仓库全程只读，跑完删临时目录。
 *   仓库根相对路径（`docs/api-contract.md` 等）仍按真实 ROOT 解析 ⇒ 用例贴近实战。
 *
 * 用例（`check-memory.mjs`：基线 1 + 负向 7 ｜ `check-skill-refs.mjs`：基线 1 + 负向 3）
 *   基线  副本未改动          ⇒ 0 致命（防「用例自身把基线弄坏」）
 *   ① 缺 TOPICS.md ｜② MEMORY.md 超 8k ｜③ 锚点越界 ｜④ 末节未登记锚点
 *   ⑤ 磁盘日志未登记 ｜⑥ 落点表路径不存在 ｜⑥b 落点表 §小节指针落空
 *   ⑦ 引用真断链 ｜⑧ `docs/` 引用断链（守住 2026-09-12 才补上的 `docs/` 前缀）
 *   ⑨ 示意引用不误报（`e.g.` 紧邻的路径必须被跳过，否则真信号会被噪声淹没）
 *
 * 两套守卫的原理差别
 *   · `check-memory.mjs` 自带 `MEMORY_DIR` 覆盖 ⇒ 副本可直接指过去。
 *   · `check-skill-refs.mjs` 原先把 `SKILLS_DIR` **硬编码**，直至 2026-09-12 才加
 *     `SKILL_REFS_DIR` 覆盖 —— 这才是它此前无法自检的**唯一**原因（并非"成本不匹配"）。
 *     夹具拷**整棵** skills 树：该守卫 walk 全量 md，小夹具会漏掉真实引用形态。
 *
 * 刻意不做
 *   - **不校验「没报致命」之外的输出**：只认退出码与命中文案，避免把措辞变动变成回归。
 *
 * 用法
 *   node scripts/test-guards.mjs        退出码 1 = 有用例失败（含「夹具失配」）
 */
import { spawnSync } from 'node:child_process';
import {
  appendFileSync,
  copyFileSync,
  cpSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const MEM = join(ROOT, '.codebuddy', 'memory');
const GUARD = join(ROOT, 'scripts', 'check-memory.mjs');
const SKILLS = join(ROOT, 'skills');
const REFS_GUARD = join(ROOT, 'scripts', 'check-skill-refs.mjs');

/** 与 check-memory.mjs 同口径：末尾换行不额外算一行 */
const linesOf = (s) => {
  const a = s.split(/\r?\n/);
  if (a.length && a[a.length - 1] === '') a.pop();
  return a;
};
const SECTION_RE = /^(#{1,4}\s|-\s*【)/;

let baseDir = null;
let sandbox = null;
let skillsBase = null;

/** 每个用例都从真实记忆层的干净副本开始，互不污染 */
const fresh = () => {
  if (!baseDir) baseDir = mkdtempSync(join(tmpdir(), 'mem-guard-'));
  sandbox = join(baseDir, 'memory');
  rmSync(sandbox, { recursive: true, force: true });
  mkdirSync(sandbox);
  for (const f of readdirSync(MEM)) copyFileSync(join(MEM, f), join(sandbox, f));
};

/**
 * skills 树的干净副本（供 check-skill-refs.mjs 的负向实证）。
 * 拷**整棵树**而非造小夹具：该守卫 walk 全量 md，小夹具会漏掉真实文件里的引用形态
 * （示意引用、上游路径、跨 skill `../` 等）；真仓库全程只读，跑完随 skillsBase 一起删除。
 */
const freshSkills = () => {
  if (!skillsBase) skillsBase = mkdtempSync(join(tmpdir(), 'skill-guard-'));
  const box = join(skillsBase, 'skills');
  rmSync(box, { recursive: true, force: true });
  cpSync(SKILLS, box, { recursive: true });
  return box;
};

const at = (f) => join(sandbox, f);
const read = (f) => readFileSync(at(f), 'utf8');
const write = (f, s) => writeFileSync(at(f), s, 'utf8');
/** 精确替换；找不到原文说明夹具已过时 ⇒ 响亮失败，不静默跳过 */
const patch = (f, from, to) => {
  const s = read(f);
  if (!s.includes(from)) throw new Error(`夹具失配：${f} 内找不到 ${JSON.stringify(from)}`);
  write(f, s.replace(from, to));
};
/** INDEX.md 的日志块 [at, end)，使 ③④ 能在正确的日志块内下手（锚点数字会跨日志重复） */
const blocks = (text) => {
  const lines = text.split('\n');
  const hdrs = [];
  lines.forEach((l, i) => {
    const m = l.match(/\*\*`(\d{4}-\d{2}-\d{2}\.md)`\*\*/);
    if (m) hdrs.push({ log: m[1], at: i });
  });
  return hdrs.map((h, k) => ({ ...h, end: k + 1 < hdrs.length ? hdrs[k + 1].at : lines.length, lines }));
};

const CASES = [
  {
    name: '基线（副本未改动）',
    expect: 0,
    mutate: () => {},
  },
  {
    name: '① 缺三层文件（删 TOPICS.md）',
    needle: '缺少三层文件',
    mutate: () => unlinkSync(at('TOPICS.md')),
  },
  {
    name: '② MEMORY.md 超 8k 预算',
    needle: '预算 8000',
    mutate: () => write('MEMORY.md', read('MEMORY.md') + 'x'.repeat(3000)),
  },
  {
    name: '③ 锚点越界（首个日志块改用 @9999）',
    needle: '锚点越界',
    mutate: () => {
      const b = blocks(read('INDEX.md'))[0];
      const ls = [...b.lines];
      for (let i = b.at; i < b.end; i++) {
        const m = ls[i].match(/@\d+/);
        if (m) {
          ls[i] = ls[i].replace(m[0], '@9999');
          return write('INDEX.md', ls.join('\n'));
        }
      }
      throw new Error('夹具失配：INDEX.md 首个日志块内没有锚点');
    },
  },
  {
    name: '④ 日志末节未登记锚点（抹掉最新日志末节锚点）',
    needle: '日志末节未登记锚点',
    mutate: () => {
      const b = blocks(read('INDEX.md')).reduce((a, c) => (c.log > a.log ? c : a));
      const secs = linesOf(read(b.log));
      let last = 0;
      secs.forEach((l, i) => {
        if (SECTION_RE.test(l)) last = i + 1;
      });
      if (!last) throw new Error(`夹具失配：${b.log} 内没有小节行`);
      const ls = [...b.lines];
      for (let i = b.at; i < b.end; i++) {
        const re = new RegExp(`@${last}\\b`);
        if (re.test(ls[i])) {
          ls[i] = ls[i].replace(re, '');
          return write('INDEX.md', ls.join('\n'));
        }
      }
      throw new Error(`夹具失配：INDEX.md 的 ${b.log} 块内找不到 @${last}`);
    },
  },
  {
    name: '⑤ 磁盘日志未登记（造 2099-12-31.md）',
    needle: '日志未登记进 INDEX.md',
    mutate: () => write('2099-12-31.md', '# 探针\n'),
  },
  {
    name: '⑥ 落点表路径不存在',
    needle: '落点表引用的路径不存在',
    mutate: () => patch('INDEX.md', '`docs/api-contract.md`', '`docs/api-contract-missing.md`'),
  },
  {
    name: '⑥b 落点表 §小节指针落空',
    needle: '落点表的小节指针落空',
    mutate: () => patch('INDEX.md', '`MEMORY.md §Skill`', '`MEMORY.md §SkillZZZ`'),
  },
];

/**
 * 引用守卫（`check-skill-refs.mjs`）的用例 —— 与上面同构，但夹具是 skills 树副本，
 * 且通过 `SKILL_REFS_DIR` 把守卫指向副本。探针一律**追加到文件末尾**，故不影响
 * 该文件既有引用的结论；每例都从干净副本重来。
 */
const REF_CASES = [
  { guard: 'refs', name: '基线（skills 副本未改动）', expect: 0 },
  {
    guard: 'refs',
    name: '⑦ 真断链：引用不存在的 references 文件',
    needle: '断链',
    mutate: (box) =>
      appendFileSync(
        join(box, 'prompt-style-library', 'SKILL.md'),
        '\n探针：`references/__no_such_file__.md`\n',
        'utf8'
      ),
  },
  {
    guard: 'refs',
    name: '⑧ docs 引用断链（验证 docs/ 前缀确实纳入校验）',
    needle: '断链',
    mutate: (box) =>
      appendFileSync(
        join(box, 'prompt-style-library', 'SKILL.md'),
        '\n探针：`docs/__no_such_doc__.md`\n',
        'utf8'
      ),
  },
  {
    guard: 'refs',
    name: '⑨ 示意引用不误报（e.g. 紧邻的路径应被跳过）',
    expect: 0,
    mutate: (box) =>
      appendFileSync(
        join(box, 'prompt-style-library', 'SKILL.md'),
        '\n示意引用（e.g. `references/__no_such_file__.md` 只是举例）\n',
        'utf8'
      ),
  },
];

/** 记忆守卫用例 + 引用守卫用例 */
const ALL_CASES = [...CASES, ...REF_CASES];

/** 按用例所属守卫准备夹具并执行 */
const runCase = (c) => {
  if (c.guard === 'refs') {
    const box = freshSkills();
    c.mutate?.(box);
    return spawnSync(process.execPath, [REFS_GUARD], {
      env: { ...process.env, SKILL_REFS_DIR: box },
      encoding: 'utf8',
    });
  }
  fresh();
  c.mutate?.();
  return spawnSync(process.execPath, [GUARD], {
    env: { ...process.env, MEMORY_DIR: sandbox },
    encoding: 'utf8',
  });
};

const passed = [];
const failed = [];
try {
  for (const c of ALL_CASES) {
    try {
      const r = runCase(c);
      const out = `${r.stdout || ''}${r.stderr || ''}`;
      const code = r.status;
      if (c.expect === 0) {
        // 两套守卫的零值文案不同：记忆层「致命 0 处」｜引用层「致命断链 0 处」
        if (code === 0 && /致命[^\n]*0 处/.test(out)) passed.push(`${c.name} → exit 0 / 致命 0`);
        else failed.push(`${c.name} → 期望 exit 0/致命 0，实得 exit ${code}：${out.trim()}`);
      } else if (code === 1 && out.includes(c.needle)) {
        passed.push(`${c.name} → exit 1 / 命中「${c.needle}」`);
      } else {
        failed.push(`${c.name} → 期望 exit 1 且含「${c.needle}」，实得 exit ${code}：${out.trim()}`);
      }
    } catch (e) {
      failed.push(`${c.name} → ${e.message}`);
    }
  }
} finally {
  if (baseDir) rmSync(baseDir, { recursive: true, force: true });
  if (skillsBase) rmSync(skillsBase, { recursive: true, force: true });
}

for (const p of passed) console.log(`✓ ${p}`);
for (const f of failed) console.log(`✗ ${f}`);
console.log(
  `守卫自检：${passed.length}/${ALL_CASES.length} 通过 ｜ check-memory.mjs 基线+①~⑥（${CASES.length} 例）｜ ` +
    `check-skill-refs.mjs 基线+⑦~⑨（${REF_CASES.length} 例）`
);
process.exit(failed.length ? 1 : 0);
