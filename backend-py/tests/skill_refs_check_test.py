"""``scripts/check_skill_refs`` 的解析基准守卫 —— 四类基准各自判对，判不了的要**说得出**为什么。

为什么单独立一套：这个脚本的判据全是「哪一类路径该按谁做基准」。写错基准不会报错，
只会把真断链**判成绿**（或者把好路径判成红），而两种错在真树上都不明显 ——
2026-11-04 实测两例：① 跨 skill 引用（`film-shot/references/x.md`）全被当「基准不明」
跳过 ⇒ 指空也能一直绿；② 含通配符的模式在收集阶段就被丢掉，连跳过计数都不进。

所以这里不测「扫得全不全」，只测**分类对不对**：每类基准各造一棵最小树，正反两面都验，
并额外跑一次真树确认没有把既有基线弄脏。

跑法::

    .venv/Scripts/python.exe -X utf8 tests/skill_refs_check_test.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_PY = Path(__file__).resolve().parents[1]
SCRIPT = BACKEND_PY / "app" / "scripts" / "check_skill_refs.py"
sys.stdout.reconfigure(encoding="utf-8")

_PASSED = 0
_FAILED = 0
SUMMARY_RE = re.compile(
    r"待校验 (\d+) 处 ｜ 致命断链 (\d+) 处 ｜ 非致命缺脚本 (\d+) 处 ｜ "
    r"资产无人指路 (\d+) 份 ｜ "
    r"跳过：上游/外来宿主 (\d+) 处 / 示意引用 (\d+) 处 / 基准不明 (\d+) 处"
)


def check(name: str, condition: object, detail: object = None) -> None:
    global _PASSED, _FAILED
    if condition:
        _PASSED += 1
        print(f"  PASS  {name}")
    else:
        _FAILED += 1
        print(f"  FAIL  {name}")
        if detail is not None:
            print(f"        <<< {detail!r}")


def run(skills_root: Path | None, *args: str) -> tuple[int, str, dict]:
    """跑一次脚本，返回 ``(退出码, 输出, 计数)``；计数解析失败时给全 0。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if skills_root is None:
        env.pop("SKILL_REFS_DIR", None)
    else:
        env["SKILL_REFS_DIR"] = str(skills_root)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    out = proc.stdout + proc.stderr
    match = SUMMARY_RE.search(out)
    counts = {
        "checked": int(match.group(1)),
        "fatal": int(match.group(2)),
        "notes": int(match.group(3)),
        "orphans": int(match.group(4)),
        "upstream": int(match.group(5)),
        "illustrative": int(match.group(6)),
        "ambiguous": int(match.group(7)),
    } if match else {}
    return proc.returncode, out, counts


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def case_cross_skill(root: Path) -> None:
    """④ 跨 skill 引用：`<另一个 skill 名>/references/x.md` 按本库 `skills/` 根解析。"""
    print("\n[④ 跨 skill 引用]")
    write(root / "demo-a" / "SKILL.md", "# A\n\n细节见 `demo-b/references/hit.md` 那份。\n")
    write(root / "demo-b" / "SKILL.md", "# B\n")
    write(root / "demo-b" / "references" / "hit.md", "# 受击\n")
    code, out, counts = run(root)
    check("跨 skill 引用指向真文件 ⇒ 绿", code == 0 and counts.get("fatal") == 0,
          (code, counts, out[-300:]))
    # 表头那行「扫描 <目录>」曾写死成 `backend-py/app/skills`：换了 SKILL_REFS_DIR 仍在报原路径，
    # 回执与实扫不一致（读的人会以为扫的是真库，从而忽略红字）。
    check("表头报的扫描目录随 `SKILL_REFS_DIR` 变（不是写死的路径）",
          str(root).replace("\\", "/") in out.replace("\\", "/"), out[-300:])
    check("跨 skill 引用真的进了「待校验」（不再算基准不明）",
          counts.get("checked") == 1 and counts.get("ambiguous") == 0, counts)

    (root / "demo-b" / "references" / "hit.md").unlink()
    code, out, counts = run(root)
    check("跨 skill 引用指空 ⇒ 致命 + 退出 1（旧行为是静默跳过）",
          code == 1 and counts.get("fatal") == 1, (code, counts))
    check("报错行点名那个串", "demo-b/references/hit.md" in out, out[-300:])


def case_library_qualified(root: Path) -> None:
    """⑥ 库限定引用：`<库目录>/<skill>/references/x.md` 从 `skills/` 根起算。

    库目录＝`skills/` 下一层**不含 SKILL.md 的容器**（`wuxia-action` 这种）。这类写法在
    库内文档与 README 里都自然，早先它不在任何基准里 ⇒ 落「基准不明」被跳过，而其中
    README 那条指向的文件**确实存在**（等于把可判的链接白放过了）。
    """
    print("\n[⑥ 库限定引用]")
    write(root / "lib-box" / "inner-skill" / "SKILL.md", "# 内\n")
    write(root / "lib-box" / "inner-skill" / "references" / "k.md", "# 库件\n")
    write(root / "holder" / "SKILL.md", "# 托\n\n原句在 `lib-box/inner-skill/references/k.md` 。\n")
    code, out, counts = run(root)
    check("库限定路径指对 ⇒ 绿且计入待校验（旧行为是落基准不明）",
          code == 0 and counts.get("checked") == 1 and counts.get("ambiguous") == 0,
          (code, counts, out[-300:]))

    (root / "lib-box" / "inner-skill" / "references" / "k.md").unlink()
    code, out, counts = run(root)
    check("库限定路径指空 ⇒ 致命 + 退出 1（库目录存在就不再宽容）",
          code == 1 and counts.get("fatal") == 1, (code, counts))

    write(root / "holder" / "SKILL.md", "# 托\n\n`no-such-lib/inner-skill/references/k.md`\n")
    code, out, counts = run(root)
    check("库名不存在 ⇒ 仍归基准不明且退出 0（库名写错不猜、也不乱判红）",
          code == 0 and counts.get("ambiguous") == 1 and counts.get("fatal") == 0, counts)


def case_orphan_assets(root: Path) -> None:
    """⑦ 反向：`references/` 与 `scripts/` 下的资产必须被本 skill 的 md 指路。

    断链是「指向空白」，孤儿是「空白无人指向」—— 两种都让知识进不了模型上下文，而检查一直
    只盯前者。2026-11-06 实测 121 份资产里 12 份孤儿，全是真的能力白废：`ui-motion` 正文
    写着 "The 8 style files" 却没说它们在 `references/styles/`、字幕技能的术语对照表无人指路
    （Step 1 反倒让用户手抄）、`skill-creator` 只指英文版 `NAMING.md` 而中文版没人读。
    """
    print("\n[⑦ 资产无人指路]")
    write(root / "orph-skill" / "references" / "t.md", "# 术语\n")
    write(root / "orph-skill" / "SKILL.md", "# 傲\n\n先按流程走一遍。\n")
    code, out, counts = run(root)
    check("references 里有文件但正文不提 ⇒ 报 1 且退出 1",
          code == 1 and counts.get("orphans") == 1, (code, counts))
    check("点名到 skill 与文件（人能直接去补指路）",
          "orph-skill" in out and "references/t.md" in out, out[-300:])

    write(root / "orph-skill" / "SKILL.md", "# 傲\n\n先读 `references/t.md` 再动手。\n")
    code, out, counts = run(root)
    check("点名文件名 ⇒ 不报（最常见的指路形态）",
          code == 0 and counts.get("orphans") == 0, (code, counts))

    (root / "orph-skill" / "references" / "t.md").unlink()
    write(root / "orph-skill" / "references" / "styles" / "A.md", "# 样式 A\n")
    write(root / "orph-skill" / "SKILL.md", "# 傲\n\n样式文件都在 `references/styles/` 下。\n")
    code, out, counts = run(root)
    check("多级目录整体指路（references/styles/）⇒ 不报",
          code == 0 and counts.get("orphans") == 0, (code, counts))

    write(root / "orph-skill" / "SKILL.md", "# 傲\n\n资源放 `references/` 目录。\n")
    code, out, counts = run(root)
    check("只说 references/ 目录而不点名 ⇒ 仍报（「有这目录」不等于「读这一份」）",
          code == 1 and counts.get("orphans") == 1, (code, counts))

    write(root / "orph-skill" / "NOTICE.md", "# 凭据\n")
    write(root / "orph-skill" / "agents" / "openai.yaml", "a: 1\n")
    write(root / "orph-skill" / "SKILL.md", "# 傲\n\n先读 `references/styles/A.md` 再动手。\n")
    code, out, counts = run(root)
    check("NOTICE.md / agents/ 这类平台与元数据文件 ⇒ 不参与判定",
          code == 0 and counts.get("orphans") == 0, (code, counts))


def case_orphan_net(root: Path) -> None:
    """⑦b 判据的**网本身**：哪些类型进网、谁说的话算数。

    两处都是 2026-11-06 真机反验才露出来的：

    ① **类型网**原本是正向白名单（`.md/.py/.json/.yaml/.yml/.ts/.sh/.txt`），有 `.ts`、有
       `.sh`、偏偏没有 `.js` —— 真树上 `multi-shot` 与 `n-storyboard` 的 `scripts/index.js`
       压根不在网内。当轮它们恰好被人指路才没露症状，但形状是错的：正向白名单意味「新加的
       类型从落地那刻起默认不受看管，且永远不报」。改成反白名单后默认在网内。
    ② **同名代答**：原来看的是**整棵树**的 md，于是抹掉 `multi-shot` 对 `scripts/index.js`
       的点名后，`n-storyboard` 里那句同名的 `entry: "scripts/index.js"` 仍在全树文本中
       ⇒ 孤儿数照旧 0、一声不响。现在裸名只在本 skill 内算数，跨 skill 必须带 `<skill>/` 前缀。
    """
    print("\n[⑦b 判据的网：类型与「谁说的话算数」]")
    lib = root

    # ① 类型网：.js 与任何一种「没预料到的扩展名」都必须在网内
    write(lib / "net-a" / "scripts" / "index.js", "// entry\n")
    write(lib / "net-a" / "SKILL.md", "# 甲\n\n按流程走。\n")
    code, out, counts = run(lib)
    check("scripts/ 下的 .js 无人指路 ⇒ 必须报（旧的白名单没有 .js，整份网外）",
          code == 1 and counts.get("orphans") == 1
          and "scripts/index.js" in out, (code, counts, out[-200:]))

    write(lib / "net-a" / "SKILL.md", "# 甲\n\n入口 `scripts/index.js`。\n")
    code, out, counts = run(lib)
    check("点名 .js ⇒ 不报", code == 0 and counts.get("orphans") == 0, (code, counts))

    write(lib / "net-a" / "references" / "shot.png", "\x89PNG")
    code, out, counts = run(lib)
    check("图片放进 references/ 且无人指路 ⇒ 不报（图无从「指路」，本就不该被要求）",
          code == 0 and counts.get("orphans") == 0, (code, counts))

    write(lib / "net-a" / "references" / "probe.mjs", "export default 1\n")
    code, out, counts = run(lib)
    check("没预料到的新类型（.mjs）无人指路 ⇒ 必须报（默认在网内，不是默认豁免）",
          code == 1 and counts.get("orphans") == 1
          and "references/probe.mjs" in out, (code, counts, out[-200:]))
    (lib / "net-a" / "references" / "probe.mjs").unlink()

    # ② 同名代答：两个 skill 各有一份同名文件，只点名了其中一个
    write(lib / "twin-x" / "scripts" / "main.js", "// x\n")
    write(lib / "twin-x" / "SKILL.md", "# 寅\n\n直接开跑。\n")
    write(lib / "twin-y" / "scripts" / "main.js", "// y\n")
    write(lib / "twin-y" / "SKILL.md", "# 卯\n\n入口 `scripts/main.js`。\n")
    code, out, counts = run(lib)
    check("同名文件：兄弟 skill 点了名、这一份没人点 ⇒ 仍报（不许被同名句代答）",
          code == 1 and counts.get("orphans") == 1
          and "twin-x" in out and "scripts/main.js" in out, (code, counts, out[-300:]))

    write(lib / "twin-x" / "SKILL.md", "# 寅\n\n见 `twin-y/scripts/main.js` 的写法。\n")
    code, out, counts = run(lib)
    check("跨 skill 指路带 `<skill>/` 前缀 ⇒ 认（这是明确的「去读那一份」）",
          code == 0 and counts.get("orphans") == 0, (code, counts))

    # 别名：目录名与 frontmatter name 不一致时，两种写法都得认
    write(lib / "twin-z" / "references" / "z.md", "# z\n")
    write(lib / "twin-z" / "SKILL.md", "---\nname: twin-zed\n---\n\n# 辰\n\n先做准备工作。\n")
    write(lib / "twin-x" / "SKILL.md",
          "# 寅\n\n见 `twin-y/scripts/main.js`，搭片法见 `twin-zed/references/z.md`。\n")
    code, out, counts = run(lib)
    check("跨 skill 用 frontmatter 别名写 ⇒ 也认（目录名与 name 本就不保证一致）",
          code == 0 and counts.get("orphans") == 0, (code, counts, out[-300:]))


def case_ambiguous(root: Path) -> None:
    """同名 skill 有多处 ⇒ 不猜，仍归「基准不明」且不判红。"""
    print("\n[同名 skill 歧义]")
    write(root / "lib-a" / "storyboard" / "SKILL.md", "# 甲\n")
    write(root / "lib-b" / "storyboard" / "SKILL.md", "# 乙\n")
    write(root / "holder" / "SKILL.md", "# 丙\n\n见 `storyboard/references/nope.md` 。\n")
    code, out, counts = run(root)
    check("同名两处 ⇒ 退出 0、不判红（猜错会把真断链判绿）",
          code == 0 and counts.get("fatal") == 0, (code, counts, out[-300:]))
    check("同名两处 ⇒ 落进「基准不明」并计数", counts.get("ambiguous") == 1, counts)


def case_relative(root: Path) -> None:
    """② 真·相对路径：`../other/references/x.md` 相对**当前文档目录**。"""
    print("\n[② 真·相对路径]")
    write(root / "lib-x" / "alpha" / "SKILL.md",
          "# α\n\n接 `../beta/references/move.md` 。\n")
    write(root / "lib-x" / "beta" / "references" / "move.md", "# 走位\n")
    code, _, counts = run(root)
    check("相对路径指对 ⇒ 绿", code == 0 and counts.get("fatal") == 0, (code, counts))

    (root / "lib-x" / "beta" / "references" / "move.md").unlink()
    code, _, counts = run(root)
    check("相对路径指空 ⇒ 致命", code == 1 and counts.get("fatal") == 1, (code, counts))


def case_asset_in_refs(root: Path) -> None:
    """① 写在 `references/foo.md` 内部的 `references/x.md` 仍按 **skill 根**解析。"""
    print("\n[① references/ 内部引用]")
    write(root / "lib-y" / "gamma" / "SKILL.md", "# γ\n\n细表见 `references/outline.md` 。\n")
    write(root / "lib-y" / "gamma" / "references" / "outline.md",
          "# 大纲\n\n细表见 `references/table.md` 。\n")
    write(root / "lib-y" / "gamma" / "references" / "table.md", "# 表\n")
    code, _, counts = run(root)
    check("附件里指同一 skill 的附件 ⇒ 绿（按 skill 根，不是 references/）",
          code == 0 and counts.get("fatal") == 0, (code, counts))

    (root / "lib-y" / "gamma" / "references" / "table.md").unlink()
    code, _, counts = run(root)
    check("上面那条指空 ⇒ 致命（真按 references/ 解析就会误判成绿）",
          code == 1 and counts.get("fatal") == 1, (code, counts))


def case_pattern(root: Path) -> None:
    """通配符模式：命中 ≥1 个真文件才算数（早先整类在收集阶段就被丢掉）。"""
    print("\n[通配符模式]")
    write(root / "lib-z" / "delta" / "SKILL.md", "# δ\n\n全套见 `references/*.md` 。\n")
    write(root / "lib-z" / "delta" / "references" / "one.md", "# 一\n")
    code, _, counts = run(root)
    check("模式命中 ⇒ 绿且计入待校验", code == 0 and counts.get("checked") == 1, (code, counts))

    (root / "lib-z" / "delta" / "references" / "one.md").unlink()
    code, out, counts = run(root)
    check("模式一个都没命中 ⇒ 致命（指着空目录写 *.md 不能再绿）",
          code == 1 and counts.get("fatal") == 1, (code, counts))
    check("报错点明是「模式未命中」而不是单个文件名", "模式未命中" in out, out[-300:])

    write(root / "lib-z" / "delta" / "SKILL.md", "# δ\n\n上游格式见 `.ci/prompts/desc-*.md` 。\n")
    code, _, counts = run(root)
    check("上游宿主前缀的模式 ⇒ 跳过、不判红", code == 0 and counts.get("upstream") == 1,
          (code, counts))


def case_name_alias(root: Path) -> None:
    """引用里写 skill 的 **frontmatter name**（不是目录名）也要能判 —— 本仓真有这种双名。"""
    print("\n[frontmatter name 作别名]")
    write(root / "snake_one" / "SKILL.md",
          "---\nname: kebab-one\n---\n\n# 一\n")
    write(root / "holder" / "SKILL.md", "# 托\n\n细节见 `kebab-one/references/x.md` 。\n")
    write(root / "snake_one" / "references" / "x.md", "# 叉\n")
    code, out, counts = run(root)
    check("按 frontmatter name 指对 ⇒ 绿且计入待校验（旧行为是落基准不明静默跳过）",
          code == 0 and counts.get("checked") == 1 and counts.get("ambiguous") == 0,
          (code, counts, out[-300:]))

    (root / "snake_one" / "references" / "x.md").unlink()
    code, _, counts = run(root)
    check("按别名指空 ⇒ 致命（别名不是「爱指哪儿指哪儿」）",
          code == 1 and counts.get("fatal") == 1, (code, counts))

    # 两个 skill 声明同一个 name ⇒ 与同名目录一样：不猜、归基准不明
    write(root / "snake_one" / "references" / "x.md", "# 叉\n")
    write(root / "twin_a" / "SKILL.md", "---\nname: kebab-one\n---\n# 甲\n")
    write(root / "twin_b" / "SKILL.md", "---\nname: kebab-one\n---\n# 乙\n")
    code, _, counts = run(root)
    check("两处声明同名 ⇒ 退出 0 且落基准不明（不猜）",
          code == 0 and counts.get("ambiguous") == 1 and counts.get("fatal") == 0, counts)


def case_real_tree() -> None:
    """真树不许被这套规则弄脏：基线仍是致命 0。"""
    print("\n[真树]")
    code, out, counts = run(None)
    check("真 `app/skills/**` 扫完退出 0", code == 0, out[-400:])
    check("真树致命 0 / 非致命 0", counts.get("fatal") == 0 and counts.get("notes") == 0, counts)
    check("真树确实扫到了东西（待校验 > 0，别是空跑）", counts.get("checked", 0) > 50, counts)


def main() -> int:
    print("=" * 68)
    print("check_skill_refs 解析基准守卫")
    print("=" * 68)

    for case in (case_cross_skill, case_library_qualified, case_orphan_assets, case_orphan_net,
                 case_ambiguous, case_relative, case_asset_in_refs, case_pattern, case_name_alias):
        tmp = Path(tempfile.mkdtemp(prefix="skillrefs-"))
        try:
            case(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    case_real_tree()

    print(f"\nSUMMARY: {_PASSED}/{_PASSED + _FAILED} passed")
    return 0 if _FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
