"""修复内核守卫：**提案 / 应用 / 回滚**（源 `assistant-runtime/repair-tools.mjs`）。

这套护栏是「自动改源码」的地基，所以每条拒绝都要有测试钉住 —— 否则某天有人
为了"方便"把「唯一命中」改成「替换第一处」，测试还全绿 ✗。

⚠️ 最要紧的一条是 **CRLF 保真**（见模块头第 3 条坑）：Python 默认换行转换会把
整份 Windows 文件改写成 LF，于是
1. 应用后的文件 diff 全是整行变更 ✗；
2. ``oldContent == 盘上内容`` 的过期校验**永久为假** ✗ ⇒ 之后任何提案都"过期"；
3. 回滚拷贝回来的字节与原文件不一致 ✗。
本测试用 ``\\r\\n`` 文件跑完整「提案 → 应用 → 回滚」并逐字节比对来钉住它。

运行::

    ./.venv/Scripts/python.exe tests/repair_tools_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "backend-py"

_RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, condition: object, detail: object = "") -> None:
    _RESULTS.append((name, bool(condition), detail))


def _expect_refused(name: str, func, *args, **kwargs) -> None:
    """断言这条操作**必须被护栏拒绝** ✓（放行 = 护栏失效）。"""
    from app.services import repair_tools as repair  # noqa: PLC0415

    try:
        func(*args, **kwargs)
    except repair.RepairError as err:
        check(name, True, str(err))
    except Exception as err:  # noqa: BLE001
        check(name, False, f"抛的不是 RepairError 而是 {type(err).__name__}: {err}")
    else:
        check(name, False, "**被放行了**")


def main() -> int:
    sys.path.insert(0, str(BP))
    from app.services import repair_tools as repair  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        app_dir = root / "frontend" / "app"
        app_dir.mkdir(parents=True)
        (root / "backend-py").mkdir()

        target = app_dir / "demo.ts"
        target.write_bytes(b"export const a = 1;\nexport const b = 2;\n")
        crlf = app_dir / "crlf.ts"
        crlf_bytes = b"line one\r\nline two\r\nline three\r\n"
        crlf.write_bytes(crlf_bytes)
        (app_dir / "node_modules").mkdir()
        (app_dir / "node_modules" / "dep.ts").write_text("x\n", encoding="utf-8")
        (app_dir / "package.json").write_text("{}\n", encoding="utf-8")
        (root / "backend-py" / "secret.py").write_text("KEY = 1\n", encoding="utf-8")
        backups = root / ".repair-backups"

        # ① 路径逃逸与绝对路径
        _expect_refused("逃逸: `../outside.ts` 被拒", repair.resolve_inside, root, "../outside.ts")
        _expect_refused("逃逸: 绝对路径被拒", repair.resolve_inside, root, str(app_dir / "demo.ts"))
        # ⚠️ 用例要写**真逃逸**：`frontend/app/../../x.ts` 退两级正好落回仓库根 ⇒ 在 root 内，
        #    放行才对 ✗；退三级才出界 ✓（本守卫第一版就写错了这条，红在用例自己身上）
        _expect_refused("逃逸: `../../../x` 出界被拒", repair.resolve_inside, root, "frontend/app/../../../x.ts")
        check("边界: `frontend/app/../../x.ts` 落在 root 内要放行",
              repair.resolve_inside(root, "frontend/app/../../x.ts") == root / "x.ts")

        # ② 拒绝段 / 拒绝文件 / 白名单 / 文本门槛
        _expect_refused(
            "拒绝段: node_modules 下的文件不能进提案", repair.make_proposal, root,
            "注入依赖", [{"path": "frontend/app/node_modules/dep.ts", "search": "x", "replace": "y"}],
        )
        _expect_refused(
            "拒绝文件: package.json 不能改", repair.make_proposal, root,
            "改依赖", [{"path": "frontend/app/package.json", "search": "{}", "replace": "{ }"}],
        )
        _expect_refused(
            "白名单: 默认不许改 backend-py/", repair.make_proposal, root,
            "改后端", [{"path": "backend-py/secret.py", "search": "KEY = 1", "replace": "KEY = 2"}],
        )
        # 反证：同一条在**显式放开**后必须放行 ✓（否则说明白名单写成了死拒绝）
        try:
            repair.make_proposal(
                root, "改后端", [{"path": "backend-py/secret.py", "search": "KEY = 1", "replace": "KEY = 2"}],
                allowed_prefixes=("frontend/app/", "backend-py/"),
            )
        except repair.RepairError as err:
            check("白名单: 显式放开后放行", False, str(err))
        else:
            check("白名单: 显式放开后放行", True)

        (app_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
        _expect_refused(
            "文本门槛: .png 不在可编辑扩展名里", repair.make_proposal, root,
            "改图", [{"path": "frontend/app/logo.png", "search": "a", "replace": "b"}],
        )

        # ③ 唯一命中 —— 0 次 / 多次 / 1 次
        _expect_refused(
            "唯一命中: search 找不到要拒", repair.make_proposal, root,
            "找不到", [{"path": "frontend/app/demo.ts", "search": "export const zzz = 9;", "replace": "x"}],
        )
        (app_dir / "dup.ts").write_bytes(b"const x = 1;\nconst x = 1;\n")
        _expect_refused(
            "唯一命中: search 出现两次要拒（防误伤同名代码）", repair.make_proposal, root,
            "多处命中", [{"path": "frontend/app/dup.ts", "search": "const x = 1;", "replace": "const x = 2;"}],
        )
        _expect_refused(
            "无变化: replace 与 search 相同要拒", repair.make_proposal, root,
            "空改", [{"path": "frontend/app/demo.ts", "search": "const b = 2;", "replace": "const b = 2;"}],
        )
        _expect_refused(
            "新建: 文件已存在时 create 要拒", repair.make_proposal, root,
            "重复建", [{"path": "frontend/app/demo.ts", "create": True, "replace": "x"}],
        )

        # ④ 正常提案 + 应用 + 备份
        proposal = repair.make_proposal(
            root, "把 b 从 2 改成 3",
            [{"path": "frontend/app/demo.ts", "search": "const b = 2;", "replace": "const b = 3;"}],
        )
        check("提案: 状态是 pending", proposal["status"] == "pending", proposal["status"])
        check("提案: 只有一条 edit", len(proposal["edits"]) == 1, len(proposal["edits"]))
        result = repair.apply_proposal(proposal, root, backups)
        check("应用: 文件内容已改", "const b = 3;" in target.read_text(encoding="utf-8"))
        check("应用: 备份目录已生成", Path(result["backupDir"]).is_dir(), result["backupDir"])
        check("应用: 备份里是旧内容",
              Path(result["backupDir"], "frontend/app/demo.ts").read_text(encoding="utf-8").find("const b = 2;") >= 0)

        # ⑤ 方案过期 —— 应用前别人改了这个文件
        stale = repair.make_proposal(
            root, "再改一次",
            [{"path": "frontend/app/demo.ts", "search": "const b = 3;", "replace": "const b = 4;"}],
        )
        target.write_bytes("export const a = 1;\nexport const b = 3;\n// 别人插入的一行\n".encode("utf-8"))
        _expect_refused("过期检测: 文件已被改过，整单拒绝", repair.apply_proposal, stale, root, backups)

        # ⑥ 回滚 —— 改过的退回、新建的删除
        rb = repair.make_proposal(
            root, "改一处 + 新建一处",
            [
                {"path": "frontend/app/demo.ts", "search": "const b = 3;", "replace": "const b = 99;"},
                {"path": "frontend/app/new.ts", "create": True, "replace": "export const fresh = true;\n"},
            ],
        )
        repair.apply_proposal(rb, root, backups)
        check("回滚前: 新建文件在盘上", (app_dir / "new.ts").exists())
        repair.rollback_proposal(rb, root, backups)
        check("回滚后: 新建的文件被删掉", not (app_dir / "new.ts").exists())
        check("回滚后: 改过的文件退回上一步内容",
              "const b = 3;" in target.read_text(encoding="utf-8"), target.read_text(encoding="utf-8")[:60])

        # ⑦ ⚠️ CRLF 保真（模块头第 3 条坑）
        crlf_proposal = repair.make_proposal(
            root, "改 CRLF 文件",
            [{"path": "frontend/app/crlf.ts", "search": "line two", "replace": "line 2"}],
        )
        repair.apply_proposal(crlf_proposal, root, backups)
        after = crlf.read_bytes()
        check("CRLF: 应用后仍是每行 \\r\\n（没被改写成 LF）",
              after == b"line one\r\nline 2\r\nline three\r\n", after)
        repair.rollback_proposal(crlf_proposal, root, backups)
        check("CRLF: 回滚后与原文件**逐字节相同**", crlf.read_bytes() == crlf_bytes, crlf.read_bytes())

        # ⑧ 写中途失败 → 已写的那几个要就地回滚（源没有这一步，本仓补的）
        pair_a = app_dir / "pair_a.ts"
        pair_b = app_dir / "pair_b.ts"
        pair_a.write_text("A0\n", encoding="utf-8")
        pair_b.write_text("B0\n", encoding="utf-8")
        two = repair.make_proposal(
            root, "两个文件一起改",
            [
                {"path": "frontend/app/pair_a.ts", "search": "A0", "replace": "A1"},
                {"path": "frontend/app/pair_b.ts", "search": "B0", "replace": "B1"},
            ],
        )
        original_write = repair._write_text_file
        calls = {"n": 0}

        def flaky(file: Path, content: str) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("磁盘满（模拟）")
            original_write(file, content)

        repair._write_text_file = flaky  # type: ignore[assignment]
        try:
            repair.apply_proposal(two, root, backups)
            check("写失败: 应当抛出", False, "被吞了")
        except OSError:
            check("写失败: 抛出原始异常", True)
        finally:
            repair._write_text_file = original_write  # type: ignore[assignment]
        check("写失败: 第 1 个文件已被就地回滚", pair_a.read_text(encoding="utf-8") == "A0\n",
              pair_a.read_text(encoding="utf-8"))
        check("写失败: 第 2 个文件保持原样", pair_b.read_text(encoding="utf-8") == "B0\n",
              pair_b.read_text(encoding="utf-8"))

        # ⑨ 只读工具
        listing = repair.list_workspace(root, "frontend/app", 1)
        listed = [item["path"] for item in listing["entries"]]
        check("列目录: 不含 node_modules", not any("node_modules" in path for path in listed), listed[:5])
        found = repair.search_source(root, "const b =", relative="frontend/app")
        check("搜索: 命中 demo.ts", any("demo.ts" in item["path"] for item in found["results"]),
              [item["path"] for item in found["results"]])
        _expect_refused("搜索: 空 query 要拒", repair.search_source, root, "   ")

    failures = [item for item in _RESULTS if not item[1]]
    for name, passed_ok, detail in _RESULTS:
        print(("PASS  " if passed_ok else "FAIL  ") + name + ("" if passed_ok else f"   <<< {detail!r}"))
    print()
    print(f"SUMMARY: {len(_RESULTS) - len(failures)}/{len(_RESULTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
