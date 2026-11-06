"""版面分栏重排：把「按 y 顺序」的 OCR 行改成「按栏序 → 栏内 y」的正确阅读顺序。

为什么必须有这一步
------------------
rapidocr 的输出只有检测框 + 文本，排序按 y。单栏页没问题；**双栏页**会把
「左栏第 1 行、右栏第 1 行、左栏第 2 行……」交替拼接 —— 字数、标点、行数全都正常，
``judge_page`` 判 ok，句子却整段错位。这类错误进库后**无法自动化发现**（结构上完全
合法），只能靠人读，所以必须在页文本落库之前修掉。

做法：纯几何，零依赖
--------------------
把每行的 ``[x0, x1]`` 投影到 x 轴，找「没有任何一行跨越」的空白带 ⇒ 那就是栏间距。
有空白带才分栏，没有就原样返回（单栏、表格页、满页插图不会被动一个字节）。

因此本模块对单栏语料是**恒等变换**，不存在「装了反而更差」的可能 —— 这也是不把
RapidLayout/PP-DocLayout 设成硬依赖的理由：分栏这件事几何就够，而且几何不会幻觉。
模型版式分析能多给「页眉/图注/表格」这类类别，属于可选的第二层增强，不是分栏的前提。

跨栏行的处理
------------
标题、整页表格线常常横跨中缝。这些行不能塞进任何一栏，否则会把一栏的结尾和另一栏
的开头拼在一起。做法是把它们当作**分段锚点**：锚点之上、锚点之下各自分栏排序，
锚点自己按 y 插在段间 —— 输出顺序就与人的读法一致了。
"""
from __future__ import annotations

__all__ = ["analyze", "order_lines", "text_of", "bbox_of"]

#: 栏间空白的最小像素宽度（150 dpi 下中缝一般 ≥20px；18 是留了余量的下限）
COLUMN_GAP_MIN_PX = 18
#: 同样的空白带相对页宽的下限（大页面按比例放大，小图按像素保底）
COLUMN_GAP_MIN_SHARE = 0.015
#: 行宽占版面宽这个比例 ⇒ 认定是跨栏行（标题/表格/分隔线）
WIDE_LINE_SHARE = 0.7
#: 少于这么多行不做分栏（两三行的页没有统计意义，判定必不稳）
MIN_LINES_FOR_LAYOUT = 6
#: 每栏至少这么多行才承认这是一栏（防止把「一个孤零零的页码」切成第二栏）
MIN_COLUMN_LINES = 2
#: 三栏以上的递归深度上限（期刊最多三栏；再深就说明检测已经不可信了）
MAX_DEPTH = 2


def bbox_of(line) -> tuple[float, float, float, float]:
    """``[框, 文本, 分]`` → ``(x0, y0, x1, y1)``。框可能是 numpy 数组或嵌套 list。"""
    pts = [(float(point[0]), float(point[1])) for point in line[0]]
    xs = [point[0] for point in pts]
    ys = [point[1] for point in pts]
    return min(xs), min(ys), max(xs), max(ys)


def text_of(line) -> str:
    """取一行的文本（`[框, 文本, 分]` 的第 2 位）。"""
    return str(line[1]) if len(line) > 1 and line[1] is not None else ""


def _yc(box: tuple[float, float, float, float]) -> float:
    return (box[1] + box[3]) / 2.0


def _xc(box: tuple[float, float, float, float]) -> float:
    return (box[0] + box[2]) / 2.0


def _blank_bands(boxes: list[tuple], *, min_gap: float) -> list[tuple[float, float]]:
    """x 轴上「没有任何一行覆盖」的空白带，按位置升序。

    事件扫描：每行的左右边各算一次进出，覆盖深度回到 0 且持续 ``min_gap`` 即是一条带。
    排序时 ``-1`` 排在 ``+1`` 之前 —— 相邻两行首尾相接（x1 恰等于下一个 x0）时不能
    被当成一条缝，否则紧排的正文会被误切成两栏。
    """
    edges: list[tuple[float, int]] = []
    for x0, _y0, x1, _y1 in boxes:
        if x1 <= x0:  # 退化的空框，不参与
            continue
        edges.append((x0, 1))
        edges.append((x1, -1))
    if not edges:
        return []
    edges.sort(key=lambda edge: (edge[0], edge[1]))
    bands: list[tuple[float, float]] = []
    depth = 0
    prev: float | None = None
    for x, delta in edges:
        if depth == 0 and prev is not None and x - prev >= min_gap:
            bands.append((prev, x))
        depth += delta
        prev = x
    return bands


def _divider(boxes: list[tuple], left: float, right: float, min_gap: float) -> tuple[float, float] | None:
    """挑一条「中间位置、最宽」的空白带当分栏线；没有则 ``None``。"""
    span = max(1.0, right - left)
    bands = _blank_bands(boxes, min_gap=min_gap)
    if not bands:
        return None
    lo = left + 0.15 * span
    hi = right - 0.15 * span
    middle = [band for band in bands if lo <= (band[0] + band[1]) / 2.0 <= hi]
    if not middle:
        return None
    return max(middle, key=lambda band: band[1] - band[0])


def _columnize(indexes: list[int], boxes: list[tuple], *, min_gap: float,
               depth: int = 0) -> list[list[int]]:
    """把一组行切成若干栏（左→右）。切不动就返回单列。"""
    if depth >= MAX_DEPTH or len(indexes) < 2 * MIN_COLUMN_LINES:
        return [indexes]
    sub = [boxes[i] for i in indexes]
    left = min(box[0] for box in sub)
    right = max(box[2] for box in sub)
    band = _divider(sub, left, right, min_gap)
    if band is None:
        return [indexes]
    cut = (band[0] + band[1]) / 2.0
    left_side = [i for i in indexes if _xc(boxes[i]) < cut]
    right_side = [i for i in indexes if _xc(boxes[i]) >= cut]
    #: 一侧不足 ``MIN_COLUMN_LINES`` ⇒ 这不是栏，是「正文 + 一个页码/图注」
    if len(left_side) < MIN_COLUMN_LINES or len(right_side) < MIN_COLUMN_LINES:
        return [indexes]
    return (_columnize(left_side, boxes, min_gap=min_gap, depth=depth + 1)
            + _columnize(right_side, boxes, min_gap=min_gap, depth=depth + 1))


def _sort_by_y(indexes: list[int], boxes: list[tuple]) -> list[int]:
    #: 按 y 中心排，同 y 时按 x —— 保证「一行被切成两段」时左右顺序不颠倒
    return sorted(indexes, key=lambda i: (_yc(boxes[i]), boxes[i][0]))


def analyze(lines: list, page_width: float | None = None) -> dict:
    """算出新的阅读顺序。

    返回 ``{"order": [索引] 或 None, "breaks": {输出位置…}, "columns": n, "anchors": n}``：

    * ``order is None`` ⇒ 保持输入顺序（单栏页的唯一出口，调用方据此零成本跳过）；
    * ``breaks`` 是**在新序列里需要硬断行**的位置（段间、栏间）；
    * ``columns`` 仅供日志/测试看，不参与判断。
    """
    count = len(lines)
    empty = {"order": None, "breaks": set(), "columns": 1, "anchors": 0}
    if count < MIN_LINES_FOR_LAYOUT:
        return empty
    try:
        boxes = [bbox_of(line) for line in lines]
    except (TypeError, ValueError, IndexError):
        #: 框的形状不认识就当没有版面信息 —— 宁可不分栏，也不能抛异常打断整本书
        return empty

    span_left = min(box[0] for box in boxes)
    span_right = max(box[2] for box in boxes)
    span = span_right - span_left
    if page_width:
        #: 页宽比行范围更可信（右侧栏可能不足行）；但不用它推翻行范围
        span = max(span, float(page_width) * 0.5)
    if span <= 0:
        return empty
    min_gap = max(COLUMN_GAP_MIN_PX, span * COLUMN_GAP_MIN_SHARE)

    #: ⚠️ 找中缝时必须**先排除整页宽的行**（标题、通栏表格、分隔线）：它们横跨整幅
    #: x 范围，只要有一行存在，「没有任何一行覆盖」的条件就永远不成立 ⇒ 双栏页里
    #: 只要带一个通栏标题，分栏就整体失效（这是实测最容易漏掉的一种版式，期刊几乎
    #: 每页都有）。宽行不用丢掉，它们在下面当分段锚点用。
    span_total = max(1.0, span_right - span_left)
    wide = {i for i, box in enumerate(boxes)
            if (box[2] - box[0]) >= WIDE_LINE_SHARE * span_total}
    narrow = [i for i in range(count) if i not in wide]
    if len(narrow) < MIN_LINES_FOR_LAYOUT:
        #: 整页都是宽行（满页表格/插图）⇒ 没有"栏"可言，按 y 读就是对的
        return empty

    bands = _blank_bands([boxes[i] for i in narrow], min_gap=min_gap)
    if not bands:
        return empty
    dividers = [(band[0] + band[1]) / 2.0 for band in bands]

    #: 跨栏行＝真正跨越某条中缝的行（正文行缩进再大也不会跨过中缝，故不会误判）
    anchors = [i for i, box in enumerate(boxes)
               if any(box[0] < cut - 1.0 and box[2] > cut + 1.0 for cut in dividers)]
    others = [i for i in range(count) if i not in set(anchors)]

    order: list[int] = []
    breaks: set[int] = set()

    def emit(sequence: list[int]) -> None:
        if not sequence:
            return
        if order:
            breaks.add(len(order))
        order.extend(sequence)

    if not anchors:
        columns = _columnize(others, boxes, min_gap=min_gap)
        if len(columns) < 2:
            #: 有空白带但切不成两栏（典型：正文右侧挂一个页码/图注）⇒ 按 y 读本来就是
            #: 对的。这里必须报「没动」，否则调用方会白白插一堆断行，而结果与不重排相同
            return empty
        for column in columns:
            emit(_sort_by_y(column, boxes))
        if len(order) != count:  # 有行既不是锚点也没进栏（不该发生）⇒ 放弃重排
            return empty
        return {"order": order, "breaks": breaks,
                "columns": len(columns), "anchors": 0}

    anchor_order = _sort_by_y(anchors, boxes)
    buckets: dict[int, list[int]] = {i: [] for i in anchor_order}
    tail: list[int] = []
    for index in others:
        target = None
        for anchor in anchor_order:
            if _yc(boxes[index]) < _yc(boxes[anchor]):
                target = anchor
                break
        (buckets[target] if target is not None else tail).append(index)

    total_columns = 0
    for anchor in anchor_order:
        bucket = buckets[anchor]
        columns = _columnize(bucket, boxes, min_gap=min_gap)
        total_columns = max(total_columns, len(columns))
        for column in columns:
            emit(_sort_by_y(column, boxes))
        emit([anchor])
    columns = _columnize(tail, boxes, min_gap=min_gap)
    total_columns = max(total_columns, len(columns))
    for column in columns:
        emit(_sort_by_y(column, boxes))

    if len(order) != count:
        return empty
    return {"order": order, "breaks": breaks,
            "columns": max(2, total_columns), "anchors": len(anchor_order)}


def order_lines(lines: list, page_width: float | None = None) -> list:
    """重排后的行列表（无版面可依据时返回**原对象**，调用方可用 ``is`` 判断是否变过）。"""
    report = analyze(lines, page_width)
    if not report["order"]:
        return lines
    return [lines[index] for index in report["order"]]
