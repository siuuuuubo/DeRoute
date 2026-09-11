"""分层评测与错误归因：把端到端 EM/F1 拆成图、节点、边、执行、合成五层。

离线运行，不调用任何模型，也不把标准答案写入推理流程。读入
- 预测：run.py 产出的 workflows.jsonl（每条记录含 nodes、status、answer 等）
- 标准：data_ori/musique_ans_v1.0_dev.jsonl（含 question_decomposition 与逐子题答案）

MuSiQue 标准拆解里，子题写成 ``X >> relation`` 或 ``#k >> relation``。这里不试图做
"语义对齐"这种脆弱匹配，而是用两路稳健信号：

1. 答案对齐（主信号）：预测节点的输出答案 vs 标准子题答案，用规范化 EM 或词项 F1 对齐。
   它直接度量"流程是否恢复了正确的中间事实"，用于计算漏拆/过拆/Node-F1。
2. 题目语义重叠（次信号）：去掉 ``#k`` 引用、轻量词干化后，比较标准子题与预测子题
   的整句词集，仅用于在"该子题漏答了"时区分"根本没拆(拆解错)"还是"拆了但答错(执行错)"。

错误归因（只对"流程完成但最终答案错"的样本）：
- decomposition：存在标准子题既未恢复答案、又没有语义对应的预测节点（漏拆/边错）。
- execution：标准子题有语义对应的预测节点，但该节点答案未命中标准子题答案。
- synthesis：所有标准子题答案都被恢复、中间步骤都对，但最终答案仍错（最后一跳/合成错）。
- no_answer：流程未完成（abort / 预算 / 修订耗尽 / 证据不足），无法定位到执行层。

这些是规则化、可复核的启发式归因，不是语义完备的判定；阈值与口径固定在下方常量里。
"""
import argparse
import re
from collections import Counter
from pathlib import Path

from evaluate import normalize_answer, read_records

ROOT = Path(__file__).resolve().parent

# 对齐阈值：答案相似度 / 题目语义重叠，达到阈值才视为"对应同一个子题"。
ANSWER_ALIGN_THRESHOLD = 0.5
QUESTION_OVERLAP_THRESHOLD = 0.4

_STOP = set("""a an the of in on at to for and or by with from as is are was were be been
it its this that these those he she they who whom whose what which when where how why do
does did done have has had can could may might must shall should will would into onto upon
not no nor so than then there here""".split())


def content_tokens(text):
    """小写、去标点、去停用词，保留单字与数字；返回词列表（有顺序、可重复用于词袋）。"""
    if not isinstance(text, str):
        return []
    raw = re.findall(r"[a-z0-9]+", text.casefold())
    return [w for w in raw if w not in _STOP]


def bag(text):
    return Counter(content_tokens(text))


def token_f1(a, b):
    p, g = Counter(content_tokens(a)), Counter(content_tokens(b))
    if not p or not g:
        return float(p == g)
    overlap = sum((p & g).values())
    return 2 * overlap / (sum(p.values()) + sum(g.values()))


def answer_similarity(pred_answer, gold_answer):
    """规范化 EM 给 1.0，否则退化为词项 F1。"""
    if not isinstance(pred_answer, str) or not isinstance(gold_answer, str):
        return 0.0
    if normalize_answer(pred_answer) == normalize_answer(gold_answer):
        return 1.0
    return token_f1(pred_answer, gold_answer)


def strip_refs(text):
    """去掉 '#k' 引用，避免把编号数字当成语义词参与匹配。"""
    return re.sub(r"#\d+", " ", text)


def stem_word(word):
    """轻量词干化：把 headquartered/headquarters 归并到 headquarter，但不动 city 这类短词。"""
    word = re.sub(r"ies$", "y", word)
    for suffix in ("ing", "edly", "es", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def question_tokens(text):
    """整句语义词：去 #k 引用、小写、去停用词、词干化。"""
    if not isinstance(text, str):
        return []
    raw = re.findall(r"[a-z0-9]+", strip_refs(text).casefold())
    return [stem_word(w) for w in raw if w not in _STOP]


def question_overlap(gold_question, pred_question):
    """标准子题与预测子题的整句词集 Jaccard 重叠。"""
    g = set(question_tokens(gold_question))
    p = set(question_tokens(pred_question))
    if not g or not p:
        return 0.0
    return len(g & p) / len(g | p)


def gold_references(gold_question):
    """从标准子题 '#k' 中提取所依赖的标准子题序号（1-based → 0-based）。"""
    return {int(m) - 1 for m in re.findall(r"#([1-9][0-9]*)", gold_question)}


def align_by_answer(pred_nodes, gold_nodes):
    """贪心把预测节点对齐到标准子题（按答案相似度）。返回 (align, missed, extra)。

    align[gold_idx] = pred_id 或 None；missed = 未被覆盖的标准子题序号；
    extra = 未对齐到任何标准子题的预测节点 id 列表。
    """
    align = [None] * len(gold_nodes)
    used = set()
    order = sorted(range(len(gold_nodes)), key=lambda i: gold_nodes[i].get("order", i))
    for gold_idx in order:
        gold_answer = gold_nodes[gold_idx]["answer"]
        best, best_score = None, ANSWER_ALIGN_THRESHOLD
        for pred in pred_nodes:
            if pred["id"] in used or not isinstance(pred.get("answer"), str):
                continue
            score = answer_similarity(pred["answer"], gold_answer)
            if score > best_score or (score >= ANSWER_ALIGN_THRESHOLD and score == best_score):
                best, best_score = pred["id"], score
        if best is not None:
            align[gold_idx] = best
            used.add(best)
    missed = [i for i in range(len(gold_nodes)) if align[i] is None]
    extra = [p["id"] for p in pred_nodes if p["id"] not in used]
    return align, missed, extra


def node_edges(align, gold_nodes, pred_nodes):
    """把两边依赖边归一化到预测节点 id 空间后求边 P/R/F1。

    标准边来自 '#k' 引用；预测边来自 depends_on。仅统计两端都能对齐的边。
    """
    pred_by_id = {p["id"]: p for p in pred_nodes}
    gold_edges = set()
    for i, g in enumerate(gold_nodes):
        for j in gold_references(g["question"]):
            if j < len(gold_nodes):
                gold_edges.add((j, i))
    mapped = {(align[j], align[i]) for (j, i) in gold_edges
              if align[j] is not None and align[i] is not None}
    pred_edges = {(d, p["id"]) for p in pred_nodes for d in p.get("depends_on", [])
                  if d in pred_by_id}
    both = mapped & pred_edges
    recall = len(both) / len(gold_edges) if gold_edges else 1.0
    precision = len(both) / len(pred_edges) if pred_edges else 1.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"gold_edges": len(gold_edges), "pred_edges": len(pred_edges),
            "matched_edges": len(both), "recall": recall, "precision": precision, "f1": f1}


def gold_nodes_from_record(record):
    """从标准记录的 question_decomposition 提取子题节点。"""
    dec = record.get("question_decomposition") or []
    return [{"order": i, "question": item.get("question", ""),
             "answer": item.get("answer", "")} for i, item in enumerate(dec)]


def pred_nodes_from_record(record):
    """从预测记录提取节点：只取最终状态（修订前的旧态在 revisions/invalidations 里）。"""
    out = []
    for n in record.get("nodes", []):
        ans = ""
        if n.get("status") == "succeeded" and isinstance(n.get("output"), dict):
            ans = n["output"].get("answer", "")
        out.append({"id": n.get("id"), "question": n.get("question", ""),
                    "depends_on": list(n.get("depends_on", [])),
                    "status": n.get("status"), "answer": ans,
                    "operation": n.get("operation", "")})
    return out


def valid_graph(nodes):
    """用现有 levels() 校验预测图无环且引用存在；不抛异常即合法。"""
    try:
        from graph import levels
        levels([{"id": n["id"], "depends_on": n["depends_on"]} for n in nodes])
        return True
    except Exception:
        return False


def attribute(record, gold_nodes, pred_nodes, align, missed, extra, final_correct):
    """对单条样本做错误归因，返回 (attribution, detail)。"""
    if record.get("status") != "succeeded":
        reason = (record.get("error") or "未完成")
        if "修订次数耗尽" in reason:
            kind = "no_answer:revision_exhausted"
        elif "预算" in reason or "调用上限" in reason:
            kind = "no_answer:budget"
        elif "无法恢复" in reason or "abort" in reason:
            kind = "no_answer:abort"
        else:
            kind = "no_answer:other"
        return kind, reason
    if final_correct:
        return "correct", "最终答案命中标准答案"

    pred_by_id = {p["id"]: p for p in pred_nodes}
    # 漏答的标准子题：看是否有语义对应的预测节点（拆了但答错 → 执行错；没拆 → 拆解错）。
    missing = []
    for i in missed:
        overlap = max((question_overlap(gold_nodes[i]["question"], p["question"])
                       for p in pred_nodes), default=0.0)
        missing.append((i, overlap))
    decomposition = [i for i, ov in missing if ov < QUESTION_OVERLAP_THRESHOLD]
    execution = [i for i, ov in missing if ov >= QUESTION_OVERLAP_THRESHOLD]

    if decomposition:
        return "decomposition", f"漏拆/边错，未覆盖标准子题 {decomposition}"
    if execution:
        return "execution", f"已拆出但答案未命中标准子题 {execution}"
    # 中间事实都恢复，但最终答案仍错：定位到最后一跳/合成。
    return "synthesis", f"中间 {len(align) - len(missed)}/{len(gold_nodes)} 个标准子题已恢复，最终答案错"


def evaluate_layers(predictions_path, gold_path):
    predictions = {}
    for record in read_records(predictions_path):
        key = record.get("task_id")
        if not isinstance(key, str) or not isinstance(record.get("status"), str):
            raise ValueError("预测日志缺少task_id/status")
        predictions[key] = record
    if not predictions:
        raise ValueError("预测日志为空")

    gold = {}
    for record in read_records(gold_path):
        key = record.get("id")
        if key not in predictions:
            continue
        if key in gold:
            raise ValueError("标准数据中存在重复id")
        if not isinstance(record.get("answer"), str) or not record["answer"].strip():
            raise ValueError(f"标准数据 {key} 缺少非空答案")
        gold[key] = record
    if set(predictions) != set(gold):
        raise ValueError("预测与标准数据的样本 id 不一致；请检查数据版本")

    rows = []
    for key, record in predictions.items():
        g = gold[key]
        gold_nodes = gold_nodes_from_record(g)
        pred_nodes = pred_nodes_from_record(record)
        final_correct = int(normalize_answer(record.get("answer", "")) == normalize_answer(g["answer"]))
        align, missed, extra = align_by_answer(pred_nodes, gold_nodes)
        edges = node_edges(align, gold_nodes, pred_nodes)
        attribution, detail = attribute(record, gold_nodes, pred_nodes,
                                        align, missed, extra, final_correct)
        node_recall = (len(gold_nodes) - len(missed)) / len(gold_nodes) if gold_nodes else 1.0
        node_precision = (len(pred_nodes) - len(extra)) / len(pred_nodes) if pred_nodes else 1.0
        rows.append({
            "task_id": key, "status": record["status"],
            "final_correct": bool(final_correct), "attribution": attribution, "detail": detail,
            "pred_nodes": len(pred_nodes), "gold_nodes": len(gold_nodes),
            "node_recall": round(node_recall, 4), "node_precision": round(node_precision, 4),
            "missed_gold": missed, "extra_pred": extra,
            "edges": {k: round(v, 4) if isinstance(v, float) else v for k, v in edges.items()},
            "graph_valid": valid_graph(pred_nodes),
        })

    n = len(rows)
    succeeded = [r for r in rows if r["status"] == "succeeded"]
    correct = [r for r in rows if r["final_correct"]]

    def mean(seq):
        return round(sum(seq) / len(seq), 4) if seq else 0.0

    def ratio(seq):
        return round(sum(seq) / n, 4) if n else 0.0

    attribution_counts = dict(Counter(r["attribution"] for r in rows))

    return {
        "evaluated": n,
        "completed": len(succeeded),
        "final_exact_match": ratio([r["final_correct"] for r in rows]),
        "graph_valid_rate": ratio([r["graph_valid"] for r in rows]),
        "node_recall": mean([r["node_recall"] for r in succeeded]),
        "node_precision": mean([r["node_precision"] for r in succeeded]),
        "edge_recall": mean([r["edges"]["recall"] for r in succeeded]),
        "edge_precision": mean([r["edges"]["precision"] for r in succeeded]),
        "mean_gold_nodes": round(sum(r["gold_nodes"] for r in rows) / n, 2) if n else 0,
        "mean_pred_nodes": round(sum(r["pred_nodes"] for r in rows) / n, 2) if n else 0,
        "under_decomposition_rate": mean([1 - r["node_recall"] for r in succeeded]),
        "over_decomposition_rate": mean([1 - r["node_precision"] for r in succeeded]),
        "attribution": attribution_counts,
        "attribution_share": {k: round(v / n, 4) for k, v in attribution_counts.items()},
        "scope": "分层指标在流程完成样本上求均值；归因对全部样本计数；阈值见模块常量",
        "results": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT) or output in (args.predictions.resolve(), args.gold.resolve()):
            raise ValueError("评测输出须在DeRoute内，且不能覆盖预测或标准数据")
        result = evaluate_layers(args.predictions, args.gold)
        from decompose import write_json
        write_json(output, result)
        print(f"分层评测 {result['evaluated']} 条：EM={result['final_exact_match']:.2%}，"
              f"图合法率={result['graph_valid_rate']:.2%}，"
              f"Node-R={result['node_recall']:.2%}/P={result['node_precision']:.2%}，"
              f"Edge-R={result['edge_recall']:.2%}/P={result['edge_precision']:.2%}")
        print(f"归因分布：{result['attribution_share']}")
        print(output)
        return 0
    except (ValueError, OSError) as exc:
        print("评测失败：" + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
