"""eval_layers 的离线测试：答案对齐、节点/边指标、错误归因。不调用模型。"""
import json
import tempfile
import unittest
from pathlib import Path

from eval_layers import (
    align_by_answer, answer_similarity, attribute, evaluate_layers,
    gold_references, gold_nodes_from_record, node_edges,
    pred_nodes_from_record, question_overlap, token_f1, valid_graph)

ROOT = Path(__file__).resolve().parents[1]


def gold_rec(task_id="s", answer="Cologne", decom=None):
    decom = decom or [
        {"id": 1, "question": "Ulrich Walter >> employer", "answer": "German Aerospace Center",
         "paragraph_support_idx": 0},
        {"id": 2, "question": "#1 >> headquarters located in the city of", "answer": "Cologne",
         "paragraph_support_idx": 1}]
    return {"id": task_id, "question": "q", "answer": answer, "answer_aliases": [],
            "question_decomposition": decom}


def pred_node(nid, question, answer, deps=None, status="succeeded"):
    return {"id": nid, "question": question, "depends_on": deps or [], "operation": "lookup",
            "status": status, "output": {"answer": answer}}


def pred_rec(task_id="s", status="succeeded", answer="Cologne", nodes=None):
    return {"task_id": task_id, "status": status, "answer": answer, "nodes": nodes or []}


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                    encoding="utf-8")


class EvalLayersTests(unittest.TestCase):
    def test_answer_similarity_em_and_partial(self):
        self.assertEqual(answer_similarity("German Aerospace Center", "german aerospace center"), 1.0)
        self.assertGreater(answer_similarity("the German Aerospace Center", "German Aerospace"), 0.5)
        self.assertEqual(token_f1("a", "a"), 1.0)

    def test_gold_references_are_zero_based(self):
        self.assertEqual(gold_references("#1 >> spouse"), {0})
        self.assertEqual(gold_references("Combine #1 and #2"), {0, 1})
        self.assertEqual(gold_references("Green >> performer"), set())

    def test_question_overlap_stems_and_strips_refs(self):
        self.assertGreater(question_overlap("Ulrich Walter >> employer", "Who employed Ulrich Walter?"), 0.4)
        self.assertGreater(question_overlap("#1 >> headquarters located in the city of",
                                             "In which city is #1 headquartered?"), 0.4)
        self.assertEqual(question_overlap("A >> relation", "Something entirely different"), 0.0)

    def test_answer_alignment_recovers_both_nodes(self):
        gold = gold_nodes_from_record(gold_rec())
        preds = pred_nodes_from_record(pred_rec(nodes=[
            pred_node("n1", "Who employed Ulrich Walter?", "German Aerospace Center"),
            pred_node("n2", "Where is #1 headquartered?", "Cologne", ["n1"])]))
        align, missed, extra = align_by_answer(preds, gold)
        self.assertEqual(set(align), {"n1", "n2"})
        self.assertEqual(missed, [])
        self.assertEqual(extra, [])

    def test_edge_f1_full_match(self):
        gold = gold_nodes_from_record(gold_rec())
        preds = pred_nodes_from_record(pred_rec(nodes=[
            pred_node("n1", "employer", "German Aerospace Center"),
            pred_node("n2", "HQ city", "Cologne", ["n1"])]))
        align, _, _ = align_by_answer(preds, gold)
        edges = node_edges(align, gold, preds)
        self.assertEqual(edges["recall"], 1.0)
        self.assertEqual(edges["precision"], 1.0)

    def test_graph_validity(self):
        self.assertTrue(valid_graph([pred_node("n1", "a", "x"),
                                     pred_node("n2", "b", "y", ["n1"])]))
        self.assertFalse(valid_graph([pred_node("n1", "a", "x", ["n2"]),
                                      pred_node("n2", "b", "y", ["n1"])]))

    def test_attribution_correct_and_synthesis(self):
        gold = gold_nodes_from_record(gold_rec())
        ok_preds = pred_nodes_from_record(pred_rec(nodes=[
            pred_node("n1", "employer", "German Aerospace Center"),
            pred_node("n2", "HQ city", "Cologne", ["n1"])]))
        align, missed, extra = align_by_answer(ok_preds, gold)
        self.assertEqual(attribute(pred_rec(), gold, ok_preds, align, missed, extra, True)[0], "correct")
        # 中间两个子题答案都恢复，但最终答案错 → 合成错
        wrong_final = pred_rec(answer="Berlin")
        self.assertEqual(attribute(wrong_final, gold, ok_preds, align, missed, extra, False)[0], "synthesis")

    def test_attribution_decomposition_when_subtask_never_created(self):
        gold = gold_nodes_from_record(gold_rec())
        # 只答了第一跳，第二跳既没恢复答案、也没有语义对应的节点 → 漏拆
        preds = pred_nodes_from_record(pred_rec(nodes=[
            pred_node("n1", "Who employed Ulrich Walter?", "German Aerospace Center")]))
        align, missed, extra = align_by_answer(preds, gold)
        self.assertEqual(missed, [1])
        kind, _ = attribute(pred_rec(answer=""), gold, preds, align, missed, extra, False)
        self.assertEqual(kind, "decomposition")

    def test_attribution_execution_when_subtask_exists_but_wrong(self):
        gold = gold_nodes_from_record(gold_rec())
        # 两个子题都拆了，但第二个答案错（语义对应，答案未命中）→ 执行错
        preds = pred_nodes_from_record(pred_rec(nodes=[
            pred_node("n1", "Who employed Ulrich Walter?", "German Aerospace Center"),
            pred_node("n2", "In which city is #1 headquartered?", "Bonn", ["n1"])]))
        align, missed, extra = align_by_answer(preds, gold)
        self.assertIn(1, missed)
        kind, _ = attribute(pred_rec(answer="Bonn"), gold, preds, align, missed, extra, False)
        self.assertEqual(kind, "execution")

    def test_attribution_no_answer_for_failed_status(self):
        gold = gold_nodes_from_record(gold_rec())
        preds = pred_nodes_from_record(pred_rec(status="failed", answer=""))
        rec = pred_rec(status="failed", answer="")
        rec["error"] = "拆解器无法恢复：缺少直接关系句"
        align, missed, extra = align_by_answer(preds, gold)
        kind, _ = attribute(rec, gold, preds, align, missed, extra, False)
        self.assertEqual(kind, "no_answer:abort")

    def test_end_to_end_report_shapes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p, g = Path(tmp) / "pred.jsonl", Path(tmp) / "gold.jsonl"
            write_jsonl(p, [pred_rec(nodes=[
                pred_node("n1", "Who employed Ulrich Walter?", "German Aerospace Center"),
                pred_node("n2", "Where is #1 headquartered?", "Cologne", ["n1"])])])
            write_jsonl(g, [gold_rec()])
            result = evaluate_layers(p, g)
            self.assertEqual(result["evaluated"], 1)
            self.assertEqual(result["final_exact_match"], 1.0)
            self.assertEqual(result["attribution"], {"correct": 1})
            self.assertEqual(result["node_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
