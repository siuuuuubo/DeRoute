"""离线测试：数据隔离、增量协议、并行分支、升级、预算和续跑。"""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from dataset import read_dataset
from compare_runs import workload
from decompose import load_prompt, run_task, summarize_calls, write_json
from executor import FINISH_PROMPT, REVIEW_PROMPT, check_finish, execute_node, rank_paragraphs, readable_paragraphs, route_node, validate_answer
from graph import canonicalize_node, levels, ready_nodes, references, resolve_inputs, validate_finish, validate_node, validate_revision, validate_temporal_scope
from evaluate import answer_scores, evaluate
from model import APIModel, CallBudget, LocalModel, ModelError, load_config, parse_json
from run import checkpoint_records, completed_records, run_selected, task_checkpoint
from decompose import fingerprint, workflow_signature, clean_paragraph_hint

ROOT = Path(__file__).resolve().parents[1]


def task():
    return {"id": "hidden-hop-id", "question": "Compare the homes of Alpha and Beta.",
            "paragraphs": [
                {"idx": 7, "title": "Alpha", "paragraph_text": "Alpha was born in AlphaTown."},
                {"idx": 9, "title": "Beta", "paragraph_text": "Beta was born in BetaTown."}]}


def node(index=1, operation="lookup", deps=None):
    questions = {1: "Where was Alpha born?", 2: "Explain where Beta was born?",
                 3: "Combine the locations #1 and #2."}
    return {"id": f"n{index}", "question": questions.get(index, "Who is #1?"),
            "operation": operation, "depends_on": deps or [], "expected_output": "short answer",
            "reason": "resolve one necessary part"}


def answer(text="AlphaTown", idx=7, quote="Alpha was born in AlphaTown.", used=None):
    return {"status": "ok", "answer": text, "evidence": [{"paragraph_idx": idx, "quote": quote}],
            "used_inputs": used or [], "reason": "supported by the supplied material",
            "support_type": "direct", "assumptions": []}


def response(value):
    return {"raw_response": json.dumps(value), "model": "test", "request_attempts": 1,
            "seconds": 0.01, "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


def config():
    return {"large": {"model": "fake-large"}, "small": {"model": "fake-small"},
            "runtime": {"max_workers": 2, "max_nodes": 8, "max_planner_calls": 16,
                        "max_model_calls": 30, "max_task_seconds": 15, "max_revisions": 2},
            "routing": {"small_max_context_chars": 6000, "small_top_k": 3,
                        "small_max_dependencies": 1, "fallback_to_large": True}}


class FakeClient:
    def __init__(self, role, gates=None, insufficient=False, quota=False):
        self.config = {"model": "fake-" + role}
        self.role, self.gates = role, gates
        self.insufficient, self.quota = insufficient, quota
        self.requests = []

    def complete(self, system, user, live=False):
        assert live
        data = json.loads(user)
        self.requests.append((system, data))
        if system == FINISH_PROMPT:
            return response({"valid": True, "reason": "the final node answers the original question"})
        if system == "PLAN":
            nodes = data["nodes"]
            if self.quota and len(nodes) >= 1:
                raise ModelError("quota", {"http_status": 402})
            if not nodes:
                action = {"action": "add", "node": node(), "continue_planning": True}
            elif len(nodes) == 1:
                action = {"action": "add", "node": node(2, "reason")}
            elif len(nodes) == 2 and all(n["status"] == "succeeded" for n in nodes):
                action = {"action": "add", "node": node(3, "summarize", ["n1", "n2"])}
            elif len(nodes) == 3 and all(n["status"] == "succeeded" for n in nodes):
                action = {"action": "finish", "final_node": "n3"}
            else:
                action = {"action": "wait"}
            return response(action)
        if self.insufficient:
            return response({"status": "insufficient", "answer": "", "evidence": [],
                             "used_inputs": [], "reason": "not found", "support_type": "insufficient", "assumptions": []})
        if "Combine" in data["question"]:
            return response({"status": "ok", "answer": "AlphaTown and BetaTown", "evidence": [],
                             "used_inputs": ["n1", "n2"], "reason": "combined both inputs",
                             "support_type": "direct", "assumptions": []})
        if self.gates:
            mine = 0 if self.role == "small" else 1
            self.gates[mine].set()
            if not self.gates[1 - mine].wait(4):
                raise ModelError("independent small/large executions did not overlap")
        if "Beta" in data["question"]:
            return response(answer("BetaTown", 9, "Beta was born in BetaTown."))
        return response(answer())


class RecoveryClient(FakeClient):
    def __init__(self, role, always_fail=False):
        super().__init__(role)
        self.always_fail = always_fail

    def complete(self, system, user, live=False):
        data = json.loads(user)
        self.requests.append((system, data))
        if system == FINISH_PROMPT:
            return response({"valid": True, "reason": "the recovered node answers the question"})
        if system == "PLAN":
            nodes = data["nodes"]
            if not nodes:
                return response({"action": "add", "node": node()})
            if nodes[0]["status"] == "needs_revision":
                return response({"action": "revise", "node": node(operation="reason")})
            if nodes[0]["status"] == "succeeded":
                return response({"action": "finish", "final_node": "n1"})
            return response({"action": "wait"})
        if data["operation"] == "lookup" or self.always_fail:
            return response(dict(answer(), status="insufficient", support_type="insufficient",
                                 reason="relation is implicit; candidate needs verification"))
        return response(answer())


class CoreTests(unittest.TestCase):
    def run_flow(self, clients=None, cfg=None, previous=None):
        clients = clients or {"small": FakeClient("small"), "large": FakeClient("large")}
        saves = []
        with redirect_stdout(io.StringIO()):
            result = run_task(task(), clients, cfg or config(), "PLAN",
                              lambda r: saves.append(copy.deepcopy(r)), previous)
        return result, clients, saves

    def test_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        for text in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}']:
            with self.assertRaises(ValueError):
                parse_json(text)

    def test_dataset_projects_away_all_gold_fields(self):
        sample = task()
        sample.update(answer="SECRET_GOLD", question_decomposition=[{"answer": "SECRET_GOLD"}],
                      answerable=True, answer_aliases=["SECRET_GOLD"])
        sample["paragraphs"][0]["is_supporting"] = True
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = Path(tmp) / "dev.jsonl"
            p.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            got = list(read_dataset(p))
        self.assertEqual(got, [task()])
        self.assertNotIn("SECRET_GOLD", json.dumps(got))

    def test_dataset_rejects_duplicate_sample_and_paragraph_ids(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = Path(tmp) / "dev.jsonl"
            p.write_text((json.dumps(task()) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复"):
                list(read_dataset(p))
            sample = task()
            sample["paragraphs"][1]["idx"] = 7
            p.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复"):
                list(read_dataset(p))

    def test_dataset_reports_line_number(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = Path(tmp) / "dev.jsonl"
            p.write_text(json.dumps(task()) + "\n{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "第2行"):
                list(read_dataset(p))

    def test_graph_rejects_forward_self_and_undeclared_dependencies(self):
        bad = node(1)
        bad.update(question="Who is #1?", depends_on=["n1"])
        with self.assertRaises(ValueError):
            validate_node(bad, [])
        bad = node(2)
        bad["question"] = "Where is #1?"
        with self.assertRaises(ValueError):
            validate_node(bad, [node()])
        bad["depends_on"] = ["n9"]
        with self.assertRaises(ValueError):
            validate_node(bad, [node()])

    def test_graph_allows_pending_dependency_and_parallel_roots(self):
        a, b = node(), node(2, "reason")
        a["status"], b["status"] = "pending", "pending"
        c = validate_node(node(3, "summarize", ["n1", "n2"]), [a, b])
        c["status"] = "pending"
        self.assertEqual(levels([a, b, c]), [["n1", "n2"], ["n3"]])
        self.assertEqual([n["id"] for n in ready_nodes([a, b, c])], ["n1", "n2"])

    def test_graph_finish_requires_all_branches_and_success(self):
        nodes = [dict(node(), status="succeeded", output=answer()),
                 dict(node(2), status="succeeded", output=answer())]
        with self.assertRaisesRegex(ValueError, "分支"):
            validate_finish("n2", nodes)
        nodes[0]["status"] = "running"
        with self.assertRaisesRegex(ValueError, "成功"):
            validate_finish("n2", nodes)

    def test_replaced_failed_branch_is_kept_out_of_a_complete_final_chain(self):
        nodes = [dict(node(), status="succeeded", output=answer()),
                 dict(node(2), status="needs_revision", error="an obsolete failed candidate"),
                 dict(node(3), question="What place contains #1?", depends_on=["n1"],
                      status="succeeded", output=answer(used=["n1"]))]
        self.assertEqual(validate_finish("n3", nodes), "AlphaTown")
        client = FakeClient("large")
        check_finish(task(), nodes, "n3", client, CallBudget(1))
        self.assertEqual([n["id"] for n in client.requests[-1][1]["nodes"]], ["n1", "n3"])
        nodes[2]["depends_on"] = ["n1", "n2"]
        with self.assertRaisesRegex(ValueError, "最终依赖链"):
            validate_finish("n3", nodes)

    def test_reference_replacement_does_not_confuse_1_and_10(self):
        n = {"question": "Compare #1 with #10.", "depends_on": ["n1", "n10"]}
        upstream = [{"id": k, "status": "succeeded", "output": answer(text)}
                    for k, text in [("n1", "A"), ("n10", "B")]]
        question, values = resolve_inputs(n, upstream)
        self.assertEqual(question, "Compare A with B.")
        self.assertEqual(set(values), {"n1", "n10"})
        self.assertEqual(references(n["question"]), {"n1", "n10"})

    def test_declared_literal_upstream_answer_is_canonicalized_without_guessing(self):
        previous = [dict(node(), status="succeeded", output=answer("North University"))]
        step = dict(node(2), question="Where is North University located?", depends_on=["n1"])
        fixed = canonicalize_node(step, previous)
        self.assertEqual(fixed["question"], "Where is #1 located?")
        self.assertEqual(validate_node(fixed, previous), fixed)
        self.assertIn("North University", step["question"])
        with self.assertRaises(ValueError):
            validate_node(canonicalize_node(dict(step, question="Where is another university?"), previous), previous)

    def test_final_check_can_add_one_missing_atomic_step(self):
        class MissingStepClient(RecoveryClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                if system == FINISH_PROMPT and len(data["nodes"]) == 1:
                    step = dict(node(2), question="What place contains #1?", depends_on=["n1"])
                    return response({"valid": False, "reason": "one relationship still missing", "next_node": step})
                if system == "PLAN" and len(data["nodes"]) == 2:
                    return response({"action": "finish", "final_node": "n2"} if data["nodes"][1]["status"] == "succeeded"
                                    else {"action": "wait"})
                if system not in ("PLAN", FINISH_PROMPT) and data.get("upstream"):
                    return response(answer(used=["n1"]))
                return super().complete(system, user, live)
        result, _, _ = self.run_flow({r: MissingStepClient(r) for r in ("small", "large")})
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        self.assertTrue(any(e.get("source") == "final_check" for e in result["events"]))
        self.assertEqual(result["nodes"][1]["depends_on"], ["n1"])

    def test_route_rules_are_deterministic(self):
        cfg = config()["routing"]
        route, _ = route_node(node(), node()["question"], {}, task()["paragraphs"], cfg)
        self.assertEqual(route["role"], "small")
        route, _ = route_node(node(2, "compare"), "Beta", {}, task()["paragraphs"], cfg)
        self.assertEqual(route["role"], "large")
        cfg["small_max_context_chars"] = 1
        route, _ = route_node(node(), "Alpha", {}, task()["paragraphs"], cfg)
        self.assertEqual(route["role"], "large")

    def test_temporal_ambiguity_routes_to_large_and_invented_year_is_rejected(self):
        route, _ = route_node(node(), "Who won in 1999–2000?", {}, task()["paragraphs"], config()["routing"])
        self.assertEqual(route["role"], "large")
        ambiguous = [{"idx": 1, "title": "Alpha", "paragraph_text": "Alpha joined X in 2001, Y in 2004 and Z in 2008."}]
        route, _ = route_node(node(), "Which organization employed Alpha?", {}, ambiguous, config()["routing"])
        self.assertEqual(route["role"], "large")
        with self.assertRaisesRegex(ValueError, "年份"):
            validate_temporal_scope(dict(node(), question="Where was Alpha in 2004?"), task(), [])
        validate_temporal_scope(dict(node(), question="Who won in the 1999–2000 season?"),
                                dict(task(), question="Who won in 1999–00?"), [])

    def test_owner_is_not_mistaken_for_owners_administrative_location(self):
        sample = dict(task(), question="What administrative territorial entity is the owner of Cedar Hall located?")
        owner = dict(node(), question="Who is the owner of Cedar Hall?", status="succeeded", output=answer("Alpha"))
        class NoModel:
            config = {"model": "not-used"}
            def complete(self, *args, **kwargs):
                raise AssertionError("the missing relationship is checked by code")
        result = check_finish(sample, [owner], "n1", NoModel(), CallBudget(1))
        self.assertFalse(result["valid"])
        self.assertEqual(result["next_node"]["depends_on"], ["n1"])
        self.assertIn("#1 located", result["next_node"]["question"])

    def test_retrieval_keeps_full_title_entity_over_partial_match(self):
        paragraphs = [
            {"idx": 0, "title": "Little Cedar", "paragraph_text": "An artist performed Little Cedar."},
            {"idx": 1, "title": "Cedar (album)", "paragraph_text": "Cedar is an album by Ada."}]
        self.assertEqual(rank_paragraphs("Which artist performed Cedar?", paragraphs)[0][1]["idx"], 1)

    def test_retrieval_matches_accents_without_modifying_source(self):
        paragraphs = [{"idx": 1, "title": "São Paulo", "paragraph_text": "A city in Brazil."},
                      {"idx": 0, "title": "Other", "paragraph_text": "A city in Egypt."}]
        before = copy.deepcopy(paragraphs)
        self.assertEqual(rank_paragraphs("Where is Sao Paulo?", paragraphs)[0][1]["idx"], 1)
        self.assertEqual(paragraphs, before)

    def test_list_answers_need_evidence_for_every_item_and_full_words(self):
        value = answer("AlphaTown and BetaTown")
        value["evidence"].append({"paragraph_idx": 9, "quote": "Beta was born in BetaTown."})
        self.assertEqual(validate_answer(value, "lookup", task()["paragraphs"], {}), value)
        with self.assertRaises(ValueError):
            validate_answer(dict(value, answer="AlphaTown and MissingTown"), "lookup", task()["paragraphs"], {})
        with self.assertRaises(ValueError):
            validate_answer(answer("Town"), "lookup", task()["paragraphs"], {})

    def test_flattened_table_rows_are_readable_without_changing_evidence(self):
        paragraphs = [{"idx": 1, "title": "Matches", "paragraph_text":
                       "Date Home team Score 1 May 2000 Alpha 2 -- 1 2 June 2001 Beta 0 -- 1"}]
        before = copy.deepcopy(paragraphs)
        readable = readable_paragraphs(paragraphs)
        self.assertIn("\n2 June 2001", readable[0]["paragraph_text"])
        validate_answer(answer("Alpha", 1, readable[0]["paragraph_text"]), "lookup", paragraphs, {})
        self.assertEqual(paragraphs, before)

    def test_evidence_rejects_fabricated_quote_and_missing_input(self):
        with self.assertRaises(ValueError):
            validate_answer(answer(quote="fabricated evidence"), "lookup", task()["paragraphs"], {})
        with self.assertRaises(ValueError):
            validate_answer(answer(), "lookup", task()["paragraphs"], {"n1": answer()})
        with self.assertRaises(ValueError):
            validate_answer(answer("unsubstantiated answer"), "lookup", task()["paragraphs"], {})

    def test_small_insufficient_escalates_once(self):
        clients = {"small": FakeClient("small", insufficient=True), "large": FakeClient("large")}
        budget = CallBudget(3)
        got = execute_node(node(), node()["question"], {}, task(), clients, budget, config()["routing"])
        self.assertEqual(got["status"], "succeeded")
        self.assertEqual(got["executed_by"], "large")
        self.assertEqual([a["status"] for a in got["attempts"]], ["rejected", "accepted"])
        self.assertEqual(len(budget.snapshot()), 2)

    def test_fallback_can_be_disabled(self):
        clients = {"small": FakeClient("small", insufficient=True), "large": FakeClient("large")}
        cfg = config()["routing"]; cfg["fallback_to_large"] = False
        got = execute_node(node(), node()["question"], {}, task(), clients, CallBudget(3), cfg)
        self.assertEqual(got["status"], "failed")
        self.assertEqual(len(clients["large"].requests), 0)

    def test_bad_quote_candidate_reaches_large_review(self):
        clients = {"small": FakeClient("small"), "large": FakeClient("large")}
        clients["small"].complete = lambda *a, **k: response(answer(quote="Alpha was born"))
        result = execute_node(node(), node()["question"], {}, task(), clients, CallBudget(4), config()["routing"])
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(result["reviewed"])
        system, payload = clients["large"].requests[0]
        self.assertEqual(system, REVIEW_PROMPT)
        self.assertEqual(payload["unverified_candidates"][0]["answer"], "AlphaTown")
        self.assertIn("答案不在", payload["validation_feedback"])
        self.assertEqual(result["attempts"][0]["status"], "rejected")

    def test_small_inference_requires_large_review(self):
        clients = {"small": FakeClient("small"), "large": FakeClient("large")}
        inferred = dict(answer(), support_type="inferred", assumptions=["A stated biographical alias links the entity."])
        clients["small"].complete = lambda *a, **k: response(inferred)
        result = execute_node(node(), node()["question"], {}, task(), clients, CallBudget(4), config()["routing"])
        self.assertEqual(result["executed_by"], "large")
        self.assertEqual(result["attempts"][0]["failure_kind"], "needs_review")
        self.assertEqual(clients["large"].requests[0][0], REVIEW_PROMPT)

    def test_large_inference_is_not_accepted_if_review_rejects(self):
        client = FakeClient("large")
        inferred = dict(answer(), support_type="inferred", assumptions=["unverified identity bridge"])
        def complete(system, user, live=False):
            return response(dict(inferred, status="insufficient", support_type="insufficient")
                            if system == REVIEW_PROMPT else inferred)
        client.complete = complete
        budget = CallBudget(4)
        result = execute_node(node(operation="reason"), "Alpha", {}, task(),
                              {"large": client}, budget, config()["routing"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(budget.snapshot()), 2)
        self.assertNotIn("output", result)

    def test_inference_still_requires_real_quotes_and_explicit_assumptions(self):
        for value in [dict(answer(), support_type="inferred"),
                      dict(answer(quote="fabricated"), support_type="inferred", assumptions=["bridge"])]:
            with self.assertRaises(ValueError):
                validate_answer(value, "reason", task()["paragraphs"], {})

    def test_failed_node_is_revised_with_history_and_failure_feedback(self):
        clients = {r: RecoveryClient(r) for r in ("small", "large")}
        result, clients, _ = self.run_flow(clients)
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        fixed = result["nodes"][0]
        self.assertEqual(fixed["revision_count"], 1)
        self.assertEqual(fixed["revisions"][0]["status"], "needs_revision")
        self.assertEqual(len(fixed["revisions"][0]["attempts"]), 2)
        failure_plans = [p for s, p in clients["large"].requests if s == "PLAN" and p["nodes"]
                         and p["nodes"][0]["status"] == "needs_revision"]
        self.assertTrue(failure_plans)
        self.assertNotIn("output", failure_plans[0]["nodes"][0])
        self.assertIn("candidate", failure_plans[0]["nodes"][0]["failure_context"]["attempts"][0])

    def test_revisions_are_bounded_when_evidence_remains_missing(self):
        cfg = config(); cfg["runtime"]["max_revisions"] = 1
        result, _, _ = self.run_flow({r: RecoveryClient(r, True) for r in ("small", "large")}, cfg)
        self.assertEqual(result["status"], "failed")
        self.assertIn("修订次数耗尽", result["error"])
        self.assertEqual(result["nodes"][0]["revision_count"], 1)

    def test_revision_can_use_new_helper_but_cannot_form_cycle(self):
        a = dict(node(), status="needs_revision")
        b = dict(node(2), question="Who follows #1?", depends_on=["n1"], status="pending")
        c = dict(node(3), question="Which alias denotes Alpha?", status="pending")
        fixed = dict(node(), question="Where was #3 born?", depends_on=["n3"])
        self.assertEqual(validate_revision(fixed, [a, b, c], 2), fixed)
        self.assertEqual(levels([fixed, b, c]), [["n3"], ["n1"], ["n2"]])
        c.update(question="Which alias denotes #2?", depends_on=["n2"])
        with self.assertRaises(ValueError):
            validate_revision(fixed, [a, b, c], 2)
        a["status"] = "succeeded"
        with self.assertRaises(ValueError):
            validate_revision(fixed, [a, b], 2)

    def test_resume_preserves_waiting_revision_and_does_not_retry_small(self):
        clients = {r: RecoveryClient(r) for r in ("small", "large")}
        _, _, saves = self.run_flow(clients)
        snapshot = next(r for r in saves if r["nodes"] and r["nodes"][0]["status"] == "needs_revision")
        result, clients, _ = self.run_flow({r: RecoveryClient(r) for r in ("small", "large")}, previous=snapshot)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(clients["small"].requests, [])

    def test_backtracking_invalidates_descendants_and_replaces_real_inputs(self):
        class BacktrackClient(FakeClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                self.requests.append((system, data))
                if system == FINISH_PROMPT:
                    return response({"valid": True, "reason": "recovered final answer"})
                if system == "PLAN":
                    nodes = data["nodes"]
                    if not nodes:
                        return response({"action": "add", "node": node()})
                    if len(nodes) == 1:
                        child = dict(node(2), question="Which place contains #1?", depends_on=["n1"])
                        return response({"action": "add", "node": child})
                    if nodes[1]["status"] == "needs_revision":
                        self.parent_was_revisable = "n1" in data["revision_targets"]
                        return response({"action": "revise", "node": dict(node(operation="reason"),
                            question="Where was Alpha born according to the birth record?")})
                    if nodes[1]["status"] == "succeeded":
                        return response({"action": "finish", "final_node": "n2"})
                    return response({"action": "wait"})
                if not data["upstream"]:
                    return response(answer("BetaTown", 9, "Beta was born in BetaTown.")
                                    if data["operation"] == "lookup" else answer())
                if data["upstream"]["n1"]["answer"] == "BetaTown":
                    return response(dict(answer(used=["n1"]), answer="", status="insufficient",
                                         support_type="insufficient", reason="wrong upstream entity"))
                return response(answer(used=["n1"]))
        clients = {r: BacktrackClient(r) for r in ("small", "large")}
        result, clients, _ = self.run_flow(clients)
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        self.assertTrue(clients["large"].parent_was_revisable)
        child = result["nodes"][1]
        self.assertEqual(child["invalidations"][0]["input_results"]["n1"]["answer"], "BetaTown")
        self.assertEqual(child["input_results"]["n1"]["answer"], "AlphaTown")
        self.assertEqual(result["nodes"][0]["revisions"][0]["status"], "succeeded")

    def test_final_scope_check_rejects_premature_completion(self):
        client = FakeClient("large")
        original = client.complete
        def complete(system, user, live=False):
            if system == FINISH_PROMPT:
                return response({"valid": False, "reason": "only an intermediate entity was answered"})
            if system == "PLAN" and any(n["status"] == "needs_revision" for n in json.loads(user)["nodes"]):
                return response({"action": "abort", "reason": "no defensible correction found"})
            return original(system, user, live)
        client.complete = complete
        result, _, _ = self.run_flow({"small": FakeClient("small"), "large": client})
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("answer", result)
        self.assertTrue(any(e["type"] == "final_checked" and not e["valid"] for e in result["events"]))
        self.assertEqual(result["nodes"][-1]["status"], "needs_revision")
        self.assertNotIn("output", result["nodes"][-1])

    def test_final_check_reopens_wrong_answer_instead_of_adding_duplicate(self):
        class FinalCorrectionClient(RecoveryClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                if system == FINISH_PROMPT:
                    if data["nodes"][0]["operation"] == "lookup":
                        duplicate = dict(node(2), question=node()["question"], reason="the existing answer has the wrong relation")
                        return response({"valid": False, "reason": "wrong relation", "next_node": duplicate})
                    return response({"valid": True, "reason": "the relation was corrected"})
                if system != "PLAN":
                    return response(answer())
                return super().complete(system, user, live)
        clients = {r: FinalCorrectionClient(r) for r in ("small", "large")}
        result, _, saves = self.run_flow(clients)
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        self.assertEqual(len(result["nodes"]), 1)
        self.assertEqual(result["nodes"][0]["revision_count"], 1)
        self.assertTrue(any(e["type"] == "revision_required" and e.get("source") == "final_check"
                            for e in result["events"]))
        waiting = next(r["nodes"][0] for r in saves if r["nodes"] and r["nodes"][0]["status"] == "needs_revision")
        self.assertNotIn("output", waiting)
        self.assertEqual(waiting["attempts"][-1]["candidate"]["answer"], "AlphaTown")

    def test_orphaned_dependency_can_be_repaired_after_finish_is_rejected(self):
        class LostBranchClient(FakeClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                if system == "PLAN":
                    nodes = data["nodes"]
                    if len(nodes) == 2:
                        if nodes[1]["status"] == "needs_revision":
                            fixed = dict(node(2, "summarize", ["n1"]), question="Combine #1 with Beta's birthplace.")
                            return response({"action": "revise", "node": fixed})
                        if all(n["status"] == "succeeded" for n in nodes):
                            return response({"action": "finish", "final_node": "n2"})
                        return response({"action": "wait"})
                if system not in ("PLAN", FINISH_PROMPT) and data.get("upstream"):
                    value = answer("AlphaTown and BetaTown", used=["n1"])
                    value["evidence"].append({"paragraph_idx": 9, "quote": "Beta was born in BetaTown."})
                    return response(value)
                return super().complete(system, user, live)
        result, _, _ = self.run_flow({r: LostBranchClient(r) for r in ("small", "large")})
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        self.assertEqual(result["nodes"][1]["depends_on"], ["n1"])
        self.assertEqual(result["nodes"][1]["input_results"]["n1"]["answer"], "AlphaTown")
        self.assertTrue(any(e["type"] == "revision_required" and e.get("source") == "graph_check"
                            for e in result["events"]))

    def test_backtracking_cannot_revise_ancestor_while_descendant_runs(self):
        a = dict(node(), status="succeeded")
        b = dict(node(2), question="Who follows #1?", depends_on=["n1"], status="needs_revision")
        c = dict(node(3), question="Who follows #1 and #2?", depends_on=["n1", "n2"], status="running")
        with self.assertRaises(ValueError):
            validate_revision(node(operation="reason"), [a, b, c], 2)

    def test_evaluation_uses_aliases_and_counts_failed_last_attempt(self):
        self.assertEqual(answer_scores("The Alpha,", ["Alpha"]), (1, 1.0))
        self.assertEqual(answer_scores("Alpha Town", ["Town"]), (0, 2 / 3))
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p, g = Path(tmp) / "pred.jsonl", Path(tmp) / "gold.jsonl"
            predictions = [{"task_id": "a", "status": "succeeded", "answer": "Alias"},
                           {"task_id": "b", "status": "succeeded", "answer": "Beta"},
                           {"task_id": "b", "status": "failed", "answer": "Beta"}]
            p.write_text("\n".join(json.dumps(x) for x in predictions))
            g.write_text('\n'.join(json.dumps(x) for x in [
                {"id": "a", "answer": "Alpha", "answer_aliases": ["Alias"]}, {"id": "b", "answer": "Beta"}]))
            result = evaluate(p, g)
            self.assertEqual(result["evaluated"], 2)
            self.assertEqual(result["exact_match"], 0.5)
            self.assertEqual(result["completed"], 1)
            g.write_text(json.dumps({"id": "a", "answer": "Alpha"}))
            with self.assertRaisesRegex(ValueError, "id"):
                evaluate(p, g)

    def test_parallel_small_and_large_and_actual_feedback(self):
        gates = (threading.Event(), threading.Event())
        clients = {"small": FakeClient("small", gates), "large": FakeClient("large", gates)}
        result, clients, saves = self.run_flow(clients)
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        self.assertEqual(result["answer"], "AlphaTown and BetaTown")
        self.assertEqual(result["layers"], [["n1", "n2"], ["n3"]])
        self.assertTrue(all(e.is_set() for e in gates))
        self.assertEqual(result["nodes"][2]["input_results"]["n1"]["answer"], "AlphaTown")
        self.assertEqual(result["nodes"][2]["resolved_question"], "Combine the locations AlphaTown and BetaTown.")
        plans = [p for system, p in clients["large"].requests if system == "PLAN"]
        self.assertTrue(any(any(n.get("output", {}).get("answer") == "AlphaTown"
                               for n in p["nodes"]) for p in plans))
        self.assertTrue(all("hidden-hop-id" not in json.dumps(p) for p in plans))
        self.assertTrue(any(r["nodes"] and r["nodes"][0]["status"] == "running" for r in saves))
        self.assertEqual(sum(v["total_tokens"] for v in result["metrics"].values()), len(result["calls"]) * 30)
        executions = [p for system, p in clients["large"].requests if system != "PLAN"]
        merge = next(p for p in executions if "Combine" in p["question"])
        self.assertEqual(merge["required_used_inputs"], ["n1", "n2"])
        self.assertEqual(merge["question_template"], "Combine the locations #1 and #2.")

    def test_add_batch_accepts_independent_nodes_and_parallel_roots(self):
        class BatchClient(FakeClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                if system == FINISH_PROMPT:
                    return response({"valid": True, "reason": "both parts answered"})
                if system == "PLAN":
                    nodes = data["nodes"]
                    if not nodes:
                        return response({"action": "add_batch", "nodes": [
                            node(1, "lookup", []), node(2, "lookup", [])]})
                    if len(nodes) == 2 and all(n["status"] == "succeeded" for n in nodes):
                        return response({"action": "add", "node": node(3, "summarize", ["n1", "n2"])})
                    if len(nodes) == 3 and all(n["status"] == "succeeded" for n in nodes):
                        return response({"action": "finish", "final_node": "n3"})
                    return response({"action": "wait"})
                return super().complete(system, user, live)
        result, _, _ = self.run_flow({r: BatchClient(r) for r in ("small", "large")})
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        self.assertEqual(result["layers"], [["n1", "n2"], ["n3"]])
        self.assertEqual(len(result["nodes"]), 3)
        self.assertTrue(any(e.get("type") == "batch_added" for e in result["events"]))

    def test_paragraph_indices_lock_valid_and_ignore_invalid(self):
        t = task()
        raw = dict(node(), paragraph_indices=[7, 9])
        clean_paragraph_hint(raw, t)
        self.assertEqual(raw["paragraph_indices"], [7, 9])
        raw = dict(node(), paragraph_indices=[999])
        clean_paragraph_hint(raw, t)
        self.assertNotIn("paragraph_indices", raw)
        route, ctx = route_node(node(), "Alpha", {}, t["paragraphs"], config()["routing"], locked=[7])
        self.assertEqual(route["role"], "small")
        self.assertEqual([p["idx"] for p in ctx], [7])

    def test_full_plan_response_is_rejected(self):
        bad = FakeClient("large")
        bad.complete = lambda *a, **k: response({"nodes": [node(), node(2)]})
        result, _, _ = self.run_flow({"large": bad, "small": FakeClient("small")})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["nodes"], [])
        self.assertEqual(len(result["calls"]), 3)

    def test_budget_is_shared_with_fallback(self):
        clients = {"small": FakeClient("small", insufficient=True), "large": FakeClient("large")}
        budget = CallBudget(1)
        result = execute_node(node(), node()["question"], {}, task(), clients, budget, config()["routing"])
        self.assertTrue(result["halt"])
        self.assertEqual(len(budget.snapshot()), 1)
        self.assertEqual(clients["large"].requests, [])

    def test_quota_checkpoint_resumes_without_repeating_successful_node(self):
        first = {"small": FakeClient("small"), "large": FakeClient("large", quota=True)}
        result, _, _ = self.run_flow(first)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["nodes"][0]["status"], "succeeded")
        second, clients, _ = self.run_flow(previous=result)
        self.assertEqual(second["status"], "succeeded")
        self.assertEqual(len(clients["small"].requests), 0)
        self.assertEqual(second["resumes"], 1)
        self.assertEqual(sum(c["purpose"] == "execute:n1:small" for c in second["calls"]), 1)

    def test_resume_rejects_changed_input(self):
        result, _, _ = self.run_flow()
        result["input_sha256"] = "changed"
        with self.assertRaises(ValueError):
            self.run_flow(previous=result)

    def test_wait_without_work_is_bounded(self):
        client = FakeClient("large")
        client.complete = lambda *a, **k: response({"action": "wait"})
        cfg = config(); cfg["runtime"]["max_planner_calls"] = 2
        result, _, _ = self.run_flow({"large": client, "small": FakeClient("small")}, cfg)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_kind"], "budget")
        self.assertFalse(result["resumable"])
        self.assertEqual(len(result["calls"]), 2)

    def test_jsonl_recovers_partial_tail(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = Path(tmp) / "workflows.jsonl"
            good = json.dumps({"task_id": "one", "status": "succeeded"}) + "\n"
            p.write_text(good + '{"task_id":', encoding="utf-8")
            self.assertEqual(set(completed_records(p)), {"one"})
            self.assertEqual(p.read_text(), good)

    def test_api_and_local_require_live_before_use(self):
        with patch("urllib.request.build_opener", side_effect=AssertionError("network")):
            with self.assertRaises(ModelError):
                APIModel({}).complete("s", "u")
            with self.assertRaises(ModelError):
                LocalModel({}).complete("s", "u")

    def test_api_truncation_retains_usage_without_key(self):
        cfg = {"model": "deepseek-test", "base_url": "https://example.invalid/v1",
               "api_key_env": "DEEPSEEK_API_KEY", "max_tokens": 100, "timeout_seconds": 10}
        body = {"choices": [{"message": {"content": '{"partial":'}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 100}}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "never-persist"}), patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(body).encode()
            with self.assertRaises(ModelError) as exc:
                APIModel(cfg).complete("s", "u", live=True)
        self.assertEqual(exc.exception.details["usage"]["completion_tokens"], 100)
        self.assertNotIn("never-persist", json.dumps(exc.exception.details))

    def test_api_and_workflow_metrics_retain_prompt_cache_usage(self):
        cfg = {"model": "deepseek-test", "base_url": "https://example.invalid/v1",
               "api_key_env": "DEEPSEEK_API_KEY", "max_tokens": 100, "timeout_seconds": 10}
        body = {"choices": [{"message": {"content": '{"valid":true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                          "prompt_cache_hit_tokens": 94, "prompt_cache_miss_tokens": 6,
                          "provider_private_field": 999}}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}), patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(body).encode()
            response_value = APIModel(cfg).complete("Return JSON.", "question", live=True)
        self.assertEqual(response_value["usage"]["prompt_cache_hit_tokens"], 94)
        self.assertEqual(response_value["usage"]["prompt_cache_miss_tokens"], 6)
        self.assertNotIn("provider_private_field", response_value["usage"])
        metrics = summarize_calls([{"purpose": "planner", "response": response_value}])
        self.assertEqual(metrics["planner"]["prompt_cache_hit_tokens"], 94)
        self.assertEqual(metrics["planner"]["prompt_cache_hit_rate"], 0.94)

    def test_api_json_mode_includes_required_format_instruction(self):
        cfg = {"model": "deepseek-test", "base_url": "https://example.invalid/v1",
               "api_key_env": "DEEPSEEK_API_KEY", "max_tokens": 100, "timeout_seconds": 10}
        body = {"choices": [{"message": {"content": '{"valid":true}'}, "finish_reason": "stop"}]}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}), patch("urllib.request.build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(body).encode()
            APIModel(cfg).complete("Check the proposed answer.", "question", live=True)
            request = opener.return_value.open.call_args.args[0]
            messages = json.loads(request.data)["messages"]
            self.assertIn("JSON", messages[0]["content"])

    def test_resume_loader_retains_failed_tasks_with_successful_nodes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = Path(tmp) / "workflows.jsonl"
            records = [{"task_id": "a", "status": "failed", "nodes": [dict(node(), status="succeeded")]},
                       {"task_id": "b", "status": "succeeded"}]
            p.write_text("\n".join(json.dumps(r) for r in records))
            self.assertEqual(set(completed_records(p)), {"b"})
            recovered = completed_records(p, successful_only=False)
            self.assertEqual(recovered["a"]["nodes"][0]["status"], "succeeded")

    def test_successful_checkpoint_is_appended_after_interrupted_log_write(self):
        cfg = config()
        record = {"task_id": task()["id"], "status": "succeeded", "wall_seconds": 1,
                  "input_sha256": fingerprint(task()), "signature": workflow_signature(cfg, "PLAN")}
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            output = Path(tmp)
            write_json(output / "checkpoint.json", record)
            with patch("run.load_prompt", return_value="PLAN"), patch("run.make_clients", return_value={}), \
                    patch("run.run_task", side_effect=AssertionError("must not call models")), redirect_stdout(io.StringIO()):
                self.assertEqual(run_selected([task()], cfg, output, resume=True), 0)
            self.assertEqual(completed_records(output / "workflows.jsonl"), {task()["id"]: record})

    def test_multiple_tasks_run_concurrently_with_separate_checkpoints(self):
        samples = [dict(task(), id=f"parallel-{i}") for i in range(2)]
        cfg = config()
        cfg["runtime"]["max_inflight_tasks"] = 2
        barrier = threading.Barrier(2)
        active = {"count": 0, "maximum": 0}
        active_lock = threading.Lock()

        def fake_run(sample, clients, config_value, prompt, save, previous=None):
            with active_lock:
                active["count"] += 1
                active["maximum"] = max(active["maximum"], active["count"])
            barrier.wait(timeout=2)
            record = {"task_id": sample["id"], "status": "succeeded", "wall_seconds": 0.01,
                      "input_sha256": fingerprint(sample),
                      "signature": workflow_signature(config_value, prompt)}
            save(record)
            with active_lock:
                active["count"] -= 1
            return record

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, patch("run.load_prompt", return_value="PLAN"), \
                patch("run.make_clients", return_value={}), patch("run.run_task", side_effect=fake_run), \
                redirect_stdout(io.StringIO()):
            output = Path(tmp)
            self.assertEqual(run_selected(samples, cfg, output), 0)
            self.assertEqual(active["maximum"], 2)
            saved = checkpoint_records(output / "checkpoints")
            self.assertEqual(set(saved), {s["id"] for s in samples})
            self.assertEqual(set(completed_records(output / "workflows.jsonl")), set(saved))
            run_metrics = parse_json((output / "run_metrics.json").read_text())
            self.assertEqual(run_metrics["max_inflight_tasks"], 2)
            self.assertEqual(run_metrics["succeeded_tasks"], 2)
            self.assertEqual(run_metrics["batch_wall_seconds"], run_metrics["cumulative_batch_wall_seconds"])

    def test_resume_uses_each_tasks_own_checkpoint(self):
        samples = [dict(task(), id="done"), dict(task(), id="paused")]
        cfg = config()
        prompt = "PLAN"
        signature = workflow_signature(cfg, prompt)
        done = {"task_id": "done", "status": "succeeded", "wall_seconds": 1,
                "input_sha256": fingerprint(samples[0]), "signature": signature}
        paused = {"task_id": "paused", "status": "paused", "wall_seconds": 2,
                  "input_sha256": fingerprint(samples[1]), "signature": signature}
        invoked = []

        def fake_run(sample, clients, config_value, prompt_value, save, previous=None):
            invoked.append((sample["id"], previous["status"]))
            record = dict(previous, status="succeeded", wall_seconds=3)
            save(record)
            return record

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, patch("run.load_prompt", return_value=prompt), \
                patch("run.make_clients", return_value={}), patch("run.run_task", side_effect=fake_run), \
                redirect_stdout(io.StringIO()):
            output = Path(tmp)
            checkpoint_dir = output / "checkpoints"
            write_json(task_checkpoint(checkpoint_dir, "done"), done)
            write_json(task_checkpoint(checkpoint_dir, "paused"), paused)
            write_json(output / "run_metrics.json", {"batch_wall_seconds": 4})
            (output / "workflows.jsonl").write_text(json.dumps(done) + "\n", encoding="utf-8")
            self.assertEqual(run_selected(samples, cfg, output, resume=True), 0)
            self.assertEqual(invoked, [("paused", "paused")])
            self.assertEqual(set(completed_records(output / "workflows.jsonl")), {"done", "paused"})
            metrics = parse_json((output / "run_metrics.json").read_text())
            self.assertAlmostEqual(metrics["cumulative_batch_wall_seconds"],
                                   4 + metrics["batch_wall_seconds"], places=3)

    def test_workload_reports_cache_and_concurrent_batch_wall_time(self):
        call = {"purpose": "planner", "response": {"usage": {
            "prompt_tokens": 100, "completion_tokens": 4, "total_tokens": 104,
            "prompt_cache_hit_tokens": 95, "prompt_cache_miss_tokens": 5}}}
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            output = Path(tmp)
            (output / "workflows.jsonl").write_text(json.dumps({
                "task_id": "a", "wall_seconds": 7, "calls": [call]}) + "\n", encoding="utf-8")
            write_json(output / "run_metrics.json", {
                "batch_wall_seconds": 1.25, "cumulative_batch_wall_seconds": 4.25})
            got = workload(output / "workflows.jsonl")
            self.assertEqual(got["wall_seconds"], 7)
            self.assertEqual(got["batch_wall_seconds"], 4.2)
            self.assertEqual(got["prompt_cache_hit_rate"], 0.95)

    def test_workload_accepts_old_logs_without_cache_fields(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "workflows.jsonl"
            path.write_text(json.dumps({"task_id": "old", "calls": []}) + "\n", encoding="utf-8")
            got = workload(path)
            self.assertEqual(got["prompt_cache_hit_tokens"], 0)
            self.assertEqual(got["prompt_cache_miss_tokens"], 0)
            self.assertEqual(got["prompt_cache_hit_rate"], 0)
            self.assertIsNone(got["batch_wall_seconds"])

    def test_prompt_loads_all_thirty_examples(self):
        prompt = load_prompt()
        self.assertEqual(prompt.count("源 ID："), 30)
        self.assertIn('{"action":"wait"}', prompt)
        self.assertIn("一次输出完整计划", prompt)

    def test_parallel_only_route_mode_keeps_serial_lookup_off_small_model(self):
        cfg = config()["routing"]
        route, _ = route_node(node(), node()["question"], {}, task()["paragraphs"], cfg,
                              allow_small=False)
        self.assertEqual(route["role"], "large")
        self.assertIn("串行关键路径", route["reason"])

    def test_atomic_checkpoint_preserves_json(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = Path(tmp) / "checkpoint.json"
            write_json(p, {"old": True})
            write_json(p, {"new": "中文"})
            self.assertEqual(parse_json(p.read_text()), {"new": "中文"})
            self.assertEqual([x.name for x in Path(tmp).iterdir()], ["checkpoint.json"])

    def test_env_allowlist_and_existing_environment_wins(self):
        cfg = json.loads((ROOT / "model.json").read_text())
        cfg["runtime"].pop("max_inflight_tasks")
        cfg["runtime"].pop("max_total_revisions")
        cfg["routing"].pop("small_route_mode")
        cfg["small"] = {"provider": "api", "model": "test", "base_url": "http://localhost:8000/v1",
                        "api_key_env": "QWEN_API_KEY", "max_tokens": 100, "timeout_seconds": 10}
        cfg["env_file"] = ".env"
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, patch.dict(os.environ, {"DEEPSEEK_API_KEY": "existing"}, clear=True):
            root = Path(tmp)
            (root / ".env").write_text("DEEPSEEK_API_KEY=ignored\nHOME=/invalid\nDEEPSEEK_BASE_URL=https://example.invalid/v1\n")
            write_json(root / "model.json", cfg)
            got = load_config(root / "model.json")
            self.assertEqual(got["large"]["base_url"], "https://example.invalid/v1")
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "existing")
            self.assertNotIn("HOME", os.environ)
            self.assertEqual(got["runtime"]["max_inflight_tasks"], 4)
            self.assertEqual(got["runtime"]["max_total_revisions"], 3)
            self.assertEqual(got["routing"]["small_route_mode"], "cost")


if __name__ == "__main__":
    unittest.main()
