"""Count real client invocations with deterministic offline models, including failures."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from decompose import available_revision_targets, planner_payload, run_task, workflow_signature, fingerprint, write_json
from executor import FINISH_PROMPT, REVIEW_PROMPT, check_finish, execute_node
from graph import normalize_planner_node, validate_node, validate_revision
from model import CallBudget, ModelError, parse_model_json
from run import run_selected
from test_core import FakeClient, answer, config, node, response, task, ROOT


def finish(key):
    return {"action": "finish", "final_node": key,
            "reason": "The final node performs the requested last relation and returns the requested answer type."}


class FrontierClient(FakeClient):
    """Two independent branches followed by a join; any polling is a test failure."""
    def complete(self, system, user, live=False):
        data = json.loads(user)
        if system == "PLAN":
            self.requests.append((system, data))
            nodes = data["nodes"]
            if not nodes:
                return response({"action": "add_batch", "nodes": [node(), node(2, "reason")]})
            if any(n["status"] in ("pending", "running") for n in nodes):
                raise AssertionError("planner called before its execution frontier finished")
            if len(nodes) == 2:
                return response({"action": "add", "node": node(3, "summarize", ["n1", "n2"])})
            return response(finish("n3"))
        return super().complete(system, user, live)


class CallReductionTests(unittest.TestCase):
    def run_flow(self, clients, cfg=None, previous=None):
        saves = []
        with redirect_stdout(io.StringIO()):
            result = run_task(task(), clients, cfg or config(), "PLAN",
                              lambda r: saves.append(copy.deepcopy(r)), previous)
        return result, saves

    def test_frontier_runs_in_parallel_without_wait_calls_or_duplicate_reviews(self):
        gates = (threading.Event(), threading.Event())
        clients = {r: FrontierClient(r, gates) for r in ("small", "large")}
        result, _ = self.run_flow(clients)
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        self.assertEqual(result["answer"], "AlphaTown and BetaTown")
        self.assertEqual(len(result["calls"]), 6)  # three planner + three executions
        self.assertEqual(result["metrics"]["planner"]["calls"], 3)
        self.assertFalse(any(e["type"] == "planner_wait" for e in result["events"]))
        self.assertFalse(any(c["purpose"].startswith(("verify:", "final_check:")) for c in result["calls"]))
        self.assertEqual(result["final_review"]["method"], "planner")
        self.assertTrue(all(g.is_set() for g in gates))

    def test_parallel_only_mode_uses_large_for_a_serial_lookup(self):
        class Client(FakeClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                self.requests.append((system, data))
                if system == "PLAN":
                    return response({"action": "add", "node": node()} if not data["nodes"] else finish("n1"))
                return response(answer())
        cfg = config(); cfg["routing"]["small_route_mode"] = "parallel_only"
        clients = {r: Client(r) for r in ("small", "large")}
        result, _ = self.run_flow(clients, cfg)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["nodes"][0]["executed_by"], "large")
        self.assertEqual(clients["small"].requests, [])
        self.assertEqual(len(result["calls"]), 3)

    def test_parallel_only_mode_still_uses_small_on_a_parallel_frontier(self):
        cfg = config(); cfg["routing"]["small_route_mode"] = "parallel_only"
        gates = (threading.Event(), threading.Event())
        clients = {r: FrontierClient(r, gates) for r in ("small", "large")}
        result, _ = self.run_flow(clients, cfg)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["nodes"][0]["executed_by"], "small")
        self.assertEqual(result["nodes"][1]["executed_by"], "large")
        self.assertTrue(all(g.is_set() for g in gates))

    def test_dependent_batch_executes_with_real_inputs_and_two_planner_calls(self):
        class Client(FakeClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                if system == "PLAN":
                    if not data["nodes"]:
                        return response({"action": "add_batch", "nodes": [
                            node(), dict(node(2), question="Where is #1?", depends_on=[])]})
                    return response(finish("n2"))
                if data.get("upstream"):
                    self.used = data["upstream"]["n1"]["answer"]
                    return response(answer(used=["n1"]))
                return super().complete(system, user, live)
        clients = {r: Client(r) for r in ("small", "large")}
        result, _ = self.run_flow(clients)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["metrics"]["planner"]["calls"], 2)
        self.assertEqual(len(result["calls"]), 4)
        self.assertEqual(result["nodes"][1]["resolved_question"], "Where is AlphaTown?")
        self.assertEqual(clients["small"].used, "AlphaTown")

    def test_bad_second_batch_node_does_not_commit_first_or_execute_it(self):
        class Client(FakeClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                self.requests.append((system, data))
                if data.get("validation_feedback"):
                    return response({"action": "abort", "reason": "invalid test batch"})
                return response({"action": "add_batch", "nodes": [node(), dict(node(2), question="Who is #99?")]})
        clients = {"large": Client("large"), "small": FakeClient("small")}
        result, _ = self.run_flow(clients)
        self.assertEqual(result["nodes"], [])
        self.assertEqual(len(result["calls"]), 2)
        self.assertEqual(clients["small"].requests, [])
        self.assertFalse(any(e["type"] in ("node_added", "batch_added") for e in result["events"]))

    def test_batch_limit_rejects_without_partial_graph(self):
        client = FakeClient("large")
        client.complete = lambda *a, **kw: response({"action": "add_batch", "nodes": [node(), node(2)]})
        cfg = config(); cfg["runtime"]["max_batch_nodes"] = 1
        result, _ = self.run_flow({"large": client, "small": FakeClient("small")}, cfg)
        self.assertEqual(result["nodes"], [])
        self.assertIn("max_batch_nodes", result["error"])

    def test_normalization_derives_dependencies_without_repairing_unknown_references(self):
        raw = dict(node(2), question="Where are #1 and #1?", depends_on=["n1", "n1", "n9"])
        raw.pop("id")
        normalized = normalize_planner_node(raw, [node()])
        self.assertEqual(normalized["id"], "n2")
        self.assertEqual(normalized["depends_on"], ["n1"])
        validate_node(normalized, [node()])
        bad = normalize_planner_node(dict(raw, question="Where is #99?"), [node()])
        with self.assertRaises(ValueError):
            validate_node(bad, [node()])
        # Declared but unresolvable literal dependencies must not silently disappear.
        unclear = normalize_planner_node(dict(node(2), depends_on=["n1"]), [node()])
        with self.assertRaises(ValueError):
            validate_node(unclear, [node()])

    def test_identical_large_failed_query_cannot_be_reexecuted_by_changing_reason(self):
        old = dict(node(), status="needs_revision", attempts=[{"role": "large", "status": "rejected"}])
        with self.assertRaisesRegex(ValueError, "没有新增信息"):
            validate_revision(dict(node(), reason="try again"), [old], 2)
        self.assertEqual(validate_revision(node(operation="reason"), [old], 2)["operation"], "reason")
        changed = dict(node(), expected_output="a city name, excluding province and country")
        self.assertEqual(validate_revision(changed, [old], 2), changed)

    def test_direct_reason_without_extractive_support_still_requires_review(self):
        client = FakeClient("large")
        def complete(system, user, live=False):
            client.requests.append((system, json.loads(user)))
            return response(answer("AlphaTown is warmer than BetaTown")) if system != REVIEW_PROMPT else response(answer())
        client.complete = complete
        budget = CallBudget(3)
        step = node(operation="reason")
        result = execute_node(step, step["question"], {}, task(),
                              {"large": client, "small": FakeClient("small")}, budget, config()["routing"])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["output"]["answer"], "AlphaTown")
        self.assertEqual(len(budget.snapshot()), 2)
        self.assertEqual(client.requests[1][0], REVIEW_PROMPT)

    def test_checkpoint_persists_terminal_status_and_elapsed_time_before_final_save(self):
        client = FakeClient("large")
        client.complete = lambda *a, **kw: response({"action": "abort", "reason": "no evidence"})
        result, saves = self.run_flow({"large": client, "small": FakeClient("small")})
        terminal = [r for r in saves if r["status"] == "failed"]
        self.assertGreaterEqual(len(terminal), 2)
        self.assertTrue(all(r["resumable"] is False for r in terminal))
        times = [r["wall_seconds"] for r in saves]
        self.assertEqual(times, sorted(times))
        self.assertEqual(times[-1], result["wall_seconds"])

    def test_planner_receives_legal_actions_remaining_budget_and_failed_actions(self):
        record = {"nodes": [dict(node(), status="needs_revision", revision_count=2)],
                  "calls": [{"purpose": "planner"}, {"purpose": "execute:n1:large"}],
                  "planner_state": {"rejected_actions": [{"action": {"action": "wait"}, "error": "no jobs"}]}}
        payload = planner_payload(task(), record, config()["runtime"], "last error")
        self.assertNotIn("wait", payload["allowed_actions"])
        self.assertNotIn("revise", payload["allowed_actions"])
        self.assertEqual(payload["revision_targets"], [])
        self.assertEqual(payload["remaining_planner_calls"], 15)
        self.assertEqual(payload["remaining_model_calls"], 28)
        self.assertEqual(payload["next_node_id"], "n2")
        self.assertEqual(payload["rejected_actions"][0]["error"], "no jobs")

    def test_task_wide_revision_budget_hides_all_revision_targets(self):
        cfg = config()["runtime"]
        cfg["max_total_revisions"] = 1
        nodes = [dict(node(), status="needs_revision", revision_count=1)]
        self.assertEqual(available_revision_targets(nodes, cfg), [])
        record = {"nodes": nodes, "calls": [], "planner_state": {}}
        payload = planner_payload(task(), record, cfg, "evidence still missing")
        self.assertNotIn("revise", payload["allowed_actions"])
        self.assertEqual(payload["remaining_total_revisions"], 0)

    def test_planner_budget_exhaustion_is_terminal_and_resume_adds_no_calls(self):
        client = FakeClient("large")
        client.complete = lambda *a, **kw: response({"action": "wait"})
        cfg = config(); cfg["runtime"]["max_planner_calls"] = 2
        first, _ = self.run_flow({"large": client, "small": FakeClient("small")}, cfg)
        self.assertEqual(first["error_kind"], "budget")
        resumed_clients = {r: FakeClient(r) for r in ("small", "large")}
        last, _ = self.run_flow(resumed_clients, cfg, first)
        self.assertEqual(first, last)
        self.assertTrue(all(not c.requests for c in resumed_clients.values()))

    def test_resuming_partial_checkpoint_does_not_replenish_model_budget(self):
        cfg = config(); cfg["runtime"]["max_model_calls"] = 1
        previous = {"task_id": task()["id"], "question": task()["question"], "input_sha256": fingerprint(task()),
                    "signature": workflow_signature(cfg, "PLAN"), "status": "paused", "nodes": [],
                    "calls": [{"purpose": "planner", "status": "succeeded", "response": response({"action": "wait"})}],
                    "events": [], "wall_seconds": 0, "resumes": 0}
        clients = {r: FakeClient(r) for r in ("small", "large")}
        result, _ = self.run_flow(clients, cfg, previous)
        self.assertEqual(result["error_kind"], "budget")
        self.assertEqual(len(result["calls"]), 1)
        self.assertTrue(all(not c.requests for c in clients.values()))

    def test_cumulative_task_time_is_not_reset_on_resume(self):
        cfg = config()
        previous = {"task_id": task()["id"], "question": task()["question"], "input_sha256": fingerprint(task()),
                    "signature": workflow_signature(cfg, "PLAN"), "status": "interrupted", "nodes": [],
                    "calls": [], "events": [], "wall_seconds": cfg["runtime"]["max_task_seconds"], "resumes": 0}
        result, _ = self.run_flow({r: FakeClient(r) for r in ("small", "large")}, cfg, previous)
        self.assertEqual(result["error_kind"], "budget")
        self.assertEqual(result["calls"], [])

    def test_rejected_feedback_survives_transport_pause_and_resume(self):
        class Client(FakeClient):
            def complete(self, system, user, live=False):
                self.requests.append((system, json.loads(user)))
                if len(self.requests) == 1:
                    return response({"action": "wait"})
                raise ModelError("temporary connection failure")
        first, _ = self.run_flow({"large": Client("large"), "small": FakeClient("small")})
        self.assertEqual(first["status"], "paused")
        self.assertTrue(first["resumable"])
        self.assertEqual(first["planner_state"]["invalid_actions"], 1)
        client = FakeClient("large")
        def abort(system, user, live=False):
            client.requests.append((system, json.loads(user)))
            return response({"action": "abort", "reason": "no new evidence"})
        client.complete = abort
        result, _ = self.run_flow({"large": client, "small": FakeClient("small")}, previous=first)
        self.assertEqual(len(result["calls"]), 3)
        payload = client.requests[0][1]
        self.assertIn("没有在途", payload["validation_feedback"])
        self.assertEqual(len(payload["rejected_actions"]), 1)
        self.assertEqual(payload["remaining_planner_calls"], 14)

    def test_transport_pause_resumes_large_fallback_without_repeating_small(self):
        class Client(FakeClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                if system == "PLAN":
                    return response({"action": "add", "node": node()} if not data["nodes"] else finish("n1"))
                raise ModelError("temporary transport failure")
        small = FakeClient("small", insufficient=True)
        first, _ = self.run_flow({"large": Client("large"), "small": small})
        self.assertEqual(first["status"], "paused")
        self.assertEqual(len(first["calls"]), 3)
        large = FakeClient("large")
        def complete(system, user, live=False):
            large.requests.append((system, json.loads(user)))
            return response(finish("n1")) if system == "PLAN" else response(answer())
        large.complete = complete
        next_small = FakeClient("small")
        result, _ = self.run_flow({"large": large, "small": next_small}, previous=first)
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        self.assertEqual(next_small.requests, [])
        self.assertEqual(sum(c["purpose"].endswith(":small") for c in result["calls"]), 1)
        self.assertEqual(len(result["calls"]), 5)

    def test_interrupted_inference_review_reuses_existing_candidate(self):
        class Client(FakeClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                if system == "PLAN":
                    return response({"action": "add", "node": node(operation="reason")})
                if system == REVIEW_PROMPT:
                    raise ModelError("temporary review transport failure")
                return response(dict(answer(), support_type="inferred", assumptions=["alias bridge"]))
        first, _ = self.run_flow({"large": Client("large"), "small": FakeClient("small")})
        self.assertEqual(first["status"], "paused")
        large = FakeClient("large")
        def complete(system, user, live=False):
            data = json.loads(user)
            large.requests.append((system, data))
            return response(finish("n1")) if system == "PLAN" else response(answer())
        large.complete = complete
        result, _ = self.run_flow({"large": large, "small": FakeClient("small")}, previous=first)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(large.requests[0][0], REVIEW_PROMPT)
        self.assertEqual(large.requests[0][1]["unverified_candidates"][0]["answer"], "AlphaTown")
        self.assertEqual(len(result["calls"]), 5)

    def test_candidate_survives_multiple_interrupted_reviews(self):
        class Client(FakeClient):
            def complete(self, system, user, live=False):
                if system == "PLAN":
                    return response({"action": "add", "node": node(operation="reason")})
                if system == REVIEW_PROMPT:
                    raise ModelError("temporary review failure")
                return response(dict(answer(), support_type="inferred", assumptions=["alias bridge"]))
        clients = {"large": Client("large"), "small": FakeClient("small")}
        first, _ = self.run_flow(clients)
        second, _ = self.run_flow(clients, previous=first)
        self.assertEqual(second["status"], "paused")
        large = FakeClient("large")
        def complete(system, user, live=False):
            large.requests.append((system, json.loads(user)))
            return response(finish("n1")) if system == "PLAN" else response(answer())
        large.complete = complete
        result, _ = self.run_flow({"large": large, "small": FakeClient("small")}, previous=second)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(large.requests[0][0], REVIEW_PROMPT)
        self.assertEqual(large.requests[0][1]["unverified_candidates"][0]["answer"], "AlphaTown")
        self.assertEqual(len(result["calls"]), 6)

    def test_resume_skips_terminal_failure_without_appending_duplicate_record(self):
        cfg = config()
        record = {"task_id": task()["id"], "status": "failed", "resumable": False, "error": "no progress",
                  "wall_seconds": 1, "input_sha256": fingerprint(task()), "signature": workflow_signature(cfg, "PLAN")}
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            output = Path(tmp)
            write_json(output / "checkpoint.json", record)
            path = output / "workflows.jsonl"
            text = json.dumps(record) + "\n"
            path.write_text(text)
            with patch("run.load_prompt", return_value="PLAN"), patch("run.make_clients", return_value={}), \
                    patch("run.run_task", side_effect=AssertionError("terminal task must not resume")), redirect_stdout(io.StringIO()):
                self.assertEqual(run_selected([task()], cfg, output, resume=True), 1)
            self.assertEqual(path.read_text(), text)

    def test_always_final_check_retains_independent_review(self):
        cfg = config(); cfg["runtime"]["final_check_mode"] = "always"
        result, _ = self.run_flow({r: FrontierClient(r) for r in ("small", "large")}, cfg)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(len(result["calls"]), 7)
        self.assertEqual(sum(c["purpose"] == "final_check:large" for c in result["calls"]), 1)

    def test_missing_finish_reason_falls_back_to_independent_review(self):
        n = dict(node(), status="succeeded", output=answer())
        budget = CallBudget(1)
        result = check_finish(task(), [n], "n1", FakeClient("large"), budget)
        self.assertTrue(result["valid"])
        self.assertEqual(len(budget.snapshot()), 1)

    def test_planner_finish_does_not_bypass_owner_location_rule(self):
        sample = dict(task(), question="What administrative territorial entity is the owner of Cedar Hall located?")
        owner = dict(node(), question="Who is the owner of Cedar Hall?", status="succeeded", output=answer("Alpha"))
        budget = CallBudget(1)
        result = check_finish(sample, [owner], "n1", FakeClient("large"), budget, planner_decision="done")
        self.assertFalse(result["valid"])
        self.assertEqual(budget.snapshot(), [])

    def test_small_input_bookkeeping_and_json_fences_do_not_trigger_fallback(self):
        client = FakeClient("small")
        value = answer(); value.pop("used_inputs")
        client.complete = lambda *a, **kw: dict(response(value), raw_response="```json\n" + json.dumps(value) + "\n```")
        budget = CallBudget(2)
        step = dict(node(2), question="Where is #1?", depends_on=["n1"])
        result = execute_node(step, "Where is AlphaTown?", {"n1": answer()}, task(),
                              {"small": client, "large": FakeClient("large")}, budget, config()["routing"])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["executed_by"], "small")
        self.assertEqual(result["output"]["used_inputs"], ["n1"])
        self.assertEqual(len(budget.snapshot()), 1)
        self.assertEqual(result["attempts"][0]["protocol_repairs"][0]["field"], "used_inputs")
        for raw in ['```json\n{"a":1,"a":2}\n```', '```json\n{"a":NaN}\n```', None]:
            with self.assertRaises(ValueError):
                parse_model_json(raw)

    def test_locked_evidence_is_used_by_scheduler_and_executor_consistently(self):
        sample = task()
        sample["paragraphs"].insert(0, {"idx": 1, "title": "Alpha", "paragraph_text": "Alpha moved in 2001, 2002 and 2003."})
        class Client(FakeClient):
            def complete(self, system, user, live=False):
                data = json.loads(user)
                if system == "PLAN":
                    return response({"action": "add", "node": dict(node(), paragraph_indices=[7])}
                                    if not data["nodes"] else finish("n1"))
                return super().complete(system, user, live)
        with redirect_stdout(io.StringIO()):
            result = run_task(sample, {r: Client(r) for r in ("small", "large")}, config(), "PLAN", lambda r: None)
        self.assertEqual(result["status"], "succeeded")
        started = next(e for e in result["events"] if e["type"] == "node_started")
        self.assertEqual(started["role"], "small")
        self.assertEqual(result["nodes"][0]["route"]["paragraph_indices"], [7])
        self.assertEqual(result["nodes"][0]["executed_by"], "small")


if __name__ == "__main__":
    unittest.main()
