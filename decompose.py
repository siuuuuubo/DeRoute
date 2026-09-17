"""DeepSeek逐步拆解；就绪节点并行执行，真实结果反馈到下一轮规划。"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from executor import ANSWER_PROMPT, REVIEW_PROMPT, FINISH_PROMPT, check_finish, execute_node, failure_context, readable_paragraphs, route_node, select_small_paragraphs
from graph import descendants_of, levels, normalize_planner_node, ready_nodes, resolve_inputs, revision_targets, validate_finish, validate_node, validate_revision, validate_temporal_scope
from model import BudgetExceeded, CallBudget, ModelError, now, parse_model_json, resumable_model_error

ROOT = Path(__file__).resolve().parent


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def workflow_signature(config, prompt):
    return fingerprint({"config": config, "prompt": prompt,
                        "answer_prompt": ANSWER_PROMPT, "review_prompt": REVIEW_PROMPT,
                        "finish_prompt": FINISH_PROMPT, "protocol_version": 12})


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_prompt():
    instruction = (ROOT / "data_prompts/decompose.txt").read_text(encoding="utf-8")
    examples = (ROOT / "data_prompts/musique_train_30shot_decomposition.txt").read_text(encoding="utf-8")
    return instruction + "\n\n<training_examples>\n" + examples + "\n</training_examples>"


def summarize_calls(calls):
    totals = {}
    for call in calls:
        role = "planner" if call["purpose"] == "planner" else call["purpose"].rsplit(":", 1)[-1]
        item = totals.setdefault(role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                       "total_tokens": 0, "prompt_cache_hit_tokens": 0,
                                       "prompt_cache_miss_tokens": 0, "seconds": 0,
                                       "physical_requests": 0, "queue_wait_seconds": 0,
                                       "service_seconds": 0, "retry_sleep_seconds": 0,
                                       "unknown_usage_calls": 0})
        response = call.get("response") or {}
        item["calls"] += 1
        attempts = response.get("request_attempts", 1)
        item["physical_requests"] += attempts if type(attempts) is int and attempts > 0 else 1
        usage = response.get("usage")
        if not usage:
            item["unknown_usage_calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            item[key] += (usage or {}).get(key, 0)
        item["seconds"] = round(item["seconds"] + response.get("seconds", 0), 3)
        for key in ("queue_wait_seconds", "service_seconds", "retry_sleep_seconds"):
            item[key] = round(item[key] + response.get(key, 0), 3)
    for item in totals.values():
        cache_total = item["prompt_cache_hit_tokens"] + item["prompt_cache_miss_tokens"]
        item["prompt_cache_hit_rate"] = round(item["prompt_cache_hit_tokens"] / cache_total, 6) if cache_total else None
    return totals


def available_revision_targets(nodes, runtime):
    """同时限制单节点和整题修订，避免在多个节点间反复消耗恢复预算。"""
    total = sum(n.get("revision_count", 0) for n in nodes)
    maximum = runtime.get("max_total_revisions", runtime["max_nodes"] * runtime["max_revisions"])
    return [] if total >= maximum else revision_targets(nodes, runtime["max_revisions"])


def planner_payload(task, record, runtime, feedback):
    # id编码了跳数和原始子问题id；只用于存储，不能作为模型输入。
    nodes = []
    for n in record["nodes"]:
        item = {k: n[k] for k in ("id", "question", "operation", "depends_on", "status")}
        item["expected_output"] = n.get("expected_output", "")
        if "paragraph_indices" in n:
            item["paragraph_indices"] = n["paragraph_indices"]
        if n["status"] == "succeeded":
            # 只把 planner 决策真正需要的字段发回给它：答案(供 #k 引用与终检)与证据性质
            # (direct/inferred，供 finish 安全校验)。evidence 逐字引文、used_inputs、reason、
            # assumptions 是 executor 内部自校验信息，planner 用不到，裁掉以缩短 prefill。
            out = n["output"]
            item["output"] = {k: out[k] for k in ("answer", "support_type") if k in out}
        elif n.get("error"):
            item["failure_context"] = failure_context(n)
        item["revision_count"] = n.get("revision_count", 0)
        nodes.append(item)
    targets = available_revision_targets(record["nodes"], runtime)
    maximum_revisions = runtime.get("max_total_revisions",
                                    runtime["max_nodes"] * runtime["max_revisions"])
    allowed = ["abort"]
    if len(nodes) < runtime["max_nodes"]:
        allowed += ["add", "add_batch"]
    if targets:
        allowed.append("revise")
    # 列出结构上允许结束的节点；语义上的最后关系由本轮 planner 判断。
    finish_targets = []
    for node in record["nodes"]:
        try:
            validate_finish(node["id"], record["nodes"])
            finish_targets.append(node["id"])
        except ValueError:
            pass
    if finish_targets:
        allowed.append("finish")
    if any(n["status"] == "running" for n in record["nodes"]):
        allowed.append("wait")
    return {"question": task["question"], "paragraphs": readable_paragraphs(task["paragraphs"]), "nodes": nodes,
            "max_nodes": runtime["max_nodes"], "max_planner_calls": runtime["max_planner_calls"],
            "max_revisions": runtime["max_revisions"],
            "max_total_revisions": maximum_revisions,
            "remaining_total_revisions": max(0, maximum_revisions - sum(
                    n.get("revision_count", 0) for n in record["nodes"])),
            "max_batch_nodes": runtime.get("max_batch_nodes", 3),
            "auto_finish_mode": runtime.get("auto_finish_mode", "safe"),
            "final_check_mode": runtime.get("final_check_mode", "planner"),
            "remaining_planner_calls": runtime["max_planner_calls"] - sum(c["purpose"] == "planner" for c in record["calls"]),
            "remaining_model_calls": runtime["max_model_calls"] - len(record["calls"]),
            "next_node_id": f"n{len(nodes) + 1}", "allowed_actions": allowed,
            "revision_targets": targets, "finish_targets": finish_targets,
            "validation_feedback": feedback,
            "rejected_actions": record.get("planner_state", {}).get("rejected_actions", [])}


def clean_paragraph_hint(raw, task):
    """拆解时可选锁定证据段落；非法或越界索引直接忽略，执行时退回词项检索兜底。"""
    if "paragraph_indices" not in raw:
        return
    valid_idx = {p["idx"] for p in task["paragraphs"]}
    hint = raw["paragraph_indices"]
    if isinstance(hint, list) and hint and all(type(i) is int and i in valid_idx for i in hint):
        raw["paragraph_indices"] = list(dict.fromkeys(hint))
    else:
        raw.pop("paragraph_indices", None)


def run_task(task, clients, config, prompt, save, previous=None):
    runtime = config["runtime"]
    signature = workflow_signature(config, prompt)
    if previous is not None:
        if previous.get("input_sha256") != fingerprint(task) or previous.get("signature") != signature:
            raise ValueError("续跑的输入或配置/提示词已变化，请使用新的输出目录")
        record = copy.deepcopy(previous)
        if record.get("status") == "succeeded" or (record.get("status") == "failed" and not record.get("resumable", False)):
            return record
        record.pop("error", None)
        for n in record["nodes"]:
            if n["status"] not in ("succeeded", "needs_revision"):
                if n.get("halt") and n.get("attempts"):
                    # 网络暂停发生在接管/复核期间时，保留候选并从大模型阶段恢复。
                    n["recovery_context"] = failure_context(n)
                    n["resume_role"] = n["attempts"][-1].get("role")
                n["status"] = "pending"
                for key in ("error", "halt", "output"):
                    n.pop(key, None)
        for call in record["calls"]:
            if call["status"] == "running":
                call["status"] = "interrupted"
        record["resumes"] += 1
    else:
        record = {"task_id": task["id"], "question": task["question"], "input_sha256": fingerprint(task),
                  "signature": signature, "started_at": now(), "nodes": [], "calls": [],
                  "events": [], "resumes": 0, "wall_seconds": 0}
    record.update(status="running", finished_at=None, resumable=True)
    record.pop("error_kind", None)
    # 预算覆盖任务整个生命周期；续跑只恢复剩余工作，不增加调用额度。
    budget = CallBudget(runtime["max_model_calls"], record["calls"])
    prior_wall_seconds = record["wall_seconds"]
    start = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=runtime["max_workers"], thread_name_prefix="execute")
    planner_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner")
    jobs = {}
    planning = None
    waiting = False
    expand_while_running = False
    planner_state = record.setdefault("planner_state", {})
    feedback = planner_state.get("validation_feedback", "")
    invalid_actions = planner_state.get("invalid_actions", 0)
    planner_count = sum(c["purpose"] == "planner" for c in record["calls"])

    def persist():
        planner_state.update(validation_feedback=feedback, invalid_actions=invalid_actions)
        record["resumable"] = record["status"] in ("paused", "interrupted", "running")
        record["wall_seconds"] = round(prior_wall_seconds + time.perf_counter() - start, 3)
        record["calls"] = budget.snapshot()
        record["metrics"] = summarize_calls(record["calls"])
        record["layers"] = levels(record["nodes"])
        save(record)

    def event(kind, **details):
        record["events"].append({"time": now(), "type": kind, **details})

    def prepare_node(raw, existing, *, revision=False):
        if revision and raw.get("id") not in available_revision_targets(existing, runtime):
            raise ValueError("该节点不在当前可修订目标中，或整题修订预算已经耗尽")
        canonical = normalize_planner_node(raw, existing, revision=revision)
        # 先恢复确定的引用，再做时间检查，以实际依赖校验年份来源。
        validate_temporal_scope(canonical, task, existing)
        staged_events = []
        if canonical != raw:
            staged_events.append(("reference_canonicalized", {"original": raw, "canonical": copy.deepcopy(canonical)}))
        raw = copy.deepcopy(canonical)
        clean_paragraph_hint(raw, task)
        if not revision and raw.get("operation") == "lookup" and not raw.get("depends_on") and "paragraph_indices" not in raw:
            # lookup 根节点默认锁定证据：大模型未显式锁定时，用检索兜底锁定。
            # 仅对无依赖的根节点做兜底——带#k的依赖节点此时尚未替换出真实实体，检索会退化，仍交由大模型显式锁定或运行时检索。
            locked = [p["idx"] for p in select_small_paragraphs(
                raw["question"], {}, task["paragraphs"], config["routing"])]
            if locked:
                raw["paragraph_indices"] = locked
                staged_events.append(("evidence_autolocked", {"node_id": raw["id"], "source": "retrieval", "indices": locked}))
        node = copy.deepcopy(validate_revision(raw, existing, runtime["max_revisions"]) if revision
                             else validate_node(raw, existing, runtime["max_nodes"]))
        return node, staged_events

    def commit_node(node, staged_events):
        node.update(status="pending", created_at=now())
        record["nodes"].append(node)
        for kind, details in staged_events:
            event(kind, **details)
        event("node_added", node_id=node["id"], depends_on=node["depends_on"])
        print(f"  新子任务 {node['id']}: {node['question']}", flush=True)
        return node

    def exhaust_budget(error):
        record.update(status="failed", resumable=False, error_kind="budget", error=error)

    def reopen_final(node, error, source):
        node.setdefault("attempts", []).append({"role": "large" if source == "final_check" else "code",
            "stage": "final_review", "status": "rejected", "candidate": node.pop("output"), "error": error})
        node.update(status="needs_revision", error=error)
        event("revision_required", node_id=node["id"], source=source, failure_context=failure_context(node))
        print(f"  最终核验未通过，重新修订 {node['id']}", flush=True)
        if not available_revision_targets(record["nodes"], runtime):
            record.update(status="failed", error="最终核验失败且修订次数耗尽：" + error)

    def finish_record(final_node, answer, review, source):
        record["final_review"] = review
        record.update(status="succeeded", final_node=final_node, answer=answer)
        record["unused_failed_nodes"] = [n["id"] for n in record["nodes"] if n["status"] == "needs_revision"]
        for key in record["unused_failed_nodes"]:
            event("failed_branch_replaced", node_id=key, final_node=final_node)
        record["answer_support"] = "inferred" if any(
            n["output"]["support_type"] == "inferred" for n in record["nodes"]
            if n["status"] == "succeeded") else "direct"
        event("finished", final_node=final_node, source=source)

    def validate_auto_finish_declaration(declaration, added_ids):
        if not isinstance(declaration, dict) or set(declaration) != {"final_node", "reason"}:
            raise ValueError("finish_after_success需要final_node和reason")
        if declaration["final_node"] not in added_ids:
            raise ValueError("finish_after_success.final_node必须是本次新增节点")
        if not isinstance(declaration["reason"], str) or not declaration["reason"].strip():
            raise ValueError("finish_after_success.reason必须说明最后关系和答案类型")

    def declare_auto_finish(declaration, added_ids):
        validate_auto_finish_declaration(declaration, added_ids)
        if runtime.get("auto_finish_mode", "safe") == "off" or runtime.get("final_check_mode") == "always":
            event("auto_finish_ignored", final_node=declaration["final_node"], reason="configuration")
            return
        planner_state["auto_finish"] = copy.deepcopy(declaration)
        event("auto_finish_declared", **declaration)

    def cancel_auto_finish(reason):
        declaration = planner_state.pop("auto_finish", None)
        if declaration:
            event("auto_finish_cancelled", final_node=declaration.get("final_node"), reason=reason)

    def try_auto_finish():
        """只对未修订、全链直接证据的预声明终点自动结束；其余情况仍让planner看到真实结果。"""
        declaration = planner_state.get("auto_finish")
        if (not declaration or runtime.get("auto_finish_mode", "safe") != "safe"
                or runtime.get("final_check_mode", "planner") != "planner"):
            return False
        final_node = declaration["final_node"]
        by_id = {n["id"]: n for n in record["nodes"]}
        node = by_id.get(final_node)
        if node is None:
            cancel_auto_finish("声明的终点不存在")
            return False
        if node["status"] in ("pending", "running") or jobs:
            return False
        if node["status"] != "succeeded":
            cancel_auto_finish("终点执行未成功")
            return False
        try:
            answer = validate_finish(final_node, record["nodes"])
        except ValueError as exc:
            cancel_auto_finish("图尚不满足结束条件：" + str(exc))
            return False
        ancestors, pending = set(), [final_node]
        while pending:
            key = pending.pop()
            if key not in ancestors:
                ancestors.add(key)
                pending.extend(by_id[key]["depends_on"])
        chain = [by_id[key] for key in ancestors]
        if any(n.get("revision_count", 0) for n in chain):
            cancel_auto_finish("最终依赖链经过修订")
            return False
        # 「发生过接管或额外复核」（attempts>1，即小模型失败后大模型接管）不再阻止自动结束：
        # fallback 节点的最终答案由大模型给出，与直接大模型 answer 同样可靠；实测取消后
        # planner finish 也总是原样采用 final_node 答案，放宽零风险。
        if any(n.get("output", {}).get("support_type") != "direct" for n in chain):
            cancel_auto_finish("最终依赖链包含间接推断或未确认证据")
            return False
        review = check_finish(task, record["nodes"], final_node, clients["large"], budget,
                              planner_decision=declaration["reason"])
        review["method"] = "planner_precommitted" if review.get("valid") else review.get("method", "rule")
        event("final_checked", **review)
        planner_state.pop("auto_finish", None)
        if not review["valid"]:
            proposed = review.get("next_node")
            if isinstance(proposed, dict):
                node = commit_node(*prepare_node(proposed, record["nodes"]))
                record["events"][-1]["source"] = "auto_finish_rule"
                print(f"  补充最后一步 {node['id']}: {node['question']}", flush=True)
            else:
                reopen_final(node, "最终关系核验未通过：" + review["reason"], "auto_finish_rule")
            return False
        finish_record(final_node, answer, review, "planner_precommitted")
        return True

    def collect_execution(future, node):
        try:
            result = future.result()
        except Exception as exc:
            result = {"status": "failed", "error": "执行器异常：" + type(exc).__name__}
        node.update(result, finished_at=now())
        event("node_finished", node_id=node["id"], status=node["status"],
              executed_by=node.get("executed_by"), escalated=len(node.get("attempts", [])) > 1)
        print(f"  {node['id']} {node['status']} ({node.get('executed_by', 'none')})", flush=True)
        if node["status"] == "succeeded":
            print("    结果：" + node["output"]["answer"], flush=True)
        elif node.get("error"):
            print("    原因：" + node["error"], flush=True)
        if node["status"] == "failed":
            if node.get("halt"):
                if node.get("error_kind") == "budget":
                    exhaust_budget(node["error"])
                elif record["status"] == "running":
                    record.update(status="paused", resumable=True, error_kind="model", error=node.get("error", "模型调用暂停"))
            else:
                node["status"] = "needs_revision"
                event("revision_required", node_id=node["id"], failure_context=failure_context(node))
                print("    等待拆解器诊断并修订", flush=True)
                if record["status"] == "running" and node.get("revision_count", 0) >= runtime["max_revisions"] and not available_revision_targets(
                        record["nodes"], runtime):
                    record.update(status="failed", error="节点修订次数耗尽：" + node.get("error", ""))

    event("resumed" if previous else "started")
    persist()
    try:
        while record["status"] == "running":
            # 先收集执行结果，保证接下来的规划能看到最新事实。
            for future, node in list(jobs.items()):
                if future.done():
                    collect_execution(future, node)
                    del jobs[future]
                    waiting = False
                    persist()
            if record["status"] != "running":
                break
            if planning is not None and planning.done():
                current = planning
                planning = None
                original_action = None
                try:
                    response = current.result()
                    planner_repairs = []
                    action = parse_model_json(response["raw_response"], planner_repairs)
                    if planner_repairs:
                        event("planner_protocol_repaired", repairs=planner_repairs)
                    if not isinstance(action, dict):
                        raise ValueError("规划结果必须是单个动作对象")
                    original_action = copy.deepcopy(action)
                    expand_while_running = action.pop("continue_planning", False)
                    if type(expand_while_running) is not bool:
                        raise ValueError("continue_planning必须是布尔值")
                    finish_after_success = action.pop("finish_after_success", None)
                    kind = action.get("action")
                    if finish_after_success is not None and kind not in ("add", "add_batch"):
                        raise ValueError("finish_after_success只能与add或add_batch同时使用")
                    if finish_after_success is not None and expand_while_running:
                        raise ValueError("声明自动结束时不能同时continue_planning")
                    if kind == "add" and set(action) == {"action", "node"}:
                        prepared = prepare_node(action["node"], record["nodes"])
                        if finish_after_success is not None:
                            validate_auto_finish_declaration(finish_after_success, {prepared[0]["id"]})
                        added = commit_node(*prepared)
                        if finish_after_success is not None:
                            declare_auto_finish(finish_after_success, {added["id"]})
                        waiting = False
                    elif kind == "add_batch" and set(action) == {"action", "nodes"}:
                        batch = action["nodes"]
                        if not isinstance(batch, list) or not batch:
                            raise ValueError("add_batch需要非空nodes列表")
                        if len(batch) > runtime.get("max_batch_nodes", 3):
                            raise ValueError("add_batch超过max_batch_nodes；只提交当前必要的有限前沿")
                        if len(batch) > runtime["max_nodes"] - len(record["nodes"]):
                            raise ValueError("add_batch节点数超过max_nodes上限")
                        staged = []
                        existing = list(record["nodes"])
                        for raw in batch:
                            prepared = prepare_node(raw, existing)
                            staged.append(prepared)
                            existing.append(dict(prepared[0], status="pending"))
                        if finish_after_success is not None:
                            validate_auto_finish_declaration(
                                finish_after_success, {prepared[0]["id"] for prepared in staged})
                        # 整批验证成功后才改变真实图，失败不留下半批节点。
                        event("batch_added", count=len(batch))
                        for prepared in staged:
                            commit_node(*prepared)
                        if finish_after_success is not None:
                            declare_auto_finish(finish_after_success, {prepared[0]["id"] for prepared in staged})
                        waiting = False
                    elif kind == "revise" and set(action) == {"action", "node"}:
                        replacement, staged_events = prepare_node(action["node"], record["nodes"], revision=True)
                        for event_kind, details in staged_events:
                            event(event_kind, **details)
                        node = next(n for n in record["nodes"] if n["id"] == replacement["id"])
                        previous_failure = failure_context(node)
                        # 回溯修改上游后，所有后代的旧答案失效；先保存原尝试，再等待新输入重跑。
                        for child in record["nodes"]:
                            if child["id"] not in descendants_of(node["id"], record["nodes"]):
                                continue
                            snapshot = copy.deepcopy({k: v for k, v in child.items() if k != "invalidations"})
                            child.setdefault("invalidations", []).append(snapshot)
                            child["status"] = "pending"
                            for key in ("output", "error", "halt", "attempts", "route", "resolved_question",
                                        "input_results", "recovery_context", "resume_role", "executed_by", "reviewed", "started_at", "finished_at"):
                                child.pop(key, None)
                            event("node_invalidated", node_id=child["id"], revised_ancestor=node["id"])
                        history = node.get("revisions", []) + [copy.deepcopy({k: v for k, v in node.items()
                            if k not in ("revisions", "recovery_context")})]
                        created_at = node["created_at"]
                        revision_count = node.get("revision_count", 0) + 1
                        node.clear()
                        node.update(replacement, status="pending", created_at=created_at,
                                    revision_count=revision_count, revisions=history,
                                    recovery_context=previous_failure)
                        event("node_revised", node_id=node["id"], revision_count=revision_count,
                              question=node["question"], reason=node["reason"])
                        print(f"  修订 {node['id']}: {node['question']}", flush=True)
                        waiting = False
                    elif kind == "wait" and set(action) == {"action"}:
                        if not jobs:
                            raise ValueError("没有在途节点可等待；请根据allowed_actions继续或abort")
                        waiting = True
                        expand_while_running = False
                        event("planner_wait")
                    elif kind == "finish" and set(action) in ({"action", "final_node"}, {"action", "final_node", "reason"}):
                        if "reason" in action and (not isinstance(action["reason"], str) or not action["reason"].strip()):
                            raise ValueError("finish.reason必须解释最后关系和答案类型如何满足原题")
                        try:
                            answer = validate_finish(action["final_node"], record["nodes"])
                        except ValueError as exc:
                            final = next((n for n in record["nodes"] if n["id"] == action["final_node"]), None)
                            if final is not None and all(n["status"] == "succeeded" for n in record["nodes"]):
                                # 修订丢失依赖导致孤立分支时，也要允许修复已完成的节点。
                                reopen_final(final, "最终依赖链核验未通过：" + str(exc), "graph_check")
                                waiting, feedback, invalid_actions = False, "", 0
                                persist()
                                continue
                            raise
                        decision = action.get("reason") if runtime.get("final_check_mode", "planner") == "planner" else None
                        review = check_finish(task, record["nodes"], action["final_node"], clients["large"], budget,
                                              planner_decision=decision)
                        event("final_checked", **review)
                        if not review["valid"]:
                            proposed = review.get("next_node")
                            if isinstance(proposed, dict):
                                proposed = normalize_planner_node(proposed, record["nodes"])
                            duplicate = isinstance(proposed, dict) and isinstance(proposed.get("question"), str) and any(
                                n["question"].strip().casefold() == proposed["question"].strip().casefold()
                                for n in record["nodes"])
                            if proposed is None or duplicate:
                                # 已有步骤答错时应重开修订，不能反复添加同一个节点或再次finish。
                                node = next(n for n in record["nodes"] if n["id"] == action["final_node"])
                                error = "最终关系核验未通过：" + review["reason"]
                                if duplicate:
                                    error += "；建议步骤已存在，请修订该链条：" + str(proposed.get("reason", ""))
                                reopen_final(node, error, "final_check")
                            else:
                                node = commit_node(*prepare_node(proposed, record["nodes"]))
                                record["events"][-1]["source"] = "final_check"
                                print(f"  补充最后一步 {node['id']}: {node['question']}", flush=True)
                            waiting = False
                        else:
                            finish_record(action["final_node"], answer, review, "planner_finish")
                    elif kind == "abort" and set(action) == {"action", "reason"}:
                        if not isinstance(action["reason"], str) or not action["reason"].strip():
                            raise ValueError("abort需要明确原因")
                        record.update(status="failed", error="拆解器无法恢复：" + action["reason"])
                        event("aborted", reason=action["reason"])
                    else:
                        raise ValueError("每次只允许一个add、add_batch、revise、wait、finish或abort动作；禁止一次输出完整计划")
                    if kind != "wait":
                        feedback = ""
                        invalid_actions = 0
                        planner_state["rejected_actions"] = []
                except BudgetExceeded as exc:
                    exhaust_budget(str(exc))
                except ModelError as exc:
                    record.update(status="paused" if resumable_model_error(exc) else "failed",
                                  error_kind="model", error=str(exc))
                except ValueError as exc:
                    expand_while_running = False
                    invalid_actions += 1
                    feedback = str(exc)
                    rejected = planner_state.setdefault("rejected_actions", [])
                    rejected.append({"action": original_action, "error": feedback})
                    planner_state["rejected_actions"] = rejected[-3:]
                    event("planner_rejected", error=feedback)
                    if invalid_actions >= 3:
                        record.update(status="failed", error="连续3个规划动作不合法：" + feedback)
                persist()
            if record["status"] != "running":
                break
            if try_auto_finish():
                persist()
                break
            if prior_wall_seconds + time.perf_counter() - start >= runtime["max_task_seconds"]:
                exhaust_budget("达到任务累计时间预算；在途调用结束后保存")
                break
            frontier = ready_nodes(record["nodes"])
            small_running = any(n["route"]["role"] == "small" for n in jobs.values())
            parallel_frontier = runtime["max_workers"] >= 2 and len(frontier) + len(jobs) >= 2
            for node in frontier:
                if len(jobs) >= runtime["max_workers"]:
                    break
                question, upstream = resolve_inputs(node, record["nodes"])
                small_mode = config["routing"].get("small_route_mode", "cost")
                allow_small = small_mode == "cost" or parallel_frontier
                route, _ = route_node(node, question, upstream, task["paragraphs"], config["routing"],
                                      locked=node.get("paragraph_indices"), allow_small=allow_small)
                if route["role"] == "small" and small_running:
                    if small_mode == "cost":
                        continue
                    allow_small = False
                    route, _ = route_node(node, question, upstream, task["paragraphs"], config["routing"],
                                          locked=node.get("paragraph_indices"), allow_small=False)
                node.update(status="running", started_at=now(), route=route,
                            resolved_question=question, input_results=upstream)
                event("node_started", node_id=node["id"], role=route["role"], reason=route["reason"])
                print(f"  {node['id']} → {route['role']}: {route['reason']}", flush=True)
                future = pool.submit(execute_node, copy.deepcopy(node), question, copy.deepcopy(upstream),
                                     task, clients, budget, config["routing"], allow_small=allow_small)
                jobs[future] = node
                small_running = small_running or route["role"] == "small"
                persist()
            if waiting and not jobs:
                waiting = False
                feedback = "没有在途任务可等待；若有needs_revision，请revise或补充独立辅助节点；否则add、finish或abort"
            # 默认为事件驱动：执行完当前前沿才规划。只有 planner 明确还有独立工作，
            # 且执行池有容量时，才允许提前扩展一次；不发请求询问“是否需要等待”。
            can_plan = not jobs or (expand_while_running and len(jobs) < runtime["max_workers"])
            if planning is None and not waiting and can_plan:
                if planner_count >= runtime["max_planner_calls"]:
                    if jobs:
                        waiting = True
                    else:
                        exhaust_budget("达到本任务累计规划调用上限")
                        break
                elif len(budget.snapshot()) >= runtime["max_model_calls"]:
                    exhaust_budget("达到本任务累计模型调用上限")
                    break
                else:
                    payload = planner_payload(task, record, runtime, feedback)
                    planning = planner_pool.submit(budget.call, clients["large"], prompt, payload, "planner")
                    expand_while_running = False
                    planner_count += 1
                    event("planner_started", call_number=planner_count)
                    persist()
            active = list(jobs) + ([planning] if planning is not None else [])
            if active:
                wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
            else:
                record.update(status="failed", error="没有可运行的节点或规划动作")
    except KeyboardInterrupt:
        record.update(status="interrupted", error="用户中断；可使用--resume继续")
        event("interrupted")
        persist()
        raise
    except Exception as exc:
        record.update(status="failed", error="调度异常：" + type(exc).__name__)
        raise
    finally:
        # 已发送的模型调用无法撤回。等待其结束以保留实际结果和Token，不虚报取消。
        planner_pool.shutdown(wait=True, cancel_futures=True)
        pool.shutdown(wait=True, cancel_futures=True)
        for future, node in jobs.items():
            if future.done() and not future.cancelled():
                terminal_state = {k: record[k] for k in ("status", "error", "error_kind") if k in record}
                collect_execution(future, node)
                if terminal_state["status"] != "running":
                    for key in ("status", "error", "error_kind"):
                        record.pop(key, None)
                    record.update(terminal_state)
            elif node["status"] == "running":
                node["status"] = "pending"
        record["finished_at"] = now()
        record["resumable"] = record["status"] in ("paused", "interrupted", "running")
        persist()
    return record
