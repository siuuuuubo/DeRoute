"""增量DAG：一次追加一个节点，依赖来自子问题中的#k引用。"""
import re


def references(question):
    return {"n" + m for m in re.findall(r"#([1-9][0-9]*)\b", question)}


def canonicalize_node(node, existing):
    """将已声明依赖里的明确上游答案还原成#k，不猜测缺失的关系或答案。"""
    if not isinstance(node, dict) or not isinstance(node.get("question"), str) or not isinstance(node.get("depends_on"), list):
        return node
    result = dict(node)
    by_id = {n["id"]: n for n in existing}
    for key in node["depends_on"]:
        if not isinstance(key, str) or key in references(result["question"]):
            continue
        parent = by_id.get(key, {})
        if parent.get("status") != "succeeded":
            continue
        answer = parent["output"]["answer"]
        if answer.strip():
            result["question"] = re.sub(r"(?<!\w)" + re.escape(answer) + r"(?!\w)",
                                        lambda _: "#" + key[1:], result["question"], flags=re.IGNORECASE)
    return result


def normalize_planner_node(node, existing, *, revision=False):
    """修复可确定的协议字段；不猜实体，也不放宽图/时间/证据检查。"""
    if not isinstance(node, dict):
        raise ValueError("节点必须是对象")
    result = dict(node)
    if not revision:
        result.setdefault("id", f"n{len(existing) + 1}")
    result.setdefault("depends_on", [])
    result = canonicalize_node(result, existing)
    if isinstance(result.get("question"), str):
        refs = list(dict.fromkeys("n" + k for k in re.findall(r"#([1-9][0-9]*)\b", result["question"])))
        # 显式引用是执行器真实使用的输入，避免模型再重复维护一份依赖集合。
        if refs or not result["depends_on"]:
            result["depends_on"] = refs
    return result


def validate_temporal_scope(node, task, existing):
    """新增的年份限制须来自原题或依赖答案，不能任意挑选生涯中的某一年。"""
    if not isinstance(node, dict) or not isinstance(node.get("question"), str) or not isinstance(node.get("depends_on"), list):
        return
    allowed = task["question"]
    for n in existing:
        if n["id"] in node.get("depends_on", []) and n.get("status") == "succeeded":
            allowed += " " + n["output"]["answer"]
    def years(text):
        values = set(re.findall(r"\b(?:18|19|20)\d{2}\b", text))
        for start, end in re.findall(r"\b((?:18|19|20)\d{2})\s*[-–—/]\s*(\d{2})\b", text):
            values.add(str(int(start[:2]) * 100 + int(end) + (100 if int(end) < int(start[-2:]) else 0)))
        return values
    if years(node["question"]) - years(allowed):
        raise ValueError("子问题新增了原题和依赖答案未要求的年份限制；请保留原问题的时间范围")


def validate_fields(node):
    fields = {"id", "question", "operation", "depends_on", "expected_output", "reason"}
    if not isinstance(node, dict) or not fields <= set(node) or set(node) - fields - {"paragraph_indices"}:
        raise ValueError("新节点只能包含id、question、operation、depends_on、expected_output、reason，可选paragraph_indices")
    if "paragraph_indices" in node:
        pi = node["paragraph_indices"]
        if not isinstance(pi, list) or any(type(i) is not int for i in pi) or len(pi) != len(set(pi)):
            raise ValueError("paragraph_indices必须是不重复的整数列表")
    if not isinstance(node["id"], str) or not re.fullmatch(r"n[1-9][0-9]*", node["id"]):
        raise ValueError("节点id格式非法")
    for key in ("question", "expected_output", "reason"):
        if not isinstance(node[key], str) or not node[key].strip():
            raise ValueError(f"{key}必须是非空字符串")
    if node["operation"] not in ("lookup", "compare", "reason", "summarize"):
        raise ValueError("operation只能为lookup、compare、reason、summarize")
    deps = node["depends_on"]
    if not isinstance(deps, list) or any(not isinstance(x, str) for x in deps):
        raise ValueError("depends_on必须是节点id列表")
    if len(deps) != len(set(deps)) or set(deps) != references(node["question"]):
        raise ValueError("depends_on必须与question里的#k引用完全一致且不能重复")
    return deps


def validate_node(node, existing, max_nodes=12):
    deps = validate_fields(node)
    if len(existing) >= max_nodes:
        raise ValueError("达到最大节点数，请在已有结果中选择最终节点")
    if node["id"] != f"n{len(existing) + 1}":
        raise ValueError("节点id必须按n1、n2依次递增")
    by_id = {n["id"]: n for n in existing}
    if set(deps) - set(by_id):
        raise ValueError("只能引用已经创建的节点，禁止未来引用、自依赖和环")
    if any(by_id[d].get("status") in ("failed", "blocked") for d in deps):
        raise ValueError("不能依赖失败节点")
    if any(n["question"].strip().casefold() == node["question"].strip().casefold() for n in existing):
        raise ValueError("不能重复创建相同子问题")
    return node


def descendants_of(key, nodes):
    descendants = {key}
    while True:
        added = {n["id"] for n in nodes if set(n["depends_on"]) & descendants} - descendants
        if not added:
            return descendants - {key}
        descendants |= added


def revision_targets(nodes, max_revisions):
    by_id = {n["id"]: n for n in nodes}
    targets = []
    for n in nodes:
        descendants = [by_id[k] for k in descendants_of(n["id"], nodes)]
        eligible = n["status"] == "needs_revision" or (n["status"] == "succeeded" and any(
            d["status"] == "needs_revision" for d in descendants))
        if eligible and n.get("revision_count", 0) < max_revisions and not any(
                d["status"] == "running" for d in descendants):
            targets.append(n["id"])
    return targets


def validate_revision(node, existing, max_revisions):
    deps = validate_fields(node)
    by_id = {n["id"]: n for n in existing}
    old = by_id.get(node["id"])
    if old is None or node["id"] not in revision_targets(existing, max_revisions):
        raise ValueError("只能修订待恢复节点或失败链上的成功祖先，且后代不能正在运行")
    if old.get("revision_count", 0) >= max_revisions:
        raise ValueError("达到该节点修订次数上限")
    if set(deps) - set(by_id) or node["id"] in deps:
        raise ValueError("修订不能自依赖或引用尚未创建的节点")
    if any(by_id[d]["status"] in ("failed", "needs_revision", "blocked") for d in deps):
        raise ValueError("修订不能依赖尚待修复的节点")
    if any(n["id"] != node["id"] and n["question"].strip().casefold() == node["question"].strip().casefold()
           for n in existing):
        raise ValueError("修订不能复制已有子问题")
    # 大模型已经尝试过同一查询时，仅改 reason 不会给执行器增加新信息。
    if old.get("status") == "needs_revision" and any(a.get("role") == "large" for a in old.get("attempts", [])):
        fields = ("question", "operation", "depends_on", "paragraph_indices", "expected_output")
        if all(node.get(k) == old.get(k) for k in fields):
            raise ValueError("修订没有新增信息：请改变问题、操作、依赖、证据或输出要求，或abort；仅改reason不会重新执行")
    # 可引用后来新增的辅助节点，但必须检查整个图，而不是只检查编号顺序。
    levels([node if n["id"] == node["id"] else n for n in existing])
    return node


def ready_nodes(nodes):
    complete = {n["id"] for n in nodes if n["status"] == "succeeded"}
    return [n for n in nodes if n["status"] == "pending" and set(n["depends_on"]) <= complete]


def validate_finish(final_node, nodes):
    by_id = {n["id"]: n for n in nodes}
    if not isinstance(final_node, str) or final_node not in by_id:
        raise ValueError("final_node必须是已有节点")
    seen, pending = set(), [final_node]
    while pending:
        key = pending.pop()
        if key not in seen:
            seen.add(key)
            pending.extend(by_id[key]["depends_on"])
    if any(by_id[k]["status"] != "succeeded" for k in seen):
        raise ValueError("最终依赖链的所有节点执行成功后才能finish；待修订节点请revise，在途节点请wait")
    unused = [n for n in nodes if n["id"] not in seen]
    if any(n["status"] in ("pending", "running") for n in unused):
        raise ValueError("仍有未执行成功的在途分支，不能finish")
    # 已被完整新链替代的失败旁支保留在日志中，不要求再制造一个无意义依赖来合并。
    if any(n["status"] != "needs_revision" for n in unused):
        raise ValueError("存在未贡献到最终答案的分支，请增加合并节点")
    return by_id[final_node]["output"]["answer"]


def resolve_inputs(node, nodes):
    by_id = {n["id"]: n for n in nodes}
    upstream = {}
    for key in node["depends_on"]:
        parent = by_id[key]
        if parent["status"] != "succeeded":
            raise ValueError("依赖结果尚未就绪")
        upstream[key] = parent["output"]
    question = re.sub(r"#([1-9][0-9]*)\b", lambda m: upstream["n" + m[1]]["answer"], node["question"])
    return question, upstream


def levels(nodes):
    remaining = {n["id"]: set(n["depends_on"]) for n in nodes}
    layers = []
    while remaining:
        ready = [key for key, deps in remaining.items() if not deps]
        if not ready:
            raise ValueError("图中存在环或悬空引用")
        layers.append(ready)
        remaining = {k: deps - set(ready) for k, deps in remaining.items() if k not in ready}
    return layers
