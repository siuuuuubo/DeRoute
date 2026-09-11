"""对比 baseline 与新版 100 题的分层评测 + 归因 + 成本（耗时/调用数）。

用法：
  python compare_runs.py --baseline outputs/baseline_100/evaluation_layers.json \\
      --new outputs/exp100_v2/evaluation_layers.json \\
      --baseline_workflows outputs/baseline_100/workflows.jsonl \\
      --new_workflows outputs/exp100_v2/workflows.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from model import parse_model_json

METRICS = [
    ("evaluated", "评测样本数"),
    ("completed", "流程完成数"),
    ("final_exact_match", "端到端 EM"),
    ("graph_valid_rate", "图合法率"),
    ("node_recall", "Node-Recall"),
    ("node_precision", "Node-Precision"),
    ("edge_recall", "Edge-Recall"),
    ("edge_precision", "Edge-Precision"),
    ("mean_gold_nodes", "平均标准子题"),
    ("mean_pred_nodes", "平均预测子题"),
    ("under_decomposition_rate", "漏拆率"),
    ("over_decomposition_rate", "过拆率"),
]

ATTRIBUTIONS = ["correct", "decomposition", "execution", "synthesis",
                "no_answer:abort", "no_answer:budget", "no_answer:revision_exhausted", "no_answer:other"]


def workload(path):
    """汇总去重任务耗时与调用；并发运行另读取批次实际墙钟。"""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tid = r.get("task_id")
            records[tid] = r  # 保留最后一次尝试
    wall = sum(r.get("wall_seconds", 0) for r in records.values())
    calls = Counter()
    roles = Counter()
    usage = Counter()
    transport = Counter()
    wait_calls = 0
    for r in records.values():
        for c in r.get("calls", []):
            purpose = c.get("purpose", "")
            role = "planner" if purpose == "planner" else purpose.split(":", 1)[0]
            calls[role] += 1
            roles["small" if purpose.endswith(":small") else "large"] += 1
            response = c.get("response") or {}
            attempts = response.get("request_attempts", 1)
            request_role = "small" if purpose.endswith(":small") else "large"
            transport[request_role + "_requests"] += attempts if type(attempts) is int and attempts > 0 else 1
            for key in ("queue_wait_seconds", "service_seconds", "retry_sleep_seconds"):
                value = response.get(key, 0)
                if type(value) in (int, float) and value >= 0:
                    transport[key] += value
            if not purpose.endswith(":small"):
                for key, value in (response.get("usage") or {}).items():
                    if key in ("prompt_tokens", "completion_tokens", "total_tokens",
                               "prompt_cache_hit_tokens", "prompt_cache_miss_tokens") and type(value) is int:
                        usage[key] += value
            if purpose == "planner":
                try:
                    wait_calls += parse_model_json(c["response"]["raw_response"]).get("action") == "wait"
                except (KeyError, ValueError, TypeError, AttributeError):
                    pass
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens",
                  "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
    usage_values = {key: usage[key] for key in usage_keys}
    cache_total = usage_values["prompt_cache_hit_tokens"] + usage_values["prompt_cache_miss_tokens"]
    metrics_path = Path(path).with_name("run_metrics.json")
    batch_wall = None
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            value = metrics.get("cumulative_batch_wall_seconds", metrics.get("batch_wall_seconds"))
            if type(value) in (int, float) and value >= 0:
                batch_wall = round(value, 1)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return {"tasks": len(records), "wall_seconds": round(wall, 1),
            "batch_wall_seconds": batch_wall,
            "planner_calls": calls.get("planner", 0),
            "execute_calls": sum(v for k, v in calls.items() if k != "planner"),
            "model_calls": sum(calls.values()), "large_calls": roles["large"], "small_calls": roles["small"],
            "large_requests": transport["large_requests"], "small_requests": transport["small_requests"],
            "queue_wait_seconds": round(transport["queue_wait_seconds"], 1),
            "service_seconds": round(transport["service_seconds"], 1),
            "retry_sleep_seconds": round(transport["retry_sleep_seconds"], 1),
            "verify_calls": calls["verify"], "final_check_calls": calls["final_check"],
            "planner_wait_calls": wait_calls,
            "planner_rejections": sum(e["type"] == "planner_rejected" for r in records.values() for e in r.get("events", [])),
            **usage_values,
            "prompt_cache_hit_rate": usage_values["prompt_cache_hit_tokens"] / cache_total if cache_total else 0}


def fmt_ratio(v):
    return f"{v:.2%}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--baseline_workflows", type=Path, required=True)
    ap.add_argument("--new_workflows", type=Path, required=True)
    args = ap.parse_args(argv)

    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    base_w = workload(args.baseline_workflows)
    new_w = workload(args.new_workflows)

    print("=" * 78)
    print(f"{'指标':<18}{'baseline':>12}{'新版':>12}{'变化':>12}")
    print("-" * 78)
    for key, label in METRICS:
        b, n = base.get(key, 0), new.get(key, 0)
        if key in ("final_exact_match", "graph_valid_rate", "node_recall", "node_precision",
                   "edge_recall", "edge_precision", "under_decomposition_rate", "over_decomposition_rate"):
            b_s, n_s, delta = fmt_ratio(b), fmt_ratio(n), f"{n - b:+.2%}"
        else:
            b_s, n_s, delta = f"{b:g}", f"{n:g}", f"{n - b:+.2g}"
        print(f"{label:<18}{b_s:>12}{n_s:>12}{delta:>12}")

    print("-" * 78)
    for key, label in [("tasks", "去重任务数"), ("wall_seconds", "各任务耗时之和(s)"),
                       ("model_calls", "全部模型调用数"), ("large_calls", "大模型调用数"),
                       ("small_calls", "小模型调用数"), ("large_requests", "大模型HTTP请求数"),
                       ("small_requests", "小模型实际请求数"), ("planner_calls", "规划调用数"),
                       ("planner_wait_calls", "规划wait调用数"), ("planner_rejections", "非法规划动作数"),
                       ("execute_calls", "非规划调用数"), ("verify_calls", "推断复核调用数"),
                       ("final_check_calls", "独立终检调用数"),
                       ("queue_wait_seconds", "请求排队时间(s)"),
                       ("service_seconds", "模型服务时间(s)"),
                       ("retry_sleep_seconds", "重试退避时间(s)"),
                       ("prompt_tokens", "大模型输入token"),
                       ("prompt_cache_hit_tokens", "缓存命中token"),
                       ("prompt_cache_miss_tokens", "缓存未命中token")]:
        b, n = base_w[key], new_w[key]
        print(f"{label:<18}{b:>12}{n:>12}{n - b:>+12.1f}")
    b, n = base_w["batch_wall_seconds"], new_w["batch_wall_seconds"]
    if b is not None or n is not None:
        b_s, n_s = ("—" if b is None else f"{b:g}"), ("—" if n is None else f"{n:g}")
        delta = "—" if b is None or n is None else f"{n - b:+.1f}"
        print(f"{'批次实际墙钟(s)':<18}{b_s:>12}{n_s:>12}{delta:>12}")
    print(f"{'缓存命中率':<18}{base_w['prompt_cache_hit_rate']:>11.2%}{new_w['prompt_cache_hit_rate']:>12.2%}"
          f"{new_w['prompt_cache_hit_rate'] - base_w['prompt_cache_hit_rate']:>+12.2%}")

    print("=" * 78)
    print("归因分布（占比）：")
    print(f"{'归因':<24}{'baseline':>12}{'新版':>12}")
    base_share = base.get("attribution_share", {})
    new_share = new.get("attribution_share", {})
    for key in ATTRIBUTIONS:
        b, n = base_share.get(key, 0), new_share.get(key, 0)
        if b or n:
            print(f"{key:<24}{b:>12.2%}{n:>12.2%}")

    # 每条样本层面的正确性变化（旧错新对 / 旧对新错）
    base_rows = {r["task_id"]: r for r in base.get("results", [])}
    new_rows = {r["task_id"]: r for r in new.get("results", [])}
    improved = [t for t in base_rows if t in new_rows and not base_rows[t]["final_correct"] and new_rows[t]["final_correct"]]
    regressed = [t for t in base_rows if t in new_rows and base_rows[t]["final_correct"] and not new_rows[t]["final_correct"]]
    print("=" * 78)
    print(f"样本级正确性翻转：旧错→新对 {len(improved)} 条；旧对→新错 {len(regressed)} 条")
    if improved:
        print("  改进:", ", ".join(improved))
    if regressed:
        print("  回退:", ", ".join(regressed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
