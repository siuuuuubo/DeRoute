"""命令入口：读取MuSiQue、逐步拆解执行、保存和断点续跑。"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import itertools
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

from dataset import read_dataset
from decompose import fingerprint, load_prompt, run_task, workflow_signature, write_json
from model import ModelError, load_config, make_clients, now, parse_json

ROOT = Path(__file__).resolve().parent


def completed_records(path, successful_only=True):
    """修复中断写出的最后半行；完整日志中的错误则明确报错。"""
    completed = {}
    if not path.exists():
        return completed
    with path.open("r+b") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                value = parse_json(line.decode("utf-8"))
            except (ValueError, UnicodeError):
                if stream.read(1):
                    raise ValueError("工作流日志中间存在损坏记录")
                stream.truncate(start)
                break
            if not line.endswith(b"\n"):
                stream.seek(0, 2)
                stream.write(b"\n")
            completed[value["task_id"]] = value
    return {key: value for key, value in completed.items()
            if not successful_only or value["status"] == "succeeded"}


def task_checkpoint(checkpoint_dir, task_id):
    """任务ID不直接进入文件名，兼容任意数据集ID并避免路径穿越。"""
    return checkpoint_dir / (fingerprint(task_id)[:24] + ".json")


def checkpoint_records(checkpoint_dir):
    """读取并核对每任务检查点；并发任务互不覆盖。"""
    records = {}
    if not checkpoint_dir.exists():
        return records
    for path in sorted(checkpoint_dir.glob("*.json")):
        value = parse_json(path.read_text(encoding="utf-8"))
        task_id = value.get("task_id") if isinstance(value, dict) else None
        if not isinstance(task_id, str) or task_checkpoint(checkpoint_dir, task_id) != path:
            raise ValueError("任务检查点文件名或task_id不合法：" + path.name)
        records[task_id] = value
    return records


def append_record(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_selected(tasks, config, output, resume=False):
    import fcntl
    started_at = now()
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    legacy_checkpoint = output / "checkpoint.json"
    checkpoint_dir = output / "checkpoints"
    results = output / "workflows.jsonl"
    metrics_path = output / "run_metrics.json"
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("该输出目录已有运行进程") from None
        has_task_checkpoints = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.json"))
        if not resume and (legacy_checkpoint.exists() or has_task_checkpoints or results.exists()):
            raise ValueError("输出目录已有记录；使用--resume继续，或指定新的--output目录")
        prompt = load_prompt()
        signature = workflow_signature(config, prompt)
        prior_batch_wall = 0
        if resume and metrics_path.is_file():
            previous_metrics = parse_json(metrics_path.read_text(encoding="utf-8"))
            value = previous_metrics.get("cumulative_batch_wall_seconds",
                                         previous_metrics.get("batch_wall_seconds", 0))
            if type(value) not in (int, float) or value < 0:
                raise ValueError("run_metrics.json中的累计墙钟时间不合法")
            prior_batch_wall = value
        saved_records = completed_records(results, successful_only=False) if resume else {}
        logged_records = dict(saved_records)
        logged_successes = {key for key, value in saved_records.items() if value["status"] == "succeeded"}
        if resume:
            previous = parse_json(legacy_checkpoint.read_text(encoding="utf-8")) if legacy_checkpoint.exists() else None
            if previous:
                saved_records[previous["task_id"]] = previous
            # 新版每任务检查点比遗留的单文件检查点更新，发生重叠时应覆盖旧状态。
            saved_records.update(checkpoint_records(checkpoint_dir))
        completed = {key: value for key, value in saved_records.items() if value["status"] == "succeeded"}
        # 所有选中样本在开始模型调用前完成续跑兼容性检查。
        for task in tasks:
            saved = saved_records.get(task["id"])
            if saved and (saved["input_sha256"] != fingerprint(task) or saved["signature"] != signature):
                raise ValueError("已有记录的输入、配置或提示词不同，请指定新的--output目录")
        clients = make_clients(config)
        success = 0
        terminal = 0
        runnable = []
        for index, task in enumerate(tasks, 1):
            if task["id"] in completed and task["id"] in logged_successes:
                print(f"[{index}/{len(tasks)}] 已完成，跳过 {task['id']}", flush=True)
                success += 1
                continue
            prior = saved_records.get(task["id"])
            if prior and prior["status"] == "failed" and not prior.get("resumable", False):
                print(f"[{index}/{len(tasks)}] 已终止，保留失败结果 {task['id']}：{prior.get('error', '')}", flush=True)
                # 快照已落盘、JSONL 尚未追加时，也要补记终止结果。
                logged = logged_records.get(task["id"])
                if logged != prior:
                    append_record(results, prior)
                    logged_records[task["id"]] = prior
                terminal += 1
                continue
            # 断在checkpoint保存成功与JSONL追加之间时，直接补记，不重复请求模型。
            if prior and prior["status"] == "succeeded":
                append_record(results, prior)
                logged_records[task["id"]] = prior
                success += 1
            else:
                runnable.append((index, task, prior))

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        max_inflight = min(config["runtime"].get("max_inflight_tasks", 1), max(1, len(runnable)))

        def execute(item):
            index, task, prior = item
            print(f"[{index}/{len(tasks)}] {task['question']}", flush=True)
            checkpoint = task_checkpoint(checkpoint_dir, task["id"])
            return index, task, run_task(task, clients, config, prompt,
                lambda value: write_json(checkpoint, value), previous=prior)

        pending = {}
        items = iter(runnable)
        stop_submitting = False
        with ThreadPoolExecutor(max_workers=max_inflight, thread_name_prefix="task") as task_pool:
            def fill():
                while not stop_submitting and len(pending) < max_inflight:
                    try:
                        item = next(items)
                    except StopIteration:
                        break
                    pending[task_pool.submit(execute, item)] = item

            fill()
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    pending.pop(future)
                    _, task, record = future.result()
                    append_record(results, record)
                    logged_records[task["id"]] = record
                    print(f"  {task['id']} 状态：{record['status']}；耗时 {record['wall_seconds']} 秒", flush=True)
                    if record["status"] == "succeeded":
                        success += 1
                    elif record["status"] in ("paused", "interrupted"):
                        stop_submitting = True
                        print("检测到可恢复暂停；不再提交新任务，等待已在途任务保存后退出：" +
                              record.get("error", ""), flush=True)
                    else:
                        terminal += 1
                fill()

        elapsed = round(time.perf_counter() - started, 3)
        write_json(metrics_path, {
            "started_at": started_at, "finished_at": now(), "batch_wall_seconds": elapsed,
            "cumulative_batch_wall_seconds": round(prior_batch_wall + elapsed, 3),
            "selected_tasks": len(tasks), "succeeded_tasks": success, "terminal_failed_tasks": terminal,
            "max_inflight_tasks": max_inflight, "stopped_for_resumable_failure": stop_submitting})
        print(f"完成 {success}/{len(tasks)}；工作流：{results}")
        return 0 if success == len(tasks) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="DeRoute：MuSiQue逐步拆解与大小模型并行执行")
    parser.add_argument("command", nargs="?", default="show", choices=["show", "run", "decompose"])
    parser.add_argument("--input", type=Path, default=ROOT / "data_test/musique_ans_v1.0_dev_test.jsonl",
                        help="MuSiQue JSONL文件或目录")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=5, help="从头选择N条，默认5条")
    selection.add_argument("--all", action="store_true", help="处理输入中的所有样本")
    selection.add_argument("--id", action="append", dest="ids", help="指定样本id，可重复使用")
    parser.add_argument("--live", action="store_true", help="允许调用远程与本地模型")
    parser.add_argument("--resume", action="store_true", help="复用完成的节点与样本，继续未完成工作")
    parser.add_argument("--config", type=Path, default=ROOT / "model.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--workers", type=int, help="覆盖并行执行节点数；单GPU仍最多一个小模型请求")
    parser.add_argument("--task-workers", type=int, help="覆盖同时处理的任务数；默认读取max_inflight_tasks")
    parser.add_argument("--small-route-mode", choices=["cost", "parallel_only"],
                        help="cost优先节省API；parallel_only仅在任务内并行前沿使用小模型")
    parser.add_argument("--auto-finish-mode", choices=["safe", "off"],
                        help="safe允许终点预声明省去末尾planner；off用于消融对照")
    args = parser.parse_args(argv)
    try:
        if args.limit < 1:
            raise ValueError("--limit必须大于0")
        data = read_dataset(args.input)
        if args.command == "show":
            count = 0
            for task in data:
                count += 1
                if count <= args.limit:
                    print(task["id"] + "：" + task["question"])
            print(f"数据集有效：{count}条；仅含问题和候选段落的白名单字段。")
            print("运行示例：bash agent.sh run.py run --limit 5 --live")
            return 0
        if not args.live:
            raise ValueError("执行需要--live；默认show只读取数据")
        if args.ids:
            ids = set(args.ids)
            tasks = [t for t in data if t["id"] in ids]
            if {t["id"] for t in tasks} != ids:
                raise ValueError("有指定id不在输入数据集中")
        else:
            tasks = list(data if args.all else itertools.islice(data, args.limit))
        if not tasks:
            raise ValueError("没有选中的测试样本")
        output = args.output.resolve()
        if not output.is_relative_to(ROOT):
            raise ValueError("输出目录必须位于DeRoute项目内")
        config = load_config(args.config)
        if args.workers is not None:
            if not 1 <= args.workers <= 8:
                raise ValueError("--workers必须是1至8")
            config["runtime"]["max_workers"] = args.workers
        if args.task_workers is not None:
            if not 1 <= args.task_workers <= 16:
                raise ValueError("--task-workers必须是1至16")
            config["runtime"]["max_inflight_tasks"] = args.task_workers
        if args.small_route_mode is not None:
            config["routing"]["small_route_mode"] = args.small_route_mode
        if args.auto_finish_mode is not None:
            config["runtime"]["auto_finish_mode"] = args.auto_finish_mode
        return run_selected(tasks, config, output, args.resume)
    except (ValueError, ModelError, OSError) as exc:
        print("错误：" + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已中断；在途调用完成后已保存在checkpoints/，可加--resume续跑。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
