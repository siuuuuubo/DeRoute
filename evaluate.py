"""离线答案评测。只读取已完成的日志；不调用模型，也不向运行流程传入标准答案。"""
import argparse
from collections import Counter
from pathlib import Path
import re
import string

from model import parse_json

ROOT = Path(__file__).resolve().parent


def normalize_answer(value):
    text = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction, answers):
    pred = normalize_answer(prediction)
    exact, best_f1 = 0, 0.0
    for answer in answers:
        gold = normalize_answer(answer)
        exact = max(exact, int(pred == gold))
        p, g = pred.split(), gold.split()
        if not p or not g:
            f1 = float(p == g)
        else:
            overlap = sum((Counter(p) & Counter(g)).values())
            f1 = 2 * overlap / (len(p) + len(g))
        best_f1 = max(best_f1, f1)
    return exact, best_f1


def read_records(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    record = parse_json(line)
                    if not isinstance(record, dict):
                        raise ValueError("记录必须是对象")
                    yield record
                except ValueError as exc:
                    raise ValueError(f"{Path(path).name}第{number}行：{exc}") from None


def evaluate(predictions_path, gold_path):
    # 同一样本重复运行取最后一次，包括失败；不能挑选历史最好结果。
    predictions = {}
    for record in read_records(predictions_path):
        if not isinstance(record.get("task_id"), str) or not isinstance(record.get("status"), str):
            raise ValueError("预测日志缺少task_id/status")
        predictions[record["task_id"]] = record
    if not predictions:
        raise ValueError("预测日志为空")
    gold = {}
    for record in read_records(gold_path):
        key = record.get("id")
        if key not in predictions:
            continue
        if key in gold:
            raise ValueError("标准数据中存在重复id")
        answer, aliases = record.get("answer"), record.get("answer_aliases", [])
        if not isinstance(answer, str) or not answer.strip() or not isinstance(aliases, list) or any(
                not isinstance(a, str) for a in aliases):
            raise ValueError("评测需要非空标准答案及合法别名列表")
        gold[key] = [answer] + [a for a in aliases if a.strip()]
    if set(predictions) != set(gold):
        raise ValueError("部分预测id未在标准数据中找到；请检查数据版本")
    rows = []
    for key, record in predictions.items():
        prediction = record.get("answer", "")
        completed = record["status"] == "succeeded"
        if completed and (not isinstance(prediction, str) or not prediction.strip()):
            raise ValueError("成功记录缺少非空答案")
        exact, f1 = answer_scores(prediction, gold[key]) if completed else (0, 0.0)
        rows.append({"task_id": key, "status": record["status"], "prediction": prediction if completed else "",
                     "gold_answer": gold[key][0], "exact_match": exact, "f1": round(f1, 6),
                     "answer_support": record.get("answer_support", "unrecorded"),
                     "error": record.get("error")})
    count = len(rows)
    return {"evaluated": count, "completed": sum(r["status"] == "succeeded" for r in rows),
            "correct": sum(r["exact_match"] for r in rows),
            "exact_match": sum(r["exact_match"] for r in rows) / count,
            "f1": sum(r["f1"] for r in rows) / count,
            "scope": "仅评测日志中的不同样本，失败计0分，重复样本取最后一次；不代表全测试集准确率",
            "normalization": "英文小写、删除ASCII标点、去除a/an/the、合并空白；别名取最高分",
            "results": rows}


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
        result = evaluate(args.predictions, args.gold)
        # 仅在独立离线命令中导入写盘工具；推理入口不导入本模块。
        from decompose import write_json
        write_json(output, result)
        print(f"评测 {result['evaluated']} 条，完成 {result['completed']} 条，正确 {result['correct']} 条；"
              f"EM={result['exact_match']:.2%}，F1={result['f1']:.2%}")
        print(output)
        return 0
    except (ValueError, OSError) as exc:
        print("评测失败：" + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
