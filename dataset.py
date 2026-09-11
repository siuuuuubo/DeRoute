"""MuSiQue JSONL 接口：只向运行流程提供问题与候选段落。"""
from pathlib import Path

from model import parse_json


def read_dataset(path):
    """逐行读取；白名单投影确保原始文件中的答案/支持标记也不会泄漏。"""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise ValueError("数据目录中没有JSONL文件")
    seen = set()
    for file in files:
        with file.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = parse_json(line)
                    if not isinstance(value, dict):
                        raise ValueError("记录必须是对象")
                    for key in ("id", "question"):
                        if not isinstance(value.get(key), str) or not value[key].strip():
                            raise ValueError(f"{key}必须是非空字符串")
                    if value["id"] in seen:
                        raise ValueError("重复的样本id")
                    raw = value.get("paragraphs")
                    if not isinstance(raw, list) or not raw:
                        raise ValueError("paragraphs必须是非空列表")
                    paragraphs, indices = [], set()
                    for para in raw:
                        if not isinstance(para, dict) or type(para.get("idx")) is not int:
                            raise ValueError("段落需要整数idx")
                        if para["idx"] < 0 or para["idx"] in indices:
                            raise ValueError("段落idx为负数或重复")
                        if any(not isinstance(para.get(k), str) for k in ("title", "paragraph_text")):
                            raise ValueError("段落需要title和paragraph_text字符串")
                        indices.add(para["idx"])
                        paragraphs.append({k: para[k] for k in ("idx", "title", "paragraph_text")})
                    seen.add(value["id"])
                    yield {"id": value["id"], "question": value["question"], "paragraphs": paragraphs}
                except ValueError as exc:
                    raise ValueError(f"{file.name}第{number}行：{exc}") from None
