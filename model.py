"""统一模型接口：DeepSeek HTTP API、已有本地Qwen和逐任务调用预算。"""
from datetime import datetime, timezone
import copy
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def now():
    return datetime.now(timezone.utc).isoformat()


class ModelError(RuntimeError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


class BudgetExceeded(ModelError):
    pass


def parse_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("JSON包含重复字段")
            result[key] = value
        return result

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("JSON包含非有限数字")
        return result

    def constant(_):
        raise ValueError("JSON包含非有限数字")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)
    except json.JSONDecodeError:
        raise ValueError("输出不是完整的严格JSON") from None


def parse_model_json(text, repairs=None):
    """修复可确定的外层包装，JSON对象本身仍使用严格校验。"""
    if not isinstance(text, str):
        raise ValueError("模型JSON响应必须是字符串")
    text = text.strip()
    if text.startswith("\ufeff"):
        text = text.removeprefix("\ufeff").lstrip()
        if repairs is not None:
            repairs.append({"field": "json_wrapper", "repair": "removed_bom"})
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].casefold() in ("```json", "```") and lines[-1] == "```":
        text = "\n".join(lines[1:-1])
        if repairs is not None:
            repairs.append({"field": "json_wrapper", "repair": "removed_markdown_fence"})
    try:
        return parse_json(text)
    except ValueError as original:
        # 只接受外层说明文本包住的单一完整对象；不修复单引号、截断、重复键或NaN。
        start, end = text.find("{"), text.rfind("}")
        if start > 0 or (end >= 0 and end < len(text) - 1):
            if start >= 0 and end > start and "{" not in text[:start] and "}" not in text[end + 1:]:
                value = parse_json(text[start:end + 1])
                if not isinstance(value, dict):
                    raise original
                if repairs is not None:
                    repairs.append({"field": "json_wrapper", "repair": "removed_surrounding_text"})
                return value
        raise original


def resumable_model_error(exc):
    status = exc.details.get("http_status")
    return not isinstance(exc, BudgetExceeded) and (status is None or status in (402, 408, 429) or status >= 500)


def positive(value, name, maximum=100000):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name}必须是1至{maximum}的整数")


def nonnegative_number(value, name, maximum=600):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f"{name}必须是0至{maximum}的有限数字")


def load_config(path):
    path = Path(path).resolve()
    cfg = parse_json(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != {"env_file", "large", "small", "routing", "runtime"}:
        raise ValueError("model.json需要env_file、large、small、routing、runtime")
    if not isinstance(cfg["env_file"], str):
        raise ValueError("env_file必须是路径字符串")
    env_file = path.parent / cfg["env_file"]
    if cfg["env_file"] and env_file.is_file():
        allowed = {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "QWEN_API_KEY", "QWEN_BASE_URL"}
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() in allowed:
                value = value.strip()
                if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
    for role in ("large", "small"):
        model = cfg[role]
        if not isinstance(model, dict) or not isinstance(model.get("model"), str) or not model["model"]:
            raise ValueError(f"{role}需要模型名称")
        positive(model.get("max_tokens"), f"{role}.max_tokens", 8192)
        provider = model.get("provider")
        if provider == "api":
            base = os.path.expandvars(model.get("base_url", "")).rstrip("/")
            url = urllib.parse.urlsplit(base)
            loopback = url.hostname in ("localhost", "127.0.0.1", "::1")
            if not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("模型URL不得含密钥、用户信息或查询参数")
            if url.scheme != "https" and not (role == "small" and loopback and url.scheme == "http"):
                raise ValueError("仅本机小模型允许HTTP，其余接口必须HTTPS")
            if model.get("api_key_env") not in {"DEEPSEEK_API_KEY", "QWEN_API_KEY"}:
                raise ValueError("模型需要已允许的密钥变量名")
            positive(model.get("timeout_seconds"), f"{role}.timeout_seconds", 600)
            positive(model.get("max_concurrent", 1), f"{role}.max_concurrent", 8)
            model.setdefault("max_retries", 2)
            model.setdefault("retry_backoff_seconds", 1)
            model.setdefault("retry_max_backoff_seconds", 8)
            if type(model["max_retries"]) is not int or not 0 <= model["max_retries"] <= 5:
                raise ValueError(f"{role}.max_retries必须是0至5的整数")
            nonnegative_number(model["retry_backoff_seconds"], f"{role}.retry_backoff_seconds", 60)
            nonnegative_number(model["retry_max_backoff_seconds"], f"{role}.retry_max_backoff_seconds", 120)
            if model["retry_max_backoff_seconds"] < model["retry_backoff_seconds"]:
                raise ValueError(f"{role}.retry_max_backoff_seconds不能小于retry_backoff_seconds")
            token_limits = model.get("purpose_max_tokens", {})
            allowed_purposes = {"planner", "answer", "review", "final_check"}
            if not isinstance(token_limits, dict) or set(token_limits) - allowed_purposes:
                raise ValueError(f"{role}.purpose_max_tokens只允许planner、answer、review、final_check")
            for purpose, limit in token_limits.items():
                positive(limit, f"{role}.purpose_max_tokens.{purpose}", model["max_tokens"])
            model["base_url"] = base
        elif provider == "local_transformers" and role == "small":
            model_path = (path.parent / model.get("model_path", "")).resolve()
            if not (model_path / "config.json").is_file():
                raise ValueError("本地模型目录不存在或缺少config.json")
            model["model_path"] = str(model_path)
            positive(model.get("max_input_tokens"), "small.max_input_tokens", 32768)
            if type(model.get("load_in_4bit")) is not bool:
                raise ValueError("load_in_4bit必须为布尔值")
        else:
            raise ValueError(f"不支持的{role}模型provider")
    limits = {"max_workers": 8, "max_nodes": 64, "max_revisions": 5, "max_planner_calls": 256,
              "max_model_calls": 512, "max_task_seconds": 36000}
    runtime = cfg["runtime"]
    optional = {"final_check_mode", "auto_finish_mode", "max_batch_nodes", "max_inflight_tasks", "max_total_revisions"}
    if (not isinstance(runtime, dict) or not set(limits) <= set(runtime)
            or set(runtime) - set(limits) - optional):
        raise ValueError("runtime字段不完整")
    for key, maximum in limits.items():
        positive(cfg["runtime"][key], key, maximum)
    cfg["runtime"].setdefault("final_check_mode", "planner")
    cfg["runtime"].setdefault("auto_finish_mode", "safe")
    cfg["runtime"].setdefault("max_batch_nodes", 3)
    cfg["runtime"].setdefault("max_inflight_tasks", 4)
    cfg["runtime"].setdefault("max_total_revisions", 3)
    if cfg["runtime"]["final_check_mode"] not in ("planner", "always"):
        raise ValueError("final_check_mode必须是planner或always")
    if cfg["runtime"]["auto_finish_mode"] not in ("safe", "off"):
        raise ValueError("auto_finish_mode必须是safe或off")
    positive(cfg["runtime"]["max_batch_nodes"], "runtime.max_batch_nodes", 8)
    positive(cfg["runtime"]["max_inflight_tasks"], "runtime.max_inflight_tasks", 16)
    positive(cfg["runtime"]["max_total_revisions"], "runtime.max_total_revisions", 32)
    routing = cfg["routing"]
    required_routing = {"small_max_context_chars", "small_top_k", "small_max_dependencies", "fallback_to_large"}
    if not required_routing <= set(routing) or set(routing) - required_routing - {"small_route_mode", "small_max_top_k"}:
        raise ValueError("routing字段不完整")
    for key in ("small_max_context_chars", "small_top_k", "small_max_dependencies"):
        positive(routing[key], key)
    if type(routing["fallback_to_large"]) is not bool:
        raise ValueError("fallback_to_large必须为布尔值")
    routing.setdefault("small_route_mode", "cost")
    routing.setdefault("small_max_top_k", routing["small_top_k"])
    positive(routing["small_max_top_k"], "routing.small_max_top_k")
    if routing["small_max_top_k"] < routing["small_top_k"]:
        raise ValueError("routing.small_max_top_k不能小于small_top_k")
    if routing["small_route_mode"] not in ("cost", "parallel_only"):
        raise ValueError("small_route_mode必须是cost或parallel_only")
    return cfg


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ModelError("接口发生重定向，请检查基础地址")


class APIModel:
    def __init__(self, config):
        self.config = config
        self.slots = threading.BoundedSemaphore(config.get("max_concurrent", 1))

    def _retry_delay(self, attempt, headers=None):
        retry_after = headers.get("Retry-After") if headers is not None else None
        if retry_after is not None:
            try:
                return min(max(0, float(retry_after)), self.config.get("retry_max_backoff_seconds", 8))
            except (TypeError, ValueError):
                try:
                    when = parsedate_to_datetime(str(retry_after))
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    return min(max(0, (when - datetime.now(timezone.utc)).total_seconds()),
                               self.config.get("retry_max_backoff_seconds", 8))
                except (TypeError, ValueError, OverflowError):
                    pass
        base = self.config.get("retry_backoff_seconds", 1)
        maximum = self.config.get("retry_max_backoff_seconds", 8)
        # 小幅抖动避免多个并发任务在限流后同时重试。
        return min(maximum, base * (2 ** (attempt - 1))) * (0.8 + random.random() * 0.4)

    def complete(self, system, user, live=False, max_tokens=None):
        if not live:
            raise ModelError("调用模型需要--live")
        key = os.environ.get(self.config["api_key_env"])
        if not key:
            raise ModelError("缺少模型密钥环境变量：" + self.config["api_key_env"])
        # JSON模式接口要求消息中明确包含JSON一词；覆盖独立核验等简短提示词。
        if "json" not in (system + " " + user).casefold():
            system += "\nReturn exactly one valid JSON object."
        output_limit = max_tokens or self.config["max_tokens"]
        payload = {"model": self.config["model"], "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0, "max_tokens": output_limit,
            "response_format": {"type": "json_object"}}
        if self.config["model"].casefold().startswith("deepseek"):
            # 不区分大小写：端点上的模型名大小写不统一（如 DeepSeek-V4-Pro / deepseek-v4-pro）。
            payload["thinking"] = {"type": "disabled"}
        request_data = json.dumps(payload, ensure_ascii=False).encode()
        details = {"model": self.config["model"], "request_attempts": 0, "max_tokens": output_limit,
                   "usage": None, "queue_wait_seconds": 0, "service_seconds": 0,
                   "retry_sleep_seconds": 0, "attempt_errors": []}
        start = time.perf_counter()
        maximum_attempts = self.config.get("max_retries", 0) + 1

        def finish_timings():
            details["seconds"] = round(time.perf_counter() - start, 3)
            for name in ("queue_wait_seconds", "service_seconds", "retry_sleep_seconds"):
                details[name] = round(details[name], 3)

        for attempt in range(1, maximum_attempts + 1):
            details["request_attempts"] = attempt
            request = urllib.request.Request(self.config["base_url"] + "/chat/completions",
                data=request_data, headers={"Authorization": "Bearer " + key,
                "Content-Type": "application/json"}, method="POST")
            queue_start = time.perf_counter()
            try:
                with self.slots:
                    details["queue_wait_seconds"] += time.perf_counter() - queue_start
                    service_start = time.perf_counter()
                    try:
                        with urllib.request.build_opener(NoRedirect).open(
                                request, timeout=self.config["timeout_seconds"]) as response:
                            encoded = response.read()
                    finally:
                        details["service_seconds"] += time.perf_counter() - service_start
                body = parse_json(encoded.decode("utf-8"))
                choice = body["choices"][0]
                content = choice["message"]["content"]
                usage = body.get("usage")
                details["usage"] = None
                if isinstance(usage, dict):
                    mapped = {k: v for k, v in usage.items() if k in {
                        "prompt_tokens", "completion_tokens", "total_tokens",
                        "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"
                    } and type(v) is int and v >= 0}
                    # 兼容只回传 prompt_tokens_details.cached_tokens 的端点（如并行科技）：
                    # 拆成 hit/miss 两个字段，下游命中率口径保持不变。
                    prompt_details = usage.get("prompt_tokens_details")
                    cached = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
                    prompt_tokens = mapped.get("prompt_tokens")
                    if ("prompt_cache_hit_tokens" not in mapped and type(cached) is int and cached >= 0
                            and type(prompt_tokens) is int and prompt_tokens >= cached):
                        mapped["prompt_cache_hit_tokens"] = cached
                        mapped["prompt_cache_miss_tokens"] = prompt_tokens - cached
                    details["usage"] = mapped
                if isinstance(content, str):
                    details["raw_response"] = content
                finish_timings()
                if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
                    raise ModelError("模型响应为空或被截断", details)
                return details
            except urllib.error.HTTPError as exc:
                details["attempt_errors"].append({"attempt": attempt, "http_status": exc.code})
                retryable = exc.code in (408, 429) or exc.code >= 500
                if retryable and attempt < maximum_attempts:
                    delay = self._retry_delay(attempt, exc.headers)
                    details["retry_sleep_seconds"] += delay
                    time.sleep(delay)
                    continue
                details["http_status"] = exc.code
                finish_timings()
                raise ModelError(f"模型请求失败，HTTP {exc.code}", details) from None
            except ModelError as exc:
                finish_timings()
                raise ModelError(str(exc), exc.details or details) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                details["attempt_errors"].append({"attempt": attempt, "error_type": type(exc).__name__})
                if attempt < maximum_attempts:
                    delay = self._retry_delay(attempt)
                    details["retry_sleep_seconds"] += delay
                    time.sleep(delay)
                    continue
                finish_timings()
                raise ModelError("模型连接错误", details) from None
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                finish_timings()
                raise ModelError("模型响应格式错误", details) from None


class LocalModel:
    """只加载一次Qwen；串行使用GPU，与远程API并行。"""
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.model = None
        self.load_error = None

    def _load(self):
        if self.load_error:
            raise ModelError(self.load_error)
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            self.torch = torch
            if self.config["load_in_4bit"] and not torch.cuda.is_available():
                raise RuntimeError("4bit Qwen需要CUDA")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_path"], local_files_only=True)
            kwargs = {"local_files_only": True, "device_map": "auto", "torch_dtype": "auto"}
            if self.config["load_in_4bit"]:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(self.config["model_path"], **kwargs)
            self.model.eval()
        except Exception as exc:
            self.load_error = "本地Qwen加载失败（" + type(exc).__name__ + "），请检查CUDA、显存和本地依赖"
            raise ModelError(self.load_error) from None

    def complete(self, system, user, live=False):
        if not live:
            raise ModelError("调用本地模型也需要--live")
        start = time.perf_counter()
        details = {"model": self.config["model"], "request_attempts": 1, "usage": None,
                   "queue_wait_seconds": 0, "service_seconds": 0, "retry_sleep_seconds": 0}

        def finish_timings():
            details["seconds"] = round(time.perf_counter() - start, 3)
            for name in ("queue_wait_seconds", "service_seconds"):
                details[name] = round(details[name], 3)

        try:
            queue_start = time.perf_counter()
            with self.lock:
                details["queue_wait_seconds"] = time.perf_counter() - queue_start
                service_start = time.perf_counter()
                try:
                    self._load()
                    prompt = self.tokenizer.apply_chat_template([
                        {"role": "system", "content": system}, {"role": "user", "content": user}],
                        tokenize=False, add_generation_prompt=True)
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    length = inputs["input_ids"].shape[-1]
                    if length > self.config["max_input_tokens"]:
                        raise ModelError("小模型输入超过Token上限，转交大模型", details)
                    with self.torch.inference_mode():
                        outputs = self.model.generate(**inputs, max_new_tokens=self.config["max_tokens"],
                            do_sample=False, pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
                    generated = outputs[0][length:]
                    content = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                    count = len(generated)
                    details.update(raw_response=content,
                        usage={"prompt_tokens": length, "completion_tokens": count, "total_tokens": length + count})
                    eos = self.model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    if not content or (count >= self.config["max_tokens"] and int(generated[-1]) not in eos):
                        raise ModelError("本地模型响应为空或被截断", details)
                finally:
                    details["service_seconds"] = time.perf_counter() - service_start
            finish_timings()
            return details
        except ModelError as exc:
            finish_timings()
            raise ModelError(str(exc), exc.details or details) from None
        except Exception as exc:
            finish_timings()
            raise ModelError("本地Qwen推理失败（" + type(exc).__name__ + "）", details) from None


def make_clients(config):
    return {"large": APIModel(config["large"]),
            "small": LocalModel(config["small"]) if config["small"]["provider"] == "local_transformers"
            else APIModel(config["small"])}


class CallBudget:
    """规划、节点执行和升级共享预算；记录每次尝试，失败也保留。"""
    def __init__(self, maximum, previous=None):
        self.maximum = maximum
        self.calls = copy.deepcopy(previous or [])
        self.lock = threading.Lock()

    def call(self, client, system, payload, purpose, *, token_purpose=None):
        if token_purpose is None:
            token_purpose = ("planner" if purpose == "planner" else "review" if purpose.startswith("verify:")
                             else "final_check" if purpose.startswith("final_check:") else "answer")
        token_limits = client.config.get("purpose_max_tokens", {})
        output_limit = token_limits.get(token_purpose)
        with self.lock:
            if len(self.calls) >= self.maximum:
                raise BudgetExceeded("达到本任务模型调用上限")
            entry = {"purpose": purpose, "model": client.config["model"],
                     "token_purpose": token_purpose, "max_tokens": output_limit or client.config.get("max_tokens"),
                     "started_at": now(), "status": "running"}
            self.calls.append(entry)
        try:
            kwargs = {"max_tokens": output_limit} if output_limit is not None else {}
            response = client.complete(system, json.dumps(payload, ensure_ascii=False), live=True, **kwargs)
            with self.lock:
                entry.update(status="succeeded", response=response, finished_at=now())
            return response
        except ModelError as exc:
            with self.lock:
                entry.update(status="failed", error=str(exc), response=exc.details, finished_at=now())
            raise
        except Exception as exc:
            with self.lock:
                entry.update(status="failed", error=type(exc).__name__, finished_at=now())
            raise ModelError("模型调用异常（" + type(exc).__name__ + "）") from None

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.calls)
