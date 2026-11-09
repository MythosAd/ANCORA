"""
generate_sgen_sft_v2.py

Build S-Gen SFT data from `seed_pool_verl_repaired.jsonl`.

Design:
1. Read repaired `(question, answer)` pairs from `seed_pool_verl_repaired.jsonl`.
2. Call the configured API provider.
3. Ask it to produce exactly one solver `<think>...</think>` block for the
   given `(question, answer)` pair.
4. Reconstruct the runtime-aligned S-Gen assistant response as:
   - one `<think>...</think>` block
   - one ```rust ... ``` block containing the original answer body
5. Save runtime-aligned chat messages for SFT.

Example:
    python qsv/generate_sgen_sft_v2.py ^
        --input qsv/data/seed_pool_verl_repaired.jsonl ^
        --output qsv/data/sgen_sft_v2.parquet ^
        --model gemini-3-flash-preview ^
        --max_samples 2000 ^
        --max_workers 8

Environment:
    GEMINI_API_KEY=...
    XIAOMIMO_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm


SGEN_SYSTEM_PROMPT = """You are an exceptionally strong Verus (Rust) expert.

Output contract:
- Output exactly one `<think>...</think>` block followed by exactly one ```rust ... ``` block.
- Output ONLY the missing function body, not the function signature.
- Do NOT repeat `fn ...`, `requires`, `ensures`, `verus!`, `use ...`, or `fn main`.
- Do NOT add prose before or after the single rust block.
- The rust block must end with the closing `}` for the function body.
- If you are unsure, still output a best-effort function body only.

Reasoning discipline inside `<think>`:
1. Identify the core objective, return contract, and proof obligations from the specification.
2. Decide the minimal solution shape actually needed:
   - direct expression
   - case split / branching
   - loop-based construction
   - proof-only / ghost-assisted reasoning
   - witness / search construction
3. List only the verifier obligations that are actually needed for that solution shape.
   Mention loop invariants, decreases, assertions, proof blocks, ghost state, or helper lemmas only if they are really needed.
4. Formulate a step-by-step body plan that matches the chosen solution shape.
5. Verify that the body-only answer is consistent with the specification and uses no unnecessary structure.
"""


SGEN_USER_TEMPLATE = """Consider the following Verus function specification:

```rust
{spec_code}
```

Your task is to complete THIS function by writing only the missing function body.
You may use conditionals, loops, invariants, assertions, proof blocks, ghost state, and local helper logic only when needed.
You must not create a new function or repeat the signature.
"""


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
RUST_BLOCK_RE = re.compile(r"```rust\s*(.*?)```", re.DOTALL | re.IGNORECASE)
SGEN_REQUIRED_LABELS = [
    "1. Core objective:",
    "2. Solution shape:",
    "3. Required verifier obligations:",
    "4. Body plan:",
    "5. Consistency check:",
]
PLACEHOLDER_PATTERNS = [
    re.compile(r"^\s*\.\.\.\s*$", re.IGNORECASE),
    re.compile(r"\b(todo|tbd|omitted)\b", re.IGNORECASE),
    re.compile(r"\bplaceholder(?:\s+text|\s+here)?\b", re.IGNORECASE),
]


PROVIDER_PRESETS = {
    "gemini": {
        "api_format": "openai-chat",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3-flash-preview",
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    },
    "deepseek": {
        "api_format": "openai-chat",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-reasoner",
        "env_keys": ["DEEPSEEK_API_KEY"],
    },
    "moonshot": {
        "api_format": "openai-chat",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2.5",
        "env_keys": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
    },
    "xiaomimo": {
        "api_format": "anthropic-messages",
        "base_url": "https://api.xiaomimimo.com/anthropic/v1",
        "model": "mimo-v2-flash",
        "env_keys": ["MIMO_API_KEY", "XIAOMIMO_API_KEY", "XIAOMI_MIMO_API_KEY", "XIAOMI_API_KEY"],
    },
    "modelscope": {
        "api_format": "openai-chat",
        "base_url": "https://api-inference.modelscope.cn/v1",
        "model": "Qwen/Qwen3-235B-A22B",
        "env_keys": ["MODELSCOPE_API_KEY"],
    },
}


def extract_think_block(text: str) -> str | None:
    m = THINK_RE.search(text or "")
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return None
    return f"<think>\n{inner}\n</think>"


def is_placeholder_text(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    return any(p.search(stripped) for p in PLACEHOLDER_PATTERNS)


def is_valid_sgen_think(think_block: str) -> bool:
    m = THINK_RE.search(think_block or "")
    body = m.group(1).strip() if m else ""
    if len(body) < 120 or is_placeholder_text(body):
        return False
    has_labels = all(label in body for label in SGEN_REQUIRED_LABELS)
    numbered = sum(1 for i in range(1, 6) if f"{i}." in body)
    return has_labels or numbered >= 4


def strip_rust_fence(text: str) -> str:
    text = (text or "").strip()
    m = RUST_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question and answer:
                rows.append(item)
    return rows


def sample_rows(
    rows: list[dict],
    max_samples: int,
    rng: random.Random,
    *,
    preserve_order: bool = False,
    offset: int = 0,
) -> list[dict]:
    rows = list(rows)
    if not preserve_order:
        rng.shuffle(rows)
    if offset > 0:
        rows = rows[offset:]
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _resolve_api_key(explicit_key: str, provider: str) -> str:
    if explicit_key:
        return explicit_key
    preset = PROVIDER_PRESETS.get(provider, {})
    for env_key in preset.get("env_keys", []):
        value = os.environ.get(env_key, "")
        if value:
            return value
    return ""


def _resolve_provider_defaults(provider: str, api_format: str, base_url: str, model: str) -> tuple[str, str, str]:
    preset = PROVIDER_PRESETS.get(provider, {})
    return (
        api_format or preset.get("api_format", "openai-chat"),
        base_url or preset.get("base_url", ""),
        model or preset.get("model", ""),
    )


def _extract_text_from_anthropic_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text") or "").strip()
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return ""


def _normalize_temperature_for_model(model: str, temperature: float) -> float:
    model_l = (model or "").lower()
    if model_l.startswith("kimi-") or model_l.startswith("moonshot-") or model_l.startswith("gemini-3"):
        return 1.0
    return temperature


def _reasoning_effort_for_model(model: str) -> str | None:
    model_l = (model or "").lower()
    if model_l.startswith("gemini-3"):
        return "high"
    return None


def provider_chat_completion(
    api_key: str,
    api_format: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
    timeout_s: int,
) -> tuple[str, str, dict]:
    if api_format == "openai-chat":
        url = f"{base_url.rstrip('/')}/chat/completions"
        effective_temperature = _normalize_temperature_for_model(model, temperature)
        reasoning_effort = _reasoning_effort_for_model(model)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "max_tokens": max_output_tokens,
        }
        if model != "deepseek-reasoner":
            payload["temperature"] = effective_temperature
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in response: {body}")
        message = (choices[0] or {}).get("message") or {}
        content = str(message.get("content") or "").strip()
        reasoning_content = str(message.get("reasoning_content") or "").strip()
        if not content and not reasoning_content:
            raise RuntimeError(f"Empty content in response: {body}")
        return content, reasoning_content, body

    if api_format == "anthropic-messages":
        url = f"{base_url.rstrip('/')}/messages"
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = _extract_text_from_anthropic_content(body.get("content"))
        if not content:
            raise RuntimeError(f"Empty anthropic-style content in response: {body}")
        return content, "", body

    raise ValueError(f"Unsupported api_format: {api_format}")


def generate_sgen_think(
    api_key: str,
    api_format: str,
    base_url: str,
    model: str,
    question: str,
    answer_body: str,
    max_output_tokens: int,
    temperature: float,
    timeout_s: int,
    retries: int,
) -> dict:
    user_prompt = build_generation_user_prompt(question, answer_body)
    feedback_suffix = ""
    for attempt in range(retries + 1):
        try:
            content, reasoning_content, raw_response = provider_chat_completion(
                api_key=api_key,
                api_format=api_format,
                base_url=base_url,
                model=model,
                system_prompt=SGEN_SYSTEM_PROMPT,
                user_prompt=user_prompt + feedback_suffix,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            think_source = ""
            source_used = ""
            if content:
                parsed = extract_think_block(content)
                if parsed is not None:
                    think_source = THINK_RE.search(parsed).group(1).strip()
                else:
                    think_source = content.strip()
                source_used = "content"

            think_block = f"<think>\n{think_source}\n</think>" if think_source else None
            if think_block is not None and is_valid_sgen_think(think_block):
                return {
                    "ok": True,
                    "think_block": think_block,
                    "think_text": think_source,
                    "source_used": source_used,
                    "raw_content": content,
                    "raw_reasoning_content": reasoning_content,
                    "wrapped_candidate": think_block,
                    "validation_error": "",
                    "raw_response": raw_response,
                }
            if attempt < retries:
                feedback_suffix = (
                    "\n\nYour previous analysis was invalid. Write five concrete numbered items. "
                    "Do not use placeholders like `...` or `TODO`."
                )
            else:
                think_preview = think_source[:1200] if think_source else content[:1200]
                return {
                    "ok": False,
                    "think_block": None,
                    "think_text": think_source,
                    "source_used": source_used,
                    "raw_content": content,
                    "raw_reasoning_content": reasoning_content,
                    "wrapped_candidate": think_block or "",
                    "validation_error": f"Invalid S-Gen think response: {think_preview}",
                    "raw_response": raw_response,
                }
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            if attempt >= retries:
                return {
                    "ok": False,
                    "think_block": None,
                    "think_text": "",
                    "source_used": "",
                    "raw_content": "",
                    "raw_reasoning_content": "",
                    "wrapped_candidate": "",
                    "validation_error": f"HTTP {e.code}: {detail}",
                    "raw_response": {"http_error": detail, "code": e.code},
                }
            time.sleep(1.0 + attempt)
        except urllib.error.URLError as e:
            if attempt >= retries:
                return {
                    "ok": False,
                    "think_block": None,
                    "think_text": "",
                    "source_used": "",
                    "raw_content": "",
                    "raw_reasoning_content": "",
                    "wrapped_candidate": "",
                    "validation_error": f"URL error: {e}",
                    "raw_response": {"url_error": str(e)},
                }
            time.sleep(1.0 + attempt)
        except Exception as e:
            if attempt >= retries:
                return {
                    "ok": False,
                    "think_block": None,
                    "think_text": "",
                    "source_used": "",
                    "raw_content": "",
                    "raw_reasoning_content": "",
                    "wrapped_candidate": "",
                    "validation_error": f"Unhandled error: {e}",
                    "raw_response": {},
                }
            time.sleep(0.5 + attempt)
    return {
        "ok": False,
        "think_block": None,
        "think_text": "",
        "source_used": "",
        "raw_content": "",
        "raw_reasoning_content": "",
        "wrapped_candidate": "",
        "validation_error": "No result after retries",
        "raw_response": {},
    }


def build_assistant_content(think_block: str, answer_body: str) -> str:
    return f"{think_block}\n\n```rust\n{answer_body.strip()}\n```"


def build_messages(question: str, assistant_content: str) -> list[dict]:
    user_content = SGEN_USER_TEMPLATE.format(spec_code=question.strip())
    return [
        {"role": "system", "content": SGEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def append_jsonl(path: str, row: dict, lock: threading.Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_generation_user_prompt(question: str, answer_body: str) -> str:
    return (
        SGEN_USER_TEMPLATE.format(spec_code=question.strip())
        + "\n\nBelow is a verified reference body for this same function.\n"
        + "Use the standard S-Gen output format:\n"
        + "- one `<think>...</think>` block\n"
        + "- followed by one ```rust ... ``` block containing the function body\n"
        + "The `<think>` should explain why the verified body satisfies the specification.\n\n"
        + "Verified reference body:\n```rust\n"
        + answer_body.strip()
        + "\n```"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="qsv/data/curated_sft/seed_pool_verl_repaired_canonicalized.jsonl")
    parser.add_argument("--output", default="qsv/data/sgen_sft_v2.parquet")
    parser.add_argument("--debug_jsonl", default="")
    parser.add_argument("--records_jsonl", default="")
    parser.add_argument("--provider", choices=["gemini", "deepseek", "moonshot", "xiaomimo", "modelscope"], default="gemini")
    parser.add_argument("--api_format", choices=["openai-chat", "anthropic-messages"], default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api_key", default="", help="Optional; falls back to provider-specific env vars")
    parser.add_argument("--base_url", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout_s", type=int, default=120)
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--preserve_order", action="store_true")
    args = parser.parse_args()

    api_format, base_url, model = _resolve_provider_defaults(args.provider, args.api_format, args.base_url, args.model)
    api_key = _resolve_api_key(args.api_key, args.provider)
    if not api_key:
        env_hint = ", ".join(PROVIDER_PRESETS.get(args.provider, {}).get("env_keys", [])) or "provider-specific env var"
        raise SystemExit(f"Missing API key for provider={args.provider}. Set one of: {env_hint}, or pass --api_key.")
    if not base_url or not model:
        raise SystemExit("Missing provider base_url/model. Pass --base_url and --model, or use a known provider preset.")

    rng = random.Random(args.seed)
    rows = sample_rows(
        load_rows(args.input),
        args.max_samples,
        rng,
        preserve_order=args.preserve_order,
        offset=max(0, args.offset),
    )
    records: list[dict] = []
    failures = 0
    write_lock = threading.Lock()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path = args.debug_jsonl or str(output_path.with_suffix(".debug.jsonl"))
    records_jsonl_path = args.records_jsonl or str(output_path.with_suffix(".records.jsonl"))
    Path(debug_path).write_text("", encoding="utf-8")
    Path(records_jsonl_path).write_text("", encoding="utf-8")

    def _task(item: dict) -> dict | None:
        question = str(item["question"]).strip()
        answer_body = strip_rust_fence(str(item["answer"]).strip())
        passthrough = {
            k: v
            for k, v in item.items()
            if k not in {"question", "answer"}
        }
        think_result = generate_sgen_think(
            api_key=api_key,
            api_format=api_format,
            base_url=base_url,
            model=model,
            question=question,
            answer_body=answer_body,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            retries=args.retries,
        )
        debug_row = {
            "question": question,
            "answer": answer_body,
            "ok": bool(think_result.get("ok")),
            "source_used": think_result.get("source_used", ""),
            "think_text": think_result.get("think_text", ""),
            "validation_error": think_result.get("validation_error", ""),
            "raw_content": think_result.get("raw_content", ""),
            "raw_reasoning_content": think_result.get("raw_reasoning_content", ""),
            "wrapped_candidate": think_result.get("wrapped_candidate", ""),
            "raw_response": think_result.get("raw_response", {}),
        }
        debug_row.update(passthrough)
        if not think_result.get("ok"):
            return {"debug_only": True, "debug_row": debug_row}
        think_block = str(think_result["think_block"])
        assistant_content = build_assistant_content(think_block, answer_body)
        record = {
            "messages": build_messages(question, assistant_content),
            "question": question,
            "answer": answer_body,
            "think": think_block,
            "assistant_response": assistant_content,
            "debug_row": debug_row,
        }
        record.update(passthrough)
        return record

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = [ex.submit(_task, item) for item in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating S-Gen SFT"):
            try:
                rec = fut.result()
                if rec is None:
                    failures += 1
                elif rec.get("debug_only"):
                    failures += 1
                    append_jsonl(debug_path, rec["debug_row"], write_lock)
                    tqdm.write(f"[FAIL] idx={rec['debug_row'].get('idx')} source={rec['debug_row'].get('source_used')}")
                else:
                    debug_row = rec.pop("debug_row")
                    append_jsonl(debug_path, debug_row, write_lock)
                    append_jsonl(records_jsonl_path, rec, write_lock)
                    records.append(rec)
                    tqdm.write(f"[OK] idx={rec.get('idx')} source={debug_row.get('source_used')}")
            except Exception as e:
                failures += 1
                tqdm.write(f"[WARN] generation failed: {e}")

    df = pd.DataFrame(records)
    df.to_parquet(args.output, index=False)
    print(f"Loaded {len(rows)} rows")
    print(f"Generated {len(records)} valid records ({failures} failed)")
    print(f"Saved to {args.output}")
    print(f"Saved debug traces to {debug_path}")
    print(f"Saved incremental records to {records_jsonl_path}")


if __name__ == "__main__":
    main()
