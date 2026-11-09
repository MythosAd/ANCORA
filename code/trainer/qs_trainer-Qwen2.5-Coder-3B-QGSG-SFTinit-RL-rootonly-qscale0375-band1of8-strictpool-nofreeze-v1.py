# Copyright 2024-2026 - Question Solver Trainer for verl
# Implements a 2-step Question-Solver generation loop with coupled rewards

import os
import uuid
import math
import json
import re
import random
import glob
import concurrent.futures
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.distributed
from transformers import PreTrainedTokenizer
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict

def to_list_of_ints(token_output):
    """Ensure that the output of apply_chat_template is a simple flat list of ints."""
    if hasattr(token_output, "data") or isinstance(token_output, dict):
        token_output = token_output["input_ids"]
    if hasattr(token_output, "tolist"):
        token_output = token_output.tolist()
    if isinstance(token_output, list) and len(token_output) > 0 and isinstance(token_output[0], list):
        token_output = token_output[0]
    return list(token_output)

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.metric_utils import reduce_metrics, compute_timing_metrics
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def _pick_existing_path(*candidates: str) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0] if candidates else ""


@dataclass
class QFilterResult:
    valid: bool
    score: float
    compiler_msg: str
    stitched_code: str
    filter_stage: str
    filter_reason: str


@dataclass
class QParseResult:
    valid: bool
    spec: str
    reject_reason: str = ""
    format_penalty: float = 1.0
    raw_char_len: int = 0
    post_char_len: int = 0
    think_char_len: int = 0
    code_block_count: int = 0
    had_placeholder: bool = False
    had_prompt_echo: bool = False
    spec_char_len: int = 0
    spec_line_count: int = 0
    fn_count: int = 0
    salvaged_primary_fn: bool = False


@dataclass
class QGenResult:
    gen_output: object          # DataProto
    turn1_prompts: list
    turn1_seeds: list
    generated_qs: list
    generated_specs_masked: list
    generated_q_bodies: list
    q_valid_flags: list
    q_err_scores: list
    sim_scores: list
    q_err_compiler_outs: list
    q_stitched_codes: list
    q_parse_results: list
    filter_stages: list
    filter_reasons: list
    q_filter_results: list
    heuristic_rejects: int
    compiler_rejects: int
    format_rejects: int
    heuristic_postcompiler_rejects: int
    bad_ensures_rejects: int
    candidate_count: int


def _to_object_array(items) -> np.ndarray:
    seq = list(items)
    arr = np.empty(len(seq), dtype=object)
    if seq:
        arr[:] = seq
    return arr


try:
    from verus.verify import verify_code
except ImportError:
    # Compatible fallback for local development when Verus is not installed
    def verify_code(c, timeout=30): return False, "", "Verifier Missing"


# qs_prompts.py
# ====================================================================
# [Prompts Configuration]
# ====================================================================


Q_GEN_FREEFORM_SYSTEM_PROMPT = """You are an expert Verus (Rust) programmer specializing in formal verification.
Your job is to generate one new curriculum question, not a solution. 
You are driven by verification curiosity: like a human learner, you naturally seek the "Zone of Proximal Development". If a problem feels too complex or dense, you instinctively probe downward (decompose, extract lemmas, simplify). If it feels foundational or solved, you instinctively probe upward (generalize, strengthen postconditions, compose).

Output contract:
- Output exactly one `<think>...</think>` block followed by exactly one ```rust ... ``` block.
- The rust block must contain exactly one target item.
- Allowed target item forms are:
  - `fn ... ensures ... {`
  - `spec fn ... {`
  - `proof fn ... {`
- Prefer canonical minimal headers (`fn`, `spec fn`, `proof fn`) over visibility modifiers such as `pub` or `open`.
- Do not include a function body.
- Do not emit extra helper items, impl blocks, structs, enums, tests, or `fn main`.
- Do not output prose outside `<think>...</think>` and the single rust block.

Reasoning discipline inside `<think>`:
Inside `<think>`, use exactly these seven labels in this order:
1. Seed core:
2. Curiosity assessment:
3. Solver-skill focus:
4. Exploration rationale:
5. Next-step child choice:
6. Check:

Requirements:
- Keep each section short, concrete, and verifier-aware.
- The child must be one plausible next-step question, not a far jump.
- The prompt will specify a curriculum direction.
- The chosen child must introduce exactly one new reusable solver obligation, not just a semantically related variation.
- That new obligation should map to a concrete S-Gen skill family, such as:
  - witness / search construction
  - stronger loop invariant
  - absence-case / Option-case handling
  - stronger quantified relation
  - one helper lemma / proof step
  - constructor exactness
- Prefer children whose solution shape stays close to the parent, so that learning the child teaches a proof skill reusable on the parent or a harder sibling.
- `Check` must review whether steps 2-6 are correct and revise them if needed.
- The six sections must absorb this reasoning discipline, not run in parallel:
  - `Seed core` = identify the core objective and current verification difficulty of the seed in plain language.
  - `Curiosity assessment` = Evaluate the implicit difficulty. Decide whether to probe DOWN (the seed is too dense; we need to isolate a subproblem/lemma) or probe UP (the seed is foundational; we need to add a new constraint/generalization).
  - `Solver-skill focus` = name the ONE reusable solver obligation that this new direction will teach.
  - `Exploration rationale` = explain why this specific upward or downward step is the most natural, manageable next epistemic leap for a verifier.
  - `Next-step child choice` = formulate the exact target item based on the exploration.
  - `Check` = verify the full reasoning chain for consistency. Ensure the child is neither a trivial mutation nor an impossible leap.

Hard bans:
- No meta commentary about the prompt, the user, or your role.
- No code before `<think>`.
- No text after the single rust block.
- No rename-only mutation.
- No child whose extra property is already implied by the seed.
- No child whose new difficulty cannot be tied to a reusable S-Gen proof or implementation skill.
"""


Q_GEN_FREEFORM_CURRICULUM_TEMPLATE = """
Below is a seed problem from the existing pool:

```rust
{seed_code}
```

Task:
- Generate one new Verus target item driven by verification curiosity.
- Analyze the seed: if it's too complex or dense, naturally probe downward (e.g., isolate a stepping-stone, extract a lemma). If it's well-understood, naturally probe upward (e.g., introduce the next logical constraint, generalize).
- Make one focused semantic change rather than a rename or superficial mutation.
- Keep the child algorithmic, verifier-friendly, and close enough to remain a learnable next step.
- Make the new difficulty correspond to one reusable solver skill, not a vague increase in complexity.
- If the child changes the semantics, rename the function accordingly.
"""

# 3. Q-Gen zero-shot template (deprecated and no longer used)

# 4. S-Gen system prompt
S_GEN_SYSTEM_PROMPT = """You are an exceptionally strong Verus (Rust) expert.

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

# 5. S-Gen task template ({spec_code} is filled with the extracted spec)
SOLVER_USER_MSG_TEMPLATE = """Consider the following Verus function specification:

```rust
{spec_code}
```

Your task is to complete THIS function by writing only the missing function body.
You may use conditionals, loops, invariants, assertions, proof blocks, ghost state, and local helper logic only when needed.
You must not create a new function or repeat the signature.
"""

# 6. Fallback 
# (Removed VAL_HOLE_SYSTEM_PROMPT to unify with strictly-formatted S_GEN_SYSTEM_PROMPT)

# ====================================================================
# [Utils] 
# ====================================================================

def classify_verus_error(v_stderr: str, v_stdout: str, stitched_code: str = "") -> tuple:
    """Unified Verus compiler output classifier.
    Returns (error_category: str, compiler_out: str).

    Error categories (used for logging/diagnostics only, NOT as reward signals):
      'empty'            Code was empty or whitespace-only
      'timeout'          Verification timed out
      'syntax_error'     Hard Rust/Verus parse or type errors
      'other_failure'    Compilation failed, no specific pattern matched
      'assertion_failed' Valid Verus structure, but proof obligations not met

    Note: S-Gen uses pure binary rewards (1.0 for verified, 0.0 otherwise).
    These categories are for filter gating and logging, not gradient computation.
    """
    compiler_out = (v_stderr or v_stdout or "").strip()[:2000]
    out_lower = compiler_out.lower()

    if compiler_out == "timeout":
        return "timeout", "timeout"
    if not stitched_code or not stitched_code.strip():
        return "empty", ""

    # 1. Hard syntax/type errors must be checked FIRST.
    # Models often hallucinate math symbols or trigger type mismatches in ensures/requires lines.
    # Checking these patterns first prevents them from being misclassified as assertion_failed.
    #
    # IMPORTANT: Rust compilation errors (e.g. E0308) go to v_stdout, while Verus verification
    # failures go to v_stderr. When both streams are non-empty, `compiler_out` only captures
    # v_stderr (due to `or` semantics), so hard error patterns in v_stdout would be missed.
    # Fix: scan hard patterns across both streams combined.
    hard_check_lower = (out_lower + " " + (v_stdout or "")[:2000].lower())
    hard_error_patterns = [
        "error[e",                   # Rust error codes (E0308, E0762, etc.)
        "unknown start of token",    # Hallucinated unicode symbols / backtick tokens
        "unrecognized token",
        "parser error",
        "mismatched types",
        "cannot find value",
        "no implementation for",
        "unexpected end of input",   # Unclosed brace / incomplete code
        "unclosed delimiter",        # Mismatched delimiters
        "mismatched closing delimiter",
        "unexpected closing delimiter",
        "unexpected token",
        "expected curly braces",
        "expected `",                # E.g. expected `;` or expected `!`
        "this block would be parsed as",
        "panic is not supported",
        "unterminated character",    # E0762 unterminated char/string literals
        "unterminated string",
        "prefix `",                  # English prose leaked into code (e.g. "Let`s prove")
    ]
    if any(p in hard_check_lower for p in hard_error_patterns):
        return "syntax_error", compiler_out

    # 2. Spec/logic failures (Actual proof failures)
    if ("assertion failed" in out_lower
            or "pre-condition" in out_lower
            or "post-condition" in out_lower
            or "requires" in out_lower
            or "ensures" in out_lower):
        # Final safety check: if it says "expected ... found", it's still a syntax/type error.
        if "expected" in out_lower and "found" in out_lower:
            return "syntax_error", compiler_out
        return "assertion_failed", compiler_out

    # 3. Generic error fallback
    if "error:" in out_lower or "syntax" in out_lower:
        return "syntax_error", compiler_out

    if compiler_out:
        return "other_failure", compiler_out
    return "other_failure", ""


def validate_response_format(text: str, continuation_from_think: bool = False, require_body_only: bool = False) -> tuple[bool, float, str]:
    """Strictly validates the format of model responses (both Q-Gen and S-Gen).
    Returns (is_valid, penalty_score, reason).
    """
    if not text or not text.strip():
        return False, 0.0, "empty_response"

    text = re.sub(r"<\s*/?\s*thinking\s*>", lambda m: "</think>" if "/" in m.group(0) else "<think>", text, flags=re.IGNORECASE)

    # 1. Check <think> tag closure. For S-Gen we prefill "<think>\n" and only decode the continuation,
    # so a missing closing tag must be detected from the continuation alone.
    if continuation_from_think:
        if "</think>" not in text:
            return False, 0.0, "unclosed_think"
    elif "<think>" in text and "</think>" not in text:
        return False, 0.0, "unclosed_think"
    
    if require_body_only:
        # S-Gen path: all rust-block checks are performed only on the text AFTER </think>.
        # This prevents a draft rust block inside <think> from fooling the format check
        # while the actual answer section (after </think>) has no rust block.
        think_close_idx = text.find("</think>")
        post_think = text[think_close_idx + len("</think>"):] if think_close_idx != -1 else text

        if "```rust" not in post_think:
            return False, 0.0, "missing_rust_block_after_think"

        last_rust_idx = post_think.rfind("```rust")
        closing_idx = post_think.find("```", last_rust_idx + 7)
        if closing_idx == -1:
            return False, 0.0, "unclosed_rust_block"

        rust_idx = post_think.find("```rust")
        leaked = post_think[:rust_idx].strip()
        if leaked:
            return False, 0.0, "prose_before_rust_block"

        rust_match = re.search(r"```rust\s*\n(.*?)```", post_think, re.DOTALL)
        if not rust_match:
            return False, 0.0, "missing_rust_block"

        rust_body = rust_match.group(1).strip()
        if not rust_body:
            return False, 0.0, "empty_rust_block"

        forbidden_patterns = [
            (r"\brequires\b", "repeated_spec_keyword"),
            (r"\bensures\b", "repeated_spec_keyword"),
            (r"\bverus!\s*\{", "repeated_verus_wrapper"),
            (r"\bfn\s+main\s*\(", "repeated_fn_main"),
            (r"^\s*use\s+", "repeated_use_statement"),
            (r"<\s*reference implementation", "placeholder_reference_impl"),
        ]
        for pattern, reason in forbidden_patterns:
            if re.search(pattern, rust_body, re.IGNORECASE | re.MULTILINE):
                return False, 0.0, reason

        return True, 0.0, ""

    # 2. Check rust code block existence (non-S-Gen path)
    if "```rust" not in text:
        return False, 0.0, "missing_rust_block"

    # 3. Check rust code block closure
    last_rust_idx = text.rfind("```rust")
    if last_rust_idx != -1:
        closing_idx = text.find("```", last_rust_idx + 7)
        if closing_idx == -1:
            return False, 0.0, "unclosed_rust_block"

    return True, 0.0, ""


def smart_stitch(prompt_text, response_text, response_is_raw=False):
    """Cleanly stitches Spec (Prompt) and Body (Response) ensuring valid Rust/Verus wrapping."""
    prompt_text = _strip_think_tags(prompt_text, is_raw_generation=False)
    response_text = _strip_think_tags(response_text, is_raw_generation=response_is_raw)

    def _strip_wrappers(text: str) -> str:
        text = (text or "").strip()
        # Keep closing braces but drop wrapper comments / synthetic mains.
        text = re.sub(r'(\})\s*//\s*End of function\b[^\n]*', r'\1', text)
        text = re.sub(r'(\})\s*//\s*verus![^\n]*', r'\1', text)
        text = re.sub(r'^\}\s*//\s*verus!\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\bfn\s+main\s*\(\s*\)\s*\{.*?\}\s*$', '', text, flags=re.DOTALL)
        text = text.replace("verus! {", "").replace("} // verus!", "")
        return text.strip()

    def _split_outer_prefix_and_body(full_code: str) -> tuple[list[str], list[str]]:
        lines = full_code.splitlines()
        outer_prefix: list[str] = []
        body_lines: list[str] = []
        in_prefix = True
        in_use_block = False
        for line in lines:
            stripped = line.strip()
            if in_prefix:
                if not stripped:
                    if in_use_block:
                        outer_prefix.append("")
                        continue
                    continue
                if in_use_block:
                    outer_prefix.append(stripped)
                    if ";" in stripped:
                        in_use_block = False
                    continue
                if stripped.startswith("#![") or stripped.startswith("#[allow("):
                    outer_prefix.append(stripped)
                    continue
                if stripped.startswith("use "):
                    outer_prefix.append(stripped)
                    if ";" not in stripped:
                        in_use_block = True
                    continue
            in_prefix = False
            body_lines.append(line)
        return outer_prefix, body_lines

    p_code = prompt_text
    p_matches = re.findall(r"```(?:rust)?\s*\n(.*?)```", prompt_text, re.DOTALL)
    if p_matches:
        valid_block = next((b for b in reversed(p_matches) if "} // End of function" not in b and "fn main() {}" not in b), None)
        p_code = valid_block if valid_block else max(p_matches, key=len)
    elif "```" in prompt_text:
        p_code = prompt_text.split("```")[-1]

    r_code = response_text
    r_matches = re.findall(r"```(?:rust)?\s*\n(.*?)```", response_text, re.DOTALL)
    if r_matches: 
        r_code = r_matches[0]
    r_code = r_code.replace("```rust", "").replace("```", "").strip()

    if "// <BODY_HOLE>" in p_code:
        return p_code.replace("// <BODY_HOLE>", r_code), p_code, r_code

    p_code = _strip_wrappers(p_code)
    r_code = _strip_wrappers(r_code)
    full_code = f"{p_code}\n{r_code}".strip()
    outer_prefix, body_lines = _split_outer_prefix_and_body(full_code)
    if not any(line == "use vstd::prelude::*;" for line in outer_prefix):
        outer_prefix.insert(0, "use vstd::prelude::*;")
    prefix = "\n".join(line for line in outer_prefix).strip()
    body = "\n".join(body_lines).strip()
    stitched = prefix + "\n\nverus! {\n\n" + body + "\n\nfn main() {}\n\n} // verus!\n"
    return stitched, p_code, r_code

def prepare_prefix_prompt(code):
    spec, _ = extract_primary_function_spec(code)
    if not spec:
        return "// No function found in spec"
    return spec


def extract_primary_function_spec(code: str) -> tuple[str, dict]:
    """Extract the most likely target function spec from a noisy Q-Gen response.

    Q-Gen often emits helper functions, a full body, or a trailing `fn main`.
    The solver still expects one target spec, so we salvage the most plausible
    non-main function and truncate it at its opening brace.
    """
    source = (code or "").strip()
    meta = {
        "fn_count": 0,
        "contains_main": False,
        "multiple_non_main": False,
        "salvaged": False,
        "chosen_name": "",
        "prefix_source": "",
    }
    if not source:
        return "", meta

    def _truncate_to_spec(fn_chunk: str) -> str:
        lines = fn_chunk.splitlines()
        kept: list[str] = []
        for line in lines:
            kept.append(line.rstrip())
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "{":
                return "\n".join(kept).strip()
            if stripped.endswith("{") and not any(tok in stripped for tok in ("==>", "<==>", "&&&", "|||", ",")):
                return "\n".join(kept).strip()
        # Fallback: if no clean body-opening line is found, keep the whole chunk.
        return "\n".join(kept).strip()

    matches = list(re.finditer(r'\b((?:(?:pub|open|closed|exec|proof|spec)\s+)*)fn\s+([A-Za-z_][A-Za-z0-9_]*)', source))
    meta["fn_count"] = len(matches)
    meta["contains_main"] = any(m.group(2) == "main" for m in matches)
    non_main_count = sum(1 for m in matches if m.group(2) != "main")
    meta["multiple_non_main"] = non_main_count > 1

    if not matches:
        return source, meta

    candidates = []
    for idx, match in enumerate(matches):
        fn_name = match.group(2)
        qualifiers = (match.group(1) or "").strip().lower()
        if fn_name == "main":
            continue

        start = match.start()
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
        chunk = source[start:next_start]
        spec = _truncate_to_spec(chunk)

        if not spec:
            continue

        spec_lower = spec.lower()
        score = 0.0
        if "ensures" in spec_lower:
            score += 5.0
        if "requires" in spec_lower:
            score += 1.0
        if "spec" not in qualifiers and "proof" not in qualifiers:
            score += 1.5
        if idx == len(matches) - 1:
            score += 0.25
        score += idx * 0.01
        prefix_source = source[:start].strip()
        candidates.append((score, spec, fn_name, prefix_source))

    if not candidates:
        open_brace_idx = source.find("{")
        spec = source if open_brace_idx == -1 else source[:open_brace_idx + 1].strip()
        meta["salvaged"] = spec.strip() != source.strip()
        return spec, meta

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_spec, best_name, best_prefix = candidates[0][1], candidates[0][2], candidates[0][3]
    meta["chosen_name"] = best_name
    meta["prefix_source"] = best_prefix
    meta["salvaged"] = best_spec.strip() != source.strip()
    return best_spec, meta


def is_collapsed_qgen_spec(spec: str) -> tuple[bool, str]:
    """Reject obvious shallow/collapsed proposer specs before costly verification."""
    if not spec or not spec.strip():
        return True, "EMPTY SPEC"

    s = spec.lower()
    fn_match = re.search(r'\bfn\s+([a-zA-Z0-9_]+)', spec)
    fn_name = fn_match.group(1).lower() if fn_match else ""
    fn_tokens = [tok for tok in fn_name.split("_") if tok]
    banned_terms = ["parity", "even", "odd", "divisible", "mod", "remainder"]
    fixed_index_terms = ["[0]", "len() - 1", "first", "last", "ends"]
    vague_name_terms = ["validate", "evaluate", "check", "assess", "analyze", "verify", "complex", "conditions", "properties"]
    algorithmic_terms = [
        "forall|", "exists|", "while", "subrange", "partition", "search", "count",
        "sorted", "rotate", "merge", "filter", "prefix", "suffix", "perm", "seq<", "vec<",
        "binary", "lower_bound", "duplicate", "interleave", "construct", "accumulate"
    ]
    shallow_bool_terms = ["is_valid", "valid", "result: bool", "-> (r: bool)", "-> (is_valid: bool)"]

    banned_hits = sum(1 for t in banned_terms if t in s)
    fixed_hits = sum(1 for t in fixed_index_terms if t in s)
    algo_hits = sum(1 for t in algorithmic_terms if t in s)
    vague_name_hits = sum(1 for t in vague_name_terms if t in fn_tokens)
    shallow_bool_hits = sum(1 for t in shallow_bool_terms if t in s)
    conjunction_hits = s.count("&&") + s.count("||")

    if banned_hits >= 2:
        return True, "COLLAPSED parity/divisibility family"
    if fixed_hits >= 3 and algo_hits == 0:
        return True, "FIXED-INDEX/ENDS shallow pattern"
    if conjunction_hits >= 4 and algo_hits == 0:
        return True, "BOOLEAN STITCHING pattern"
    if vague_name_hits >= 2 and algo_hits == 0:
        return True, "VAGUE NAME / no algorithmic target"
    if vague_name_hits >= 3 and conjunction_hits >= 3:
        return True, "VAGUE NAME + stitched conditions"
    if shallow_bool_hits > 0 and conjunction_hits >= 3 and algo_hits <= 1:
        return True, "BOOLEAN VALIDATOR / shallow condition mashup"
    if ("forall|" in s or "exists|" in s) and conjunction_hits >= 5 and algo_hits <= 2:
        return True, "QUANTIFIER STITCHING / pseudo-complex validator"

    ensures_match = re.search(r'ensures\s*(.*?)\{', spec, re.DOTALL)
    if ensures_match:
        ensures_body = ensures_match.group(1).strip()
        if len(ensures_body) < 80 and algo_hits == 0:
            return True, "TOO SHORT / likely one-expression spec"
        if len(ensures_body) > 220 and vague_name_hits >= 2 and algo_hits <= 1:
            return True, "LONG VAGUE ENSURES / semantic padding"
        if ensures_body.count("==>") + ensures_body.count("<==>") >= 4 and algo_hits <= 1:
            return True, "IMPLICATION CHAIN / shallow validator"

    return False, ""


def has_body_like_tokens_in_spec(spec: str) -> tuple[bool, str]:
    """Reject obvious cases where Q-Gen outputs a function body instead of spec-only code."""
    target_source = _extract_target_fn_source(spec)
    if not target_source or "{" not in target_source:
        return False, ""
    lowered = target_source.lower()
    brace_idx = lowered.rfind("{")
    after_open = lowered[brace_idx + 1:]
    # Spec-only output is allowed to end right after the opening brace.
    if not after_open.strip():
        return False, ""
    body_markers = [
        "let ",
        "let\t",
        "while ",
        "for ",
        "return ",
        "if ",
        "match ",
        "loop ",
        ".push(",
        "=",
        ";",
    ]
    for marker in body_markers:
        if marker in after_open:
            return True, f"BODY TOKEN IN SPEC ({marker.strip()})"
    return False, ""


def has_illegal_characters(spec: str) -> tuple[bool, str]:
    """Check for non-ASCII math symbols or other illegal characters that cause model hallucination issues.
    These symbols (for example common unicode logic symbols) often indicate the model is 'dreaming' up math 
    rather than writing valid Verus/Rust code.
    """
    # List of common math symbols models hallucinate instead of using Verus/Rust operators
    # Pattern: (regex, human_readable_name)
    illegal_patterns = [
        ('[\u2228\u2227\u00ac\u21d2\u21d4\u2200\u2203]', "Unicode Math Symbol"),
        ('[\u27e8\u27e9]', "Unicode Angle Bracket"),
        ('[\u2260\u2264\u2265]', "Unicode Comparison Symbol"),
    ]
    
    for pattern, name in illegal_patterns:
        if re.search(pattern, spec):
            return True, name
            
    return False, ""


def is_bad_ensures_suspect(spec: str) -> tuple[bool, str]:
    """Cheap static checks for likely-unimplementable postconditions."""
    if not spec:
        return False, ""
    s = spec.lower()
    req_match = re.search(r'requires\s*(.*?)ensures', spec, re.DOTALL | re.IGNORECASE)
    ensures_match = re.search(r'ensures\s*(.*?)\{', spec, re.DOTALL | re.IGNORECASE)
    req = req_match.group(1).lower() if req_match else ""
    ens = ensures_match.group(1).lower() if ensures_match else ""
    if not ens:
        return False, ""

    ens_compact = re.sub(r"\s+", " ", ens)
    fn_match = re.search(r"\bfn\s+([a-zA-Z0-9_]+)", spec, re.IGNORECASE)
    fn_name = fn_match.group(1).lower() if fn_match else ""
    ret_ty = _infer_return_type_from_spec(spec).replace(" ", "").lower()
    ret_name = _infer_return_name_from_spec(spec).lower()

    # Direct self-contradictions indicate a malformed template rather than a
    # meaningful verification task.
    if re.search(r'([a-z_][a-z0-9_\.]*len\(\))\s*!=\s*\1', ens_compact):
        return True, "BAD ENSURES / self-contradictory length clause"

    # Unconstrained sign/value guarantees.
    # Skip very short ensures (likely spec-helper functions like `fn sum()`).
    # Skip specs with forall/exists quantifiers (they have real semantic structure).
    if len(ens_compact) > 30 and "forall|" not in ens and "exists|" not in ens and (">= 0" in ens or "<= 0" in ens) and not any(tok in req for tok in [">= 0", "<= 0", "nonnegative", "positive", "negative"]):
        if any(tok in ens for tok in ["max", "min", "sum", "difference", "gradient", "slope"]):
            return True, "BAD ENSURES / unconstrained sign claim"

    # Index-like returns without clear non-empty guard.
    # Do not hard-reject here: some valid search/lower_bound style specs rely on
    # idx <= len and option-style semantics. This is better treated as a weak-spec
    # soft signal than as an invalidation.

    # Count impossibility: count tied too tightly without provenance.
    # Exempt: `count == expr` where expr has arithmetic operators (e.g. `n*(n+1)/2`) -- those are valid formulas.
    if "count" in s and "count ==" in ens and not any(tok in ens for tok in ["exists|", "forall|", "<=", ">="]):
        count_eq_match = re.search(r'count\s*==\s*(.{1,30})', ens)
        if count_eq_match:
            rhs = count_eq_match.group(1)
            has_formula = any(op in rhs for op in ['*', '/', '(', 'len()', '==>', '<==>', '@'])
            if not has_formula:
                return True, "BAD ENSURES / brittle count equality"

    # Weak counting specs: only say "if element matches then count > 0", which does not define a count.
    if "count" in s and "forall|" in ens and any(tok in ens for tok in ["count > 0", "count == 0"]):
        if "exists|" not in ens and "result.len()" not in ens:
            return True, "BAD ENSURES / weak count implication"

    # Count specs that bind the final count directly to the witness index are almost always wrong.
    if "count" in s and "forall|" in ens and any(tok in ens for tok in ["count == i", "count > i", "count == j", "count > j"]):
        return True, "BAD ENSURES / brittle count-by-index rule"

    # Self-referential count updates in ensures are not meaningful postconditions.
    if "count" in s and any(tok in ens_compact for tok in ["count == count + 1", "count==count+1", "count == count", "count==count"]):
        return True, "BAD ENSURES / self-referential count rule"

    # Vec/result construction with weak provenance is a common false positive on
    # legitimate construct/merge tasks. Leave this to weak-spec scoring.

    # Incomplete if-expression leaked into ensures.
    # EXCEPTION: In Verus, `ensures result == if cond { a } else { b }` is valid syntax.
    # Our ensures regex truncates at the first `{`, making these look incomplete.
    # Only flag if `== if` is NOT near the end of ens (i.e., not a truncated Verus if-expr).
    if "== if" in ens_compact or "==if" in ens_compact:
        if_pos = max(ens_compact.rfind("== if"), ens_compact.rfind("==if"))
        is_truncated_verus_if = if_pos >= 0 and if_pos > len(ens_compact) - 40
        if not is_truncated_verus_if and " else " not in ens_compact and "else" not in ens:
            return True, "BAD ENSURES / incomplete conditional expression"

    # NOTE: shifted-index hard rejection was too aggressive in v5.6 and killed
    # many legitimate merge/scan specs. We intentionally do not hard-reject on
    # generic `i+1` / `i-1` patterns here; those should be handled later by
    # solver utility / trivial priors rather than by a front-door invalidation.

    # Fixed-width arithmetic masquerading as sequence algorithm.
    # Exemption: specs with concrete `== value` equalities at fixed indices are
    # legitimate small-vec constructions (e.g., s[0] == n, s[1] == n+1).
    fixed_index_refs = len(re.findall(r"\[[012]\]", ens))
    has_concrete_equality = bool(re.search(r"\[[012]\]\s*==\s*\w", ens))
    if fixed_index_refs >= 3 and not has_concrete_equality and "forall|" not in ens and "exists|" not in ens:
        return True, "BAD ENSURES / fixed-index arithmetic template"

    # Fake prefix/suffix/counting tasks that hard-code a few indices instead of a range relation.
    if any(tok in fn_name for tok in ["prefix", "suffix", "sum", "accumulate"]):
        small_fixed_refs = len(re.findall(r"\[[0-3]\]", ens))
        if small_fixed_refs >= 3 and "subrange" not in ens and "take(" not in ens and "drop(" not in ens:
            return True, "BAD ENSURES / fake prefix-sum style fixed indices"

    # Weak count bound only: keep as soft signal, not hard reject.

    # Fake prefix/sum/scan tasks: only a loose numeric upper bound, no range relation.
    if any(tok in fn_name for tok in ["prefix", "sum", "scan", "accumulate"]):
        has_only_bound = any(tok in ens for tok in ["sum <=", "count <=", "res <=", "result <="])
        has_range_semantics = any(tok in ens for tok in ["forall|", "exists|", "set::new", "subrange", "take(", "drop(", "<==>"])
        if has_only_bound and not has_range_semantics:
            return True, "BAD ENSURES / weak prefix-or-sum bound only"

    # Easy boolean predicate tasks are frequent in repaired datasets and seed pool.
    # Keep them out of hard reject; weak-spec penalty can still downweight them.

    # Nested result indexing must guarantee the inner vector is long enough.
    if "result[i][1]" in ens or "result[j][1]" in ens:
        has_inner_len_lb = any(tok in ens for tok in ["result[i].len() >= 2", "result[j].len() >= 2", "2 <= result[i].len()", "2 <= result[j].len()"])
        if not has_inner_len_lb:
            return True, "BAD ENSURES / nested result index without inner-length guard"

    # Partition/count specs with impossible end equalities.
    if any(tok in s for tok in ["partition", "count"]) and "arr.len() == " in ens:
        if "partition + 1" in ens or "count + 1" in ens:
            return True, "BAD ENSURES / brittle terminal equality"

    # New patterns (v5.15+)

    # Pattern A: forall strict-bound + exists equality on return variable -> contradiction.
    # e.g., "forall a[i] < max" combined with "exists a[i] == max" is self-contradictory:
    # no element can be both strictly less than max AND equal to max simultaneously.
    # Also catches the symmetric form "forall max > a[i]" + "exists max == a[i]".
    if ret_name and "forall|" in ens and "exists|" in ens:
        rn = re.escape(ret_name)
        clauses = [c for c in _split_top_level_spec_clauses(ens) if c.strip()]
        forall_clauses = [c for c in clauses if "forall|" in c]
        exists_clauses = [c for c in clauses if "exists|" in c]
        # Strict upper bound must come from a quantified upper-bound clause,
        # not from a sentinel guard like "max_profit > 0 ==> ...".
        has_strict_bound = any(
            bool(re.search(rf'<(?!=)\s*{rn}\b(?!\s*[\+\-\*/])', clause)) or
            bool(re.search(rf'\b{rn}\s*>(?![>=])\s*[A-Za-z_@\[]', clause))
            for clause in forall_clauses
        )
        # Equality witness must come from an existential witness clause.
        # Guards:
        #  - shifted equality: "m == a[i] + 1" is valid (m = max(a)+1 is consistent with forall a[i] < m)
        #  - literal sentinel: "x == -1" or "x == 0" are sentinel returns, not structural contradictions
        #    (e.g., first_e returns x=-1 when not found; forall i < x is an index bound, not a value bound)
        has_eq_witness = any(
            bool(re.search(rf'==\s*{rn}\b(?!\s*[\+\-\*/])', clause)) or
            # Exclude: "==" followed by ">" (Verus implication ==>), literal integers, arithmetic offsets
            bool(re.search(rf'\b{rn}\s*==(?![=>])(?!\s*-?\d)(?!\s*[\w\[\]\s\.]*[\+\-\/]\s*\d)', clause))
            for clause in exists_clauses
        )
        if has_strict_bound and has_eq_witness:
            return True, "BAD ENSURES / forall strict-bound contradicts exists equality"

    # Pattern B: index [k - 1] inside a forall where k starts at 0 -> OOB at k=0.
    # e.g., "forall|k| 0 <= k < arr.len() ==> result[k] == arr[k - 1]" panics at k=0.
    # Uses a backreference to ensure both "0 <= var" and "[var - 1]" are in the SAME forall
    # clause, preventing false positives on multi-forall specs like remove_element where a
    # different forall (starting at k, not 0) legitimately uses [i-1].
    oob_m = re.search(
        r'forall\|\s*(\w+)\s*(?::\s*\w+\s*)?\|\s*0\s*<=\s*\1\b[^{,]*\[\1(?:\s+as\s+int)?\s*-\s*1\]',
        ens, re.IGNORECASE
    )
    if oob_m:
        var_name = oob_m.group(1)
        return True, f"BAD ENSURES / [{var_name}-1] index is out-of-bounds when {var_name} starts from 0"

    # Pattern C: result <==> P and !result <==> P with identical predicate -> contradiction.
    # result = P and !result = P -> P AND !P, which is impossible for any boolean result.
    if "<==>" in ens_compact and "result" in ens_compact:
        res_bico = re.findall(r'\bresult\s*<==>\s*([^\n,{]{4,}?)(?=\s*,|\s*$)', ens_compact)
        neg_bico = re.findall(r'!\s*result\s*<==>\s*([^\n,{]{4,}?)(?=\s*,|\s*$)', ens_compact)
        if res_bico and neg_bico:
            norm_res = re.sub(r'\s+', ' ', res_bico[0].strip())
            norm_neg = re.sub(r'\s+', ' ', neg_bico[0].strip())
            if norm_res == norm_neg and len(norm_res) > 6:
                return True, "BAD ENSURES / result <==> P and !result <==> same P (tautological contradiction)"

    # Pattern D: max-named function whose postcondition uses a lower-bound (i.e. min) semantics.
    # Correct max spec: forall a[i] <= max (max is an upper bound).
    # Inverted / misnamed: forall a[i] >= max (max is a lower bound -> describes minimum).
    # The model always tries to implement max and perpetually fails on these.
    if ret_name and "max" in fn_name and "min" not in fn_name and "forall|" in ens:
        rn = re.escape(ret_name)
        # Lower-bound indicator: all elements >= ret (ret is a floor value, not a ceiling)
        has_lb_spec = (
            bool(re.search(rf'forall\|[^{{]+>=\s*{rn}\b', ens, re.DOTALL)) or
            bool(re.search(rf'\b{rn}\s*<=(?!=)\s', ens))
        )
        # Upper-bound indicator: all elements <= ret (ret is a ceiling value -- correct for max)
        has_ub_spec = (
            bool(re.search(rf'<=\s*{rn}\b', ens)) or
            bool(re.search(rf'\b{rn}\s*>=', ens))
        )
        if has_lb_spec and not has_ub_spec:
            return True, "BAD ENSURES / max-named function with min-style lower-bound postcondition"

    return False, ""


def is_weak_spec_suspect(spec: str) -> tuple[bool, str]:
    """Heuristic soft detector for low-information specs that are valid but poor training targets."""
    if not spec:
        return False, ""
    s = spec.lower()
    ensures_match = re.search(r'ensures\s*(.*?)\{', spec, re.DOTALL | re.IGNORECASE)
    ens = ensures_match.group(1).lower() if ensures_match else ""
    if not ens:
        return False, ""
    ens_compact = re.sub(r"\s+", " ", ens)
    fn_match = re.search(r"\bfn\s+([a-zA-Z0-9_]+)", spec, re.IGNORECASE)
    fn_name = fn_match.group(1).lower() if fn_match else ""
    ret_ty = _infer_return_type_from_spec(spec).replace(" ", "").lower()
    ret_name = _infer_return_name_from_spec(spec).lower()
    req_match = re.search(r'requires\s*(.*?)ensures', spec, re.DOTALL | re.IGNORECASE)
    req = req_match.group(1).lower() if req_match else ""

    # Vacuous existential implication: the witness does not establish the
    # property, so the spec is effectively almost-always-true.
    for match in re.finditer(r'exists\|[^|]+\|\s*([^\n\r{,]{1,160})', ens, re.IGNORECASE | re.DOTALL):
        body = re.sub(r"\s+", " ", match.group(1)).strip()
        if "==>" not in body:
            continue
        antecedent = body.split("==>", 1)[0]
        if "&&" not in antecedent and "||" not in antecedent:
            return True, "WEAK SPEC / vacuous existential implication"

    # Construct/transform Vec tasks need content provenance, not just a target
    # length. Pure size-only specs produce many meaningless training targets.
    construct_name_hits = [
        "interleave", "merge", "join", "append", "concat", "concatenate",
        "insert", "reverse", "clone", "copy", "replace", "remove", "filter",
    ]
    if ret_ty.startswith("vec<") and any(tok in fn_name for tok in construct_name_hits):
        has_len_relation = "len()" in ens or ".len()" in ens
        has_content_relation = any(tok in ens for tok in [
            "forall|", "exists|", "subrange", "take(", "drop(", "filter(",
            "map(", "[", "@",
        ])
        if has_len_relation and not has_content_relation:
            return True, "WEAK SPEC / vector construction only fixes length"

    # Search/index tasks without an explicit non-empty guard are often valid but
    # semantically weak or underspecified. Downweight rather than reject.
    # Exempt: functions with forall/exists quantifiers (already have real semantics),
    #         and search/bound/partition functions (empty arrays are valid inputs).
    search_exempt = any(tok in fn_name for tok in ["search", "bound", "lower_", "upper_", "partition", "bisect", "find"])
    if not search_exempt and "forall|" not in ens and "exists|" not in ens:
        if any(tok in s for tok in ["idx:", "index:"]) and any(tok in ens for tok in ["idx <", "index <", "result.unwrap() <", "partition <", "res <"]):
            if "len() > 0" not in req and "a.len() > 0" not in req and "arr.len() > 0" not in req and "vec.len() > 0" not in req:
                return True, "WEAK SPEC / index without non-empty guard"

    # Count tasks with only trivial bounds and no actual semantic relation.
    # Only trigger when "count" is explicitly in the function name or spec text,
    # not just because the return type is usize (too broad, causes false positives).
    if "count" in fn_name or "count:" in s:
        has_only_bounds = any(tok in ens for tok in ["count <=", "0 <= count"])
        has_real_semantics = any(tok in ens for tok in ["<==>", "set::new", "forall|", "exists|", "count == ", "== count"])
        if has_only_bounds and not has_real_semantics:
            return True, "WEAK SPEC / count bound only"

    # Prefix/sum/scan names without true range semantics.
    if any(tok in fn_name for tok in ["prefix", "sum", "scan", "accumulate"]):
        has_only_numeric_bound = any(tok in ens for tok in ["sum <=", "count <=", "res <=", "result <=", "<= arr.len()", "<= end - start"])
        has_range_relation = any(tok in ens for tok in ["forall|", "exists|", "set::new", "subrange", "take(", "drop(", "<==>"])
        if has_only_numeric_bound and not has_range_relation:
            return True, "WEAK SPEC / fake prefix-sum-or-scan"

    # Easy boolean predicate tasks that collapse to one simple pass/check.
    if ret_ty == "bool" and any(tok in fn_name for tok in ["contains", "exists", "has_", "have_", "all_", "any_", "find_first", "find_last"]):
        if "<==>" in ens and all(tok not in ens for tok in ["set::new", "subrange", "take(", "drop(", "chain(", "result[", "indices["]):
            return True, "WEAK SPEC / easy boolean predicate"

    # Vec extraction tasks with only size and index-bounds, but no stronger relational content.
    if ret_ty.startswith("vec<") and "indices" in fn_name:
        if "indices.len() ==" in ens and "forall|" in ens and all(tok not in ens for tok in ["indices[i] == ", "list[indices[i]]", "arr[indices[i]]", "result[i] =="]):
            return True, "WEAK SPEC / weak index-extraction relation"

    # Weak vector provenance: relation is too loose to define a meaningful construct task.
    if "vec<" in s or "vec<" in ens or "result: vec" in s or "-> (c: vec" in s or "-> (merged: vec" in s:
        has_len_relation = "len()" in ens or ".len()" in ens
        has_quantified_relation = "forall|" in ens or "exists|" in ens
        has_index_relation = "[" in ens and "]" in ens
        has_direct_relational_shape = any(tok in ens for tok in ["==", "subrange", "take(", "drop(", "chain(", "merge", "interleave"])
        if not (has_len_relation or has_quantified_relation or has_index_relation or has_direct_relational_shape):
            return True, "WEAK SPEC / vector result lacks provenance"

    # Scalar-return specs occasionally smuggle in extra input-only postconditions
    # instead of further constraining the result, which is usually a template leak.
    if ret_name and ret_ty not in ("", "bool", "()") and not ret_ty.startswith("vec<"):
        clauses = [re.sub(r"\s+", " ", c.strip().lower()) for c in _split_top_level_spec_clauses(ens) if c.strip()]
        if len(clauses) >= 2:
            for clause in clauses:
                if ret_name in clause or "result[" in clause or "result@" in clause or "res[" in clause or "res@" in clause:
                    continue
                if any(tok in clause for tok in ["forall|", "exists|", "==>", "<==>"]):
                    continue
                if any(sym in clause for sym in ["==", "!=", "<=", ">=", "<", ">"]):
                    return True, "WEAK SPEC / stray input-only ensures clause"

    # Tautological or nearly tautological ensures.
    tautology_markers = [
        "start <= end",
        "start <= arr.len()",
        "end <= arr.len()",
        "arr.len() >= 0",
        "list.len() >= 0",
    ]
    if sum(1 for tok in tautology_markers if tok in ens_compact) >= 2 and len(ens_compact) < 220:
        return True, "WEAK SPEC / mostly tautological ensures"

    return False, ""


def infer_unsolved_family(spec: str) -> str:
    s = spec.lower()
    if any(k in s for k in ["increment_each", "replace_", "transform_", "shift_left", "normalize_", "filter_not_", "concatenate_lasts"]):
        return "elementwise_transform"
    if any(k in s for k in ["subarray", "kadane"]):
        return "subarray"
    if any(k in s for k in ["partition", "split", "segment"]):
        return "partition"
    if any(k in s for k in ["upper_bound", "lower_bound", "binary search", "search", "index"]):
        return "search"
    if any(k in s for k in ["count", "occurrences", "greater_than", "zeros", "leaves"]):
        return "count"
    if any(k in s for k in ["max", "maximum", "minimum", "gradient", "slope"]):
        return "max_select"
    if any(k in s for k in ["construct", "interleav", "merge", "result vector"]):
        return "construct"
    return "other"


def infer_unsolved_template_key(spec: str) -> str:
    s = spec.lower()
    fn_match = re.search(r'\bfn\s+([a-zA-Z0-9_]+)', spec)
    fn_name = fn_match.group(1).lower() if fn_match else "unknown"
    fn_name = re.sub(r'_(unsolved|variant|v\d+)$', '', fn_name)

    if any(k in fn_name for k in ["increment_each", "replace", "transform", "shift_left", "normalize", "filter_not", "concatenate_lasts"]):
        return "elementwise_transform"
    if "count_occurrences" in fn_name or "count_" in fn_name:
        return "count_template"
    if "max_subarray" in fn_name or "maximum_subarray" in fn_name:
        return "subarray_max_template"
    if "find_max" in fn_name or "find_min" in fn_name:
        return "extrema_template"
    if "partition" in fn_name:
        return "partition_template"
    if "binary_search" in fn_name or "upper_bound" in fn_name or "lower_bound" in fn_name:
        return "search_template"

    ret = _infer_return_type_from_spec(spec).replace(" ", "").lower()
    ensures_match = re.search(r'ensures\s*(.*?)\{', spec, re.DOTALL | re.IGNORECASE)
    ens = ensures_match.group(1).lower() if ensures_match else ""
    if ret.startswith("vec<") and "forall|" in ens and "len()" in ens:
        return "vec_map_template"
    if "count" in s and "forall|" in ens:
        return "count_template"
    return fn_name


def _infer_return_type_from_spec(spec: str) -> str:
    target_spec = _extract_target_fn_source(spec)
    m = re.search(r'->\s*\((?:[A-Za-z_][A-Za-z0-9_]*\s*:\s*)?([^)]+)\)', target_spec, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'->\s*([^{\n]+)', target_spec)
    return m.group(1).strip() if m else ""


def _infer_return_name_from_spec(spec: str) -> str:
    m = re.search(r'->\s*\(([A-Za-z_][A-Za-z0-9_]*)\s*:', _extract_target_fn_source(spec), re.DOTALL)
    return m.group(1).strip() if m else ""


def _split_top_level_spec_clauses(block: str) -> list[str]:
    if not block:
        return []
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in block:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            clause = "".join(cur).strip()
            if clause:
                parts.append(clause)
            cur = []
            continue
        cur.append(ch)
    clause = "".join(cur).strip()
    if clause:
        parts.append(clause)
    return parts


def _infer_fn_params_from_spec(spec: str) -> list[tuple[str, str]]:
    m = re.search(r'fn\s+[A-Za-z_][A-Za-z0-9_]*\s*\((.*?)\)\s*->', _extract_target_fn_source(spec), re.DOTALL)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw:
        return []
    parts = []
    cur = []
    depth = 0
    for ch in raw:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    out = []
    for part in parts:
        if ":" not in part:
            continue
        name, ty = part.split(":", 1)
        out.append((name.strip(), ty.strip()))
    return out


def _extract_fn_name_from_spec(spec: str) -> str:
    match = _select_target_fn_match(spec or "")
    return match.group(2) if match else ""


def _iter_fn_matches(spec: str) -> list[re.Match]:
    return list(re.finditer(r'\b((?:(?:pub|open|closed|exec|proof|spec)\s+)*)fn\s+([A-Za-z_][A-Za-z0-9_]*)', spec or ""))


def _select_target_fn_match(spec: str) -> re.Match | None:
    matches = _iter_fn_matches(spec)
    if not matches:
        return None

    non_main = [m for m in matches if m.group(2) != "main"]
    candidates = non_main if non_main else matches
    non_helper = [
        m for m in candidates
        if "spec" not in (m.group(1) or "").lower() and "proof" not in (m.group(1) or "").lower()
    ]
    return non_helper[-1] if non_helper else candidates[-1]


def _extract_target_fn_source(spec: str) -> str:
    source = (spec or "").strip()
    if not source:
        return ""
    match = _select_target_fn_match(source)
    return source[match.start():].strip() if match is not None else source


QGEN_ALLOWED_TARGET_PATTERNS = [
    ("spec_fn", re.compile(r"^\s*(?:pub\s+)?(?:open\s+|closed\s+)?spec\s+fn\b", re.IGNORECASE)),
    ("proof_fn", re.compile(r"^\s*proof\s+fn\b", re.IGNORECASE)),
    ("fn", re.compile(r"^\s*(?:pub\s+)?fn\b", re.IGNORECASE)),
]


def _normalize_qgen_target_source(spec: str) -> str:
    source = (spec or "").strip()
    source = re.sub(r"^\s*pub\s+open\s+spec\s+fn\b", "spec fn", source, count=1, flags=re.IGNORECASE)
    source = re.sub(r"^\s*pub\s+closed\s+spec\s+fn\b", "spec fn", source, count=1, flags=re.IGNORECASE)
    source = re.sub(r"^\s*open\s+spec\s+fn\b", "spec fn", source, count=1, flags=re.IGNORECASE)
    source = re.sub(r"^\s*closed\s+spec\s+fn\b", "spec fn", source, count=1, flags=re.IGNORECASE)
    source = re.sub(r"^\s*pub\s+spec\s+fn\b", "spec fn", source, count=1, flags=re.IGNORECASE)
    source = re.sub(r"^\s*pub\s+fn\b", "fn", source, count=1, flags=re.IGNORECASE)
    return source.strip()


def _detect_qgen_target_kind(spec: str) -> str | None:
    source = _normalize_qgen_target_source(_extract_target_fn_source(spec))
    if not source:
        return None
    for kind, pattern in QGEN_ALLOWED_TARGET_PATTERNS:
        if pattern.search(source):
            return kind
    return None


def _sanitize_spec_prefix(prefix: str) -> str:
    """Keep helper declarations/comments, but strip top-level wrapper/import boilerplate.

    Move-mode synthesis should preserve helper spec/proof functions that the target
    depends on, but it must not carry outer `use ...` imports or an existing
    `verus! {` wrapper into the compiler filter, otherwise we end up nesting
    wrappers and systematically producing syntax errors.
    """
    source = (prefix or "").strip()
    if not source:
        return ""

    sanitized: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            if sanitized and sanitized[-1] != "":
                sanitized.append("")
            continue
        if re.match(r"^(?:pub\s+)?use\b", stripped):
            continue
        if re.match(r"^#!?\[allow\(", stripped):
            continue
        if re.match(r"^verus!\s*\{$", stripped):
            continue
        sanitized.append(line.rstrip())

    while sanitized and not sanitized[0].strip():
        sanitized.pop(0)
    while sanitized and not sanitized[-1].strip():
        sanitized.pop()
    return "\n".join(sanitized).strip()


def _extract_prefix_decl_blocks(prefix: str) -> list[tuple[str, str]]:
    source = (prefix or "").strip()
    if not source:
        return []

    decl_pattern = re.compile(
        r'\b((?:(?:pub|open|closed|exec|proof|spec|broadcast)\s+)*)((?:fn|enum|struct|type|trait))\s+([A-Za-z_][A-Za-z0-9_]*)'
    )
    matches = list(decl_pattern.finditer(source))
    if not matches:
        return []

    blocks: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
        block = source[start:end].strip()
        name = match.group(3)
        if block and name:
            blocks.append((name, block))
    return blocks


def _merge_spec_prefixes(seed_prefix: str, candidate_prefix: str) -> str:
    seed_clean = _sanitize_spec_prefix(seed_prefix)
    cand_clean = _sanitize_spec_prefix(candidate_prefix)
    if not seed_clean:
        return cand_clean
    if not cand_clean:
        return seed_clean

    seed_blocks = _extract_prefix_decl_blocks(seed_clean)
    cand_blocks = _extract_prefix_decl_blocks(cand_clean)
    if not seed_blocks and not cand_blocks:
        if cand_clean == seed_clean:
            return seed_clean
        if cand_clean in seed_clean:
            return seed_clean
        if seed_clean in cand_clean:
            return cand_clean
        return (seed_clean + "\n\n" + cand_clean).strip()

    merged: dict[str, str] = {}
    order: list[str] = []
    for name, block in seed_blocks:
        if name not in merged:
            order.append(name)
        merged[name] = block
    for name, block in cand_blocks:
        if name not in merged:
            order.append(name)
            merged[name] = block
    return "\n\n".join(merged[name] for name in order if merged.get(name, "").strip()).strip()


def _merge_freeform_spec_with_seed_prefix(seed_spec: str, candidate_spec: str, candidate_prefix: str = "") -> str:
    seed_prefix, _, _, _ = _split_spec_sections_with_prefix(seed_spec)
    merged_prefix = _merge_spec_prefixes(seed_prefix, candidate_prefix)
    _, signature, requires, ensures = _split_spec_sections_with_prefix(candidate_spec)
    if not signature:
        return (candidate_spec or "").strip()
    return _render_spec_sections(signature, list(requires), list(ensures), prefix=merged_prefix)


def _rename_fn_in_signature(signature: str, new_name: str) -> str:
    return re.sub(
        r'(\bfn\s+)([A-Za-z_][A-Za-z0-9_]*)',
        lambda m: f"{m.group(1)}{new_name}",
        signature,
        count=1,
    )


def _normalize_spec_clause(clause: str) -> str:
    return re.sub(r"\s+", " ", (clause or "").strip().rstrip(",")).lower()


def _split_spec_sections_with_prefix(spec: str) -> tuple[str, str, list[str], list[str]]:
    source = (spec or "").strip()
    if not source:
        return "", "", [], []

    match = _select_target_fn_match(source)
    prefix = source[:match.start()].rstrip() if match is not None else ""
    target_source = source[match.start():].strip() if match is not None else source

    open_idx = target_source.rfind("{")
    head = target_source[:open_idx].rstrip() if open_idx != -1 else target_source
    req_match = re.search(r'\brequires\b', head, re.IGNORECASE)
    ens_match = re.search(r'\bensures\b', head, re.IGNORECASE)

    if req_match and ens_match and req_match.start() < ens_match.start():
        signature = head[:req_match.start()].rstrip()
        req_block = head[req_match.end():ens_match.start()].strip()
        ens_block = head[ens_match.end():].strip()
    elif ens_match:
        signature = head[:ens_match.start()].rstrip()
        req_block = ""
        ens_block = head[ens_match.end():].strip()
    else:
        signature = head.rstrip()
        req_block = ""
        ens_block = ""

    requires = [c.rstrip(",") for c in _split_top_level_spec_clauses(req_block)]
    ensures = [c.rstrip(",") for c in _split_top_level_spec_clauses(ens_block)]
    return prefix, signature, requires, ensures


def _split_spec_sections(spec: str) -> tuple[str, list[str], list[str]]:
    _, signature, requires, ensures = _split_spec_sections_with_prefix(spec)
    return signature, requires, ensures


def _render_spec_sections(signature: str, requires: list[str], ensures: list[str], prefix: str = "") -> str:
    lines = []
    if prefix and prefix.strip():
        lines.append(prefix.rstrip())
    lines.append(signature.rstrip())
    if requires:
        lines.append("    requires")
        for clause in requires:
            lines.append(f"        {clause.rstrip(',')},")
    if ensures:
        lines.append("    ensures")
        for clause in ensures:
            lines.append(f"        {clause.rstrip(',')},")
    lines.append("{")
    return "\n".join(lines).strip()


def _render_external_body_spec(spec: str) -> str:
    prefix, signature, requires, ensures = _split_spec_sections_with_prefix(spec)
    if not signature:
        return (spec or "").strip()
    prefix = _sanitize_spec_prefix(prefix)
    signature = "#[verifier::external_body]\n" + signature.lstrip()
    return _render_spec_sections(signature, requires, ensures, prefix=prefix)


def _canonical_type_string(ty: str) -> str:
    return re.sub(r"\s+", "", ty or "")


def _container_inner_type(ty: str) -> str:
    ty_compact = _canonical_type_string(ty)
    patterns = [
        r'^&\[(.+)\]$',
        r'^&Vec<(.+)>$',
        r'^Vec<(.+)>$',
        r'^Seq<(.+)>$',
        r'^&Seq<(.+)>$',
    ]
    for pattern in patterns:
        match = re.match(pattern, ty_compact)
        if match:
            return match.group(1)
    return ""


def _is_slice_like_type(ty: str) -> bool:
    ty_compact = _canonical_type_string(ty)
    return any(tok in ty_compact for tok in ["&[", "&Vec<", "Vec<", "Seq<", "&Seq<"])


def _is_vec_like_type(ty: str) -> bool:
    ty_compact = _canonical_type_string(ty)
    return ty_compact.startswith("Vec<") or ty_compact.startswith("&Vec<")


def _is_unsigned_numeric_type(ty: str) -> bool:
    return _canonical_type_string(ty) in {"usize", "u8", "u16", "u32", "u64", "nat"}


def _is_signed_numeric_type(ty: str) -> bool:
    return _canonical_type_string(ty) in {"isize", "i8", "i16", "i32", "i64", "int"}


def _is_numeric_type(ty: str) -> bool:
    return _is_unsigned_numeric_type(ty) or _is_signed_numeric_type(ty)


def _render_numeric_cast(expr: str, ret_ty: str) -> str:
    ret = ret_ty.replace(" ", "")
    if ret in {"usize", "u8", "u16", "u32", "u64", "isize", "i8", "i16", "i32", "i64", "int", "nat"}:
        return f"({expr}) as {ret}"
    return expr


def _build_seed_sanity_bodies(spec: str) -> list[tuple[str, str]]:
    ret = _infer_return_type_from_spec(spec).replace(" ", "")
    params = _infer_fn_params_from_spec(spec)
    first_slice = next((name for name, ty in params if "&[" in ty or "&Vec<" in ty), None)
    first_nested_slice = next((name for name, ty in params if "&[&[" in ty or "&Vec<&[" in ty or "&[Vec<" in ty), None)
    first_numeric = next((name for name, ty in params if ty.replace(" ", "") in {"u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize", "int", "nat"}), None)
    bodies: list[tuple[str, str]] = []
    if not ret:
        return bodies

    if ret == "bool":
        bodies.extend([
            ("const_false", "false\n}"),
            ("const_true", "true\n}"),
        ])
    elif ret in {"u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize", "int", "nat"}:
        bodies.extend([
            ("const_zero", "0\n}"),
            ("const_one", "1\n}"),
        ])
    elif ret.startswith("Vec<"):
        bodies.append(("empty_vec", "Vec::new()\n}"))
    elif ret.startswith("Option<"):
        bodies.append(("const_none", "None\n}"))
    elif ret.startswith("Result<"):
        bodies.append(("const_err", "Err(())\n}"))

    # Cheap syntactic baselines for common shallow array indexing specs.
    if first_slice is not None:
        if ret in {"i32", "u32", "usize", "u64"}:
            bodies.append(("first_elem", f"{first_slice}[0]\n}}"))
            bodies.append(("last_elem", f"{first_slice}[{first_slice}.len() - 1]\n}}"))
        elif ret == "bool":
            bodies.append(("const_len_check", f"{first_slice}.len() == 0\n}}"))
        elif ret.startswith("Vec<"):
            bodies.append(("copy_vec", f"{first_slice}.to_vec()\n}}"))
    if first_nested_slice is not None and ret.startswith("Vec<"):
        bodies.append(("extract_heads_like", f"{first_nested_slice}.iter().map(|xs| xs[0]).collect()\n}}"))
        bodies.append(("extract_tails_like", f"{first_nested_slice}.iter().map(|xs| xs[xs.len() - 1]).collect()\n}}"))
    if first_numeric is not None and ret in {"i32", "u32", "usize", "u64", "isize", "i64"}:
        bodies.append(("echo_numeric_arg", f"{_render_numeric_cast(first_numeric, ret)}\n}}"))
    if first_slice is not None and ret in {"i32", "u32", "usize", "u64", "isize", "i64"}:
        bodies.append(("len_as_value", f"{_render_numeric_cast(f'{first_slice}.len()', ret)}\n}}"))
        if ret.startswith("i") or ret.startswith("u") or ret in {"int", "nat"}:
            bodies.append(("count_positive", f"{_render_numeric_cast(f'{first_slice}.iter().filter(|&&x| x > 0).count()', ret)}\n}}"))
            bodies.append(("count_even", f"{_render_numeric_cast(f'{first_slice}.iter().filter(|&&x| x % 2 == 0).count()', ret)}\n}}"))
    if len(params) >= 2 and ret in {"usize", "u32", "u64", "i32", "i64", "isize"}:
        p0n, p0t = params[0]
        p1n, p1t = params[1]
        p0tc = p0t.replace(" ", "")
        p1tc = p1t.replace(" ", "")
        if ("&[" in p0tc or "&Vec<" in p0tc) and ("&[" in p1tc or "&Vec<" in p1tc):
            bodies.append(("common_prefix_len", f"{_render_numeric_cast(f'{p0n}.iter().zip({p1n}.iter()).take_while(|(x,y)| x == y).count()', ret)}\n}}"))
            mismatch_expr = f"({p0n}.iter().zip({p1n}.iter()).position(|(x,y)| x != y).unwrap_or(::core::cmp::min({p0n}.len(), {p1n}.len())))"
            bodies.append(("first_mismatch_index_or_len", f"{_render_numeric_cast(mismatch_expr, ret)}\n}}"))
        if ("&[" in p0tc or "&Vec<" in p0tc) and p1tc in {"i32","i64","isize","u32","u64","usize","int","nat","u8"} and ret in {"usize", "u32", "u64", "i32", "i64", "isize"}:
            bodies.append(("count_equals_scalar", f"{_render_numeric_cast(f'{p0n}.iter().filter(|&&x| x == {p1n}).count()', ret)}\n}}"))
            occurrence_expr = f"({p0n}.iter().position(|&x| x == {p1n}).unwrap_or({p0n}.len()))"
            bodies.append(("first_occurrence_or_len", f"{_render_numeric_cast(occurrence_expr, ret)}\n}}"))
    if first_slice is not None and ret.startswith("Vec<"):
        inner = ret[4:-1] if ret.endswith(">") else ""
        if inner in {"i32", "i64", "isize", "u32", "u64", "usize", "int", "nat"}:
            bodies.append(("minus_one_map", f"{first_slice}.iter().map(|&x| x - 1).collect()\n}}"))
            bodies.append(("plus_one_map", f"{first_slice}.iter().map(|&x| x + 1).collect()\n}}"))
            bodies.append(("filter_nonnegative", f"{first_slice}.iter().copied().filter(|&x| x >= 0).collect()\n}}"))
    if len(params) >= 2 and ret == "bool":
        p0n, p0t = params[0]
        p1n, p1t = params[1]
        p0tc = p0t.replace(" ", "")
        p1tc = p1t.replace(" ", "")
        if ("&[" in p0tc or "&Vec<" in p0tc) and ("&[" in p1tc or "&Vec<" in p1tc):
            bodies.append(("starts_with_slice", f"{p0n}.starts_with({p1n})\n}}"))
            bodies.append(("all_equal_zip", f"{p0n}.len() == {p1n}.len() && {p0n}.iter().zip({p1n}.iter()).all(|(x,y)| x == y)\n}}"))

    deduped: list[tuple[str, str]] = []
    seen = set()
    for item in bodies:
        if item[0] not in seen:
            deduped.append(item)
            seen.add(item[0])
    return deduped


def trivial_seed_solution_passes(spec: str, timeout: int = 10) -> tuple[bool, str]:
    """Run a few dumb baselines; if any verify, the spec is too shallow for the seed pool."""
    for label, body in _build_seed_sanity_bodies(spec):
        try:
            full_code, _, _ = smart_stitch(spec, body)
            verified, _, _ = verify_code(full_code, timeout=timeout)
            if verified:
                return True, label
        except Exception:
            continue
    return False, ""


def estimate_cheap_solver_pass_rate(spec: str, timeout: int = 6) -> tuple[float, list[str]]:
    """Estimate how much of the spec can already be solved by a tiny cheap-solver bank."""
    bodies = _build_seed_sanity_bodies(spec)
    if not bodies:
        return 0.0, []
    tried = 0
    hits: list[str] = []
    for label, body in bodies[:8]:
        try:
            full_code, _, _ = smart_stitch(spec, body)
            verified, _, _ = verify_code(full_code, timeout=timeout)
            tried += 1
            if verified:
                hits.append(label)
        except Exception:
            tried += 1
            continue
    if tried == 0:
        return 0.0, []
    return float(len(hits)) / float(tried), hits


def estimate_trivial_prior_from_spec(spec: str) -> tuple[float, str]:
    """Cheap prior that a spec can be solved by a trivial baseline."""
    if not spec:
        return 0.0, ""
    s = spec.lower()
    ens_match = re.search(r'ensures\s*(.*?)\{', spec, re.DOTALL | re.IGNORECASE)
    ens = ens_match.group(1).lower() if ens_match else ""
    ens_compact = re.sub(r"\s+", " ", ens)
    fn_match = re.search(r'\bfn\s+([a-zA-Z0-9_]+)', spec, re.IGNORECASE)
    fn_name = fn_match.group(1).lower() if fn_match else ""
    ret_ty = _infer_return_type_from_spec(spec).replace(" ", "").lower()

    if any(tok in ens_compact for tok in ["result == 1", "res == 1", "count == 1", "result == true", "res == true", "result == false", "res == false"]):
        return 0.90, "TRIVIAL PRIOR / constant output"
    if any(tok in ens_compact for tok in ["result == a[0]", "res == a[0]", "result == arr[0]", "res == arr[0]"]):
        return 0.85, "TRIVIAL PRIOR / first element"
    if ret_ty.startswith("vec<") and any(tok in fn_name for tok in ["construct_result", "copy_", "clone_", "identity_", "extract_", "tails", "heads", "elements", "prefix_values"]):
        if "result.len() == a.len()" in ens_compact or "tails.len() == lists.len()" in ens_compact:
            if any(tok in ens_compact for tok in [
                "result[i] == a[i]",
                "tails[i] == lists[i][lists[i].len() - 1]",
                "heads[i] == lists[i][0]",
                "result[i] == arr[i]",
            ]):
                return 0.90, "TRIVIAL PRIOR / copy-or-direct-extract vector"
    if any(tok in fn_name for tok in ["extract_tails", "extract_heads", "first_element", "last_element", "find_first", "find_last"]):
        if any(tok in ens_compact for tok in [
            "tails[i] == lists[i][lists[i].len() - 1]",
            "heads[i] == lists[i][0]",
            "result == a[0]",
            "res == a[0]",
        ]):
            return 0.90, "TRIVIAL PRIOR / direct extraction"
    if "count == count + 1" in ens_compact or "count==count+1" in ens_compact or "count == count" in ens_compact or "count==count" in ens_compact:
        return 0.95, "TRIVIAL PRIOR / self-referential count"
    if ret_ty == "bool" and any(tok in fn_name for tok in ["contains", "exists", "has_", "have_", "all_", "any_"]):
        if "<==>" in ens and all(tok not in ens for tok in ["set::new", "subrange", "take(", "drop(", "chain(", "result[", "indices["]):
            return 0.65, "TRIVIAL PRIOR / easy boolean predicate"
    if "count" in fn_name or "count" in ens:
        has_only_bounds = any(tok in ens_compact for tok in ["count <=", "0 <= count", "count <="])
        has_real_semantics = any(tok in ens for tok in ["<==>", "set::new", "forall|", "exists|", "count == "])
        if has_only_bounds and not has_real_semantics:
            return 0.75, "TRIVIAL PRIOR / weak count bound"
    if any(tok in fn_name for tok in ["prefix", "sum", "scan", "accumulate"]):
        has_only_bound = any(tok in ens_compact for tok in ["sum <=", "count <=", "res <=", "result <=", "<= arr.len()", "<= end - start"])
        has_range_semantics = any(tok in ens for tok in ["forall|", "exists|", "set::new", "subrange", "take(", "drop(", "<==>"])
        if has_only_bound and not has_range_semantics:
            return 0.70, "TRIVIAL PRIOR / weak prefix-sum bound"
    if any(tok in ens_compact for tok in ["a[i] == a[i]", "arr[i] == arr[i]", "list[i] == list[i]"]):
        return 0.90, "TRIVIAL PRIOR / tautological identity"
    if ret_ty.startswith("vec<") and "indices" in fn_name and "indices.len() ==" in ens and "forall|" in ens and all(tok not in ens for tok in ["indices[i] == ", "list[indices[i]]", "arr[indices[i]]", "result[i] =="]):
        return 0.60, "TRIVIAL PRIOR / weak index extraction"
    return 0.0, ""


def _strip_think_tags(text, is_raw_generation=False):
    """Removes <think>...</think> blocks from text before code extraction, avoiding draft code."""
    text = re.sub(r"<\s*/?\s*thinking\s*>", lambda m: "</think>" if "/" in m.group(0) else "<think>", text, flags=re.IGNORECASE)
    text_lower = text.lower()
    
    if is_raw_generation:
        close_idx = text_lower.find('</think>')
        if close_idx != -1:
            text = text[close_idx + len('</think>'):]
        else:
            if '<think>' in text_lower:
                text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
                return text
            # If the model emits a direct code/spec response without any reasoning tags,
            # keep the original content instead of collapsing it to empty.
            return text
            
    # Generic strip for non-raw text or if hallucinated later
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # [FIX] Legacy support (if is_raw_generation=False but text starts with reasoning)
    if not is_raw_generation and '</think>' in text.lower() and '<think>' not in text.lower():
        text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
    return text

def extract_code_from_text(text):
    text = _strip_think_tags(text, is_raw_generation=True)
    matches = re.findall(r"```(?:rust)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        scored_blocks = []
        for idx, block in enumerate(matches):
            block_s = block.strip()
            block_lower = block_s.lower()
            score = 0.0
            if "fn " in block_lower:
                score += 4.0
            if "ensures" in block_lower:
                score += 3.0
            if "requires" in block_lower:
                score += 1.0
            # Prefer later blocks when structural quality is similar.
            score += idx * 0.01
            scored_blocks.append((score, block_s))
        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        extracted = scored_blocks[0][1]
        if extracted:
            return extracted
    return text.strip()

def extract_spec_and_impl(text):
    """[Spec-Only] Extracts Spec from Q-Gen output. Impl is no longer generated.
    Returns (spec, '') - second element kept empty for backward compatibility."""
    text = _strip_think_tags(text, is_raw_generation=True)
    if text.count("```") % 2 != 0:
        text += "\n```"
        
    matches = re.findall(r"```(?:rust)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not matches:
        return text.strip(), ""
    scored_blocks = []
    for idx, block in enumerate(matches):
        block_s = block.strip()
        block_lower = block_s.lower()
        score = 0.0
        if "fn " in block_lower:
            score += 4.0
        if "ensures" in block_lower:
            score += 3.0
        if "requires" in block_lower:
            score += 1.0
        # If the model emits multiple code blocks, the last one is often the refined answer.
        score += idx * 0.01
        scored_blocks.append((score, block_s))
    scored_blocks.sort(key=lambda x: x[0], reverse=True)
    return scored_blocks[0][1], ""



def pad_and_stack_tensors(input_tensors, pad_id, max_len=None, device=None, left_pad=True):
    """Generic, heavily vectorized dynamic truncation and padding engine."""
    if not input_tensors:
        return torch.empty((0, max_len or 0), dtype=torch.long, device=device), torch.empty((0, max_len or 0), dtype=torch.long, device=device)
        
    if not max_len: 
        max_len = max((len(t) for t in input_tensors), default=0)
        
    padded_ids, padded_masks = [], []
    for t in input_tensors:
        if len(t) > max_len: t = t[-max_len:] if left_pad else t[:max_len]
        req_pad = max_len - len(t)
        if req_pad > 0:
            pad_vec = torch.full((req_pad,), pad_id, dtype=torch.long)
            mask_vec = torch.zeros((req_pad,), dtype=torch.long)
            if left_pad:
                ids = torch.cat([pad_vec, t.cpu()])
                mask = torch.cat([mask_vec, torch.ones_like(t).cpu()])
            else:
                ids = torch.cat([t.cpu(), pad_vec])
                mask = torch.cat([torch.ones_like(t).cpu(), mask_vec])
        else:
            ids = t.cpu()
            mask = torch.ones_like(t).cpu()
            
        if device is not None:
            ids = ids.to(device)
            mask = mask.to(device)
            
        padded_ids.append(ids)
        padded_masks.append(mask)

    return torch.stack(padded_ids), torch.stack(padded_masks)


# ====================================================================
# [Similarity] MinHash-based Approximate Jaccard Similarity
# Pure Python, no external dependencies (uses hashlib + stdlib only)
# ====================================================================

import hashlib
import struct

_MINHASH_NUM_PERM = 64    # Number of hash permutations
_MINHASH_NGRAM_N = 3      # Character n-gram size
_MINHASH_MAX_INT = (1 << 32)  # 32-bit hash space

# Pre-generate (a, b) pairs for linear congruential hash permutations
# h_i(x) = (a_i * x + b_i) mod prime
_MERSENNE_PRIME = (1 << 61) - 1
_rng_state = 12345
def _next_rng():
    global _rng_state
    _rng_state = (_rng_state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return _rng_state

_HASH_A = [_next_rng() % _MERSENNE_PRIME for _ in range(_MINHASH_NUM_PERM)]
_HASH_B = [_next_rng() % _MERSENNE_PRIME for _ in range(_MINHASH_NUM_PERM)]

def _text_to_shingles(text: str, n: int = _MINHASH_NGRAM_N) -> list:
    """Convert text to char-level n-gram tokens (shingles) after normalization."""
    t = re.sub(r'[^\w\s]', ' ', text.lower())
    t = re.sub(r'\s+', ' ', t).strip()
    if len(t) < n:
        return [t] if t else []
    return [t[i:i+n] for i in range(len(t) - n + 1)]

def _shingle_hash(shingle: str) -> int:
    """Deterministic 32-bit hash for a shingle using MD5."""
    h = hashlib.md5(shingle.encode('utf-8')).digest()
    return struct.unpack('<I', h[:4])[0]  # Take first 4 bytes as uint32

def compute_minhash(text: str) -> list:
    """
    Compute a MinHash signature (list of num_perm ints) for a given text.
    The signature is a compact approximation of the text's shingle set.
    Jaccard(A, B) ~= fraction of positions where signature[A][i] == signature[B][i].
    """
    shingles = _text_to_shingles(text)
    if not shingles:
        return [_MINHASH_MAX_INT] * _MINHASH_NUM_PERM
    
    sig = [_MINHASH_MAX_INT] * _MINHASH_NUM_PERM
    for shingle in shingles:
        h = _shingle_hash(shingle)
        for i in range(_MINHASH_NUM_PERM):
            ph = int((_HASH_A[i] * h + _HASH_B[i]) % _MERSENNE_PRIME) % _MINHASH_MAX_INT
            if ph < sig[i]:
                sig[i] = ph
    return sig

def minhash_jaccard(sig_a: list, sig_b: list) -> float:
    """Estimate Jaccard similarity from two MinHash signatures. O(num_perm)."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)

def ngram_jaccard(a: str, b: str, n: int = _MINHASH_NGRAM_N) -> float:
    """MinHash-based Jaccard similarity (replaces old char N-gram Jaccard)."""
    if not a or not b:
        return 0.0
    return minhash_jaccard(compute_minhash(a), compute_minhash(b))

def spec_sim_to_seeds(spec: str, seeds: list, sig_cache: dict = None) -> float:
    """Max Jaccard similarity between `spec` and any seed question in `seeds`.
    seeds: list of (seed_dict, label) tuples as used by turn1_seeds.
    sig_cache: optional dict mapping seed question_str -> precomputed MinHash signature.
    Returns S_max in [0, 1].
    """
    if not spec or not seeds:
        return 0.0
    spec_sig = compute_minhash(spec)
    max_sim = 0.0
    for item in seeds:
        seed_dict = item[0] if isinstance(item, (tuple, list)) else item
        seed_text = seed_dict.get('question', '') if isinstance(seed_dict, dict) else str(seed_dict)
        if sig_cache is not None and seed_text in sig_cache:
            seed_sig = sig_cache[seed_text]
        else:
            seed_sig = compute_minhash(seed_text)
        sim = minhash_jaccard(spec_sig, seed_sig)
        if sim > max_sim:
            max_sim = sim
    return max_sim


def spec_sim_to_pool(spec: str, sig_cache: dict) -> float:
    """[Anti-Hack Fix 1] Max Jaccard similarity between `spec` and ANY question in the seed pool.
    Uses precomputed MinHash signatures from the pool's sig_cache for O(N_pool) lookup.
    """
    if not spec or not sig_cache:
        return 0.0
    spec_sig = compute_minhash(spec)
    max_sim = 0.0
    for existing_sig in sig_cache.values():
        sim = minhash_jaccard(spec_sig, existing_sig)
        if sim > max_sim:
            max_sim = sim
    return max_sim


def extract_fn_signature(spec: str) -> str:
    """[Anti-Hack Fix 2] Extract normalized function signature (types only, name removed) from Verus spec.
    Returns a canonical string like '(i32,&[i32])->bool' for structural comparison.
    """
    m = re.search(r'fn\s+\w+\s*\(([^)]*)\)(?:\s*->\s*(\S+))?', spec)
    if not m:
        return ""
    params = m.group(1).strip()
    ret = (m.group(2) or "").strip()
    # Normalize: strip param names, keep only types
    param_types = []
    for p in params.split(','):
        p = p.strip()
        if ':' in p:
            param_types.append(p.split(':')[-1].strip())
        elif p:
            param_types.append(p)
    return f"({','.join(param_types)})->{ret}"


def signature_similarity(spec_a: str, spec_b: str) -> float:
    """[Anti-Hack Fix 2] Binary similarity: 1.0 if function signatures match structurally, else 0.0.
    Catches 'Minimal Variation' hacks where only variable/function names are changed.
    """
    sig_a = extract_fn_signature(spec_a)
    sig_b = extract_fn_signature(spec_b)
    if not sig_a or not sig_b:
        return 0.0
    return 1.0 if sig_a == sig_b else 0.0


# ====================================================================
# [State Management] Seed Pool Manager (O(1) Hash Map Deduplication)
# ====================================================================

class SeedPoolManager:
    def __init__(
        self,
        data_dir,
        checkpoint_dir=None,
        downgrade_fail_threshold=2,
        q_gen_n=8,
        kin_prior_weight: float = 0.10,
        path_credit_gamma: float = 0.70,
    ):
        self.data_dir = data_dir
        self.q_gen_n = q_gen_n
        self.kin_prior_weight = float(kin_prior_weight)
        self.path_credit_gamma = float(path_credit_gamma)
        # downgrade_fail_threshold kept for API compat but no longer used (downgrade removed)
        # DAG-UCT statistics
        # value_sums[q] accumulates rollout reward for sampled selection paths.
        self.value_sums = {}   # question_str -> float
        # Prefer the canonicalized curated seed pool so Q-Gen training, seed selection,
        # and repaired validation all see the same trimmed question style.
        # Falls back to older raw files for backwards compatibility.
        # This file is read-only - training never overwrites it.
        self.seed_path = _pick_existing_path(
            os.path.join(data_dir, "curated_sft", "seed_pool_all_canonicalized.jsonl"),
            os.path.join(data_dir, "seed_pool_all_canonicalized.jsonl"),
            os.path.join(data_dir, "seed_pool_all.jsonl"),
            os.path.join(data_dir, "seed_pool.jsonl"),
        )
        self.checkpoint_dir = checkpoint_dir or data_dir
        # seed_pool_evolved.jsonl - full pool state for checkpoint resume.
        # Contains both the base entries and everything discovered during training.
        self.evolved_seed_path = os.path.join(self.checkpoint_dir, "seed_pool_evolved.jsonl")
        self.pool = {}          # question_str -> dict  (active pool for Q-Gen sampling)
        self.sig_cache = {}     # question_str -> MinHash sig  (full pool, for dedup)

        # Search graph topology.
        # Main curriculum edges are tracked as directed parent -> child links.
        # Structural similarity links are kept separately as weak priors.
        self.graph_edges = {}   # question_str -> {'parents': set, 'children': set, 'child_edge_types': dict, 'parent_edge_types': dict}
        self.kin_neighbors = {}  # question_str -> set[question_str]
        self.edge_visit_counts = {}  # parent_q -> {child_q -> float}
        self.stay_visit_counts = {}  # question_str -> float

        # UCT sampling state
        # visit_counts[q] = how many sampled seed paths traversed q.
        # _total_visits   = number of sampled seed paths overall.
        self.visit_counts = {}  # question_str -> float
        self._total_visits = 0.0

        # Path for persisting graph + DAG-UCT state between runs.
        self.graph_path = os.path.join(self.checkpoint_dir, "seed_pool_graph.json")

    @staticmethod
    def _is_output_shape_error(error_type) -> bool:
        return str(error_type or "") in {"format_error", "syntax_error", "output_shape_error"}

    def _ensure_graph_node(self, question: str):
        q = question.strip()
        if q not in self.graph_edges:
            self.graph_edges[q] = {
                'parents': set(),
                'children': set(),
                'child_edge_types': {},
                'parent_edge_types': {},
            }
        if q not in self.kin_neighbors:
            self.kin_neighbors[q] = set()
        if q not in self.edge_visit_counts:
            self.edge_visit_counts[q] = {}
        if q not in self.stay_visit_counts:
            self.stay_visit_counts[q] = 0.0
        if q not in self.visit_counts:
            self.visit_counts[q] = 0.0
        if q not in self.value_sums:
            self.value_sums[q] = 0.0

    def _node_value(self, question: str) -> float:
        q = question.strip()
        visits = float(self.visit_counts.get(q, 0.0))
        if visits > 0.0:
            return float(self.value_sums.get(q, 0.0)) / max(visits, 1e-8)
        # For unvisited nodes, use the same neighborhood difficulty signal that
        # drives action hints and target-pass-rate rewards, rather than the
        # node's direct eval pass_rate alone.
        entry = self.pool.get(q)
        return self.neighborhood_pass_rate(q) if entry is not None else 0.0

    def _kin_prior(self, question: str) -> float:
        q = question.strip()
        if self.kin_prior_weight <= 0.0:
            return 0.0
        neighbors = [n for n in self.kin_neighbors.get(q, set()) if n in self.pool]
        if not neighbors:
            return 0.0
        return self.kin_prior_weight * float(np.mean([self._node_value(n) for n in neighbors]))

    def _edge_visits(self, parent_q: str, child_q: str) -> float:
        return float(self.edge_visit_counts.get(parent_q, {}).get(child_q, 0.0))

    def _root_score(self, question: str, c: float = 1.0, extra_visits: int = 0) -> float:
        q = question.strip()
        if q not in self.pool:
            return float("-inf")
        n_q = float(self.visit_counts.get(q, 0.0)) + float(extra_visits)
        exploit = self._node_value(q)
        explore = c * math.sqrt(math.log(self._total_visits + 1.0) / (n_q + 1.0))
        return exploit + explore + self._kin_prior(q)

    def _child_score(self, parent_q: str, child_q: str, c: float = 1.0, extra_edge_visits: int = 0) -> float:
        if child_q not in self.pool:
            return float("-inf")
        parent_visits = float(self.visit_counts.get(parent_q, 0.0))
        edge_visits = self._edge_visits(parent_q, child_q) + float(extra_edge_visits)
        exploit = self._node_value(child_q)
        explore = c * math.sqrt(math.log(parent_visits + 1.0) / (edge_visits + 1.0))
        return exploit + explore + self._kin_prior(child_q)

    def _stay_score(self, question: str, c: float = 1.0, extra_stays: int = 0) -> float:
        q = question.strip()
        if q not in self.pool:
            return float("-inf")
        parent_visits = float(self.visit_counts.get(q, 0.0))
        stay_visits = float(self.stay_visit_counts.get(q, 0.0)) + float(extra_stays)
        exploit = self._node_value(q)
        explore = c * math.sqrt(math.log(parent_visits + 1.0) / (stay_visits + 1.0))
        return exploit + explore + self._kin_prior(q)

    def _load_from_file(self, path, reset_pass_rate=False, mark_original: bool = False):
        """Internal: load seed pool entries from a JSONL file.
        reset_pass_rate=True: ignore stored pass_rate and set all entries to 0.0.
        Used on fresh start so historical pass_rates from a previous run or
        curated dataset do not bias DAG-UCT before the current model has been
        evaluated at all.
        mark_original=True: tag every loaded entry as is_original=True so the
        pruner never removes them."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    q = item['question'].strip()
                    a = item.get('answer', '').strip()

                    # [ANTIGRAVITY FIX] Filter out hallucinated/overlong seeds when loading
                    # Relaxed limits to 2000 (spec) and 3000 (impl) to accommodate legitimately complex examples
                    # Also filter out extremely short/trivial implementations (< 10 chars)
                    if len(q) > 2000 or len(a) > 3000 or len(a) < 10:
                        continue

                    pr = 0.0 if reset_pass_rate else float(item.get('pass_rate', 0.0))
                    # When mark_original is True, force is_original=True; otherwise preserve
                    # the stored field (checkpoint resume path) or default to False.
                    if mark_original:
                        is_original = True
                    else:
                        is_original = bool(item.get('is_original', False))
                    self.pool[q] = {
                        'generation_idx': item.get('generation_idx', 0),
                        'question': q,
                        'answer': a,
                        'pass_rate': pr,
                        'is_original': is_original,
                        'non_format_failures': int(item.get('non_format_failures', 0)),
                    }
                    if item.get('error_type'):
                        self.pool[q]['error_type'] = item.get('error_type')
                    self.sig_cache[q] = compute_minhash(q)
                    self._ensure_graph_node(q)
            label = " (pass_rate reset to 0.0)" if reset_pass_rate else ""
            print(f"[SEED] Loaded {len(self.pool)} deduplicated examples from {path}{label}.")
            return True
        except Exception as e:
            print(f"[WARN] Failed to load seed pool from {path}: {e}")
            return False

    def load(self):
        """Load seed pool.
        - Fresh start: load canonicalized seed_pool_all when available, with pass_rate reset to 0.0,
          so DAG-UCT evaluates all problems from scratch with the current model.
        - Checkpoint resume: load seed_pool_evolved.jsonl keeping earned pass_rates.
        """
        loaded = False
        if os.path.exists(self.evolved_seed_path):
            print(f"[SEED] Resuming from checkpoint: {self.evolved_seed_path}", flush=True)
            loaded = self._load_from_file(self.evolved_seed_path, reset_pass_rate=False)
        elif os.path.exists(self.seed_path):
            print(f"[SEED] Fresh start from base: {self.seed_path}", flush=True)
            loaded = self._load_from_file(self.seed_path, reset_pass_rate=True, mark_original=True)
        # Restore graph topology + DAG-UCT state (best-effort; non-fatal on fresh start)
        self._load_graph()
        return loaded

    def save(self):
        """Save evolved pool and graph state."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        try:
            with open(self.evolved_seed_path, 'w', encoding='utf-8') as f:
                for item in self.pool.values():
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[WARN] Failed to save seed pool: {e}")
        self._save_graph()

    def _save_graph(self):
        """Persist graph topology + DAG-UCT stats to seed_pool_graph.json."""
        try:
            serializable_edges = {
                k: {
                    'parents': list(v['parents']),
                    'children': list(v['children']),
                    'child_edge_types': v.get('child_edge_types', {}),
                    'parent_edge_types': v.get('parent_edge_types', {}),
                }
                for k, v in self.graph_edges.items()
            }
            graph_state = {
                'edges': serializable_edges,
                'kin_neighbors': {k: list(v) for k, v in self.kin_neighbors.items()},
                'visit_counts': self.visit_counts,
                'total_visits': self._total_visits,
                'edge_visit_counts': self.edge_visit_counts,
                'stay_visit_counts': self.stay_visit_counts,
                'value_sums': self.value_sums,
            }
            with open(self.graph_path, 'w', encoding='utf-8') as f:
                json.dump(graph_state, f, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Failed to save graph state: {e}")

    def _load_graph(self):
        """Restore graph topology + DAG-UCT stats from seed_pool_graph.json."""
        if not os.path.exists(self.graph_path):
            return
        try:
            with open(self.graph_path, 'r', encoding='utf-8') as f:
                graph_state = json.load(f)
            for q in self.pool:
                self._ensure_graph_node(q)
            for k, v in graph_state.get('edges', {}).items():
                self._ensure_graph_node(k)
                self.graph_edges[k] = {
                    'parents': set(v.get('parents', [])),
                    'children': set(v.get('children', [])),
                    'child_edge_types': dict(v.get('child_edge_types', {})),
                    'parent_edge_types': dict(v.get('parent_edge_types', {})),
                }
            self.kin_neighbors = {k: set(v) for k, v in graph_state.get('kin_neighbors', {}).items()}
            self.visit_counts = {k: float(v) for k, v in graph_state.get('visit_counts', {}).items()}
            self._total_visits = float(graph_state.get('total_visits', 0))
            self.edge_visit_counts = {
                parent_q: {child_q: float(cnt) for child_q, cnt in child_map.items()}
                for parent_q, child_map in graph_state.get('edge_visit_counts', {}).items()
            }
            self.stay_visit_counts = {k: float(v) for k, v in graph_state.get('stay_visit_counts', {}).items()}
            loaded_value_sums = graph_state.get('value_sums')
            if loaded_value_sums is None:
                loaded_value_sums = graph_state.get('subtree_solve_counts', {})
            self.value_sums = {k: float(v) for k, v in loaded_value_sums.items()}
            for q in self.pool:
                self._ensure_graph_node(q)
            print(f"[SEED] Loaded graph: {len(self.graph_edges)} nodes, {self._total_visits} total visits.", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to load graph state: {e}")

    def add_edge(self, src_q: str, dst_q: str, edge_type: str) -> bool:
        """Add a directed edge src_q -> dst_q in the DAG.
        src_q is the generator node; dst_q is the node it maps to (dedup target or child).
        Skips self-loops and missing nodes silently.
        """
        if src_q == dst_q:
            return False
        self._ensure_graph_node(src_q)
        self._ensure_graph_node(dst_q)
        if edge_type == 'similar_kin':
            existed = dst_q in self.kin_neighbors[src_q]
            self.kin_neighbors[src_q].add(dst_q)
            self.kin_neighbors[dst_q].add(src_q)
            return not existed
        existed = dst_q in self.graph_edges[src_q]['children']
        self.graph_edges[src_q]['children'].add(dst_q)
        self.graph_edges[src_q]['child_edge_types'][dst_q] = edge_type
        self.graph_edges[dst_q]['parents'].add(src_q)
        self.graph_edges[dst_q]['parent_edge_types'][src_q] = edge_type
        return not existed

    def question_similarity(self, spec_a: str, spec_b: str) -> float:
        a = (spec_a or "").strip()
        b = (spec_b or "").strip()
        if not a or not b:
            return 0.0
        sig_a = self.sig_cache.get(a)
        if sig_a is None:
            sig_a = compute_minhash(a)
        sig_b = self.sig_cache.get(b)
        if sig_b is None:
            sig_b = compute_minhash(b)
        return float(minhash_jaccard(sig_a, sig_b))

    def neighborhood_pass_rate(self, question: str) -> float:
        """Compute the effective pass_rate of a node's neighborhood.

        For nodes with direct pass_rate > 0, returns that value.
        For nodes with pass_rate == 0 (e.g. original roots never directly solved
        by S-Gen), returns the mean pass_rate of their children as a proxy —
        this tells us how easy S-Gen finds the neighborhood around this node.

        Returns 0.0 if the node has no children or is unknown.
        """
        q = question.strip()
        entry = self.pool.get(q)
        if entry is None:
            return 0.0
        p = entry['pass_rate']
        if p > 0.0:
            return p
        # Node's own pass_rate is 0 — use children as proxy
        children = self.graph_edges.get(q, {}).get('children', set())
        if children:
            child_prs = [self.pool[c]['pass_rate'] for c in children if c in self.pool]
            if child_prs:
                return sum(child_prs) / len(child_prs)
        # No children or no child pass_rates — fall back to global pool mean
        # so that unexplored roots get a realistic target instead of 0.0 → 0.50
        all_prs = [e['pass_rate'] for e in self.pool.values() if e['pass_rate'] > 0]
        return sum(all_prs) / len(all_prs) if all_prs else 0.15

    # Root-level UCT score used for diagnostics and root sampling.
    def ucb_score(self, question: str, c: float = 1.0, extra_visits: int = 0) -> float:
        """Root-level UCT score used for diagnostics and root sampling."""
        return self._root_score(question, c=c, extra_visits=extra_visits)

    # Traverse the curriculum DAG by comparing the local stay arm against child arms.
    def mcts_select(
        self,
        c: float = 1.0,
        batch_bonus: dict | None = None,
        edge_bonus: dict | None = None,
        stay_bonus: dict | None = None,
        descend_margin: float = 0.0,
        eligible_keys: set | None = None,
    ) -> tuple[str, list[str]]:
        """Traverse the curriculum DAG and return the selected seed plus path.

        `batch_bonus`, `edge_bonus`, and `stay_bonus` are within-batch penalties
        so repeated selections diversify across roots, edges, and stay actions.
        If `eligible_keys` is set, only those seeds are considered during traversal.
        """
        if not self.pool:
            return "", []

        active = eligible_keys if eligible_keys is not None else set(self.pool.keys())
        bonus = batch_bonus or {}
        edge_bonus = edge_bonus or {}
        stay_bonus = stay_bonus or {}

        roots = [
            q for q in active
            if q in self.pool and not any(p in self.pool for p in self.graph_edges.get(q, {}).get('parents', set()))
        ]
        if not roots:
            roots = list(active & set(self.pool.keys()))

        current = max(roots, key=lambda q: self._root_score(q, c, extra_visits=bonus.get(q, 0)))
        path = [current]

        # Compare the local stay arm against the best outgoing child arm.
        # Intermediate nodes may still be valuable seeds, especially early in
        # training when entropy is high and one more generation from the current
        # node can be more useful than forcing a descent to the frontier.
        for _ in range(20):   # depth cap to avoid O(depth) stalls
            children = [
                child for child in self.graph_edges.get(current, {}).get('children', set())
                if child in self.pool and (eligible_keys is None or child in eligible_keys)
            ]
            if not children:
                break  # leaf: expand here

            current_score = self._stay_score(current, c, extra_stays=stay_bonus.get(current, 0))
            best_child = max(
                children,
                key=lambda child: self._child_score(
                    current,
                    child,
                    c,
                    extra_edge_visits=edge_bonus.get(current, {}).get(child, 0),
                ),
            )
            best_child_score = self._child_score(
                current,
                best_child,
                c,
                extra_edge_visits=edge_bonus.get(current, {}).get(best_child, 0),
            )
            if current_score + float(descend_margin) >= best_child_score:
                break
            current = best_child
            path.append(current)

        return current, path

    # Mix root sampling with DAG traversal to produce a batch of seed paths.
    def ucb_sample(self, k: int, c: float = 1.0, root_ratio: float = 0.25, descend_margin: float = 0.0,
                   ability: float | None = None, delta: float = 0.3, min_window_frac: float = 0.15) -> list:
        """Return k seeds via mixed root sampling and DAG-UCT traversal.

        Root slots are restricted to root nodes so the original seed set stays
        active even after deeper curriculum branches appear.

        The remaining slots use DAG traversal with local stay-vs-child comparisons.

        Within-batch bonuses penalize repeating the same root, edge, or stay
        action inside one sampling call.

        If *ability* is not None, an adaptive difficulty window restricts
        sampling to seeds whose pass_rate falls in [ability - delta, ability + delta].
        Falls back to the full pool when fewer than min_window_frac of seeds pass.
        """
        if not self.pool:
            return []

        # --- Adaptive difficulty window ---
        eligible_keys = None
        if ability is not None:
            lo = max(ability - delta, 0.0)
            hi = min(ability + delta, 1.0)
            eligible_keys = set(
                q for q, it in self.pool.items()
                if lo <= float(it.get('pass_rate', 0.0)) <= hi
            )
            # Fall back to full pool if the window is too narrow
            if len(eligible_keys) < max(k, len(self.pool) * min_window_frac):
                eligible_keys = None

        roots = [
            q for q in self.pool
            if not any(p in self.pool for p in self.graph_edges.get(q, {}).get('parents', set()))
        ]
        if eligible_keys is not None:
            roots = [q for q in roots if q in eligible_keys]
        if not roots:
            roots = list(eligible_keys) if eligible_keys is not None else list(self.pool.keys())

        batch_bonus: dict[str, int] = {}
        batch_edge_bonus: dict[str, dict[str, int]] = {}
        batch_stay_bonus: dict[str, int] = {}
        results = []

        n_roots = max(1, round(k * root_ratio)) if roots else 0
        n_mcts = k - n_roots

        for _ in range(n_roots):
            best = max(roots, key=lambda q: self._root_score(q, c, extra_visits=batch_bonus.get(q, 0)))
            path = [best]
            results.append({"seed": self.pool[best], "path": path})
            for node_q in path:
                batch_bonus[node_q] = batch_bonus.get(node_q, 0) + 1
            batch_stay_bonus[best] = batch_stay_bonus.get(best, 0) + 1

        for _ in range(n_mcts):
            node, path = self.mcts_select(c, batch_bonus, batch_edge_bonus, batch_stay_bonus, descend_margin=descend_margin, eligible_keys=eligible_keys)
            results.append({"seed": self.pool[node], "path": path})
            for node_q in path:
                batch_bonus[node_q] = batch_bonus.get(node_q, 0) + 1
            for parent_q, child_q in zip(path, path[1:]):
                batch_edge_bonus.setdefault(parent_q, {})
                batch_edge_bonus[parent_q][child_q] = batch_edge_bonus[parent_q].get(child_q, 0) + 1
            batch_stay_bonus[node] = batch_stay_bonus.get(node, 0) + 1

        return results
    def record_selection_path(self, path: list[str]):
        """Record one sampled seed-selection path."""
        if not path:
            return
        self._total_visits += 1.0
        for q in path:
            self._ensure_graph_node(q)
            self.visit_counts[q] = float(self.visit_counts.get(q, 0.0)) + 1.0
        for parent_q, child_q in zip(path, path[1:]):
            self.edge_visit_counts.setdefault(parent_q, {})
            self.edge_visit_counts[parent_q][child_q] = float(self.edge_visit_counts[parent_q].get(child_q, 0.0)) + 1.0
        terminal_q = path[-1].strip()
        self.stay_visit_counts[terminal_q] = float(self.stay_visit_counts.get(terminal_q, 0.0)) + 1.0

    def backpropagate_path(self, path: list[str], observed_reward: float):
        """Backpropagate reward along the actual sampled path only.

        Credit is discounted from leaf to root so near-terminal nodes
        receive stronger signal than distant ancestors.
        """
        if not path:
            return
        reward = float(max(observed_reward, 0.0))
        gamma = float(min(max(self.path_credit_gamma, 0.0), 1.0))
        depth_from_leaf = 0
        for q in reversed(path):
            self._ensure_graph_node(q)
            discounted_reward = reward * (gamma ** depth_from_leaf if gamma < 1.0 else 1.0)
            self.value_sums[q] = float(self.value_sums.get(q, 0.0)) + discounted_reward
            depth_from_leaf += 1



    def add_or_update(self, question, answer, pass_rate, generation_idx, error_type=None, raw_response=None):
        q = question.strip()
        if not q: return False  # Only reject empty questions
        pass_rate = float(pass_rate)  # Ensure native Python float for JSON serialization

        # 1. Exact Match Handling
        if q in self.pool:
            old_pr = self.pool[q]['pass_rate']
            new_pr = max(old_pr, pass_rate)  # binary: solved stays solved
            self.pool[q]['pass_rate'] = new_pr
            if pass_rate > 0.0:
                self.pool[q]['last_solved_step'] = int(generation_idx)
            if pass_rate >= old_pr and answer:
                self.pool[q]['answer'] = answer
            # Permanently keep the best-ever CoT+answer (immune to decay)
            if answer and pass_rate > 0.0 and pass_rate >= float(self.pool[q].get('best_pass_rate', 0.0)):
                self.pool[q]['best_answer'] = answer
                self.pool[q]['best_pass_rate'] = pass_rate
                self.pool[q]['best_step'] = int(generation_idx)
                if raw_response:
                    self.pool[q]['best_raw_response'] = raw_response
            if pass_rate >= 1.0:
                self.pool[q]['non_format_failures'] = 0
                self.pool[q].pop('error_type', None) # Clear error if solved
            elif error_type and pass_rate <= 0.0:
                self.pool[q]['error_type'] = error_type
                self.pool[q]['non_format_failures'] = 0
            elif pass_rate <= 0.0:
                self.pool[q]['non_format_failures'] = int(self.pool[q].get('non_format_failures', 0)) + 1
                self.pool[q].pop('error_type', None)
            self._ensure_graph_node(q)
            return True

        # 2. Fuzzy Match Deduplication (against full evolved pool)
        #
        # Why threshold = 0.97 (not 0.85):
        #   Verus specs share heavy boilerplate: fn / ensures / forall / arr / i32 / result ...
        #   These keywords dominate the 3-gram pool, so two genuinely *different* functions
        #   that share the same skeleton (e.g. find_max vs find_min, count_pos vs count_neg)
        #   routinely score J ~= 0.85-0.93.  A threshold of 0.85 silently discards those as
        #   duplicates and collapses the pool into one family - exactly mode collapse.
        #   At 0.97 only specs that are character-for-character nearly identical are blocked;
        #   every legitimate algorithmic variant enters the pool as its own node.
        #
        # Kinship edges (J > 0.70):
        #   Questions that share heavy structure but differ in algorithm are still connected
        #   with a weak 'similar_kin' prior, but they ARE added to the pool as
        #   independent nodes so DAG-UCT can sample and train on both.
        EXACT_DUP_THRESH = 0.97   # block pool entry: virtually character-identical
        KIN_EDGE_THRESH  = 0.70   # draw kinship edge, but still add to pool

        q_sig = compute_minhash(q)
        for existing_q, existing_sig in self.sig_cache.items():
            sim = minhash_jaccard(q_sig, existing_sig)
            if sim >= EXACT_DUP_THRESH:
                # Character-identical duplicate: merge pass_rate, discard q
                self.pool[existing_q]['pass_rate'] = max(self.pool[existing_q]['pass_rate'], pass_rate)
                if pass_rate > 0.0:
                    self.pool[existing_q]['last_solved_step'] = int(generation_idx)
                    if answer and pass_rate >= float(self.pool[existing_q].get('best_pass_rate', 0.0)):
                        self.pool[existing_q]['best_answer'] = answer
                        self.pool[existing_q]['best_pass_rate'] = pass_rate
                        self.pool[existing_q]['best_step'] = int(generation_idx)
                        if raw_response:
                            self.pool[existing_q]['best_raw_response'] = raw_response
                if pass_rate >= 1.0:
                    self.pool[existing_q]['non_format_failures'] = 0
                    self.pool[existing_q].pop('error_type', None)
                elif error_type:
                    self.pool[existing_q]['error_type'] = error_type
                    self.pool[existing_q]['non_format_failures'] = 0
                elif pass_rate <= 0.0:
                    self.pool[existing_q]['non_format_failures'] = int(self.pool[existing_q].get('non_format_failures', 0)) + 1
                    self.pool[existing_q].pop('error_type', None)
                self._ensure_graph_node(existing_q)
                return False  # Treated as duplicate; q not added
            elif sim >= KIN_EDGE_THRESH:
                # Structurally similar but algorithmically distinct: add a weak
                # kinship prior, but still keep q as an independent searchable node.
                self.add_edge(existing_q, q, 'similar_kin')

        # 3. Completely New Question
        entry = {
            'generation_idx': generation_idx, 'question': q, 'answer': answer,
            'pass_rate': pass_rate,
            'is_original': False,
            'non_format_failures': 0,
        }
        if error_type and pass_rate <= 0.0:
            entry['error_type'] = error_type
        elif pass_rate <= 0.0:
            entry['non_format_failures'] = 1
        if pass_rate > 0.0:
            entry['last_solved_step'] = int(generation_idx)
            if answer:
                entry['best_answer'] = answer
                entry['best_pass_rate'] = pass_rate
                entry['best_step'] = int(generation_idx)
                if raw_response:
                    entry['best_raw_response'] = raw_response
        self.pool[q] = entry
        self.sig_cache[q] = q_sig
        self._ensure_graph_node(q)

        return True

    def decay_pass_rates(self, decay: float, current_step: int | None = None):
        """Apply per-step exponential decay to solved nodes (pass_rate > 0).

        Models catastrophic forgetting: a node that was solved once but never
        re-visited slowly drifts from upgrade zone (p=1.0) back into similar
        (0 < p < 1) and eventually downgrade (p=0.0).
        When the solver re-solves it, add_or_update sets pass_rate back to 1.0.
        All nodes with pass_rate > 0 are decayed so they don't permanently
        stick at 0.99 after the first decay step.

        Newly solved nodes keep their upgrade status for the rest of the current
        step so metrics and next-step seed sampling can observe the solved state.
        """
        for entry in self.pool.values():
            if entry['pass_rate'] > 0.0:
                if current_step is not None and int(entry.get('last_solved_step', -1)) >= int(current_step):
                    continue
                entry['pass_rate'] = round(entry['pass_rate'] * decay, 6)
        
        # Also decay DAG-UCT statistics to prevent stale signals from dominating.
        for q in self.visit_counts:
            self.visit_counts[q] = float(self.visit_counts[q]) * decay
        self._total_visits = float(self._total_visits) * decay
        for q in self.value_sums:
            self.value_sums[q] = float(self.value_sums[q]) * decay
        for q in self.stay_visit_counts:
            self.stay_visit_counts[q] = float(self.stay_visit_counts[q]) * decay
        for parent_q in self.edge_visit_counts:
            for child_q in self.edge_visit_counts[parent_q]:
                self.edge_visit_counts[parent_q][child_q] = float(self.edge_visit_counts[parent_q][child_q]) * decay

    def prune_unsolvable(self, min_visits: int = 10) -> int:
        """Remove non-original nodes whose path value has stayed at zero after
        sufficient exploration.

        Conditions for pruning:
          - is_original is False (never prune the 437 base-pool seeds)
          - value_sums == 0.0  (no positive reward has been backpropagated
            through this node)
          - visit_counts >= min_visits (enough exploration to be confident)

        Using path value instead of pass_rate ensures we keep internal nodes
        that were useful routing points for rewarded descendants.

        Also cleans up the associated DAG-UCT bookkeeping for pruned nodes.
        """
        to_prune = []
        for q, entry in self.pool.items():
            if entry.get('is_original', False):
                continue
            if self.value_sums.get(q, 0.0) > 0.0:
                continue   # positive path value observed - keep
            if self.visit_counts.get(q, 0.0) < min_visits:
                continue
            to_prune.append(q)

        for q in to_prune:
            self.pool.pop(q, None)
            self.sig_cache.pop(q, None)
            self.visit_counts.pop(q, None)
            self.value_sums.pop(q, None)
            self.edge_visit_counts.pop(q, None)
            self.stay_visit_counts.pop(q, None)
            neighbors = self.kin_neighbors.pop(q, set())
            for neighbor_q in neighbors:
                self.kin_neighbors.get(neighbor_q, set()).discard(q)
            if q in self.graph_edges:
                for parent_q in list(self.graph_edges[q].get('parents', set())):
                    self.graph_edges.get(parent_q, {}).get('children', set()).discard(q)
                    self.graph_edges.get(parent_q, {}).get('child_edge_types', {}).pop(q, None)
                    self.edge_visit_counts.get(parent_q, {}).pop(q, None)
                for child_q in list(self.graph_edges[q].get('children', set())):
                    self.graph_edges.get(child_q, {}).get('parents', set()).discard(q)
                    self.graph_edges.get(child_q, {}).get('parent_edge_types', {}).pop(q, None)
                self.graph_edges.pop(q, None)

        if to_prune:
            print(f"[SEED] Pruned {len(to_prune)} zero-value non-original nodes from pool.", flush=True)

        return len(to_prune)


# ====================================================================
# [Main Trainer] Question Solver Trainer
# ====================================================================

class QuestionSolverTrainer(RayPPOTrainer):
    def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, **kwargs):
        print(f"[QuestionSolverTrainer] Initializing trainer with config: {config.trainer.project_name}", flush=True)
        # Pop custom reward fields that the new RayPPOTrainer.__init__ no longer accepts
        self.reward_fn = kwargs.pop("reward_fn", None)
        self.val_reward_fn = kwargs.pop("val_reward_fn", None)
        
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, **kwargs)
        self.k_answers = config.algorithm.get("k_answers", 8)
        self.max_shots = config.algorithm.get("max_shots", 1)
        self.q_gen_n = config.algorithm.get("q_gen_n", 8)
        
        # [Corrective Expansion] S-Gen second chance: failed first-round answers get corrective sampling
        self.s_corrective_n = config.algorithm.get("s_corrective_n", 8)          # Number of corrective samples per failed answer
        self.s_corrective_top_m = config.algorithm.get("s_corrective_top_m", 2)  # Only expand top-M most promising failures per question
        self.s_corrective_reward_scale = config.algorithm.get("s_corrective_reward_scale", 0.5)  # Reward scale for corrective pass
        
        with open_dict(config):
            config.actor_rollout_ref.rollout.n = self.k_answers + 1

        self.val_datasets = {}
        data_dir = os.environ.get("DATA_DIR", "/workspace/verl/qsv/data")
        
        # [ANTIGRAVITY UPDATE] Use Repaired Verified Datasets (Parquet format)
        repaired_files = [
            "humaneval_verl_repaired_canonicalized.parquet",
            "mbpp_verl_repaired_canonicalized.parquet",
            "dafny2verus_verl_repaired_canonicalized.parquet",
            "seed_pool_verl_repaired_canonicalized.parquet",
        ]
        
        for fname in repaired_files:
            fpath = _pick_existing_path(
                os.path.join(data_dir, "curated_sft", fname),
                os.path.join(data_dir, fname),
                os.path.join("qsv/data/curated_sft", fname),
                os.path.join("qsv/data", fname),
            )
            
            if os.path.exists(fpath):
                try:
                    name = fname.replace("_verl_repaired_canonicalized.parquet", "")
                    self.val_datasets[name] = pd.read_parquet(fpath)
                    print(f"[QuestionSolverTrainer] Loaded validation dataset: {name} ({len(self.val_datasets[name])} samples)", flush=True)
                    print(f"Current Chat Template: {self.tokenizer.chat_template}, Validation file: {fpath}", flush=True)
                except Exception as e:
                    print(f"Error loading {fpath}: {e}", flush=True)
            else:
                print(f"[WARN] Validation file not found: {fpath}", flush=True)

        self.total_verified = 0
        self.total_solutions = 0
        self.seed_pool_mgr = SeedPoolManager(
            data_dir,
            checkpoint_dir=self.config.trainer.default_local_dir,
            downgrade_fail_threshold=config.algorithm.get("downgrade_fail_threshold", 2),
            q_gen_n=self.q_gen_n,
            kin_prior_weight=config.algorithm.get("mcts_kin_prior_weight", 0.10),
            path_credit_gamma=config.algorithm.get("path_credit_gamma", 0.70),
        )
        self.seed_pool = None

        self.q_gen_temperature = float(config.actor_rollout_ref.rollout.get("temperature", 1.0))

        # [Anti-Hack Fix 3] Running EMA baseline for Q-Gen scores
        self._q_score_ema = 0.0
        self._q_score_ema_initialized = False
        # [Adaptive Difficulty] EMA of S-Gen pass rate (solver ability estimate)
        self._s_ability_ema = 0.0
        self._s_ability_ema_alpha = float(config.algorithm.get("s_ability_ema_alpha", 0.1))
        # Question pass rate tracking history
        self.q_passrate_history = {}
        self.solverable_log_path = os.path.join(self.config.trainer.default_local_dir, "solverable_question.log")
        self.unsolverable_log_path = os.path.join(self.config.trainer.default_local_dir, "unsolverable_question.log")
        
        # Enable special handling for thinking models (e.g., DeepSeek R1, Qwen3-Thinking)
        model_path = config.actor_rollout_ref.model.get('path', '').lower()
        self.is_thinking_model = 'thinking' in model_path or 'r1' in model_path or os.environ.get('IS_THINKING_MODEL', '0') == '1'
        if self.is_thinking_model:
            print(f"[QuestionSolverTrainer] Detected THINKING model mode for path: {model_path}", flush=True)
            self.q_gen_temperature = 0.6

        # This trainer merges Q and S samples into one PPO batch and recomputes
        # old/current log-probs with a single batch-level temperature. Mixed
        # branch temperatures therefore produce incorrect PPO ratios.
        rollout_temperature = float(config.actor_rollout_ref.rollout.temperature)
        if abs(self.q_gen_temperature - rollout_temperature) > 1e-8:
            print(
                "[QuestionSolverTrainer][WARN] q_gen_temperature differs from rollout.temperature; "
                f"forcing Q rollout temperature {self.q_gen_temperature} -> {rollout_temperature} "
                "to keep PPO log-prob computation aligned.",
                flush=True,
            )
            self.q_gen_temperature = rollout_temperature

        # <think> prefix is forced by appending {"role":"assistant","content":"<think>\n"} to
        # raw_prompt messages; the agent loop detects the trailing assistant role and uses
        # continue_final_message=True so the model continues from "<think>\n" in-place.


    def _get_q_branch_warmup_scale(self) -> float:
        warm_cfg = self.config.algorithm.get("q_branch_warmup", {}) if hasattr(self.config, "algorithm") else {}
        enabled = bool(warm_cfg.get("enabled", True))
        if not enabled:
            return 1.0
        warmup_steps = max(int(warm_cfg.get("steps", 100)), 1)
        min_scale = float(warm_cfg.get("min_scale", 0.25))
        min_scale = max(0.0, min(min_scale, 1.0))
        progress = min(float(self.global_steps) / float(warmup_steps), 1.0)
        return float(min_scale + (1.0 - min_scale) * progress)

    def _get_q_gen_max_tokens(self) -> int:
        algo_cfg = self.config.algorithm if hasattr(self.config, "algorithm") else {}
        return max(int(algo_cfg.get("q_gen_max_tokens", 256)), 96)

    def _choose_qgen_direction(self, seed_spec: str, neighborhood_pr: float) -> str:
        """Pick a stable up/down curriculum direction for runtime Q-Gen.

        SFT Q-Gen data uses explicit direction labels. At RL time we route each
        seed into one direction deterministically using the current solver
        ability estimate when available, and a stable hash near the frontier so
        prompts remain aligned with the SFT format without adding randomness.
        """
        algo_cfg = self.config.algorithm if hasattr(self.config, "algorithm") else {}
        margin = max(float(algo_cfg.get("q_gen_direction_margin", 0.10)), 0.0)
        default_frontier = float(algo_cfg.get("q_gen_direction_frontier", 0.50))
        frontier = self._s_ability_ema if self._s_ability_ema > 0.0 else default_frontier

        if neighborhood_pr >= frontier + margin:
            return "up"
        if neighborhood_pr <= frontier - margin:
            return "down"

        stable = abs(hash(seed_spec.strip()))
        return "up" if (stable % 2 == 0) else "down"

    def _parse_q_candidate_response(self, raw_text: str, seed_spec: str = "", **kwargs) -> QParseResult:
        return self._parse_q_candidate_response_freeform(raw_text, seed_spec=seed_spec)

    def _parse_q_candidate_response_freeform(self, raw_text: str, seed_spec: str = "") -> QParseResult:
        algo_cfg = self.config.algorithm if hasattr(self.config, "algorithm") else {}
        hard_max_raw_chars = max(int(algo_cfg.get("q_gen_hard_max_raw_chars", 12000)), 1024)
        # 0 disables the hard think-length cap for Q-Gen. We keep raw/spec limits,
        # but do not want to choke off reasoning traces by default.
        hard_max_think_chars = max(int(algo_cfg.get("q_gen_hard_max_think_chars", 0)), 0)
        soft_max_post_chars = max(int(algo_cfg.get("q_gen_soft_max_post_chars", 1600)), 256)
        soft_max_spec_chars = max(int(algo_cfg.get("q_gen_soft_max_spec_chars", 2200)), 256)
        soft_max_spec_lines = max(int(algo_cfg.get("q_gen_soft_max_spec_lines", 32)), 8)
        hard_max_spec_chars = max(int(algo_cfg.get("q_gen_hard_max_spec_chars", 6000)), soft_max_spec_chars)
        hard_max_spec_lines = max(int(algo_cfg.get("q_gen_hard_max_spec_lines", 96)), soft_max_spec_lines)

        raw = (raw_text or "").strip()
        raw = re.sub(r"<\s*/?\s*thinking\s*>", lambda m: "</think>" if "/" in m.group(0) else "<think>", raw, flags=re.IGNORECASE)
        if not raw:
            return QParseResult(valid=False, spec="", reject_reason="empty_q_response")

        raw_char_len = len(raw)
        raw_lower = raw.lower()
        think_close_idx = raw_lower.find("</think>")
        has_q_think_open = "<think>" in raw_lower
        has_q_think_close = think_close_idx != -1
        if has_q_think_open and not has_q_think_close:
            return QParseResult(valid=False, spec="", reject_reason="q_unclosed_think", raw_char_len=raw_char_len)
        if not has_q_think_open and not has_q_think_close:
            return QParseResult(valid=False, spec="", reject_reason="q_missing_think_close", raw_char_len=raw_char_len)
        if has_q_think_open and has_q_think_close:
            think_segments = re.findall(r"<think>(.*?)</think>", raw, flags=re.DOTALL | re.IGNORECASE)
            think_char_len = sum(len(seg.strip()) for seg in think_segments)
        else:
            think_char_len = max(0, think_close_idx)
        post = _strip_think_tags(raw, is_raw_generation=True).strip()
        post_char_len = len(post)
        code_blocks = re.findall(r"```(?:rust)?\s*\n?(.*?)```", post, re.DOTALL | re.IGNORECASE)
        code_block_count = len(code_blocks)

        had_placeholder = bool(
            re.search(
                r"<\s*your novel function spec|<\s*reference implementation|<function body only",
                raw,
                re.IGNORECASE,
            )
        )
        had_prompt_echo = bool(
            re.search(
                r"mentally attempt to solve the seed|solvability:|generation directive says:|plan the new novel problem accordingly|response format:|output exactly one",
                raw,
                re.IGNORECASE,
            )
        )

        if had_placeholder:
            return QParseResult(
                valid=False,
                spec="",
                reject_reason="q_placeholder_leak",
                raw_char_len=raw_char_len,
                post_char_len=post_char_len,
                think_char_len=think_char_len,
                code_block_count=code_block_count,
                had_placeholder=True,
                had_prompt_echo=had_prompt_echo,
            )
        if raw_char_len > hard_max_raw_chars:
            return QParseResult(
                valid=False,
                spec="",
                reject_reason="q_response_too_long",
                raw_char_len=raw_char_len,
                post_char_len=post_char_len,
                think_char_len=think_char_len,
                code_block_count=code_block_count,
                had_placeholder=had_placeholder,
                had_prompt_echo=had_prompt_echo,
            )
        if hard_max_think_chars > 0 and think_char_len > hard_max_think_chars:
            return QParseResult(
                valid=False,
                spec="",
                reject_reason="q_think_too_long",
                raw_char_len=raw_char_len,
                post_char_len=post_char_len,
                think_char_len=think_char_len,
                code_block_count=code_block_count,
                had_placeholder=had_placeholder,
                had_prompt_echo=had_prompt_echo,
            )

        if code_blocks:
            candidate = extract_code_from_text(post)
            rust_idx = post.find("```rust")
            prose_prefix = post[:rust_idx].strip() if rust_idx != -1 else ""
        else:
            candidate = post
            prose_prefix = ""

        cleaned_lines = []
        drop_patterns = [
            r"^\s*\d+\.\s*mentally attempt",
            r"^\s*\d+\.\s*solvability",
            r"^\s*\d+\.\s*generation directive says",
            r"^\s*\d+\.\s*plan the new novel problem",
            r"^\s*\*\*response format",
            r"^\s*critical:",
            r"^\s*output exactly one",
            r"^\s*do not output",
            r"^\s*<\s*your novel function spec",
        ]
        for raw_line in candidate.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if any(re.search(pat, lowered) for pat in drop_patterns):
                continue
            cleaned_lines.append(line)
        candidate_clean = "\n".join(cleaned_lines).strip()
        spec, primary_meta = extract_primary_function_spec(candidate_clean)
        spec = (spec or candidate_clean).strip()
        candidate_prefix = str(primary_meta.get("prefix_source", "") or "")
        if spec and seed_spec:
            spec = _merge_freeform_spec_with_seed_prefix(seed_spec, spec, candidate_prefix=candidate_prefix).strip()
        spec_line_count = len([ln for ln in spec.splitlines() if ln.strip()])
        spec_char_len = len(spec)
        fn_count = int(primary_meta.get("fn_count", 0))
        salvaged_primary_fn = bool(primary_meta.get("salvaged", False))

        if prose_prefix and len(prose_prefix) > 40:
            return QParseResult(
                valid=False,
                spec=spec,
                reject_reason="q_prose_before_rust_block",
                raw_char_len=raw_char_len,
                post_char_len=post_char_len,
                think_char_len=think_char_len,
                code_block_count=code_block_count,
                had_placeholder=had_placeholder,
                had_prompt_echo=had_prompt_echo,
                spec_char_len=spec_char_len,
                spec_line_count=spec_line_count,
                fn_count=fn_count,
                salvaged_primary_fn=salvaged_primary_fn,
            )
        if not spec:
            return QParseResult(
                valid=False,
                spec="",
                reject_reason="empty_q_spec_after_parse",
                raw_char_len=raw_char_len,
                post_char_len=post_char_len,
                think_char_len=think_char_len,
                code_block_count=code_block_count,
                had_placeholder=had_placeholder,
                had_prompt_echo=had_prompt_echo,
            )
        if spec_char_len > hard_max_spec_chars or spec_line_count > hard_max_spec_lines:
            return QParseResult(
                valid=False,
                spec=spec,
                reject_reason="q_spec_too_long_hard",
                raw_char_len=raw_char_len,
                post_char_len=post_char_len,
                think_char_len=think_char_len,
                code_block_count=code_block_count,
                had_placeholder=had_placeholder,
                had_prompt_echo=had_prompt_echo,
                spec_char_len=spec_char_len,
                spec_line_count=spec_line_count,
                fn_count=fn_count,
                salvaged_primary_fn=salvaged_primary_fn,
            )

        format_penalty = 1.0
        # NOTE: <think> blocks are REQUIRED by the system prompt, so we do NOT
        # penalize their presence.  Only penalize their *absence* (already
        # caught earlier as q_missing_think_close / q_unclosed_think).
        if had_prompt_echo:
            format_penalty *= 0.60
        if prose_prefix:
            format_penalty *= 0.75
        if code_block_count == 0:
            format_penalty *= 0.75
        if code_block_count > 1:
            format_penalty *= max(0.45, 0.85 / float(code_block_count))
        if salvaged_primary_fn:
            format_penalty *= 0.82
        if fn_count > 1:
            format_penalty *= 0.88
        if bool(primary_meta.get("contains_main", False)):
            format_penalty *= 0.90
        if post_char_len > soft_max_post_chars:
            format_penalty *= max(0.25, float(soft_max_post_chars) / float(post_char_len))
        if spec_char_len > soft_max_spec_chars:
            format_penalty *= max(0.35, float(soft_max_spec_chars) / float(spec_char_len))
        if spec_line_count > soft_max_spec_lines:
            format_penalty *= max(0.35, float(soft_max_spec_lines) / float(spec_line_count))
        format_penalty = float(min(max(format_penalty, 0.05), 1.0))

        return QParseResult(
            valid=True,
            spec=spec,
            reject_reason="",
            format_penalty=format_penalty,
            raw_char_len=raw_char_len,
            post_char_len=post_char_len,
            think_char_len=think_char_len,
            code_block_count=code_block_count,
            had_placeholder=had_placeholder,
            had_prompt_echo=had_prompt_echo,
            spec_char_len=spec_char_len,
            spec_line_count=spec_line_count,
            fn_count=fn_count,
            salvaged_primary_fn=salvaged_primary_fn,
        )

    def _make_q_filter_result(
        self,
        valid: bool,
        score: float,
        compiler_msg: str,
        stitched_code: str,
        filter_stage: str,
        filter_reason: str,
    ) -> QFilterResult:
        return QFilterResult(
            valid=bool(valid),
            score=float(score),
            compiler_msg=str(compiler_msg),
            stitched_code=str(stitched_code),
            filter_stage=str(filter_stage),
            filter_reason=str(filter_reason),
        )

    def _q_heuristic_filter_penalty(self, reason: str) -> float:
        """Return a diagnostic score for Q filter reject reasons (logging only).

        These scores are stored in QFilterResult for diagnostics and logged to
        tracking. They do NOT enter PPO gradients (q_filter_adv_weight=0).
        """
        algo_cfg = self.config.algorithm if hasattr(self.config, "algorithm") else {}
        scale = float(algo_cfg.get("q_filter_penalty_scale", 1.0))
        penalties = {
            "q_move_missing_json": -0.18,
            "q_move_invalid_json": -0.18,
            "q_move_unknown_id": -0.16,
            "q_move_apply_failed": -0.16,
            "q_placeholder_leak": -0.35,
            "prompt_echo_in_spec": -0.30,
            "q_multiple_rust_blocks": -0.28,
            "q_response_too_long": -0.28,
            "q_think_too_long": -0.25,
            "q_spec_too_long_hard": -0.22,
            "q_prose_before_rust_block": -0.22,
            "illegal_unicode_symbol": -0.20,
            "missing_fn": -0.18,
            "missing_ensures": -0.18,
            "trivial_ensures": -0.18,
            "leaked_function_body": -0.18,
            "empty_q_response": -0.15,
            "empty_q_spec_after_parse": -0.15,
            "empty_spec": -0.15,
        }
        return float(scale * penalties.get(str(reason or "").strip(), -0.10))

    def _q_compiler_filter_penalty(self, err_cat: str) -> float:
        """Return a diagnostic score for Q compiler filter rejects (logging only)."""
        algo_cfg = self.config.algorithm if hasattr(self.config, "algorithm") else {}
        scale = float(algo_cfg.get("q_filter_penalty_scale", 1.0))
        penalties = {
            "syntax_error": -0.20,
            "type_mismatch": -0.18,
            "undefined_symbol": -0.16,
            "unsupported_method": -0.14,
            "assertion_failed": -0.08,
            "timeout": -0.06,
            "other_failure": -0.10,
        }
        return float(scale * penalties.get(str(err_cat or "").strip(), -0.10))

    def _q_valid_floor(self) -> float:
        """No reward floor: Q-Gen gradient comes exclusively from S-Gen pass rate.
        A valid spec that the solver cannot solve scores 4p(1-p) = 0, giving no
        gradient - entropy is preserved rather than driven toward copy."""
        return 0.0

    def _q_valid_style_penalty(self, parse_info: QParseResult | None) -> float:
        """Diagnostic score for valid-but-templatey Q responses (logging only).

        Stored in q_filter_scores for tracking. Does NOT enter PPO gradients
        (q_filter_adv_weight=0). Retained for monitoring proposer style drift.
        """
        if parse_info is None or not parse_info.valid:
            return 0.0

        algo_cfg = self.config.algorithm if hasattr(self.config, "algorithm") else {}
        max_penalty = float(algo_cfg.get("q_valid_style_penalty_max", 0.18))
        soft_raw_chars = max(int(algo_cfg.get("q_valid_style_soft_raw_chars", 2400)), 256)
        # 0 disables the soft think-length style penalty for Q-Gen.
        soft_think_chars = max(int(algo_cfg.get("q_valid_style_soft_think_chars", 0)), 0)

        penalty = 0.0
        penalty -= min(max(1.0 - float(parse_info.format_penalty), 0.0) * 0.18, 0.18)
        if parse_info.had_prompt_echo:
            penalty -= 0.10
        if parse_info.had_placeholder:
            penalty -= 0.10
        if parse_info.raw_char_len > soft_raw_chars:
            overflow = float(parse_info.raw_char_len - soft_raw_chars) / float(max(soft_raw_chars, 1))
            penalty -= min(0.10, 0.05 * overflow)
        if soft_think_chars > 0 and parse_info.think_char_len > soft_think_chars:
            overflow = float(parse_info.think_char_len - soft_think_chars) / float(max(soft_think_chars, 1))
            penalty -= min(0.08, 0.04 * overflow)
        if parse_info.code_block_count != 1:
            penalty -= 0.06

        return float(max(-max_penalty, penalty))

    def _q_pool_quality_gate(self, parse_info: QParseResult | None, q_debug: dict | None) -> tuple[bool, str]:
        """Decide whether a Q candidate is clean enough to enter the seed pool."""
        if parse_info is None or not parse_info.valid:
            return False, "parse_invalid"

        algo_cfg = self.config.algorithm if hasattr(self.config, "algorithm") else {}
        min_format_penalty = float(algo_cfg.get("q_pool_min_format_penalty", 0.60))
        max_raw_chars = max(int(algo_cfg.get("q_pool_max_raw_chars", 3200)), 256)
        # 0 disables the seed-pool gate on Q think length.
        max_think_chars = max(int(algo_cfg.get("q_pool_max_think_chars", 0)), 0)

        if float(parse_info.format_penalty) < min_format_penalty:
            return False, "low_format_penalty"
        if bool(parse_info.had_prompt_echo):
            return False, "prompt_echo"
        if bool(parse_info.had_placeholder):
            return False, "placeholder_leak"
        if int(parse_info.raw_char_len) > max_raw_chars:
            return False, "raw_too_long"
        if max_think_chars > 0 and int(parse_info.think_char_len) > max_think_chars:
            return False, "think_too_long"
        if int(parse_info.code_block_count) != 1:
            return False, "bad_code_block_count"

        return True, ""

    @staticmethod
    def _extract_fn_name(spec: str) -> str:
        match = re.search(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", spec or "")
        return match.group(1) if match else ""

    @staticmethod
    def _seed_question_from_meta(seed_meta: dict | None) -> str:
        if not seed_meta:
            return ""
        shots = seed_meta.get("shots", [])
        if not shots:
            return ""
        seed_dict = shots[0][0] if isinstance(shots[0], (tuple, list)) else shots[0]
        if not isinstance(seed_dict, dict):
            return ""
        return str(seed_dict.get("question", "")).strip()

    def _is_exact_duplicate_like(
        self,
        seed_spec: str,
        candidate_spec: str,
        seed_sim: float,
        struct_sim: float,
    ) -> tuple[bool, bool, str, str]:
        gate_cfg = self.config.algorithm.get("seed_pool_gate", {}) if hasattr(self.config, "algorithm") else {}
        dup_seed_sim_threshold = float(gate_cfg.get("duplicate_seed_sim_threshold", 0.92))
        dup_struct_sim_threshold = float(gate_cfg.get("duplicate_struct_sim_threshold", 0.99))
        seed_fn = self._extract_fn_name(seed_spec)
        cand_fn = self._extract_fn_name(candidate_spec)
        same_fn_name = bool(seed_fn and cand_fn and seed_fn == cand_fn)
        duplicate_like = bool(
            struct_sim >= dup_struct_sim_threshold
            or (seed_sim >= dup_seed_sim_threshold and same_fn_name)
        )
        return duplicate_like, same_fn_name, seed_fn, cand_fn


    def _run_q_static_filter(self, spec: str) -> QFilterResult | None:
        """Run only bottom-line static checks before compiler verification.

        These checks are intentionally minimal and objective:
        - check for illegal unicode symbols (hallucinations)
        - empty / missing fn
        - missing or trivial ensures
        - obvious body leakage into the spec
        """
        if not spec or not spec.strip():
            return self._make_q_filter_result(False, -0.40, "[EMPTY SPEC]", "", "heuristic", "empty_spec")

        # 1. Check for illegal unicode math symbols (Hallucinations)
        has_illegal, illegal_reason = has_illegal_characters(spec)
        if has_illegal:
            return self._make_q_filter_result(False, self._q_heuristic_filter_penalty(illegal_reason), f"[{illegal_reason}]", "", "heuristic", illegal_reason)

        # 2. Basic structural checks
        target_kind = _detect_qgen_target_kind(spec)
        if target_kind is None:
            return self._make_q_filter_result(False, self._q_heuristic_filter_penalty("missing_fn"), "[NO TARGET ITEM FOUND IN SPEC]", "", "heuristic", "missing_fn")

        if target_kind == "fn" and "ensures" not in spec:
            return self._make_q_filter_result(False, self._q_heuristic_filter_penalty("missing_ensures"), "[NO ensures CLAUSE IN SPEC]", "", "heuristic", "missing_ensures")

        if re.search(r"solvability:|generation directive says|response format:|your novel function spec", spec, re.IGNORECASE):
            return self._make_q_filter_result(False, self._q_heuristic_filter_penalty("prompt_echo_in_spec"), "[PROMPT ECHO IN SPEC]", "", "heuristic", "prompt_echo_in_spec")

        ensures_match = re.search(r'ensures\s*(.*?)\{', spec, re.DOTALL)
        if target_kind == "fn" and ensures_match:
            ensures_body = ensures_match.group(1).replace(" ", "").replace("\n", "").replace(",", "")
            if ensures_body == "true" or not ensures_body:
                return self._make_q_filter_result(False, self._q_heuristic_filter_penalty("trivial_ensures"), "[TRIVIAL ensures true]", "", "heuristic", "trivial_ensures")

        reject_body_like, reject_body_reason = has_body_like_tokens_in_spec(spec)
        if reject_body_like:
            return self._make_q_filter_result(False, self._q_heuristic_filter_penalty(reject_body_reason), f"[{reject_body_reason}]", "", "heuristic", reject_body_reason)

        return None

    def _run_q_compiler_filter(self, spec: str) -> QFilterResult:
        """Verify spec syntax/types via external_body wrapping."""
        target_kind = _detect_qgen_target_kind(spec)
        if target_kind in {"spec_fn", "proof_fn"}:
            return self._make_q_filter_result(True, 0.0, "", spec, "passed_nonexec", target_kind)

        spec_trimmed = spec.rstrip()
        stub_body = "\n    unimplemented!()\n}"
        rendered_spec = _render_external_body_spec(spec_trimmed)
        if rendered_spec.endswith('{'):
            external_fn = rendered_spec + stub_body
        else:
            external_fn = rendered_spec + " {" + stub_body
        # Close any still-open local containers preserved in the spec prefix,
        # such as `impl Foo { ... fn target(...) {`.
        extra_closer_count = max(external_fn.count("{") - external_fn.count("}"), 0)
        extra_closers = ""
        if extra_closer_count > 0:
            extra_closers = "\n" + "\n".join("}" for _ in range(extra_closer_count))
        stitched_code = "use vstd::prelude::*;\nverus! {\n" + external_fn + extra_closers + "\n} // verus!\nfn main() {}"

        try:
            verified, v_stdout, v_stderr = verify_code(stitched_code, timeout=30)
        except Exception:
            verified, v_stdout, v_stderr = False, "", "timeout"

        err_cat, compiler_out = classify_verus_error(v_stderr, v_stdout, stitched_code)
        if not verified:
            score = self._q_compiler_filter_penalty(err_cat)
            return self._make_q_filter_result(False, score, f"[{err_cat}] {compiler_out}", stitched_code, "compiler", err_cat)

        return self._make_q_filter_result(True, 0.0, "", stitched_code, "passed", "")

    def _run_q_postcompiler_heuristic_filter(self, spec: str, stitched_code: str) -> QFilterResult | None:
        """Post-compiler heuristic checks for semantically suspicious specs.

        These run AFTER the compiler passes, so the spec is syntactically valid.
        Scores are intentionally lenient (between compiler-fail and valid-floor)
        to avoid over-filtering while still providing a learning signal.

        Score hierarchy:
          static fail  : -0.50 ~ -0.30
          compiler fail: -0.30 ~ -0.10
          >>> heuristic fail: -0.15 ~ -0.05 <<<  (this layer)
          valid floor  : +0.05
        """
        target_spec = _normalize_qgen_target_source(_extract_target_fn_source(spec))
        target_kind = _detect_qgen_target_kind(target_spec)
        if target_kind in {"spec_fn", "proof_fn"}:
            return None

        # 1. Bad ensures: likely-unimplementable postconditions
        bad_ensures, bad_reason = is_bad_ensures_suspect(target_spec)
        if bad_ensures:
            return self._make_q_filter_result(
                False, -0.15,
                f"[POST-COMPILER HEURISTIC] {bad_reason}",
                stitched_code, "heuristic_postcompiler", bad_reason,
            )

        # 2. Weak spec: valid but low-information training targets
        weak_spec, weak_reason = is_weak_spec_suspect(target_spec)
        if weak_spec:
            return self._make_q_filter_result(
                False, -0.05,
                f"[POST-COMPILER HEURISTIC] {weak_reason}",
                stitched_code, "heuristic_postcompiler", weak_reason,
            )

        return None  # All checks passed

    def _evaluate_q_candidate_filter(self, spec: str) -> QFilterResult:
        # Ordering matters (3-stage pipeline):
        # 1. minimal bottom-line static checks with graded penalties
        # 2. compiler / implementability check
        # 3. post-compiler heuristic checks (bad ensures, weak spec)
        static_result = self._run_q_static_filter(spec)
        if static_result is not None:
            return static_result
        compiler_result = self._run_q_compiler_filter(spec)
        if not compiler_result.valid:
            return compiler_result
        # Compiler passed - run post-compiler heuristic checks
        heuristic_result = self._run_q_postcompiler_heuristic_filter(spec, compiler_result.stitched_code)
        if heuristic_result is not None:
            return heuristic_result
        return compiler_result  # All 3 stages passed

    # [REMOVED] Bridge, verified_unsolved, survivor, preference, move catalog — all inert mechanisms deleted in v5.45 refactor.

    @staticmethod
    def _pass_at_k(n, c, k):
        return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)

    def _get_eval_rolls(self):
        return int(self.config.trainer.get("eval_rolls", 10))

    def _get_checkpoint_eval_paths(self):
        checkpoint_root = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_root):
            checkpoint_root = os.path.join(os.getcwd(), checkpoint_root)

        checkpoint_paths = []
        if os.path.isdir(checkpoint_root):
            for path in glob.glob(os.path.join(checkpoint_root, "global_step_*")):
                if not os.path.isdir(path):
                    continue
                try:
                    step = int(os.path.basename(path).split("global_step_")[-1])
                except ValueError:
                    continue
                actor_path = os.path.join(path, "actor")
                if os.path.isdir(actor_path):
                    checkpoint_paths.append((step, actor_path))

        checkpoint_paths.sort(key=lambda x: x[0])
        return checkpoint_paths

    def _run_eval_only_all_checkpoints(self, logger):
        if not self.val_datasets:
            print("[WARN] No validation datasets available for only_eval.", flush=True)
            return

        eval_rolls = self._get_eval_rolls()

        print(f"[Eval] Running base model evaluation with n_samples={eval_rolls}", flush=True)
        self.global_steps = 0
        base_metrics = {}
        for val_name, val_df in self.val_datasets.items():
            base_metrics.update(self._validate(val_name, val_df, n_samples=eval_rolls))
        base_metrics["eval/source"] = "base_model"
        base_metrics["eval/source_step"] = 0
        logger.log(data=base_metrics, step=0)

        checkpoint_paths = self._get_checkpoint_eval_paths()
        if not checkpoint_paths:
            print("[Eval] No checkpoints found after base model evaluation.", flush=True)
            return

        for step, actor_path in checkpoint_paths:
            print(f"[Eval] Loading checkpoint for evaluation: step={step}, path={actor_path}", flush=True)
            self.actor_rollout_wg.load_checkpoint(
                actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
            self.global_steps = step
            ckpt_metrics = {}
            for val_name, val_df in self.val_datasets.items():
                ckpt_metrics.update(self._validate(val_name, val_df, n_samples=eval_rolls))
            ckpt_metrics["eval/source"] = f"checkpoint_{step}"
            ckpt_metrics["eval/source_step"] = step
            logger.log(data=ckpt_metrics, step=step)

    # -------------------------- Stage Breakdown Modules --------------------------
    def _dump_token_debug(
        self,
        step,
        stage_name,
        input_ids,
        response_ids,
        dataset_name="",
        extra_rows=None,
        input_texts=None,
        output_texts=None,
    ):
        """Debug tool: records decoded validation/generation samples to JSONL for inspection.

        When `extra_rows` is provided, it must be a list with the same length as the
        decoded batch. Each dict is merged into the corresponding JSONL entry.
        """
        try:
            local_dir = getattr(self.config.trainer, "default_local_dir", "./")
            log_dir = os.path.join(local_dir, "token_debug_logs")
            os.makedirs(log_dir, exist_ok=True)

            file_name = f"{stage_name}_{dataset_name}.jsonl" if dataset_name else f"{stage_name}.jsonl"
            log_file = os.path.join(log_dir, file_name)

            if input_texts is not None:
                in_decoded = list(input_texts)
            else:
                in_decoded = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            if output_texts is not None:
                out_decoded = list(output_texts)
            else:
                out_decoded = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            with open(log_file, "a", encoding="utf-8") as f:
                for i in range(len(in_decoded)):
                    entry = {
                        "step": step,
                        "stage": stage_name,
                        "dataset": dataset_name,
                        "idx": i,
                        "raw_input_decoded": in_decoded[i],
                        "raw_output_decoded": out_decoded[i],
                    }
                    if extra_rows and i < len(extra_rows) and isinstance(extra_rows[i], dict):
                        entry.update(extra_rows[i])
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] Failed to write token debug log for {stage_name}: {e}")

    def _dump_validation_summary(self, step, dataset_name, summary_rows):
        """Write one row per validation question with success_count / total_count."""
        try:
            if not summary_rows:
                return
            local_dir = getattr(self.config.trainer, "default_local_dir", "./")
            log_dir = os.path.join(local_dir, "token_debug_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"validate_summary_{dataset_name}.jsonl")
            with open(log_file, "a", encoding="utf-8") as f:
                for row in summary_rows:
                    entry = {"step": step, "dataset": dataset_name}
                    entry.update(row)
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] Failed to write validation summary for {dataset_name}: {e}")

    def _rollout_generate(self, input_ids_list, raw_prompts, meta_info=None, extra_non_tensor=None):
        """Centralized generation helper: pad -> generate -> unpad -> decode.
        
        Args:
            input_ids_list: List[List[int]] - tokenized prompt ids per sample.
            raw_prompts: List[List[Dict]] - chat-template messages per sample (used by vLLM agent_loop).
            meta_info: Optional dict of meta_info to set on the DataProto (e.g. temperature, stop_strings).
            extra_non_tensor: Optional dict of extra non_tensor_batch fields (e.g. uid, original_raw_prompt).
        
        Returns:
            (gen_output: DataProto, decoded_responses: List[str], timing: dict)
        """
        # Safety truncation: keep the tail so behavior matches left padding everywhere else.
        # This preserves the most recent user/error context for long multi-turn prompts.
        MAX_SAFE_PROMPT = self.config.actor_rollout_ref.rollout.get("prompt_length", 4000)
        input_ids_truncated = [ids[-MAX_SAFE_PROMPT:] for ids in input_ids_list]
        input_tensors = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_truncated]
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded_ids, padded_masks = pad_and_stack_tensors(input_tensors, pad_id, device=torch.device('cpu'), left_pad=True)
        
        batch_dict = TensorDict({
            'input_ids': padded_ids,
            'attention_mask': padded_masks
        }, batch_size=(len(input_ids_list),))
        
        non_tensor = {'raw_prompt': _to_object_array(raw_prompts)}
        if extra_non_tensor:
            non_tensor.update(extra_non_tensor)
        
        gen_batch = DataProto(batch=batch_dict, non_tensor_batch=non_tensor)
        if meta_info:
            gen_batch.meta_info.update(meta_info)
        
        num_workers = self.config.actor_rollout_ref.rollout.agent.get("num_workers", 1)
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, num_workers)
        gen_output = unpad_dataproto(
            self.async_rollout_manager.generate_sequences(gen_batch_padded), pad_size
        )
        
        timing = gen_output.meta_info.pop("timing", {})
        resp_ids = gen_output.batch.get('responses', gen_output.batch['input_ids'])
        decoded = self.tokenizer.batch_decode(resp_ids, skip_special_tokens=True)
        
        return gen_output, decoded, timing

    def _generate_questions(self, batch, timing_raw):
        """Turn 1: Generate Questions (Proposer) with DAG-UCT-guided seed sampling.

        Replaces the previous uniform random.choice() with UCB-weighted sampling
        so the generator prioritises seeds at the frontier of model capability
        (pass_rate ~= 0.5, variance term 4p(1-p) maximized) while still exploring
        under-visited nodes via the UCB exploration bonus.
        """
        with marked_timer("gen_q", timing_raw, color="red"):
            batch_size = len(batch.batch['input_ids']) if 'input_ids' in batch.batch else len(batch.non_tensor_batch.get('uid', []))
            print(f"[Stage 1 - Q-Gen] Generating questions: batch_size={batch_size}, n_samples={self.q_gen_n}", flush=True)
            turn1_prompts, current_input_ids_list, turn1_seeds = [], [], []

            # --- 1. Seed Sampling ---
            # Default mode keeps the current DAG-UCT behavior. Ablations can switch
            # to root-only or uniform sampling from the launcher without touching
            # trainer code again.
            algo_cfg = self.config.algorithm if hasattr(self.config, "algorithm") else {}
            ucb_c = float(algo_cfg.get("ucb_c", 1.0))
            ucb_root_ratio = float(algo_cfg.get("ucb_root_ratio", 0.25))
            ucb_descend_margin = float(algo_cfg.get("ucb_descend_margin", 0.0))
            seed_sampling_mode = str(algo_cfg.get("seed_sampling_mode", "ucb")).strip().lower()
            diff_delta = float(algo_cfg.get("difficulty_delta", 0.3))
            diff_enabled = bool(algo_cfg.get("adaptive_difficulty", False))
            ability = self._s_ability_ema if (diff_enabled and self._s_ability_ema > 0.0) else None
            print(f"[Stage 1 - Q-Gen] seed_sampling_mode={seed_sampling_mode}", flush=True)

            ucb_seed_items = []
            if self.seed_pool:
                if seed_sampling_mode == "ucb":
                    ucb_seed_items = self.seed_pool_mgr.ucb_sample(
                        batch_size,
                        c=ucb_c,
                        root_ratio=ucb_root_ratio,
                        descend_margin=ucb_descend_margin,
                        ability=ability,
                        delta=diff_delta,
                    )
                elif seed_sampling_mode == "root_only":
                    ucb_seed_items = self.seed_pool_mgr.ucb_sample(
                        batch_size,
                        c=ucb_c,
                        root_ratio=1.0,
                        descend_margin=ucb_descend_margin,
                        ability=ability,
                        delta=diff_delta,
                    )
                elif seed_sampling_mode in {"uniform_root", "uniform_pool"}:
                    root_keys = [
                        q for q in self.seed_pool_mgr.pool
                        if not any(p in self.seed_pool_mgr.pool for p in self.seed_pool_mgr.graph_edges.get(q, {}).get('parents', set()))
                    ]
                    candidate_keys = root_keys if seed_sampling_mode == "uniform_root" else list(self.seed_pool_mgr.pool.keys())
                    if not candidate_keys:
                        candidate_keys = list(self.seed_pool_mgr.pool.keys())
                    ucb_seed_items = [
                        {"seed": self.seed_pool_mgr.pool[random.choice(candidate_keys)], "path": []}
                        for _ in range(batch_size)
                    ]
                else:
                    raise ValueError(
                        f"Unsupported algorithm.seed_sampling_mode={seed_sampling_mode!r}. "
                        "Expected one of: ucb, root_only, uniform_root, uniform_pool."
                    )

            # --- 2. Prepare Prompts (Original Batch Size) ---
            for i in range(batch_size):
                sampled_seed = ucb_seed_items[i] if i < len(ucb_seed_items) else None

                if sampled_seed:
                    seed_item = sampled_seed['seed']
                    seed_path = sampled_seed.get('path', [seed_item['question'].strip()])
                    seed_spec = seed_item['question'].strip()
                    nbr_pr = self.seed_pool_mgr.neighborhood_pass_rate(seed_spec)
                    direction = self._choose_qgen_direction(seed_spec, nbr_pr)
                    if seed_path:
                        self.seed_pool_mgr.record_selection_path(seed_path)
                    user_content = Q_GEN_FREEFORM_CURRICULUM_TEMPLATE.format(
                        seed_code=seed_spec,
                        direction=direction,
                    )
                    turn1_seeds.append({
                        "shots": [seed_item],
                        "full_examples": [seed_item],
                        "selection_path": list(seed_path),
                        "qgen_direction": direction,
                        "qgen_neighborhood_pass_rate": nbr_pr,
                    })
                else:
                    user_content = Q_GEN_FREEFORM_CURRICULUM_TEMPLATE.format(
                        seed_code="fn target(xs: &[int]) -> (r: int)\n    ensures\n        true,\n{",
                        direction="up",
                    )
                    turn1_seeds.append({
                        "shots": [],
                        "full_examples": [],
                        "selection_path": [],
                        "qgen_direction": "up",
                        "qgen_neighborhood_pass_rate": 0.5,
                    })

                messages = [
                    {"role": "system", "content": Q_GEN_FREEFORM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": "<think>\n"},
                ]
                turn1_prompts.append(messages)

                chat_ids = to_list_of_ints(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        continue_final_message=True,
                        add_generation_prompt=False,
                        add_special_tokens=False,
                    )
                )
                current_input_ids_list.append(chat_ids)
                

            expanded_input_ids_list = []
            for ids in current_input_ids_list:
                expanded_input_ids_list.extend([ids] * self.q_gen_n)
                
            expanded_turn1_prompts = []
            expanded_turn1_seeds = []
            for p, s in zip(turn1_prompts, turn1_seeds):
                expanded_turn1_prompts.extend([p] * self.q_gen_n)
                expanded_turn1_seeds.extend([s] * self.q_gen_n)
            
            # Propagate non_tensor_batch attributes from parent batch
            extra_non_tensor = {}
            if hasattr(batch, 'non_tensor_batch'):
                for key in ['data_source', 'reward_model', 'ability', 'extra_info', 'uid']:
                    if key in batch.non_tensor_batch:
                        parent_vals = batch.non_tensor_batch[key]
                        expanded_parent_vals = np.repeat(parent_vals, self.q_gen_n) if isinstance(parent_vals, (np.ndarray, list)) else parent_vals
                        extra_non_tensor[key] = _to_object_array(expanded_parent_vals)
                        
            # --- 3. Execute Generation (Rollout) ---
            gen_batch_q_output, generated_q_bodies, q_timing = self._rollout_generate(
                input_ids_list=expanded_input_ids_list,
                raw_prompts=expanded_turn1_prompts,
                meta_info={
                    "global_steps": self.global_steps, 
                    "stop_strings": ["<|im_start|>assistant", "<|im_end|>", "```rust\nfn main", "\nfn main(", "\nfn main", "\n[", "New Problem"],
                    "temperature": self.q_gen_temperature,
                    "max_tokens": self._get_q_gen_max_tokens(),
                },
                extra_non_tensor=extra_non_tensor
            )
            timing_raw.update(q_timing)
            
            # --- 4. Verify All Candidates ---
            print(f"[Stage 1 - Q-Gen] Verifying {len(generated_q_bodies)} candidates...", flush=True)
            
            # Parse & Prepare
            q_parse_results = [
                self._parse_q_candidate_response(
                    generated_q_bodies[i],
                    self._seed_question_from_meta(expanded_turn1_seeds[i] if i < len(expanded_turn1_seeds) else None),
                )
                for i in range(len(generated_q_bodies))
            ]
            generated_specs = [res.spec for res in q_parse_results]

            q_filter_results = [None] * len(q_parse_results)
            parse_valid_indices = [i for i, res in enumerate(q_parse_results) if res.valid]
            parse_valid_specs = [q_parse_results[i].spec for i in parse_valid_indices]
            parse_valid_filter_results = []
            if parse_valid_specs:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(parse_valid_specs))) as executor:
                    parse_valid_filter_results = list(executor.map(self._evaluate_q_candidate_filter, parse_valid_specs))
            parse_valid_iter = iter(parse_valid_filter_results)
            for idx, parse_res in enumerate(q_parse_results):
                if not parse_res.valid:
                    q_filter_results[idx] = self._make_q_filter_result(
                        False,
                        self._q_heuristic_filter_penalty(parse_res.reject_reason),
                        f"[Q FORMAT] {parse_res.reject_reason}",
                        "",
                        "heuristic",
                        parse_res.reject_reason,
                    )
                else:
                    q_filter_results[idx] = next(parse_valid_iter)

            # --- 5. Compute N-gram Similarity vs Seed Shots (S_max per candidate) ---
            # expanded_turn1_seeds is B*N list; indexing aligns with B*N candidates
            # [Anti-Hack Fix 1] S_max = max similarity vs BOTH few-shot seeds AND entire seed pool
            pool_sig_cache = self.seed_pool_mgr.sig_cache
            sim_scores = [
                max(
                    spec_sim_to_seeds(
                        generated_specs[i],
                        expanded_turn1_seeds[i].get("shots", []) if isinstance(expanded_turn1_seeds[i], dict) else expanded_turn1_seeds[i],
                        pool_sig_cache,
                    ),
                    spec_sim_to_pool(generated_specs[i], pool_sig_cache)
                )
                for i in range(len(generated_specs))
            ]

            q_valid_flags = [res.valid for res in q_filter_results]
            q_err_scores = [res.score for res in q_filter_results]
            q_err_compiler_outs = [res.compiler_msg for res in q_filter_results]
            q_stitched_codes = [res.stitched_code for res in q_filter_results]
            q_filter_stages = [res.filter_stage for res in q_filter_results]
            q_filter_reasons = [res.filter_reason for res in q_filter_results]

            format_filtered = sum(1 for res in q_parse_results if not res.valid)
            heuristic_filtered = sum(
                1 for res in q_filter_results
                if not res.valid and res.filter_stage == "heuristic" and not str(res.filter_reason).startswith("q_")
            )
            compiler_filtered = sum(1 for res in q_filter_results if not res.valid and res.filter_stage == "compiler")
            heuristic_postcompiler_filtered = sum(1 for res in q_filter_results if not res.valid and res.filter_stage == "heuristic_postcompiler")
            bad_ensures_rejects = sum(1 for res in q_filter_results if not res.valid and "BAD ENSURES" in res.compiler_msg)
            total_valid = sum(q_valid_flags)
            total_candidates = len(q_valid_flags)
            solver_samples = total_valid * self.k_answers
            mean_sim = float(np.mean(sim_scores)) if sim_scores else 0.0

            print(
                f"[Stage 1 - Q-Gen] Filtered candidates: format={format_filtered}, heuristic={heuristic_filtered}, "
                f"compiler={compiler_filtered}, heuristic_postcompiler={heuristic_postcompiler_filtered}, "
                f"to_solver_questions={total_valid}/{total_candidates}, "
                f"to_solver_samples={solver_samples}.",
                flush=True,
            )
            # Print one sample Q-Compiler error per step for sanity checking
            for res in q_filter_results:
                if not res.valid and res.filter_stage == "compiler":
                    print(f"[Q-Compiler-Err] score={res.score:.2f}  {res.compiler_msg[:200]!r}", flush=True)
                    break
            print(f"[Stage 1 - Q-Gen] Seed Similarity: mean={mean_sim:.3f}, max={max(sim_scores, default=0.0):.3f}", flush=True)

            # --- 6. Return All Candidates (Refactored for Batch Structure) ---
            # Instead of Best-of-N selection, we return ALL B*N candidates.
            # Output Batch Size: B * N
            
            # Reconstruct Prompt + Gen for logging/debugging (for ALL candidates)
            final_generated_qs = []
            final_generated_specs_masked = []
            final_generated_q_bodies = []
            final_q_valid_flags = q_valid_flags # List of B*N bools
            
            for i in range(len(generated_q_bodies)):
                final_q_body = generated_q_bodies[i]
                final_q_full = self.tokenizer.apply_chat_template(
                    expanded_turn1_prompts[i],
                    tokenize=False,
                    continue_final_message=True,
                    add_generation_prompt=False,
                    add_special_tokens=False,
                ) + final_q_body
                final_spec_masked = prepare_prefix_prompt(generated_specs[i])
                
                final_generated_qs.append(final_q_full)
                final_generated_specs_masked.append(final_spec_masked)
                final_generated_q_bodies.append(final_q_body)

            # --- 7. Finalize Output DataProto (KeyError Fix) ---
            # We re-attach 'prompts' (Prompt-Only) to ensure correct response_mask computation.
            parent_pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            # Use same device and padding as input_ids
            max_safe_prompt = self.config.actor_rollout_ref.rollout.get("prompt_length", 4000)
            q_prompt_tensors = [torch.tensor(p[-max_safe_prompt:], dtype=torch.long) for p in expanded_input_ids_list]
            q_prompts_padded, _ = pad_and_stack_tensors(q_prompt_tensors, parent_pad_id, device=gen_batch_q_output.batch['input_ids'].device, left_pad=True)
            gen_batch_q_output.batch['prompts'] = q_prompts_padded
            gen_batch_q_output.batch['is_question'] = torch.ones(len(expanded_input_ids_list), dtype=torch.bool, device=gen_batch_q_output.batch['input_ids'].device)
            gen_batch_q_output.non_tensor_batch['raw_prompt'] = _to_object_array(expanded_turn1_prompts)

            return QGenResult(
                gen_output=gen_batch_q_output,
                turn1_prompts=expanded_turn1_prompts,
                turn1_seeds=expanded_turn1_seeds,
                generated_qs=final_generated_qs,
                generated_specs_masked=final_generated_specs_masked,
                generated_q_bodies=final_generated_q_bodies,
                q_valid_flags=final_q_valid_flags,
                q_err_scores=q_err_scores,
                sim_scores=sim_scores,
                q_err_compiler_outs=q_err_compiler_outs,
                q_stitched_codes=q_stitched_codes,
                q_parse_results=q_parse_results,
                filter_stages=q_filter_stages,
                filter_reasons=q_filter_reasons,
                q_filter_results=q_filter_results,
                format_rejects=format_filtered,
                heuristic_rejects=heuristic_filtered,
                compiler_rejects=compiler_filtered,
                heuristic_postcompiler_rejects=heuristic_postcompiler_filtered,
                bad_ensures_rejects=bad_ensures_rejects,
                candidate_count=total_candidates,
            )




    def _generate_solutions(self, batch, generated_specs_masked, timing_raw, q_valid_flags=None, q_seeds=None):
        """Turn 2: Generate Solutions (Solver) - Sparse Batch Mode
        
        Only generates solutions for VALID questions.
        Returns a "Ragged" / Sparse batch of size [N_valid * K].
        """
        with marked_timer("gen_s", timing_raw, color="magenta"):
            total_batch_size = len(generated_specs_masked) # B * N
            
            # 1. Filter Valid Indices
            valid_indices = [i for i, v in enumerate(q_valid_flags) if v] if q_valid_flags else list(range(total_batch_size))
            n_valid = len(valid_indices)
            
            n_solver_samples = n_valid * self.k_answers
            print(
                f"[Stage 2 - S-Gen] Valid questions: {n_valid}/{total_batch_size}. "
                f"Sampling k={self.k_answers} solutions per valid question -> {n_solver_samples} solver samples.",
                flush=True,
            )

            # --- Case: All Invalid (Return Empty) ---
            if not valid_indices:
                print(f"[Stage 2 - S-Gen] All questions invalid. Returning empty batch.", flush=True)
                # Create empty DataProto
                device = batch.batch.get('input_ids', torch.empty(0)).device
                empty_tensor = torch.empty((0, 1), dtype=torch.long, device=device)
                s_batch_dict = TensorDict({'input_ids': empty_tensor, 'attention_mask': empty_tensor, 'responses': empty_tensor}, batch_size=(0,))
                s_gen_output = DataProto(batch=s_batch_dict, non_tensor_batch={'raw_prompt': _to_object_array([])})
                return s_gen_output, [], s_gen_output

            # 2. Prepare Valid Prompts
            turn2_prompts_valid = []
            for i in valid_indices:
                spec_code = generated_specs_masked[i]
                val_usr_msg = SOLVER_USER_MSG_TEMPLATE.format(spec_code=spec_code)

                target_msgs = [
                    {"role": "system", "content": S_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": val_usr_msg}
                ]
                turn2_prompts_valid.append(target_msgs)

            # 3. Tokenize & Pad Valid Batch
            device = batch.batch.get('input_ids', torch.empty(0)).device
            input_ids_list = []
            
            for p in turn2_prompts_valid:
                p.append({"role": "assistant", "content": "<think>\n"})
                ids = to_list_of_ints(self.tokenizer.apply_chat_template(p, tokenize=True, continue_final_message=True, add_generation_prompt=False, add_special_tokens=False))
                input_ids_list.append(ids)

            input_tensors = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list]
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            valid_input_ids, valid_attention_mask = pad_and_stack_tensors(input_tensors, pad_id, device=device, left_pad=True)
            
            # 4. Create Valid DataProto & Repeat K times
            valid_batch_dict = TensorDict({
                'input_ids': valid_input_ids, 
                'attention_mask': valid_attention_mask, 
                'prompts': valid_input_ids
            }, batch_size=(n_valid,))
            
            gen_batch_s_valid = DataProto(batch=valid_batch_dict, non_tensor_batch={'raw_prompt': _to_object_array(turn2_prompts_valid)})
            
            # Propagate extra info from parent batch (Sparse Slice)
            if hasattr(batch, 'non_tensor_batch'):
                 for key in ['data_source', 'reward_model', 'ability', 'extra_info', 'uid']:
                    if key in batch.non_tensor_batch:
                        parent_vals = batch.non_tensor_batch[key]
                        
                        # Expand B -> B*N
                        if isinstance(parent_vals, (list, np.ndarray)):
                             expanded_parent_vals = np.repeat(parent_vals, self.q_gen_n) if isinstance(parent_vals, np.ndarray) else [x for x in parent_vals for _ in range(self.q_gen_n)]
                        else:
                             expanded_parent_vals = parent_vals
                             
                        # Slice to Valid
                        if isinstance(expanded_parent_vals, (list, np.ndarray)):
                             sliced = [expanded_parent_vals[i] for i in valid_indices] if isinstance(expanded_parent_vals, list) else expanded_parent_vals[valid_indices]
                             gen_batch_s_valid.non_tensor_batch[key] = sliced
            
            # Repeat K times
            gen_batch_s_valid = gen_batch_s_valid.repeat(repeat_times=self.k_answers, interleave=True)
            gen_batch_s_valid.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
            gen_batch_s_valid.meta_info['stop_strings'] = [
                "<|im_start|>assistant",
                "<|im_end|>",
                "```rust\nfn ",
                "\nfn main",
                "\nHere is",
                "\nExplanation:",
                "<reference implementation",
            ]

            # 5. Execute rollout (returns sparse batch) - pad for worker divisibility
            num_workers = self.config.actor_rollout_ref.rollout.agent.get("num_workers", 1)
            gen_batch_s_padded, s_pad_size = pad_dataproto_to_divisor(gen_batch_s_valid, num_workers)
            gen_batch_s_output = unpad_dataproto(self.async_rollout_manager.generate_sequences(gen_batch_s_padded), s_pad_size)
            
            # --- KeyError Fix: Re-attach Prompts (use original unpadded input_ids) ---
            gen_batch_s_output.batch['prompts'] = gen_batch_s_valid.batch['input_ids']
            gen_batch_s_output.batch['is_question'] = torch.zeros(len(gen_batch_s_output), dtype=torch.bool, device=gen_batch_s_output.batch['input_ids'].device)
            
            timing_raw.update(gen_batch_s_output.meta_info.get("timing", {}))
            
            # Decode Sparse
            response_ids = gen_batch_s_output.batch.get('responses', gen_batch_s_output.batch['input_ids'])
            decoded_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            return gen_batch_s_output, decoded_responses, gen_batch_s_valid

    def _verify_solutions(self, generated_q_bodies, decoded_responses, turn1_prompts, generated_qs, metrics, timing_raw, q_valid_flags=None, q_err_scores=None, q_sim_scores=None, q_err_compiler_outs=None, generated_specs_masked=None, q_stitched_codes=None, q_filter_results=None, q_seeds=None, q_parse_results=None):
        """Turn 3: Verify Solutions & Compute Scores (Sparse Batch Handler)"""
        with marked_timer("verify_s", timing_raw, color="yellow"):
            # generated_q_bodies: [B * N]
            # decoded_responses: [N_valid * K] (Sparse)
            
            total_qs = len(generated_q_bodies)
            
            solve_scores = []     # Will contain [N_valid * K] scores
            question_scores = []  # Will contain [B * N] scores
            pass_rates = []       # Will contain [B * N] pass rates
            s_reward_signals = [] # Will contain [N_valid] flags
            q_reward_signals = [] # Will contain [B] flags
            q_base_scores = [0.0] * total_qs
            q_filter_scores = [0.0] * total_qs
            q_struct_sims = [0.0] * total_qs
            q_seed_sims_effective = [0.0] * total_qs
            q_format_penalties = [1.0] * total_qs
            
            clean_answers = []    # Sparse [N_valid * K]
            full_codes = []       # Sparse [N_valid * K]
            verus_outputs = []    # Sparse [N_valid * K]
            strict_passes = []    # Sparse [N_valid * K]
            debug_infos = []      # Dense [B * N]
            
            parent_pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            # Iterator for sparse responses
            resp_iter = iter(decoded_responses)
            
            # 1. Prepare Verification Tasks (Only for Valid)
            verify_tasks = []
            
            # We need to map tasks back to their origin for reconstruction, 
            # but simpler to just process sparse list and then map scores?
            # Actually, we need to know WHICH Q body corresponds to the response.
            
            # Use `q_valid_flags` to skip invalid Qs
            valid_indices = [i for i, v in enumerate(q_valid_flags) if v] if q_valid_flags else list(range(total_qs))
            
            # Construct tasks aligned with `decoded_responses`
            # `decoded_responses` is flattened [Valid_Q1_K1, Valid_Q1_K2, ..., Valid_Q2_K1...]
            # Use pre-computed generated_specs_masked directly (already extract_spec_and_impl +
            # prepare_prefix_prompt applied) to avoid re-extracting from q_body and risking
            # picking up the wrong rust block when q_body has been prefixed with '# Reasoning\n'.
            
            for q_idx in valid_indices:
                # Prefer the already-prepared spec_masked; fall back to re-extracting from q_body
                if generated_specs_masked and q_idx < len(generated_specs_masked):
                    spec_for_verify = generated_specs_masked[q_idx]
                else:
                    spec_raw, _ = extract_spec_and_impl(generated_q_bodies[q_idx])
                    spec_for_verify = prepare_prefix_prompt(spec_raw)
                for _ in range(self.k_answers):
                    try:
                        resp = next(resp_iter)
                    except StopIteration:
                        resp = "" # Error fallback
                        
                    verify_tasks.append((spec_for_verify, resp))

            # 2. Parallel Verification
            if verify_tasks:
                print(f"[Stage 3 - Verify] Verifying {len(verify_tasks)} solutions...", flush=True)
                
                # Define helper locally or use lambda if simple
                def process_verification(args):
                    # args[0] is already the prepared spec_masked (truncated to opening `{`)
                    # args[1] is the raw S-Gen response (function body only)
                    spec_masked, a_text = args
                    
                    # 1. Format Validation (Hard Filter)
                    # We must ensure the model follows formatting rules BEFORE trying to compile.
                    is_format_ok, format_penalty, format_reason = validate_response_format(
                        a_text,
                        continuation_from_think=True,
                        require_body_only=True,
                    )
                    if not is_format_ok:
                        # Format failures score 0: filter is a hard gate, not a gradient source.
                        return 0.0, False, "", "", f"[{format_reason}] Format failed."

                    stitched_code, _, clean_body = smart_stitch(spec_masked, a_text, response_is_raw=True)

                    verified, v_stdout, v_stderr = False, "", ""
                    
                    if stitched_code and stitched_code.strip():
                        # Minimal cleanup
                        if "use builtin" in stitched_code:
                            stitched_code = stitched_code.replace("use builtin::*;", "// ...").replace("use builtin_macros::*;", "// ...")
                        
                        # Verify
                        try:
                            verified, v_stdout, v_stderr = verify_code(stitched_code, timeout=30)
                        except:
                            verified, v_stdout, v_stderr = False, "", "timeout"
                    
                    # Compute dense reward for S-Gen
                    compiler_out = (v_stderr or v_stdout or "").strip()[:2000]
                    if verified:
                        s_score = 1.0
                        err_cat = "verified"
                    else:
                        err_cat, _ = classify_verus_error(v_stderr, v_stdout, stitched_code)
                        s_score = 0.0  # all failures: binary 0/1 reward

                    stdout_tail = (v_stdout or "").strip()[-120:]
                    return s_score, bool(verified), stitched_code, clean_body, f"[{err_cat}] {compiler_out}|||stdout_tail={stdout_tail}"

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(verify_tasks))) as executor:
                     verify_results = list(executor.map(process_verification, verify_tasks))
                # Print one sample partial-reward line per step for sanity checking
                for _sc, _sp, _fc, _cb, _vo in verify_results:
                    if "[assertion_failed]" in _vo:
                        print(f"[S-partial] score={_sc:.2f}  {_vo[:200]!r}", flush=True)
                        break
            else:
                verify_results = []
            # =====================================================
            # 2b. [Corrective Expansion] Second chance for failed S-Gen
            # For questions where first-round pass_rate=0, take top-M failed
            # samples and do corrective expansion (with error feedback).
            # If any corrective sample passes, signal propagates back.
            # =====================================================
            corrective_results = {}  # Map: valid_q_idx -> {corrective_pass: bool, corrective_pass_rate: float, best_solution: str}
            corrective_batch_outputs = []  # Store DataProto for corrective samples (for PPO training)
            corrective_all_scores = []  # All corrective scores for GRPO
            
            if self.s_corrective_n > 0 and verify_results:
                # Group verify_results by question: each valid Q has K consecutive results
                n_valid = len(valid_indices)
                q_first_round_results = {}  # valid_q_local_idx -> list of (verify_idx, score, error_out)
                for vq_local_idx in range(n_valid):
                    base = vq_local_idx * self.k_answers
                    entries = []
                    for k_idx in range(self.k_answers):
                        vi = base + k_idx
                        if vi < len(verify_results):
                            sc, sp, fc, cb, vo = verify_results[vi]
                            entries.append((vi, sc, sp, vo))
                    q_first_round_results[vq_local_idx] = entries
                
                # Identify questions with pass_rate=0 after the first verification pass
                corrective_candidates = []
                for vq_local_idx, entries in q_first_round_results.items():
                    any_pass = any(sp for _, _, sp, _ in entries)
                    if not any_pass and entries:
                        # Sort by score descending (most promising failures first)
                        ranked = sorted(entries, key=lambda x: x[1], reverse=True)
                        # Take top-M
                        top_m = ranked[:self.s_corrective_top_m]
                        corrective_candidates.append((vq_local_idx, top_m))
                
                if corrective_candidates:
                    total_corrective_roots = sum(len(top_m) for _, top_m in corrective_candidates)
                    total_corrective_samples = total_corrective_roots * self.s_corrective_n
                    total_corrective = total_corrective_roots
                    print(
                        f"[Stage 3 - Corrective] Zero-pass questions: {len(corrective_candidates)}. "
                        f"Selected top-{self.s_corrective_top_m} failed answers per zero-pass question -> "
                        f"{total_corrective_roots} corrective roots. "
                        f"Sampling n={self.s_corrective_n} fixes per root -> {total_corrective_samples} corrective samples.",
                        flush=True,
                    )
            # Corrective example dumps are intentionally suppressed in normal training logs.
                    
                    # Build corrective prompts
                    corrective_input_ids_list = []
                    corrective_raw_prompts = []
                    corrective_map = []  # (vq_local_idx, verify_idx) for tracking
                    
                    for vq_local_idx, top_m_entries in corrective_candidates:
                        for verify_idx, sc, sp, error_out in top_m_entries:
                            spec_masked = verify_tasks[verify_idx][0]
                            orig_response = verify_tasks[verify_idx][1]
                            # Use CLEANED code body instead of raw (potentially huge) reasoning block
                            clean_body = verify_results[verify_idx][3] or ""
                            # Format nicely for the model
                            assistant_content = f"```rust\n{clean_body.strip()}\n```" if clean_body.strip() else orig_response

                            # Parse error to extract line-level information
                            # Verus errors often contain "error[E0XXX]: message\n --> filename:LINE:COL"
                            error_msg = error_out[:1500]
                            
                            # Build corrective prompt with error context pointing to code location
                            corrective_msgs = [
                                {"role": "system", "content": S_GEN_SYSTEM_PROMPT},
                                {"role": "user", "content": SOLVER_USER_MSG_TEMPLATE.format(spec_code=spec_masked)},
                                {"role": "assistant", "content": assistant_content},
                                {"role": "user", "content": (
                                    f"Your solution failed Verus verification. Here is the compiler output:\n"
                                    f"```\n{error_msg}\n```\n\n"
                                    f"Please carefully analyze the error messages above. Pay attention to:\n"
                                    f"- Line numbers and column positions in the error output\n"
                                    f"- The specific verification condition that failed (requires/ensures/invariant)\n"
                                    f"- Any type mismatches or undefined variables\n\n"
                                    f"Fix ONLY the function body. Output the corrected implementation in a single ```rust block."
                                )}
                            ]
                            corrective_raw_prompts.append(corrective_msgs)
                            corrective_map.append((vq_local_idx, verify_idx))
                            
                            corrective_msgs.append({"role": "assistant", "content": "<think>\n"})
                            ids = to_list_of_ints(self.tokenizer.apply_chat_template(
                                corrective_msgs, tokenize=True, continue_final_message=True, add_generation_prompt=False, add_special_tokens=False
                            ))
                            corrective_input_ids_list.append(ids)
                    
                    # Expand each corrective prompt by s_corrective_n
                    expanded_corrective_ids = []
                    expanded_corrective_prompts = []
                    expanded_corrective_map = []
                    for idx, ids in enumerate(corrective_input_ids_list):
                        for _ in range(self.s_corrective_n):
                            expanded_corrective_ids.append(ids)
                            expanded_corrective_prompts.append(corrective_raw_prompts[idx])
                            expanded_corrective_map.append(corrective_map[idx])
                    
                    # Generate corrective solutions
                    corrective_output, corrective_bodies, corrective_timing = self._rollout_generate(
                        input_ids_list=expanded_corrective_ids,
                        raw_prompts=expanded_corrective_prompts,
                        meta_info={
                            "global_steps": self.global_steps,
                            "temperature": self.config.actor_rollout_ref.rollout.temperature,
                            "stop_strings": [
                                "<|im_start|>assistant",
                                "<|im_end|>",
                                "```rust\nfn ",
                                "\nfn main",
                                "\nHere is",
                                "\nExplanation:",
                                "<reference implementation",
                            ],
                        }
                    )
                    timing_raw.update(corrective_timing)
                    
                    # Verify corrective solutions
                    corrective_verify_tasks = []
                    for idx, (vq_local_idx, verify_idx) in enumerate(expanded_corrective_map):
                        spec_masked = verify_tasks[verify_idx][0]
                        corrective_verify_tasks.append((spec_masked, corrective_bodies[idx]))
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(corrective_verify_tasks))) as executor:
                        corrective_verify_results = list(executor.map(process_verification, corrective_verify_tasks))
                    
                    # Aggregate corrective results per original failure
                    n_per_failure = self.s_corrective_n
                    n_failures = len(corrective_input_ids_list)
                    corrective_success_total = 0
                    corrective_question_success_total = 0
                    kept_failure_indices = []
                    
                    for failure_idx in range(n_failures):
                        base = failure_idx * n_per_failure
                        vq_local_idx, orig_verify_idx = corrective_map[failure_idx]
                        
                        c_scores = [corrective_verify_results[base + j][0] for j in range(n_per_failure)]
                        c_strict = [corrective_verify_results[base + j][1] for j in range(n_per_failure)]
                        c_v_outs = [corrective_verify_results[base + j][4] for j in range(n_per_failure)]
                        c_pass = sum(1 for sp in c_strict if sp)

                        # Store per-question corrective results
                        if vq_local_idx not in corrective_results:
                            corrective_results[vq_local_idx] = {
                                'corrective_pass': False,
                                'corrective_pass_rate': 0.0,
                                'best_solution': '',
                                'best_score': float("-inf"),
                                'corrective_scores': [],
                                'corrective_strict_passes': [],
                                'corrective_v_outs': [],
                            }

                        corrective_results[vq_local_idx]['corrective_scores'].extend(c_scores)
                        corrective_results[vq_local_idx]['corrective_strict_passes'].extend(c_strict)
                        corrective_results[vq_local_idx]['corrective_v_outs'].extend(c_v_outs)                        
                        if c_pass > 0:
                            kept_failure_indices.append(failure_idx)
                            if not corrective_results[vq_local_idx]['corrective_pass']:
                                corrective_question_success_total += 1
                            corrective_results[vq_local_idx]['corrective_pass'] = True
                            corrective_success_total += 1
                            # Update the ORIGINAL verify_result with scaled reward
                            orig_sc, orig_sp, orig_fc, orig_cb, orig_vo = verify_results[orig_verify_idx]
                            scaled_reward = self.s_corrective_reward_scale  # e.g., 0.5
                            verify_results[orig_verify_idx] = (
                                scaled_reward, orig_sp, orig_fc, orig_cb,
                                orig_vo + f" [corrective {c_pass}/{n_per_failure}]"
                            )
                            # Find best corrective solution
                            best_idx = max(range(n_per_failure), key=lambda j: corrective_verify_results[base + j][0])
                            best_solution = corrective_verify_results[base + best_idx][3]  # clean_body
                            best_score = corrective_verify_results[base + best_idx][0]
                            if best_score > corrective_results[vq_local_idx]['best_score']:
                                corrective_results[vq_local_idx]['best_solution'] = best_solution
                                corrective_results[vq_local_idx]['best_score'] = best_score
                    
                    # Compute corrective pass_rate per Q
                    for vq_local_idx, cr in corrective_results.items():
                        all_c_strict = cr['corrective_strict_passes']
                        cr['corrective_pass_rate'] = sum(1 for sp in all_c_strict if sp) / max(len(all_c_strict), 1)
                    
                    # Only keep corrective roots that had at least one successful fix.
                    # This avoids spending PPO budget on groups that are pure negative signal.
                    kept_sample_indices = []
                    for failure_idx in kept_failure_indices:
                        base = failure_idx * n_per_failure
                        kept_sample_indices.extend(range(base, base + n_per_failure))

                    corrective_roots_kept_for_ppo = len(kept_failure_indices)
                    corrective_samples_kept_for_ppo = len(kept_sample_indices)

                    # Corrective GRPO: store only kept corrective scores for advantage computation
                    corrective_all_scores = [
                        corrective_verify_results[idx][0] * self.s_corrective_reward_scale
                        for idx in kept_sample_indices
                    ]
                    
                    metrics.update({
                        "corrective/n_questions": len(corrective_candidates),
                        "corrective/n_roots": total_corrective_roots,
                        "corrective/n_expanded": total_corrective,
                        "corrective/n_expanded_samples": total_corrective_samples,
                        "corrective/n_roots_kept_for_ppo": corrective_roots_kept_for_ppo,
                        "corrective/n_samples_kept_for_ppo": corrective_samples_kept_for_ppo,
                        "corrective/success_total": corrective_success_total,
                        "corrective/success_rate": corrective_success_total / max(total_corrective, 1),
                        "corrective/success_questions": corrective_question_success_total,
                        "corrective/success_question_rate": corrective_question_success_total / max(len(corrective_candidates), 1),
                    })
                    print(
                        f"[Stage 3 - Corrective] Successful roots: {corrective_success_total}/{total_corrective_roots}; "
                        f"questions rescued: {corrective_question_success_total}/{len(corrective_candidates)}. "
                        f"PPO kept roots/samples: {corrective_roots_kept_for_ppo}/{total_corrective_roots} roots, "
                        f"{corrective_samples_kept_for_ppo}/{total_corrective_samples} samples.",
                        flush=True,
                    )
                    
                    # Store corrective DataProto for PPO batch if needed
                    if (
                        corrective_output is not None
                        and len(corrective_output.batch['input_ids']) > 0
                        and corrective_samples_kept_for_ppo > 0
                    ):
                        keep_idx_tensor = torch.tensor(
                            kept_sample_indices,
                            dtype=torch.long,
                            device=corrective_output.batch['input_ids'].device,
                        )
                        corrective_output = corrective_output[keep_idx_tensor]
                        kept_corrective_ids = [expanded_corrective_ids[idx] for idx in kept_sample_indices]
                        # Fix Prompts for mask alignment
                        max_safe_prompt = self.config.actor_rollout_ref.rollout.get("prompt_length", 4000)
                        c_prompt_tensors = [
                            torch.tensor(ids[-max_safe_prompt:], dtype=torch.long)
                            for ids in kept_corrective_ids
                        ]
                        c_prompts_padded, _ = pad_and_stack_tensors(c_prompt_tensors, parent_pad_id, device=corrective_output.batch['input_ids'].device, left_pad=True)
                        corrective_output.batch['prompts'] = c_prompts_padded
                        corrective_output.batch['is_question'] = torch.zeros(len(corrective_output), dtype=torch.bool, device=corrective_output.batch['input_ids'].device)
                        corrective_batch_outputs.append(corrective_output)
            
            # 3. Process Results & Reconstruct Dense Scores
            
            # `verify_results` corresponds to `valid_indices * K`
            res_iter = iter(verify_results)

            # Pre-compute dense-index -> local-valid-index mapping (O(1) lookup
            # instead of O(B) sum(q_valid_flags[:i]) on every iteration).
            _valid_idx_map = {}
            _running_valid = 0
            for _di in range(total_qs):
                if q_valid_flags and _di < len(q_valid_flags) and q_valid_flags[_di]:
                    _valid_idx_map[_di] = _running_valid
                    _running_valid += 1

            for i in range(total_qs):
                is_valid = q_valid_flags[i] if q_valid_flags else True
                parse_info = q_parse_results[i] if (q_parse_results and i < len(q_parse_results)) else None
                
                if is_valid:
                    if parse_info is not None:
                        q_format_penalties[i] = float(parse_info.format_penalty)
                    # Consumes K results
                    k_scores = []
                    k_debug_samples = []
                    
                    for _ in range(self.k_answers):
                        try:
                            score, strict_pass, full_code, clean_ans, v_out = next(res_iter)
                        except StopIteration:
                             score, strict_pass, full_code, clean_ans, v_out = 0.0, False, "", "", "Error: Missing Result"
                        
                        solve_scores.append(score)
                        strict_passes.append(strict_pass)
                        clean_answers.append(clean_ans)
                        full_codes.append(full_code)
                        verus_outputs.append(v_out)
                        
                        k_scores.append(score)
                        k_debug_samples.append((full_code, v_out, strict_pass))
                    
                    # Compute Q Stats
                    k_arr = np.array(k_scores)
                    k_pass = np.array([sp for _, _, sp in k_debug_samples], dtype=bool)
                    m_pass = int(k_pass.sum())
                    K_total = len(k_arr)
                    s_reward_signals.append(float(k_arr.max() > k_arr.min()))
                    
                    p = m_pass / K_total
                    
                    # [Corrective Expansion] If pass_rate=0 but corrective found a solution,
                    # use corrective_pass_rate as effective pass_rate (scaled down)
                    # This smooths the Q-Gen curriculum for "hard but solvable with hints" questions
                    vq_local_idx_for_corrective = _valid_idx_map.get(i)
                    
                    if p == 0.0 and vq_local_idx_for_corrective is not None and vq_local_idx_for_corrective in corrective_results:
                        cr = corrective_results[vq_local_idx_for_corrective]
                        if cr['corrective_pass']:
                            # Corrective found a solution! Use scaled pass_rate
                            p_corrective = cr['corrective_pass_rate'] * self.s_corrective_reward_scale
                            p = p_corrective  # Override 0.0 with corrective signal
                    
                    # s_max combines MinHash similarity (from Q-Gen) with structural signature match.
                    s_max = float(q_sim_scores[i]) if (q_sim_scores and i < len(q_sim_scores)) else 0.0
                    struct_sim = 0.0
                    if generated_specs_masked and i < len(generated_specs_masked):
                        for sq_key in self.seed_pool_mgr.pool:
                            if signature_similarity(generated_specs_masked[i], sq_key) > 0.5:
                                struct_sim = 1.0
                                break
                    # Treat exact structural duplicates as fully non-novel.
                    s_max = max(s_max, struct_sim)
                    q_struct_sims[i] = float(struct_sim)
                    q_seed_sims_effective[i] = float(s_max)

                    # Extract seed info for DAG backpropagation
                    seed_nbr_pr = 0.0
                    seed_q = ""
                    if q_seeds and i < len(q_seeds):
                        shots = q_seeds[i].get("shots", [])
                        if shots:
                            seed_dict = shots[0][0] if isinstance(shots[0], (tuple, list)) else shots[0]
                            seed_q = seed_dict.get('question', '').strip() if isinstance(seed_dict, dict) else ''
                            if seed_q:
                                seed_nbr_pr = self.seed_pool_mgr.neighborhood_pass_rate(seed_q)

                    # Q-Gen reward: only reward "barely solvable" questions.
                    # With k_answers=8, the target band is exactly one strict pass.
                    solver_utility = 1.0 if m_pass == 1 else 0.0
                    q_score = solver_utility

                    style_filter_score = self._q_valid_style_penalty(parse_info)
                    duplicate_like, same_fn_name, seed_fn_name, cand_fn_name = self._is_exact_duplicate_like(
                        seed_q,
                        generated_specs_masked[i] if (generated_specs_masked and i < len(generated_specs_masked)) else "",
                        s_max,
                        struct_sim,
                    )
                    q_base_scores[i] = float(solver_utility)
                    q_filter_scores[i] = float(style_filter_score)
                    pass_rates.append(p)
                    question_scores.append(float(q_score))

                    debug_infos.append({
                        "pass_rate": float(p),
                        "score": float(q_score),
                        "solver_utility": float(solver_utility),
                        "policy_score": float(q_base_scores[i]),
                        "seed_nbr_pr": float(seed_nbr_pr),
                        "seed_sim": float(s_max),
                        "struct_sim": float(struct_sim),
                        "format_penalty": float(q_format_penalties[i]),
                        "style_filter_score": float(style_filter_score),
                        "seed_fn_name": str(seed_fn_name),
                        "candidate_fn_name": str(cand_fn_name),
                        "same_fn_name": bool(same_fn_name),
                        "exact_duplicate_like": bool(duplicate_like),
                        "parse_reject_reason": str(getattr(parse_info, "reject_reason", "")) if parse_info is not None else "",
                        "q_stitched_code": q_stitched_codes[i] if (q_stitched_codes and i < len(q_stitched_codes)) else "",
                    })
                    
                else:
                    # Invalid Q (failed parse/filter gate): main solver reward stays 0,
                    # while the separate filter branch can still apply a mild negative
                    # shaping signal during PPO.
                    err_compiler_out = q_err_compiler_outs[i] if (q_err_compiler_outs and i < len(q_err_compiler_outs)) else ""
                    filter_score = 0.0
                    if q_filter_results is not None and i < len(q_filter_results):
                        q_filter_result = q_filter_results[i]
                        if q_filter_result is not None:
                            filter_score = float(getattr(q_filter_result, "score", 0.0))
                    if parse_info is not None and not parse_info.valid:
                        q_format_penalties[i] = 0.0
                    q_filter_scores[i] = float(filter_score)
                    pass_rates.append(0.0)
                    question_scores.append(0.0)

                    debug_infos.append({
                        "pass_rate": 0.0,
                        "score": 0.0,
                        "filter_score": float(filter_score),
                        "policy_score": 0.0,
                        "format_penalty": float(q_format_penalties[i]),
                        "parse_reject_reason": str(getattr(parse_info, "reject_reason", "")) if parse_info is not None else "",
                        "q_stitched_code": q_stitched_codes[i] if (q_stitched_codes and i < len(q_stitched_codes)) else "",
                        "q_verus_output": err_compiler_out
                    })

            question_scores_safe_raw = np.array(question_scores, dtype=np.float32)
            q_base_scores_raw = np.array(q_base_scores, dtype=np.float32)
            q_filter_scores_raw = np.array(q_filter_scores, dtype=np.float32)
            solve_scores_for_ppo = list(solve_scores)
            strict_passes_for_ppo = list(strict_passes)

            # =====================================================
            # [Diversity Penalty] Within-group pairwise similarity
            # Convert
            question_scores_safe = np.array(question_scores_safe_raw, dtype=np.float32)
            pass_rates = np.array(pass_rates, dtype=np.float32)
            solve_scores_arr = np.array(solve_scores, dtype=np.float32)
            solve_scores_arr_for_ppo = np.array(solve_scores_for_ppo, dtype=np.float32)
            
            # Compute Q reward signal ratio (per prompt, N candidates)
            if self.q_gen_n > 0 and len(question_scores_safe) % self.q_gen_n == 0:
                q_scores_reshaped = question_scores_safe.reshape(-1, self.q_gen_n)
                q_maxs = q_scores_reshaped.max(axis=1)
                q_mins = q_scores_reshaped.min(axis=1)
                for i in range(len(q_maxs)):
                    q_reward_signals.append(float(q_maxs[i] > q_mins[i]))

            # Metrics
            if len(solve_scores) > 0:
                metrics.update({
                    "reward/solve_scores_mean": solve_scores_arr.mean(),
                    "reward/ratio_perfect_s": (solve_scores_arr > 0.99).mean(),
                    "reward/ratio_has_signal_s": float(np.mean(s_reward_signals)) if len(s_reward_signals) > 0 else 0.0
                })
            
            # Compute pass count metrics instead of percentages
            n_valid = sum(q_valid_flags) if q_valid_flags else 0
            if n_valid > 0:
                pass_rates_valid = pass_rates[q_valid_flags]
                one_pass_rate = 1.0 / max(self.k_answers, 1)
                cnt_full = int((pass_rates_valid >= 1.0 - 1e-6).sum())
                cnt_zero = int((pass_rates_valid <= 1e-6).sum())
                cnt_partial = int(((pass_rates_valid > 1e-6) & (pass_rates_valid < 1.0 - 1e-6)).sum())
                cnt_one_pass = int((np.abs(pass_rates_valid - one_pass_rate) <= 1e-6).sum())
            else:
                cnt_full = cnt_zero = cnt_partial = cnt_one_pass = 0

            metrics.update({
                "reward/question_scores_mean": question_scores_safe.mean(),
                "reward/pass_rate_mean": pass_rates.mean(),
                "reward/ratio_full_pass": float(cnt_full) / max(n_valid, 1),
                "reward/ratio_zero_pass": float(cnt_zero) / max(n_valid, 1),
                "reward/ratio_partial_pass": float(cnt_partial) / max(n_valid, 1),
                "reward/ratio_one_pass": float(cnt_one_pass) / max(n_valid, 1),
                "reward/ratio_valid_q": np.mean(q_valid_flags) if q_valid_flags else 0.0,
                "reward/ratio_has_signal_q": float(np.mean(q_reward_signals)) if len(q_reward_signals) > 0 else 0.0
            })
            valid_q_idx = [i for i, v in enumerate(q_valid_flags or [True] * total_qs) if v]
            if valid_q_idx:
                metrics.update({
                    "reward/q_filter_score_mean": float(np.mean([q_filter_scores[i] for i in range(total_qs)])),
                    "reward/q_solver_utility_mean": float(np.mean([debug_infos[i].get("solver_utility", q_base_scores[i]) for i in valid_q_idx])),
                    "reward/q_policy_score_mean": float(np.mean([q_base_scores[i] for i in valid_q_idx])),
                    "reward/q_pass_rate_mean": float(np.mean([debug_infos[i].get("pass_rate", 0.0) for i in valid_q_idx])),
                    "reward/q_seed_sim_effective_mean": float(np.mean([q_seed_sims_effective[i] for i in valid_q_idx])),
                    "reward/q_struct_sim_mean": float(np.mean([q_struct_sims[i] for i in valid_q_idx])),
                    "reward/q_format_penalty_mean": float(np.mean([q_format_penalties[i] for i in valid_q_idx])),
                    "reward/q_seed_nbr_pr_mean": float(np.mean([debug_infos[i].get("seed_nbr_pr", 0.0) for i in valid_q_idx])),
                    "reward/q_style_filter_score_mean": float(np.mean([debug_infos[i].get("style_filter_score", 0.0) for i in valid_q_idx])),
                    "reward/q_raw_char_len_mean": float(np.mean([debug_infos[i].get("raw_char_len", 0.0) for i in valid_q_idx])),
                    "reward/q_think_char_len_mean": float(np.mean([debug_infos[i].get("think_char_len", 0.0) for i in valid_q_idx])),
                    "reward/q_code_block_count_mean": float(np.mean([debug_infos[i].get("code_block_count", 0.0) for i in valid_q_idx])),
                    "reward/q_spec_char_len_mean": float(np.mean([debug_infos[i].get("spec_char_len", 0.0) for i in valid_q_idx])),
                    "reward/q_spec_line_count_mean": float(np.mean([debug_infos[i].get("spec_line_count", 0.0) for i in valid_q_idx])),
                })
                print(
                    f"[Q-Reward] solver={metrics['reward/q_solver_utility_mean']:.3f}, "
                    f"pass_rate={metrics['reward/q_pass_rate_mean']:.3f}, "
                    f"final={metrics['reward/question_scores_mean']:.3f}, "
                    f"seed_sim={metrics['reward/q_seed_sim_effective_mean']:.3f}",
                    flush=True,
                )

            # Similarity metrics (for monitoring novelty over training)
            if q_sim_scores:
                valid_sims = [q_sim_scores[i] for i, v in enumerate(q_valid_flags or [True]*len(q_sim_scores)) if v and i < len(q_sim_scores)]
                metrics.update({
                    "novelty/seed_sim_mean": float(np.mean(q_sim_scores)),
                    "novelty/seed_sim_max": float(np.max(q_sim_scores)),
                    "novelty/seed_sim_valid_mean": float(np.mean(valid_sims)) if valid_sims else 0.0,
                })
            
            n_verified = int(sum(1 for sp in strict_passes if sp))
            n_total_solutions = len(solve_scores_arr)
            print(f"[Metrics] Verified Solutions: {n_verified}/{n_total_solutions}, "
                  f"Full-Pass: {cnt_full}/{n_valid}, Zero-Pass: {cnt_zero}/{n_valid}, "
                  f"Partial-Pass: {cnt_partial}/{n_valid}, One-Pass: {cnt_one_pass}/{n_valid}", flush=True)

            reward_extra_infos = {
                "solve_scores": solve_scores,           # Sparse [N_valid * K]
                "solve_scores_raw": solve_scores,
                "solve_scores_for_ppo": solve_scores_for_ppo,
                "strict_passes": strict_passes,
                "strict_passes_for_ppo": strict_passes_for_ppo,
                "verus_outputs": verus_outputs,          # Sparse [N_valid * K], aligned with solve_scores
                "full_codes": full_codes,                # Sparse [N_valid * K], stitched code per S-Gen answer
                "question_scores": question_scores_safe.tolist(),  # Dense [B * N], one score per Q candidate
                "question_solver_scores": q_base_scores_raw.tolist(),
                "question_filter_scores": q_filter_scores_raw.tolist(),
                "debug_infos": debug_infos
            }
            
            return solve_scores_arr_for_ppo, question_scores_safe, pass_rates, clean_answers, reward_extra_infos, corrective_results, corrective_batch_outputs, corrective_all_scores

    def _prepare_unified_batch(self, batch_q_output, batch_s_output, corrective_batch_outputs=None):
        """Turn 4: Construct Unified PPO Batch."""
        pad_token_id = self.tokenizer.pad_token_id
        def ensure_mask(batch_proto):
              if 'response_mask' not in batch_proto.batch:
                   input_ids = batch_proto.batch['input_ids']
                   attention_mask = batch_proto.batch['attention_mask']
                   actual_pad_id = pad_token_id if pad_token_id is not None else 0
                   if 'prompts' in batch_proto.batch:
                        prompts = batch_proto.batch['prompts']
                        p_mask = (prompts != actual_pad_id)                        
                        p_lens = p_mask.sum(dim=1)
                   else:
                        responses = batch_proto.batch.get('responses', input_ids)
                        r_valid_mask = (responses != actual_pad_id)
                        r_lens = r_valid_mask.sum(dim=1)
                        total_valid_mask = (input_ids != actual_pad_id)
                        total_lens = total_valid_mask.sum(dim=1)
                        p_lens = total_lens - r_lens
                   
                   r_mask = torch.zeros_like(input_ids)
                   for i in range(len(input_ids)):
                        # Find the first non-pad token position
                        valid_starts = (input_ids[i] != actual_pad_id).nonzero(as_tuple=True)[0]
                        start = valid_starts[0] if len(valid_starts) > 0 else 0
                        r_mask[i, start + p_lens[i]:] = 1
                   batch_proto.batch['response_mask'] = r_mask * attention_mask
              return batch_proto

        batch_q_ready = ensure_mask(batch_q_output)
        batch_s_ready = ensure_mask(batch_s_output)
        
        # Reorder batches: all proposer rows (is_question=1) first, then all solver rows (is_question=0)
        # This makes Advantage computation much easier (contiguous indices)
        protos_q = [batch_q_ready]
        
        protos_s_main = [batch_s_ready]
        protos_s_corrective = list(corrective_batch_outputs) if corrective_batch_outputs else []
        protos_s = protos_s_main + protos_s_corrective
        
        all_protos = protos_q + protos_s
        batches = [p.batch for p in all_protos]
        
        all_keys = set()
        for b in batches:
            all_keys.update(b.keys())
        all_keys = list(all_keys)
        
        TOTAL_SEQ_KEYS = {'input_ids', 'attention_mask', 'position_ids', 'prompts'}
        RESP_SEQ_KEYS = {'responses', 'log_probs', 'response_mask'}
        
        max_total_len = 0
        max_resp_len = 0
        for k in all_keys:
            for b in batches:
                if k in b and b[k].dim() > 1:
                    if k in TOTAL_SEQ_KEYS:
                        max_total_len = max(max_total_len, b[k].shape[1])
                    elif k in RESP_SEQ_KEYS:
                        max_resp_len = max(max_resp_len, b[k].shape[1])
        
        tensors_cat = {}
        for k in all_keys:
            target_len = 0
            if k in TOTAL_SEQ_KEYS: target_len = max_total_len
            elif k in RESP_SEQ_KEYS: target_len = max_resp_len
            
            def prepare_side(batch_side, reference_batch):
                if k in batch_side:
                    t = batch_side[k]
                    if target_len > 0 and t.dim() > 1 and t.shape[1] < target_len:
                        fill_val = pad_token_id if pad_token_id is not None and k in ['input_ids', 'prompts', 'responses'] else 0
                        t = torch.cat([t, torch.full((t.shape[0], target_len - t.shape[1]) + t.shape[2:], fill_val, dtype=t.dtype, device=t.device)], dim=1)
                    return t
                else:
                    sample = reference_batch[k]
                    this_size = batch_side.batch_size[0]
                    fill_val = pad_token_id if pad_token_id is not None and k in ['input_ids', 'prompts', 'responses'] else 0
                    if target_len > 0 and sample.dim() > 1:
                        shape = (this_size, target_len) + sample.shape[2:]
                    else:
                        shape = (this_size,) + sample.shape[1:]
                    return torch.full(shape, fill_val, dtype=sample.dtype, device=sample.device)

            prepared_tensors = []
            ref_batch = next((b for b in batches if k in b), batches[0])
            for b in batches:
                prepared_tensors.append(prepare_side(b, ref_batch))
            
            tensors_cat[k] = torch.cat(prepared_tensors, dim=0)
             
        non_tensor_cat = {}
        # Determine common keys across all available non_tensor_batches
        common_nt_keys = set()
        if all_protos:
            common_nt_keys = set(all_protos[0].non_tensor_batch.keys())
            for p in all_protos[1:]:
                common_nt_keys &= set(p.non_tensor_batch.keys())

        for k in common_nt_keys:
             vals = [p.non_tensor_batch[k] for p in all_protos]
             if all(isinstance(v, np.ndarray) for v in vals):
                 try:
                     non_tensor_cat[k] = np.concatenate(vals)
                 except ValueError:
                     merged = []
                     for v in vals:
                         merged.extend(v.tolist())
                     non_tensor_cat[k] = _to_object_array(merged)
             elif all(isinstance(v, list) for v in vals):
                 non_tensor_cat[k] = sum(vals, [])
             elif all(isinstance(v, (list, np.ndarray)) for v in vals):
                 merged = []
                 for v in vals:
                     if isinstance(v, np.ndarray):
                         merged.extend(v.tolist())
                     else:
                         merged.extend(v)
                 non_tensor_cat[k] = _to_object_array(merged)
             else:
                 non_tensor_cat[k] = vals[0]
        
        non_tensor_cat['uid'] = np.array([f"gen_{i}" for i in range(tensors_cat['input_ids'].shape[0])])
        
        unified_batch = DataProto(batch=TensorDict(tensors_cat, batch_size=tensors_cat['input_ids'].shape[0]), non_tensor_batch=non_tensor_cat)
        unified_batch.meta_info = deepcopy(batch_s_ready.meta_info) 
        
        # Build is_question mask consistent with our reordered concatenation
        n_q = sum(p.batch.shape[0] for p in protos_q)
        n_s_main = sum(p.batch.shape[0] for p in protos_s_main)
        n_s_corrective = sum(p.batch.shape[0] for p in protos_s_corrective)
        n_s = n_s_main + n_s_corrective
        is_q_mask = torch.cat([
            torch.ones(n_q, dtype=torch.bool),
            torch.zeros(n_s, dtype=torch.bool)
        ]).to(tensors_cat['input_ids'].device)

        unified_batch.batch['is_question'] = is_q_mask
        is_corrective_mask = torch.cat([
            torch.zeros(n_q, dtype=torch.bool),
            torch.zeros(n_s_main, dtype=torch.bool),
            torch.ones(n_s_corrective, dtype=torch.bool),
        ]).to(tensors_cat['input_ids'].device)
        unified_batch.batch['is_corrective'] = is_corrective_mask
        unified_batch.meta_info["n_q"] = n_q
        unified_batch.meta_info["n_s_main"] = n_s_main
        unified_batch.meta_info["n_s_corrective"] = n_s_corrective
        print(
            f"[Stage 4 - Batch] Unified PPO batch: q={n_q}, s_main={n_s_main}, "
            f"s_corrective={n_s_corrective}, total={n_q + n_s}.",
            flush=True,
        )
        return unified_batch

    def _compute_coupled_advantages(
        self,
        batch,
        rewards_q,
        rewards_s,
        strict_passes_s=None,
        rewards_c=None,
        rewards_q_solver=None,
        rewards_q_filter=None,
        q_valid_flags=None,
    ):
        """
        [Sparse Batch Update]
        rewards_q: Dense [B * N]
        rewards_s: Sparse [N_valid * K]
        rewards_c: Sparse Corrective expansion (optional)
        q_valid_flags: Dense [B * N] bool — if set, format-invalid Q specs are
            hard-discarded from GRPO group normalization (Kimi-Prover style).
            They receive zero advantage and contribute no gradient.
        """
        is_question = batch.batch['is_question'] 
        is_corrective = batch.batch['is_corrective'] if 'is_corrective' in batch.batch else torch.zeros_like(is_question)
        response_mask = batch.batch['response_mask']
        device = batch.batch['input_ids'].device
        
        # 1. Proposer (Q-Gen) Advantage
        r_q_all = torch.tensor(rewards_q, device=device, dtype=torch.float32)
        q_solver_weight = float(getattr(self.config.algorithm, "q_solver_adv_weight", 1.0))
        q_filter_weight = float(getattr(self.config.algorithm, "q_filter_adv_weight", 0.0))

        def _normalize_q_component(raw_vals: torch.Tensor, keep_mask: torch.Tensor | None = None) -> torch.Tensor:
            # Group-relative normalization for the main Q solver reward.
            # Use mean absolute reward as the scale so the same code remains
            # stable if we later introduce small signed shaping terms.
            if len(raw_vals) == 0:
                return raw_vals
            if keep_mask is None:
                keep_mask = torch.ones_like(raw_vals, dtype=torch.bool)
            if self.q_gen_n > 0 and len(raw_vals) >= self.q_gen_n and len(raw_vals) % self.q_gen_n == 0:
                raw_view = raw_vals.view(-1, self.q_gen_n)
                keep_view = keep_mask.view(-1, self.q_gen_n)
                out_view = torch.zeros_like(raw_view)
                for gi in range(raw_view.shape[0]):
                    active_mask = keep_view[gi]
                    if not torch.any(active_mask):
                        continue
                    active_vals = raw_view[gi][active_mask]
                    mean_r = active_vals.mean()
                    mean_abs = active_vals.abs().mean().clamp_min(1e-3)
                    out_view[gi][active_mask] = (active_vals - mean_r) / mean_abs
                return out_view.view(-1)
            if torch.any(keep_mask):
                active_vals = raw_vals[keep_mask]
                mean_r = active_vals.mean()
                mean_abs = active_vals.abs().mean().clamp_min(1e-3)
                out = torch.zeros_like(raw_vals)
                out[keep_mask] = (active_vals - mean_r) / mean_abs
                return out
            return torch.zeros_like(raw_vals)

        def _scale_q_filter_component(raw_vals: torch.Tensor, keep_mask: torch.Tensor | None = None) -> torch.Tensor:
            # Filter penalties are behavior shaping, not within-group ranking.
            # Keep them signed and globally scaled so a whole bad group still
            # receives a consistent push away from malformed/template outputs.
            if len(raw_vals) == 0:
                return raw_vals
            if keep_mask is None:
                keep_mask = torch.ones_like(raw_vals, dtype=torch.bool)
            if not torch.any(keep_mask):
                return torch.zeros_like(raw_vals)
            active_vals = raw_vals[keep_mask]
            mean_abs = active_vals.abs().mean().clamp_min(1e-3)
            out = torch.zeros_like(raw_vals)
            out[keep_mask] = active_vals / mean_abs
            return out

        # Hard-discard: format-invalid Q specs are excluded from GRPO normalization.
        # They get zero advantage → zero gradient. Only on-manifold samples participate.
        if q_valid_flags is not None and len(q_valid_flags) == len(r_q_all):
            q_keep_mask = torch.tensor(q_valid_flags, device=device, dtype=torch.bool)
        else:
            q_keep_mask = torch.ones_like(r_q_all, dtype=torch.bool)

        if rewards_q_solver is not None:
            r_q_solver = torch.tensor(rewards_q_solver, device=device, dtype=torch.float32)
            adv_q_solver = _normalize_q_component(r_q_solver, q_keep_mask)
            adv_q_values = q_solver_weight * adv_q_solver
        else:
            adv_q_values = _normalize_q_component(r_q_all, q_keep_mask)
        r_q_filter = torch.tensor(rewards_q_filter or [], device=device, dtype=torch.float32)
        q_filter_weight_effective = q_filter_weight
        if len(r_s_base := torch.tensor(rewards_s, device=device, dtype=torch.float32)) == 0:
            q_filter_weight_effective = 0.0
        if rewards_q_filter is not None and q_filter_weight_effective > 0.0:
            adv_q_filter = _scale_q_filter_component(r_q_filter, q_keep_mask)
            adv_q_values = adv_q_values + q_filter_weight_effective * adv_q_filter
        
        # 2. Solver (S-Gen + Corrective) Advantage
        strict_s_base = torch.tensor(strict_passes_s or [], device=device, dtype=torch.bool)
        r_c_all = torch.tensor(rewards_c or [], device=device, dtype=torch.float32)
        r_s_all = torch.cat([r_s_base, r_c_all])
        q_group_count = (len(r_q_all) // self.q_gen_n) if self.q_gen_n > 0 and len(r_q_all) % self.q_gen_n == 0 else 0
        s_group_count = (len(r_s_all) // self.k_answers) if self.k_answers > 0 and len(r_s_all) % self.k_answers == 0 else 0
        s_base_group_count = (len(r_s_base) // self.k_answers) if self.k_answers > 0 and len(r_s_base) % self.k_answers == 0 else 0
        c_group_count = (len(r_c_all) // self.k_answers) if self.k_answers > 0 and len(r_c_all) % self.k_answers == 0 else 0
        if len(r_c_all) > 0 and self.s_corrective_n != self.k_answers:
            print(
                f"[Stage 4 - Adv][WARN] corrective samples are generated with s_corrective_n={self.s_corrective_n}, "
                f"but solver advantages are grouped with k_answers={self.k_answers}. "
                f"Current implementation groups corrective rewards by k_answers.",
                flush=True,
            )
        if len(r_s_all) > 0 and self.k_answers > 0 and len(r_s_all) % self.k_answers != 0:
            print(
                f"[Stage 4 - Adv][WARN] Solver reward count {len(r_s_all)} is not divisible by k_answers={self.k_answers}. "
                f"Falling back to global normalization instead of group normalization.",
                flush=True,
            )
        
        if len(r_s_all) > 0:
            # Advantage = (r - mean(r)) / (mean(r) + eps)
            #
            # For binary r in {0, 1} this has desirable properties:
            #   all-0 group : numerator=0  -> adv=0, no gradient (entropy preserved)
            #   all-1 group : numerator=0  -> adv=0, no gradient
            #   m/k pass    : success adv = (1-p)/p  (grows as 1/p - amplifies rare solves)
            #                 failure adv = -1        (constant, independent of p)
            # The 1/p amplification is exactly what sparse binary rewards need:
            # a single success in a group of 8 gets advantage 7x, not just ~2.6x.
            EPS_S = 1e-4
            if self.k_answers > 0 and len(r_s_all) % self.k_answers == 0:
                r_s_reshaped = r_s_all.view(-1, self.k_answers)
                mean_s = r_s_reshaped.mean(dim=1, keepdim=True)
                adv_s_values = ((r_s_reshaped - mean_s) / (mean_s + EPS_S)).view(-1)
            else:
                mean_s = r_s_all.mean()
                adv_s_values = (r_s_all - mean_s) / (mean_s + EPS_S)
        else:
            adv_s_values = torch.empty(0, device=device)
             
        # Get mask indices
        q_indices = torch.nonzero(is_question, as_tuple=True)[0]
        s_indices = torch.nonzero(~is_question, as_tuple=True)[0]
        s_main_indices = torch.nonzero((~is_question) & (~is_corrective), as_tuple=True)[0]
        s_corrective_indices = torch.nonzero((~is_question) & is_corrective, as_tuple=True)[0]
        
        token_level_adv = torch.zeros_like(response_mask, dtype=torch.float32)
        token_level_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        algo_cfg = self.config.algorithm if hasattr(self.config, "algorithm") else {}
        freeze_q_branch = bool(algo_cfg.get("freeze_q_branch", False))
        freeze_s_branch = bool(algo_cfg.get("freeze_s_branch", False))
        
        raw_q_tokens = response_mask[q_indices].sum().item() if len(q_indices) > 0 else 0
        raw_s_tokens = response_mask[s_indices].sum().item() if len(s_indices) > 0 else 0
        raw_s_main_tokens = response_mask[s_main_indices].sum().item() if len(s_main_indices) > 0 else 0
        raw_s_corrective_tokens = response_mask[s_corrective_indices].sum().item() if len(s_corrective_indices) > 0 else 0
        if raw_q_tokens > 0 and raw_s_tokens > 0:
            total_tokens = raw_q_tokens + raw_s_tokens
            scale_q = float(total_tokens / (2.0 * raw_q_tokens))
            scale_s = float(total_tokens / (2.0 * raw_s_tokens))
        else:
            scale_q, scale_s = 1.0, 1.0
        avg_q_resp_len = (raw_q_tokens / max(len(q_indices), 1)) if len(q_indices) > 0 else 0.0
        avg_s_main_resp_len = (raw_s_main_tokens / max(len(s_main_indices), 1)) if len(s_main_indices) > 0 else 0.0
        avg_s_corrective_resp_len = (raw_s_corrective_tokens / max(len(s_corrective_indices), 1)) if len(s_corrective_indices) > 0 else 0.0
        
        global_pass_rate = strict_s_base.float().mean().item() if len(strict_s_base) > 0 else 0.0
        balance_boost_q = 1.0
        q_warmup_scale = 1.0
        q_branch_weight_raw = scale_q * balance_boost_q if raw_q_tokens > 0 else 0.0
        q_l2_balance_ratio = float(algo_cfg.get("q_adv_l2_balance_ratio", 0.5))
        q_branch_weight_min = float(algo_cfg.get("q_branch_weight_min", 0.125))
        q_branch_weight_max = float(algo_cfg.get("q_branch_weight_max", 4.0))
        raw_q_adv = response_mask[q_indices] * adv_q_values[:len(q_indices)].unsqueeze(1) if len(q_indices) > 0 else None
        raw_s_adv = response_mask[s_indices] * adv_s_values[:len(s_indices)].unsqueeze(1) if len(s_indices) > 0 else None
        raw_q_adv_l2 = raw_q_adv.norm().item() if raw_q_adv is not None else 0.0
        raw_s_adv_l2 = raw_s_adv.norm().item() if raw_s_adv is not None else 0.0
        target_s_adv_l2 = scale_s * raw_s_adv_l2 if raw_s_adv_l2 > 0.0 else 0.0
        if raw_q_adv_l2 > 0.0 and raw_s_adv_l2 > 0.0:
            target_q_branch_weight = q_l2_balance_ratio * target_s_adv_l2 / max(raw_q_adv_l2, 1e-6)
            q_branch_weight = float(np.clip(target_q_branch_weight, q_branch_weight_min, q_branch_weight_max))
            q_loss_scale = q_branch_weight / max(q_branch_weight_raw, 1e-6) if q_branch_weight_raw > 0.0 else 0.0
        else:
            q_loss_scale = float(algo_cfg.get("q_loss_scale", 1.0))
            q_branch_weight = float(np.clip(q_branch_weight_raw * q_loss_scale, q_branch_weight_min, q_branch_weight_max))
        s_branch_weight = scale_s if raw_s_tokens > 0 else 0.0

        if len(q_indices) > 0:
            token_level_adv[q_indices] = response_mask[q_indices] * (adv_q_values[:len(q_indices)] * q_branch_weight).unsqueeze(1)
            token_level_scores[q_indices] = response_mask[q_indices] * r_q_all[:len(q_indices)].unsqueeze(1)
            
        if len(s_indices) > 0:
            token_level_adv[s_indices] = response_mask[s_indices] * (adv_s_values[:len(s_indices)] * scale_s).unsqueeze(1)
            token_level_scores[s_indices] = response_mask[s_indices] * r_s_all[:len(s_indices)].unsqueeze(1)
            
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        token_level_adv = token_level_adv * ppo_mini_batch_size

        # [Branch isolation] Zero out Q or S advantages for diagnostic runs.
        # freeze_q_branch=true → only S updates; freeze_s_branch=true → only Q updates.
        if freeze_q_branch and len(q_indices) > 0:
            token_level_adv[q_indices] = 0.0
        if freeze_s_branch and len(s_indices) > 0:
            token_level_adv[s_indices] = 0.0

        # [Branch adv_norm] Proxy for per-branch gradient drive.
        q_adv_l2 = token_level_adv[q_indices].norm().item() if len(q_indices) > 0 else 0.0
        s_adv_l2 = token_level_adv[s_indices].norm().item() if len(s_indices) > 0 else 0.0

        batch.batch['advantages'] = token_level_adv
        batch.batch['returns'] = token_level_adv
        batch.batch['token_level_scores'] = token_level_scores
        print(
            f"[Stage 4 - Adv] Groups: q={q_group_count} (q_gen_n={self.q_gen_n}), "
            f"s_total={s_group_count} [s_main={s_base_group_count}, s_corrective={c_group_count}] "
            f"with group_size={self.k_answers}.",
            flush=True,
        )
        print(
            f"[Stage 4 - Adv] Tokens: q={raw_q_tokens}, s_main={raw_s_main_tokens}, "
            f"s_corrective={raw_s_corrective_tokens}, s_total={raw_s_tokens}. "
            f"Avg response len: q={avg_q_resp_len:.1f}, s_main={avg_s_main_resp_len:.1f}, "
            f"s_corrective={avg_s_corrective_resp_len:.1f}.",
            flush=True,
        )
        print(
            f"[Stage 4 - Adv] Branch weights: q={q_branch_weight:.4f} "
            f"(scale_q={scale_q:.4f}, boost_q={balance_boost_q:.4f}, warmup_q={q_warmup_scale:.4f}), "
            f"s={s_branch_weight:.4f} (scale_s={scale_s:.4f}). "
            f"Q loss scale={q_loss_scale:.4f}, raw_l2: q={raw_q_adv_l2:.4f}, s={raw_s_adv_l2:.4f}, "
            f"target_s_l2={target_s_adv_l2:.4f}. "
            f"Q component weights: solver={q_solver_weight:.3f}, filter={q_filter_weight_effective:.3f}. "
            f"Reward means: q={r_q_all.mean().item():.4f}, s_main={(r_s_base.mean().item() if len(r_s_base) > 0 else 0.0):.4f}, "
            f"s_corrective={(r_c_all.mean().item() if len(r_c_all) > 0 else 0.0):.4f}.",
            flush=True,
        )
        
        return {
            "adv/q_mean": adv_q_values.mean().item(),
            "adv/s_mean": adv_s_values.mean().item() if len(adv_s_values) > 0 else 0.0,
            "adv/q_scale_token": scale_q,
            "adv/s_scale_token": scale_s,
            "adv/balance_boost_q": balance_boost_q,
            "adv/q_warmup_scale": q_warmup_scale,
            "adv/q_solver_adv_weight": q_solver_weight,
            "adv/q_filter_adv_weight": q_filter_weight_effective,
            "adv/q_loss_scale": q_loss_scale,
            "adv/q_raw_adv_l2": raw_q_adv_l2,
            "adv/s_raw_adv_l2": raw_s_adv_l2,
            "adv/s_target_adv_l2": target_s_adv_l2,
            "adv/q_branch_weight": q_branch_weight,
            "adv/s_branch_weight": s_branch_weight,
            "adv/q_groups": q_group_count,
            "adv/s_groups_total": s_group_count,
            "adv/s_groups_main": s_base_group_count,
            "adv/s_groups_corrective": c_group_count,
            "adv/s_group_size": self.k_answers,
            "adv/corrective_sample_group_size": self.s_corrective_n,
            "adv/corrective_grouping_mismatch": float(len(r_c_all) > 0 and self.s_corrective_n != self.k_answers),
            "adv/s_used_global_norm": float(len(r_s_all) > 0 and self.k_answers > 0 and len(r_s_all) % self.k_answers != 0),
            "adv/tokens_q": raw_q_tokens,
            "adv/tokens_s_main": raw_s_main_tokens,
            "adv/tokens_s_corrective": raw_s_corrective_tokens,
            "adv/tokens_s_total": raw_s_tokens,
            "adv/avg_resp_len_q": avg_q_resp_len,
            "adv/avg_resp_len_s_main": avg_s_main_resp_len,
            "adv/avg_resp_len_s_corrective": avg_s_corrective_resp_len,
            "adv/reward_mean_q": r_q_all.mean().item(),
            "adv/reward_mean_q_filter": r_q_filter.mean().item() if len(r_q_filter) > 0 else 0.0,
            "adv/reward_mean_s_main": r_s_base.mean().item() if len(r_s_base) > 0 else 0.0,
            "adv/reward_mean_s_corrective": r_c_all.mean().item() if len(r_c_all) > 0 else 0.0,
            "adv/n_q_samples": len(q_indices),
            "adv/n_q_discarded": int((~q_keep_mask).sum().item()),
            "adv/n_q_active": int(q_keep_mask.sum().item()),
            "adv/n_s_main_samples": len(s_main_indices),
            "adv/n_s_corrective_samples": len(s_corrective_indices),
            "adv/q_adv_l2": q_adv_l2,
            "adv/s_adv_l2": s_adv_l2,
        }

    def _update_policy_step(self, metric_prefix, batch, timing_raw, metrics):
        """Turn 5: Update Weights (Actor & Critic)."""
        total_batch_size = len(batch.batch['input_ids'])
        n_gpus = max(1, self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes)
        micro_bsz = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        print(
            f"[Stage 5 - RL Opt] total={total_batch_size}, n_gpus={n_gpus}, "
            f"per_gpu={total_batch_size // n_gpus}, "
            f"ppo_mini_batch_size={self.config.actor_rollout_ref.actor.ppo_mini_batch_size}, "
            f"micro={micro_bsz}",
            flush=True,
        )

        if self.use_critic:
            with marked_timer(f"{metric_prefix}/update_critic", timing_raw, color="pink"):
                critic_output = self._update_critic(batch)
            metrics.update({f"{metric_prefix}/{k}": v for k, v in reduce_metrics(critic_output.meta_info["metrics"]).items()})

        with marked_timer(f"{metric_prefix}/update_actor", timing_raw, color="green"):
            actor_output = self._update_actor(batch)
        metrics.update({f"{metric_prefix}/{k}": v for k, v in reduce_metrics(actor_output.meta_info["metrics"]).items()})

    def _emit_step_log(self, metrics):
        def _fmt(v):
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        def _get(key, default="na"):
            return _fmt(metrics.get(key, default))

        q_exploration_low = int(
            float(metrics.get("novelty/seed_sim_valid_mean", 0.0)) > 0.75
            and float(metrics.get("diversity/frac_penalized", 0.0)) > 0.85
        )
        corrective_diluting_s = int(
            float(metrics.get("adv/tokens_s_total", 0.0)) > 0.0
            and float(metrics.get("adv/tokens_s_corrective", 0.0)) / max(float(metrics.get("adv/tokens_s_total", 1.0)), 1.0) > 0.25
            and float(metrics.get("adv/reward_mean_s_corrective", 0.0)) < 0.0
        )
        q_collapse_parity = int(
            float(metrics.get("reward/q_seed_sim_effective_mean", 0.0)) > 0.7
        )

        s_corrective_token_frac = float(metrics.get('adv/tokens_s_corrective', 0.0)) / max(float(metrics.get('adv/tokens_s_total', 1.0)), 1.0)
        s_corrective_sample_frac = float(metrics.get('adv/n_s_corrective_samples', 0.0)) / max(float(metrics.get('adv/n_s_main_samples', 0.0)) + float(metrics.get('adv/n_s_corrective_samples', 0.0)), 1.0)
        lines = [
            "===== STEP_LOG_BEGIN =====",
            f"step={self.global_steps}",
            f"exp={self.config.trainer.experiment_name}",
            "",
            "[QGEN]",
            f"q_gen_n={self.q_gen_n}",
            f"q_valid_rate={_get('reward/ratio_valid_q')}",
            f"heuristic_rejects={_get('qgen/heuristic_rejects')}",
            f"compiler_rejects={_get('qgen/compiler_rejects')}",
            f"bad_ensures_rejects={_get('qgen/bad_ensures_rejects')}",
            f"to_solver_questions={_get('qgen/to_solver_questions')}",
            f"to_solver_samples={_get('qgen/to_solver_samples')}",
            f"seed_sim_max={_get('novelty/seed_sim_max')}",
            f"seed_sim_valid_mean={_get('novelty/seed_sim_valid_mean')}",
            "",
            "[QREWARD]",
            f"q_score_mean={_get('reward/question_scores_mean')}",
            f"q_solver_utility_mean={_get('reward/q_solver_utility_mean')}",
            f"q_pass_rate_mean={_get('reward/q_pass_rate_mean')}",
            f"q_filter_score_mean={_get('reward/q_filter_score_mean')}",
            f"q_seed_sim_effective_mean={_get('reward/q_seed_sim_effective_mean')}",
            f"q_struct_sim_mean={_get('reward/q_struct_sim_mean')}",
            "",
            "[SMAIN]",
            f"s_main_samples={_get('adv/n_s_main_samples')}",
            f"pass_rate_mean={_get('reward/pass_rate_mean')}",
            f"s_main_reward_mean={_get('adv/reward_mean_s_main')}",
            f"s_ability_ema={_get('adaptive/s_ability_ema')}",
            "",
            "[CORRECTIVE]",
            f"zero_pass_questions={_get('corrective/n_questions')}",
            f"roots={_get('corrective/n_roots')}",
            f"roots_kept_for_ppo={_get('corrective/n_roots_kept_for_ppo')}",
            f"samples_kept_for_ppo={_get('corrective/n_samples_kept_for_ppo')}",
            f"success_question_rate={_get('corrective/success_question_rate')}",
            f"reward_mean={_get('adv/reward_mean_s_corrective')}",
            "",
            "[BATCH]",
            f"s_corrective_samples={_get('adv/n_s_corrective_samples')}",
            f"s_corrective_tokens={_get('adv/tokens_s_corrective')}",
            f"s_corrective_token_frac={_fmt(s_corrective_token_frac)}",
            f"s_corrective_sample_frac={_fmt(s_corrective_sample_frac)}",
            "",
            "[ADV]",
            f"q_warmup_scale={_get('adv/q_warmup_scale')}",
            f"q_branch_weight={_get('adv/q_branch_weight')}",
            f"s_branch_weight={_get('adv/s_branch_weight')}",
            "",
            "[SEEDPOOL]",
            f"seed_total={_get('seed_pool/total')}",
            f"unsolved_original_roots={_get('seed_pool/unsolved_original_roots')}",
            f"high_similarity_filtered={_get('seed_pool/high_similarity_filtered')}",
            f"shallow_filtered={_get('seed_pool/shallow_filtered')}",
            f"short_impl_filtered={_get('seed_pool/short_impl_filtered')}",
            f"mutation_filtered={_get('seed_pool/mutation_filtered')}",
            f"quality_filtered={_get('seed_pool/quality_filtered')}",
            "",
            "[PERF]",
            f"gen_q_s={_get('timing_s/gen_q')}",
            f"gen_s_s={_get('timing_s/gen_s')}",
            f"verify_s_s={_get('timing_s/verify_s')}",
            f"update_actor_s={_get('timing_s/train/update_actor')}",
            f"actor_entropy={_get('actor/entropy')}",
            f"q_entropy={_get('actor/q_entropy')}",
            f"s_entropy={_get('actor/s_entropy')}",
            f"actor_grad_norm={_get('train/actor/grad_norm')}",
            f"lr={_get('train/actor/lr')}",
            "",
            "[FLAGS]",
            f"q_collapse_parity={q_collapse_parity}",
            f"q_exploration_low={q_exploration_low}",
            f"corrective_diluting_s={corrective_diluting_s}",
            "===== STEP_LOG_END =====",
        ]
        step_log_text = "\n".join(lines)
        try:
            step_log_dir = os.path.join(self.config.trainer.default_local_dir, "step_logs")
            os.makedirs(step_log_dir, exist_ok=True)
            append_path = os.path.join(step_log_dir, "step_blocks.log")
            latest_path = os.path.join(step_log_dir, "latest_step.log")
            step_path = os.path.join(step_log_dir, f"step_{self.global_steps}.log")
            with open(append_path, "a", encoding="utf-8") as f:
                f.write(step_log_text + "\n")
            with open(latest_path, "w", encoding="utf-8") as f:
                f.write(step_log_text + "\n")
            with open(step_path, "w", encoding="utf-8") as f:
                f.write(step_log_text + "\n")
        except Exception as e:
            print(f"[WARN] Failed to write step log file: {e}", flush=True)

    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        """Logs detailed rollout data (Q, S, Reward) to JSONL for debugging."""
        os.makedirs(rollout_data_dir, exist_ok=True)
        log_file = os.path.join(rollout_data_dir, f"rollout_step_{self.global_steps}.jsonl")
        
        try:
            prompts = batch.non_tensor_batch['raw_prompt']
            responses = batch.batch['responses']
            is_question_mask = batch.batch['is_question']
            
            decoded_responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            
            solve_scores = reward_extra_infos_dict.get('solve_scores', [])
            verus_outputs = reward_extra_infos_dict.get('verus_outputs', [])
            full_codes = reward_extra_infos_dict.get('full_codes', [])
            question_scores = reward_extra_infos_dict.get('question_scores', [])
            debug_infos = reward_extra_infos_dict.get('debug_infos', [])
            
            s_idx = 0
            q_idx = 0
            with open(log_file, 'w', encoding='utf-8') as f:
                for i, (p, r) in enumerate(zip(prompts, decoded_responses)):
                    is_q = bool(is_question_mask[i].item())
                    
                    if not is_q: # S-Gen
                        score = solve_scores[s_idx] if s_idx < len(solve_scores) else None
                        v_out = verus_outputs[s_idx] if s_idx < len(verus_outputs) else None
                        f_code = full_codes[s_idx] if s_idx < len(full_codes) else None
                        d_info = {"verus_output": v_out, "full_code": f_code}
                        s_idx += 1
                    else: # Q-Gen
                        score = question_scores[q_idx] if q_idx < len(question_scores) else None
                        d_info = debug_infos[q_idx] if q_idx < len(debug_infos) else None
                        q_idx += 1

                    # Fix formatting damage caused by calling str() directly on a NumPy array
                    p_out = p.tolist() if isinstance(p, np.ndarray) else p

                    log_entry = {
                        "step": self.global_steps,
                        "idx": i,
                        "prompt": p_out,
                        "response": r,
                        "is_question": is_q,
                        "question_score": score if is_q else None,
                        "solve_score": score if not is_q else None,
                        "debug_info": d_info
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    
        except Exception as e:
            print(f"[WARN] Failed to log rollout data: {e}")

    def _log_rollout_questions(
        self,
        generated_specs_masked,
        q_valid_flags,
        q_err_scores,
        q_err_compiler_outs,
        q_stitched_codes,
        q_sim_scores,
        reward_extra_infos,
        out_dir,
    ):
        """Persist per-rollout question records with deduplication for later audit."""
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"questions_step_{self.global_steps}.jsonl")
        debug_infos = reward_extra_infos.get("debug_infos", []) if reward_extra_infos else []
        q_filter_scores = reward_extra_infos.get("question_filter_scores", []) if reward_extra_infos else []
        filter_stages = reward_extra_infos.get("filter_stages", []) if reward_extra_infos else []
        filter_reasons = reward_extra_infos.get("filter_reasons", []) if reward_extra_infos else []
        seen = set()

        try:
            # 1. Aggregate pass rates for identical sequences in THIS rollout
            q_step_aggr = {}
            for i, spec in enumerate(generated_specs_masked):
                q_text = (spec or "").strip()
                if not q_text:
                    continue
                is_valid = bool(q_valid_flags[i]) if i < len(q_valid_flags) else False
                # debug_infos is DENSE [B*N] — one entry per question regardless
                # of validity — so index with the dense loop variable `i`.
                debug_info = debug_infos[i] if i < len(debug_infos) else {}
                
                # Only track questions that are valid for solver 
                if is_valid:
                    pr = float(debug_info.get("pass_rate", 0.0))
                    if q_text not in q_step_aggr:
                        q_step_aggr[q_text] = []
                    q_step_aggr[q_text].append(pr)
            
            # 2. Update global history and rewrite log files
            if q_step_aggr:
                for q_text, pr_list in q_step_aggr.items():
                    pr_str = " ".join([f"passrate {pr*100:.0f}%" for pr in pr_list])
                    
                    if not hasattr(self, "q_passrate_history"):
                        self.q_passrate_history = {}
                    
                    if q_text not in self.q_passrate_history:
                        is_seed = False
                        if self.seed_pool_mgr and self.seed_pool_mgr.pool:
                            is_seed = q_text in self.seed_pool_mgr.pool
                        self.q_passrate_history[q_text] = {"is_seed": is_seed, "history": []}
                    
                    ever_solved_this_step = any(pr > 0.0 for pr in pr_list)
                    self.q_passrate_history[q_text]["history"].append((int(self.global_steps), pr_str, ever_solved_this_step))

                # Rewrite logs
                try:
                    s_path = getattr(self, "solverable_log_path", os.path.join(self.config.trainer.default_local_dir, "solverable_question.log"))
                    u_path = getattr(self, "unsolverable_log_path", os.path.join(self.config.trainer.default_local_dir, "unsolverable_question.log"))
                    with open(s_path, "w", encoding="utf-8") as fs, open(u_path, "w", encoding="utf-8") as fu:
                        for q_text, data in self.q_passrate_history.items():
                            is_seed = data["is_seed"]
                            history = data["history"]
                            
                            history_str = " --> ".join([f"step {step} -- {pr_str}" for step, pr_str, _ in history])
                            seed_mark = " [SEED_POOL_ORIGINAL]" if is_seed else ""
                            # Using repr to keep standard text single-line but the user might expect literal newlines, let's substitute explicit newlines
                            clean_q_text = q_text.replace('\n', '\\n')
                            line_out = f"question : {clean_q_text} , {history_str}{seed_mark}\n"
                            
                            ever_solved = any(solved for _, _, solved in history)
                            if ever_solved:
                                fs.write(line_out)
                            else:
                                fu.write(line_out)
                except Exception as e:
                    print(f"[WARN] Failed to write passrate logs: {e}", flush=True)

            with open(out_path, "w", encoding="utf-8") as f:
                for i, spec in enumerate(generated_specs_masked):
                    q_text = (spec or "").strip()
                    if not q_text:
                        continue
                    if q_text in seen:
                        continue
                    seen.add(q_text)

                    is_valid = bool(q_valid_flags[i]) if i < len(q_valid_flags) else False
                    # debug_infos is DENSE [B*N] — index with dense `i`.
                    debug_info = debug_infos[i] if i < len(debug_infos) else {}

                    record = {
                        "step": int(self.global_steps),
                        "rollout_question_idx": int(i),
                        "question": q_text,
                        "valid_for_solver": is_valid,
                        "filter_stage": filter_stages[i] if i < len(filter_stages) else ("passed" if is_valid else "unknown"),
                        "filter_reason": filter_reasons[i] if i < len(filter_reasons) else "",
                        "compiler_filter_msg": q_err_compiler_outs[i] if i < len(q_err_compiler_outs) else "",
                        "filter_score": float(q_filter_scores[i]) if i < len(q_filter_scores) else (float(q_err_scores[i]) if i < len(q_err_scores) else 0.0),
                        "seed_sim_proxy": float(q_sim_scores[i]) if i < len(q_sim_scores) else 0.0,
                        "stitched_code": q_stitched_codes[i] if i < len(q_stitched_codes) else "",
                        "soft_scores": {
                            "score": float(debug_info.get("score", 0.0)),
                            "score_pre_div": float(debug_info.get("score_pre_div", 0.0)),
                            "base_score": float(debug_info.get("base_score", 0.0)),
                            "filter_score": float(q_filter_scores[i]) if i < len(q_filter_scores) else float(debug_info.get("filter_score", 0.0)),
                            "solver_utility": float(debug_info.get("solver_utility", 0.0)),
                            "policy_score": float(debug_info.get("policy_score", debug_info.get("solver_utility", 0.0))),
                            "p_clean": float(debug_info.get("p_clean", 0.0)),
                            "trivial_prior": float(debug_info.get("trivial_prior", 0.0)),
                            "cheap_solver_pass_rate": float(debug_info.get("cheap_solver_pass_rate", 0.0)),
                            "weak_spec_penalty": float(debug_info.get("weak_spec_penalty", 0.0)),
                            "seed_sim": float(debug_info.get("seed_sim", 0.0)),
                            "struct_sim": float(debug_info.get("struct_sim", 0.0)),
                            "family_escape_score": float(debug_info.get("family_escape_score", 0.0)),
                            "sibling_max_sim": float(debug_info.get("sibling_max_sim", 0.0)),
                            "format_penalty": float(debug_info.get("format_penalty", 1.0)),
                            "style_filter_score": float(debug_info.get("style_filter_score", 0.0)),
                        },
                        "soft_flags": {
                            "weak_spec_suspect": bool(debug_info.get("weak_spec_suspect", False)),
                            "bad_ensures_suspect": bool(debug_info.get("bad_ensures_suspect", False)),
                            "trivial_baseline_pass": bool(debug_info.get("trivial_baseline_pass", False)),
                        },
                        "soft_reasons": {
                            "weak_spec_reason": str(debug_info.get("weak_spec_reason", "")),
                            "bad_ensures_reason": str(debug_info.get("bad_ensures_reason", "")),
                            "trivial_reason": str(debug_info.get("trivial_reason", "")),
                            "parse_reject_reason": str(debug_info.get("parse_reject_reason", "")),
                        },
                        "format_debug": {
                            "raw_char_len": int(debug_info.get("raw_char_len", 0)),
                            "post_char_len": int(debug_info.get("post_char_len", 0)),
                            "think_char_len": int(debug_info.get("think_char_len", 0)),
                            "code_block_count": int(debug_info.get("code_block_count", 0)),
                            "spec_char_len": int(debug_info.get("spec_char_len", 0)),
                            "spec_line_count": int(debug_info.get("spec_line_count", 0)),
                            "fn_count": int(debug_info.get("fn_count", 0)),
                            "salvaged_primary_fn": bool(debug_info.get("salvaged_primary_fn", False)),
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] Failed to log rollout questions: {e}", flush=True)

    # ========================== Data Loading & Evaluation ==========================



    def _validate(self, dataset_name, dataset_df, n_samples=10):
        """Runs rigorous multi-threaded evaluation on verified datasets."""
        print(f"Running validation for {dataset_name} ({len(dataset_df)} samples, n={n_samples})...")
        metrics, problem_results = {}, []
        raw_val_batch_size = self.config.data.get("val_batch_size", 16)
        if raw_val_batch_size is None: raw_val_batch_size = 16
        val_batch_size = max(1, raw_val_batch_size // n_samples) if n_samples > 1 else raw_val_batch_size
        num_batches = math.ceil(len(dataset_df) / val_batch_size)
        
        def process_verification_sample(args):
            p_text, full_text = args
            stitched_result = smart_stitch(p_text, full_text)
            stitched_code = stitched_result[0] if isinstance(stitched_result, tuple) else stitched_result
            verified, v_stdout, v_stderr = False, "", ""
            
            if stitched_code:
                if "use builtin" in stitched_code:
                     stitched_code = stitched_code.replace("use builtin::*;", "// use builtin::*;").replace("use builtin_macros::*;", "// use builtin_macros::*;")
                if not re.search(r"fn\s+main", stitched_code): stitched_code += "\nfn main() {}\n"
                verified, v_stdout, v_stderr = verify_code(stitched_code, timeout=30)
            return verified, stitched_code, v_stdout, v_stderr

        for b_idx in range(num_batches):
            chunk = dataset_df.iloc[b_idx*val_batch_size : (b_idx+1)*val_batch_size]
            n_chunk = len(chunk)
            raw_prompts = chunk['question'].tolist()

            # Tokenize for generation using the exact same zero-shot solver template.
            input_ids_list = []
            messages_list = []

            for p in raw_prompts:
                # 1. Extract Spec from prompt (handle both raw text and wrapped code)
                raw_p_text = str(p)
                # For validation datasets, the 'question' might already be the raw spec without ```rust
                # If extract_code_from_text returns empty, we should fallback to the raw text
                spec_code = extract_code_from_text(raw_p_text)
                if not spec_code.strip():
                    spec_code = raw_p_text.strip()

                msgs = [
                    {"role": "system", "content": S_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": SOLVER_USER_MSG_TEMPLATE.format(spec_code=spec_code)},
                ]
                
                messages_list.append(msgs)

                # Apply Chat Template + force <think> prefix via partial assistant turn
                msgs.append({"role": "assistant", "content": "<think>\n"})
                ids = to_list_of_ints(self.tokenizer.apply_chat_template(msgs, tokenize=True, continue_final_message=True, add_generation_prompt=False, add_special_tokens=False))
                input_ids_list.append(ids)

            processed_input_ids = input_ids_list
            filtered_raw_prompts = raw_prompts
            filtered_messages = messages_list

            # --- Multi-sample expansion (qs_trainer.py logic) ---
            n_filtered = len(processed_input_ids)
            if n_samples > 1:
                expanded_input_ids = []
                expanded_raw_prompts = []
                expanded_messages = []
                for i in range(n_filtered):
                    for _ in range(n_samples):
                        expanded_input_ids.append(processed_input_ids[i])
                        expanded_raw_prompts.append(filtered_raw_prompts[i])
                        expanded_messages.append(filtered_messages[i])
                processed_input_ids = expanded_input_ids
                working_raw_prompts = expanded_raw_prompts
                working_messages = expanded_messages
            else:
                working_raw_prompts = filtered_raw_prompts
                working_messages = filtered_messages

            # Execute Generation via shared helper
            gen_output, decoded_outputs, val_timing = self._rollout_generate(
                input_ids_list=processed_input_ids,
                raw_prompts=working_messages,
                meta_info={'temperature': 1},
                extra_non_tensor={
                    'uid': np.arange(len(processed_input_ids)),
                    'original_raw_prompt': _to_object_array(working_raw_prompts)
                }
            )

            tasks = [(working_raw_prompts[i], full_text) for i, full_text in enumerate(decoded_outputs)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(tasks))) as executor:
                results = list(executor.map(process_verification_sample, tasks))

            resp_ids = gen_output.batch.get('responses', gen_output.batch['input_ids'])
            extra_rows = []
            summary_rows = []

            for p_idx in range(n_filtered):
                start = p_idx * n_samples
                question_results = results[start:start + n_samples]
                success_count = sum(1 for (v, _, _, _) in question_results if v)
                global_q_idx = b_idx * val_batch_size + p_idx
                question_text = str(filtered_raw_prompts[p_idx]) if p_idx < len(filtered_raw_prompts) else ""
                summary_rows.append({
                    "question_idx": int(global_q_idx),
                    "success_count": int(success_count),
                    "total_count": int(n_samples),
                    "success_rate": float(success_count / max(n_samples, 1)),
                    "verified_any": bool(success_count > 0),
                    "verified_all": bool(success_count == n_samples),
                    "question_text": question_text,
                })
                for s_idx, (verified, stitched_code, v_stdout, v_stderr) in enumerate(question_results):
                    extra_rows.append({
                        "question_idx": int(global_q_idx),
                        "sample_idx_within_question": int(s_idx),
                        "success_count_for_question": int(success_count),
                        "total_count_for_question": int(n_samples),
                        "success_rate_for_question": float(success_count / max(n_samples, 1)),
                        "verified_any_for_question": bool(success_count > 0),
                        "verified_all_for_question": bool(success_count == n_samples),
                        "verify_success": bool(verified),
                        "question_text": question_text,
                        "stitched_code": stitched_code,
                        "verify_stdout": v_stdout,
                        "verify_stderr": v_stderr,
                    })
                problem_results.append((n_samples, success_count))
            problem_results.extend([(0, 0)] * (n_chunk - n_filtered))

            self._dump_token_debug(
                getattr(self, 'global_steps', 0),
                "validate",
                processed_input_ids,
                resp_ids,
                dataset_name=dataset_name,
                extra_rows=extra_rows,
                input_texts=self.tokenizer.batch_decode(processed_input_ids, skip_special_tokens=True),
                output_texts=decoded_outputs,
            )
            self._dump_validation_summary(
                getattr(self, 'global_steps', 0),
                dataset_name,
                summary_rows,
            )

        valid_results = [(n, c) for n, c in problem_results if n > 0]
        if valid_results:
            eligible = [(n, c) for n, c in valid_results if n >= 1]
            if eligible:
                metrics[f"val/{dataset_name}/pass@1"] = float(
                    np.mean([self._pass_at_k(n, c, 1) for n, c in eligible])
                )

        # Write eval pass_rate back to root nodes in seed pool.
        # This gives root nodes a ground-truth difficulty anchor so that
        # neighborhood_pass_rate() and the contraction bonus use the ROOT's
        # actual solvability, not the (circular) average of its children.
        if hasattr(self, 'seed_pool_mgr') and self.seed_pool_mgr:
            n_updated = 0
            current_step = getattr(self, 'global_steps', 0)
            for p_idx, (n_s, n_c) in enumerate(problem_results):
                if n_s <= 0:
                    continue
                q_text = dataset_df.iloc[p_idx]['question'].strip() if p_idx < len(dataset_df) else ''
                if not q_text or q_text not in self.seed_pool_mgr.pool:
                    continue
                eval_pr = n_c / n_s
                entry = self.seed_pool_mgr.pool[q_text]
                # Only update if eval shows the root is solvable (pr > 0)
                # and the eval result is better than current (handles decay)
                if eval_pr > entry.get('pass_rate', 0.0):
                    entry['pass_rate'] = round(eval_pr, 6)
                    entry['last_solved_step'] = int(current_step)
                    n_updated += 1
            if n_updated > 0:
                print(f"[Eval→Pool] Updated {n_updated} root nodes with eval pass_rates", flush=True)

        return metrics

    # ========================== Orchestration ==========================

    def fit(self):
        print("[QuestionSolverTrainer] Starting custom fit loop", flush=True)
        logger = Tracking(
            project_name=self.config.trainer.project_name, experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger, config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        val_only = bool(self.config.trainer.get("val_only", False) or self.config.trainer.get("only_eval", False))
        if val_only:
            self._run_eval_only_all_checkpoints(logger)
            return

        self._load_checkpoint()

        if not self.seed_pool_mgr.load():
            print("[WARN] No seed pool found. Starting with empty pool.", flush=True)
        
        self.seed_pool = self.seed_pool_mgr.pool
        total_seeds = len(self.seed_pool)
        
        k_shots = self.config.algorithm.get("max_shots", 3)
        self.current_k_shots = min(total_seeds, k_shots) if total_seeds > 0 else 0

        val_before_train = self.config.trainer.get("val_before_train", True)
        eval_rolls = self._get_eval_rolls()
        
        if val_before_train and self.val_datasets:
            for val_name, val_df in self.val_datasets.items():
                logger.log(data=self._validate(val_name, val_df, n_samples=eval_rolls), step=self.global_steps)

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="QS Training")
        train_iterator = iter(self.train_dataloader)
        
        while self.global_steps < self.total_training_steps:
            try: batch_dict = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dataloader)
                batch_dict = next(train_iterator)
                
            metrics, timing_raw = {}, {}

            with marked_timer("preprocess", timing_raw):
                batch = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.q_gen_temperature
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

            # --- Stage 1: Generate Questions (Proposer) ---
            qr = self._generate_questions(batch, timing_raw)
            q_gen_output        = qr.gen_output
            t1_prompts          = qr.turn1_prompts
            t1_seeds            = qr.turn1_seeds
            generated_qs        = qr.generated_qs
            generated_specs_masked = qr.generated_specs_masked
            generated_q_bodies  = qr.generated_q_bodies
            q_valid_flags       = qr.q_valid_flags
            q_err_scores        = qr.q_err_scores
            q_sim_scores        = qr.sim_scores
            q_err_compiler_outs = qr.q_err_compiler_outs
            q_stitched_codes    = qr.q_stitched_codes

            metrics["qgen/heuristic_rejects"]             = float(qr.heuristic_rejects)
            metrics["qgen/heuristic_reject_rate"]         = float(qr.heuristic_rejects) / max(float(qr.candidate_count), 1.0)
            metrics["qgen/format_rejects"]                = float(qr.format_rejects)
            metrics["qgen/format_reject_rate"]            = float(qr.format_rejects) / max(float(qr.candidate_count), 1.0)
            metrics["qgen/compiler_rejects"]              = float(qr.compiler_rejects)
            metrics["qgen/compiler_reject_rate"]          = float(qr.compiler_rejects) / max(float(qr.candidate_count), 1.0)
            metrics["qgen/to_solver_questions"]           = float(sum(1 for x in qr.q_valid_flags if x))
            metrics["qgen/to_solver_samples"]             = float(sum(1 for x in qr.q_valid_flags if x) * self.k_answers)
            metrics["qgen/bad_ensures_rejects"]           = float(qr.bad_ensures_rejects)
            if qr.q_parse_results:
                metrics["qgen/salvaged_primary_fn_rate"] = float(np.mean([
                    1.0 if getattr(res, "salvaged_primary_fn", False) else 0.0
                    for res in qr.q_parse_results
                ]))
                metrics["qgen/multi_fn_candidate_rate"] = float(np.mean([
                    1.0 if int(getattr(res, "fn_count", 0)) > 1 else 0.0
                    for res in qr.q_parse_results
                ]))
                metrics["qgen/spec_soft_overflow_rate"] = float(np.mean([
                    1.0 if (
                        int(getattr(res, "spec_char_len", 0)) > int(self.config.algorithm.get("q_gen_soft_max_spec_chars", 2200))
                        or int(getattr(res, "spec_line_count", 0)) > int(self.config.algorithm.get("q_gen_soft_max_spec_lines", 32))
                    ) else 0.0
                    for res in qr.q_parse_results
                ]))
            # --- Stage 2: Generate Solutions (Solver) ---
            s_gen_output, decoded_responses, _ = self._generate_solutions(batch, generated_specs_masked, timing_raw, q_valid_flags=q_valid_flags, q_seeds=t1_seeds)

            # --- Stage 3: Verify and Track Submissions (Verifier) ---
            solve_scores, question_scores_safe, pass_rates, clean_answers, reward_extra_infos, corrective_results, corrective_batch_outputs, corrective_all_scores = self._verify_solutions(
                generated_q_bodies, decoded_responses, t1_prompts, generated_qs, metrics, timing_raw,
                q_valid_flags=q_valid_flags, q_err_scores=q_err_scores, q_sim_scores=q_sim_scores, q_err_compiler_outs=q_err_compiler_outs,
                generated_specs_masked=generated_specs_masked, q_stitched_codes=q_stitched_codes,
                q_filter_results=qr.q_filter_results, q_seeds=t1_seeds, q_parse_results=qr.q_parse_results,
            )
            # Attach filter_stages/filter_reasons to reward_extra_infos for _log_rollout_questions
            reward_extra_infos["filter_stages"] = qr.filter_stages
            reward_extra_infos["filter_reasons"] = qr.filter_reasons

            # [Adaptive Difficulty] Update solver ability EMA using only specs where
            # 0 < p < 1 (contested). p=0 and p=1 are uninformative about true ability.
            contested = pass_rates[(pass_rates > 0.0) & (pass_rates < 1.0)]
            if len(contested) > 0:
                alpha = self._s_ability_ema_alpha
                self._s_ability_ema = alpha * float(np.median(contested)) + (1.0 - alpha) * self._s_ability_ema
            metrics["adaptive/s_ability_ema"] = self._s_ability_ema

            # Dynamic Update (Seed Pool O(1) Deduplication & EMA)
            # [Spec-Only FIX] Use S-Gen's (or Corrective's) best verified solution as the seed answer
            # since Q-Gen no longer generates the implementation body.
            seed_gate_cfg = self.config.algorithm.get("seed_pool_gate", {}) if hasattr(self.config, "algorithm") else {}
            anchor_phase_cfg = self.config.algorithm.get("anchor_phase", {}) if hasattr(self.config, "algorithm") else {}
            freeze_seed_pool = bool(anchor_phase_cfg.get("freeze_seed_pool", False))
            seed_min_impl_chars = int(seed_gate_cfg.get("min_impl_chars", 20))
            seed_max_spec_sim = float(seed_gate_cfg.get("max_seed_sim", 0.90))
            seed_reject_struct_dup = bool(seed_gate_cfg.get("reject_struct_dup", True))
            seed_short_impl_rejects = 0
            seed_overlong_rejects = 0
            seed_unsolved_rejects = 0
            seed_mutation_rejects = 0
            seed_shallow_rejects = 0
            seed_similarity_rejects = 0
            seed_quality_rejects = 0
            for i_q in range(len(pass_rates)):
                if q_valid_flags[i_q]:
                    q_spec = generated_specs_masked[i_q]
                    q_debug = reward_extra_infos["debug_infos"][i_q] if i_q < len(reward_extra_infos.get("debug_infos", [])) else {}
                    q_parse_info = qr.q_parse_results[i_q] if i_q < len(qr.q_parse_results) else None

                    # Find the corresponding original valid_idx for solve_scores map
                    valid_idx = sum(q_valid_flags[:i_q])
                    start_idx = valid_idx * self.k_answers
                    end_idx = start_idx + self.k_answers
                    
                    # Extract the scores and answers for this specific question
                    q_s_scores = solve_scores[start_idx:end_idx] if len(solve_scores) >= end_idx else []
                    q_s_strict_passes = reward_extra_infos.get("strict_passes", [])[start_idx:end_idx] if len(reward_extra_infos.get("strict_passes", [])) >= end_idx else []
                    q_s_answers = clean_answers[start_idx:end_idx] if len(clean_answers) >= end_idx else []
                    
                    # Extract verifier outputs
                    all_v_outs = reward_extra_infos.get("verus_outputs", [])
                    q_s_v_outs = all_v_outs[start_idx:end_idx] if len(all_v_outs) >= end_idx else []
                    
                    best_ans = ""
                    best_raw = ""
                    best_sc = -1
                    q_s_raw = decoded_responses[start_idx:end_idx] if len(decoded_responses) >= end_idx else []
                    for _j, _sc in enumerate(q_s_scores):
                        if _j < len(q_s_strict_passes) and q_s_strict_passes[_j] and _sc > best_sc:
                            best_sc = _sc
                            best_ans = q_s_answers[_j] if _j < len(q_s_answers) else ""
                            best_raw = q_s_raw[_j] if _j < len(q_s_raw) else ""

                    # If first round failed, check corrective results for a solution
                    if best_sc < 0 and valid_idx in corrective_results:
                        cr = corrective_results[valid_idx]
                        if cr['corrective_pass']:
                            best_ans = cr['best_solution']
                            best_raw = cr.get('best_raw_response', '')
                    
                    # Quality gate for seed-pool admission. Under UCB, admitting
                    # unsolved compiler-valid nodes proved too permissive: the pool
                    # gets flooded with zero-pass / format-error nodes that soak up
                    # exploration budget. We therefore keep the main training pool
                    # solved-only and use gating to reject unresolved candidates.
                    seed_sim = float(q_debug.get("seed_sim", q_sim_scores[i_q] if q_sim_scores and i_q < len(q_sim_scores) else 0.0))
                    struct_sim = float(q_debug.get("struct_sim", 0.0))
                    reject_shallow, _ = is_collapsed_qgen_spec(q_spec)
                    trivial_pass, _ = trivial_seed_solution_passes(q_spec) if not reject_shallow else (False, "")
                    pool_quality_ok, pool_quality_reason = self._q_pool_quality_gate(q_parse_info, q_debug)
                    whole_pool_sim = 0.0
                    q_spec_stripped = q_spec.strip()
                    if q_spec_stripped:
                        for pool_q in self.seed_pool_mgr.pool.keys():
                            if pool_q == q_spec_stripped:
                                continue
                            sim = self.seed_pool_mgr.question_similarity(q_spec_stripped, pool_q)
                            if sim > whole_pool_sim:
                                whole_pool_sim = sim
                    max_pool_sim = float(seed_gate_cfg.get("max_pool_sim", 1.0))

                    if seed_sim > seed_max_spec_sim or (seed_reject_struct_dup and struct_sim >= 1.0):
                        seed_similarity_rejects += 1
                    elif whole_pool_sim >= max_pool_sim:
                        seed_similarity_rejects += 1
                    elif reject_shallow:
                        seed_shallow_rejects += 1
                    elif trivial_pass:
                        seed_mutation_rejects += 1
                    elif not pool_quality_ok:
                        seed_quality_rejects += 1
                    elif freeze_seed_pool:
                        pass
                    else:
                        if best_ans and (seed_min_impl_chars <= len(best_ans) <= 3000) and len(q_spec) <= 2000:
                            # Solved: enter pool with pass_rate=1.0 and the verified answer.
                            updated = self.seed_pool_mgr.add_or_update(q_spec, best_ans, 1.0, self.global_steps, raw_response=best_raw)
                        else:
                            updated = False
                            if best_ans:
                                seed_overlong_rejects += 1
                            else:
                                seed_unsolved_rejects += 1

                        if updated:
                            seed_q_str = None
                            if i_q < len(t1_seeds):
                                shots = t1_seeds[i_q].get("shots", [])
                                if shots:
                                    seed_dict = shots[0][0] if isinstance(shots[0], (tuple, list)) else shots[0]
                                    seed_q_str = seed_dict.get('question', '').strip() if isinstance(seed_dict, dict) else None
                            if seed_q_str and seed_q_str != q_spec.strip():
                                self.seed_pool_mgr.add_edge(seed_q_str, q_spec.strip(), "curriculum")
                        selection_path = list(t1_seeds[i_q].get("selection_path", [])) if i_q < len(t1_seeds) else []
                        # PPO learns from a softer local novelty penalty, but DAG credit
                        # Tree credit should track clean solver utility only.
                        # Structural/format diagnostics are handled separately.
                        tree_reward = float(max(q_debug.get("solver_utility", 0.0), 0.0))
                        progress_gain = float(max(q_debug.get("progress_gain", 0.0), 0.0))
                        seed_nbr_pr = float(q_debug.get("seed_nbr_pr", 0.0))
                        struct_sim = float(q_debug.get("struct_sim", 0.0))
                        if struct_sim >= 1.0:
                            tree_reward = 0.0
                        if seed_nbr_pr > 0.0:
                            tree_reward *= progress_gain
                        self.seed_pool_mgr.backpropagate_path(selection_path, tree_reward)

                    if best_ans and (seed_min_impl_chars > len(best_ans)):
                        seed_short_impl_rejects += 1

            self.seed_pool = self.seed_pool_mgr.pool
            # Forgetting decay: solved nodes (pass_rate=1.0) drift back toward
            # similar/downgrade if the model stops solving them.
            _decay = float(self.config.algorithm.get("pass_rate_decay", 0.99)) if hasattr(self.config, "algorithm") else 0.99
            if _decay < 1.0 and not freeze_seed_pool:
                self.seed_pool_mgr.decay_pass_rates(_decay, current_step=self.global_steps)
            # Always persist observational snapshots, even during anchor/freeze phases.
            # Freezing only means "do not mutate the active training pool", not
            # "hide the current pool state from analysis".
            self.seed_pool_mgr.save()
            if not freeze_seed_pool:
                prune_min_visits = int(self.config.algorithm.get("mcts_prune_min_visits", 0))
                if prune_min_visits > 0:
                    # Only prune once the solver has demonstrated some capability.
                    # During the zero-signal phase pass_rate=0 reflects model weakness,
                    # not unsolvability; guard against premature eviction.
                    global_any_solved = float(metrics.get("reward/pass_rate_mean", 0.0)) > 0.01
                    if global_any_solved:
                        n_pruned = self.seed_pool_mgr.prune_unsolvable(min_visits=prune_min_visits)
                        if n_pruned > 0:
                            metrics["seed_pool/pruned"] = float(n_pruned)

            # UCB / graph diagnostics
            pool_items = list(self.seed_pool_mgr.pool.values())
            if pool_items:
                ucb_c_diag = float(self.config.algorithm.get("ucb_c", 1.0)) if hasattr(self.config, "algorithm") else 1.0
                ucb_scores_all = [self.seed_pool_mgr.ucb_score(it['question'], c=ucb_c_diag) for it in pool_items]
                pass_rates_all = [it['pass_rate'] for it in pool_items]
                # DAG-UCT value diagnostics
                path_value_sums = [self.seed_pool_mgr.value_sums.get(it['question'], 0.0) for it in pool_items]
                n_productive = sum(1 for s in path_value_sums if s > 0.0)
                metrics["pool/ucb_score_mean"]       = float(np.mean(ucb_scores_all))
                metrics["pool/ucb_score_max"]        = float(np.max(ucb_scores_all))
                metrics["pool/pass_rate_mean"]       = float(np.mean(pass_rates_all))
                metrics["pool/pass_rate_std"]        = float(np.std(pass_rates_all))
                metrics["pool/n_productive_nodes"] = float(n_productive)
                metrics["pool/path_value_sum"]     = float(sum(path_value_sums))
                metrics["pool/total_visits"]      = float(self.seed_pool_mgr._total_visits)
                metrics["pool/graph_edge_nodes"]  = float(len(self.seed_pool_mgr.graph_edges))

            # --- Stage 4: Sync Advantages (Tensor Broadcasting) ---
            with marked_timer("prepare_batch", timing_raw):
                unified_batch = self._prepare_unified_batch(
                    q_gen_output, s_gen_output, 
                    corrective_batch_outputs=corrective_batch_outputs
                )
                old_log_prob, _ = self._compute_old_log_prob(unified_batch)

                # Log actor/entropy (matching ray_trainer.py logic)
                if "entropys" in old_log_prob.batch:
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = unified_batch.batch["response_mask"]
                    is_question_mask = unified_batch.batch["is_question"]
                    is_corrective_mask = unified_batch.batch["is_corrective"] if "is_corrective" in unified_batch.batch else torch.zeros_like(is_question_mask)
                    actor_config = self.config.actor_rollout_ref.actor
                    entropy_agg = agg_loss(
                        loss_mat=entropys,
                        loss_mask=response_masks,
                        loss_agg_mode=actor_config.loss_agg_mode,
                        loss_scale_factor=actor_config.get("loss_scale_factor", 1.0),
                    )
                    metrics["actor/entropy"] = entropy_agg.detach().item()

                    q_response_mask = response_masks * is_question_mask.unsqueeze(1)
                    s_response_mask = response_masks * (~is_question_mask).unsqueeze(1)
                    s_corrective_response_mask = response_masks * ((~is_question_mask) & is_corrective_mask).unsqueeze(1)

                    if q_response_mask.sum().item() > 0:
                        q_entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=q_response_mask,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.get("loss_scale_factor", 1.0),
                        )
                        metrics["actor/q_entropy"] = q_entropy_agg.detach().item()

                    if s_response_mask.sum().item() > 0:
                        s_entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=s_response_mask,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.get("loss_scale_factor", 1.0),
                        )
                        metrics["actor/s_entropy"] = s_entropy_agg.detach().item()

                    if s_corrective_response_mask.sum().item() > 0:
                        s_corrective_entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=s_corrective_response_mask,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.get("loss_scale_factor", 1.0),
                        )
                        metrics["actor/s_corrective_entropy"] = s_corrective_entropy_agg.detach().item()
                    old_log_prob.batch.pop("entropys")

                unified_batch = unified_batch.union(old_log_prob)

            with marked_timer("compute_adv", timing_raw):
                metrics.update(self._compute_coupled_advantages(
                    unified_batch,
                    question_scores_safe,
                    solve_scores,
                    strict_passes_s=reward_extra_infos.get("strict_passes_for_ppo", reward_extra_infos.get("strict_passes")),
                    rewards_c=corrective_all_scores,
                    rewards_q_solver=reward_extra_infos.get("question_solver_scores"),
                    rewards_q_filter=reward_extra_infos.get("question_filter_scores"),
                    q_valid_flags=q_valid_flags,
                ))
                if "global_token_num" not in unified_batch.meta_info: unified_batch.meta_info["global_token_num"] = unified_batch.batch['response_mask'].sum(dim=1).tolist()

            # --- Stage 5: Weight Opt (PPO Actor/Critic) ---
            self._update_policy_step("train", unified_batch, timing_raw, metrics)

            # --- Stage 6: Telemetry & Checkpoints ---
            metrics.update(compute_timing_metrics(unified_batch, timing_raw))
            if self.config.trainer.get("log_rollout", True):
                self._log_rollout_data(
                    batch=unified_batch,
                    reward_extra_infos_dict=reward_extra_infos,
                    timing_raw=timing_raw,
                    rollout_data_dir=os.path.join(self.config.trainer.default_local_dir, "rollout_logs"),
                )
                self._log_rollout_questions(
                    generated_specs_masked=generated_specs_masked,
                    q_valid_flags=q_valid_flags,
                    q_err_scores=q_err_scores,
                    q_err_compiler_outs=q_err_compiler_outs,
                    q_stitched_codes=q_stitched_codes,
                    q_sim_scores=q_sim_scores,
                    reward_extra_infos=reward_extra_infos,
                    out_dir=os.path.join(self.config.trainer.default_local_dir, "question_rollouts"),
                )

            if self.val_datasets and (self.global_steps + 1) % self.config.trainer.get("test_freq", 10) == 0:
                for val_name, val_df in self.val_datasets.items():
                    metrics.update(self._validate(val_name, val_df, n_samples=eval_rolls))

            # Do not report solved/unsolved from the instantaneous pass_rate because
            # pass_rate is intentionally decayed over time. Use whether the node has
            # ever received positive evidence instead.
            n_ever_solved = sum(
                1 for it in self.seed_pool.values()
                if float(it.get('pass_rate', 0.0)) > 0.0 or int(it.get('last_solved_step', -1)) >= 0
            )
            n_never_solved = len(self.seed_pool) - n_ever_solved
            n_unsolved_original_roots = sum(
                1 for q, it in self.seed_pool.items()
                if bool(it.get("is_original", False))
                and int(it.get("last_solved_step", -1)) < 0
            )
            metrics.update({
                'seed_pool/high_similarity_filtered': float(seed_similarity_rejects),
                'seed_pool/shallow_filtered': float(seed_shallow_rejects),
                'seed_pool/short_impl_filtered': float(seed_short_impl_rejects),
                'seed_pool/unsolved_filtered': float(seed_unsolved_rejects),
                'seed_pool/mutation_filtered': float(seed_mutation_rejects),
                'seed_pool/quality_filtered': float(seed_quality_rejects),
                'seed_pool/total': len(self.seed_pool),
                'seed_pool/ever_solved': n_ever_solved,
                'seed_pool/never_solved': n_never_solved,
                'seed_pool/unsolved_original_roots': n_unsolved_original_roots,
            })

            self._emit_step_log(metrics)
            logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)
            self.global_steps += 1

            if self.config.trainer.save_freq > 0 and (self.global_steps >= self.total_training_steps or self.global_steps % self.config.trainer.save_freq == 0):
                with marked_timer("save_checkpoint", timing_raw, color="green"): self._save_checkpoint()
