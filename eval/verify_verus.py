"""
Verify with auto-repair for common brace issues.
Repairs: strip redundant leading {, append missing }, normalize inline invariants.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

VERUS_BIN = "/workspace/verl/verus/verus-x86-linux/verus"
CONTAINER = "verl_eval"
ENV_PREFIX = 'export PATH="/root/.cargo/bin:$PATH"'


def stitch_code(spec_code: str, body_code: str) -> str:
    spec = spec_code.strip()
    body = body_code.strip()
    for wrapper in ["verus! {", "} // verus!"]:
        spec = spec.replace(wrapper, "")
        body = body.replace(wrapper, "")
    spec = spec.strip()
    body = body.strip()

    body_lines = []
    extra_uses = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("use "):
            extra_uses.append(stripped)
        elif stripped == "fn main() {}":
            continue
        else:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()

    combined = f"{spec}\n{body}"
    uses = ["use vstd::prelude::*;"] + extra_uses
    use_block = "\n".join(sorted(set(uses)))
    return f"{use_block}\n\nverus! {{\n\n{combined}\n\nfn main() {{}}\n\n}} // verus!\n"


def repair_body(spec_code: str, body_code: str) -> tuple[str, str]:
    """Try to repair common body issues. Returns (repaired_body, repair_kind)."""
    body = body_code.strip()
    repairs = []

    # Repair 1: body starts with { but spec already ends with {
    spec_stripped = spec_code.strip()
    for wrapper in ["verus! {", "} // verus!"]:
        spec_stripped = spec_stripped.replace(wrapper, "")
    spec_stripped = spec_stripped.strip()
    spec_ends_with_brace = spec_stripped.rstrip().endswith("{")

    if spec_ends_with_brace and body.startswith("{"):
        body = body[1:].strip()
        repairs.append("strip_leading_brace")

    # Repair 2: normalize inline loop invariants (while cond \n invariant ... { → while cond \n invariant ... \n{)
    # This is already handled by Verus in most cases

    # Repair 3: append missing closing braces
    code = stitch_code(spec_code, body)
    opens = code.count("{")
    closes = code.count("}")
    if opens > closes:
        diff = opens - closes
        body = body + "\n" + ("}\n" * diff)
        repairs.append(f"append_{diff}_braces")

    # Repair 4: if body ends with extra }, remove
    code2 = stitch_code(spec_code, body)
    opens2 = code2.count("{")
    closes2 = code2.count("}")
    if closes2 > opens2:
        diff = closes2 - opens2
        for _ in range(diff):
            body = body.rstrip()
            if body.endswith("}"):
                body = body[:-1].rstrip()
        repairs.append(f"strip_{diff}_braces")

    repair_kind = "+".join(repairs) if repairs else "none"
    return body, repair_kind


def verify_in_docker(code: str, idx: int, sol_idx: int, timeout: int = 30) -> tuple[bool, str]:
    tmp_name = f"/tmp/verify_{idx}_{sol_idx}_{threading.current_thread().ident}.rs"
    b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    write_cmd = f'echo "{b64}" | base64 -d > {tmp_name}'
    verify_cmd = f'{ENV_PREFIX} && {VERUS_BIN} {tmp_name} --multiple-errors 5 2>&1; echo "EXIT_CODE:$?"'
    cleanup_cmd = f'rm -f {tmp_name}'
    full_cmd = f'{write_cmd} && {verify_cmd}; {cleanup_cmd}'

    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER, "bash", "-c", full_cmd],
            capture_output=True, text=True, timeout=timeout + 10,
            env={**os.environ, "MSYS_NO_PATHCONV": "1"},
        )
        output = result.stdout.strip()
        exit_code = -1
        lines = output.split("\n")
        for line in reversed(lines):
            if line.startswith("EXIT_CODE:"):
                try:
                    exit_code = int(line.split(":")[1])
                except ValueError:
                    pass
                break
        if exit_code == 0:
            lower = output.lower()
            if "error" not in lower or "0 errors" in lower:
                return True, ""
        err_lines = [l for l in lines if "error" in l.lower() and not l.startswith("EXIT_CODE")]
        snippet = "; ".join(err_lines[:3])[:200] if err_lines else output[-200:]
        return False, snippet
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["docker", "exec", CONTAINER, "bash", "-c", f"rm -f {tmp_name}"],
            capture_output=True, timeout=10,
            env={**os.environ, "MSYS_NO_PATHCONV": "1"},
        )
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="qsv/data/curated_sft/diverse_sgen_gemini_unverified.jsonl")
    parser.add_argument("--output", default="qsv/data/curated_sft/diverse_sgen_gemini_verified_v2.jsonl")
    parser.add_argument("--skip_verified", default="qsv/data/curated_sft/diverse_sgen_gemini_verified.jsonl",
                        help="Skip already-verified records from v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    entries = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Load already-verified from v1 (pass through directly)
    already_verified = []
    skip_keys = set()
    if args.skip_verified and os.path.exists(args.skip_verified):
        with open(args.skip_verified, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    already_verified.append(r)
                    skip_keys.add((r["idx"], r["solution_idx"]))

    to_verify = [e for e in entries if (e["idx"], e["solution_idx"]) not in skip_keys]
    print(f"Total: {len(entries)}, already verified: {len(already_verified)}, to verify (with repair): {len(to_verify)}")

    # Write already-verified to output first
    with open(args.output, "w", encoding="utf-8") as f:
        for r in already_verified:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_lock = threading.Lock()
    ok_count = len(already_verified)
    fail_count = 0
    repair_stats = {}
    t_start = time.time()

    def _verify(entry):
        spec = entry.get("question", "").strip()
        body_orig = entry.get("answer_api", "").strip()
        if not spec or not body_orig:
            return entry, False, "empty", "none"

        # Try original first
        code = stitch_code(spec, body_orig)
        verified, err = verify_in_docker(code, entry.get("idx", 0), entry.get("solution_idx", 0), args.timeout)
        if verified:
            return entry, True, "", "original_ok"

        # Try repaired
        body_repaired, repair_kind = repair_body(spec, body_orig)
        if repair_kind != "none":
            code2 = stitch_code(spec, body_repaired)
            verified2, err2 = verify_in_docker(code2, entry.get("idx", 0), entry.get("solution_idx", 0), args.timeout)
            if verified2:
                entry["answer_api"] = body_repaired
                return entry, True, "", f"repaired_{repair_kind}"
            return entry, False, err2, f"repair_failed_{repair_kind}"

        return entry, False, err, "original_failed"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_verify, e) for e in to_verify]
        for fut in as_completed(futures):
            entry, verified, err, status = fut.result()
            repair_stats[status] = repair_stats.get(status, 0) + 1
            if verified:
                ok_count += 1
                entry["verified"] = True
                entry["verified_repair_status"] = status
                with write_lock:
                    with open(args.output, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            else:
                fail_count += 1

            done = ok_count + fail_count - len(already_verified)
            if done % 20 == 0 or done == len(to_verify):
                elapsed = time.time() - t_start
                print(f"  [{done}/{len(to_verify)}] new_verified={ok_count - len(already_verified)} failed={fail_count} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Total verified: {ok_count} (prev={len(already_verified)} + new={ok_count - len(already_verified)})")
    print(f"Failed: {fail_count}")
    print(f"\nRepair stats:")
    for k, v in sorted(repair_stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
