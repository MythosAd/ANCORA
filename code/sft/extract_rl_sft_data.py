#!/usr/bin/env python3
"""
Extract high-quality Q-Gen and S-Gen SFT data from RL rollout logs (v4, v6),
deduplicate using MinHash similarity, merge with existing
all_merged_sft_v3.parquet to produce all_merged_sft_v4.parquet.

Strategy:
- S-Gen: collect ALL verified solutions (solve_score=1.0), but filter by
  min response length (collapse indicator) and deduplicate per spec.
  For each unique spec, keep the LONGEST verified solution (best quality).
- Q-Gen: collect questions that were actually solved (pass_rate > 0),
  keep one per unique spec.
- Cross-dedup against existing SFT data using MinHash Jaccard.
"""

import json
import os
import re
import sys
import hashlib
import struct
from pathlib import Path
from collections import defaultdict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Inlined MinHash utilities from qs_trainer.py (pure Python, no torch) ───

_MINHASH_NUM_PERM = 64
_MINHASH_NGRAM_N = 3
_MINHASH_MAX_INT = (1 << 32)
_MERSENNE_PRIME = (1 << 61) - 1
_rng_state = 12345

def _next_rng():
    global _rng_state
    _rng_state = (_rng_state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return _rng_state

_HASH_A = [_next_rng() % _MERSENNE_PRIME for _ in range(_MINHASH_NUM_PERM)]
_HASH_B = [_next_rng() % _MERSENNE_PRIME for _ in range(_MINHASH_NUM_PERM)]

def _text_to_shingles(text: str, n: int = _MINHASH_NGRAM_N) -> list:
    t = re.sub(r'[^\w\s]', ' ', text.lower())
    t = re.sub(r'\s+', ' ', t).strip()
    if len(t) < n:
        return [t] if t else []
    return [t[i:i+n] for i in range(len(t) - n + 1)]

def _shingle_hash(shingle: str) -> int:
    h = hashlib.md5(shingle.encode('utf-8')).digest()
    return struct.unpack('<I', h[:4])[0]

def compute_minhash(text: str) -> list:
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
    if not sig_a or not sig_b:
        return 0.0
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def extract_fn_signature(spec: str) -> str:
    m = re.search(r'fn\s+\w+\s*\(([^)]*)\)(?:\s*->\s*(\S+))?', spec)
    if not m:
        return ""
    params = m.group(1).strip()
    ret = (m.group(2) or "").strip()
    param_types = []
    for p in params.split(','):
        p = p.strip()
        if ':' in p:
            param_types.append(p.split(':')[-1].strip())
        elif p:
            param_types.append(p)
    return f"({','.join(param_types)})->{ret}"

# ─── Config ───────────────────────────────────────────────────────
ROLLOUT_DIRS = {
    "v4": "qsv/GPT_analysis/SFTinit-RLv4/rollout_logs",
    "v6": "qsv/GPT_analysis/Qwen2.5-Coder-3B-Instruct-QGSG-SFTinit-RL-v6/rollout_logs",
}
EXISTING_SFT = "qsv/data/curated_sft/all_merged_sft_v3.parquet"
OUTPUT_SFT = "qsv/data/curated_sft/all_merged_sft_v4.parquet"

# Filtering
MIN_S_RESPONSE_CHARS = 150       # Filter collapsed short responses
MIN_Q_RESPONSE_CHARS = 80
MAX_RESPONSE_CHARS = 12000
DEDUP_SIM_THRESHOLD = 0.80       # MinHash Jaccard > this → duplicate
# Use ALL steps but let response length filter remove collapse artifacts
V4_MAX_STEP = 706
V6_MAX_STEP = 427


def extract_spec_from_prompt(prompt_messages):
    """Extract the Verus spec from the user message in s_gen prompt."""
    for msg in prompt_messages:
        if msg.get("role") == "user":
            content = msg["content"]
            m = re.search(r"```rust\s*\n(.*?)```", content, re.DOTALL)
            if m:
                return m.group(1).strip()
    return ""


def clean_response(response: str) -> str:
    return response.strip()


def load_all_rollout_data():
    """Load Q-Gen and S-Gen entries from all rollout logs."""
    all_q = []
    all_s = []

    for run_name, rollout_dir in ROLLOUT_DIRS.items():
        max_step = V4_MAX_STEP if run_name == "v4" else V6_MAX_STEP
        base = os.path.join(str(PROJECT_ROOT), rollout_dir)
        files = sorted(Path(base).glob("rollout_step_*.jsonl"))
        print(f"  [{run_name}] Found {len(files)} rollout files")

        q_raw = s_raw = q_kept = s_kept = 0
        for fpath in files:
            step = int(fpath.stem.split("_")[-1])
            if step > max_step:
                continue
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    is_q = d.get("is_question", False)
                    prompt = d.get("prompt", [])
                    response = clean_response(d.get("response", ""))
                    debug = d.get("debug_info", {})
                    if isinstance(debug, str):
                        try:
                            debug = json.loads(debug)
                        except:
                            debug = {}

                    if is_q:
                        q_raw += 1
                        pass_rate = debug.get("pass_rate", 0) or 0
                        if pass_rate <= 0:
                            continue
                        if len(response) < MIN_Q_RESPONSE_CHARS:
                            continue
                        if len(response) > MAX_RESPONSE_CHARS:
                            continue
                        if "<think>" not in response:
                            continue
                        stitched = debug.get("q_stitched_code", "")
                        q_kept += 1
                        all_q.append({
                            "run": run_name, "step": step,
                            "prompt": prompt, "response": response,
                            "spec": stitched,
                            "pass_rate": pass_rate,
                            "resp_len": len(response),
                        })
                    else:
                        s_raw += 1
                        solve_score = d.get("solve_score", 0) or 0
                        if solve_score < 1.0:
                            continue
                        verus_out = str(debug.get("verus_output", ""))
                        if "verified" not in verus_out.lower():
                            continue
                        if len(response) < MIN_S_RESPONSE_CHARS:
                            continue
                        if len(response) > MAX_RESPONSE_CHARS:
                            continue
                        spec = extract_spec_from_prompt(prompt)
                        if len(spec) < 20:
                            continue
                        s_kept += 1
                        all_s.append({
                            "run": run_name, "step": step,
                            "prompt": prompt, "response": response,
                            "spec": spec,
                            "solve_score": solve_score,
                            "resp_len": len(response),
                        })

        print(f"  [{run_name}] Q: {q_raw} total → {q_kept} kept | "
              f"S: {s_raw} total → {s_kept} kept (verified + len filter)")

    return all_q, all_s


def per_spec_dedup(entries, label):
    """For entries solving the same spec, keep only the longest response."""
    print(f"\n  Per-spec dedup for {len(entries)} {label} entries...")
    # Group by spec MinHash (approximate grouping)
    spec_groups = defaultdict(list)
    for e in entries:
        spec_text = e.get("spec", "")
        sig = tuple(compute_minhash(spec_text))
        # Use a coarser grouping: first 16 elements of minhash
        coarse_key = sig[:16]
        spec_groups[coarse_key].append(e)

    # Within each group, do pairwise sim check and keep best per cluster
    kept = []
    for group_key, group in spec_groups.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        # Compute full minhash for fine dedup
        for e in group:
            if "_sig" not in e:
                e["_sig"] = compute_minhash(e.get("spec", ""))
        # Greedy cluster: pick longest, remove similar
        group.sort(key=lambda x: x["resp_len"], reverse=True)
        cluster_reps = []
        for e in group:
            is_dup = False
            for rep in cluster_reps:
                sim = minhash_jaccard(e["_sig"], rep["_sig"])
                if sim > DEDUP_SIM_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                cluster_reps.append(e)
        kept.extend(cluster_reps)

    print(f"  {label}: {len(entries)} → {len(kept)} after per-spec dedup")
    return kept


def cross_dedup(entries, label, existing_sigs):
    """Remove entries too similar to existing SFT data."""
    print(f"\n  Cross-dedup {len(entries)} {label} against {len(existing_sigs)} existing...")
    kept = []
    removed = 0
    for e in entries:
        if "_sig" not in e:
            e["_sig"] = compute_minhash(e.get("spec", "") or e.get("response", ""))
        is_dup = False
        for esig in existing_sigs:
            if minhash_jaccard(e["_sig"], esig) > DEDUP_SIM_THRESHOLD:
                is_dup = True
                removed += 1
                break
        if not is_dup:
            kept.append(e)
    print(f"  {label}: {len(entries)} → {len(kept)} (removed {removed} existing-dups)")
    return kept


def format_sft_row(entry, data_source):
    return {
        "messages": entry["prompt"] + [{"role": "assistant", "content": entry["response"]}],
        "data_source": data_source,
    }


def main():
    print("=" * 60)
    print("Extract RL rollout SFT data -> all_merged_sft_v4.parquet")
    print("=" * 60)

    # 1. Load existing SFT data
    existing_path = os.path.join(str(PROJECT_ROOT), EXISTING_SFT)
    df_existing = pd.read_parquet(existing_path)
    print(f"\nExisting SFT data: {len(df_existing)} rows")
    print(df_existing["data_source"].value_counts().to_string())

    # 2. Build MinHash signatures from existing data
    print("\nBuilding MinHash signatures from existing SFT data...")
    existing_sigs = []
    for _, row in df_existing.iterrows():
        msgs = row["messages"]
        for m in msgs:
            if m.get("role") == "user":
                existing_sigs.append(compute_minhash(m["content"]))
                break
    print(f"  Built {len(existing_sigs)} existing signatures")

    # 3. Load rollout data (all steps, with len filter)
    print("\n--- Loading rollout data ---")
    all_q, all_s = load_all_rollout_data()
    print(f"\nTotal after basic filter: {len(all_q)} Q-Gen, {len(all_s)} S-Gen")

    # 4. Response length distribution
    print("\n--- Response length analysis ---")
    if all_s:
        lens = [e["resp_len"] for e in all_s]
        lens.sort()
        print(f"  S-Gen: min={min(lens)}, p25={lens[len(lens)//4]}, "
              f"median={lens[len(lens)//2]}, p75={lens[3*len(lens)//4]}, max={max(lens)}")
    if all_q:
        lens = [e["resp_len"] for e in all_q]
        lens.sort()
        print(f"  Q-Gen: min={min(lens)}, p25={lens[len(lens)//4]}, "
              f"median={lens[len(lens)//2]}, p75={lens[3*len(lens)//4]}, max={max(lens)}")

    # 5. Per-spec dedup (keep longest per unique spec)
    print("\n--- Per-spec deduplication ---")
    all_s = per_spec_dedup(all_s, "S-Gen")
    all_q = per_spec_dedup(all_q, "Q-Gen")

    # 6. Cross-dedup against existing SFT
    print("\n--- Cross-dedup against existing SFT ---")
    all_s = cross_dedup(all_s, "S-Gen", existing_sigs)
    all_q = cross_dedup(all_q, "Q-Gen", existing_sigs)

    # 7. Final intra-set dedup (S vs Q overlap)
    if all_q and all_s:
        q_sigs_set = [e["_sig"] for e in all_q]
        print(f"\n  Removing S-Gen entries that overlap with Q-Gen...")
        s_before = len(all_s)
        all_s_clean = []
        for e in all_s:
            overlap = any(minhash_jaccard(e["_sig"], qs) > DEDUP_SIM_THRESHOLD for qs in q_sigs_set)
            if not overlap:
                all_s_clean.append(e)
        print(f"  S-Gen: {s_before} → {len(all_s_clean)} after Q-overlap removal")
        all_s = all_s_clean

    # 8. Similarity stats
    print("\n--- Final similarity stats ---")
    for label, entries in [("Q-Gen", all_q), ("S-Gen", all_s)]:
        if len(entries) < 2:
            continue
        sample = entries[:min(200, len(entries))]
        sims = []
        for i, e in enumerate(sample):
            best = max(
                (minhash_jaccard(e["_sig"], e2["_sig"])
                 for j, e2 in enumerate(entries) if j != i),
                default=0
            )
            sims.append(best)
        print(f"  {label} ({len(entries)} entries): "
              f"mean_max_sim={sum(sims)/len(sims):.3f}, "
              f"max={max(sims):.3f}")

    # 9. Step distribution of kept entries
    print("\n--- Step distribution of final entries ---")
    for label, entries in [("Q-Gen", all_q), ("S-Gen", all_s)]:
        if not entries:
            continue
        step_counts = defaultdict(int)
        for e in entries:
            bucket = (e["step"] // 50) * 50
            step_counts[bucket] += 1
        buckets = sorted(step_counts.items())
        print(f"  {label}: " + ", ".join(f"[{b}-{b+49}]:{c}" for b, c in buckets))

    # 10. Format and merge
    print("\n--- Formatting SFT rows ---")
    new_rows = []
    for e in all_q:
        new_rows.append(format_sft_row(e, "rl_v4v6_qgen_sft"))
    for e in all_s:
        new_rows.append(format_sft_row(e, "rl_v4v6_rollout_sft"))

    print(f"  New Q-Gen rows: {len(all_q)}")
    print(f"  New S-Gen rows: {len(all_s)}")

    df_new = pd.DataFrame(new_rows)
    df_merged = pd.concat([df_existing, df_new], ignore_index=True)

    print(f"\n--- Final Dataset ---")
    print(f"  Existing: {len(df_existing)}")
    print(f"  New:      {len(df_new)}")
    print(f"  Merged:   {len(df_merged)}")
    print(f"\nDistribution:")
    print(df_merged["data_source"].value_counts().to_string())

    # 11. Save
    output_path = os.path.join(str(PROJECT_ROOT), OUTPUT_SFT)
    df_merged.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # 12. Per-run breakdown
    print("\n--- Per-run breakdown ---")
    for run_name in ROLLOUT_DIRS:
        q_from = sum(1 for e in all_q if e["run"] == run_name)
        s_from = sum(1 for e in all_s if e["run"] == run_name)
        print(f"  {run_name}: {q_from} Q-Gen + {s_from} S-Gen = {q_from + s_from} total")

    print("\nDone!")


if __name__ == "__main__":
    main()
