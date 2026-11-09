"""Log-based audits used in the ANCORA paper (Sections 4.5, 5.6, 5.7).

All audits read a run's own logs, not its configuration or code:

  seeds      fraction of Proposer prompts whose seed is an admitted (non-root)
             node of the final seed-pool graph, i.e. measured curriculum descent;
             also counts dual-parent ("Seed A / Seed B") crossover prompts
  hacking    reward values of Solver rollouts, and verified Solver code blocks
             that contain assume(, admit(, or #[verifier ...] attributes
  exposure   Proposer/Solver prompts that contain a reference implementation
             from a seed pool
  weights    per-step Proposer branch scale and warm-up scale from a training log

Inputs: rollout_logs.tar.gz (one rollout_step_<N>.jsonl per step, records with
keys step, prompt, response, is_question, question_score, solve_score),
seed_pool_graph.json (keys edges/visit_counts/...), a seed-pool JSONL with
question/answer fields, and a training log with "step:N - key:value ..." lines.

Example:
  python rollout_audits.py seeds    --rollouts rollout_logs.tar.gz --graph seed_pool_graph.json
  python rollout_audits.py hacking  --rollouts rollout_logs.tar.gz
  python rollout_audits.py exposure --rollouts rollout_logs.tar.gz --pool ../../data/seed_pool_ttt.jsonl
  python rollout_audits.py weights  --log training.log
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import tarfile

RUST = re.compile(r"```rust\s*\n(.*?)```", re.S)
STEP_FILE = re.compile(r"rollout_step_(\d+)\.jsonl$")
CROSSOVER = re.compile(r"two seed problems|Seed A:", re.I)
BYPASS = {
    "assume": re.compile(r"assume\s*\("),
    "admit": re.compile(r"admit\s*\("),
    "verifier_attr": re.compile(r"#\s*\[\s*verifier"),
}


def prompt_text(p) -> str:
    if isinstance(p, str):
        return p
    if isinstance(p, list):
        return "\n".join(prompt_text(x.get("content", "")) if isinstance(x, dict) else prompt_text(x) for x in p)
    return str(p)


def ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def records(path: str, every: int):
    """Yield rollout records from steps divisible by `every`."""
    with tarfile.open(path) as tf:
        for m in tf:
            mm = STEP_FILE.search(m.name)
            if not m.isfile() or not mm or int(mm.group(1)) % every:
                continue
            for line in tf.extractfile(m):
                yield json.loads(line)


def audit_seeds(a):
    edges = json.load(open(a.graph, encoding="utf-8"))["edges"]
    roots = {ws(q) for q, e in edges.items() if not e.get("parents")}
    admitted = {ws(q) for q, e in edges.items() if e.get("parents")}
    c = collections.Counter()
    for r in records(a.rollouts, a.every):
        if not r.get("is_question"):
            continue
        p = prompt_text(r.get("prompt"))
        blocks = [ws(b) for b in RUST.findall(p)]
        if CROSSOVER.search(p):
            c["crossover_prompts"] += 1
        if any(b in admitted for b in blocks):
            c["admitted_seed"] += 1
        elif any(b in roots for b in blocks):
            c["root_seed"] += 1
        else:
            # Seeds pruned before the graph was saved can only be admitted nodes.
            c["unmatched_seed"] += 1
    total = c["admitted_seed"] + c["root_seed"] + c["unmatched_seed"]
    print(dict(c), f"roots={len(roots)} admitted_nodes={len(admitted)}")
    if total:
        print(f"descent fraction = {(c['admitted_seed'] + c['unmatched_seed']) / total:.4f}")


def audit_hacking(a):
    solve, question, hits = collections.Counter(), collections.Counter(), collections.Counter()
    verified = 0
    for r in records(a.rollouts, a.every):
        if r.get("question_score") is not None:
            question[round(float(r["question_score"]), 3)] += 1
        if r.get("solve_score") is None:
            continue
        s = float(r["solve_score"])
        solve[round(s, 3)] += 1
        if s <= 0:
            continue
        verified += 1
        resp = r.get("response") or ""
        code = "\n".join(RUST.findall(resp)) or resp
        for name, pat in BYPASS.items():
            if pat.search(code):
                hits[name] += 1
    print("solve_score values:", sorted(solve.items()))
    print("question_score values:", sorted(question.items()))
    print(f"verified Solver rollouts: {verified}; containing:", dict(hits))


def audit_exposure(a):
    def core(s):
        s = s.replace("fn main() {}", "").replace("} // verus!", "")
        return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", s))

    pool = [json.loads(l) for l in open(a.pool, encoding="utf-8") if l.strip()]
    answers = [core(x["answer"]) for x in pool]
    answers = [x for x in answers if len(x) >= a.min_chars]
    c = collections.Counter()
    for r in records(a.rollouts, a.every):
        role = "proposer" if r.get("is_question") else "solver"
        c[role] += 1
        p = core(prompt_text(r.get("prompt")))
        if any(x in p for x in answers):
            c[role + "_contains_full_reference"] += 1
    print(dict(c), f"reference answers checked: {len(answers)}")


def audit_weights(a):
    keys = ("adv/q_loss_scale", "adv/q_warmup_scale")
    vals = {k: [] for k in keys}
    modes = collections.Counter()
    for line in open(a.log, encoding="utf-8", errors="replace"):
        for m in re.findall(r"seed_sampling_mode=([a-z_]+)", line):
            modes[m] += 1
        if not re.search(r"step:\d+ - ", line):
            continue
        for k in keys:
            m = re.search(re.escape(k) + r":([-0-9.e]+)", line)
            if m:
                vals[k].append(float(m.group(1)))
    print("seed_sampling_mode prints:", dict(modes))
    for k, v in vals.items():
        if v:
            v_sorted = sorted(v)
            print(f"{k}: steps={len(v)} distinct={len(set(round(x, 4) for x in v))} "
                  f"min={v_sorted[0]:.4f} median={v_sorted[len(v) // 2]:.4f} max={v_sorted[-1]:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("seeds", "hacking", "exposure"):
        p = sub.add_parser(name)
        p.add_argument("--rollouts", required=True)
        p.add_argument("--every", type=int, default=1, help="only read steps divisible by this number")
        if name == "seeds":
            p.add_argument("--graph", required=True)
        if name == "exposure":
            p.add_argument("--pool", required=True)
            p.add_argument("--min-chars", type=int, default=60)
    p = sub.add_parser("weights")
    p.add_argument("--log", required=True)
    a = ap.parse_args()
    {"seeds": audit_seeds, "hacking": audit_hacking, "exposure": audit_exposure, "weights": audit_weights}[a.cmd](a)


if __name__ == "__main__":
    main()
