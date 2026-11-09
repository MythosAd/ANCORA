# Seed Pools

| File | Count | Role |
|---|---|---|
| `seed_pool_ttt.jsonl` | 436 | Curated roots of the reported run (test-time-training setting). |
| `seed_pool_transfer.jsonl` | 234 | Filtered Dafny2Verus specifications, used as roots by the D2V-only runs (e.g., the curriculum-descent and Proposer-reward runs in the paper). |

Each file is a JSON-Lines stream with one seed record per line.

## Composition

Matching specification text after removing comments, `use` lines and `verus!` wrappers:

- `seed_pool_ttt.jsonl` contains all 241 Dafny2Verus tasks, 73 of the 74 MBPP-Verus tasks, 56 of the 73 HumanEval-Verus tasks, and about 70 elementary specifications (e.g., `signum`, `clamp`, `safe_abs`) that are not evaluation tasks. The HumanEval-Verus tasks that are absent were never used as seeds in the reported run.
- `seed_pool_transfer.jsonl` consists of 234 of the 241 Dafny2Verus tasks.

The evaluation splits are those released with PSV.

## How the pools are used

These are **seeds, not held-out sets.** During RL, a seed's specification (without its reference implementation) conditions the Proposer, and the Solver solves the proposals derived from it. Evaluating on tasks that are also seeds is therefore test-time training. In the retained rollouts of the reported run, no Solver prompt contains a reference implementation.

In the reported run, novel verified proposals were admitted to the pool during training (1,737 nodes by step 3,199) but, because the root quota was ρ = 1, they were never used as seeds.

An earlier version of this file described `seed_pool_transfer.jsonl` as a held-out set that the model never trains on. That was incorrect: in the runs that use it, its specifications are the seeds.

## Schema

```jsonc
{
  "generation_idx": 0,              // origin tag (0 = original/curated)
  "question": "fn fast_add(...)",   // Verus specification: signature, requires/ensures, open body brace
  "answer":   "...",                // a reference body that verifies; not shown to the model during RL
  "is_original": true,              // always true in the released pools
  "difficulty": 1.0                 // optional; only present in seed_pool_ttt.jsonl
}
```

## Filtering applied

Both files are post-filter snapshots of what the trainer loads at the start of training:

1. `len(question) <= 2000`: drop overlong specifications
2. `len(answer) <= 3000`: drop overlong implementations
3. `len(answer) >= 5`: drop trivially short or placeholder implementations
4. Deduplicate on `question` text

Raw counts before filtering: 241 to 234 for `seed_pool_transfer.jsonl`; 457 to 436 for `seed_pool_ttt.jsonl`.

## Reproducing the filter

```python
import json

seen = set()
out = []
with open("seed_pool_transfer.jsonl") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        q, a = item["question"].strip(), item["answer"].strip()
        if len(q) > 2000 or len(a) > 3000 or len(a) < 5:
            continue
        if q in seen:
            continue
        seen.add(q)
        out.append(item)
print(len(out))  # 234 / 436
```

## License

Same as the parent repository (Apache-2.0) for the files added here. The specifications and reference implementations derive from publicly released benchmark sets; please also respect their original licenses.
