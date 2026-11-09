# Warm-up Data Verification Script

`verify_verus.py` is the script used to verify API-generated Solver demonstrations for the warm-up (SFT) data. It is **not** the scorer behind the training rewards or the pass@k numbers reported in the paper.

| | `verify_verus.py` (this directory) | Training rewards and reported pass@k |
|---|---|---|
| Purpose | Filter teacher-generated warm-up data | Score Solver rollouts during RL and evaluation |
| Auto-repair | Yes: strips a redundant leading `{`, appends or strips braces, normalizes inline invariants | No |
| Output | Verified / not verified per entry, plus the repair applied | Binary verifier outcome per attempt |

The scorer used for training and evaluation is part of the trainer (see [`../code/README.md`](../code/README.md)). The Solver's body is appended to the fixed specification; responses that repeat `requires`/`ensures`, contain a `verus!` wrapper, `fn main`, or `use` statements, or give a placeholder implementation are rejected; and the reward is 1 only if Verus verifies the result.

An earlier version of this file described the script as the evaluation harness and as matching a partial-credit training score. Both statements were incorrect: training rewards in the reported run are binary.

## Container assumption

The script calls Verus inside a local Docker container named `verl_eval` that has:

- the `verus` binary at `/workspace/verl/verus/verus-x86-linux/verus`
- a working `cargo` toolchain at `/root/.cargo/bin`

Edit `CONTAINER` and `VERUS_BIN` at the top of the script to point to your own Verus install.

## Usage

```bash
python verify_verus.py --input <unverified.jsonl> --output <verified.jsonl> --workers 8 --timeout 30
```

Each input line is a JSON object with fields `question` (the specification), `answer_api` (the candidate body), `idx`, and `solution_idx`. Entries already listed in the file given by `--skip_verified` are skipped. See the source for the repair rules.
