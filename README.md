# ANCORA

ANCORA is a self-play reinforcement-learning method in which one language-model policy both **proposes** formal specifications and **solves** them, with the [Verus](https://verus-lang.github.io/verus/) verifier providing the reward.

**Paper:** *ANCORA: Learning to Question via Manifold-Anchored Self-Play for Verifiable Reasoning*, [arXiv:2604.27644](https://arxiv.org/abs/2604.27644).

---

## Method in brief

- **Gate, do not grade.** Structural checks only accept or reject outputs; every learning reward is a binary verifier outcome.
- **One policy, two roles.** As *Proposer* (Q-Gen), the policy writes a one-step extension of a seed specification. As *Solver* (S-Gen), it writes only the function body, which is appended to the fixed specification and checked by Verus. Responses that repeat `requires`/`ensures`, a `verus!` wrapper, `fn main`, or `use` statements are rejected, so the Solver cannot change the specification.
- **Rewards.** Solver: 1 if the implementation verifies, else 0. Proposer: 1 if exactly one of K = 8 Solver attempts verifies the proposal (Band-1-of-K), else 0.
- **Update.** Two-level group-relative advantages with mean normalization: across the N = 8 proposals of a seed, and across the K = 8 attempts on a proposal.
- **Warm-up.** Three stages of self-distilled SFT on verifier-accepted rollouts (Dafny2Verus pass@1: 0.4% for the base model, 26.6% after warm-up).
- **UCB-guided curriculum DAG.** Seeds are selected by UCB; novel, Solver-verified proposals are admitted to the seed pool. In the reported run, proposals were conditioned on the curated roots.

## Main result

Test-time-training setting, Qwen2.5-Coder-3B-Instruct, n = 100 samples per task, single training run. Pass@1 (%):

| | Dafny2Verus | MBPP-Verus | HumanEval-Verus |
|---|---:|---:|---:|
| PSV, 1-shot (as reported by PSV) | 65.63 | 36.78 | 19.07 |
| ANCORA, 0-shot | 81.46 | 44.11 | 19.15 |
| ANCORA, PSV-format 1-shot | 74.28 | 41.15 | 18.67 |

Pass@5/10, the transfer setting, and task-level intervals are in Table 2 and Section 5 of the paper. The HumanEval-Verus margins over PSV are inconclusive.

## What's in this repository

| Path | Contents |
|---|---|
| `data/seed_pool_ttt.jsonl` | The 436 curated roots of the reported run; composition in [`data/README.md`](data/README.md). |
| `data/seed_pool_transfer.jsonl` | The 234 filtered Dafny2Verus specifications used as roots by the D2V-only runs. |
| `eval/verify_verus.py` | The script used to verify API-generated warm-up data. It applies brace auto-repair and is **not** the scorer behind the training rewards or the reported pass@k; see [`eval/README.md`](eval/README.md). |
| `code/` | The code of the reported run as it was run: trainer snapshot, launch commands, evaluation wrappers, warm-up data scripts, and the log-based audit scripts used in the paper; see [`code/README.md`](code/README.md). |
| `figs/` | Figures from the paper. |

The code is released for inspection; it depends on a customized verl installation and our container setup and is not expected to run unchanged. Trained checkpoints and the warm-up training data are not included.

## Citation

```bibtex
@article{yang2026ancora,
  title   = {{ANCORA}: Learning to Question via Manifold-Anchored Self-Play for Verifiable Reasoning},
  author  = {Yang, Chengcao},
  journal = {arXiv preprint arXiv:2604.27644},
  year    = {2026}
}
```

## License

Code: Apache-2.0 (see [`LICENSE`](LICENSE)).
The seed pools are derived from the Verus benchmark splits released with PSV (Dafny2Verus, MBPP-Verus, HumanEval-Verus) and a small set of elementary specifications; please also respect the licenses of the original benchmark sets.

## Acknowledgements

Built on [verl](https://github.com/volcengine/verl) for distributed RL infrastructure and [Verus](https://verus-lang.github.io/verus/) for formal verification.
