# Code of the reported run

This directory contains the code that produced the run reported in the paper, **as it was run**. It is released for inspection and as a record. It depends on a customized [verl](https://github.com/volcengine/verl) installation and on our Apptainer container and cluster layout, so it is not expected to run unchanged elsewhere. Trained checkpoints are not available.

The earliest runs predate systematic version control, and the code changed between runs; the audit scripts below check the run's logged behaviour.

## Contents

| Path | What it is |
|---|---|
| `trainer/qs_trainer-Qwen2.5-Coder-3B-QGSG-SFTinit-RL-rootonly-qscale0375-band1of8-strictpool-nofreeze-v1.py` | Trainer snapshot of the reported run (5,702 lines; `QuestionSolverTrainer`). |
| `trainer/qs_trainer_v45_rootonly_resume.py` | Wrapper that loads the snapshot above; used for the resumed segment (steps 2,000–3,200). |
| `eval/qs_trainer_rebuttal_1shot.py` | Evaluation-only wrapper for the PSV-format 1-shot evaluation (last row of the main results table). |
| `eval/qs_trainer_paper_version.py` | Later trainer version that the 1-shot wrapper loads for its validation code. |
| `launch/train_steps_0000-2000.command.txt` | Command line of steps 0–2,000, copied from the run's log. |
| `launch/train_steps_2000-3200.command.txt` | Command line of the resumed segment, copied from its log. |
| `launch/resume_steps_2000-3200.sh` | Launcher of the resumed segment. |
| `launch/eval_0shot_n100.sh` | Launcher of the n = 100, 0-shot evaluations. |
| `launch/eval_psv_1shot_n100.sh` | Launcher of the PSV-format 1-shot evaluation. |
| `sft/generate_sgen_sft_v2.py` | Generates teacher reasoning for warm-up Solver demonstrations through an API (keys are read from environment variables). |
| `sft/extract_rl_sft_data.py` | Extracts verifier-accepted rollouts from early RL runs for the warm-up stages. |
| `audits/rollout_audits.py` | Log-based audits used in the paper: seed sampling (curriculum descent and crossover), reward values and verifier-bypass constructs, reference-answer exposure, and branch weights. |

The training data of the warm-up stages is not included.

## How the files relate to the reported run

- **Steps 0–2,000.** The launch command is recorded in the log; the trainer file used at launch was not recorded separately. The run's logged behaviour matches the snapshot in `trainer/`: every step prints `seed_sampling_mode=root_only` (which the snapshot maps to a root quota of 1), all 204,800 logged proposals are conditioned on curated roots, and the logged Proposer branch scale varies from step to step as the snapshot's ℓ2-balancing computes it.
- **Steps 2,000–3,200.** The resumed segment sets `trainer_impl=qsv.qs_trainer_v45_rootonly_resume`, which loads the snapshot explicitly. The reported checkpoint lies in this segment.
- **Evaluation.** The n = 100 evaluations were run with later versions of the trainer code (the 0-shot evaluation with the mainline trainer of 2026-04-26, the 1-shot evaluation with `eval/qs_trainer_paper_version.py`). These versions apply the same Solver output rules as the snapshot: the six rejection patterns for repeated `requires`/`ensures`, `verus!` wrappers, `fn main`, `use` statements, and placeholder implementations.
- **Settings that are configured but inactive.** The command line enables a Proposer-weight warm-up (`q_branch_warmup`) and sets `ucb_root_ratio=0.25`. The snapshot ignores both: the logged warm-up scale is 1.0 at every step, and `root_only` sampling overrides the root ratio.

## Reproducing the audits

```bash
python audits/rollout_audits.py seeds    --rollouts rollout_logs.tar.gz --graph seed_pool_graph.json
python audits/rollout_audits.py hacking  --rollouts rollout_logs.tar.gz
python audits/rollout_audits.py exposure --rollouts rollout_logs.tar.gz --pool data/seed_pool_ttt.jsonl
python audits/rollout_audits.py weights  --log training.log
```

On the reported run, these give: 204,800 of 204,800 proposals conditioned on curated roots and no crossover prompts; Solver rewards in {0, 1} with 586,188 verified rollouts, one of which contains `assume(`; Proposer rewards in {0, 1}, with 950 frontier events; no prompt containing a full reference implementation; and 527 distinct Proposer branch scales over the first 2,020 steps, with a warm-up scale of 1.0 throughout. The run logs themselves are not part of this repository.

## Edits made for release

The only change to any file is in the three launchers, where the cluster user name was replaced by `${USER}`. SHA-256 of the unmodified Python files:

```text
1940bf9213f9fb90cb64b2629ba024952a6ad37f39071a81181541d11db69880  trainer/qs_trainer-...-nofreeze-v1.py
22c6aaf3f98aa868b8d20a206373c9338efe5dddfd15ce3980e0d7e3477d80c2  trainer/qs_trainer_v45_rootonly_resume.py
fc9f30c7249ebcfae1b80cfa094599cc40bbd83a688d802e30f4490a62ba3ce7  eval/qs_trainer_paper_version.py
e36d8144df94c4f3243f5576e337d4023dd04a0cb4bbb76c375c68be2adafac8  eval/qs_trainer_rebuttal_1shot.py
1d7c21b4c78a2df4ec98dec22bc5f0b5075617d1966578a8346dae299fbae5a8  sft/extract_rl_sft_data.py
addb3773ee1f62ce0485a554356dc529c52c2a7955826c9417f3d3258bd217bc  sft/generate_sgen_sft_v2.py
```
