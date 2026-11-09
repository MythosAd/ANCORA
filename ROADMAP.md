# Release Roadmap

## Released

| Asset | Path |
|---|---|
| Seed pools (436 roots of the reported run; 234 Dafny2Verus roots) | `data/` |
| Code of the reported run, as run (trainer snapshot, launch commands, evaluation wrappers, warm-up data scripts) | `code/` |
| Log-based audit scripts used in the paper | `code/audits/` |
| Warm-up data verification script | `eval/verify_verus.py` |
| Figures | `figs/` |

The code depends on a customized [verl](https://github.com/volcengine/verl) installation and on our Apptainer container setup, so it is not expected to run unchanged elsewhere; see [`code/README.md`](code/README.md).

## Not included

- Warm-up (SFT) training data.
- Trained checkpoints (not available).
