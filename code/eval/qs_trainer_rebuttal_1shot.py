"""Evaluation-only trainer for the NeurIPS rebuttal matched 1-shot comparison.

The training/evaluation base is the archived paper snapshot.  The only behavioral
change is validation prompting: every rollout receives one independently sampled
PSV-format Dafny2Verus gold exemplar, with the current D2V task excluded by stable
index.  This module must not be used for training.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np

from qsv.psv_one_shot import (
    build_psv_one_shot_messages,
    file_sha256,
    load_psv_spec_completion_pairs,
)


_PAPER_TRAINER_PATH = (
    Path(__file__).resolve().parent
    / "archive"
    / "trainers"
    / "qs_trainer_paper_version.py"
)
_PAPER_MODULE_NAME = "qsv._archived_paper_trainer_for_rebuttal_1shot"
_SPEC = importlib.util.spec_from_file_location(_PAPER_MODULE_NAME, _PAPER_TRAINER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load archived paper trainer: {_PAPER_TRAINER_PATH}")
_PAPER = importlib.util.module_from_spec(_SPEC)
sys.modules[_PAPER_MODULE_NAME] = _PAPER
_SPEC.loader.exec_module(_PAPER)


class QuestionSolverTrainer(_PAPER.QuestionSolverTrainer):
    """Paper-snapshot trainer with an exact PSV-format 1-shot validation path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        trainer_cfg = self.config.trainer

        source_path = str(trainer_cfg.get("psv_fewshot_data", "")).strip()
        if not source_path:
            raise ValueError("trainer.psv_fewshot_data is required")
        self.psv_fewshot_pairs, self.psv_fewshot_sha256 = (
            load_psv_spec_completion_pairs(source_path)
        )
        expected_pool_size = int(
            trainer_cfg.get("psv_fewshot_expected_pool_size", 241)
        )
        if len(self.psv_fewshot_pairs) != expected_pool_size:
            raise ValueError(
                "Unexpected PSV exemplar pool size: "
                f"{len(self.psv_fewshot_pairs)} != {expected_pool_size}"
            )

        self.psv_fewshot_seed = int(trainer_cfg.get("psv_fewshot_seed", 42))
        self.eval_temperature = float(trainer_cfg.get("eval_temperature", 0.8))
        self.eval_top_p = float(
            self.config.actor_rollout_ref.rollout.get("top_p", 1.0)
        )
        self.eval_dump_full_traces = bool(
            trainer_cfg.get("eval_dump_full_traces", False)
        )

        requested = trainer_cfg.get("eval_dataset_names", None)
        if requested is not None:
            requested_names = [str(name) for name in list(requested)]
            missing = [name for name in requested_names if name not in self.val_datasets]
            if missing:
                raise ValueError(
                    f"Requested validation datasets are unavailable: {missing}; "
                    f"loaded={list(self.val_datasets)}"
                )
            self.val_datasets = {
                name: self.val_datasets[name] for name in requested_names
            }

        print(
            "[PSV-1SHOT] protocol=exact_psv_messages "
            "assistant_prefix=none exemplar_sampling=independent_per_rollout "
            f"seed_scheme='<seed>:<dataset>' seed={self.psv_fewshot_seed} "
            f"pool_size={len(self.psv_fewshot_pairs)} "
            f"pool_sha256={self.psv_fewshot_sha256} "
            f"paper_trainer_sha256={file_sha256(_PAPER_TRAINER_PATH)} "
            f"temperature={self.eval_temperature} top_p={self.eval_top_p} "
            f"datasets={list(self.val_datasets)}",
            flush=True,
        )

    def _dump_protocol_audit(self, dataset_name, rows):
        """Persist a small raw prompt/output sample without retaining every rollout."""

        if not rows:
            return
        output_path = (
            Path(str(self.config.trainer.default_local_dir))
            / "protocol_audit.jsonl"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                record = {
                    "step": int(getattr(self, "global_steps", 0)),
                    "dataset": dataset_name,
                    "prompt_protocol": "psv_exact_1shot",
                    "fewshot_pool_sha256": self.psv_fewshot_sha256,
                    "fewshot_seed": self.psv_fewshot_seed,
                    **row,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _validate(self, dataset_name, dataset_df, n_samples=10, pass_at_ks=None):
        """Run n-sample verifier evaluation with a fresh PSV exemplar per rollout."""

        if pass_at_ks is None:
            pass_at_ks = [1]
        print(
            f"Running PSV-matched 1-shot validation for {dataset_name} "
            f"({len(dataset_df)} samples, n={n_samples}, pass@k={pass_at_ks})...",
            flush=True,
        )
        metrics, problem_results = {}, []
        raw_val_batch_size = self.config.data.get("val_batch_size", 16)
        if raw_val_batch_size is None:
            raw_val_batch_size = 16
        val_batch_size = (
            max(1, raw_val_batch_size // n_samples)
            if n_samples > 1
            else raw_val_batch_size
        )
        num_batches = math.ceil(len(dataset_df) / val_batch_size)
        rng = random.Random(f"{self.psv_fewshot_seed}:{dataset_name}")

        def process_verification_sample(args):
            prompt_text, full_text = args
            stitched_result = _PAPER.smart_stitch(prompt_text, full_text)
            stitched_code = (
                stitched_result[0]
                if isinstance(stitched_result, tuple)
                else stitched_result
            )
            verified, verifier_stdout, verifier_stderr = False, "", ""
            if stitched_code:
                if "use builtin" in stitched_code:
                    stitched_code = stitched_code.replace(
                        "use builtin::*;", "// use builtin::*;"
                    ).replace(
                        "use builtin_macros::*;", "// use builtin_macros::*;"
                    )
                if not re.search(r"fn\s+main", stitched_code):
                    stitched_code += "\nfn main() {}\n"
                verified, verifier_stdout, verifier_stderr = _PAPER.verify_code(
                    stitched_code, timeout=30
                )
            return verified, stitched_code, verifier_stdout, verifier_stderr

        for batch_index in range(num_batches):
            chunk = dataset_df.iloc[
                batch_index * val_batch_size : (batch_index + 1) * val_batch_size
            ]
            raw_prompts = chunk["question"].tolist()
            processed_input_ids = []
            working_raw_prompts = []
            working_messages = []
            exemplar_records = []

            for (_, row), raw_prompt in zip(chunk.iterrows(), raw_prompts):
                raw_prompt_text = str(raw_prompt)
                spec_code = _PAPER.extract_code_from_text(raw_prompt_text)
                if not spec_code.strip():
                    spec_code = raw_prompt_text.strip()

                exclude_pool_index = None
                if dataset_name == "dafny2verus" and "idx" in row:
                    exclude_pool_index = int(row["idx"])

                for _sample_index in range(n_samples):
                    messages, exemplar = build_psv_one_shot_messages(
                        spec_code,
                        self.psv_fewshot_pairs,
                        rng,
                        exclude_pool_index=exclude_pool_index,
                    )
                    token_ids = _PAPER.to_list_of_ints(
                        self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            continue_final_message=False,
                            add_generation_prompt=True,
                            add_special_tokens=False,
                        )
                    )
                    processed_input_ids.append(token_ids)
                    working_raw_prompts.append(raw_prompt)
                    working_messages.append(messages)
                    exemplar_records.append(
                        {
                            "pool_index": exemplar.pool_index,
                            "task_id": exemplar.task_id,
                            "excluded_pool_index": exclude_pool_index,
                        }
                    )

            n_filtered = len(raw_prompts)
            generation_output, decoded_outputs, _val_timing = self._rollout_generate(
                input_ids_list=processed_input_ids,
                raw_prompts=working_messages,
                meta_info={"temperature": self.eval_temperature},
                extra_non_tensor={
                    "uid": np.arange(len(processed_input_ids)),
                    "original_raw_prompt": _PAPER._to_object_array(
                        working_raw_prompts
                    ),
                },
            )

            tasks = list(zip(working_raw_prompts, decoded_outputs))
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(32, len(tasks))
            ) as executor:
                results = list(executor.map(process_verification_sample, tasks))

            response_ids = generation_output.batch.get(
                "responses", generation_output.batch["input_ids"]
            )
            extra_rows = []
            summary_rows = []
            audit_rows = []

            for problem_index in range(n_filtered):
                start = problem_index * n_samples
                stop = start + n_samples
                question_results = results[start:stop]
                success_count = sum(
                    1 for verified, _, _, _ in question_results if verified
                )
                global_question_index = (
                    batch_index * val_batch_size + problem_index
                )
                question_text = str(raw_prompts[problem_index])
                question_exemplars = exemplar_records[start:stop]
                excluded_index = question_exemplars[0]["excluded_pool_index"]
                summary_rows.append(
                    {
                        "question_idx": int(global_question_index),
                        "success_count": int(success_count),
                        "total_count": int(n_samples),
                        "success_rate": float(success_count / max(n_samples, 1)),
                        "verified_any": bool(success_count > 0),
                        "verified_all": bool(success_count == n_samples),
                        "question_text": question_text,
                        "prompt_protocol": "psv_exact_1shot",
                        "fewshot_pool_size": len(self.psv_fewshot_pairs),
                        "fewshot_pool_sha256": self.psv_fewshot_sha256,
                        "fewshot_seed": self.psv_fewshot_seed,
                        "excluded_self_pool_index": excluded_index,
                        "distinct_exemplar_count": len(
                            {
                                item["pool_index"]
                                for item in question_exemplars
                            }
                        ),
                        "eval_temperature": self.eval_temperature,
                        "eval_top_p": self.eval_top_p,
                    }
                )
                for sample_index, (
                    verified,
                    stitched_code,
                    verifier_stdout,
                    verifier_stderr,
                ) in enumerate(question_results):
                    exemplar_record = question_exemplars[sample_index]
                    extra_rows.append(
                        {
                            "question_idx": int(global_question_index),
                            "sample_idx_within_question": int(sample_index),
                            "success_count_for_question": int(success_count),
                            "total_count_for_question": int(n_samples),
                            "success_rate_for_question": float(
                                success_count / max(n_samples, 1)
                            ),
                            "verified_any_for_question": bool(success_count > 0),
                            "verified_all_for_question": bool(
                                success_count == n_samples
                            ),
                            "verify_success": bool(verified),
                            "question_text": question_text,
                            "stitched_code": stitched_code,
                            "verify_stdout": verifier_stdout,
                            "verify_stderr": verifier_stderr,
                            "fewshot_exemplar_pool_index": exemplar_record[
                                "pool_index"
                            ],
                            "fewshot_exemplar_task_id": exemplar_record["task_id"],
                            "excluded_self_pool_index": exemplar_record[
                                "excluded_pool_index"
                            ],
                        }
                    )
                problem_results.append((n_samples, success_count))

                if batch_index == 0 and problem_index < 3:
                    sample_offset = start
                    audit_result = results[sample_offset]
                    audit_rows.append(
                        {
                            "question_idx": int(global_question_index),
                            "sample_idx_within_question": 0,
                            "fewshot_exemplar_pool_index": exemplar_records[
                                sample_offset
                            ]["pool_index"],
                            "fewshot_exemplar_task_id": exemplar_records[
                                sample_offset
                            ]["task_id"],
                            "excluded_self_pool_index": exemplar_records[
                                sample_offset
                            ]["excluded_pool_index"],
                            "raw_input_decoded": self.tokenizer.decode(
                                processed_input_ids[sample_offset],
                                skip_special_tokens=True,
                            ),
                            "raw_output_decoded": decoded_outputs[sample_offset],
                            "verify_success": bool(audit_result[0]),
                            "stitched_code": audit_result[1],
                            "verify_stdout": audit_result[2],
                            "verify_stderr": audit_result[3],
                        }
                    )

            if self.eval_dump_full_traces:
                self._dump_token_debug(
                    getattr(self, "global_steps", 0),
                    "validate",
                    processed_input_ids,
                    response_ids,
                    dataset_name=dataset_name,
                    extra_rows=extra_rows,
                    input_texts=self.tokenizer.batch_decode(
                        processed_input_ids, skip_special_tokens=True
                    ),
                    output_texts=decoded_outputs,
                )
            self._dump_protocol_audit(dataset_name, audit_rows)
            self._dump_validation_summary(
                getattr(self, "global_steps", 0),
                dataset_name,
                summary_rows,
            )

        valid_results = [(n, c) for n, c in problem_results if n > 0]
        for k in pass_at_ks:
            eligible = [(n, c) for n, c in valid_results if n >= k]
            if eligible:
                metrics[f"val/{dataset_name}/pass@{k}"] = float(
                    np.mean(
                        [self._pass_at_k(n, c, k) for n, c in eligible]
                    )
                )
        return metrics
