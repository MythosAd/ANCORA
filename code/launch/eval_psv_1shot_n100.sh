#!/bin/bash
#SBATCH --job-name=ancora-rb-1shot
#SBATCH --account=chenjun3
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# NeurIPS rebuttal, evaluation only:
#   sbatch qsv/run_apptainer_rebuttal_matched_1shot.sh r1
#
# The job loads the retained legacy step-2400 actor and evaluates D2V/MBPP/HumanEval
# with PSV's exact 1-shot message format, independently sampling one D2V gold exemplar
# per rollout.  It writes into a separate eval directory so historical 0-shot raw
# summaries remain untouched.

set -euo pipefail
module purge
module load google/go
module load apptainer/1.4.5
module load nvidia/cuda/12.9

cd "$HOME/project/GSV"
mkdir -p slurm_logs ray_tmp

RUN_TAG="${1:-r1}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[FATAL] Unsafe RUN_TAG: $RUN_TAG"
    exit 2
fi

VER="rebuttal1shot"
EVAL_STEP=2400
N_SAMPLES=100
VAL_BATCH=400
FEWSHOT_SEED=42
EXPECTED_POOL_SIZE=241
EXPECTED_PSV_SHA256="171F56746977CA97563CD6E3F1614375CF85E6E463EC7101231010210E20E853"
EXPECTED_PAPER_TRAINER_SHA256="FC9F30C7249EBCFAE1B80CFA094599CC40BBD83A688D802E30F4490A62BA3CE7"

HOST_CKPT_ROOT="/scratch/${USER}/checkpoints"
RUN_NAME="Qwen2.5-Coder-3B-QGSG-SFTinit-RL-rootonly-qscale0375-band1of8-strictpool-nofreeze-v1"
SOURCE_CKPT="${HOST_CKPT_ROOT}/${RUN_NAME}"
TARGET_STEP_DIR="${SOURCE_CKPT}/global_step_${EVAL_STEP}"
TARGET_ACTOR="${TARGET_STEP_DIR}/actor"
SFT_INIT="${HOST_CKPT_ROOT}/Qwen2.5-Coder-3B-QGSG-SFT-v7-merged/global_step_1038_merged"
PSV_POOL="$HOME/project/GSV/psv/data/dafny2verus.jsonl"
PAPER_TRAINER="$HOME/project/GSV/qsv/archive/trainers/qs_trainer_paper_version.py"
EVAL_NAME="ancora-rebuttal-v45-step${EVAL_STEP}-psv1shot-seed${FEWSHOT_SEED}-${RUN_TAG}"
EVAL_DIR="${HOST_CKPT_ROOT}/${EVAL_NAME}"

for required_dir in "$TARGET_ACTOR" "$SFT_INIT"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "[FATAL] Required directory missing: $required_dir"
        exit 3
    fi
done
for required_file in "$PSV_POOL" "$PAPER_TRAINER"; do
    if [[ ! -f "$required_file" ]]; then
        echo "[FATAL] Required file missing: $required_file"
        exit 4
    fi
done

actual_psv_sha256="$(sha256sum "$PSV_POOL" | awk '{print toupper($1)}')"
actual_paper_sha256="$(sha256sum "$PAPER_TRAINER" | awk '{print toupper($1)}')"
if [[ "$actual_psv_sha256" != "$EXPECTED_PSV_SHA256" ]]; then
    echo "[FATAL] PSV pool SHA mismatch: $actual_psv_sha256"
    exit 5
fi
if [[ "$actual_paper_sha256" != "$EXPECTED_PAPER_TRAINER_SHA256" ]]; then
    echo "[FATAL] paper trainer SHA mismatch: $actual_paper_sha256"
    exit 6
fi
if [[ -e "$EVAL_DIR" ]]; then
    echo "[FATAL] Eval directory already exists; use a new RUN_TAG: $EVAL_DIR"
    exit 7
fi

mkdir -p "$EVAL_DIR"
ln -s "$TARGET_STEP_DIR" "$EVAL_DIR/global_step_${EVAL_STEP}"

GIT_SHA="$( (cat "$HOME/project/GSV/qsv/GIT_SHA.txt" 2>/dev/null \
            || (cd "$HOME/project/GSV/qsv" && git rev-parse HEAD 2>/dev/null) \
            || echo unknown) | tr -d '\r\n')"
LOG_FILE="${EVAL_DIR}/eval_${VER}_step${EVAL_STEP}_n${N_SAMPLES}_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
META_FILE="${EVAL_DIR}/run.meta"
{
    echo "status=running"
    echo "job_id=${SLURM_JOB_ID:-unknown}"
    echo "git_sha=$GIT_SHA"
    echo "source_checkpoint=$TARGET_ACTOR"
    echo "eval_step=$EVAL_STEP"
    echo "n_samples=$N_SAMPLES"
    echo "fewshot_seed=$FEWSHOT_SEED"
    echo "psv_pool_sha256=$actual_psv_sha256"
    echo "paper_trainer_sha256=$actual_paper_sha256"
    echo "started_at=$(date --iso-8601=seconds)"
} > "$META_FILE"

echo "================================================================"
echo "GIT_SHA=$GIT_SHA"
echo "EVAL_NAME=$EVAL_NAME"
echo "EVAL_DIR=$EVAL_DIR"
echo "TARGET_ACTOR=$TARGET_ACTOR"
echo "PROMPT_PROTOCOL=psv_exact_1shot"
echo "PSV_POOL_SIZE=$EXPECTED_POOL_SIZE"
echo "PSV_POOL_SHA256=$actual_psv_sha256"
echo "PAPER_TRAINER_SHA256=$actual_paper_sha256"
echo "FEWSHOT_SEED_SCHEME=${FEWSHOT_SEED}:<dataset>"
echo "DATASETS=humaneval,mbpp,dafny2verus"
echo "N_SAMPLES=$N_SAMPLES TEMPERATURE=0.8 TOP_P=0.9 RESPONSE_LENGTH=2048"
echo "================================================================"

export RAY_memory_usage_threshold=0.98
# CUDA nodes in this cluster can inherit ROCm visibility variables from the
# submission environment. PyTorch rejects workers when both ROCr/HIP and CUDA
# visibility variables are present, even if the ROCm value is empty.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
# vLLM's hybrid memory pool rejects expandable-segment allocators. Clear both
# the legacy and current PyTorch configuration names instead of inheriting an
# interactive-shell setting.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF

set +e
apptainer exec \
    --nv \
    --bind "$HOME/project/GSV:/workspace/verl" \
    --bind "$HOME/project/model:/workspace/models" \
    --bind "$HOME/project/GSV/ray_tmp:/tmp" \
    --bind "$HOME/scratch/my_pylibs:/my_custom_libs" \
    --bind "$HOME/scratch/rust_env:/rust_env" \
    --bind "${HOST_CKPT_ROOT}:${HOST_CKPT_ROOT}" \
    --env PYTHONPATH=/my_custom_libs:/workspace/verl \
    --env RAY_memory_usage_threshold="$RAY_memory_usage_threshold" \
    --env TORCH_COMPILE_DISABLE=1 \
    --env WANDB_MODE=offline \
    --env HYDRA_FULL_ERROR=1 \
    "$HOME/project/verl_vllm.sif" \
    bash -c "
        set -euo pipefail
        unset ROCR_VISIBLE_DEVICES
        unset HIP_VISIBLE_DEVICES
        unset PYTORCH_CUDA_ALLOC_CONF
        unset PYTORCH_ALLOC_CONF
        ulimit -n 51200
        ulimit -u 65535
        cd /workspace/verl
        export RUSTUP_HOME=/rust_env/rustup
        export CARGO_HOME=/rust_env/cargo
        export PATH=/rust_env/cargo/bin:\$PATH:/workspace/verl/verus/verus-x86-linux
        export DATA_DIR=/workspace/verl/qsv/data
        python3 -u -X faulthandler qsv/main_qs.py \
            algorithm.adv_estimator=grpo \
            +algorithm.trainer_impl=qsv.qs_trainer_rebuttal_1shot \
            data.train_files=/workspace/verl/qsv/data/seed.parquet \
            data.val_files=/workspace/verl/qsv/data/seed.parquet \
            data.train_batch_size=8 \
            data.val_batch_size=${VAL_BATCH} \
            data.max_prompt_length=4096 \
            data.max_response_length=2048 \
            data.filter_overlong_prompts=True \
            data.truncation=right \
            +algorithm.q_gen_n=8 \
            +algorithm.enable_retry=false \
            +algorithm.s_corrective_n=0 \
            +algorithm.s_corrective_top_m=0 \
            +algorithm.q_reward_mode=band_1of8 \
            +algorithm.q_branch_weight_mode=static \
            +algorithm.q_loss_scale=0.375 \
            +algorithm.seed_pool_override=/workspace/verl/qsv/data/curated_sft/dafny2verus_verl_repaired_canonicalized.jsonl \
            +algorithm.local_test_mode=false \
            +algorithm.k_answers=8 \
            +algorithm.max_shots=6 \
            actor_rollout_ref.model.path=${SFT_INIT} \
            actor_rollout_ref.actor.optim.optimizer=AdamW \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=4096 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.actor.grad_clip=1.0 \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=False \
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
            actor_rollout_ref.rollout.max_model_len=8192 \
            actor_rollout_ref.rollout.prompt_length=4096 \
            actor_rollout_ref.rollout.response_length=2048 \
            actor_rollout_ref.rollout.enforce_eager=True \
            actor_rollout_ref.rollout.temperature=0.8 \
            actor_rollout_ref.rollout.top_p=0.9 \
            actor_rollout_ref.rollout.agent.num_workers=2 \
            reward_model.use_reward_loop=False \
            algorithm.use_kl_in_reward=False \
            trainer.critic_warmup=0 \
            trainer.logger='[console,wandb]' \
            trainer.project_name=question_solver_rl_eval \
            trainer.experiment_name=${EVAL_NAME} \
            trainer.n_gpus_per_node=2 \
            trainer.nnodes=1 \
            trainer.save_freq=999999 \
            trainer.test_freq=999999 \
            trainer.total_training_steps=0 \
            trainer.default_local_dir=${EVAL_DIR} \
            trainer.val_before_train=false \
            +trainer.eval_rolls=${N_SAMPLES} \
            +trainer.eval_steps=[${EVAL_STEP}] \
            +trainer.skip_base_eval=true \
            +trainer.require_ckpt=true \
            +trainer.log_rollout=false \
            +trainer.psv_fewshot_data=/workspace/verl/psv/data/dafny2verus.jsonl \
            +trainer.psv_fewshot_expected_pool_size=${EXPECTED_POOL_SIZE} \
            +trainer.psv_fewshot_seed=${FEWSHOT_SEED} \
            +trainer.eval_temperature=0.8 \
            +trainer.eval_dataset_names='[humaneval,mbpp,dafny2verus]' \
            +trainer.eval_dump_full_traces=false \
            trainer.val_only=true
    " 2>&1 | awk -v v="[$VER]" '{print v" "$0; fflush()}' | tee "$LOG_FILE"
pipeline_status=("${PIPESTATUS[@]}")
job_status="${pipeline_status[0]}"
for component_status in "${pipeline_status[@]}"; do
    if [[ "$component_status" -ne 0 ]]; then
        job_status="$component_status"
        break
    fi
done
set -e

{
    echo "exit_code=$job_status"
    echo "completed_at=$(date --iso-8601=seconds)"
    if [[ "$job_status" -eq 0 ]]; then
        echo "status=complete"
    else
        echo "status=failed"
    fi
} >> "$META_FILE"

exit "$job_status"
