#!/bin/bash
# ============================================================================
# v45 final-checkpoint pass@1/5/10 evaluation (n=100)
# Skips base_model eval. Fails fast if global_step_3200/actor missing.
# ============================================================================
cd ~/project/GSV
IMAGE_NAME="$HOME/project/verl_vllm.sif "
VER=v45eval
printf '\033kqsv-%s\033\\' "$VER"

# ----- paths (host == container, via passthrough bind) ------------------------
HOST_CKPT_ROOT="/scratch/${USER}/checkpoints"
RUN_NAME="Qwen2.5-Coder-3B-QGSG-SFTinit-RL-rootonly-qscale0375-band1of8-strictpool-nofreeze-v1"
CKPT_DIR="${HOST_CKPT_ROOT}/${RUN_NAME}"
EVAL_STEP="${1:-2400}"   # override via CLI: ./run_apptainer_v45_eval_n100.sh 2800
TARGET_ACTOR="${CKPT_DIR}/global_step_${EVAL_STEP}/actor"
SFT_INIT="${HOST_CKPT_ROOT}/Qwen2.5-Coder-3B-QGSG-SFT-v7-merged/global_step_1038_merged"
N_SAMPLES=100
VAL_BATCH=400

echo "================================================================"
echo "  EVAL  v=$VER  step=$EVAL_STEP  n=$N_SAMPLES"
echo "  HOST_CKPT_ROOT : $HOST_CKPT_ROOT"
echo "  RUN_NAME       : $RUN_NAME"
echo "  CKPT_DIR       : $CKPT_DIR"
echo "  TARGET_ACTOR   : $TARGET_ACTOR"
echo "  SFT_INIT       : $SFT_INIT"
echo "================================================================"

# ----- host-side fail-fast check ----------------------------------------------
if [ ! -d "$TARGET_ACTOR" ]; then
    echo "[ERROR] Checkpoint actor dir not found on host: $TARGET_ACTOR"
    echo "[ERROR] Available steps under ${CKPT_DIR}:"
    ls -1d ${CKPT_DIR}/global_step_* 2>/dev/null | head -20
    exit 1
fi
if [ ! -d "$SFT_INIT" ]; then
    echo "[ERROR] SFT init dir not found on host: $SFT_INIT"
    exit 1
fi
echo "[ok] host paths verified."

LOG_FILE="eval_${VER}_step${EVAL_STEP}_n${N_SAMPLES}_$(date +%Y%m%d_%H%M%S).log"
export RAY_memory_usage_threshold=0.98
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apptainer exec \
    --nv \
    --bind $HOME/project/GSV:/workspace/verl \
    --bind $HOME/project/model:/workspace/models \
    --bind ./ray_tmp:/tmp \
    --bind $HOME/scratch/my_pylibs:/my_custom_libs \
    --bind $HOME/scratch/rust_env:/rust_env \
    --bind ${HOST_CKPT_ROOT}:${HOST_CKPT_ROOT} \
    --env PYTHONPATH=/workspace/verl \
    --env RAY_memory_usage_threshold=$RAY_memory_usage_threshold \
    --env TORCH_COMPILE_DISABLE=1 \
    --env WANDB_MODE=offline \
    --env HYDRA_FULL_ERROR=1 \
    $IMAGE_NAME \
    bash -c " \
        unset ROCR_VISIBLE_DEVICES; \
        set -ex; \
        ulimit -n 51200; \
        ulimit -u 65535; \
        cd /workspace/verl; \
        export RUSTUP_HOME=/rust_env/rustup; \
        export CARGO_HOME=/rust_env/cargo; \
        export PATH=/rust_env/cargo/bin:\$PATH:/workspace/verl/verus/verus-x86-linux; \
        export DATA_DIR=/workspace/verl/qsv/data; \
        echo '[container] verifying ckpt actor dir:'; \
        ls -1d ${TARGET_ACTOR} || { echo '[FATAL] ckpt actor missing inside container'; exit 2; }; \
        python3 -u -X faulthandler qsv/main_qs.py \
            algorithm.adv_estimator=grpo \
            data.train_files=/workspace/verl/qsv/data/seed.parquet \
            data.val_files=/workspace/verl/qsv/data/seed.parquet \
            data.train_batch_size=8 \
            data.val_batch_size=${VAL_BATCH} \
            data.max_prompt_length=4096 \
            data.max_response_length=4096 \
            data.filter_overlong_prompts=True \
            data.truncation='right' \
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
            actor_rollout_ref.rollout.response_length=4096 \
            actor_rollout_ref.rollout.enforce_eager=True \
            actor_rollout_ref.rollout.temperature=1.0 \
            actor_rollout_ref.rollout.agent.num_workers=2 \
            reward_model.use_reward_loop=False \
            algorithm.use_kl_in_reward=False \
            trainer.critic_warmup=0 \
            trainer.logger='[console,wandb]' \
            trainer.project_name='question_solver_rl_eval' \
            trainer.experiment_name='Qwen2.5-Coder-3B-v45-eval-step${EVAL_STEP}-n${N_SAMPLES}' \
            trainer.n_gpus_per_node=2 \
            trainer.nnodes=1 \
            trainer.save_freq=999999 \
            trainer.test_freq=999999 \
            trainer.total_training_steps=0 \
            trainer.default_local_dir=${CKPT_DIR} \
            trainer.val_before_train=false \
            +trainer.eval_rolls=${N_SAMPLES} \
            +trainer.eval_steps=[${EVAL_STEP}] \
            +trainer.skip_base_eval=true \
            +trainer.require_ckpt=true \
            +trainer.log_rollout=false \
            trainer.val_only=true"   2>&1 | awk -v v="[$VER]" '{print v" "$0; fflush()}' | tee "$LOG_FILE"
