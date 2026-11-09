#!/bin/bash
#SBATCH --job-name=qwen2.5-3B-qsv
#SBATCH --account=chenjun3
#SBATCH --partition=a100x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2          # 修改：脚本中 trainer.n_gpus_per_node=1，所以申请1张卡即可
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mem=128G


# docker image 

# 加载环境，下载镜像
module load google/go
module load apptainer/1.4.5
apptainer pull qwen_env.sif docker://registry.cn-beijing.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.11.0
mkdir -p ~/scratch/apptainer_cache
mkdir -p ~/scratch/apptainer_tmp
export APPTAINER_CACHEDIR=~/scratch/apptainer_cache
export APPTAINER_TMPDIR=~/scratch/apptainer_tmp
export APPTAINER_DOWNLOAD_CONCURRENCY=1
APPTAINER_DOWNLOAD_CONCURRENCY=1 apptainer pull verl_vllm.sif docker://v6wkgkj5.mirror.aliyuncs.com/verlai/verl:vllm012.dev3
APPTAINER_DOWNLOAD_CONCURRENCY=1 apptainer pull verl_vllm.sif docker://docker.m.daocloud.io/verlai/verl:vllm012.dev3
APPTAINER_DOWNLOAD_CONCURRENCY=1 apptainer pull verl_vllm.sif docker://docker.nju.edu.cn/verlai/verl:vllm012.dev3
APPTAINER_DOWNLOAD_CONCURRENCY=1 apptainer pull verl_vllm.sif docker://docker.1panel.live/verlai/verl:vllm012.latest
APPTAINER_DOWNLOAD_CONCURRENCY=1 apptainer pull verl_vllm.sif docker://docker.1panel.live/verlai/verl:vllm012.latest

docker pull verlai/verl:vllm016.dev.qwen3_5
# 创建目录

# 下载模型
# modelscope download --model  Qwen/Qwen3.5-4B --local_dir 
modelscope download --model  Qwen/Qwen2.5-Coder-3B --local_dir  Qwen2.5-Coder-3B
# 
mkdir -p $HOME/scratch/lean_env
cd $HOME/scratch/lean_env
# 下载 Linux 二进制包 (v4.27.0 zip版)
# 如果网络受限，可以使用我提供的 python 脚本: python qsv/lean_env/download_lean_zip.py
wget https://github.com/leanprover/lean4/releases/download/v4.27.0/lean-4.27.0-linux.zip
# 解压 (使用 unzip)
unzip lean-4.27.0-linux.zip
# 此时路径应为: $HOME/scratch/lean_env/lean-4.27.0-linux/bin/lean

tmux new -s my_install_SIF_session
tmux attach -t my_install_SIF_session
APPTAINER_DOWNLOAD_CONCURRENCY=1 apptainer pull verl_vllm_12.sif docker://docker.1panel.live/verlai/verl:vllm012.latest

apptainer pull vllm016.dev.qwen3_5.sif docker://docker.1panel.live/verlai/verl:vllm016.dev.qwen3_5
# --- 准备工作 ---
#  挂载目录， gpu 
# docker run -it -d --name verl_test --gpus all -v e:\GSV:/workspace/verl -v e:\GSV\model:/workspace/models --ipc=host verlai/verl:vllm012.latest bash
#  下载rustup环境
# docker exec verl_test bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"

# export CARGO_HOME="$HOME/scratch/rust_env/cargo"
# export RUSTUP_HOME="$HOME/scratch/rust_env/rustup"
# docker exec verl_eval bash -c "source /root/.cargo/env && rustup toolchain install 1.91.0-x86_64-unknown-linux-gnu"
# 检查 docker exec verl_eval bash -c "source /root/.cargo/env && rustup show"
# 切换到提交作业的目录的
cd $SLURM_SUBMIT_DIR
tmux new -s my_debug_session
tmux attach -t my_debug_session
tmux new -s my_debug_session_2
tmux attach -t my_debug_session_2
tmux new -s my_debug_session_3
tmux attach -t my_debug_session_3
tmux new -s my_debug_session_4
tmux attach -t my_debug_session_4
srun --account=chenjun3 --partition=a100x4 --nodes=1 --gres=gpu:2 --cpus-per-task=16 --mem=128G --pty /bin/bash

srun --account=chenjun3 --partition=gpu --nodes=1 --gres=gpu:2 --cpus-per-task=16 --mem=64G --pty /bin/bash

# --- 加载环境 ---
echo "Loading Apptainer"
# 根据你之前的 module avail，建议优先用 apptainer，如果没有则换回 singularity
module purge
module load google/go
module load apptainer/1.4.5
module load nvidia/cuda/12.9

# 查看文件目录层级大小。 方便删除不必须要的文件
du -h --max-depth=1 ~ | sort -h
# /bin/myDiskQuota
# --- 设置 Docker 脚本中需要的环境变量 ---
export RAY_memory_usage_threshold=0.98




# 测试verus 环境
# 在集群上运行这个命令测试环境
apptainer exec \
    --nv \
    --bind $HOME/project/GSV:/workspace/verl \
    --bind $HOME/scratch/rust_env:/rust_env \
    /home/${USER}/project/verl_vllm.sif \
    bash -c "
        export RUSTUP_HOME=/rust_env/rustup
        export CARGO_HOME=/rust_env/cargo
        export PATH=/lean4_root/bin:/rust_env/cargo/bin:\$PATH:/workspace/verl/verus/verus-x86-linux

        
        echo '=== 检查环境变量 ==='
        echo \"RUSTUP_HOME: \$RUSTUP_HOME\"
        echo \"CARGO_HOME: \$CARGO_HOME\"
        echo \"PATH: \$PATH\"
        
        echo ''
        echo '=== 检查 rustup ==='
        which rustup
        rustup --version
        
        echo ''
        echo '=== 检查 cargo ==='
        which cargo
        cargo --version
        
        echo ''
        echo '=== 检查 verus ==='
        which verus
        ls -la /workspace/verl/verus/verus-x86-linux/verus
        
        echo ''
        echo '=== 测试 verus 运行 ==='
        chmod -R +x /workspace/verl/verus/verus-x86-linux/verus
        /workspace/verl/verus/verus-x86-linux/verus --version || echo 'verus --version failed'
    "
#  chmod -R +x ~/project/GSV/verus/verus-x86-linux   手动赋予权限，否则没有权限
#  rustup toolchain install 1.91.0-x86_64-unknown-linux-gnu
# 确保日志和 checkpoint 目录存在
mkdir -p slurm_logs
mkdir -p $HOME/scratch/checkpoints

# --- 执行命令 ---
echo "Starting Container..."

cd ~/project/GSV
# /bin/myDiskQuota
IMAGE_NAME="$HOME/project/verl_vllm.sif "
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
export RAY_memory_usage_threshold=0.98
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
apptainer exec \
    --nv \
    --bind $HOME/project/GSV:/workspace/verl \
    --bind $HOME/project/model:/workspace/models \
    --bind ./ray_tmp:/tmp \
    --bind $HOME/scratch/my_pylibs:/my_custom_libs \
    --bind $HOME/scratch/rust_env:/rust_env \
    --bind $HOME/scratch/checkpoints:/workspace/verl/checkpoints \
    --env PYTHONPATH=/workspace/verl \
    --env RAY_memory_usage_threshold=$RAY_memory_usage_threshold \
    --env TORCH_COMPILE_DISABLE=1 \
    --env WANDB_MODE=offline \
    --env HYDRA_FULL_ERROR=1 \
    $IMAGE_NAME \
    bash -c " \
        unset ROCR_VISIBLE_DEVICES; \
        set -x; \
        ulimit -n 51200; \
        ulimit -u 65535; \
        cd /workspace/verl; \
        export RUSTUP_HOME=/rust_env/rustup; \
        export CARGO_HOME=/rust_env/cargo; \
        export PATH=/rust_env/cargo/bin:\$PATH:/workspace/verl/verus/verus-x86-linux; \
        export DATA_DIR=/workspace/verl/qsv/data; \
        python3 -u -X faulthandler qsv/main_qs.py \
            algorithm.adv_estimator=grpo \
            data.train_files=/workspace/verl/qsv/data/seed.parquet \
            data.val_files=/workspace/verl/qsv/data/seed.parquet \
            data.train_batch_size=8 \
            +algorithm.trainer_impl=qsv.qs_trainer_v45_rootonly_resume \
            +algorithm.q_gen_n=8 \
            +algorithm.enable_retry=false \
            +algorithm.s_corrective_n=0 \
            +algorithm.s_corrective_top_m=0 \
            +algorithm.q_branch_warmup.enabled=true \
            +algorithm.q_branch_warmup.steps=100 \
            +algorithm.q_branch_warmup.min_scale=0.25 \
            +algorithm.anchor_phase.freeze_seed_pool=false \
            +algorithm.pass_rate_decay=0.996 \
            +algorithm.seed_pool_gate.min_impl_chars=64 \
            +algorithm.seed_pool_gate.max_seed_sim=0.78 \
            +algorithm.seed_pool_gate.max_pool_sim=0.72 \
            +algorithm.seed_pool_gate.max_sibling_sim=0.65 \
            +algorithm.seed_pool_gate.reject_struct_dup=true \
            +algorithm.seed_pool_gate.duplicate_seed_sim_threshold=0.80 \
            +algorithm.seed_pool_gate.duplicate_struct_sim_threshold=0.90 \
            +algorithm.s_corrective_reward_scale=0.5 \
            +algorithm.seed_sampling_mode=root_only \
            +algorithm.mcts_prune_min_visits=0 \
            +algorithm.ucb_root_ratio=0.25 \
            +algorithm.ucb_c=1.5 \
            +algorithm.ucb_descend_margin=-0.05 \
            +algorithm.mcts_kin_prior_weight=0.1 \
            +algorithm.path_credit_gamma=0.70 \
            +algorithm.adaptive_difficulty=false \
            +algorithm.difficulty_delta=0.30 \
            +algorithm.s_ability_ema_alpha=0.10 \
            +algorithm.q_filter_adv_weight=0.0 \
            +algorithm.q_loss_scale=0.375 \
            +algorithm.q_filter_penalty_scale=0.50 \
            +algorithm.q_gen_max_tokens=256 \
            +algorithm.q_gen_hard_max_think_chars=0 \
            +algorithm.q_gen_hard_max_raw_chars=12000 \
            +algorithm.q_gen_soft_max_spec_chars=2400 \
            +algorithm.q_gen_soft_max_spec_lines=40 \
            +algorithm.q_gen_hard_max_spec_chars=7000 \
            +algorithm.q_gen_hard_max_spec_lines=96 \
            +algorithm.q_valid_style_penalty_max=0.08 \
            +algorithm.q_valid_style_soft_raw_chars=3200 \
            +algorithm.q_valid_style_soft_think_chars=0 \
            +algorithm.q_pool_min_format_penalty=0.75 \
            +algorithm.q_pool_max_raw_chars=3200 \
            +algorithm.q_pool_max_think_chars=0 \
+algorithm.freeze_s_branch=false \
            +algorithm.freeze_q_branch=false \
            data.val_batch_size=32 \
            data.max_prompt_length=4096 \
            data.max_response_length=4096 \
            data.filter_overlong_prompts=True \
            data.truncation='right' \
            actor_rollout_ref.model.path=/workspace/verl/checkpoints/Qwen2.5-Coder-3B-QGSG-SFT-v7-merged/global_step_1038_merged \
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
            actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
            actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=False \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
            actor_rollout_ref.rollout.max_model_len=8192 \
            actor_rollout_ref.rollout.prompt_length=4096 \
            actor_rollout_ref.rollout.enforce_eager=True \
            actor_rollout_ref.rollout.response_length=4096 \
            actor_rollout_ref.rollout.agent.num_workers=2 \
            reward_model.use_reward_loop=False \
            actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            algorithm.use_kl_in_reward=False \
            +algorithm.max_shots=6 \
            +algorithm.local_test_mode=false \
            +algorithm.k_answers=8 \
            trainer.critic_warmup=0 \
            trainer.logger='[console,wandb]' \
            trainer.project_name='question_solver_rl' \
            trainer.experiment_name='Qwen2.5-Coder-3B-QGSG-SFTinit-RL-rootonly-qscale0375-band1of8-strictpool-nofreeze-v45' \
            trainer.n_gpus_per_node=2 \
            trainer.nnodes=1 \
            trainer.save_freq=400 \
            trainer.test_freq=100 \
            trainer.total_training_steps=3200 \
            trainer.default_local_dir=/workspace/verl/checkpoints/Qwen2.5-Coder-3B-QGSG-SFTinit-RL-rootonly-qscale0375-band1of8-strictpool-nofreeze-v1 \
            trainer.val_before_train=false \
            +trainer.log_rollout=true \
            trainer.val_only=false"   2>&1 | tee "$LOG_FILE"
