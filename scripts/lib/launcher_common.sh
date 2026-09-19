#!/usr/bin/env bash
set -euo pipefail

launcher_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="${REPO_ROOT:-${launcher_root}}"

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    printf '[ERROR] %s must be set.\n' "${name}" >&2
    exit 2
  fi
}

check_ray_ports() {
  local component_port
  for component_port in \
    "${MASTER_PORT}" \
    "${RAY_DASHBOARD_PORT}" \
    "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
    "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
    "${RAY_METRICS_EXPORT_PORT}" \
    "${RAY_RUNTIME_ENV_AGENT_PORT}"; do
    if (( component_port >= RAY_MIN_WORKER_PORT && component_port <= RAY_MAX_WORKER_PORT )); then
      printf '[ERROR] Ray component port %s overlaps worker range %s-%s.\n' \
        "${component_port}" "${RAY_MIN_WORKER_PORT}" "${RAY_MAX_WORKER_PORT}" >&2
      exit 2
    fi
  done
}

start_ray() {
  local node_rank="$1"
  if (( node_rank == 0 )); then
    ray start --head \
      --port="${MASTER_PORT}" \
      --node-ip-address="${MASTER_ADDR}" \
      --include-dashboard=false \
      "${RAY_PORT_ARGS[@]}"
  else
    until ray start --address="${MASTER_ADDR}:${MASTER_PORT}" "${RAY_PORT_ARGS[@]}" --block; do
      sleep 3
    done
  fi
}

wait_for_gpus() {
  local expected_gpus="$1"
  local max_wait_seconds="${MAX_WAIT_SECONDS:-6000}"
  local start_time
  local ray_gpu_status
  local cluster_gpus
  local available_gpus
  local node_summary

  start_time="$(date +%s)"
  while true; do
    ray_gpu_status="$(python3 -c "
import ray
from ray._private.state import available_resources_per_node

try:
    ray.init(address='auto', logging_level='error', ignore_reinit_error=True)
    cluster_gpus = ray.cluster_resources().get('GPU', 0)
    per_node = available_resources_per_node()
    available_gpus = sum(
        node_info.get('GPU', node_info.get('NPU', 0))
        for node_info in per_node.values()
    )
    node_summary = ','.join(
        node_id[-8:] + ':' + str(node_info.get('GPU', node_info.get('NPU', 0)))
        for node_id, node_info in sorted(per_node.items())
    )
    print(f'{int(cluster_gpus)} {int(available_gpus)} {node_summary}')
    ray.shutdown()
except Exception:
    print('0 0 unavailable')
")"
    read -r cluster_gpus available_gpus node_summary <<< "${ray_gpu_status}"
    local elapsed=$(( "$(date +%s)" - start_time ))
    printf '[INFO] [%ss/%ss] cluster=%s/%s available=%s/%s per_node=%s\n' \
      "${elapsed}" "${max_wait_seconds}" "${cluster_gpus}" "${expected_gpus}" \
      "${available_gpus}" "${expected_gpus}" "${node_summary}"

    if (( cluster_gpus >= expected_gpus && available_gpus >= expected_gpus )); then
      return
    fi
    if (( elapsed > max_wait_seconds )); then
      printf '[ERROR] Timed out waiting for %s GPUs.\n' "${expected_gpus}" >&2
      exit 1
    fi
    sleep 5
  done
}

run_training() {
  local experiment_name="$1"
  shift

  require_value STUDENT_MODEL
  require_value TEACHER_MODEL
  require_value TRAIN_FILE
  require_value VAL_FILE

  local max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
  local max_response_length="${MAX_RESPONSE_LENGTH:-16384}"
  local train_batch_size="${TRAIN_BATCH_SIZE:-1024}"
  local mini_batch_size="${PPO_MINI_BATCH_SIZE:-1024}"
  local micro_batch_size="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
  local nnodes="${NNODES:-${DISTRIBUTED_NODE_COUNT:-1}}"
  local gpus_per_node="${N_GPUS_PER_NODE:-8}"
  local node_rank="${NODE_RANK:-${DISTRIBUTED_NODE_RANK:-0}}"
  local actor_max_token_len=$((max_prompt_length + max_response_length))
  local output_root="${OUTPUT_ROOT:-${repo_root}/outputs}"
  local ckpts_dir="${CKPTS_DIR:-${output_root}/checkpoints/${experiment_name}}"
  local tensorboard_dir="${TENSORBOARD_DIR:-${output_root}/tensorboard/${experiment_name}}"

  export PYTHONPATH="${repo_root}/verl${PYTHONPATH:+:${PYTHONPATH}}"
  export TENSORBOARD_DIR="${tensorboard_dir}"
  export MASTER_ADDR="${MASTER_ADDR:-$(printf '%s' "${DISTRIBUTED_MASTER_HOSTS:-localhost}" | cut -d, -f1)}"
  export MASTER_PORT="${MASTER_PORT:-6386}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
  fi
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
  export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-3600}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

  local visible_gpu_count="${gpus_per_node}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    visible_gpu_count="${#visible_gpu_ids[@]}"
  fi
  printf '[INFO] node=%s/%s master=%s:%s visible_gpus=%s\n' \
    "${node_rank}" "${nnodes}" "${MASTER_ADDR}" "${MASTER_PORT}" "${visible_gpu_count}"
  printf '[INFO] checkpoints=%s\n[INFO] tensorboard=%s\n' "${ckpts_dir}" "${tensorboard_dir}"

  RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
  RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-54405}"
  RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-54005}"
  RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-48002}"
  RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-51001}"
  RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-52000}"
  RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-53999}"
  RAY_PORT_ARGS=(
    --dashboard-port="${RAY_DASHBOARD_PORT}"
    --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}"
    --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}"
    --metrics-export-port="${RAY_METRICS_EXPORT_PORT}"
    --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}"
    --min-worker-port="${RAY_MIN_WORKER_PORT}"
    --max-worker-port="${RAY_MAX_WORKER_PORT}"
  )

  check_ray_ports
  trap "if (( ${node_rank} == 0 )); then ray stop || true; fi" EXIT
  start_ray "${node_rank}"

  if (( node_rank == 0 )); then
    wait_for_gpus "$((nnodes * gpus_per_node))"
    cd "${repo_root}/verl"
    # RADAR is the default actor optimizer. AdamW alternative:
    # actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
    # actor_rollout_ref.actor.optim.optimizer=AdamW \
    python3 -m verl.trainer.main_ppo \
      data.train_files="${TRAIN_FILE}" \
      data.val_files="${VAL_FILE}" \
      data.prompt_key=prompt \
      data.filter_overlong_prompts=True \
      data.truncation=error \
      data.shuffle=True \
      data.seed="${DATA_SEED:-42}" \
      data.return_raw_chat=True \
      +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING:-False}" \
      data.max_prompt_length="${max_prompt_length}" \
      data.max_response_length="${max_response_length}" \
      data.train_batch_size="${train_batch_size}" \
      algorithm.adv_estimator=grpo \
      algorithm.rollout_correction.rollout_is=token \
      algorithm.rollout_correction.rollout_is_threshold=5.0 \
      algorithm.rollout_correction.rollout_rs=null \
      algorithm.rollout_correction.bypass_mode=false \
      algorithm.use_kl_in_reward=False \
      algorithm.kl_ctrl.kl_coef=0.0 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.path="${STUDENT_MODEL}" \
      +actor_rollout_ref.ref.model.path="${TEACHER_MODEL}" \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
      actor_rollout_ref.actor.optim.optimizer_impl=rad.optim \
      actor_rollout_ref.actor.optim.optimizer=RADAR \
      actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-5}" \
      actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
      actor_rollout_ref.actor.ppo_mini_batch_size="${mini_batch_size}" \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${micro_batch_size}" \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.0 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_max_token_len}" \
      actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-False}" \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}" \
      actor_rollout_ref.rollout.calculate_log_probs=true \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-4}" \
      actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}" \
      actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-4}" \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.rollout.max_num_batched_tokens="${actor_max_token_len}" \
      actor_rollout_ref.rollout.temperature="${TEMPERATURE:-1.0}" \
      actor_rollout_ref.rollout.top_p="${TOP_P:-1.0}" \
      actor_rollout_ref.rollout.top_k="${TOP_K:--1}" \
      actor_rollout_ref.rollout.val_kwargs.do_sample=True \
      actor_rollout_ref.rollout.val_kwargs.temperature="${TEMPERATURE:-1.0}" \
      actor_rollout_ref.rollout.val_kwargs.top_p="${TOP_P:-1.0}" \
      actor_rollout_ref.rollout.val_kwargs.top_k="${TOP_K:--1}" \
      actor_rollout_ref.rollout.val_kwargs.n=1 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE:-4}" \
      actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD:-True}" \
      reward_model.reward_manager="${REWARD_MANAGER:-dapo}" \
      trainer.logger="${LOGGER:-[\"console\",\"tensorboard\"]}" \
      trainer.critic_warmup=0 \
      trainer.val_before_train=False \
      trainer.log_val_generations=0 \
      trainer.project_name=opd \
      trainer.experiment_name="${experiment_name}" \
      trainer.n_gpus_per_node="${gpus_per_node}" \
      trainer.nnodes="${nnodes}" \
      trainer.test_freq="${TEST_FREQ:-10}" \
      trainer.save_freq="${SAVE_FREQ:-5}" \
      trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-20}" \
      trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP:-20}" \
      trainer.total_epochs="${TOTAL_EPOCHS:-3}" \
      trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-150}" \
      trainer.default_local_dir="${ckpts_dir}" \
      trainer.resume_mode=auto \
      "$@"
  fi
}
