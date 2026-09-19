#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/launcher_common.sh"
run_training "${EXPERIMENT_NAME:-gammaopd}" \
  actor_rollout_ref.actor.policy_loss.causal_opd=True \
  actor_rollout_ref.actor.policy_loss.causal_gamma="${CAUSAL_GAMMA:-0.99}" \
  actor_rollout_ref.actor.policy_loss.use_reward_advantage=True \
  actor_rollout_ref.actor.policy_loss.scale_reverse_kl_by_abs_mean=True \
  actor_rollout_ref.actor.policy_loss.apply_softsign_to_opd_advantages=True \
  "$@"
