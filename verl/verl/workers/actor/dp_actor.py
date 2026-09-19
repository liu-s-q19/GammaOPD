# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = [
    "apply_aopd_piecewise_token_objective",
    "apply_reopold_signal_processor",
    "build_response_row_mask_from_attention",
    "DataParallelPPOActor",
    "compute_adaptive_opd_coef_update",
    "compute_aopd_local_kl",
    "combine_local_and_causal_opd",
    "discounted_reverse_cumsum",
    "mix_reward_and_opd_advantages",
    "masked_abs_mean_per_sample",
    "normalize_discounted_suffix_values",
    "prepare_response_logits_for_local_group_stats",
    "scale_by_masked_abs_mean_per_sample",
    "validate_reopold_config",
]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def discounted_reverse_cumsum(x: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute a masked discounted suffix return over token-level values."""
    x = x * mask
    out = torch.zeros_like(x)
    running = torch.zeros_like(x[:, 0])
    for t in range(x.shape[1] - 1, -1, -1):
        running = x[:, t] + gamma * running
        running = running * mask[:, t]
        out[:, t] = running
    return out * mask


def normalize_discounted_suffix_values(
    values: torch.Tensor, mask: torch.Tensor, gamma: float, eps: float = 1e-6
) -> torch.Tensor:
    """Normalize masked discounted suffix values by their discounted suffix weights."""
    suffix_weights = discounted_reverse_cumsum(torch.ones_like(values), mask, gamma)
    normalized = values / suffix_weights.clamp_min(eps)
    return normalized * mask.to(dtype=values.dtype)


def combine_local_and_causal_opd(
    token_reverse_kl: torch.Tensor, mask: torch.Tensor, gamma: float, alpha: float
) -> torch.Tensor:
    """Keep the token-local OPD term and add a scaled future-token causal correction."""
    token_reverse_kl = token_reverse_kl * mask.to(dtype=token_reverse_kl.dtype)
    causal_reverse_kl = discounted_reverse_cumsum(token_reverse_kl, mask, gamma)
    future_reverse_kl = (causal_reverse_kl - token_reverse_kl) * mask.to(dtype=token_reverse_kl.dtype)
    return -(token_reverse_kl + alpha * future_reverse_kl)


def masked_abs_mean_per_sample(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute per-sample masked absolute mean, falling back to 1 for empty masks."""
    mask = mask.to(dtype=values.dtype)
    abs_sum = (values.abs() * mask).sum(dim=-1, keepdim=True)
    count = mask.sum(dim=-1, keepdim=True)
    return torch.where(count > 0, abs_sum / count.clamp_min(1.0), torch.ones_like(abs_sum))


def scale_by_masked_abs_mean_per_sample(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale each sample by its masked absolute mean while keeping masked tokens zero."""
    scale = masked_abs_mean_per_sample(values, mask)
    scaled = values / scale.clamp_min(eps)
    return scaled * mask.to(dtype=values.dtype)


def mix_reward_and_opd_advantages(
    reward_advantages: torch.Tensor,
    opd_advantages: torch.Tensor,
    reward_advantage_coef: float,
    opd_advantage_coef: float,
    use_reward_advantage: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale reward/OPD advantages independently and return the final mixed advantages."""
    scaled_reward_advantages = reward_advantages * reward_advantage_coef
    scaled_opd_advantages = opd_advantages * opd_advantage_coef
    if use_reward_advantage:
        final_advantages = scaled_reward_advantages + scaled_opd_advantages
    else:
        final_advantages = scaled_opd_advantages
    return scaled_reward_advantages, scaled_opd_advantages, final_advantages


def compute_adaptive_opd_coef_update(
    current_coef: float,
    current_ema_kl: float | None,
    observed_reverse_kl: float,
    target_reverse_kl: float,
    ema_decay: float,
    update_rate: float,
    coef_min: float,
    coef_max: float,
) -> tuple[float, float]:
    """Update the adaptive OPD multiplier from a step-level reverse-KL observation."""
    if current_ema_kl is None:
        next_ema_kl = observed_reverse_kl
    else:
        next_ema_kl = ema_decay * current_ema_kl + (1.0 - ema_decay) * observed_reverse_kl
    next_coef = current_coef * torch.exp(
        torch.tensor(update_rate * (next_ema_kl - target_reverse_kl), dtype=torch.float32)
    ).item()
    next_coef = min(max(next_coef, coef_min), coef_max)
    return next_coef, next_ema_kl


def topk_log_probs_from_logits(logits: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token ids, token log-probs, and covered mass for the top-k support."""
    topk_logits, topk_ids = torch.topk(logits, k=topk, dim=-1)
    log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
    topk_log_probs = topk_logits - log_norm
    topk_mass = topk_log_probs.exp().sum(dim=-1)
    return topk_ids, topk_log_probs, topk_mass


def build_full_response_topk_ids(
    batch_size: int, seqlen: int, response_length: int, topk_ids: torch.Tensor
) -> torch.Tensor:
    """Place response-position top-k ids back into full-sequence layout aligned with logits slicing."""
    full_topk_ids = torch.zeros(
        batch_size,
        seqlen,
        topk_ids.size(-1),
        device=topk_ids.device,
        dtype=topk_ids.dtype,
    )
    full_topk_ids[:, -response_length - 1 : -1] = topk_ids
    return full_topk_ids


def build_response_row_mask_from_attention(attention_mask: torch.Tensor, response_length: int) -> torch.Tensor:
    """Map response-token positions into the flattened unpadded row layout."""
    batch_size, seqlen = attention_mask.shape
    response_position_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    response_position_mask[:, -response_length - 1 : -1] = True
    response_position_mask &= attention_mask.to(dtype=torch.bool)
    flat_attention_mask = attention_mask.reshape(batch_size * seqlen).to(dtype=torch.bool)
    return response_position_mask.reshape(batch_size * seqlen)[flat_attention_mask]


def prepare_response_logits_for_local_group_stats(
    logits_rmpad: torch.Tensor,
    attention_mask: torch.Tensor,
    response_length: int,
    use_ulysses_sp: bool,
    pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align response-token logits for local-group stats under both plain and Ulysses-SP layouts."""
    response_row_mask_rmpad = build_response_row_mask_from_attention(attention_mask, response_length)
    if use_ulysses_sp:
        logits_rmpad = gather_outputs_and_unpad(
            logits_rmpad,
            gather_dim=0,
            unpad_dim=0,
            padding_size=pad_size,
            grad_scaler=False,
        )
    return logits_rmpad[response_row_mask_rmpad], response_row_mask_rmpad


def compute_local_group_baseline(
    student_topk_log_probs: torch.Tensor,
    ref_topk_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    renorm: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the detached local-group mean reward and covered mass."""
    student_probs = student_topk_log_probs.detach().exp()
    covered_mass = student_probs.sum(dim=-1)
    topk_rewards = ref_topk_log_probs.detach() - student_topk_log_probs.detach()
    if renorm:
        student_weights = student_probs / covered_mass.clamp_min(eps).unsqueeze(-1)
    else:
        student_weights = student_probs
    local_group_mean = (student_weights * topk_rewards).sum(dim=-1)
    local_group_mean = local_group_mean * response_mask.to(dtype=local_group_mean.dtype)
    covered_mass = covered_mass * response_mask.to(dtype=covered_mass.dtype)
    return local_group_mean, covered_mass


def compute_aopd_local_kl(
    student_topk_log_probs: torch.Tensor,
    ref_topk_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute teacher-top-k forward-KL style local correction for AOPD non-positive tokens."""
    teacher_probs = ref_topk_log_probs.detach().exp()
    local_kl = (teacher_probs * (ref_topk_log_probs.detach() - student_topk_log_probs)).sum(dim=-1)
    return local_kl * response_mask.to(dtype=local_kl.dtype)


def apply_aopd_piecewise_token_objective(
    sampled_reward: torch.Tensor,
    local_kl: torch.Tensor,
    response_mask: torch.Tensor,
    positive_only_floor: float,
    nonpositive_coef: float,
) -> torch.Tensor:
    """Keep positive sampled-token OPD rewards and replace non-positive tokens with local-KL correction."""
    mask = response_mask.to(dtype=sampled_reward.dtype)
    positive_mask = (sampled_reward > positive_only_floor).to(dtype=sampled_reward.dtype) * mask
    nonpositive_mask = mask - positive_mask
    positive_term = sampled_reward * positive_mask
    nonpositive_term = -(nonpositive_coef * local_kl) * nonpositive_mask
    return (positive_term + nonpositive_term) * mask


def _compute_reopold_abs_quantile_mask(
    clipped_reward: torch.Tensor, response_mask: torch.Tensor, reward_quantile: float
) -> torch.Tensor:
    """Compute a per-sample absolute reward quantile mask over response tokens."""
    abs_reward = clipped_reward.abs()
    thresholds = torch.zeros_like(abs_reward[:, :1])
    for row_idx in range(abs_reward.shape[0]):
        valid = response_mask[row_idx].bool()
        if valid.any():
            thresholds[row_idx, 0] = torch.quantile(abs_reward[row_idx, valid], reward_quantile)
    return (abs_reward >= thresholds).to(dtype=clipped_reward.dtype) * response_mask.to(dtype=clipped_reward.dtype)


def apply_reopold_signal_processor(
    sampled_reward: torch.Tensor,
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    clip_min: float,
    clip_max: float,
    mask_mode: str,
    entropy_threshold: float,
    reward_quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply REOPOLD fixed-reward clipping and token masking to OPD sampled rewards."""
    mask = response_mask.to(dtype=sampled_reward.dtype)
    clipped_reward = torch.clamp(sampled_reward.detach(), min=clip_min, max=clip_max) * mask
    if mask_mode == "none":
        reopold_mask = mask
    elif mask_mode == "entropy":
        reopold_mask = ((entropy >= entropy_threshold).to(dtype=sampled_reward.dtype) * mask)
    elif mask_mode == "abs_quantile":
        reopold_mask = _compute_reopold_abs_quantile_mask(clipped_reward, response_mask, reward_quantile)
    else:
        raise ValueError(f"Unsupported REOPOLD mask mode: {mask_mode!r}")
    masked_reward = clipped_reward * reopold_mask
    return masked_reward, reopold_mask


def apply_suffix_group_baseline(
    sampled_reward: torch.Tensor,
    token_baseline: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate a token baseline into a suffix baseline and subtract it from causal rewards."""
    masked_reward = sampled_reward * response_mask.to(dtype=sampled_reward.dtype)
    suffix_group_baseline = discounted_reverse_cumsum(token_baseline, response_mask, gamma)
    opd_advantages = masked_reward - alpha * suffix_group_baseline
    return opd_advantages, suffix_group_baseline


def validate_suffix_group_baseline_config(policy_loss_config) -> None:
    """Validate suffix-baseline config constraints before actor updates."""
    if not policy_loss_config.use_suffix_group_baseline:
        return
    if not policy_loss_config.causal_opd:
        raise ValueError("policy_loss.use_suffix_group_baseline requires policy_loss.causal_opd=True")
    if not policy_loss_config.only_reverse_kl_advantages:
        raise ValueError(
            "policy_loss.use_suffix_group_baseline requires policy_loss.only_reverse_kl_advantages=True"
        )


def validate_causal_opd_extensions(policy_loss_config) -> None:
    """Validate new causal OPD extension flags before actor updates."""
    if not (policy_loss_config.normalize_causal_by_discounted_weight or policy_loss_config.use_local_future_correction):
        return
    if not policy_loss_config.causal_opd:
        raise ValueError("causal OPD extensions require policy_loss.causal_opd=True")
    if not policy_loss_config.only_reverse_kl_advantages:
        raise ValueError("causal OPD extensions require policy_loss.only_reverse_kl_advantages=True")
    if policy_loss_config.use_local_future_correction and policy_loss_config.use_suffix_group_baseline:
        raise ValueError("policy_loss.use_local_future_correction cannot be combined with use_suffix_group_baseline")
    if policy_loss_config.use_local_future_correction and policy_loss_config.use_local_group_baseline:
        raise ValueError("policy_loss.use_local_future_correction cannot be combined with use_local_group_baseline")
    if policy_loss_config.normalize_causal_by_discounted_weight and policy_loss_config.use_local_future_correction:
        raise ValueError(
            "policy_loss.normalize_causal_by_discounted_weight cannot be combined with "
            "use_local_future_correction until a combined formula is explicitly defined"
        )


def validate_aopd_config(policy_loss_config) -> None:
    """Validate AOPD constraints before actor updates."""
    if not policy_loss_config.use_aopd:
        return
    if not policy_loss_config.only_reverse_kl_advantages:
        raise ValueError("policy_loss.use_aopd requires policy_loss.only_reverse_kl_advantages=True")
    if policy_loss_config.aopd_nonpositive_mode != "local_topk_kl":
        raise ValueError(
            f"Unsupported policy_loss.aopd_nonpositive_mode={policy_loss_config.aopd_nonpositive_mode!r}; "
            "only 'local_topk_kl' is implemented"
        )
    if policy_loss_config.aopd_teacher_topk <= 0:
        raise ValueError("policy_loss.aopd_teacher_topk must be > 0 when AOPD is enabled")


def validate_reopold_config(policy_loss_config) -> None:
    """Validate REOPOLD constraints before actor updates."""
    if not policy_loss_config.use_reopold:
        return
    if not policy_loss_config.only_reverse_kl_advantages:
        raise ValueError("policy_loss.use_reopold requires policy_loss.only_reverse_kl_advantages=True")
    if policy_loss_config.causal_opd:
        raise ValueError("policy_loss.use_reopold currently cannot be combined with policy_loss.causal_opd")
    if policy_loss_config.reopold_mask_mode not in {"none", "entropy", "abs_quantile"}:
        raise ValueError(
            f"Unsupported policy_loss.reopold_mask_mode={policy_loss_config.reopold_mask_mode!r}; "
            "only 'none', 'entropy', and 'abs_quantile' are implemented"
        )
    if policy_loss_config.reopold_reward_clip_min > policy_loss_config.reopold_reward_clip_max:
        raise ValueError("policy_loss.reopold_reward_clip_min must be <= reopold_reward_clip_max")
    if not 0.0 <= policy_loss_config.reopold_reward_quantile <= 1.0:
        raise ValueError("policy_loss.reopold_reward_quantile must be in [0.0, 1.0]")


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None
        policy_loss_cfg = self.config.get("policy_loss", None)
        if policy_loss_cfg is not None:
            self._adaptive_opd_advantage_coef = float(policy_loss_cfg.opd_advantage_coef)
        else:
            self._adaptive_opd_advantage_coef = 1.0
        self._adaptive_opd_advantage_coef_ema_kl = None

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        compute_local_group_stats=False,
        gather_local_group_ref_log_probs=False,
        local_group_topk=8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # reset input_ids, attention_mask, position_ids to ref model inputs if ref model input_ids is different from actor input_ids
            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                response_position_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
                response_position_mask[:, -response_length - 1 : -1] = True
                response_position_mask &= attention_mask.to(dtype=torch.bool)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    local_group_topk_ids_rmpad = None
                    local_group_student_log_probs_rmpad = None
                    local_group_ref_log_probs_rmpad = None
                    if compute_local_group_stats or gather_local_group_ref_log_probs:
                        if local_group_topk > logits_rmpad.shape[-1]:
                            raise ValueError(
                                f"local_group_topk ({local_group_topk}) cannot exceed vocab size ({logits_rmpad.shape[-1]})"
                            )
                        response_logits_rmpad, response_row_mask_rmpad = prepare_response_logits_for_local_group_stats(
                            logits_rmpad=logits_rmpad,
                            attention_mask=attention_mask,
                            response_length=response_length,
                            use_ulysses_sp=self.use_ulysses_sp,
                            pad_size=pad_size if self.use_ulysses_sp else 0,
                        )
                        local_group_topk_ids_rmpad, local_group_student_log_probs_rmpad, _ = topk_log_probs_from_logits(
                            response_logits_rmpad, local_group_topk
                        )
                        if gather_local_group_ref_log_probs:
                            topk_ids = micro_batch["opd_local_topk_ids"]
                            response_topk_ids_rmpad = topk_ids[response_position_mask[:, -response_length - 1 : -1]]
                            log_norm_rmpad = torch.logsumexp(response_logits_rmpad, dim=-1, keepdim=True)
                            local_group_ref_log_probs_rmpad = torch.gather(
                                response_logits_rmpad,
                                dim=-1,
                                index=response_topk_ids_rmpad,
                            ) - log_norm_rmpad

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                full_local_group_topk_ids = None
                full_local_group_student_log_probs = None
                full_local_group_ref_log_probs = None
                if not self.use_fused_kernels and (compute_local_group_stats or gather_local_group_ref_log_probs):
                    full_local_group_topk_ids = torch.zeros(
                        batch_size,
                        seqlen,
                        local_group_topk_ids_rmpad.size(-1),
                        device=local_group_topk_ids_rmpad.device,
                        dtype=local_group_topk_ids_rmpad.dtype,
                    )
                    full_local_group_topk_ids[response_position_mask] = local_group_topk_ids_rmpad
                    full_local_group_student_log_probs = torch.zeros(
                        batch_size,
                        seqlen,
                        local_group_student_log_probs_rmpad.size(-1),
                        device=local_group_student_log_probs_rmpad.device,
                        dtype=local_group_student_log_probs_rmpad.dtype,
                    )
                    full_local_group_student_log_probs[response_position_mask] = local_group_student_log_probs_rmpad
                    if local_group_ref_log_probs_rmpad is not None:
                        full_local_group_ref_log_probs = torch.zeros(
                            batch_size,
                            seqlen,
                            local_group_ref_log_probs_rmpad.size(-1),
                            device=local_group_ref_log_probs_rmpad.device,
                            dtype=local_group_ref_log_probs_rmpad.dtype,
                        )
                        full_local_group_ref_log_probs[response_position_mask] = local_group_ref_log_probs_rmpad

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                if compute_local_group_stats or gather_local_group_ref_log_probs:
                    if gather_local_group_ref_log_probs:
                        local_group_topk_ids = micro_batch["opd_local_topk_ids"]
                    else:
                        local_group_topk_ids = full_local_group_topk_ids[:, -response_length - 1 : -1]
                    local_group_student_log_probs = full_local_group_student_log_probs[:, -response_length - 1 : -1]
                    if full_local_group_ref_log_probs is not None:
                        local_group_ref_log_probs = full_local_group_ref_log_probs[:, -response_length - 1 : -1]
                    else:
                        local_group_ref_log_probs = None
                else:
                    local_group_topk_ids = None
                    local_group_student_log_probs = None
                    local_group_ref_log_probs = None

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])

                    local_group_topk_ids = None
                    local_group_student_log_probs = None
                    local_group_ref_log_probs = None
                    if compute_local_group_stats or gather_local_group_ref_log_probs:
                        if local_group_topk > logits.shape[-1]:
                            raise ValueError(
                                f"local_group_topk ({local_group_topk}) cannot exceed vocab size ({logits.shape[-1]})"
                            )
                        log_norm = torch.logsumexp(logits, dim=-1, keepdim=True)
                        if compute_local_group_stats:
                            topk_logits, local_group_topk_ids = torch.topk(logits, k=local_group_topk, dim=-1)
                            local_group_student_log_probs = topk_logits - log_norm
                        if gather_local_group_ref_log_probs:
                            topk_ids = micro_batch["opd_local_topk_ids"]
                            local_group_topk_ids = topk_ids
                            local_group_ref_log_probs = torch.gather(logits, dim=-1, index=topk_ids) - log_norm
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            if compute_local_group_stats or gather_local_group_ref_log_probs:
                return entropy, log_probs, local_group_topk_ids, (
                    local_group_student_log_probs if compute_local_group_stats else local_group_ref_log_probs
                )

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        compute_local_group_stats = data.meta_info.get("compute_local_group_stats", False)
        gather_local_group_ref_log_probs = data.meta_info.get("gather_local_group_ref_log_probs", False)
        local_group_topk = data.meta_info.get("local_group_topk")
        if compute_local_group_stats or gather_local_group_ref_log_probs:
            if local_group_topk is None:
                policy_loss_cfg = getattr(self.config, "policy_loss", None)
                if policy_loss_cfg is None:
                    raise ValueError("local_group_topk must be provided when policy_loss config is unavailable")
                local_group_topk = policy_loss_cfg.local_group_topk
        if (compute_local_group_stats or gather_local_group_ref_log_probs) and self.use_fused_kernels:
            raise NotImplementedError("local-group OPD statistics are not supported with fused kernels yet")
        if (compute_local_group_stats or gather_local_group_ref_log_probs) and self.use_ulysses_sp:
            raise NotImplementedError("local-group OPD statistics are not supported with ulysses SP yet")
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys() # handle when ref input_ids is different from actor input_ids
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        if gather_local_group_ref_log_probs:
            select_keys.append("opd_local_topk_ids")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        local_group_ids_lst = []
        local_group_vals_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    compute_local_group_stats=compute_local_group_stats,
                    gather_local_group_ref_log_probs=gather_local_group_ref_log_probs,
                    local_group_topk=local_group_topk,
                )
            if compute_local_group_stats or gather_local_group_ref_log_probs:
                entropy, log_probs, local_group_topk_ids, local_group_log_probs = outputs
                local_group_ids_lst.append(local_group_topk_ids)
                local_group_vals_lst.append(local_group_log_probs)
            else:
                entropy, log_probs = outputs
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if compute_local_group_stats or gather_local_group_ref_log_probs:
                local_group_ids = restore_dynamic_batch(torch.concat(local_group_ids_lst, dim=0), batch_idx_list)
                local_group_vals = restore_dynamic_batch(torch.concat(local_group_vals_lst, dim=0), batch_idx_list)
            else:
                local_group_ids = None
                local_group_vals = None
        else:
            if compute_local_group_stats or gather_local_group_ref_log_probs:
                local_group_ids = torch.concat(local_group_ids_lst, dim=0)
                local_group_vals = torch.concat(local_group_vals_lst, dim=0)
            else:
                local_group_ids = None
                local_group_vals = None

        if compute_local_group_stats or gather_local_group_ref_log_probs:
            return log_probs, entropys, local_group_ids, local_group_vals

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
         # Include base model log probs for corrected reward computation
        # These are computed when actor_rollout_ref.model.base_model_path and
        # actor_rollout_ref.ref.model.base_model_path are both specified
        if "base_log_prob" in data.batch.keys():
            select_keys.append("base_log_prob")
        if "base_ref_log_prob" in data.batch.keys():
            select_keys.append("base_ref_log_prob")
        # Include ref_log_prob for only_reverse_kl_advantages mode
        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in data.batch.keys():
            if "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")
        if (
            self.config.policy_loss.use_local_group_baseline
            or self.config.policy_loss.use_suffix_group_baseline
            or self.config.policy_loss.use_aopd
        ):
            for key in (
                "opd_local_topk_ids",
                "opd_local_student_log_probs",
                "opd_local_ref_log_probs",
            ):
                if key in data.batch.keys() and key not in select_keys:
                    select_keys.append(key)
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include opd_teacher for multi-teacher distillation
        if "opd_teacher" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("opd_teacher")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        adaptive_reverse_kl_means = []
        validate_suffix_group_baseline_config(self.config.policy_loss)
        validate_causal_opd_extensions(self.config.policy_loss)
        validate_aopd_config(self.config.policy_loss)
        validate_reopold_config(self.config.policy_loss)
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    reward_advantages = model_inputs["advantages"]
                    advantages = reward_advantages

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    if self.config.policy_loss.use_reopold and self.config.policy_loss.reopold_mask_mode == "entropy":
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # only use reverse KL for advantages if only_reverse_kl_advantages is True
                    if self.config.policy_loss.only_reverse_kl_advantages:
                        # Corrected reverse KL with base model normalization if base log probs are available
                        # Formula: (log_prob_actor - log_prob_ref) - (log_prob_actor_base - log_prob_ref_base)
                        # This removes the base model bias from both actor and ref models
                        if "base_log_prob" in model_inputs and "base_ref_log_prob" in model_inputs:
                            lambda_vals = self.config.policy_loss.lambda_vals

                            if self.config.policy_loss.multi_teacher_distill:
                                #### multi-teacher distillation ####
                                if "opd_teacher" in model_inputs:
                                    opd_teacher = model_inputs["opd_teacher"]
                                    batch_size = old_log_prob.shape[0]

                                    reverse_kl = torch.zeros_like(old_log_prob)

                                    for i in range(batch_size):
                                        teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                                        # TODO: need to improve the logic here
                                        if teacher_type == "math":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        elif teacher_type == "code":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["base_ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        else:
                                            reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                else:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                #### multi-teacher distillation ####
                            else:
                                #### single-teacher distillation ####
                                reverse_kl = old_log_prob - model_inputs["base_log_prob"]
                                reward_correction = model_inputs["ref_log_prob"] - model_inputs["base_log_prob"]

                                if lambda_vals == 1.0:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                else:
                                    reverse_kl = reverse_kl - reward_correction * lambda_vals
                                #### single-teacher distillation ####
                        else:
                            # Standard reverse KL: log(π_actor / π_ref) = log_prob_actor - log_prob_ref
                            reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                        token_reverse_kl = reverse_kl
                        causal_reverse_kl = token_reverse_kl
                        if self.config.policy_loss.causal_opd:
                            causal_reverse_kl = discounted_reverse_cumsum(
                                token_reverse_kl, response_mask, self.config.policy_loss.causal_gamma
                            )
                            if self.config.policy_loss.normalize_causal_by_discounted_weight:
                                causal_reverse_kl = normalize_discounted_suffix_values(
                                    causal_reverse_kl, response_mask, self.config.policy_loss.causal_gamma
                                )
                        sampled_reward = -(causal_reverse_kl)
                        reopold_mask = None
                        if self.config.policy_loss.use_reopold:
                            sampled_reward, reopold_mask = apply_reopold_signal_processor(
                                sampled_reward=sampled_reward,
                                entropy=entropy,
                                response_mask=response_mask,
                                clip_min=self.config.policy_loss.reopold_reward_clip_min,
                                clip_max=self.config.policy_loss.reopold_reward_clip_max,
                                mask_mode=self.config.policy_loss.reopold_mask_mode,
                                entropy_threshold=self.config.policy_loss.reopold_entropy_threshold,
                                reward_quantile=self.config.policy_loss.reopold_reward_quantile,
                            )
                        opd_advantages = sampled_reward
                        if self.config.policy_loss.use_local_future_correction:
                            opd_advantages = combine_local_and_causal_opd(
                                token_reverse_kl,
                                response_mask,
                                gamma=self.config.policy_loss.causal_gamma,
                                alpha=self.config.policy_loss.future_correction_alpha,
                            )
                        local_group_mean = None
                        local_group_covered_mass = None
                        if (
                            self.config.policy_loss.use_local_group_baseline
                            or self.config.policy_loss.use_suffix_group_baseline
                            or self.config.policy_loss.use_aopd
                        ):
                            required_keys = (
                                "opd_local_student_log_probs",
                                "opd_local_ref_log_probs",
                            )
                            for key in required_keys:
                                if key not in model_inputs:
                                    raise ValueError(f"Missing {key} for local-group OPD baseline")
                            local_group_mean, local_group_covered_mass = compute_local_group_baseline(
                                model_inputs["opd_local_student_log_probs"],
                                model_inputs["opd_local_ref_log_probs"],
                                response_mask,
                                self.config.policy_loss.local_group_renorm,
                                self.config.policy_loss.local_group_eps,
                            )
                        aopd_local_kl = None
                        if self.config.policy_loss.use_aopd:
                            aopd_local_kl = compute_aopd_local_kl(
                                model_inputs["opd_local_student_log_probs"],
                                model_inputs["opd_local_ref_log_probs"],
                                response_mask,
                            )
                        suffix_group_baseline = None
                        if self.config.policy_loss.use_suffix_group_baseline:
                            opd_advantages, suffix_group_baseline = apply_suffix_group_baseline(
                                sampled_reward=sampled_reward,
                                token_baseline=local_group_mean,
                                response_mask=response_mask,
                                gamma=self.config.policy_loss.causal_gamma,
                                alpha=self.config.policy_loss.suffix_group_baseline_alpha,
                            )
                        elif self.config.policy_loss.use_local_group_baseline:
                            opd_advantages = sampled_reward - local_group_mean
                        if self.config.policy_loss.use_aopd:
                            opd_advantages = apply_aopd_piecewise_token_objective(
                                sampled_reward=opd_advantages,
                                local_kl=aopd_local_kl,
                                response_mask=response_mask,
                                positive_only_floor=self.config.policy_loss.aopd_positive_only_floor,
                                nonpositive_coef=self.config.policy_loss.aopd_nonpositive_coef,
                            )
                        opd_scale = torch.ones_like(opd_advantages[:, :1])
                        if self.config.policy_loss.scale_reverse_kl_by_abs_mean:
                            opd_scale = masked_abs_mean_per_sample(opd_advantages, response_mask)
                            opd_advantages = scale_by_masked_abs_mean_per_sample(opd_advantages, response_mask)
                        if self.config.policy_loss.apply_softsign_to_opd_advantages:
                            opd_advantages = opd_advantages / (1.0 + opd_advantages.abs())
                        opd_advantage_coef_effective = float(self.config.policy_loss.opd_advantage_coef)
                        if self.config.policy_loss.adaptive_opd_coef:
                            opd_advantage_coef_effective = self._adaptive_opd_advantage_coef
                        scaled_reward_advantages, scaled_opd_advantages, advantages = mix_reward_and_opd_advantages(
                            reward_advantages=reward_advantages,
                            opd_advantages=opd_advantages,
                            reward_advantage_coef=self.config.policy_loss.reward_advantage_coef,
                            opd_advantage_coef=opd_advantage_coef_effective,
                            use_reward_advantage=self.config.policy_loss.use_reward_advantage,
                        )
                        micro_batch_metrics["actor/opd/token_reverse_kl_mean"] = (
                            verl_F.masked_mean(token_reverse_kl.detach(), response_mask).item()
                        )
                        micro_batch_metrics["actor/opd/token_reverse_kl_abs_mean"] = (
                            verl_F.masked_mean(token_reverse_kl.detach().abs(), response_mask).item()
                        )
                        micro_batch_metrics["actor/opd/causal_reverse_kl_mean"] = (
                            verl_F.masked_mean(causal_reverse_kl.detach(), response_mask).item()
                        )
                        micro_batch_metrics["actor/opd/causal_reverse_kl_abs_mean"] = (
                            verl_F.masked_mean(causal_reverse_kl.detach().abs(), response_mask).item()
                        )
                        micro_batch_metrics["actor/opd/causal_sampled_reward_mean"] = (
                            verl_F.masked_mean(sampled_reward.detach(), response_mask).item()
                        )
                        micro_batch_metrics["actor/opd/causal_sampled_reward_abs_mean"] = (
                            verl_F.masked_mean(sampled_reward.detach().abs(), response_mask).item()
                        )
                        if self.config.policy_loss.use_reopold:
                            micro_batch_metrics["actor/opd/reopold_mask_ratio"] = (
                                reopold_mask.sum() / response_mask.sum().clamp_min(1.0)
                            ).item()
                            micro_batch_metrics["actor/opd/reopold_reward_mean"] = (
                                verl_F.masked_mean(sampled_reward.detach(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/reopold_reward_abs_mean"] = (
                                verl_F.masked_mean(sampled_reward.detach().abs(), response_mask).item()
                            )
                        if self.config.policy_loss.normalize_causal_by_discounted_weight:
                            micro_batch_metrics["actor/opd/normalized_causal"] = 1.0
                        if self.config.policy_loss.use_local_future_correction:
                            micro_batch_metrics["actor/opd/future_correction_alpha"] = (
                                self.config.policy_loss.future_correction_alpha
                            )
                        micro_batch_metrics["actor/opd/optimized_adv_mean"] = (
                            verl_F.masked_mean(opd_advantages.detach(), response_mask).item()
                        )
                        micro_batch_metrics["actor/opd/optimized_adv_abs_mean"] = (
                            verl_F.masked_mean(opd_advantages.detach().abs(), response_mask).item()
                        )
                        if (
                            self.config.policy_loss.use_local_group_baseline
                            or self.config.policy_loss.use_suffix_group_baseline
                            or self.config.policy_loss.use_aopd
                        ):
                            micro_batch_metrics["actor/opd/local_group_mean"] = (
                                verl_F.masked_mean(local_group_mean.detach(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/local_group_abs_mean"] = (
                                verl_F.masked_mean(local_group_mean.detach().abs(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/local_group_covered_mass"] = (
                                verl_F.masked_mean(local_group_covered_mass.detach(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/local_adv_mean"] = (
                                verl_F.masked_mean(opd_advantages.detach(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/local_adv_abs_mean"] = (
                                verl_F.masked_mean(opd_advantages.detach().abs(), response_mask).item()
                            )
                        if self.config.policy_loss.use_aopd:
                            aopd_positive_mask = ((sampled_reward > self.config.policy_loss.aopd_positive_only_floor) * response_mask).to(
                                dtype=sampled_reward.dtype
                            )
                            aopd_nonpositive_mask = response_mask.to(dtype=sampled_reward.dtype) - aopd_positive_mask
                            micro_batch_metrics["actor/opd/aopd_positive_ratio"] = (
                                aopd_positive_mask.sum() / response_mask.sum().clamp_min(1.0)
                            ).item()
                            micro_batch_metrics["actor/opd/aopd_positive_adv_mean"] = (
                                ((sampled_reward.detach() * aopd_positive_mask).sum())
                                / aopd_positive_mask.sum().clamp_min(1.0)
                            ).item()
                            micro_batch_metrics["actor/opd/aopd_nonpositive_kl_mean"] = (
                                ((aopd_local_kl.detach() * aopd_nonpositive_mask).sum())
                                / aopd_nonpositive_mask.sum().clamp_min(1.0)
                            ).item()
                            micro_batch_metrics["actor/opd/aopd_nonpositive_kl_abs_mean"] = (
                                ((aopd_local_kl.detach().abs() * aopd_nonpositive_mask).sum())
                                / aopd_nonpositive_mask.sum().clamp_min(1.0)
                            ).item()
                            micro_batch_metrics["actor/opd/aopd_final_adv_mean"] = (
                                verl_F.masked_mean(opd_advantages.detach(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/aopd_final_adv_abs_mean"] = (
                                verl_F.masked_mean(opd_advantages.detach().abs(), response_mask).item()
                            )
                        if self.config.policy_loss.use_suffix_group_baseline:
                            micro_batch_metrics["actor/opd/suffix_group_mean"] = (
                                verl_F.masked_mean(suffix_group_baseline.detach(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/suffix_group_abs_mean"] = (
                                verl_F.masked_mean(suffix_group_baseline.detach().abs(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/suffix_group_alpha"] = (
                                self.config.policy_loss.suffix_group_baseline_alpha
                            )
                            micro_batch_metrics["actor/opd/suffix_adv_mean"] = (
                                verl_F.masked_mean(opd_advantages.detach(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/suffix_adv_abs_mean"] = (
                                verl_F.masked_mean(opd_advantages.detach().abs(), response_mask).item()
                            )
                        reverse_kl_mean = verl_F.masked_mean(reverse_kl.detach(), response_mask).item()
                        micro_batch_metrics["actor/opd/reverse_kl_mean"] = reverse_kl_mean
                        micro_batch_metrics["actor/opd/reverse_kl_abs_mean"] = (
                            verl_F.masked_mean(reverse_kl.detach().abs(), response_mask).item()
                        )
                        micro_batch_metrics["actor/opd/opd_adv_abs_mean"] = (
                            verl_F.masked_mean(scaled_opd_advantages.detach().abs(), response_mask).item()
                        )
                        micro_batch_metrics["actor/opd/opd_adv_mean"] = (
                            verl_F.masked_mean(scaled_opd_advantages.detach(), response_mask).item()
                        )
                        micro_batch_metrics["actor/opd/opd_scale_abs_mean"] = opd_scale.detach().mean().item()
                        micro_batch_metrics["actor/opd/opd_advantage_coef"] = opd_advantage_coef_effective
                        if self.config.policy_loss.adaptive_opd_coef:
                            adaptive_reverse_kl_means.append(reverse_kl_mean)
                        if self.config.policy_loss.use_reward_advantage:
                            micro_batch_metrics["actor/opd/reward_adv_mean"] = (
                                verl_F.masked_mean(reward_advantages.detach(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/reward_adv_abs_mean"] = (
                                verl_F.masked_mean(reward_advantages.detach().abs(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/reward_adv_scaled_mean"] = (
                                verl_F.masked_mean(scaled_reward_advantages.detach(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/reward_adv_scaled_abs_mean"] = (
                                verl_F.masked_mean(scaled_reward_advantages.detach().abs(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/reward_adv_coef"] = (
                                self.config.policy_loss.reward_advantage_coef
                            )
                            micro_batch_metrics["actor/opd/final_adv_mean"] = (
                                verl_F.masked_mean(advantages.detach(), response_mask).item()
                            )
                            micro_batch_metrics["actor/opd/final_adv_abs_mean"] = (
                                verl_F.masked_mean(advantages.detach().abs(), response_mask).item()
                            )
                   
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        if self.config.policy_loss.adaptive_opd_coef and adaptive_reverse_kl_means:
            observed_reverse_kl = sum(adaptive_reverse_kl_means) / len(adaptive_reverse_kl_means)
            next_coef, next_ema_kl = compute_adaptive_opd_coef_update(
                current_coef=self._adaptive_opd_advantage_coef,
                current_ema_kl=self._adaptive_opd_advantage_coef_ema_kl,
                observed_reverse_kl=observed_reverse_kl,
                target_reverse_kl=float(self.config.policy_loss.target_reverse_kl),
                ema_decay=float(self.config.policy_loss.opd_coef_ema_decay),
                update_rate=float(self.config.policy_loss.opd_coef_update_rate),
                coef_min=float(self.config.policy_loss.opd_advantage_coef_min),
                coef_max=float(self.config.policy_loss.opd_advantage_coef_max),
            )
            self._adaptive_opd_advantage_coef = next_coef
            self._adaptive_opd_advantage_coef_ema_kl = next_ema_kl
            append_to_dict(
                metrics,
                {
                    "actor/opd/opd_advantage_coef_ema_kl": next_ema_kl,
                    "actor/opd/opd_advantage_coef_target_kl": float(self.config.policy_loss.target_reverse_kl),
                },
            )
        self.actor_optimizer.zero_grad()
        return metrics
