# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = ["PolicyLossConfig", "ActorConfig", "FSDPActorConfig", "McoreActorConfig"]


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
        lambda_vals (float): Lambda coefficient for on-policy distillation.
            When lambda_vals=1.0, uses standard OPD.
        causal_opd (bool): Whether to replace token-local OPD log-ratio advantages with a causal suffix return.
        causal_gamma (float): Discount factor for causal OPD suffix return.
        normalize_causal_by_discounted_weight (bool): Whether to divide causal OPD suffix returns by the
            masked discounted suffix weight so each token sees an average future discrepancy instead of a sum.
        use_local_future_correction (bool): Whether to keep the token-local OPD term and add only a scaled
            future-token causal correction on top of it.
        future_correction_alpha (float): Multiplier applied to the future-token causal correction term.
        use_reward_advantage (bool): Whether to add trainer-computed reward advantages back into OPD.
        reward_advantage_coef (float): Multiplier applied to trainer-computed reward advantages before mixing.
        opd_advantage_coef (float): Multiplier applied to OPD advantages before final mixing.
        adaptive_opd_coef (bool): Whether to update the OPD multiplier from step-level reverse-KL feedback.
        target_reverse_kl (float): Reverse-KL target used by the adaptive OPD multiplier update.
        opd_coef_ema_decay (float): EMA decay used to smooth observed reverse-KL before multiplier updates.
        opd_coef_update_rate (float): Exponential update rate used by the adaptive OPD multiplier.
        opd_advantage_coef_min (float): Lower bound for the adaptive OPD multiplier.
        opd_advantage_coef_max (float): Upper bound for the adaptive OPD multiplier.
        scale_reverse_kl_by_abs_mean (bool): Whether to scale per-sample OPD advantages by masked abs mean.
        apply_softsign_to_opd_advantages (bool): Whether to apply softsign to OPD advantages before mixing.
        use_local_group_baseline (bool): Whether to subtract a detached student top-k local baseline from
            sampled-token OPD advantages. Only applies when only_reverse_kl_advantages=True.
        local_group_topk (int): Number of student top-k tokens used to build the detached local baseline.
        local_group_renorm (bool): Whether to renormalize the student top-k mass before averaging local-group
            rewards. When false, uses the truncated top-k expectation without conditional renormalization.
        local_group_eps (float): Epsilon used when normalizing the covered top-k mass for the local baseline.
        use_aopd (bool): Whether to enable asymmetric OPD in the reverse-KL advantage path.
        aopd_nonpositive_mode (str): Non-positive token handling mode for AOPD.
        aopd_teacher_topk (int): Teacher top-k support size used by AOPD local correction.
        aopd_nonpositive_coef (float): Multiplier applied to the non-positive AOPD correction.
        aopd_positive_only_floor (float): Threshold separating positive and non-positive token regions in AOPD.
        use_reopold (bool): Whether to enable REOPOLD fixed-reward clipping and masking on the OPD path.
        reopold_reward_clip_min (float): Lower clipping bound for REOPOLD fixed token rewards.
        reopold_reward_clip_max (float): Upper clipping bound for REOPOLD fixed token rewards.
        reopold_mask_mode (str): REOPOLD token mask mode. Options: 'none', 'entropy', 'abs_quantile'.
        reopold_entropy_threshold (float): Entropy threshold used by REOPOLD entropy masking.
        reopold_reward_quantile (float): Absolute reward quantile used by REOPOLD quantile masking.
        use_suffix_group_baseline (bool): Whether to aggregate the detached token-level top-k baseline into
            a discounted suffix baseline before subtracting it from causal OPD returns.
        suffix_group_baseline_alpha (float): Multiplier applied to the suffix baseline before subtraction.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
    only_reverse_kl_advantages: bool = False
    lambda_vals: float = 1.0
    multi_teacher_distill: bool = False
    causal_opd: bool = False
    causal_gamma: float = 0.0
    normalize_causal_by_discounted_weight: bool = False
    use_local_future_correction: bool = False
    future_correction_alpha: float = 1.0
    use_reward_advantage: bool = False
    reward_advantage_coef: float = 1.0
    opd_advantage_coef: float = 1.0
    adaptive_opd_coef: bool = False
    target_reverse_kl: float = 0.05
    opd_coef_ema_decay: float = 0.9
    opd_coef_update_rate: float = 0.01
    opd_advantage_coef_min: float = 0.1
    opd_advantage_coef_max: float = 4.0
    scale_reverse_kl_by_abs_mean: bool = False
    apply_softsign_to_opd_advantages: bool = False
    use_local_group_baseline: bool = False
    local_group_topk: int = 8
    local_group_renorm: bool = True
    local_group_eps: float = 1e-6
    use_aopd: bool = False
    aopd_nonpositive_mode: str = "local_topk_kl"
    aopd_teacher_topk: int = 8
    aopd_nonpositive_coef: float = 1.0
    aopd_positive_only_floor: float = 0.0
    use_reopold: bool = False
    reopold_reward_clip_min: float = -2.0
    reopold_reward_clip_max: float = 2.0
    reopold_mask_mode: str = "entropy"
    reopold_entropy_threshold: float = 0.0
    reopold_reward_quantile: float = 0.5
    use_suffix_group_baseline: bool = False
    suffix_group_baseline_alpha: float = 1.0


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        entropy_coeff (float): Entropy coefficient for regularization.
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0
    use_kl_loss: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    data_loader_seed = 1
    rollout_n: int = 1  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        data_loader_seed (Optional[int]): Seed for data loader. If None, uses global seed.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    data_loader_seed: Optional[int] = None
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
