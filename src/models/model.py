"""Decoder-only language model built from selective SSM blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ssm import SSMState, SelectiveSSM


@dataclass
class ModelConfig:
    """Configuration for :class:`SSMLanguageModel`.

    ``n_layer`` is accepted as an alias for ``n_layers`` to make small
    experiment configurations easy to port from common decoder-only models.
    ``d_inner`` defaults to ``expand * d_model``.
    """

    vocab_size: int
    d_model: int
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    d_inner: Optional[int] = None
    dt_rank: int | str = "auto"
    norm_eps: float = 1e-5
    dropout: float = 0.0
    dt_min: float = 1e-3
    dt_max: float = 1.0
    tie_embeddings: bool = True
    ignore_index: int = -100
    max_seq_len: int = 4096
    pad_token_id: Optional[int] = None
    n_layer: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_layer is not None:
            self.n_layers = self.n_layer
        self.n_layer = self.n_layers
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.d_state <= 0 or self.d_conv <= 0 or self.expand <= 0:
            raise ValueError("d_state, d_conv, and expand must be positive")
        if self.d_inner is None:
            self.d_inner = self.expand * self.d_model
        if self.d_inner <= 0:
            raise ValueError("d_inner must be positive")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.pad_token_id is not None and not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be a valid vocabulary id")


class RMSNorm(nn.Module):
    """Root-mean-square normalization without a learned bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)


class GatedMLP(nn.Module):
    """SwiGLU-style feed-forward network."""

    def __init__(self, d_model: int, d_inner: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(value * F.silu(gate)))


class SSMBlock(nn.Module):
    """Pre-norm residual selective mixer followed by a gated MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.mixer = SelectiveSSM(
            d_model=config.d_model,
            d_state=config.d_state,
            d_inner=config.d_inner,
            dt_rank=config.dt_rank,
            d_conv=config.d_conv,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
        )
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = GatedMLP(config.d_model, config.d_inner, config.dropout)

    def forward(
        self,
        x: Tensor,
        state: Optional[SSMState] = None,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, SSMState]:
        mixed, next_state = self.mixer(
            self.norm1(x), state=state, return_state=True
        )
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        return (x, next_state) if return_state else x

    def step(self, x: Tensor, state: Optional[SSMState]) -> tuple[Tensor, SSMState]:
        mixed, next_state = self.mixer.step(self.norm1(x), state)
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        return x, next_state


@dataclass
class ModelCache:
    """Per-layer recurrent states used by incremental decoding."""

    states: tuple[SSMState, ...]

    def __len__(self) -> int:
        return len(self.states)


@dataclass
class CausalLMOutput:
    """Convenient output object that also supports tuple unpacking."""

    logits: Tensor
    loss: Optional[Tensor] = None
    cache: Optional[ModelCache] = None

    def __iter__(self):
        yield self.logits
        if self.loss is not None:
            yield self.loss
        elif self.cache is not None:
            yield self.cache

    def __getitem__(self, index: int):
        values = [self.logits]
        if self.loss is not None:
            values.append(self.loss)
        elif self.cache is not None:
            values.append(self.cache)
        return values[index]


class SSMLanguageModel(nn.Module):
    """Decoder-only autoregressive LM whose mixer contains no attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(SSMBlock(config) for _ in range(config.n_layers))
        self.norm_f = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
        self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        """Number of unique parameters, respecting tied embedding weights."""

        return self.num_parameters()

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of scalar parameters in the model."""

        parameters: Iterable[nn.Parameter] = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Method alias for :meth:`num_parameters`."""

        return self.num_parameters(trainable_only=trainable_only)

    def init_cache(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> ModelCache:
        """Create zero recurrent state for every SSM block."""

        if device is None:
            device = self.tok_embeddings.weight.device
        if dtype is None:
            dtype = self.tok_embeddings.weight.dtype
        return ModelCache(
            tuple(
                block.mixer.initial_state(batch_size, device=device, dtype=dtype)
                for block in self.blocks
            )
        )

    def _coerce_cache(
        self, cache: Optional[ModelCache | tuple[SSMState, ...] | list[SSMState]], batch_size: int
    ) -> Optional[ModelCache]:
        if cache is None:
            return None
        if isinstance(cache, ModelCache):
            result = cache
        elif isinstance(cache, (tuple, list)):
            result = ModelCache(tuple(cache))
        else:
            raise TypeError("cache must be a ModelCache or a sequence of SSMState objects")
        if len(result.states) != len(self.blocks):
            raise ValueError(f"cache has {len(result.states)} layers, expected {len(self.blocks)}")
        for state in result.states:
            if state.conv_state.size(0) != batch_size:
                raise ValueError("cache batch size does not match input_ids")
        return result

    def _loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if targets.ndim != 2 or targets.shape[:2] != logits.shape[:2]:
            raise ValueError(
                f"targets must have shape {tuple(logits.shape[:2])}, got {tuple(targets.shape)}"
            )
        if targets.dtype != torch.long:
            targets = targets.long()
        # At position t, the decoder predicts token t+1.  The final logit has
        # no next-token target and is intentionally excluded.
        if logits.size(1) < 2:
            return logits.sum() * 0.0
        shifted_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
        shifted_targets = targets[:, 1:].contiguous().view(-1)
        if not torch.any(shifted_targets != self.config.ignore_index):
            return shifted_logits.sum() * 0.0
        return F.cross_entropy(
            shifted_logits,
            shifted_targets,
            ignore_index=self.config.ignore_index,
        )

    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
        *,
        labels: Optional[Tensor] = None,
        cache: Optional[ModelCache | tuple[SSMState, ...] | list[SSMState]] = None,
        use_cache: bool = False,
    ) -> Tensor | CausalLMOutput:
        """Compute causal logits and, optionally, aligned next-token loss.

        With no ``targets`` and ``use_cache=False`` this returns a logits
        tensor directly.  Supplying targets returns ``CausalLMOutput`` with a
        ``loss`` field; ``use_cache=True`` also returns the updated cache.
        """

        if labels is not None:
            if targets is not None:
                raise ValueError("provide only one of targets and labels")
            targets = labels
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, time], got {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if input_ids.size(1) == 0:
            raise ValueError("input_ids cannot be empty")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contains a token outside the configured vocabulary")

        cache = self._coerce_cache(cache, input_ids.size(0))
        if cache is not None and input_ids.size(1) == 1:
            step_logits, next_cache = self.step(input_ids[:, 0], cache)
            logits = step_logits.unsqueeze(1)
        else:
            x = self.tok_embeddings(input_ids)
            states = cache.states if cache is not None else (None,) * len(self.blocks)
            next_states = []
            for block, state in zip(self.blocks, states):
                x, next_state = block(x, state=state, return_state=True)
                next_states.append(next_state)
            logits = F.linear(self.norm_f(x), self.lm_head.weight)
            next_cache = ModelCache(tuple(next_states))

        loss = self._loss(logits, targets) if targets is not None else None
        if loss is None and not use_cache:
            return logits
        return CausalLMOutput(logits=logits, loss=loss, cache=next_cache if use_cache else None)

    def step(self, input_ids: Tensor, cache: Optional[ModelCache] = None) -> tuple[Tensor, ModelCache]:
        """Run one token through every block using recurrent state updates."""

        if input_ids.ndim == 2:
            if input_ids.size(1) != 1:
                raise ValueError("step accepts [batch] or [batch, 1] input_ids")
            input_ids = input_ids[:, 0]
        if input_ids.ndim != 1:
            raise ValueError(f"step expects [batch] input_ids, got {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contains a token outside the configured vocabulary")
        cache = self._coerce_cache(cache, input_ids.size(0))
        if cache is None:
            cache = self.init_cache(input_ids.size(0), device=input_ids.device)

        x = self.tok_embeddings(input_ids)
        next_states = []
        for block, state in zip(self.blocks, cache.states):
            x, next_state = block.step(x, state)
            next_states.append(next_state)
        logits = F.linear(self.norm_f(x), self.lm_head.weight)
        return logits, ModelCache(tuple(next_states))

    @staticmethod
    def _filter_logits(logits: Tensor, top_k: Optional[int], top_p: Optional[float]) -> Tensor:
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if top_p is not None:
            if not 0.0 < top_p <= 1.0:
                raise ValueError("top_p must be in (0, 1]")
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """Autoregressively append tokens, optionally using filtered sampling.

        ``do_sample=False`` or ``temperature=0`` selects argmax at every
        position, making greedy generation deterministic.  Sampling uses
        temperature scaling followed by top-k and/or nucleus (top-p) filtering.
        """

        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must be a non-empty [batch, time] tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id must be a valid vocabulary id")
        generated = input_ids.long().clone()
        if max_new_tokens == 0:
            return generated

        was_training = self.training
        self.eval()
        try:
            initial = self(generated, use_cache=True)
            if not isinstance(initial, CausalLMOutput) or initial.cache is None:
                raise RuntimeError("internal cache creation failed")
            logits = initial.logits[:, -1]
            cache = initial.cache
            finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)

            for _ in range(max_new_tokens):
                greedy = not do_sample or temperature == 0.0
                if greedy:
                    next_token = logits.argmax(dim=-1)
                else:
                    filtered = self._filter_logits(logits / temperature, top_k, top_p)
                    next_token = torch.multinomial(filtered.softmax(dim=-1), num_samples=1).squeeze(-1)
                if eos_token_id is not None:
                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                generated = torch.cat((generated, next_token[:, None]), dim=1)
                if eos_token_id is not None:
                    finished |= next_token == eos_token_id
                    if torch.all(finished):
                        break
                logits, cache = self.step(next_token, cache)
        finally:
            self.train(was_training)
        return generated


# Short aliases keep the module convenient for small experiments while the
# descriptive class name remains the canonical API.
Model = SSMLanguageModel
SSMModel = SSMLanguageModel
