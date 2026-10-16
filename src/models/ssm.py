"""A small, explicit selective state-space mixer.

The mixer uses a diagonal continuous-time state matrix.  For a token ``t``
and channel ``i`` its state update is

``h[t, i] = exp(delta[t, i] * A[i]) * h[t - 1, i]``
``          + delta[t, i] * B[t] * u[t, i]``
``y[t, i] = sum_n(C[t, n] * h[t, i, n]) + D[i] * u[t, i]``

where ``A`` is strictly negative and ``delta`` is positive.  Consequently,
the homogeneous dynamics are contractive instead of relying on an unstable
unconstrained recurrent matrix.  ``B``, ``C``, and ``delta`` are selected
from the current token, which gives the mixer input-dependent memory.

The full-sequence path performs the recurrence for every time step in one
batched call.  ``step`` performs the same update for exactly one token and
keeps both the causal-convolution history and SSM state for generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class SSMState:
    """Persistent state required by one :class:`SelectiveSSM` layer."""

    conv_state: Tensor
    ssm_state: Tensor


class SelectiveSSM(nn.Module):
    """Mamba-inspired selective SSM without attention.

    Parameters
    ----------
    d_model:
        Input and output width.
    d_state:
        Number of diagonal state values per inner channel.
    d_inner:
        Width of the input-dependent SSM branch.
    dt_rank:
        Rank used to produce the per-channel time step.  ``"auto"`` uses
        ``ceil(d_model / 16)``.
    d_conv:
        Width of the causal depthwise convolution preceding the SSM.
    dt_min, dt_max:
        Bounds for the positive time step.  Bounding it prevents excessively
        large input injections while ``A < 0`` guarantees stable decay.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_inner: Optional[int] = None,
        dt_rank: int | str = "auto",
        d_conv: int = 4,
        dt_min: float = 1e-3,
        dt_max: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_state <= 0:
            raise ValueError("d_state must be positive")
        if d_inner is None:
            d_inner = 2 * d_model
        if d_inner <= 0:
            raise ValueError("d_inner must be positive")
        if dt_rank == "auto":
            dt_rank = max(1, math.ceil(d_model / 16))
        if not isinstance(dt_rank, int) or dt_rank <= 0:
            raise ValueError("dt_rank must be a positive integer or 'auto'")
        if d_conv <= 0:
            raise ValueError("d_conv must be positive")
        if not 0.0 < dt_min <= dt_max:
            raise ValueError("dt_min and dt_max must satisfy 0 < dt_min <= dt_max")

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model)

        # A = -exp(A_log) is strictly negative for every channel/state pair.
        # Initial time constants cover a useful range without requiring a
        # matrix exponential or a learned unconstrained recurrent matrix.
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(a.log().repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.conv1d.weight, std=0.02)
        nn.init.zeros_(self.conv1d.bias)
        nn.init.normal_(self.x_proj.weight, std=self.d_inner**-0.5)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank**-0.5, self.dt_rank**-0.5)

        # Set dt_proj's bias using the inverse softplus so the initial step
        # is close to a small, positive time constant.
        dt_init = math.sqrt(self.dt_min * self.dt_max)
        dt_init = min(max(dt_init, self.dt_min), self.dt_max)
        inverse_softplus = math.log(math.expm1(dt_init))
        nn.init.constant_(self.dt_proj.bias, inverse_softplus)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> SSMState:
        """Return a zero state for ``batch_size`` independent sequences."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device is None:
            device = self.A_log.device
        if dtype is None:
            dtype = self.A_log.dtype
        return SSMState(
            conv_state=torch.zeros(
                batch_size, self.d_inner, max(0, self.d_conv - 1), device=device, dtype=dtype
            ),
            ssm_state=torch.zeros(
                batch_size, self.d_inner, self.d_state, device=device, dtype=dtype
            ),
        )

    def _check_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, time, d_model], got shape {tuple(x.shape)}")
        if x.size(-1) != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {x.size(-1)}")

    def _coerce_state(self, state: Optional[SSMState], x: Tensor) -> SSMState:
        if state is None:
            return self.initial_state(x.size(0), device=x.device, dtype=x.dtype)
        if not isinstance(state, SSMState):
            raise TypeError("state must be an SSMState returned by initial_state or forward")
        expected_conv = (x.size(0), self.d_inner, max(0, self.d_conv - 1))
        expected_ssm = (x.size(0), self.d_inner, self.d_state)
        if tuple(state.conv_state.shape) != expected_conv:
            raise ValueError(
                f"conv state has shape {tuple(state.conv_state.shape)}, expected {expected_conv}"
            )
        if tuple(state.ssm_state.shape) != expected_ssm:
            raise ValueError(
                f"SSM state has shape {tuple(state.ssm_state.shape)}, expected {expected_ssm}"
            )
        return SSMState(
            conv_state=state.conv_state.to(device=x.device, dtype=x.dtype),
            ssm_state=state.ssm_state.to(device=x.device, dtype=x.dtype),
        )

    def _causal_conv_full(self, x: Tensor, state: SSMState) -> tuple[Tensor, Tensor]:
        """Apply the depthwise causal convolution and return its next history."""

        # x is [B, T, I], while Conv1d consumes [B, I, T].  Prefixing the
        # saved history makes this path exactly equivalent to repeated steps.
        current = x.transpose(1, 2)
        if x.size(1) == 0:
            return x, state.conv_state
        conv_input = torch.cat((state.conv_state, current), dim=-1)
        conv_output = self.conv1d(conv_input).transpose(1, 2)
        if self.d_conv == 1:
            next_conv_state = current.new_empty(current.size(0), current.size(1), 0)
        else:
            next_conv_state = conv_input[:, :, -(self.d_conv - 1) :]
        return conv_output, next_conv_state

    def _causal_conv_step(self, x: Tensor, state: SSMState) -> tuple[Tensor, Tensor]:
        current = x.unsqueeze(-1)
        conv_input = torch.cat((state.conv_state, current), dim=-1)
        conv_output = self.conv1d(conv_input).squeeze(-1)
        if self.d_conv == 1:
            next_conv_state = current.new_empty(current.size(0), current.size(1), 0)
        else:
            next_conv_state = conv_input[:, :, -(self.d_conv - 1) :]
        return conv_output, next_conv_state

    def _select_parameters(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        params = self.x_proj(u)
        dt_raw, b, c = params.split((self.dt_rank, self.d_state, self.d_state), dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw)).clamp(self.dt_min, self.dt_max)
        # Bounding B and C keeps the input-dependent injection/readout well
        # behaved while retaining token-selective dynamics.
        return delta, torch.tanh(b), torch.tanh(c), self._stable_A(u.dtype, u.device)

    def _stable_A(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        return -torch.exp(self.A_log).to(device=device, dtype=dtype)

    def _update_state(
        self,
        u: Tensor,
        z: Tensor,
        delta: Tensor,
        b: Tensor,
        c: Tensor,
        state: Tensor,
        a: Tensor,
    ) -> tuple[Tensor, Tensor]:
        decay = torch.exp(delta.unsqueeze(-1) * a.unsqueeze(0))
        injected = delta.unsqueeze(-1) * b.unsqueeze(1) * u.unsqueeze(-1)
        next_state = decay * state + injected
        readout = torch.einsum("bin,bn->bi", next_state, c)
        y = readout + self.D.to(dtype=u.dtype, device=u.device) * u
        return self.out_proj(y * F.silu(z)), next_state

    def forward(
        self,
        x: Tensor,
        state: Optional[SSMState] = None,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, SSMState]:
        """Run a complete sequence, optionally continuing a previous state."""

        self._check_input(x)
        state = self._coerce_state(state, x)
        if x.size(1) == 0:
            y = x.new_empty(x.shape)
            return (y, state) if return_state else y

        projected = self.in_proj(x)
        x_branch, z = projected.chunk(2, dim=-1)
        conv_output, next_conv_state = self._causal_conv_full(x_branch, state)
        u = F.silu(conv_output)
        delta, b, c, a = self._select_parameters(u)

        # This loop is over time only; all batch items, inner channels, and
        # state values are updated together in each iteration.
        ssm_state = state.ssm_state
        outputs = []
        for t in range(x.size(1)):
            output, ssm_state = self._update_state(
                u[:, t], z[:, t], delta[:, t], b[:, t], c[:, t], ssm_state, a
            )
            outputs.append(output)
        y = torch.stack(outputs, dim=1)
        next_state = SSMState(next_conv_state, ssm_state)
        return (y, next_state) if return_state else y

    def scan(
        self, x: Tensor, state: Optional[SSMState] = None
    ) -> tuple[Tensor, SSMState]:
        """Explicit full-sequence scan alias returning the final state."""

        return self.forward(x, state=state, return_state=True)

    def full_scan(
        self, x: Tensor, state: Optional[SSMState] = None
    ) -> tuple[Tensor, SSMState]:
        """Readable alias for :meth:`scan`."""

        return self.scan(x, state=state)

    def step(self, x: Tensor, state: Optional[SSMState] = None) -> tuple[Tensor, SSMState]:
        """Update the SSM for one token and return ``[batch, d_model]`` output."""

        if x.ndim != 2 or x.size(-1) != self.d_model:
            raise ValueError(f"expected [batch, {self.d_model}], got shape {tuple(x.shape)}")
        state = self._coerce_state(state, x.unsqueeze(1))
        projected = self.in_proj(x)
        x_branch, z = projected.chunk(2, dim=-1)
        conv_output, next_conv_state = self._causal_conv_step(x_branch, state)
        u = F.silu(conv_output)
        delta, b, c, a = self._select_parameters(u)
        y, next_ssm_state = self._update_state(
            u, z, delta, b, c, state.ssm_state, a
        )
        return y, SSMState(next_conv_state, next_ssm_state)
