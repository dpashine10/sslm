"""Offline checkpoint durability and resume tests."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.training.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    load_latest_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


def test_checkpoint_round_trip_restores_state_and_rng(tmp_path) -> None:
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = model(torch.randn(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    path = save_checkpoint(
        tmp_path,
        step=3,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        tokens_seen=123,
        config={"model": {"kind": "fixture"}},
        validation_loss=1.25,
    )
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()
    payload = load_checkpoint(path)
    assert payload["step"] == 3
    assert payload["tokens_seen"] == 123
    assert payload["validation_loss"] == 1.25
    assert load_latest_checkpoint(tmp_path)["step"] == 3
    expected_next = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    resumed_model = torch.nn.Linear(4, 3)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_model.load_state_dict(payload["model"])
    resumed_optimizer.load_state_dict(payload["optimizer"])
    for actual, expected in zip(resumed_model.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected)
    assert resumed_optimizer.state

    # Consume values, then put the exact saved random stream back.
    random.random()
    np.random.rand()
    torch.rand(1)
    restore_rng_state(payload["rng_state"])
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == expected_next


def test_incomplete_checkpoint_is_ignored(tmp_path) -> None:
    (tmp_path / "checkpoint_step_00000009.pt").write_bytes(b"partial")
    assert load_latest_checkpoint(tmp_path) is None
