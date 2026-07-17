import math

import torch

from hralign.trainer import build_scheduler


def make_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return torch.optim.Adam(
        [
            {
                "params": [parameter],
                "lr": 1e-4,
                "layer_decay": 1.0,
            }
        ]
    )


def test_scheduler_exactly_reproduces_released_checkpoint_lr():
    scheduler = build_scheduler(
        make_optimizer(),
        base_lr=1e-4,
        warmup_start_lr=1e-6,
        warmup_epochs=10.0,
        schedule_epochs=300.0,
        steps_per_epoch=283,
        end_lr=1e-6,
    )

    assert scheduler.lr_at_step(0) == 1e-6
    assert math.isclose(
        scheduler.lr_at_step(2829),
        9.996501766784452e-5,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert scheduler.lr_at_step(2830) == 1e-4
    assert math.isclose(
        scheduler.lr_at_step(300 * 283),
        1e-6,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_scheduler_resume_rejects_changed_epoch_geometry():
    original = build_scheduler(
        make_optimizer(),
        base_lr=1e-4,
        warmup_start_lr=1e-6,
        warmup_epochs=10.0,
        schedule_epochs=300.0,
        steps_per_epoch=283,
        end_lr=1e-6,
    )
    original.set_step(123)
    state = original.state_dict()

    resumed = build_scheduler(
        make_optimizer(),
        base_lr=1e-4,
        warmup_start_lr=1e-6,
        warmup_epochs=10.0,
        schedule_epochs=300.0,
        steps_per_epoch=283,
        end_lr=1e-6,
    )
    resumed.load_state_dict(state)
    assert resumed.last_completed_step == 123
    assert resumed.optimizer.param_groups[0]["lr"] == resumed.lr_at_step(123)

    incompatible = build_scheduler(
        make_optimizer(),
        base_lr=1e-4,
        warmup_start_lr=1e-6,
        warmup_epochs=10.0,
        schedule_epochs=300.0,
        steps_per_epoch=280,
        end_lr=1e-6,
    )
    try:
        incompatible.load_state_dict(state)
    except ValueError as error:
        assert "configuration changed" in str(error)
    else:
        raise AssertionError("Expected changed steps_per_epoch to be rejected.")
