from pathlib import Path

import pytest

from hralign.checkpoint import compare_model_state_layout
from hralign.models import (
    HRAlignR3ML,
    extract_model_state,
    load_torch_checkpoint,
)
from hralign.trainer import build_optimizer


UNADAPTED = Path("/home/paichichi/data/pretrain/UnadaptedR3M.pt")
ADAPTED = Path("/home/paichichi/data/pretrain/AdaptedR3M.pyth")


@pytest.mark.skipif(
    not (UNADAPTED.is_file() and ADAPTED.is_file()),
    reason="Local released R3M checkpoints are unavailable.",
)
def test_export_layout_matches_released_adapted_r3m():
    model = HRAlignR3ML(UNADAPTED)
    produced = model.official_model_state()
    reference = extract_model_state(load_torch_checkpoint(ADAPTED))
    report = compare_model_state_layout(produced, reference)
    assert len(produced) == len(reference) == 438
    assert report["missing"] == []
    assert report["unexpected"] == []
    assert report["shape_mismatches"] == {}
    assert model.trainable_parameter_count() == {
        "adapters": 1582080,
        "language_projection": 1574912,
        "total": 3156992,
    }
    optimizer = build_optimizer(
        model,
        learning_rate=1e-4,
        weight_decay=1e-4,
        zero_weight_decay_1d=True,
    )
    assert [len(group["params"]) for group in optimizer.param_groups] == [10, 10]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        1e-4,
        0.0,
    ]


@pytest.mark.skipif(
    not (UNADAPTED.is_file() and ADAPTED.is_file()),
    reason="Local released R3M checkpoints are unavailable.",
)
def test_release_changes_only_all_resnet_bn_running_buffers():
    unadapted = {
        key.removeprefix("module."): value
        for key, value in extract_model_state(
            load_torch_checkpoint(UNADAPTED)
        ).items()
    }
    adapted = extract_model_state(load_torch_checkpoint(ADAPTED))
    changed = [
        key
        for key, value in adapted.items()
        if key in unadapted
        and value.shape == unadapted[key].shape
        and not value.equal(unadapted[key])
    ]
    assert len(changed) == 159
    assert all(
        key.endswith(
            ("running_mean", "running_var", "num_batches_tracked")
        )
        for key in changed
    )
