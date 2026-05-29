"""Tests for the public metrics exposed by :mod:`tsseg_eval`."""

import numpy as np
import pytest

from tsseg_eval import ari, covering, f1, nmi, sms, wari, wnmi
from tsseg_eval.metrics import (
    DEFAULT_SMS_WEIGHTS,
    labels_to_change_points,
    state_matching_score,
)


# ----------------------------- fixtures --------------------------------


def _three_segments():
    gt = np.array([0] * 30 + [1] * 30 + [2] * 40)
    pred = np.array([0] * 28 + [1] * 32 + [2] * 40)
    return gt, pred


# ------------------------------ identity --------------------------------


@pytest.mark.parametrize("metric", [covering, nmi, ari, wari, wnmi, sms])
def test_perfect_match_is_one(metric):
    gt = np.array([0] * 20 + [1] * 20 + [2] * 20)
    assert metric(gt, gt) == pytest.approx(1.0)


def test_f1_perfect_match_is_one():
    gt = np.array([0] * 20 + [1] * 20 + [2] * 20)
    assert f1(gt, gt) == pytest.approx(1.0)


# --------------------------- constant labels ----------------------------


@pytest.mark.parametrize("metric", [covering, nmi, ari, wari, wnmi, sms])
def test_constant_labels_match_is_one(metric):
    const = np.zeros(50, dtype=int)
    assert metric(const, const) == pytest.approx(1.0)


# --------------------- inverse labels behave correctly -------------------


def test_inverse_labelling_is_perfect_for_label_invariant_metrics():
    # NMI/ARI are invariant under relabelling; SMS also remaps optimally.
    gt = np.array([0] * 20 + [1] * 20)
    pred = np.array([1] * 20 + [0] * 20)
    assert nmi(gt, pred) == pytest.approx(1.0)
    assert ari(gt, pred) == pytest.approx(1.0)
    assert sms(gt, pred) == pytest.approx(1.0)


# ---------------------------- bounds ------------------------------------


@pytest.mark.parametrize(
    "metric", [covering, nmi, ari, wari, wnmi, sms]
)
def test_metric_within_bounds(metric):
    gt, pred = _three_segments()
    value = metric(gt, pred)
    # All these metrics are bounded above by 1.
    assert value <= 1.0 + 1e-9
    # ARI/WARI can technically dip below 0 for adversarial inputs, but on this
    # well-behaved example they should be comfortably non-negative.
    assert value >= -1.0 - 1e-9


# --------------------------- helpers ------------------------------------


def test_labels_to_change_points_includes_boundaries():
    labels = [0, 0, 1, 1, 2]
    cps = labels_to_change_points(labels)
    assert cps[0] == 0
    assert cps[-1] == len(labels)
    assert 2 in cps and 4 in cps


# -------------------- sms safety: weights & empty -----------------------


def test_sms_default_weights_are_not_mutated():
    gt, pred = _three_segments()
    sms(gt, pred)
    assert DEFAULT_SMS_WEIGHTS == {
        "delay": 0.1,
        "transition": 0.3,
        "isolation": 0.8,
        "missing": 0.5,
    }


def test_sms_empty_input_raises():
    with pytest.raises(ValueError):
        state_matching_score(np.array([], dtype=int), np.array([], dtype=int))


def test_sms_return_errors_format():
    gt, pred = _three_segments()
    score, errors = sms(gt, pred), None
    score, errors = state_matching_score(gt, pred, return_errors=True)
    assert isinstance(errors, list)
    for err in errors:
        assert set(err.keys()) >= {"type", "start", "end", "size", "penalty"}


# ------------------------ f_score margin scaling ------------------------


def test_f_score_default_margin_scales_with_series_length():
    # 1000-point series with one true change-point at 500; prediction off by 5.
    gt = np.zeros(1000, dtype=int)
    gt[500:] = 1
    pred = np.zeros(1000, dtype=int)
    pred[505:] = 1
    # Default margin should be ~10 (1% of 1000) so a 5-sample shift matches.
    assert f1(gt, pred) == pytest.approx(1.0)
