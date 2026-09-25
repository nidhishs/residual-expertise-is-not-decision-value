"""Acceptance tests: verify paper claims against pre-computed `latest/` results.

Marked with @pytest.mark.acceptance; skip cleanly when `latest/` is absent.
Run with: pytest -m acceptance

"""

from __future__ import annotations

import json

import pytest

from paths import experiment_results

pytestmark = pytest.mark.acceptance


def _load_latest(experiment: str, filename: str):
    path = experiment_results(experiment) / "latest" / filename
    if not path.exists():
        pytest.skip(f"No latest run at {path}; re-run the experiment first.")
    return json.loads(path.read_text())


def test_e1_dissociation_narrative():
    """E1 certifies the dissociation is constructible: C3 has high LL gain but zero BR."""
    summary = {r["condition"]: r for r in _load_latest("experiment_1", "summary.json")}

    # C1 (positive control): estimator ranks true BR correctly, detects positive BR.
    assert summary["C1"]["mean_spearman_rho"] > 0.5
    assert summary["C1"]["mean_frac_br_pos_hat"] > 0.10

    # C2 (weak null): no false positives: structural zero, must be exact.
    assert summary["C2"]["mean_fpr"] == 0.0
    assert summary["C2"]["mean_frac_br_pos_hat"] == 0.0

    # C3 (strong null): zero-BR certificate: structural zero, must be exact.
    assert summary["C3"]["mean_fpr"] == 0.0
    assert summary["C3"]["mean_frac_br_pos_hat"] == 0.0

    # The dissociation: C3 is more informative than C1, yet has zero BR.
    assert (
        summary["C3"]["mean_ll_gain"] > summary["C1"]["mean_ll_gain"]
    ), "C3 should beat C1 on LL gain (the dissociation: informativeness ≠ decision value)"


def test_e2_reward_asymmetry_narrative():
    """E2 reproduces the paper's reward-conditional dissociation on CheXpert."""
    agg = _load_latest("experiment_2", "results.json")["aggregate"]

    # R1 is positive on all 25 pairs; R2 has a negative mean despite 15
    # individually positive pairs (the magnitude, not a majority, drives the mean).
    assert agg["R1"]["mean_rho_ll_br"] > 0
    assert agg["R1"]["n_rho_positive"] == 25
    assert agg["R2"]["mean_rho_ll_br"] < 0
    assert agg["R2"]["n_rho_positive"] == 15

    # Under R3, only seven pairs have non-zero boundary regret, all with
    # negative correlation; nearly all top-expertise cases carry zero BR.
    assert agg["R3"]["mean_rho_ll_br"] < 0
    assert agg["R3"]["n_rho_positive"] == 0
    assert agg["R3"]["n_rho_negative"] == 7
    assert agg["R3"]["mean_top_decile_frac_br_zero"] > 0.9
    assert (
        agg["R1"]["mean_top_decile_frac_br_zero"]
        < agg["R2"]["mean_top_decile_frac_br_zero"]
        < agg["R3"]["mean_top_decile_frac_br_zero"]
    )
