from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from criteo_experiment.experiment import (
    allocation_audit,
    binary_outcome_effect,
    categorical_balance,
    estimate_outcome_effects,
    minimum_detectable_effect,
    required_total_sample_size,
    standardized_mean_difference,
    summarize_population,
)


def test_allocation_audit_uses_documented_ratio() -> None:
    result = allocation_audit(85, 100, expected_proportion=0.85)

    assert result["treatment_proportion"] == pytest.approx(0.85)
    assert result["control_proportion"] == pytest.approx(0.15)
    assert result["proportion_difference"] == pytest.approx(0.0)
    assert result["z_statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_standardized_mean_difference_uses_pooled_variance() -> None:
    result = standardized_mean_difference(1.0, 0.0, 1.0, 1.0)

    assert result == pytest.approx(1.0)


def test_categorical_balance_keeps_all_levels() -> None:
    result = categorical_balance(
        {"a": 80, "b": 19, "rare": 1},
        {"a": 50, "b": 50},
    )

    assert result["level_count"] == 3
    assert result["maximum_absolute_standardized_difference"] > 0
    assert result["weighted_rms_standardized_difference"] > 0


def test_categorical_balance_is_zero_for_identical_distributions() -> None:
    result = categorical_balance({"a": 70, "b": 30}, {"a": 70, "b": 30})

    assert result["maximum_absolute_standardized_difference"] == pytest.approx(0)
    assert result["weighted_rms_standardized_difference"] == pytest.approx(0)


def test_binary_outcome_effect_reports_risk_difference_and_relative_lift() -> None:
    result = binary_outcome_effect(60, 100, 40, 100)

    assert result["absolute_risk_difference"] == pytest.approx(0.20)
    lower, upper = result["absolute_risk_difference_ci"]
    assert lower < 0.20 < upper
    assert result["confidence_level"] == pytest.approx(0.95)
    assert result["relative_risk"] == pytest.approx(1.5)
    assert result["relative_lift"] == pytest.approx(0.5)
    assert result["relative_lift_percent"] == pytest.approx(50.0)
    assert result["incremental_outcomes_per_100000_assigned"] == pytest.approx(
        20_000
    )


def test_binary_outcome_confidence_interval_regression() -> None:
    result = binary_outcome_effect(60, 100, 40, 100)

    # These fixed bounds come from an independent Wald interval calculation.
    lower, upper = result["absolute_risk_difference_ci"]
    assert result["absolute_risk_difference"] == pytest.approx(0.2)
    assert lower == pytest.approx(0.06420971191085939)
    assert upper == pytest.approx(0.33579028808914063)
    assert result["confidence_level"] == pytest.approx(0.95)


def test_mde_increases_with_target_power() -> None:
    mde_80 = minimum_detectable_effect(
        0.05, 850_000, 150_000, power=0.80
    )
    mde_90 = minimum_detectable_effect(
        0.05, 850_000, 150_000, power=0.90
    )

    assert 0 < mde_80["absolute_effect"] < mde_90["absolute_effect"]
    assert mde_80["relative_effect"] == pytest.approx(
        mde_80["absolute_effect"] / 0.05
    )


def test_mde_numerical_regression() -> None:
    result = minimum_detectable_effect(
        0.05,
        treatment_count=850_000,
        control_count=150_000,
        alpha=0.05,
        power=0.80,
    )

    # This fixed reference comes from an independent unequal allocation calculation.
    assert result["absolute_effect"] == pytest.approx(
        0.001723908697789732,
        rel=1e-9,
    )


def test_required_sample_size_is_positive() -> None:
    required = required_total_sample_size(
        baseline_rate=0.05,
        target_absolute_effect=0.002,
        treatment_proportion=0.85,
        power=0.80,
    )

    assert required > 0


def test_primary_effect_uses_treatment_and_preserves_duplicate_rows(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "experiment.parquet"
    duplicate_control = {
        "treatment": 0,
        "exposure": 0,
        "visit": 0,
        "conversion": 0,
    }
    pl.DataFrame(
        [
            duplicate_control,
            duplicate_control.copy(),
            {"treatment": 1, "exposure": 0, "visit": 1, "conversion": 0},
            {"treatment": 1, "exposure": 1, "visit": 1, "conversion": 1},
        ]
    ).write_parquet(parquet_path)

    population = summarize_population(parquet_path)
    effects = estimate_outcome_effects(population)

    assert population["total_count"] == 4
    assert population["control_count"] == 2
    assert population["exposure"]["treatment_positive"] == 1
    assert effects["visit"]["absolute_risk_difference"] == pytest.approx(1.0)
