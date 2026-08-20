from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from criteo_experiment.dataset import FEATURE_COLUMNS
from criteo_experiment.uplift import (
    cumulative_gain_curve,
    fit_feature_preprocessor,
    predict_s_learner_potential_outcomes,
    predict_t_learner_potential_outcomes,
    split_modeling_data,
    targeting_policy_table,
    transform_features,
    transformed_outcome,
    validate_model_features,
)


class ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: sparse.csr_matrix) -> np.ndarray:
        positive = np.full(features.shape[0], self.probability)
        return np.column_stack([1 - positive, positive])


class InteractionProbabilityModel:
    def predict_proba(self, features: sparse.csr_matrix) -> np.ndarray:
        values = features.toarray()
        original_columns = (values.shape[1] - 1) // 2
        treatment = values[:, original_columns]
        interactions = values[:, original_columns + 1 :]
        positive = 0.2 + 0.1 * treatment + 0.05 * interactions.sum(axis=1)
        return np.column_stack([1 - positive, positive])


def _synthetic_frame() -> pd.DataFrame:
    rows = []
    for treatment in (0, 1):
        for visit in (0, 1):
            for conversion in (0, 1):
                for repeat in range(10):
                    row = {
                        name: float(repeat % 3) for name in FEATURE_COLUMNS
                    }
                    row.update(
                        {
                            "treatment": treatment,
                            "visit": visit,
                            "conversion": conversion,
                            "_source_row": len(rows),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "forbidden", ["treatment", "exposure", "visit", "conversion"]
)
def test_model_features_reject_treatment_and_outcome_leakage(
    forbidden: str,
) -> None:
    with pytest.raises(ValueError, match="treatment or post assignment"):
        validate_model_features([*FEATURE_COLUMNS, forbidden])


def test_split_is_reproducible_and_preserves_duplicate_rows() -> None:
    frame = _synthetic_frame()
    first = split_modeling_data(frame, random_seed=42)
    second = split_modeling_data(frame, random_seed=42)

    assert [len(first[name]) for name in ("train", "validation", "test")] == [
        48,
        16,
        16,
    ]
    for name in first:
        assert first[name]["_source_row"].tolist() == second[name][
            "_source_row"
        ].tolist()
    source_rows = {
        name: set(split["_source_row"]) for name, split in first.items()
    }
    assert source_rows["train"].isdisjoint(source_rows["validation"])
    assert source_rows["train"].isdisjoint(source_rows["test"])
    assert source_rows["validation"].isdisjoint(source_rows["test"])
    combined = pd.concat(first.values(), ignore_index=True)
    duplicate_subset = [*FEATURE_COLUMNS, "treatment", "visit", "conversion"]
    assert len(combined) == len(frame)
    assert combined.duplicated(subset=duplicate_subset).sum() == frame.duplicated(
        subset=duplicate_subset
    ).sum()


def test_transformed_outcome_uses_unequal_propensity() -> None:
    result = transformed_outcome(
        outcome=np.array([1, 1, 0, 1]),
        treatment=np.array([1, 0, 1, 0]),
        propensity=0.8,
    )

    assert result == pytest.approx([1.25, -5.0, 0.0, -5.0])
    with pytest.raises(ValueError, match="propensity"):
        transformed_outcome([1], [1], propensity=1.0)


def test_s_learner_cate_uses_treatment_interactions() -> None:
    features = sparse.csr_matrix([[1.0, 2.0], [2.0, 0.0]])

    p1, p0 = predict_s_learner_potential_outcomes(
        InteractionProbabilityModel(), features
    )

    assert p0 == pytest.approx([0.2, 0.2])
    assert p1 - p0 == pytest.approx([0.25, 0.20])


def test_t_learner_cate_subtracts_control_probability() -> None:
    features = sparse.csr_matrix(np.ones((3, 2)))

    p1, p0 = predict_t_learner_potential_outcomes(
        ConstantProbabilityModel(0.4),
        ConstantProbabilityModel(0.1),
        features,
    )

    assert p1 - p0 == pytest.approx([0.3, 0.3, 0.3])


def test_ranking_direction_and_cumulative_gain() -> None:
    signal = np.array([4.0, 3.0, -1.0, -2.0])
    good = cumulative_gain_curve(signal, signal, curve_points=4)
    bad = cumulative_gain_curve(-signal, signal, curve_points=4)

    assert good["policy_gain_per_100000"] == pytest.approx(
        [0.0, 100_000.0, 175_000.0, 150_000.0, 100_000.0]
    )
    assert good["auuc_above_random_per_100000"] == pytest.approx(68_750.0)
    assert good["auuc_above_random_per_100000"] > bad[
        "auuc_above_random_per_100000"
    ]


def test_random_targeting_baseline_connects_zero_and_treat_all() -> None:
    result = cumulative_gain_curve(
        np.array([3.0, 2.0, 1.0, 0.0]),
        np.array([2.0, 0.0, 1.0, 1.0]),
        curve_points=4,
    )

    assert result["random_gain_per_100000"] == pytest.approx(
        [0.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0]
    )
    assert result["treat_all_gain_per_100000"] == pytest.approx(100_000.0)


def test_targeting_fraction_uses_population_denominator() -> None:
    scores = np.arange(10, dtype=float)[::-1]
    signal = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    outcome = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])

    result = targeting_policy_table(
        scores,
        signal,
        outcome,
        treatment,
        fractions=(0.5, 1.0),
    )

    assert result[0]["targeted_count"] == 5
    assert result[0]["estimated_incremental_outcome_rate_among_targeted"] == 1.0
    assert result[0]["incremental_outcomes_per_100000_population"] == 50_000
    assert result[0]["random_targeting_per_100000_population"] == 30_000
    assert result[0]["ipw_transformed_outcome_rate_among_targeted"] == 1.2


def test_targeting_interval_is_not_degenerate_with_zero_events() -> None:
    result = targeting_policy_table(
        scores=np.array([4.0, 3.0, 2.0, 1.0]),
        evaluation_signal=np.zeros(4),
        outcome=np.zeros(4, dtype=int),
        treatment=np.array([1, 0, 1, 0]),
        fractions=(1.0,),
        outcome_name="conversion",
    )

    lower, upper = result[0]["observed_group_effect"][
        "observed_difference_ci"
    ]
    assert lower < 0 < upper
    assert result[0]["insufficient_control_events"] is True


def test_treat_all_policy_equals_full_observed_difference() -> None:
    scores = np.array([0.4, 0.3, 0.2, 0.1])
    treatment = np.array([1, 1, 0, 0])
    outcome = np.array([1, 0, 0, 0])
    observed_difference = outcome[treatment == 1].mean() - outcome[
        treatment == 0
    ].mean()

    result = targeting_policy_table(
        scores,
        evaluation_signal=np.zeros(4),
        outcome=outcome,
        treatment=treatment,
        fractions=(1.0,),
    )[0]

    expected_gain = observed_difference * 100_000
    assert result["incremental_outcomes_per_100000_population"] == pytest.approx(
        expected_gain
    )
    assert result["random_targeting_per_100000_population"] == pytest.approx(
        expected_gain
    )
    assert result["incremental_outcomes_above_random_per_100000"] == pytest.approx(
        0.0
    )


def test_test_transform_reuses_training_preprocessor() -> None:
    train = _synthetic_frame().iloc[:40].copy()
    test = train.iloc[:2].copy()
    test.loc[:, "f1"] = 99.0
    preprocessor = fit_feature_preprocessor(train)
    categories_before = tuple(
        tuple(values)
        for values in preprocessor.named_transformers_["categorical"].categories_
    )

    transformed = transform_features(preprocessor, test)
    categories_after = tuple(
        tuple(values)
        for values in preprocessor.named_transformers_["categorical"].categories_
    )

    assert transformed.shape[0] == 2
    assert categories_after == categories_before
    assert 99.0 not in categories_after[0]
