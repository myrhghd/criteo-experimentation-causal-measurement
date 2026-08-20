"""Audit randomization and estimate benchmark population treatment effects."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from criteo_experiment.dataset import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    DEFAULT_PARQUET_PATH,
    FEATURE_COLUMNS,
)

EXPECTED_TREATMENT_PROPORTION = 0.85
BALANCE_THRESHOLD = 0.10
OUTCOMES = ("visit", "conversion")
DEFAULT_SAMPLE_SIZE = 200_000
DEFAULT_RANDOM_SEED = 42
DEFAULT_FIGURES_DIR = Path("reports/figures")


class ExperimentError(RuntimeError):
    """Indicate that the experiment audit could not complete."""


def allocation_audit(
    treatment_count: int,
    total_count: int,
    expected_proportion: float = EXPECTED_TREATMENT_PROPORTION,
) -> dict[str, float | int]:
    """Compare an observed treatment allocation with its documented ratio."""

    if total_count <= 0 or not 0 < expected_proportion < 1:
        raise ValueError("Allocation inputs must define a nonempty valid design")
    if not 0 <= treatment_count <= total_count:
        raise ValueError("Treatment count must fall within the total count")

    observed = treatment_count / total_count
    expected_count = total_count * expected_proportion
    standard_error = math.sqrt(
        total_count * expected_proportion * (1 - expected_proportion)
    )
    z_statistic = (treatment_count - expected_count) / standard_error
    p_value = float(2 * stats.norm.sf(abs(z_statistic)))
    return {
        "total_observations": total_count,
        "treatment_observations": treatment_count,
        "control_observations": total_count - treatment_count,
        "treatment_proportion": observed,
        "control_proportion": 1 - observed,
        "expected_treatment_proportion": expected_proportion,
        "proportion_difference": observed - expected_proportion,
        "difference_percentage_points": (observed - expected_proportion) * 100,
        "count_difference_from_expected": treatment_count - expected_count,
        "z_statistic": z_statistic,
        "p_value": p_value,
    }


def standardized_mean_difference(
    treatment_mean: float,
    control_mean: float,
    treatment_standard_deviation: float,
    control_standard_deviation: float,
) -> float:
    """Calculate a standardized mean difference with pooled group variance."""

    pooled_standard_deviation = math.sqrt(
        (
            treatment_standard_deviation**2
            + control_standard_deviation**2
        )
        / 2
    )
    if pooled_standard_deviation == 0:
        return 0.0 if treatment_mean == control_mean else math.inf
    return (treatment_mean - control_mean) / pooled_standard_deviation


def categorical_balance(
    treatment_counts: Mapping[object, int],
    control_counts: Mapping[object, int],
) -> dict[str, object]:
    """Summarize level specific standardized differences for one feature."""

    treatment_total = sum(treatment_counts.values())
    control_total = sum(control_counts.values())
    if treatment_total <= 0 or control_total <= 0:
        raise ValueError("Both groups must contain categorical observations")

    levels = set(treatment_counts).union(control_counts)
    weighted_squared_sum = 0.0
    maximum_record: dict[str, object] | None = None
    total = treatment_total + control_total

    # Feature codes remain nominal, so each level becomes an indicator proportion.
    # Maximum absolute and weighted RMS SMDs summarize the level comparisons.
    for level in levels:
        treatment_proportion = treatment_counts.get(level, 0) / treatment_total
        control_proportion = control_counts.get(level, 0) / control_total
        pooled_standard_deviation = math.sqrt(
            (
                treatment_proportion * (1 - treatment_proportion)
                + control_proportion * (1 - control_proportion)
            )
            / 2
        )
        difference = treatment_proportion - control_proportion
        level_smd = (
            difference / pooled_standard_deviation
            if pooled_standard_deviation
            else 0.0
        )
        pooled_weight = (
            treatment_counts.get(level, 0) + control_counts.get(level, 0)
        ) / total
        weighted_squared_sum += pooled_weight * level_smd**2
        record = {
            "level": _json_scalar(level),
            "treatment_proportion": treatment_proportion,
            "control_proportion": control_proportion,
            "standardized_difference": level_smd,
        }
        if maximum_record is None or abs(level_smd) > abs(
            float(maximum_record["standardized_difference"])
        ):
            maximum_record = record

    assert maximum_record is not None
    return {
        "level_count": len(levels),
        "maximum_absolute_standardized_difference": abs(
            float(maximum_record["standardized_difference"])
        ),
        "weighted_rms_standardized_difference": math.sqrt(
            weighted_squared_sum
        ),
        "maximum_difference_level": maximum_record,
    }


def binary_outcome_effect(
    treatment_positive: int,
    treatment_count: int,
    control_positive: int,
    control_count: int,
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Estimate a binary outcome intention to treat effect."""

    if treatment_count <= 0 or control_count <= 0:
        raise ValueError("Both experiment groups must contain observations")
    if not 0 <= treatment_positive <= treatment_count:
        raise ValueError("Treatment positives must fall within treatment count")
    if not 0 <= control_positive <= control_count:
        raise ValueError("Control positives must fall within control count")

    treatment_rate = treatment_positive / treatment_count
    control_rate = control_positive / control_count
    risk_difference = treatment_rate - control_rate
    z_critical = float(stats.norm.ppf(1 - alpha / 2))
    # The interval uses each randomized group's variance without null pooling.
    difference_standard_error = math.sqrt(
        treatment_rate * (1 - treatment_rate) / treatment_count
        + control_rate * (1 - control_rate) / control_count
    )
    confidence_interval = (
        max(-1.0, risk_difference - z_critical * difference_standard_error),
        min(1.0, risk_difference + z_critical * difference_standard_error),
    )

    # Equal outcome rates define the null, so its hypothesis test pools the rate.
    pooled_rate = (treatment_positive + control_positive) / (
        treatment_count + control_count
    )
    null_standard_error = math.sqrt(
        pooled_rate
        * (1 - pooled_rate)
        * (1 / treatment_count + 1 / control_count)
    )
    if null_standard_error:
        z_statistic = risk_difference / null_standard_error
    else:
        z_statistic = 0.0 if risk_difference == 0 else math.inf
    # Full data z statistics can make ordinary p values underflow to zero.
    # Log space retains useful tail information through log10_p_value.
    log_p_value = min(
        0.0,
        math.log(2) + float(stats.norm.logsf(abs(z_statistic))),
    )
    p_value = (
        math.exp(log_p_value)
        if log_p_value > math.log(np.finfo(float).tiny)
        else 0.0
    )

    relative_risk: float | None = None
    relative_risk_interval: tuple[float, float] | None = None
    relative_lift: float | None = None
    if treatment_positive and control_positive and control_rate:
        relative_risk = treatment_rate / control_rate
        log_relative_risk_standard_error = math.sqrt(
            1 / treatment_positive
            - 1 / treatment_count
            + 1 / control_positive
            - 1 / control_count
        )
        log_relative_risk = math.log(relative_risk)
        relative_risk_interval = (
            math.exp(
                log_relative_risk
                - z_critical * log_relative_risk_standard_error
            ),
            math.exp(
                log_relative_risk
                + z_critical * log_relative_risk_standard_error
            ),
        )
        relative_lift = relative_risk - 1

    return {
        "treatment_count": treatment_count,
        "treatment_positive": treatment_positive,
        "treatment_rate": treatment_rate,
        "control_count": control_count,
        "control_positive": control_positive,
        "control_rate": control_rate,
        "confidence_level": 1 - alpha,
        "absolute_risk_difference": risk_difference,
        "absolute_risk_difference_ci": list(confidence_interval),
        "standard_error": difference_standard_error,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "log_p_value": log_p_value,
        "log10_p_value": log_p_value / math.log(10),
        "relative_risk": relative_risk,
        "relative_risk_ci": (
            list(relative_risk_interval)
            if relative_risk_interval is not None
            else None
        ),
        "relative_lift": relative_lift,
        "relative_lift_percent": (
            relative_lift * 100 if relative_lift is not None else None
        ),
        "incremental_outcomes_per_100000_assigned": risk_difference * 100_000,
    }


def minimum_detectable_effect(
    baseline_rate: float,
    treatment_count: int,
    control_count: int,
    *,
    power: float,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Calculate an asymptotic MDE using actual unequal group sizes."""

    if not 0 < baseline_rate < 1:
        raise ValueError("Baseline rate must fall strictly between zero and one")
    if treatment_count <= 0 or control_count <= 0:
        raise ValueError("Both group sizes must be positive")

    # Cohen's h handles binary proportions while preserving the actual allocation.
    # This is prospective sensitivity, not retrospective observed power.
    ratio = treatment_count / control_count
    standardized_effect = float(
        NormalIndPower().solve_power(
            effect_size=None,
            nobs1=control_count,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative="two-sided",
        )
    )
    baseline_transform = 2 * math.asin(math.sqrt(baseline_rate))
    detectable_rate = math.sin(
        (baseline_transform + abs(standardized_effect)) / 2
    ) ** 2
    absolute_effect = detectable_rate - baseline_rate
    return {
        "power": power,
        "alpha": alpha,
        "absolute_effect": absolute_effect,
        "relative_effect": absolute_effect / baseline_rate,
        "relative_effect_percent": absolute_effect / baseline_rate * 100,
    }


def required_total_sample_size(
    baseline_rate: float,
    target_absolute_effect: float,
    treatment_proportion: float,
    *,
    power: float,
    alpha: float = 0.05,
) -> int:
    """Calculate total sample size for a target binary outcome effect."""

    target_rate = baseline_rate + target_absolute_effect
    if not 0 < baseline_rate < 1 or not 0 < target_rate < 1:
        raise ValueError("Baseline and target rates must fall between zero and one")
    if not 0 < treatment_proportion < 1:
        raise ValueError("Treatment proportion must fall between zero and one")

    standardized_effect = abs(
        float(proportion_effectsize(target_rate, baseline_rate))
    )
    ratio = treatment_proportion / (1 - treatment_proportion)
    control_count = float(
        NormalIndPower().solve_power(
            effect_size=standardized_effect,
            nobs1=None,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative="two-sided",
        )
    )
    return math.ceil(control_count * (1 + ratio))


def summarize_population(parquet_path: Path) -> dict[str, object]:
    """Aggregate experiment groups without removing any input rows."""

    scan = pl.scan_parquet(parquet_path)
    expressions: list[pl.Expr] = [
        pl.len().alias("total_count"),
        pl.col("treatment").sum().alias("treatment_count"),
        pl.col("exposure").sum().alias("exposure_total"),
        (pl.col("exposure") * pl.col("treatment"))
        .sum()
        .alias("exposure_treatment"),
        (pl.col("exposure") * (1 - pl.col("treatment")))
        .sum()
        .alias("exposure_control"),
    ]
    for outcome in OUTCOMES:
        expressions.extend(
            [
                pl.col(outcome).sum().alias(f"{outcome}_total"),
                (pl.col(outcome) * pl.col("treatment"))
                .sum()
                .alias(f"{outcome}_treatment"),
                (pl.col(outcome) * (1 - pl.col("treatment")))
                .sum()
                .alias(f"{outcome}_control"),
            ]
        )

    row = scan.select(expressions).collect(engine="streaming").row(named=True)
    total_count = int(row["total_count"])
    treatment_count = int(row["treatment_count"])
    control_count = total_count - treatment_count
    return {
        "total_count": total_count,
        "treatment_count": treatment_count,
        "control_count": control_count,
        "outcomes": {
            outcome: {
                "total_positive": int(row[f"{outcome}_total"]),
                "treatment_positive": int(row[f"{outcome}_treatment"]),
                "control_positive": int(row[f"{outcome}_control"]),
            }
            for outcome in OUTCOMES
        },
        "exposure": {
            "total_positive": int(row["exposure_total"]),
            "treatment_positive": int(row["exposure_treatment"]),
            "control_positive": int(row["exposure_control"]),
        },
    }


def estimate_outcome_effects(
    population: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Estimate outcome effects using randomized treatment assignment groups."""

    treatment_count = int(population["treatment_count"])
    control_count = int(population["control_count"])
    outcome_counts = population["outcomes"]
    return {
        outcome: binary_outcome_effect(
            int(outcome_counts[outcome]["treatment_positive"]),
            treatment_count,
            int(outcome_counts[outcome]["control_positive"]),
            control_count,
        )
        for outcome in OUTCOMES
    }


def _continuous_balance(parquet_path: Path) -> dict[str, dict[str, float]]:
    treatment_filter = pl.col("treatment") == 1
    control_filter = pl.col("treatment") == 0
    expressions = []
    for feature in CONTINUOUS_FEATURES:
        expressions.extend(
            [
                pl.col(feature)
                .filter(treatment_filter)
                .mean()
                .alias(f"{feature}_treatment_mean"),
                pl.col(feature)
                .filter(control_filter)
                .mean()
                .alias(f"{feature}_control_mean"),
                pl.col(feature)
                .filter(treatment_filter)
                .std(ddof=1)
                .alias(f"{feature}_treatment_sd"),
                pl.col(feature)
                .filter(control_filter)
                .std(ddof=1)
                .alias(f"{feature}_control_sd"),
            ]
        )
    row = (
        pl.scan_parquet(parquet_path)
        .select(expressions)
        .collect(engine="streaming")
        .row(named=True)
    )
    results = {}
    for feature in CONTINUOUS_FEATURES:
        treatment_mean = float(row[f"{feature}_treatment_mean"])
        control_mean = float(row[f"{feature}_control_mean"])
        treatment_sd = float(row[f"{feature}_treatment_sd"])
        control_sd = float(row[f"{feature}_control_sd"])
        smd = standardized_mean_difference(
            treatment_mean,
            control_mean,
            treatment_sd,
            control_sd,
        )
        results[feature] = {
            "treatment_mean": treatment_mean,
            "control_mean": control_mean,
            "treatment_standard_deviation": treatment_sd,
            "control_standard_deviation": control_sd,
            "standardized_mean_difference": smd,
            "absolute_standardized_mean_difference": abs(smd),
            "variance_ratio": treatment_sd**2 / control_sd**2,
        }
    return results


def _categorical_balance(parquet_path: Path) -> dict[str, dict[str, object]]:
    results = {}
    for feature in CATEGORICAL_FEATURES:
        counts = (
            pl.scan_parquet(parquet_path)
            .group_by([feature, "treatment"])
            .agg(pl.len().alias("count"))
            .collect(engine="streaming")
        )
        treatment_counts: dict[object, int] = {}
        control_counts: dict[object, int] = {}
        for row in counts.iter_rows(named=True):
            target = treatment_counts if row["treatment"] == 1 else control_counts
            target[row[feature]] = int(row["count"])
        results[feature] = categorical_balance(
            treatment_counts, control_counts
        )
    return results


def _sample_parquet_rows(
    parquet_path: Path,
    columns: Sequence[str],
    sample_size: int,
    random_seed: int,
) -> pa.Table:
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    sample_size = min(sample_size, total_rows)
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")

    # A seeded subset avoids fitting all 14 million rows while remaining reproducible.
    random_generator = np.random.default_rng(random_seed)
    selected_indices = np.sort(
        random_generator.choice(total_rows, size=sample_size, replace=False)
    )
    selected_batches = []
    selected_position = 0
    row_offset = 0
    for batch in parquet_file.iter_batches(
        columns=list(columns), batch_size=262_144
    ):
        batch_end = row_offset + batch.num_rows
        next_position = int(
            np.searchsorted(selected_indices, batch_end, side="left")
        )
        if next_position > selected_position:
            local_indices = (
                selected_indices[selected_position:next_position] - row_offset
            )
            selected_batches.append(
                batch.take(pa.array(local_indices, type=pa.int64()))
            )
        selected_position = next_position
        row_offset = batch_end
    return pa.Table.from_batches(selected_batches)


def _treatment_predictability(
    parquet_path: Path,
    sample_size: int,
    random_seed: int,
) -> dict[str, object]:
    columns = list(FEATURE_COLUMNS) + ["treatment"]
    sampled = _sample_parquet_rows(
        parquet_path, columns, sample_size, random_seed
    ).to_pandas()
    features = sampled.loc[:, list(FEATURE_COLUMNS)]
    treatment = sampled["treatment"].astype(int)
    train_features, test_features, train_treatment, test_treatment = (
        train_test_split(
            features,
            treatment,
            test_size=0.25,
            random_state=random_seed,
            stratify=treatment,
        )
    )

    # This is a randomization diagnostic, not a tuned prediction model. Results near
    # the constant propensity baseline support the assignment audit.
    preprocessing = ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), list(CONTINUOUS_FEATURES)),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        sparse_threshold=1.0,
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=200,
                    random_state=random_seed,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(train_features, train_treatment)
    predicted_probability = model.predict_proba(test_features)[:, 1]
    baseline_probability = float(train_treatment.mean())
    baseline_predictions = np.full(len(test_treatment), baseline_probability)
    classifier = model.named_steps["classifier"]
    return {
        "purpose": "randomization diagnostic only",
        "sample_size": int(len(sampled)),
        "random_seed": random_seed,
        "training_observations": int(len(train_treatment)),
        "test_observations": int(len(test_treatment)),
        "model": "regularized logistic regression",
        "continuous_preprocessing": "standard scaling",
        "categorical_preprocessing": "sparse one hot encoding",
        "roc_auc": float(roc_auc_score(test_treatment, predicted_probability)),
        "baseline_roc_auc": float(
            roc_auc_score(test_treatment, baseline_predictions)
        ),
        "log_loss": float(log_loss(test_treatment, predicted_probability)),
        "baseline_log_loss": float(
            log_loss(test_treatment, baseline_predictions)
        ),
        "log_loss_improvement": float(
            log_loss(test_treatment, baseline_predictions)
            - log_loss(test_treatment, predicted_probability)
        ),
        "iterations": int(classifier.n_iter_[0]),
    }


def _create_balance_figure(
    continuous: Mapping[str, Mapping[str, float]],
    categorical: Mapping[str, Mapping[str, object]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    labels = list(CONTINUOUS_FEATURES) + list(CATEGORICAL_FEATURES)
    values = [
        continuous[name]["absolute_standardized_mean_difference"]
        for name in CONTINUOUS_FEATURES
    ] + [
        categorical[name]["maximum_absolute_standardized_difference"]
        for name in CATEGORICAL_FEATURES
    ]
    colors = ["#2864a6"] * len(CONTINUOUS_FEATURES) + ["#7b4f9d"] * len(
        CATEGORICAL_FEATURES
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    positions = np.arange(len(labels))
    bars = axis.barh(positions, values, color=colors)
    axis.axvline(
        BALANCE_THRESHOLD,
        color="#b33a3a",
        linestyle="--",
        linewidth=1.2,
    )
    axis.bar_label(
        bars,
        labels=[f"{value:.3f}" for value in values],
        padding=3,
        fontsize=8.5,
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Absolute standardized difference")
    axis.set_title("Pretreatment covariate balance", pad=25)
    axis.text(
        0.5,
        1.015,
        "Categorical features show maximum level specific |SMD|",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    axis.set_xlim(0, BALANCE_THRESHOLD * 1.05)
    axis.legend(
        handles=[
            Patch(facecolor="#2864a6", label="Continuous feature"),
            Patch(facecolor="#7b4f9d", label="Categorical feature"),
            Line2D(
                [0],
                [0],
                color="#b33a3a",
                linestyle="--",
                linewidth=1.2,
                label="0.10 practical balance threshold",
            ),
        ],
        frameon=False,
        loc="lower right",
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _create_effect_figure(
    effects: Mapping[str, Mapping[str, object]], output_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Visit", "Conversion"]
    estimates = np.array(
        [effects[outcome]["absolute_risk_difference"] for outcome in OUTCOMES]
    )
    intervals = np.array(
        [effects[outcome]["absolute_risk_difference_ci"] for outcome in OUTCOMES]
    )
    errors = np.vstack(
        [estimates - intervals[:, 0], intervals[:, 1] - estimates]
    )
    figure, axis = plt.subplots(figsize=(9, 3.8))
    positions = np.arange(len(labels))
    axis.errorbar(
        estimates * 100,
        positions,
        xerr=errors * 100,
        fmt="o",
        color="#2864a6",
        ecolor="#2864a6",
        capsize=4,
    )
    axis.axvline(0, color="#555555", linewidth=1)
    for position, estimate, interval in zip(
        positions, estimates, intervals, strict=True
    ):
        axis.annotate(
            (
                f"+{estimate * 100:.3f} pp  "
                f"[95% CI {interval[0] * 100:.3f}, "
                f"{interval[1] * 100:.3f}]"
            ),
            xy=(interval[1] * 100, position),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
        )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(-0.05, max(1.75, float(intervals[:, 1].max() * 165)))
    axis.set_xlabel("Absolute ITT effect in percentage points")
    axis.set_title("Treatment assignment effects", pad=25)
    axis.text(
        0.5,
        1.02,
        "Point = ITT estimate; horizontal interval = 95% confidence interval",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _create_sensitivity_figure(
    effects: Mapping[str, Mapping[str, object]],
    power: Mapping[str, Mapping[str, object]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Visit", "Conversion"]
    positions = np.arange(len(labels))
    series = [
        (
            "Observed ITT effect",
            "o",
            "#2864a6",
            0.18,
            [
                float(effects[outcome]["absolute_risk_difference"]) * 100
                for outcome in OUTCOMES
            ],
            3,
        ),
        (
            "80% power MDE",
            "s",
            "#4f8a5b",
            0.0,
            [
                float(power[outcome]["mde"]["0.8"]["absolute_effect"])
                * 100
                for outcome in OUTCOMES
            ],
            4,
        ),
        (
            "90% power MDE",
            "^",
            "#b36b2c",
            -0.18,
            [
                float(power[outcome]["mde"]["0.9"]["absolute_effect"])
                * 100
                for outcome in OUTCOMES
            ],
            4,
        ),
    ]

    figure, axis = plt.subplots(figsize=(9, 4.5))
    all_values = []
    for label, marker, color, offset, values, precision in series:
        values_array = np.asarray(values)
        all_values.extend(values)
        axis.scatter(
            values_array,
            positions + offset,
            marker=marker,
            color=color,
            s=58,
            label=label,
            zorder=3,
        )
        for value, position in zip(values, positions + offset, strict=True):
            axis.annotate(
                f"{value:.{precision}f} pp",
                xy=(value, position),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=color,
            )

    # The log scale keeps smaller MDE thresholds visible beside observed effects.
    # Exact percentage point labels preserve meaning on the transformed axis.
    axis.set_xscale("log")
    axis.set_xlim(min(all_values) * 0.65, max(all_values) * 1.8)
    axis.set_yticks(positions, labels)
    axis.set_ylim(len(labels) - 0.55, -0.55)
    axis.set_xlabel("Absolute effect in percentage points (log scale)")
    axis.set_title("Observed effects and experiment sensitivity", pad=25)
    axis.text(
        0.5,
        1.02,
        "Observed effects exceed the minimum detectable effect thresholds",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    axis.grid(axis="x", which="both", color="#dddddd", linewidth=0.7)
    axis.legend(frameon=False, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def analyze_experiment(
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    *,
    predictability_sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    figures_directory: Path | None = DEFAULT_FIGURES_DIR,
) -> dict[str, object]:
    """Run the experiment design audit and benchmark population ITT analysis."""

    parquet_path = Path(parquet_path)
    if not parquet_path.is_file():
        raise ExperimentError(f"Validated Parquet file does not exist: {parquet_path}")
    required_columns = set(FEATURE_COLUMNS).union(
        {"treatment", "exposure", *OUTCOMES}
    )
    observed_columns = set(pq.read_schema(parquet_path).names)
    missing_columns = sorted(required_columns.difference(observed_columns))
    if missing_columns:
        raise ExperimentError(
            f"Experiment dataset is missing columns: {', '.join(missing_columns)}"
        )

    population = summarize_population(parquet_path)
    allocation = allocation_audit(
        int(population["treatment_count"]),
        int(population["total_count"]),
    )
    continuous = _continuous_balance(parquet_path)
    categorical = _categorical_balance(parquet_path)
    predictability = _treatment_predictability(
        parquet_path, predictability_sample_size, random_seed
    )
    effects = estimate_outcome_effects(population)
    outcome_descriptives = {}
    power = {}
    for outcome in OUTCOMES:
        counts = population["outcomes"][outcome]
        treatment_count = int(population["treatment_count"])
        control_count = int(population["control_count"])
        outcome_descriptives[outcome] = {
            "overall_count": int(population["total_count"]),
            "overall_positive": int(counts["total_positive"]),
            "overall_rate": int(counts["total_positive"])
            / int(population["total_count"]),
            "treatment_count": treatment_count,
            "treatment_positive": int(counts["treatment_positive"]),
            "treatment_rate": int(counts["treatment_positive"])
            / treatment_count,
            "control_count": control_count,
            "control_positive": int(counts["control_positive"]),
            "control_rate": int(counts["control_positive"]) / control_count,
        }
        baseline_rate = outcome_descriptives[outcome]["control_rate"]
        power[outcome] = {
            "alpha": 0.05,
            "baseline_control_rate": baseline_rate,
            "mde": {
                str(target_power): minimum_detectable_effect(
                    baseline_rate,
                    treatment_count,
                    control_count,
                    power=target_power,
                )
                for target_power in (0.80, 0.90)
            },
        }

    # Exposure follows assignment, so it remains descriptive and never replaces
    # randomized treatment in the primary causal comparison.
    exposure_counts = population["exposure"]
    exposure = {
        "post_assignment_variable": True,
        "causal_effect_estimated": False,
        "treated_positive": int(exposure_counts["treatment_positive"]),
        "treated_count": int(population["treatment_count"]),
        "treated_exposure_rate": int(exposure_counts["treatment_positive"])
        / int(population["treatment_count"]),
        "control_positive": int(exposure_counts["control_positive"]),
        "control_count": int(population["control_count"]),
        "control_exposure_rate": int(exposure_counts["control_positive"])
        / int(population["control_count"]),
        "control_has_no_exposure": int(exposure_counts["control_positive"]) == 0,
    }

    maximum_continuous = max(
        value["absolute_standardized_mean_difference"]
        for value in continuous.values()
    )
    maximum_categorical = max(
        float(value["maximum_absolute_standardized_difference"])
        for value in categorical.values()
    )
    maximum_balance_difference = max(
        maximum_continuous, maximum_categorical
    )
    figures: list[str] = []
    if figures_directory is not None:
        figures_directory = Path(figures_directory)
        balance_path = figures_directory / "covariate-balance.png"
        effects_path = figures_directory / "treatment-effects.png"
        sensitivity_path = figures_directory / "experiment-sensitivity.png"
        _create_balance_figure(continuous, categorical, balance_path)
        _create_effect_figure(effects, effects_path)
        _create_sensitivity_figure(effects, power, sensitivity_path)
        figures = [
            str(balance_path),
            str(effects_path),
            str(sensitivity_path),
        ]

    return {
        "analysis_population": {
            "parquet_path": str(parquet_path),
            "all_rows_preserved": True,
            "total_observations": int(population["total_count"]),
            "treatment_variable": "treatment",
            "primary_outcome": "visit",
            "secondary_outcome": "conversion",
            "population_scope": "released CRITEO-UPLIFTv2.1 benchmark population",
        },
        "methods": {
            "sample_ratio_test": "two sided normal score test against 0.85",
            "risk_difference_interval": "unpooled Wald 95% confidence interval",
            "null_hypothesis_test": "two sided pooled two proportion z test",
            "relative_risk_interval": "log scale Wald 95% confidence interval",
            "mde": "two sample normal approximation using Cohen h",
            "categorical_balance": (
                "level indicator standardized differences summarized by maximum "
                "absolute and pooled frequency weighted RMS values"
            ),
        },
        "allocation": allocation,
        "covariate_balance": {
            "threshold_definition": (
                "Absolute standardized differences below 0.10 are treated as "
                "not practically meaningful."
            ),
            "continuous": continuous,
            "categorical": categorical,
            "maximum_absolute_standardized_difference": maximum_balance_difference,
            "all_features_below_threshold": maximum_balance_difference
            < BALANCE_THRESHOLD,
        },
        "treatment_predictability": predictability,
        "outcome_descriptives": outcome_descriptives,
        "intention_to_treat_effects": effects,
        "power": power,
        "exposure_diagnostics": exposure,
        "figures": figures,
        "limitations": [
            (
                "Effects apply to the released benchmark population after its "
                "documented preprocessing and should not be represented as original "
                "campaign incrementality or advertising ROI."
            ),
            (
                "Exact repeated rows are retained because the release has no unique "
                "user identifier and equality of anonymized rows does not establish "
                "record duplication."
            ),
            (
                "Exposure occurs after assignment and is summarized descriptively; "
                "it is not substituted for treatment in causal comparisons."
            ),
            (
                "Confidence intervals and tests treat released rows as independent "
                "observations because no user or experiment cluster identifier is "
                "available."
            ),
        ],
    }


def _json_scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible Criteo causal measurement analysis."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser(
        "experiment", help="Audit randomization and estimate ITT effects."
    )
    experiment.add_argument("--data", type=Path, default=DEFAULT_PARQUET_PATH)
    experiment.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE
    )
    experiment.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    experiment.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR
    )
    experiment.add_argument("--no-figures", action="store_true")
    uplift = commands.add_parser(
        "uplift", help="Fit and evaluate S Learner and T Learner baselines."
    )
    from criteo_experiment.uplift import DEFAULT_MODELING_SAMPLE_SIZE

    uplift.add_argument("--data", type=Path, default=DEFAULT_PARQUET_PATH)
    uplift.add_argument(
        "--sample-size", type=int, default=DEFAULT_MODELING_SAMPLE_SIZE
    )
    uplift.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    uplift.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR
    )
    uplift.add_argument("--no-figures", action="store_true")
    return parser


def main() -> None:
    """Run the causal measurement command line interface."""

    parser = _build_parser()
    arguments = parser.parse_args()
    figures_directory = None if arguments.no_figures else arguments.figures_dir
    from criteo_experiment.uplift import UpliftError, analyze_uplift

    try:
        if arguments.command == "experiment":
            result = analyze_experiment(
                arguments.data,
                predictability_sample_size=arguments.sample_size,
                random_seed=arguments.seed,
                figures_directory=figures_directory,
            )
        else:
            result = analyze_uplift(
                arguments.data,
                sample_size=arguments.sample_size,
                random_seed=arguments.seed,
                figures_directory=figures_directory,
            )
    except (ExperimentError, UpliftError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
