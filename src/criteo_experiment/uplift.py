"""Fit and evaluate reproducible uplift modeling baselines."""

from __future__ import annotations

import gc
import math
import resource
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.proportion import confint_proportions_2indep

from criteo_experiment.dataset import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    DEFAULT_PARQUET_PATH,
    FEATURE_COLUMNS,
)
from criteo_experiment.experiment import (
    estimate_outcome_effects,
    summarize_population,
)

PREFERRED_MODELING_SAMPLE_SIZE = 5_000_000
DEFAULT_MODELING_SAMPLE_SIZE = 2_000_000
DEFAULT_RANDOM_SEED = 42
TREATMENT_PROPENSITY = 0.85
OUTCOMES = ("visit", "conversion")
TARGETING_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
REGULARIZATION_CANDIDATES = (1e-5, 1e-4)
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_FIGURES_DIR = Path("reports/figures")
MINIMUM_CONTROL_EVENTS = 20
MODEL_FEATURES = FEATURE_COLUMNS
FORBIDDEN_MODEL_FEATURES = {"treatment", "exposure", "visit", "conversion"}


class UpliftError(RuntimeError):
    """Indicate that the uplift baseline workflow could not complete."""


def validate_model_features(feature_names: Sequence[str]) -> None:
    """Require the documented pretreatment feature set and reject leakage."""

    observed = set(feature_names)
    leaked = sorted(observed.intersection(FORBIDDEN_MODEL_FEATURES))
    if leaked:
        raise ValueError(
            f"Model features contain treatment or post assignment data: {leaked}"
        )
    expected = set(MODEL_FEATURES)
    if observed != expected or len(feature_names) != len(MODEL_FEATURES):
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        raise ValueError(
            "Model features must match f0 through f11: "
            f"missing={missing}, unexpected={unexpected}"
        )


def sample_modeling_data(
    parquet_path: Path,
    sample_size: int,
    random_seed: int,
) -> pd.DataFrame:
    """Select an exact reproducible row sample without changing source data."""

    validate_model_features(MODEL_FEATURES)
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    if sample_size <= 0:
        raise ValueError("Modeling sample size must be positive")
    if sample_size > total_rows:
        raise ValueError(
            f"Modeling sample size {sample_size} exceeds {total_rows} source rows"
        )

    # Only pretreatment features and randomized design fields are materialized.
    columns = [*MODEL_FEATURES, "treatment", *OUTCOMES]
    generator = np.random.default_rng(random_seed)
    selected_indices = np.sort(
        generator.choice(total_rows, size=sample_size, replace=False)
    )
    selected_batches = []
    selected_position = 0
    row_offset = 0
    for batch in parquet_file.iter_batches(columns=columns, batch_size=262_144):
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

    sampled_table = pa.Table.from_batches(selected_batches)
    sampled = sampled_table.to_pandas(split_blocks=True)
    sampled["_source_row"] = selected_indices
    for name in (*CONTINUOUS_FEATURES, "treatment", *OUTCOMES):
        dtype = np.float32 if name in CONTINUOUS_FEATURES else np.int8
        sampled[name] = sampled[name].astype(dtype, copy=False)
    return sampled


def _stratification_key(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["treatment"].to_numpy(dtype=np.int8) * 4
        + frame["visit"].to_numpy(dtype=np.int8) * 2
        + frame["conversion"].to_numpy(dtype=np.int8)
    )


def split_modeling_data(
    frame: pd.DataFrame,
    random_seed: int,
) -> dict[str, pd.DataFrame]:
    """Create reproducible 60%, 20%, and 20% stratified populations."""

    strata = _stratification_key(frame)
    _, stratum_counts = np.unique(strata, return_counts=True)
    if int(stratum_counts.min()) < 5:
        raise ValueError("Every observed split stratum must contain at least five rows")

    indices = np.arange(len(frame))
    train_indices, holdout_indices = train_test_split(
        indices,
        test_size=0.40,
        random_state=random_seed,
        stratify=strata,
    )
    validation_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=0.50,
        random_state=random_seed,
        stratify=strata[holdout_indices],
    )
    return {
        "train": frame.iloc[train_indices].reset_index(drop=True),
        "validation": frame.iloc[validation_indices].reset_index(drop=True),
        "test": frame.iloc[test_indices].reset_index(drop=True),
    }


def summarize_split(frame: pd.DataFrame) -> dict[str, object]:
    """Report treatment and outcome distributions for one modeling split."""

    treatment = frame["treatment"].to_numpy(dtype=np.int8)
    summary: dict[str, object] = {
        "observations": len(frame),
        "treatment_count": int(treatment.sum()),
        "control_count": int(len(frame) - treatment.sum()),
        "treatment_rate": float(treatment.mean()),
    }
    for outcome in OUTCOMES:
        values = frame[outcome].to_numpy(dtype=np.int8)
        treatment_values = values[treatment == 1]
        control_values = values[treatment == 0]
        summary[outcome] = {
            "positive": int(values.sum()),
            "rate": float(values.mean()),
            "treatment_positive": int(treatment_values.sum()),
            "treatment_rate": float(treatment_values.mean()),
            "control_positive": int(control_values.sum()),
            "control_rate": float(control_values.mean()),
        }
    return summary


def build_feature_preprocessor() -> ColumnTransformer:
    """Create sparse preprocessing that respects documented feature roles."""

    validate_model_features(MODEL_FEATURES)
    return ColumnTransformer(
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


def fit_feature_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Fit feature preprocessing on training data only."""

    preprocessor = build_feature_preprocessor()
    preprocessor.fit(frame.loc[:, list(MODEL_FEATURES)])
    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
) -> sparse.csr_matrix:
    """Apply an already fitted training preprocessor to another split."""

    transformed = preprocessor.transform(frame.loc[:, list(MODEL_FEATURES)])
    return sparse.csr_matrix(transformed, dtype=np.float32)


def transformed_outcome(
    outcome: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    propensity: float = TREATMENT_PROPENSITY,
) -> np.ndarray:
    """Calculate the inverse propensity weighted transformed outcome."""

    if not 0 < propensity < 1:
        raise ValueError("Treatment propensity must fall strictly between zero and one")
    outcome_array = np.asarray(outcome, dtype=np.float64)
    treatment_array = np.asarray(treatment, dtype=np.float64)
    if outcome_array.shape != treatment_array.shape:
        raise ValueError("Outcome and treatment arrays must have equal shapes")
    if not np.isin(treatment_array, (0, 1)).all():
        raise ValueError("Treatment values must be binary")

    # Unequal assignment requires both inverse propensity weights for unbiased Z.
    return (
        outcome_array * treatment_array / propensity
        - outcome_array * (1 - treatment_array) / (1 - propensity)
    )


def _new_classifier(
    alpha: float,
    random_seed: int,
    max_iterations: int,
) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iterations,
        tol=1e-4,
        shuffle=True,
        random_state=random_seed,
        average=True,
    )


def _fit_classifier(
    classifier: SGDClassifier,
    features: sparse.csr_matrix,
    outcome: np.ndarray,
) -> tuple[float, bool]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(features, outcome)
    elapsed = time.perf_counter() - started
    warned = any(issubclass(item.category, ConvergenceWarning) for item in captured)
    return elapsed, warned


def _positive_probability(
    classifier: object,
    features: sparse.csr_matrix,
) -> np.ndarray:
    probabilities = np.asarray(classifier.predict_proba(features))[:, 1]
    return np.clip(probabilities, 1e-7, 1 - 1e-7)


def _s_learner_design(
    features: sparse.csr_matrix,
    treatment: np.ndarray,
) -> sparse.csr_matrix:
    treatment_column = sparse.csr_matrix(
        np.asarray(treatment, dtype=np.float32).reshape(-1, 1)
    )
    interactions = features.multiply(treatment_column)
    # Interactions let a linear S Learner vary treatment response with every feature.
    return sparse.hstack(
        [features, treatment_column, interactions],
        format="csr",
        dtype=np.float32,
    )


def predict_s_learner_potential_outcomes(
    classifier: object,
    features: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict S Learner outcomes under treatment and control."""

    control = np.zeros(features.shape[0], dtype=np.float32)
    treatment = np.ones(features.shape[0], dtype=np.float32)
    control_design = _s_learner_design(features, control)
    control_probability = _positive_probability(classifier, control_design)
    del control_design
    treatment_design = _s_learner_design(features, treatment)
    treatment_probability = _positive_probability(classifier, treatment_design)
    return treatment_probability, control_probability


def predict_t_learner_potential_outcomes(
    treatment_classifier: object,
    control_classifier: object,
    features: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict T Learner outcomes under treatment and control."""

    return (
        _positive_probability(treatment_classifier, features),
        _positive_probability(control_classifier, features),
    )


@dataclass
class SLearnerModel:
    """Store one sparse logistic S Learner baseline."""

    classifier: SGDClassifier
    fit_seconds: float
    convergence_warning: bool

    @classmethod
    def fit(
        cls,
        features: sparse.csr_matrix,
        treatment: np.ndarray,
        outcome: np.ndarray,
        *,
        alpha: float,
        random_seed: int,
        max_iterations: int,
    ) -> SLearnerModel:
        classifier = _new_classifier(alpha, random_seed, max_iterations)
        design = _s_learner_design(features, treatment)
        fit_seconds, warned = _fit_classifier(classifier, design, outcome)
        del design
        return cls(classifier, fit_seconds, warned)

    def predict_potential_outcomes(
        self, features: sparse.csr_matrix
    ) -> tuple[np.ndarray, np.ndarray]:
        return predict_s_learner_potential_outcomes(self.classifier, features)

    def convergence(self, max_iterations: int) -> dict[str, object]:
        iterations = int(self.classifier.n_iter_)
        return {
            "iterations": iterations,
            "maximum_iterations": max_iterations,
            "reached_iteration_limit": iterations >= max_iterations,
            "convergence_warning": self.convergence_warning,
        }


@dataclass
class TLearnerModel:
    """Store separate sparse logistic models for treatment and control."""

    treatment_classifier: SGDClassifier
    control_classifier: SGDClassifier
    fit_seconds: float
    treatment_convergence_warning: bool
    control_convergence_warning: bool

    @classmethod
    def fit(
        cls,
        features: sparse.csr_matrix,
        treatment: np.ndarray,
        outcome: np.ndarray,
        *,
        alpha: float,
        random_seed: int,
        max_iterations: int,
    ) -> TLearnerModel:
        treated = treatment == 1
        control = ~treated
        treatment_classifier = _new_classifier(
            alpha, random_seed, max_iterations
        )
        control_classifier = _new_classifier(
            alpha, random_seed, max_iterations
        )
        treatment_seconds, treatment_warned = _fit_classifier(
            treatment_classifier, features[treated], outcome[treated]
        )
        control_seconds, control_warned = _fit_classifier(
            control_classifier, features[control], outcome[control]
        )
        return cls(
            treatment_classifier,
            control_classifier,
            treatment_seconds + control_seconds,
            treatment_warned,
            control_warned,
        )

    def predict_potential_outcomes(
        self, features: sparse.csr_matrix
    ) -> tuple[np.ndarray, np.ndarray]:
        return predict_t_learner_potential_outcomes(
            self.treatment_classifier,
            self.control_classifier,
            features,
        )

    def convergence(self, max_iterations: int) -> dict[str, object]:
        treatment_iterations = int(self.treatment_classifier.n_iter_)
        control_iterations = int(self.control_classifier.n_iter_)
        return {
            "treatment_iterations": treatment_iterations,
            "control_iterations": control_iterations,
            "maximum_iterations": max_iterations,
            "treatment_reached_iteration_limit": (
                treatment_iterations >= max_iterations
            ),
            "control_reached_iteration_limit": (
                control_iterations >= max_iterations
            ),
            "treatment_convergence_warning": self.treatment_convergence_warning,
            "control_convergence_warning": self.control_convergence_warning,
        }


def _stable_descending_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores), kind="mergesort")


def cumulative_gain_curve(
    scores: np.ndarray,
    evaluation_signal: np.ndarray,
    *,
    curve_points: int = 100,
) -> dict[str, object]:
    """Calculate deterministic policy gain and random targeting curves."""

    if curve_points <= 0:
        raise ValueError("Curve points must be positive")
    scores = np.asarray(scores, dtype=np.float64)
    signal = np.asarray(evaluation_signal, dtype=np.float64)
    if scores.shape != signal.shape or scores.ndim != 1:
        raise ValueError("Scores and evaluation signal must be equal length vectors")
    if len(scores) == 0:
        raise ValueError("Ranking evaluation requires observations")

    order = _stable_descending_order(scores)
    cumulative = np.cumsum(signal[order])
    fractions = np.linspace(0.0, 1.0, curve_points + 1)
    counts = np.ceil(fractions * len(scores)).astype(int)
    counts[0] = 0
    policy_gain = np.zeros_like(fractions)
    selected = counts > 0
    policy_gain[selected] = cumulative[counts[selected] - 1] / len(scores) * 100_000
    treat_all = float(signal.mean() * 100_000)
    random_gain = fractions * treat_all
    difference = policy_gain - random_gain
    interval_widths = np.diff(fractions)
    auuc_above_random = float(
        np.sum((difference[:-1] + difference[1:]) / 2 * interval_widths)
    )
    return {
        "definition": (
            "Trapezoidal area between ranked and random cumulative inverse "
            "propensity weighted gain curves, in incremental outcomes per "
            "100,000 population members."
        ),
        "fractions": fractions.tolist(),
        "targeted_counts": counts.tolist(),
        "policy_gain_per_100000": policy_gain.tolist(),
        "random_gain_per_100000": random_gain.tolist(),
        "treat_all_gain_per_100000": treat_all,
        "auuc_above_random_per_100000": auuc_above_random,
    }


def _observed_group_effect(
    outcome: np.ndarray,
    treatment: np.ndarray,
) -> dict[str, object]:
    treated = treatment == 1
    control = ~treated
    if not treated.any() or not control.any():
        raise ValueError("An evaluated group must contain treatment and control rows")
    treatment_count = int(treated.sum())
    control_count = int(control.sum())
    treatment_positive = int(outcome[treated].sum())
    control_positive = int(outcome[control].sum())
    treatment_rate = treatment_positive / treatment_count
    control_rate = control_positive / control_count
    confidence_interval = confint_proportions_2indep(
        treatment_positive,
        treatment_count,
        control_positive,
        control_count,
        compare="diff",
        method="newcomb",
    )
    return {
        "treatment_count": treatment_count,
        "control_count": control_count,
        "treatment_positive": treatment_positive,
        "control_positive": control_positive,
        "treatment_rate": treatment_rate,
        "control_rate": control_rate,
        "observed_difference": treatment_rate - control_rate,
        "observed_difference_ci": [float(value) for value in confidence_interval],
        "confidence_level": 0.95,
        "confidence_interval_method": "Newcombe score interval",
    }


def targeting_policy_table(
    scores: np.ndarray,
    evaluation_signal: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    fractions: Sequence[float] = TARGETING_FRACTIONS,
    *,
    outcome_name: str | None = None,
) -> list[dict[str, object]]:
    """Evaluate fixed targeting fractions on one held out population."""

    scores = np.asarray(scores, dtype=np.float64)
    signal = np.asarray(evaluation_signal, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.int8)
    treatment = np.asarray(treatment, dtype=np.int8)
    if not (scores.shape == signal.shape == outcome.shape == treatment.shape):
        raise ValueError("Targeting evaluation arrays must have equal shapes")
    order = _stable_descending_order(scores)
    total = len(scores)
    treat_all_effect = _observed_group_effect(outcome, treatment)
    treat_all_rate = float(treat_all_effect["observed_difference"])
    rows = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Targeting fractions must fall within zero and one")
        targeted_count = min(total, max(1, math.ceil(fraction * total)))
        targeted = order[:targeted_count]
        effect = _observed_group_effect(
            outcome[targeted], treatment[targeted]
        )
        actual_fraction = targeted_count / total
        targeted_rate = float(effect["observed_difference"])
        policy_per_100000 = targeted_rate * actual_fraction * 100_000
        random_per_100000 = treat_all_rate * actual_fraction * 100_000
        rows.append(
            {
                "fraction": float(fraction),
                "targeted_count": targeted_count,
                "estimation_method": (
                    "Held out treatment minus control rate, scaled by the targeted "
                    "population fraction"
                ),
                "estimated_incremental_outcome_rate_among_targeted": targeted_rate,
                "incremental_outcomes_per_100000_population": policy_per_100000,
                "random_targeting_per_100000_population": random_per_100000,
                "incremental_outcomes_above_random_per_100000": (
                    policy_per_100000 - random_per_100000
                ),
                "ipw_transformed_outcome_rate_among_targeted": float(
                    signal[targeted].mean()
                ),
                "observed_group_effect": effect,
                "insufficient_control_events": (
                    outcome_name == "conversion"
                    and int(effect["control_positive"])
                    < MINIMUM_CONTROL_EVENTS
                ),
                "control_event_threshold": MINIMUM_CONTROL_EVENTS,
            }
        )
    return rows


def uplift_decile_table(
    scores: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    *,
    outcome_name: str,
    groups: int = 10,
) -> list[dict[str, object]]:
    """Summarize predicted and observed uplift in held out ranked groups."""

    order = _stable_descending_order(scores)
    rows = []
    for rank, indices in enumerate(np.array_split(order, groups), start=1):
        effect = _observed_group_effect(outcome[indices], treatment[indices])
        control_events = int(effect["control_positive"])
        rows.append(
            {
                "rank": rank,
                "rank_definition": "1 is highest predicted uplift",
                "observations": len(indices),
                "mean_predicted_uplift": float(np.mean(scores[indices])),
                **effect,
                "insufficient_control_events": (
                    outcome_name == "conversion"
                    and control_events < MINIMUM_CONTROL_EVENTS
                ),
                "control_event_threshold": MINIMUM_CONTROL_EVENTS,
            }
        )
    return rows


def _safe_auc(outcome: np.ndarray, probability: np.ndarray) -> float | None:
    if np.unique(outcome).size < 2:
        return None
    return float(roc_auc_score(outcome, probability))


def predictive_diagnostics(
    outcome: np.ndarray,
    treatment: np.ndarray,
    treatment_probability: np.ndarray,
    control_probability: np.ndarray,
) -> dict[str, object]:
    """Report factual prediction checks without treating them as uplift metrics."""

    treated = treatment == 1
    control = ~treated
    factual_probability = np.where(
        treated, treatment_probability, control_probability
    )

    def group_metrics(mask: np.ndarray, probability: np.ndarray) -> dict[str, object]:
        values = outcome[mask]
        predictions = probability[mask]
        return {
            "observations": int(mask.sum()),
            "positive": int(values.sum()),
            "roc_auc": _safe_auc(values, predictions),
            "log_loss": float(log_loss(values, predictions, labels=[0, 1])),
        }

    return {
        "role": (
            "Factual outcome sanity checks only; uplift ranking determines model "
            "selection."
        ),
        "overall": {
            "roc_auc": _safe_auc(outcome, factual_probability),
            "log_loss": float(
                log_loss(outcome, factual_probability, labels=[0, 1])
            ),
        },
        "treated_p1": group_metrics(treated, treatment_probability),
        "control_p0": group_metrics(control, control_probability),
    }


def cate_summary(
    scores: np.ndarray,
    held_out_test_ate: float,
    full_benchmark_itt: float,
) -> dict[str, float]:
    """Summarize an estimated conditional treatment effect distribution."""

    values = np.asarray(scores, dtype=np.float64)
    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    mean = float(values.mean())
    return {
        "mean": mean,
        "standard_deviation": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "percentile_05": float(quantiles[0]),
        "percentile_25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "percentile_75": float(quantiles[3]),
        "percentile_95": float(quantiles[4]),
        "maximum": float(values.max()),
        "proportion_positive": float(np.mean(values > 0)),
        "proportion_negative": float(np.mean(values < 0)),
        "held_out_test_ate": held_out_test_ate,
        "mean_minus_held_out_test_ate": mean - held_out_test_ate,
        "full_benchmark_itt": full_benchmark_itt,
        "mean_minus_full_benchmark_itt": mean - full_benchmark_itt,
    }


def _matrix_storage_bytes(matrix: sparse.csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _model_report(
    learner_name: str,
    train_features: sparse.csr_matrix,
    train_treatment: np.ndarray,
    train_outcome: np.ndarray,
    validation_features: sparse.csr_matrix,
    validation_treatment: np.ndarray,
    validation_outcome: np.ndarray,
    validation_signal: np.ndarray,
    *,
    random_seed: int,
    max_iterations: int,
) -> tuple[SLearnerModel | TLearnerModel, dict[str, object]]:
    candidates = []
    selected_model: SLearnerModel | TLearnerModel | None = None
    selected_score = -math.inf
    selected_alpha = None

    for alpha in REGULARIZATION_CANDIDATES:
        if learner_name == "s_learner":
            model: SLearnerModel | TLearnerModel = SLearnerModel.fit(
                train_features,
                train_treatment,
                train_outcome,
                alpha=alpha,
                random_seed=random_seed,
                max_iterations=max_iterations,
            )
        else:
            model = TLearnerModel.fit(
                train_features,
                train_treatment,
                train_outcome,
                alpha=alpha,
                random_seed=random_seed,
                max_iterations=max_iterations,
            )
        p1, p0 = model.predict_potential_outcomes(validation_features)
        ranking = cumulative_gain_curve(p1 - p0, validation_signal)
        selection_score = float(ranking["auuc_above_random_per_100000"])
        candidate = {
            "alpha": alpha,
            "validation_auuc_above_random_per_100000": selection_score,
            "fit_seconds": model.fit_seconds,
            "convergence": model.convergence(max_iterations),
        }
        candidates.append(candidate)
        if selection_score > selected_score:
            selected_model = model
            selected_score = selection_score
            selected_alpha = alpha
        del p1, p0
        gc.collect()

    assert selected_model is not None
    return selected_model, {
        "estimator": "SGDClassifier with logistic loss and L2 regularization",
        "selection_rule": (
            "Maximum validation AUUC above random; predictive AUC is not the "
            "selection criterion."
        ),
        "regularization_candidates": list(REGULARIZATION_CANDIDATES),
        "candidates": candidates,
        "selected_alpha": selected_alpha,
        "selected_validation_auuc_above_random_per_100000": selected_score,
        "selected_fit_seconds": selected_model.fit_seconds,
        "selected_convergence": selected_model.convergence(max_iterations),
    }


def _evaluate_model(
    model: SLearnerModel | TLearnerModel,
    test_features: sparse.csr_matrix,
    test_outcome: np.ndarray,
    test_treatment: np.ndarray,
    test_signal: np.ndarray,
    held_out_test_ate: float,
    full_benchmark_itt: float,
    outcome_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    # Final ranking metrics are computed only after validation selects the model.
    p1, p0 = model.predict_potential_outcomes(test_features)
    scores = p1 - p0
    ranking = cumulative_gain_curve(scores, test_signal)
    policy = targeting_policy_table(
        scores,
        test_signal,
        test_outcome,
        test_treatment,
        outcome_name=outcome_name,
    )
    result = {
        "predictive_diagnostics": predictive_diagnostics(
            test_outcome, test_treatment, p1, p0
        ),
        "ranking": ranking,
        "targeting_policy": policy,
        "uplift_deciles": uplift_decile_table(
            scores,
            test_outcome,
            test_treatment,
            outcome_name=outcome_name,
        ),
        "estimated_cate_distribution": cate_summary(
            scores,
            held_out_test_ate,
            full_benchmark_itt,
        ),
    }
    figure_data = {
        "ranking": ranking,
        "targeting_policy": policy,
    }
    return result, figure_data


def _create_gain_figure(
    outcome_name: str,
    models: Mapping[str, Mapping[str, object]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"s_learner": "S Learner", "t_learner": "T Learner"}
    colors = {"s_learner": "#2864a6", "t_learner": "#b36b2c"}
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for learner_name, data in models.items():
        ranking = data["ranking"]
        axis.plot(
            np.asarray(ranking["fractions"]) * 100,
            ranking["policy_gain_per_100000"],
            label=labels[learner_name],
            color=colors[learner_name],
            linewidth=2,
        )
    reference = models["s_learner"]["ranking"]
    axis.plot(
        np.asarray(reference["fractions"]) * 100,
        reference["random_gain_per_100000"],
        label="Random targeting",
        color="#666666",
        linestyle="--",
    )
    axis.axhline(
        float(reference["treat_all_gain_per_100000"]),
        label="Treat all total gain",
        color="#4f8a5b",
        linestyle=":",
    )
    axis.set_xlabel("Population targeted (%)")
    axis.set_ylabel(f"Incremental {outcome_name}s per 100,000 population members")
    axis.set_title(
        "Higher cumulative gain at the same targeting fraction is better",
        fontsize=9.5,
        color="#555555",
        pad=8,
    )
    figure.suptitle(
        f"{outcome_name.title()} uplift ranking on the held out test set",
        y=0.98,
    )
    axis.legend(frameon=False)
    axis.grid(color="#e2e2e2", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _create_policy_figure(
    outcomes: Mapping[str, Mapping[str, Mapping[str, object]]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"s_learner": "S Learner", "t_learner": "T Learner"}
    colors = {"s_learner": "#2864a6", "t_learner": "#b36b2c"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, outcome_name in zip(axes, OUTCOMES, strict=True):
        reference_policy = outcomes[outcome_name]["s_learner"][
            "targeting_policy"
        ]
        fractions = np.array([row["fraction"] for row in reference_policy]) * 100
        for learner_name in ("s_learner", "t_learner"):
            policy = outcomes[outcome_name][learner_name]["targeting_policy"]
            gains = [
                row["incremental_outcomes_per_100000_population"]
                for row in policy
            ]
            axis.plot(
                fractions,
                gains,
                marker="o",
                label=labels[learner_name],
                color=colors[learner_name],
            )
        random_gain = [
            row["random_targeting_per_100000_population"]
            for row in reference_policy
        ]
        axis.plot(
            fractions,
            random_gain,
            label="Random targeting",
            color="#666666",
            linestyle="--",
        )
        axis.set_title(outcome_name.title())
        axis.set_xlabel("Population targeted (%)")
        axis.set_ylabel("Incremental outcomes per 100,000")
        axis.grid(color="#e2e2e2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
    )
    figure.suptitle("Held out targeting policy comparison", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _peak_memory_bytes() -> int:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def analyze_uplift(
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    *,
    sample_size: int = DEFAULT_MODELING_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    propensity: float = TREATMENT_PROPENSITY,
    figures_directory: Path | None = DEFAULT_FIGURES_DIR,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> dict[str, object]:
    """Fit and evaluate S Learner and T Learner uplift baselines."""

    started = time.perf_counter()
    parquet_path = Path(parquet_path)
    if not parquet_path.is_file():
        raise UpliftError(f"Validated Parquet file does not exist: {parquet_path}")
    if not 0 < propensity < 1:
        raise ValueError("Treatment propensity must fall strictly between zero and one")
    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be positive")
    required_columns = {*MODEL_FEATURES, "treatment", *OUTCOMES}
    observed_columns = set(pq.read_schema(parquet_path).names)
    missing = sorted(required_columns.difference(observed_columns))
    if missing:
        raise UpliftError(f"Modeling dataset is missing columns: {missing}")

    source_rows = pq.ParquetFile(parquet_path).metadata.num_rows
    sampled = sample_modeling_data(parquet_path, sample_size, random_seed)
    splits = split_modeling_data(sampled, random_seed)
    split_summaries = {
        name: summarize_split(frame) for name, frame in splits.items()
    }
    del sampled
    gc.collect()

    # Fitting once on train prevents category and scaling information from leaking.
    preprocessor = fit_feature_preprocessor(splits["train"])
    matrices = {
        name: transform_features(preprocessor, frame)
        for name, frame in splits.items()
    }
    matrix_report = {
        name: {
            "rows": matrix.shape[0],
            "columns": matrix.shape[1],
            "nonzero_values": int(matrix.nnz),
            "storage_bytes": _matrix_storage_bytes(matrix),
        }
        for name, matrix in matrices.items()
    }

    population = summarize_population(parquet_path)
    experimental_effects = estimate_outcome_effects(population)
    outcome_results: dict[str, object] = {}
    figure_results: dict[str, dict[str, object]] = {}
    for outcome_name in OUTCOMES:
        train_outcome = splits["train"][outcome_name].to_numpy(dtype=np.int8)
        validation_outcome = splits["validation"][outcome_name].to_numpy(
            dtype=np.int8
        )
        test_outcome = splits["test"][outcome_name].to_numpy(dtype=np.int8)
        train_treatment = splits["train"]["treatment"].to_numpy(dtype=np.int8)
        validation_treatment = splits["validation"]["treatment"].to_numpy(
            dtype=np.int8
        )
        test_treatment = splits["test"]["treatment"].to_numpy(dtype=np.int8)
        validation_signal = transformed_outcome(
            validation_outcome, validation_treatment, propensity
        )
        test_signal = transformed_outcome(
            test_outcome, test_treatment, propensity
        )
        full_benchmark_itt = float(
            experimental_effects[outcome_name]["absolute_risk_difference"]
        )
        held_out_test_ate = float(
            _observed_group_effect(test_outcome, test_treatment)[
                "observed_difference"
            ]
        )
        model_results = {}
        model_figure_results = {}
        for learner_name in ("s_learner", "t_learner"):
            model, fitting = _model_report(
                learner_name,
                matrices["train"],
                train_treatment,
                train_outcome,
                matrices["validation"],
                validation_treatment,
                validation_outcome,
                validation_signal,
                random_seed=random_seed,
                max_iterations=max_iterations,
            )
            evaluation, figure_data = _evaluate_model(
                model,
                matrices["test"],
                test_outcome,
                test_treatment,
                test_signal,
                held_out_test_ate,
                full_benchmark_itt,
                outcome_name,
            )
            model_results[learner_name] = {
                "fitting": fitting,
                "held_out_test_evaluation": evaluation,
            }
            model_figure_results[learner_name] = figure_data
            del model
            gc.collect()
        outcome_results[outcome_name] = {
            "held_out_test_ate": held_out_test_ate,
            "full_benchmark_itt": full_benchmark_itt,
            "models": model_results,
        }
        figure_results[outcome_name] = model_figure_results

    figures = []
    if figures_directory is not None:
        figures_directory = Path(figures_directory)
        for outcome_name in OUTCOMES:
            output_path = figures_directory / f"{outcome_name}-uplift-gain.png"
            _create_gain_figure(
                outcome_name, figure_results[outcome_name], output_path
            )
            figures.append(str(output_path))
        policy_path = figures_directory / "uplift-targeting-policy.png"
        _create_policy_figure(figure_results, policy_path)
        figures.append(str(policy_path))

    result = {
        "modeling_population": {
            "parquet_path": str(parquet_path),
            "source_observations": source_rows,
            "source_rows_modified_or_deduplicated": False,
            "preferred_sample_size": PREFERRED_MODELING_SAMPLE_SIZE,
            "actual_sample_size": sample_size,
            "sample_without_replacement": True,
            "random_seed": random_seed,
            "split_strategy": (
                "60% train, 20% validation, and 20% test stratified by treatment, "
                "visit, and conversion"
            ),
            "splits": split_summaries,
        },
        "feature_design": {
            "model_features": list(MODEL_FEATURES),
            "excluded_variables": [
                "treatment",
                "exposure",
                "visit",
                "conversion",
            ],
            "continuous_features": list(CONTINUOUS_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "continuous_preprocessing": "training fitted standard scaling",
            "categorical_preprocessing": (
                "training fitted sparse one hot encoding with unknown categories "
                "ignored"
            ),
            "transformed_matrices": matrix_report,
        },
        "methods": {
            "treatment_propensity": propensity,
            "transformed_outcome": "Z = Y*T/p - Y*(1-T)/(1-p)",
            "s_learner": (
                "Sparse logistic model with treatment main effect and treatment "
                "interactions for every transformed pretreatment feature"
            ),
            "t_learner": "Separate sparse logistic treatment and control models",
            "model_selection_population": "validation split only",
            "final_evaluation_population": "untouched test split only",
            "targeting_uncertainty": (
                "Newcombe 95% score intervals for observed treatment minus control "
                "rates in held out targeted groups"
            ),
            "targeting_policy_value": (
                "Held out treatment minus control rate multiplied by the targeted "
                "population fraction; inverse propensity weighting remains the "
                "ranking evaluation signal"
            ),
        },
        "outcomes": outcome_results,
        "figures": figures,
        "limitations": [
            (
                "Models use a reproducible sample for local computational "
                "practicality rather than all released rows."
            ),
            (
                "Conversion is rare, so control event warnings identify ranked "
                "groups with unstable observed differences."
            ),
            (
                "Targeting group confidence intervals quantify outcome sampling "
                "uncertainty conditional on the fitted ranking. They exclude "
                "uncertainty from model training or refitting."
            ),
            (
                "Results apply to the released CRITEO-UPLIFTv2.1 benchmark "
                "population and do not establish production campaign value."
            ),
            (
                "Predicted uplift estimates conditional treatment effects; no "
                "individual counterfactual outcome is observed."
            ),
        ],
        "runtime_seconds": time.perf_counter() - started,
        "approximate_peak_memory_bytes": _peak_memory_bytes(),
    }
    return result
