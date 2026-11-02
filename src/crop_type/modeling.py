"""Training and evaluation for the two crop-type classifiers."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from docx import Document
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .data import (BANDS, CLASSES, CUTOFFS, ID_COLS, StudyYear,
                   heldout_partition, read_study_year, signature)

LOG = logging.getLogger(__name__)
SEED = 1337

RF_SEARCH = {
    "model__n_estimators": [300, 600, 900],
    "model__max_depth": [None, 12, 24],
    "model__max_features": ["sqrt", 0.5, 0.8],
    "model__min_samples_leaf": [1, 2, 4],
    "model__min_samples_split": [2, 5],
}
LGBM_SEARCH = {
    "model__n_estimators": [200, 400, 800, 1200],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__num_leaves": [15, 31, 63],
    "model__min_child_samples": [10, 20, 40],
    "model__colsample_bytree": [0.7, 0.85, 1.0],
    "model__reg_lambda": [0.0, 1.0, 5.0],
    "model__subsample": [0.8, 1.0],
    "model__subsample_freq": [1],
}


@dataclass(frozen=True)
class RunConfig:
    algorithm: str
    year: int
    data: Path
    output_root: Path
    partition_file: Path | None = None
    pilot: bool = False
    workers: int = 1

    @property
    def n_search(self) -> int:
        return 2 if self.pilot else 12

    @property
    def outer_folds(self) -> int:
        return 2 if self.pilot else 5

    @property
    def inner_folds(self) -> int:
        return 2 if self.pilot else 3

    @property
    def n_permutations(self) -> int:
        return 2 if self.pilot else 20

    @property
    def cutoffs(self) -> tuple[str, ...]:
        return CUTOFFS[:1] if self.pilot else CUTOFFS


def build_pipeline(algorithm: str, *, n_jobs: int = -1) -> Pipeline:
    if algorithm == "RF":
        estimator = RandomForestClassifier(
            class_weight="balanced", bootstrap=True, criterion="gini",
            n_jobs=n_jobs, random_state=SEED,
        )
    elif algorithm == "LGBM":
        estimator = LGBMClassifier(
            objective="multiclass", num_class=3, class_weight="balanced",
            n_jobs=n_jobs, random_state=SEED, verbosity=-1,
        )
    else:
        raise ValueError(f"Unrecognized algorithm: {algorithm}")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=False)),
        ("variance", VarianceThreshold(threshold=0)),
        ("model", estimator),
    ])


def make_search(config: RunConfig, seed: int = SEED) -> RandomizedSearchCV:
    search_space = RF_SEARCH if config.algorithm == "RF" else LGBM_SEARCH
    if config.pilot:
        search_space = {**search_space, "model__n_estimators": [20, 40]}
    return RandomizedSearchCV(
        estimator=build_pipeline(config.algorithm),
        param_distributions=search_space,
        n_iter=config.n_search,
        scoring="f1_macro",
        cv=StratifiedGroupKFold(n_splits=config.inner_folds,
                                shuffle=True, random_state=SEED + 17),
        n_jobs=config.workers, refit=True, random_state=seed,
        return_train_score=False, error_score="raise", verbose=0,
    )


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def class_weights(targets: np.ndarray) -> dict[str, float]:
    counts = np.bincount(targets, minlength=3)
    return {name: float(len(targets) / (3 * counts[i])) if counts[i] else float("nan")
            for i, name in enumerate(CLASSES)}


def permutation_importance(model: Pipeline, features: pd.DataFrame,
                           targets: np.ndarray, repeats: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Permute every dated column of a band jointly, using a single row ordering."""
    baseline = precision_recall_fscore_support(
        targets, model.predict(features), labels=[0, 1, 2],
        average=None, zero_division=0,
    )[2]
    records = []
    for band_idx, band in enumerate(BANDS):
        columns = [col for col in features if col.startswith(f"{band}__")]
        if not columns:
            raise ValueError(f"Band {band} has no columns")
        for iteration in range(repeats):
            shuffled = features.copy(deep=True)
            rng = np.random.default_rng(SEED + 10000 * band_idx + iteration)
            order = rng.permutation(len(features))
            shuffled.loc[:, columns] = features.iloc[order][columns].to_numpy()
            after = precision_recall_fscore_support(
                targets, model.predict(shuffled), labels=[0, 1, 2],
                average=None, zero_division=0,
            )[2]
            for i, name in enumerate(CLASSES):
                records.append({
                    "band": band, "crop": name, "repeat": iteration + 1,
                    "baseline_f1": float(baseline[i]), "permuted_f1": float(after[i]),
                    "f1_drop": float(baseline[i] - after[i]),
                })
    details = pd.DataFrame.from_records(records)
    summary = (details.groupby(["band", "crop"], sort=False)["f1_drop"]
               .agg(["mean", "std"]).reset_index()
               .rename(columns={"mean": "mean_f1_drop", "std": "sd_f1_drop"}))
    summary["positive_f1_drop"] = summary["mean_f1_drop"].clip(lower=0)
    denominator = summary.groupby("crop")["positive_f1_drop"].transform("sum")
    summary["positive_share_pct"] = np.where(
        denominator > 0, 100 * summary["positive_f1_drop"] / denominator, np.nan,
    )
    shares = (summary.pivot(index="band", columns="crop", values="positive_share_pct")
              .reindex(index=BANDS, columns=CLASSES).reset_index())
    return details, summary, shares


def native_band_importance(model: Pipeline, algorithm: str) -> pd.DataFrame:
    imputer = model.named_steps["imputer"]
    variance = model.named_steps["variance"]
    names = imputer.get_feature_names_out()[variance.get_support()]
    estimator = model.named_steps["model"]
    if algorithm == "LGBM":
        values = estimator.booster_.feature_importance(importance_type="gain")
        value_name = "total_split_gain"
    else:
        values = estimator.feature_importances_
        value_name = "gini_impurity_decrease"
    if len(names) != len(values):
        raise RuntimeError("Feature-importance and transformed input shapes differ")
    frame = pd.DataFrame({"feature": names, value_name: values})
    frame["band"] = frame["feature"].str.split("__", n=1).str[0]
    return (frame.groupby("band", as_index=False)[value_name]
            .sum().sort_values(value_name, ascending=False))


def _write_report(path: Path, config: RunConfig, sections: list[dict], summary: pd.DataFrame) -> None:
    doc = Document()
    doc.add_heading(f"Crop-type classification: {config.algorithm} ({config.year})", 0)
    doc.add_paragraph("Field-level held-out evaluation and nested cross-validation")
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    for cell, name in zip(table.rows[0].cells,
                          ["Cutoff", "Accuracy", "Macro F1", "Kappa", "Nested CV macro F1"]):
        cell.text = name
    for row in summary.itertuples():
        cells = table.add_row().cells
        for cell, value in zip(cells, [row.cutoff, f"{row.test_accuracy:.4f}",
                                       f"{row.test_f1_macro:.4f}", f"{row.test_kappa:.4f}",
                                       f"{row.nested_cv_f1_macro_mean:.4f}"]):
            cell.text = value
    for section in sections:
        doc.add_heading(f"Cutoff: {section['cutoff']}", level=1)
        for heading, data in (("Confusion matrix", section["confusion"]),
                              ("Class-specific metrics", section["classification"]),
                              ("Permutation importance (%)", section["importance"])):
            doc.add_heading(heading, level=2)
            tab = doc.add_table(rows=1, cols=len(data.columns) + 1)
            tab.style = "Table Grid"
            for cell, label in zip(tab.rows[0].cells, ["Class/Index", *map(str, data.columns)]):
                cell.text = label
            for key, series in data.iterrows():
                cells = tab.add_row().cells
                for cell, value in zip(cells, [key, *series.tolist()]):
                    cell.text = f"{value:.4f}" if isinstance(value, float) else str(value)
    doc.save(path)


def run(config: RunConfig) -> dict:
    """Run one algorithm/year with the original study hyperparameter settings."""
    if config.algorithm not in ("RF", "LGBM"):
        raise ValueError(config.algorithm)
    study = read_study_year(config.data, config.year)
    train_idx, test_idx = heldout_partition(study, SEED, config.partition_file)
    y = study.targets
    groups = study.groups
    y_train, y_test = y[train_idx], y[test_idx]
    train_groups = groups[train_idx]
    output = config.output_root / f"{config.algorithm}_{config.year}_revised_results"
    if config.pilot:
        output = output.with_name(output.name + "_PILOT")
    output.mkdir(parents=True, exist_ok=True)
    train_hash = signature(study.frame, train_idx)
    test_hash = signature(study.frame, test_idx)
    LOG.info("%s %s: %d train / %d test; test signature %s",
             config.algorithm, config.year, len(train_idx), len(test_idx), test_hash)
    summary_rows, sections, fold_rows, tuning_rows, parameters = [], [], [], [], []
    predictions, importance, perm_raw, perm_summary, perm_shares = [], [], [], [], []

    for cutoff in config.cutoffs:
        features = study.features(cutoff)
        development, heldout = features.iloc[train_idx], features.iloc[test_idx]
        outer = StratifiedGroupKFold(n_splits=config.outer_folds,
                                     shuffle=True, random_state=SEED + 41)
        outer_metrics = []
        for fold_num, (tr, val) in enumerate(
                outer.split(development, y_train, train_groups), start=1):
            search = make_search(config)
            search.fit(development.iloc[tr], y_train[tr], groups=train_groups[tr])
            metrics = scores(y_train[val], search.predict(development.iloc[val]))
            outer_metrics.append(metrics)
            fold_rows.append({"algorithm": config.algorithm, "year": config.year,
                              "cutoff": cutoff, "outer_fold": fold_num,
                              "n_train": len(tr), "n_validation": len(val),
                              "inner_best_f1": float(search.best_score_), **metrics})
        final_search = make_search(config)
        final_search.fit(development, y_train, groups=train_groups)
        fitted = final_search.best_estimator_
        pred = fitted.predict(heldout)
        test_metrics = scores(y_test, pred)
        LOG.info("%s %s %s: accuracy=%.4f F1=%.4f kappa=%.4f",
                 config.algorithm, config.year, cutoff, test_metrics["accuracy"],
                 test_metrics["f1_macro"], test_metrics["kappa"])
        parameters.append({
            "year": config.year, "algorithm": config.algorithm,
            "cutoff": cutoff, "best_params": final_search.best_params_,
            "training_crop_counts": dict(zip(CLASSES, map(int, np.bincount(y_train, minlength=3)))),
            "training_class_weights": class_weights(y_train),
            "final_inner_best_f1_macro": float(final_search.best_score_),
        })
        for res in pd.DataFrame(final_search.cv_results_).itertuples():
            tuning_rows.append({
                "year": config.year, "algorithm": config.algorithm,
                "cutoff": cutoff, "n_estimators": int(res.params.get("model__n_estimators", 0)),
                "params": json.dumps(res.params, sort_keys=True),
                "rank_f1": int(res.rank_test_score),
                "mean_inner_cv_f1": float(res.mean_test_score),
                "sd_inner_cv_f1": float(res.std_test_score),
            })
        folded = pd.DataFrame(outer_metrics)
        record = {"year": config.year, "algorithm": config.algorithm,
                  "cutoff": cutoff, "n_train": len(train_idx), "n_test": len(test_idx),
                  "n_raw_features": features.shape[1],
                  "n_selected_features": len(fitted.named_steps["model"].feature_importances_),
                  "best_inner_cv_f1": float(final_search.best_score_)}
        record.update({f"test_{k}": v for k, v in test_metrics.items()})
        for name in test_metrics:
            record[f"nested_cv_{name}_mean"] = float(folded[name].mean())
            record[f"nested_cv_{name}_sd"] = float(folded[name].std(ddof=1))
        summary_rows.append(record)

        confusion = pd.DataFrame(
            confusion_matrix(y_test, pred, labels=[0, 1, 2]),
            index=[f"true_{c}" for c in CLASSES],
            columns=[f"pred_{c}" for c in CLASSES],
        )
        classification = (pd.DataFrame(classification_report(
            y_test, pred, labels=[0, 1, 2], target_names=CLASSES,
            output_dict=True, zero_division=0,
        )).T.reset_index().rename(columns={"index": "class"}))
        prediction_df = study.frame.iloc[test_idx][list(ID_COLS)].reset_index(drop=True).copy()
        prediction_df.insert(0, "year", config.year)
        prediction_df.insert(1, "cutoff", cutoff)
        prediction_df["true_class"] = [CLASSES[i] for i in y_test]
        prediction_df["predicted_class"] = [CLASSES[i] for i in pred]
        prediction_df["correct"] = prediction_df["true_class"].eq(prediction_df["predicted_class"])
        predictions.append(prediction_df)

        raw, grouped, shares = permutation_importance(
            fitted, heldout, y_test, config.n_permutations,
        )
        for frame in (raw, grouped, shares):
            frame.insert(0, "cutoff", cutoff)
            frame.insert(0, "year", config.year)
        perm_raw.append(raw)
        perm_summary.append(grouped)
        perm_shares.append(shares)
        native = native_band_importance(fitted, config.algorithm)
        native.insert(0, "cutoff", cutoff)
        native.insert(0, "year", config.year)
        importance.append(native)
        sections.append({"cutoff": cutoff, "confusion": confusion,
                         "classification": classification.set_index("class"),
                         "importance": shares.set_index("band")})
        # The fitted models are retained for subsequent analyses, not committed to Git.
        joblib.dump(fitted, output / f"model_{cutoff.replace('-', '')}.joblib", compress=3)

    exports = {
        "test_and_nested_cv_summary.csv": pd.DataFrame(summary_rows),
        "nested_cv_outer_folds.csv": pd.DataFrame(fold_rows),
        "hyperparameter_sensitivity.csv": pd.DataFrame(tuning_rows),
        "heldout_field_predictions.csv": pd.concat(predictions, ignore_index=True),
        "permutation_repeats_signed_f1.csv": pd.concat(perm_raw, ignore_index=True),
        "permutation_mean_sd_by_band_crop.csv": pd.concat(perm_summary, ignore_index=True),
        "permutation_positive_shares_pct.csv": pd.concat(perm_shares, ignore_index=True),
        "native_global_band_importance.csv": pd.concat(importance, ignore_index=True),
    }
    membership = study.frame.iloc[np.concatenate([train_idx, test_idx])][list(ID_COLS)].copy()
    membership.insert(0, "partition", ["TRAIN"] * len(train_idx) + ["TEST"] * len(test_idx))
    exports["field_partition_ids.csv"] = membership
    for filename, data in exports.items():
        data.to_csv(output / filename, index=False)
    manifest = {
        "year": config.year, "algorithm": config.algorithm,
        "seed": SEED, "test_signature": test_hash, "train_signature": train_hash,
        "test_fraction": len(test_idx) / len(study.frame),
        "train_count": len(train_idx), "test_count": len(test_idx),
        "bands": BANDS, "band_file_sha256": study.hashes,
        "group_key": "Crop|aid|County (unique record key)",
        "cutoffs": config.cutoffs, "search_iterations_per_fit": config.n_search,
        "outer_cv_folds": config.outer_folds, "inner_cv_folds": config.inner_folds,
        "permutation_repeats": config.n_permutations,
        "class_weight_scheme": "balanced: n_training/(3*n_class_training), recomputed in every fit",
        "software_versions": {
            "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
        },
        "partition_mode": "fixed" if config.partition_file else "generated",
        "pilot": config.pilot,
        "selected_parameters": parameters,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    _write_report(output / f"{config.algorithm}_{config.year}_report.docx",
                  config, sections, exports["test_and_nested_cv_summary.csv"])
    return {"output_dir": str(output), "summary": exports["test_and_nested_cv_summary.csv"],
            "train_signature": train_hash, "test_signature": test_hash}
