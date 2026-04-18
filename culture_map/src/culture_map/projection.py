import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS, MANUAL_GPT3_DEFAULT_SCORES, PART1_TABLES, PART2_GROUPS
from .paper_assets import ensure_assets


class ProjectionModel(object):
    def __init__(self, coef_rc1, coef_rc2, features=None, provenance=None, validation=None):
        self.features = list(features or FEATURE_COLUMNS)
        self.coef_rc1 = np.asarray(coef_rc1, dtype=float)
        self.coef_rc2 = np.asarray(coef_rc2, dtype=float)
        self.provenance = provenance or {}
        self.validation = validation or {}

    def project(self, frame):
        missing = [column for column in self.features if column not in frame.columns]
        if missing:
            raise ValueError("Missing feature columns: {}".format(", ".join(missing)))
        matrix = frame[self.features].astype(float).to_numpy()
        design = np.column_stack([np.ones(len(matrix)), matrix])
        projected = frame.copy()
        projected["RC1"] = design.dot(self.coef_rc1)
        projected["RC2"] = design.dot(self.coef_rc2)
        return projected

    def to_dict(self):
        return {
            "features": self.features,
            "coef_rc1": self.coef_rc1.tolist(),
            "coef_rc2": self.coef_rc2.tolist(),
            "provenance": self.provenance,
            "validation": self.validation,
        }


def _fit_affine(features_frame, target_series):
    clean = pd.concat([features_frame, target_series], axis=1).dropna()
    matrix = clean[features_frame.columns].astype(float).to_numpy()
    design = np.column_stack([np.ones(len(matrix)), matrix])
    target = clean[target_series.name].astype(float).to_numpy()
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predictions = design.dot(coefficients)
    residual = target - predictions
    max_error = float(np.max(np.abs(residual))) if len(residual) else 0.0
    rmse = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else 0.0
    return coefficients, max_error, rmse


def derive_projection_model(data_dir):
    ensure_assets(data_dir)
    data_dir = Path(data_dir)
    feature_blocks = []
    rc1_blocks = []
    rc2_blocks = []
    validation = {}

    for model_name, spec in PART2_GROUPS.items():
        scores = pd.read_csv(data_dir / spec["scores_filename"])
        coords = pd.read_csv(data_dir / spec["coords_filename"])

        averaged = scores.groupby("country")[FEATURE_COLUMNS].mean().reset_index()
        merged = averaged.merge(coords, left_on="country", right_on="country.territory", how="inner")

        feature_blocks.append(merged[FEATURE_COLUMNS])
        rc1_blocks.append(merged[spec["rc1_column"]].rename("RC1"))
        rc2_blocks.append(merged[spec["rc2_column"]].rename("RC2"))

    features = pd.concat(feature_blocks, ignore_index=True)
    rc1_target = pd.concat(rc1_blocks, ignore_index=True)
    rc2_target = pd.concat(rc2_blocks, ignore_index=True)

    coef_rc1, _, _ = _fit_affine(features, rc1_target)
    coef_rc2, _, _ = _fit_affine(features, rc2_target)
    model = ProjectionModel(
        coef_rc1=coef_rc1,
        coef_rc2=coef_rc2,
        features=FEATURE_COLUMNS,
        provenance={
            "method": "Affine projection recovered from released part 2 OSF score tables and map coordinates",
            "source_models": list(PART2_GROUPS.keys()),
        },
    )

    for model_name, spec in PART2_GROUPS.items():
        scores = pd.read_csv(data_dir / spec["scores_filename"])
        coords = pd.read_csv(data_dir / spec["coords_filename"])
        averaged = scores.groupby("country")[FEATURE_COLUMNS].mean().reset_index()
        merged = averaged.merge(coords, left_on="country", right_on="country.territory", how="inner")
        projected = model.project(merged)
        rc1_error = np.abs(projected["RC1"] - merged[spec["rc1_column"]])
        rc2_error = np.abs(projected["RC2"] - merged[spec["rc2_column"]])
        validation[model_name] = {
            "rows": int(len(merged)),
            "max_abs_error_rc1": float(rc1_error.max()),
            "max_abs_error_rc2": float(rc2_error.max()),
            "rmse_rc1": float(np.sqrt(np.mean(rc1_error ** 2))),
            "rmse_rc2": float(np.sqrt(np.mean(rc2_error ** 2))),
        }

    model.validation = validation
    return model


def save_projection_model(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True))
    return output_path


def project_wide_table(wide_table, projection_model, label=None, model_name=None):
    projected = projection_model.project(wide_table)
    if model_name is not None:
        projected["model"] = model_name
    if label is not None:
        projected["label"] = label
    return projected


def summarise_projected_points(projected, label, model_name):
    return pd.DataFrame(
        [
            {
                "label": label,
                "model": model_name,
                "RC1": float(projected["RC1"].mean()),
                "RC2": float(projected["RC2"].mean()),
                "n_variants": int(len(projected)),
            }
        ]
    )


def load_paper_general_points(data_dir, projection_model):
    ensure_assets(data_dir)
    data_dir = Path(data_dir)
    rows = []

    for label, filename in PART1_TABLES.items():
        table = pd.read_csv(data_dir / filename)
        projected = project_wide_table(table, projection_model, label=label, model_name=label)
        rows.append(summarise_projected_points(projected, label=label, model_name=label))

    gpt3_scores = pd.DataFrame([MANUAL_GPT3_DEFAULT_SCORES])
    gpt3_projected = project_wide_table(gpt3_scores, projection_model, label="GPT-3", model_name="GPT-3")
    rows.append(summarise_projected_points(gpt3_projected, label="GPT-3", model_name="GPT-3"))

    combined = pd.concat(rows, ignore_index=True)
    ordered_labels = ["GPT-3", "GPT-3.5-turbo", "GPT-4", "GPT-4-turbo", "GPT-4o"]
    combined["label"] = pd.Categorical(combined["label"], categories=ordered_labels, ordered=True)
    return combined.sort_values("label").reset_index(drop=True)
