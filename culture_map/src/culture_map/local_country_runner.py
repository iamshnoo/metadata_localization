from collections import OrderedDict
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from .paper_assets import ensure_assets, load_country_map, load_descriptors, load_questions
from .projection import derive_projection_model, project_wide_table
from .scoring import responses_to_wide_table, score_response
from .utils import ensure_dir, load_env_file, write_jsonl


DEFAULT_BASE_URL = "www.worldvaluessurvey.org/wvs-source"

METACUL_COUNTRY_SPECS = OrderedDict(
    [
        ("bd", {"country": "Bangladesh", "continent": "Asia"}),
        ("ca", {"country": "Canada", "continent": "America"}),
        ("gb", {"country": "Great Britain", "continent": "Europe"}),
        ("gh", {"country": "Ghana", "continent": "Africa"}),
        ("hk", {"country": "Hong Kong SAR", "continent": "Asia"}),
        ("ie", {"country": "Ireland", "continent": "Europe"}),
        ("in", {"country": "India", "continent": "Asia"}),
        ("ke", {"country": "Kenya", "continent": "Africa"}),
        ("my", {"country": "Malaysia", "continent": "Asia"}),
        ("ng", {"country": "Nigeria", "continent": "Africa"}),
        ("ph", {"country": "Philippines", "continent": "Asia"}),
        ("pk", {"country": "Pakistan", "continent": "Asia"}),
        ("sg", {"country": "Singapore", "continent": "Asia"}),
        ("us", {"country": "United States", "continent": "America"}),
        ("za", {"country": "South Africa", "continent": "Africa"}),
    ]
)

MAPLE_VARIANTS = OrderedDict(
    [
        ("maple_1b_tplus_eplus", {"label": "MAPLE 1B (T+, I+)", "model_path": "/path/to/metacul/models/combined_with_metadata_1b", "train_metadata": True, "eval_metadata": True, "size": "1B"}),
        ("maple_1b_tplus_eminus", {"label": "MAPLE 1B (T+, I-)", "model_path": "/path/to/metacul/models/combined_with_metadata_1b", "train_metadata": True, "eval_metadata": False, "size": "1B"}),
        ("maple_1b_tminus_eplus", {"label": "MAPLE 1B (T-, I+)", "model_path": "/path/to/metacul/models/combined_without_metadata_1b", "train_metadata": False, "eval_metadata": True, "size": "1B"}),
        ("maple_1b_tminus_eminus", {"label": "MAPLE 1B (T-, I-)", "model_path": "/path/to/metacul/models/combined_without_metadata_1b", "train_metadata": False, "eval_metadata": False, "size": "1B"}),
        ("maple_3b_tplus_eplus", {"label": "MAPLE 3B (T+, I+)", "model_path": "/path/to/metacul/models/combined_with_metadata_3b", "train_metadata": True, "eval_metadata": True, "size": "3B"}),
        ("maple_3b_tplus_eminus", {"label": "MAPLE 3B (T+, I-)", "model_path": "/path/to/metacul/models/combined_with_metadata_3b", "train_metadata": True, "eval_metadata": False, "size": "3B"}),
        ("maple_3b_tminus_eplus", {"label": "MAPLE 3B (T-, I+)", "model_path": "/path/to/metacul/models/combined_without_metadata_3b", "train_metadata": False, "eval_metadata": True, "size": "3B"}),
        ("maple_3b_tminus_eminus", {"label": "MAPLE 3B (T-, I-)", "model_path": "/path/to/metacul/models/combined_without_metadata_3b", "train_metadata": False, "eval_metadata": False, "size": "3B"}),
    ]
)

Y003_TARGET_QUALITIES = [
    "Independence",
    "Determination, perseverance",
    "Religious faith",
    "Obedience",
]
Y003_FILLERS = [
    "Good manners",
    "Hard work",
    "Feeling of responsibility",
    "Tolerance and respect for other people",
    "Imagination",
]


def _load_hf_modules():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("This command requires torch and transformers in the active environment.") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _variant_output_paths(output_dir, variant_name):
    output_dir = Path(output_dir)
    return {
        "responses_csv": output_dir / "{}_responses.csv".format(variant_name),
        "raw_jsonl": output_dir / "{}_responses.jsonl".format(variant_name),
        "wide_csv": output_dir / "{}_wide_table.csv".format(variant_name),
        "projected_csv": output_dir / "{}_variant_projection.csv".format(variant_name),
        "country_mean_csv": output_dir / "{}_country_mean_projection.csv".format(variant_name),
        "distance_csv": output_dir / "{}_country_distance_summary.csv".format(variant_name),
        "overall_csv": output_dir / "{}_overall_summary.csv".format(variant_name),
    }


def _is_complete(paths):
    return all(path.exists() for path in paths.values())


def _build_prompt_text(descriptor, prompt, country_code, continent, with_metadata, base_url):
    core = "{}\n\n{}".format(str(descriptor).strip(), str(prompt).strip())
    if not with_metadata:
        return "CONTENT:\n{}".format(core), {"url_country_tag": None, "url_continent_tag": None, "metadata_source": "none"}

    content = (
        "URL: {base_url}/{country_code}\n"
        "COUNTRY: {country_code}\n"
        "CONTINENT: {continent}\n\n"
        "TITLE: World Values Survey responses for the country {country_code}\n\n"
        "CONTENT:\n"
        "{core}"
    ).format(base_url=base_url, country_code=country_code, continent=continent, core=core)
    return content, {"url_country_tag": country_code, "url_continent_tag": continent, "metadata_source": "from_country_spec"}


def _candidate_texts_for_scale(scale):
    if scale in ("a008", "g006"):
        return [" {}".format(i) for i in range(1, 5)]
    if scale in ("f063", "f118", "f120"):
        return [" {}".format(i) for i in range(1, 11)]
    if scale == "e018":
        return [" 1", " 2", " 3"]
    if scale == "e025":
        return [" A", " B", " C"]
    if scale == "a165":
        return [" A", " B"]
    if scale == "y002":
        return [" {}, {}".format(first, second) for first, second in permutations([1, 2, 3, 4], 2)]
    if scale == "y003":
        candidates = []
        for mask in range(16):
            chosen = []
            for idx, quality in enumerate(Y003_TARGET_QUALITIES):
                if mask & (1 << idx):
                    chosen.append(quality)
            fillers_needed = max(0, 5 - len(chosen))
            candidates.append(" " + ", ".join(chosen + Y003_FILLERS[:fillers_needed]))
        return candidates
    raise ValueError("Unsupported scale '{}'".format(scale))


def _predict_candidate_by_loglikelihood(torch_mod, model, tokenizer, prompt_text, candidates):
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    scores_sum = []
    scores_avg = []
    with torch_mod.no_grad():
        for candidate in candidates:
            cand_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            if cand_ids.numel() == 0:
                scores_sum.append(float("-inf"))
                scores_avg.append(float("-inf"))
                continue
            input_ids = torch_mod.cat([prompt_ids, cand_ids], dim=1)
            attention_mask = torch_mod.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch_mod.log_softmax(logits[:, :-1, :], dim=-1)
            start = prompt_ids.shape[1]
            target = input_ids[:, 1:]
            cand_target = target[:, start - 1 :]
            cand_log_probs = log_probs[:, start - 1 :, :].gather(-1, cand_target.unsqueeze(-1)).squeeze(-1)
            score_sum = float(cand_log_probs.sum().item())
            token_length = int(cand_ids.shape[1])
            scores_sum.append(score_sum)
            scores_avg.append(score_sum / max(token_length, 1))
    best_idx = int(np.argmax(scores_avg)) if scores_avg else 0
    return {
        "selected_text": str(candidates[best_idx]).strip(),
        "selected_index": best_idx,
        "candidate_loglikelihood_sums": scores_sum,
        "candidate_loglikelihood_avgs": scores_avg,
    }


def _build_country_distance_summary(country_means, human_targets, variant_name, label):
    rows = []
    for _, point in country_means.iterrows():
        deltas = human_targets.copy()
        deltas["distance"] = ((deltas["RC1_final"] - float(point["RC1"])) ** 2 + (deltas["RC2_final"] - float(point["RC2"])) ** 2) ** 0.5
        deltas = deltas.sort_values(["distance", "country"]).reset_index(drop=True)
        target_country = point["country"]
        target_idx = deltas.index[deltas["country"] == target_country]
        if len(target_idx) == 0:
            raise ValueError("Target country '{}' missing from human target set".format(target_country))
        target_rank = int(target_idx[0]) + 1
        target_row = deltas.iloc[target_idx[0]]
        nearest_row = deltas.iloc[0]
        rows.append(
            {
                "variant": variant_name,
                "label": label,
                "country": target_country,
                "country_code": point["country_code"],
                "continent": point["continent"],
                "category": point["Category"],
                "RC1": float(point["RC1"]),
                "RC2": float(point["RC2"]),
                "target_distance": float(target_row["distance"]),
                "target_rank": target_rank,
                "nearest_country": nearest_row["country"],
                "nearest_distance": float(nearest_row["distance"]),
                "top1_hit": int(nearest_row["country"] == target_country),
                "top3_hit": int(target_rank <= 3),
            }
        )
    return pd.DataFrame(rows)


def _build_overall_summary(distance_df, variant_name, spec):
    return pd.DataFrame(
        [
            {
                "variant": variant_name,
                "label": spec["label"],
                "model_path": spec["model_path"],
                "size": spec["size"],
                "train_metadata": bool(spec["train_metadata"]),
                "eval_metadata": bool(spec["eval_metadata"]),
                "n_countries": int(len(distance_df)),
                "mean_target_distance": float(distance_df["target_distance"].mean()),
                "median_target_distance": float(distance_df["target_distance"].median()),
                "mean_target_rank": float(distance_df["target_rank"].mean()),
                "top1_rate": float(distance_df["top1_hit"].mean()),
                "top3_rate": float(distance_df["top3_hit"].mean()),
            }
        ]
    )


def run_local_country_models(
    data_dir,
    output_dir,
    variants=None,
    country_codes=None,
    env_file=None,
    base_url=DEFAULT_BASE_URL,
    temperature=0.0,
    top_p=1.0,
    max_new_tokens=64,
    overwrite=False,
):
    del temperature, top_p, max_new_tokens

    if env_file:
        load_env_file(env_file)

    ensure_assets(data_dir)
    output_dir = ensure_dir(output_dir)
    questions = load_questions(data_dir)
    descriptors = load_descriptors(data_dir)
    projection_model = derive_projection_model(data_dir)
    human_df = load_country_map(data_dir)

    variant_names = list(variants) if variants else list(MAPLE_VARIANTS.keys())
    invalid_variants = [name for name in variant_names if name not in MAPLE_VARIANTS]
    if invalid_variants:
        raise ValueError("Unknown variant(s): {}".format(", ".join(sorted(invalid_variants))))

    selected_country_codes = list(country_codes) if country_codes else list(METACUL_COUNTRY_SPECS.keys())
    invalid_country_codes = [code for code in selected_country_codes if code not in METACUL_COUNTRY_SPECS]
    if invalid_country_codes:
        raise ValueError("Unknown country code(s): {}".format(", ".join(sorted(invalid_country_codes))))

    target_country_names = [METACUL_COUNTRY_SPECS[code]["country"] for code in selected_country_codes]
    human_targets = human_df[human_df["country"].isin(target_country_names)].copy()
    if len(human_targets) != len(selected_country_codes):
        missing = sorted(set(target_country_names) - set(human_targets["country"]))
        raise ValueError("Missing human map coordinates for: {}".format(", ".join(missing)))

    torch_mod, AutoModelForCausalLM, AutoTokenizer = _load_hf_modules()

    all_country_means = []
    all_distance_summaries = []
    all_overall_summaries = []

    for variant_name in variant_names:
        spec = MAPLE_VARIANTS[variant_name]
        paths = _variant_output_paths(output_dir, variant_name)
        if not overwrite and _is_complete(paths):
            all_country_means.append(pd.read_csv(paths["country_mean_csv"]))
            all_distance_summaries.append(pd.read_csv(paths["distance_csv"]))
            all_overall_summaries.append(pd.read_csv(paths["overall_csv"]))
            continue

        if overwrite:
            for path in paths.values():
                if path.exists():
                    path.unlink()

        model_path = spec["model_path"]
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                fix_mistral_regex=True,
                local_files_only=True,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            local_files_only=True,
        )

        response_rows = []
        raw_rows = []
        total_requests = len(selected_country_codes) * len(descriptors) * len(questions)
        completed_requests = 0
        print("[{}] starting {} prompt-scale requests".format(variant_name, total_requests))

        for country_code in selected_country_codes:
            country_spec = METACUL_COUNTRY_SPECS[country_code]
            target_country = country_spec["country"]
            continent = country_spec["continent"]
            for _, descriptor_row in descriptors.iterrows():
                descriptor = descriptor_row["respondent_descriptor"]
                prompt_variant = str(descriptor_row["#variant"])
                for _, question_row in questions.iterrows():
                    scale = str(question_row["scale"])
                    prompt_text, meta_info = _build_prompt_text(descriptor, question_row["prompt"], country_code, continent, bool(spec["eval_metadata"]), base_url)
                    candidates = _candidate_texts_for_scale(scale)
                    predicted = _predict_candidate_by_loglikelihood(torch_mod, model, tokenizer, prompt_text, candidates)
                    selected_text = predicted["selected_text"]
                    score = score_response(scale, selected_text)

                    response_rows.append(
                        {
                            "#variant": prompt_variant,
                            "country": target_country,
                            "country_code": country_code,
                            "continent": continent,
                            "scale": scale,
                            "survey_question": question_row["prompt"],
                            "generated_text": selected_text,
                            "score": score,
                            "variant": variant_name,
                            "label": spec["label"],
                            "model": model_path,
                            "train_metadata": bool(spec["train_metadata"]),
                            "eval_metadata": bool(spec["eval_metadata"]),
                            **meta_info,
                        }
                    )
                    raw_rows.append(
                        {
                            "variant": variant_name,
                            "label": spec["label"],
                            "model": model_path,
                            "country": target_country,
                            "country_code": country_code,
                            "continent": continent,
                            "#variant": prompt_variant,
                            "scale": scale,
                            "prompt": prompt_text,
                            "candidates": candidates,
                            "selected_text": selected_text,
                            "score": score,
                            "selected_index": predicted["selected_index"],
                            "candidate_loglikelihood_sums": predicted["candidate_loglikelihood_sums"],
                            "candidate_loglikelihood_avgs": predicted["candidate_loglikelihood_avgs"],
                        }
                    )
                    completed_requests += 1
                    if completed_requests % 25 == 0 or completed_requests == total_requests:
                        print("[{}] completed {}/{} requests".format(variant_name, completed_requests, total_requests))

        del model
        if torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()

        responses_df = pd.DataFrame(response_rows)
        responses_df.to_csv(paths["responses_csv"], index=False)
        write_jsonl(paths["raw_jsonl"], raw_rows)

        wide_table = responses_to_wide_table(responses_df)
        wide_table = wide_table.merge(responses_df[["country", "country_code", "continent"]].drop_duplicates(), on="country", how="left")
        wide_table.to_csv(paths["wide_csv"], index=False)

        projected = project_wide_table(wide_table, projection_model, label=spec["label"], model_name=model_path)
        projected["variant"] = variant_name
        projected["train_metadata"] = bool(spec["train_metadata"])
        projected["eval_metadata"] = bool(spec["eval_metadata"])
        projected.to_csv(paths["projected_csv"], index=False)

        country_means = projected.groupby(
            ["variant", "label", "model", "train_metadata", "eval_metadata", "country", "country_code", "continent"],
            as_index=False,
        )[["RC1", "RC2"]].mean()
        country_means = country_means.merge(human_targets[["country", "Category"]].drop_duplicates(), on="country", how="left")
        country_means.to_csv(paths["country_mean_csv"], index=False)

        distance_df = _build_country_distance_summary(country_means, human_targets, variant_name, spec["label"])
        distance_df.to_csv(paths["distance_csv"], index=False)

        overall_df = _build_overall_summary(distance_df, variant_name, spec)
        overall_df.to_csv(paths["overall_csv"], index=False)

        all_country_means.append(country_means)
        all_distance_summaries.append(distance_df)
        all_overall_summaries.append(overall_df)

    combined_means = pd.concat(all_country_means, ignore_index=True) if all_country_means else pd.DataFrame()
    combined_distances = pd.concat(all_distance_summaries, ignore_index=True) if all_distance_summaries else pd.DataFrame()
    combined_overall = pd.concat(all_overall_summaries, ignore_index=True) if all_overall_summaries else pd.DataFrame()

    if not combined_means.empty:
        combined_means.to_csv(Path(output_dir) / "all_variant_country_mean_projection.csv", index=False)
    if not combined_distances.empty:
        combined_distances.to_csv(Path(output_dir) / "all_variant_country_distance_summary.csv", index=False)
    if not combined_overall.empty:
        combined_overall.to_csv(Path(output_dir) / "all_variant_overall_summary.csv", index=False)

    return {
        "country_mean_projection": combined_means,
        "country_distance_summary": combined_distances,
        "overall_summary": combined_overall,
    }
