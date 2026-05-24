import argparse
from pathlib import Path

import pandas as pd

from .anthropic_runner import run_part1_models as run_anthropic_part1_models
from .gemini_runner import run_part1_models as run_gemini_part1_models
from .openai_runner import run_part1_models
from .paper_assets import ensure_assets, get_template_image_path, load_country_map
from .plotting import plot_culture_map, plot_culture_map_with_template_shapes, plot_template_map
from .projection import derive_projection_model, load_paper_general_points, save_projection_model
from .together_runner import run_part1_models as run_together_part1_models
from .utils import ensure_dir


def cmd_download_paper_assets(args):
    downloaded = ensure_assets(args.data_dir, overwrite=args.overwrite)
    for filename, path in downloaded.items():
        print("{}\t{}".format(filename, path))
    return 0


def cmd_derive_projection(args):
    model = derive_projection_model(args.data_dir)
    output_path = save_projection_model(model, args.output)
    print(output_path)
    return 0


def cmd_plot_map(args):
    data_dir = Path(args.data_dir)
    human_df = load_country_map(data_dir)
    projection_model = derive_projection_model(data_dir) if args.with_paper_models else None
    paper_points = load_paper_general_points(data_dir, projection_model) if args.with_paper_models else None

    if paper_points is not None:
        paper_points_path = Path(args.paper_points_output)
        ensure_dir(paper_points_path.parent)
        paper_points.to_csv(paper_points_path, index=False)

    if args.official_template:
        plot_template_map(
            template_image_path=get_template_image_path(data_dir),
            overlay_point_paths=args.points_csv,
            paper_points=paper_points,
            output_path=args.output,
            hide_title=args.hide_template_title,
            hide_source_block=args.hide_template_source_block,
            add_axis_end_labels=args.axis_end_labels,
            add_zero_guides=args.zero_guides,
            exclude_labels=args.exclude_label,
        )
    elif args.traced_template_shapes:
        plot_culture_map_with_template_shapes(
            human_df=human_df,
            template_image_path=get_template_image_path(data_dir),
            overlay_point_paths=args.points_csv,
            paper_points=paper_points,
            output_path=args.output,
            exclude_labels=args.exclude_label,
        )
    else:
        plot_culture_map(
            human_df=human_df,
            overlay_point_paths=args.points_csv,
            paper_points=paper_points,
            output_path=args.output,
            title=args.title,
            exclude_labels=args.exclude_label,
        )
    print(args.output)
    return 0


def cmd_run_openai_part1(args):
    combined = run_part1_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=args.model,
        recent=args.recent,
        env_file=args.env_file,
        base_url=args.base_url,
        temperature=args.temperature,
        delay=args.delay,
        overwrite=args.overwrite,
    )
    print(Path(args.output_dir) / "all_model_mean_projection.csv")
    print(combined.to_string(index=False))
    return 0


def cmd_run_anthropic_part1(args):
    combined = run_anthropic_part1_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=args.model,
        recent=args.recent,
        env_file=args.env_file,
        base_url=args.base_url,
        temperature=args.temperature,
        delay=args.delay,
        overwrite=args.overwrite,
    )
    print(Path(args.output_dir) / "all_model_mean_projection.csv")
    print(combined.to_string(index=False))
    return 0


def cmd_run_gemini_part1(args):
    combined = run_gemini_part1_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=args.model,
        recent=args.recent,
        env_file=args.env_file,
        base_url=args.base_url,
        temperature=args.temperature,
        delay=args.delay,
        overwrite=args.overwrite,
    )
    print(Path(args.output_dir) / "all_model_mean_projection.csv")
    print(combined.to_string(index=False))
    return 0


def cmd_run_together_part1(args):
    combined = run_together_part1_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=args.model,
        recent=args.recent,
        env_file=args.env_file,
        base_url=args.base_url,
        temperature=args.temperature,
        delay=args.delay,
        overwrite=args.overwrite,
    )
    print(Path(args.output_dir) / "all_model_mean_projection.csv")
    print(combined.to_string(index=False))
    return 0


def cmd_run_local_country_eval(args):
    from .local_country_runner import run_local_country_models

    outputs = run_local_country_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        variants=args.variant,
        country_codes=args.country_code,
        env_file=args.env_file,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        overwrite=args.overwrite,
    )
    print(Path(args.output_dir) / "all_variant_overall_summary.csv")
    overall = outputs["overall_summary"]
    if isinstance(overall, pd.DataFrame) and not overall.empty:
        print(overall.to_string(index=False))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="culture-map")
    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser("download-paper-assets", help="Download the paper's released OSF assets")
    download_parser.add_argument("--data-dir", default="data/paper_osf")
    download_parser.add_argument("--overwrite", action="store_true")
    download_parser.set_defaults(func=cmd_download_paper_assets)

    derive_parser = subparsers.add_parser("derive-projection", help="Recover the paper's score-to-map projection")
    derive_parser.add_argument("--data-dir", default="data/paper_osf")
    derive_parser.add_argument("--output", default="outputs/derived_projection.json")
    derive_parser.set_defaults(func=cmd_derive_projection)

    plot_parser = subparsers.add_parser("plot-map", help="Render the Matplotlib culture map")
    plot_parser.add_argument("--data-dir", default="data/paper_osf")
    plot_parser.add_argument("--points-csv", action="append", default=[])
    plot_parser.add_argument("--with-paper-models", action="store_true")
    plot_parser.add_argument("--paper-points-output", default="outputs/paper_general_model_points.csv")
    plot_parser.add_argument("--output", default="outputs/culture_map.png")
    plot_parser.add_argument("--title", default="")
    plot_parser.add_argument("--official-template", action="store_true")
    plot_parser.add_argument("--hide-template-title", action="store_true")
    plot_parser.add_argument("--hide-template-source-block", action="store_true")
    plot_parser.add_argument("--axis-end-labels", action="store_true")
    plot_parser.add_argument("--zero-guides", action="store_true")
    plot_parser.add_argument("--traced-template-shapes", action="store_true")
    plot_parser.add_argument("--exclude-label", action="append", default=[])
    plot_parser.set_defaults(func=cmd_plot_map)

    run_parser = subparsers.add_parser("run-openai-part1", help="Run the paper's part 1 prompt suite against OpenAI models")
    run_parser.add_argument("--data-dir", default="data/paper_osf")
    run_parser.add_argument("--output-dir", default="outputs/openai_runs")
    run_parser.add_argument("--env-file", default=None)
    run_parser.add_argument("--base-url", default="https://api.openai.com/v1")
    run_parser.add_argument("--model", action="append")
    run_parser.add_argument("--recent", action="store_true")
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--delay", type=float, default=0.0)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.set_defaults(func=cmd_run_openai_part1)

    anthropic_parser = subparsers.add_parser("run-anthropic-part1", help="Run the paper's part 1 prompt suite against Anthropic models")
    anthropic_parser.add_argument("--data-dir", default="data/paper_osf")
    anthropic_parser.add_argument("--output-dir", default="outputs/anthropic_runs")
    anthropic_parser.add_argument("--env-file", default=None)
    anthropic_parser.add_argument("--base-url", default="https://api.anthropic.com/v1")
    anthropic_parser.add_argument("--model", action="append")
    anthropic_parser.add_argument("--recent", action="store_true")
    anthropic_parser.add_argument("--temperature", type=float, default=0.0)
    anthropic_parser.add_argument("--delay", type=float, default=0.0)
    anthropic_parser.add_argument("--overwrite", action="store_true")
    anthropic_parser.set_defaults(func=cmd_run_anthropic_part1)

    gemini_parser = subparsers.add_parser("run-gemini-part1", help="Run the paper's part 1 prompt suite against Gemini models")
    gemini_parser.add_argument("--data-dir", default="data/paper_osf")
    gemini_parser.add_argument("--output-dir", default="outputs/gemini_runs")
    gemini_parser.add_argument("--env-file", default=None)
    gemini_parser.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta")
    gemini_parser.add_argument("--model", action="append")
    gemini_parser.add_argument("--recent", action="store_true")
    gemini_parser.add_argument("--temperature", type=float, default=0.0)
    gemini_parser.add_argument("--delay", type=float, default=0.0)
    gemini_parser.add_argument("--overwrite", action="store_true")
    gemini_parser.set_defaults(func=cmd_run_gemini_part1)

    together_parser = subparsers.add_parser("run-together-part1", help="Run the paper's part 1 prompt suite against Together models")
    together_parser.add_argument("--data-dir", default="data/paper_osf")
    together_parser.add_argument("--output-dir", default="outputs/together_runs")
    together_parser.add_argument("--env-file", default=None)
    together_parser.add_argument("--base-url", default="https://api.together.xyz/v1")
    together_parser.add_argument("--model", action="append")
    together_parser.add_argument("--recent", action="store_true")
    together_parser.add_argument("--temperature", type=float, default=0.0)
    together_parser.add_argument("--delay", type=float, default=0.0)
    together_parser.add_argument("--overwrite", action="store_true")
    together_parser.set_defaults(func=cmd_run_together_part1)

    local_country_parser = subparsers.add_parser(
        "run-local-country-eval",
        help="Run raw pretrained MAPLE checkpoints on WVS questions and score target-country projection quality",
    )
    local_country_parser.add_argument("--data-dir", default="data/paper_osf")
    local_country_parser.add_argument("--output-dir", default="outputs/local_country_eval")
    local_country_parser.add_argument("--env-file", default=None)
    local_country_parser.add_argument(
        "--base-url",
        default="https://www.worldvaluessurvey.org/WVSContents.jsp?CMSID=tradrat",
    )
    local_country_parser.add_argument(
        "--chat-template-path",
        default=None,
        help="Reserved for compatibility with older cluster launchers.",
    )
    local_country_parser.add_argument("--variant", action="append")
    local_country_parser.add_argument("--country-code", action="append")
    local_country_parser.add_argument("--temperature", type=float, default=0.0)
    local_country_parser.add_argument("--top-p", type=float, default=1.0)
    local_country_parser.add_argument("--max-new-tokens", type=int, default=32)
    local_country_parser.add_argument("--overwrite", action="store_true")
    local_country_parser.set_defaults(func=cmd_run_local_country_eval)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
