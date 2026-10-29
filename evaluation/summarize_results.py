"""Summarize saved evaluate_all.py results for LDM, REPA and REPA-E.

Example:
    python evaluation/summarize_results.py --ldm-results RESULTS/ldm \\
        --repa-results RESULTS/repa --repae-results RESULTS/repae --output-dir RESULTS

Only saved result files are read here. This module never runs evaluations,
loads models or generates counterfactual images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


METHODS = ("LDM", "REPA", "REPA-E")
COLORS = {"LDM": "#4477AA", "REPA": "#EE7733", "REPA-E": "#228833"}
ATTRIBUTES = ("view", "density", "scanner", "age")
METRICS = (("mae", "MAE ↓"), ("mse", "MSE ↓"),
           ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓"))
COMPARISONS = {
    "vae_vs_input": "VAE reconstruction vs input",
    "cycle_vs_input": "No-edit cycle vs input",
    "cf_vs_input": "Counterfactual vs input",
    "cf_vs_cycle": "Counterfactual vs no-edit cycle",
    "generation": "Generation",
    "reconstruction_vae": "VAE reconstruction",
    "reconstruction_cycle": "No-edit cycle",
}


def _number(value):
    if value is None or str(value).strip() in ("", "None", "null"):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _flag(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def _format(value):
    value = _number(value)
    if value is None:
        return "N/A"
    if value == 0:
        return "0"
    if abs(value) < .0001 or abs(value) >= 10000:
        return f"{value:.3g}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    if not rows:
        return "_No results were requested or recorded for this table._\n"
    lines = ["| " + " | ".join(map(_escape, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(_escape, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["method", "family", "comparison", "attribute", "target", "metric", "value", "status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(runs):
    rows = []
    for method in METHODS:
        if method not in runs:
            continue
        path = Path(runs[method]) / "summary.csv"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row, method=method))
    if not rows:
        raise ValueError("No evaluation metrics were found in the selected runs.")
    return rows


def _find(rows, method, family, comparison, attribute="none", metric=None, target=None):
    found = [row for row in rows if row["method"] == method and row.get("family") == family
             and row.get("comparison") == comparison and row.get("attribute", "none") == attribute
             and (metric is None or row.get("metric") == metric)
             and (target is None or row.get("target") == target)]
    if len(found) > 1:
        raise ValueError(f"Duplicate metric rows for {method}, {family}, {comparison}, {attribute}, {metric}, {target}.")
    return found[0] if found else None


def _score(row):
    if row is None or row.get("status", "ok") != "ok":
        return None
    return _number(row.get("value"))


def _attributes(rows, protocol):
    recorded = {row.get("attribute") for row in rows if row.get("attribute") in ATTRIBUTES}
    requested = protocol.get("settings", {}).get("attributes", [])
    if isinstance(requested, str):
        requested = requested.split()
    recorded.update(attribute for attribute in requested if attribute in ATTRIBUTES)
    return [attribute for attribute in ATTRIBUTES if attribute in recorded]


def _identity_table(rows, comparison, methods, attributes):
    table_rows = []
    for attribute in attributes:
        for method in methods:
            selected = [_find(rows, method, "identity", comparison, attribute, metric)
                        for metric, _ in METRICS]
            if not any(selected):
                continue
            n = next((row.get("n_generated") or row.get("n_real") or "N/A"
                      for row in selected if row), "N/A")
            label = COMPARISONS.get(comparison, comparison) if attribute == "none" else attribute.title()
            table_rows.append([label, method, n] + [_format(_score(row)) for row in selected])
    return _table(["Comparison / intervention", "Method", "Images", "MAE ↓", "MSE ↓", "SSIM ↑", "LPIPS ↓"], table_rows)


def _grouped_bars(ax, categories, values, methods, ylabel):
    x = np.arange(len(categories), dtype=float)
    width = .78 / max(1, len(methods))
    finite = [value for series in values.values() for value in series if value is not None]
    if finite:
        lower = min(0., min(finite))
        upper = max(0., max(finite))
        extent = upper - lower or max(abs(upper), 1.)
        ax.set_ylim(lower - (.08 * extent if lower < 0 else 0), upper + .22 * extent)
    else:
        ax.set_ylim(0, 1)
    for index, method in enumerate(methods):
        positions = x + (index - (len(methods) - 1) / 2) * width
        for position, value in zip(positions, values[method]):
            if value is None:
                # Annotation only: an unavailable result is never plotted as zero.
                ax.text(position, .025, "N/A", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, rotation=90, color=COLORS[method])
            else:
                ax.bar(position, value, width=width * .92, color=COLORS[method], zorder=3)
                ax.annotate(_format(value), (position, value), xytext=(0, 4 if value >= 0 else -4),
                            textcoords="offset points", ha="center",
                            va="bottom" if value >= 0 else "top", fontsize=7, rotation=45)
    ax.set_xticks(x, categories)
    ax.set_xlim(-.5, len(categories) - .5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.22, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if not finite:
        ax.text(.5, .55, "No available scores", transform=ax.transAxes,
                ha="center", va="center", color="#666666")


def _save_figure(fig, output_dir, stem):
    output_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _identity_figure(output_dir, rows, methods, attributes, comparison, synthetic=False):
    labels = [attribute.title() for attribute in attributes]
    if not labels:
        labels = ["No interventions"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title) in zip(axes.flat, METRICS):
        values = {method: [_score(_find(rows, method, "identity", comparison, attribute, metric))
                           for attribute in attributes] or [None] for method in methods}
        _grouped_bars(ax, labels, values, methods, title)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.suptitle(COMPARISONS[comparison], fontsize=16, fontweight="bold", y=.99)
    fig.legend([Patch(color=COLORS[method]) for method in methods], methods,
               loc="upper center", bbox_to_anchor=(.5, .955), ncol=len(methods), frameon=False)
    if synthetic:
        fig.text(.5, .895, "SYNTHETIC PREVIEW — NOT BENCHMARK RESULTS", ha="center",
                 fontsize=10, color="#9B2626", fontweight="bold")
    fig.text(.5, .015, "Image similarity proxies; low distance alone does not establish identity preservation or edit success.\n"
             "Bars show means across images. No uncertainty intervals are plotted; N/A means unavailable or not requested.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .075, 1, .91), h_pad=2.5)
    return _save_figure(fig, output_dir, f"identity_{comparison}")


def _quality_specs(rows, attributes):
    specs = [(comparison, "none", COMPARISONS[comparison])
             for comparison in ("generation", "reconstruction_vae", "reconstruction_cycle")
             if any(row.get("comparison") == comparison and row.get("family") != "identity" for row in rows)]
    specs.extend(("counterfactual_target_weighted_mean", attribute, attribute.title() + " CF\n(target-weighted mean)")
                 for attribute in attributes
                 if any(row.get("comparison") == "counterfactual_target_weighted_mean"
                        and row.get("attribute") == attribute for row in rows))
    return specs


def _quality_figure(output_dir, rows, methods, specs, synthetic=False):
    fig, axes = plt.subplots(2, 1, figsize=(max(11, len(specs) * 1.9), 9))
    for ax, family, title in zip(axes, ("fid_inception", "fd_dino"),
                                 ("Inception FID ↓", "DINO Fréchet distance ↓")):
        categories = [(attribute.title() + " CF*" if comparison == "counterfactual_target_weighted_mean"
                       else label.replace("VAE reconstruction", "VAE\nreconstruction"))
                      for comparison, attribute, label in specs] or ["No distribution tasks"]
        values = {method: [_score(_find(rows, method, family, comparison, attribute))
                           for comparison, attribute, _ in specs] or [None] for method in methods}
        _grouped_bars(ax, categories, values, methods, title)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.suptitle("Distribution quality comparison", fontsize=16, fontweight="bold", y=.995)
    fig.legend([Patch(color=COLORS[method]) for method in methods], methods,
               loc="upper center", bbox_to_anchor=(.5, .965), ncol=len(methods), frameon=False)
    if synthetic:
        fig.text(.5, .92, "SYNTHETIC PREVIEW — NOT BENCHMARK RESULTS", ha="center",
                 fontsize=10, color="#9B2626", fontweight="bold")
    fig.text(.5, .012, "* CF bars are generated-count-weighted means of observational target-group distances, not pooled FIDs.\n"
             "Feature spaces have separate scales. Small/rank-deficient cohorts can yield noisy estimates; inspect the support tables.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .075, 1, .92), h_pad=3)
    return _save_figure(fig, output_dir, "distribution_quality")


def _support(row):
    if not row:
        return "not recorded"
    flags = []
    if row.get("status") != "ok":
        flags.append(row.get("status") or "status unavailable")
    if _flag(row.get("low_sample_count")):
        flags.append("low sample count")
    if _flag(row.get("rank_deficient")):
        flags.append("rank deficient")
    return "; ".join(flags) or "ok"


def _quality_table(rows, methods, specs):
    tables = []
    for family, title in (("fid_inception", "Inception FID ↓"), ("fd_dino", "DINO FD ↓")):
        table_rows = []
        for comparison, attribute, label in specs:
            for method in methods:
                row = _find(rows, method, family, comparison, attribute)
                count = f"{row.get('n_real') or '—'} / {row.get('n_generated') or '—'}" if row else "— / —"
                table_rows.append([label.replace("\n", " "), method, _format(_score(row)), count, _support(row)])
        tables.append(f"### {title}\n\n" + _table(["Task", "Method", "Score ↓", "Real / generated", "Support"], table_rows))
    return "\n".join(tables)


def build_report(output_dir: Path, runs: dict, protocol: dict) -> list[Path]:
    """Write tables and PNG/PDF figures; return every produced artifact path.

    ``runs`` maps LDM, REPA and REPA-E labels to completed evaluate_all folders.
    The caller validates that the checkpoint steps and evaluation cohorts match.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [method for method in METHODS if method in runs]
    if not methods:
        raise ValueError("At least one recognized method is required.")
    rows = _load_rows(runs)
    attributes = _attributes(rows, protocol)
    combined = output_dir / "metrics.csv"
    targets_path = output_dir / "counterfactual_targets.csv"
    _write_csv(combined, rows)
    target_rows = [row for row in rows if row.get("comparison") == "counterfactual_target"]
    _write_csv(targets_path, target_rows)
    specs = _quality_specs(rows, attributes)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        figures = []
        for comparison in ("cf_vs_input", "cf_vs_cycle"):
            figures.extend(_identity_figure(output_dir, rows, methods, attributes, comparison,
                                             synthetic=bool(protocol.get("synthetic_preview"))))
        figures.extend(_quality_figure(output_dir, rows, methods, specs,
                                      synthetic=bool(protocol.get("synthetic_preview"))))
    settings = protocol.get("settings", {})
    experiment_rows = []
    cohort_rows = []
    for method in methods:
        experiment = protocol.get("experiments", {}).get(method, {})
        experiment_rows.append([method, experiment.get("name", "—"), experiment.get("checkpoint", "—")])
        config_path = Path(runs[method]) / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        cohort_rows.append([method, config.get("num_images", "—"), config.get("split", settings.get("split", "—")),
                            config.get("indices_sha256", "—"), config.get("status", "—")])
    step = protocol.get("step", "unspecified")
    lines = [f"# LDM / REPA / REPA-E comparison — step {step}", "",
             "All methods are compared at the same training step using the same saved evaluation protocol. "
             "Image-similarity scores do not by themselves establish patient identity preservation or a successful intervention.", "",
             "## Experiments", "", _table(["Method", "Experiment", "Checkpoint"], experiment_rows),
             "## Evaluation protocol", "", _table(["Setting", "Value"], [
                 ["Checkpoint step", step], ["Weights", settings.get("weights", "ema")],
                 ["Split", settings.get("split", "test")], ["Requested images", settings.get("num_samples", "—")],
                 ["Euler steps per direction", settings.get("num_steps", "—")], ["Seed", settings.get("seed", "—")],
                 ["Interventions", ", ".join(attributes) or "none"],
                 ["Image sampling", "FP32; posterior mode; saved BN; CF/cycle CFG1"],
                 ["Metric input", protocol.get("image_storage", {}).get("format", "recorded in config.json")],
                 ["Generation CFG", settings.get("cfg_scale", 1)],
                 ["Generation Heun correction", settings.get("heun", False)],
             ]), _table(["Method", "Images", "Split", "Selected-indices SHA256", "Status"], cohort_rows),
             "Metrics use saved images. Sampling provenance and image precision are recorded in each method's `config.json`.", "",
             "## Reconstruction baselines", "",
             "VAE reconstruction measures the compression/reconstruction floor. The no-edit cycle also includes flow inversion and return error. "
             "These rows use the full selected cohort (`attribute=none`).", "",
             _identity_table(rows, "vae_vs_input", methods, ["none"]),
             _identity_table(rows, "cycle_vs_input", methods, ["none"]),
             "## Counterfactual image similarity", "",
             "Metrics are means across eligible images for each intervention. Lower MAE, MSE and LPIPS indicate greater similarity; "
             "higher SSIM indicates greater similarity. No arbitrary combined score or overall winner is calculated.", "",
             "### Counterfactual vs input", "", _identity_table(rows, "cf_vs_input", methods, attributes),
             "![Counterfactual vs input](figures/identity_cf_vs_input.png)", "",
             "### Counterfactual vs matched no-edit cycle", "",
             "This comparison measures the change relative to the no-intervention output from the same inferred noise; "
             "it is not a direct patient-identity test.", "",
             _identity_table(rows, "cf_vs_cycle", methods, attributes),
             "![Counterfactual vs no-edit cycle](figures/identity_cf_vs_cycle.png)", "",
             "## Distribution quality", "",
             "Inception FID and DINO FD use different feature spaces and should be read on their separate scales. "
             "Counterfactual summaries are **generated-count-weighted means of target-group distances, not pooled FIDs**. "
             "Their real references are observational target categories/age bins; other covariates can differ. "
             "For weighted means, the real count is omitted because each target has its own reference cohort.", "",
             _quality_table(rows, methods, specs), "![Distribution quality](figures/distribution_quality.png)", "",
             "### Target-group support", "",
             "[Counterfactual target-group table (CSV)](counterfactual_targets.csv) contains each method, feature backend, "
             "attribute, target, score, real/generated counts, status, low-sample flag and covariance-rank flag. "
             "`N/A` denotes unavailable or unrequested scores and is never plotted as zero. "
             "Groups with fewer than two images cannot be scored; computable small or rank-deficient groups can still be noisy. "
             "An unavailable target group makes that attribute's weighted summary unavailable.", "",
             "## Downloadable data and figures", "",
             "- [All metric rows (CSV)](metrics.csv)",
             "- [Counterfactual target groups (CSV)](counterfactual_targets.csv)",
             "- [Counterfactual vs input (PDF)](figures/identity_cf_vs_input.pdf)",
             "- [Counterfactual vs cycle (PDF)](figures/identity_cf_vs_cycle.pdf)",
             "- [Distribution quality (PDF)](figures/distribution_quality.pdf)", "",
             "Figures show point estimates without error bars. Any `std` values in the source CSV describe variation across images, "
             "not confidence intervals. The reported metrics should be interpreted alongside intervention effectiveness and example images. "
             "DINOv2 is also the REPA training teacher, so the Inception and similarity results provide additional perspectives.", ""]
    report_path = output_dir / "report.md"
    if protocol.get("synthetic_preview"):
        lines[2:2] = ["> **SYNTHETIC PREVIEW — NOT BENCHMARK RESULTS.** All values are artificial test fixtures.", ""]
    report_path.write_text("\n".join(lines))
    return [report_path, combined, targets_path, *figures]


PROTOCOL_ARGUMENTS = (
    "weights", "split", "num_samples", "num_steps", "seed", "cfg_scale", "heun",
    "evaluations", "identity_metrics", "tasks", "attributes", "target_view",
    "target_density", "target_scanner", "target_age", "age_bin_edges", "feature_model",
)
PROTOCOL_METADATA = (
    "step", "resolution", "split", "split_csv_sha256", "indices_sha256", "num_images",
    "weights", "inference", "feature_extractors", "source_sha256",
    "image_storage", "sampling_source_sha256", "reference_images_sha256",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_results(runs):
    """Validate completed, comparable result files before writing a report."""
    configs, selections, fingerprints = {}, {}, {}
    for method in METHODS:
        directory = Path(runs[method]).expanduser().resolve()
        for filename in ("config.json", "indices.json", "summary.csv"):
            if not (directory / filename).is_file():
                raise FileNotFoundError(f"{method}: missing {filename} in {directory}")
        try:
            config = json.loads((directory / "config.json").read_text())
            selection = json.loads((directory / "indices.json").read_text())
        except (ValueError, OSError) as error:
            raise ValueError(f"{method}: cannot read saved evaluation metadata: {error}") from error
        if config.get("status") != "complete":
            raise ValueError(f"{method}: evaluation is not complete: {directory}")
        missing = [key for key in (*PROTOCOL_METADATA, "arguments", "checkpoint", "experiment")
                   if key not in config]
        if missing:
            raise ValueError(f"{method}: config.json is missing {', '.join(missing)}")
        missing = [key for key in PROTOCOL_ARGUMENTS if key not in config["arguments"]]
        if missing:
            raise ValueError(f"{method}: saved arguments are missing {', '.join(missing)}")
        indices = selection.get("indices", [])
        if (not indices or any(type(index) is not int or index < 0 for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError(f"{method}: indices.json needs nonempty, unique nonnegative integer indices")
        indices_hash = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
        if (selection.get("split") != config["split"]
                or selection.get("split_csv_sha256") != config["split_csv_sha256"]
                or len(indices) != config["num_images"]
                or indices_hash != config["indices_sha256"]):
            raise ValueError(f"{method}: indices.json does not match config.json")
        with (directory / "summary.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"family", "comparison", "attribute", "target", "metric", "value", "status"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{method}: summary.csv is missing required metric columns")
            rows = list(reader)
        if not rows:
            raise ValueError(f"{method}: summary.csv contains no metric results")
        keys = [(row["family"], row["comparison"], row["attribute"], row["target"], row["metric"])
                for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{method}: summary.csv contains duplicate metric rows")
        for row in rows:
            if row.get("status") == "ok" and _number(row.get("value")) is None:
                raise ValueError(f"{method}: summary.csv has an invalid score with status=ok")
            for key in ("step", "checkpoint", "weights", "split", "num_images", "evaluation_id"):
                if key in row and key in config and str(row[key]) != str(config[key]):
                    raise ValueError(f"{method}: summary.csv {key} does not match config.json")
        configs[method], selections[method] = config, selection
        fingerprints[method] = {filename: _sha256(directory / filename)
                                for filename in ("config.json", "indices.json", "summary.csv")}
    reference = configs["LDM"]
    for method in METHODS[1:]:
        if selections[method] != selections["LDM"]:
            raise ValueError(f"{method}: selected test images differ from LDM")
        for key in PROTOCOL_METADATA:
            if configs[method][key] != reference[key]:
                raise ValueError(f"{method}: {key} differs from LDM")
        for key in PROTOCOL_ARGUMENTS:
            if configs[method]["arguments"][key] != reference["arguments"][key]:
                raise ValueError(f"{method}: evaluation setting {key} differs from LDM")
    protocol = {
        "step": reference["step"], "settings": reference["arguments"],
        "experiments": {method: {"name": config["experiment"], "checkpoint": config["checkpoint"]}
                        for method, config in configs.items()},
        "resolution": reference["resolution"], "split_csv_sha256": reference["split_csv_sha256"],
        "indices_sha256": reference["indices_sha256"],
        "feature_extractors": reference["feature_extractors"], "source_sha256": reference["source_sha256"],
        "image_storage": reference["image_storage"],
        "sampling_source_sha256": reference["sampling_source_sha256"],
        "reference_images_sha256": reference["reference_images_sha256"],
    }
    return protocol, selections["LDM"], fingerprints


def summarize(args):
    runs = {method: Path(path).expanduser().resolve() for method, path in zip(
        METHODS, (args.ldm_results, args.repa_results, args.repae_results))}
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir in runs.values():
        raise ValueError("Choose an output directory separate from the individual evaluation folders")
    protocol, selection, fingerprints = validate_results(runs)
    artifacts = build_report(output_dir, runs, protocol)
    (output_dir / "indices.json").write_text(json.dumps(selection, indent=2) + "\n")
    metadata = {"status": "complete", "protocol": protocol,
                "results": {method: str(path) for method, path in runs.items()},
                "result_sha256": fingerprints, "artifacts": [str(path) for path in artifacts]}
    temporary = output_dir / ".comparison.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    temporary.replace(output_dir / "comparison.json")
    print(f"Summary saved: {output_dir / 'report.md'}")
    return output_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ldm-results", required=True, help="Completed LDM evaluate_all.py result directory")
    parser.add_argument("--repa-results", required=True, help="Completed REPA evaluate_all.py result directory")
    parser.add_argument("--repae-results", required=True, help="Completed REPA-E evaluate_all.py result directory")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison tables and figures")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        summarize(parse_args())
    except (ValueError, OSError, KeyError) as error:
        print(f"Cannot summarize evaluations: {error}", file=sys.stderr)
        sys.exit(1)
