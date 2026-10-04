"""Review allocation under scarce human input — Experiment 3."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np

import utils
from experiments.experiment_2.rewards import REWARDS
from paths import dataset_prepared

from .allocation import (
    BASELINE,
    aggregate_pairs,
    aggregate_reward_sensitivity,
    evaluate_pair,
    reward_sensitivity_pair,
)
from .synthetic import config_names, generate_synthetic_pair

logger = logging.getLogger("experiment_3")


@click.group()
def main() -> None:
    """Review allocation experiment (Experiment 3)."""


# fmt: off
@main.command()
@click.option("--n-samples", default=10_000, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--output-dir", default=None, type=click.Path(path_type=Path), help="Output directory.")
# fmt: on
def synthetic(n_samples: int, seed: int, output_dir: Path | None) -> None:
    """Synthetic regime: binary-state binary-action allocation."""
    rng = np.random.default_rng(seed)
    out_dir = utils.make_run_dir("experiment_3", output_dir)
    utils.start_run(out_dir, "experiment_3", "synthetic regime")

    pair_results: list[dict] = []
    for name in config_names():
        b_x, b_xh, h, y, meta = generate_synthetic_pair(name, n=n_samples, rng=rng)
        metrics = evaluate_pair(b_x, b_xh, h, y, REWARDS, rng=rng)
        pair_results.append({"tag": name, "meta": meta, "metrics": metrics})

    payload = {
        "results": pair_results,
        "aggregate": aggregate_pairs(pair_results, REWARDS),
    }
    utils.finalize_run(out_dir, payload, logger)
    _log_summary(payload["aggregate"], q_key="0.20")


# fmt: off
@main.command()
@click.option("--dataset", required=True, help="Dataset name (e.g. chexpert).")
@click.option("--model", default=None, help="Model directory name.")
@click.option("--seed", default=0, show_default=True)
@click.option("--output-dir", default=None, type=click.Path(path_type=Path), help="Output directory.")
# fmt: on
def real(dataset: str, model: str | None, seed: int, output_dir: Path | None) -> None:
    """Real regime: walk data/prepared/<dataset>/<model>/pairs/*.npz."""
    rng = np.random.default_rng(seed)
    all_npz = _collect_pair_paths(dataset, model)

    out_dir = utils.make_run_dir("experiment_3", output_dir)
    utils.start_run(out_dir, "experiment_3", f"real regime (dataset={dataset})")

    n_total = len(all_npz)
    pair_results: list[dict] = []
    for i, npz_path in enumerate(all_npz, 1):
        logger.info(f"[{i}/{n_total}] {npz_path.stem}")
        b_x, b_xh, h, y, meta = utils.load_pair(npz_path)
        metrics = evaluate_pair(b_x, b_xh, h, y, REWARDS, rng=rng)
        pair_results.append({"tag": npz_path.stem, "meta": meta, "metrics": metrics})

    payload = {
        "results": pair_results,
        "aggregate": aggregate_pairs(pair_results, REWARDS),
    }
    utils.finalize_run(out_dir, payload, logger)
    _log_summary(payload["aggregate"], q_key="0.20")


# fmt: off
@main.command(name="reward-sensitivity")
@click.option("--dataset", required=True, help="Dataset name (e.g. chexpert/gt).")
@click.option("--model", default=None, help="Model directory name.")
@click.option("--config", "config_name", type=click.Choice(list(REWARDS)), default="R1", show_default=True, help="Reward used to rank cases.")
@click.option("--eval", "eval_name", type=click.Choice(list(REWARDS)), default="R2", show_default=True, help="Reward used for actions and utility.")
@click.option("--seed", default=0, show_default=True)
@click.option("--output-dir", default=None, type=click.Path(path_type=Path), help="Output directory.")
# fmt: on
def reward_sensitivity(
    dataset: str,
    model: str | None,
    config_name: str,
    eval_name: str,
    seed: int,
    output_dir: Path | None,
) -> None:
    """Misspecified reward: rank cases by R_config, realize utility under R_eval."""
    rng = np.random.default_rng(seed)
    all_npz = _collect_pair_paths(dataset, model)

    out_dir = utils.make_run_dir("experiment_3_reward_sensitivity", output_dir)
    utils.start_run(
        out_dir, "experiment_3", f"{config_name} ranking, {eval_name} utility"
    )

    pair_results: list[dict] = []
    for npz_path in all_npz:
        b_x, b_xh, h, y, meta = utils.load_pair(npz_path)
        result = reward_sensitivity_pair(
            b_x, b_xh, h, y, REWARDS[config_name], REWARDS[eval_name], rng=rng
        )
        pair_results.append({"tag": npz_path.stem, "meta": meta, **result})

    aggregate = aggregate_reward_sensitivity(pair_results)
    payload = {
        "config_reward": config_name,
        "eval_reward": eval_name,
        "results": pair_results,
        "aggregate": aggregate,
    }
    utils.finalize_run(out_dir, payload, logger)
    for qk, row in aggregate["budgets"].items():
        logger.info(
            f"  q={qk}: {config_name}-ranked={row['mean_utility_config_ranking']:.4f}  "
            f"{eval_name}-ranked={row['mean_utility_eval_ranking']:.4f}  "
            f"W/L/T={row['n_wins']}/{row['n_losses']}/{row['n_ties']}  "
            f"overlap={row['mean_overlap']:.3f}"
        )


def _collect_pair_paths(dataset: str, model: str | None) -> list[Path]:
    """Sorted data/prepared/<dataset>/<model>/pairs/*.npz for one or all models."""
    prepared_root = dataset_prepared(dataset)

    if model is not None:
        model_dirs = [prepared_root / model]
    else:
        model_dirs = sorted(
            d for d in prepared_root.iterdir() if (d / "pairs").is_dir()
        )
        if not model_dirs:
            raise click.ClickException(
                f"No model directories with pairs/ found under {prepared_root}"
            )

    all_npz: list[Path] = []
    for model_dir in model_dirs:
        pairs_dir = model_dir / "pairs"
        npz_paths = sorted(pairs_dir.glob("*.npz"))
        if not npz_paths:
            raise click.ClickException(f"No .npz pair files found in {pairs_dir}")
        all_npz.extend(npz_paths)
    return all_npz


def _log_summary(aggregate: dict, *, q_key: str) -> None:
    """One-line per-reward summary at a chosen budget."""
    for reward_name in REWARDS:
        row = aggregate["policies"][reward_name][q_key]
        br_gain = row["BR_hat"]["mean_utility_gain"]
        br_win = row["BR_hat"]["win_rate_vs_baseline"]
        mg_gain = row[BASELINE]["mean_utility_gain"]
        ent_gain = row["Entropy"]["mean_utility_gain"]
        logger.info(
            f"  {reward_name} q={q_key}: BR_hat gain={br_gain:+.4f}  "
            f"win_vs_{BASELINE}={br_win:.1%}  "
            f"{BASELINE}={mg_gain:+.4f}  Entropy={ent_gain:+.4f}"
        )
