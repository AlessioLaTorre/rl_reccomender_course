"""Create employability plots and tables from an experiment directory."""

import argparse
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


matplotlib.use("Agg")
FILENAME = re.compile(r"(?P<reward>.+)_k(?P<k>\d+)_seed(?P<seed>\d+)\.txt$")


def read_run(path: Path, root: Path, max_step):
    """Read one seed. For Optuna studies, use only the selected best trial."""
    match = FILENAME.fullmatch(path.name)
    if match is None:
        return None
    experiment = path.parent.parent.parent
    trial = next((p for p in path.parents if re.fullmatch(r"trial\d+", p.name)), None)
    if trial is not None:
        experiment = trial.parent
        best_path = experiment / "best_params.json"
        if not best_path.is_file():
            raise ValueError("Study has no selected best trial")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        if int(trial.name[5:]) != best["trial"]:
            raise ValueError("Not the study's selected best trial")

    status_path = path.with_suffix(".json")
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict):
            raise ValueError("Status JSON must contain an object")
        if status.get("status", "completed") != "completed" or status.get("completed") is False:
            raise ValueError("Run is not completed")

    config_path = path.with_suffix(".yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    config = config or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration YAML must contain a mapping")
    algorithm = {"ppo": "PPO", "ppo_mask": "MaskablePPO"}.get(config.get("model"), "unspecified")
    # Keep datasets of different sizes separate when configurations are available.
    dataset = "unspecified"
    if "nb_courses" in config and "nb_jobs" in config:
        dataset = f"{config['nb_courses']}courses_{config['nb_jobs']}jobs_seed{config.get('dataset_seed', 42)}"

    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] not in {3, 6, 10}:
        raise ValueError(f"Expected 3, 6 or 10 columns; found {frame.shape[1]}")
    frame = frame.iloc[:, :2].copy()
    frame.columns = ["step", "jobs"]
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Empty or non-finite log")
    if not frame.step.gt(0).all() or not frame.step.eq(frame.step.astype(int)).all():
        raise ValueError("Steps must be positive integers")
    if frame.groupby("step").jobs.nunique().gt(1).any():
        raise ValueError("Conflicting employability values at the same step")
    frame = frame.drop_duplicates("step").sort_values("step")
    limit = config.get("total_steps")
    if max_step is not None:
        limit = min(int(limit), max_step) if limit is not None else max_step
    if limit is not None:
        frame = frame[frame.step <= int(limit)]
    if frame.empty:
        raise ValueError("No evaluations within the selected budget")
    try:
        source = str(experiment.relative_to(root)) or "."
    except ValueError:
        source = str(experiment)
    return {
        "experiment": source, "algorithm": algorithm, "dataset": dataset,
        "reward": match["reward"], "k": int(match["k"]), "seed": int(match["seed"]),
        "path": str(path), "frame": frame.set_index("step").jobs,
    }


def save_plot(fig, output: Path, name: str):
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(runs: list, output: Path, last_n: int):
    """Compare the same evaluation steps and show the actual seed counts."""
    seed_rows, summaries, curve_rows = [], [], []
    for dataset, k in sorted({(run["dataset"], run["k"]) for run in runs}):
        selected = [run for run in runs if run["dataset"] == dataset and run["k"] == k]
        common = sorted(set.intersection(*(set(run["frame"].index) for run in selected)))
        if not common:
            raise ValueError(f"No common evaluation steps for {dataset}, k={k}")
        groups = {}
        for run in selected:
            key = (run["experiment"], run["algorithm"], run["reward"])
            group = groups.setdefault(key, {})
            if run["seed"] in group:
                raise ValueError(f"Duplicate seed {run['seed']} for {key}, k={k}")
            group[run["seed"]] = run

        fig, ax = plt.subplots(figsize=(10, 5))
        bar_fig, bar_ax = plt.subplots(figsize=(10, 5))
        labels, values, errors = [], [], []
        used_n = min(last_n, len(common))
        for (experiment, algorithm, reward), group in sorted(groups.items()):
            seeds = sorted(group)
            matrix = pd.DataFrame({seed: group[seed]["frame"].loc[common] for seed in seeds})
            means, stds = matrix.mean(axis=1), matrix.std(axis=1, ddof=1)
            tails = matrix.tail(used_n).mean(axis=0)
            finals = matrix.iloc[-1]
            best_step = int(means.idxmax())
            metadata = {"experiment": experiment, "algorithm": algorithm, "dataset": dataset,
                        "reward": reward, "k": k}
            summaries.append({
                **metadata, "n_seeds": len(seeds), "seeds": ",".join(map(str, seeds)),
                "end_step": int(common[-1]), "last_n_used": used_n,
                "final_mean": float(finals.mean()), "final_std": float(finals.std(ddof=1)),
                "tail_mean": float(tails.mean()), "tail_std": float(tails.std(ddof=1)),
                "best_mean": float(means.loc[best_step]),
                "best_std": float(stds.loc[best_step]), "best_step": best_step,
            })
            for seed in seeds:
                seed_rows.append({**metadata, "seed": seed, "path": group[seed]["path"],
                                  "end_step": int(common[-1]), "last_n_used": used_n,
                                  "final_jobs": float(finals[seed]), "tail_jobs": float(tails[seed])})
            curve = pd.DataFrame({"step": common, "mean": means.to_numpy(), "std": stds.to_numpy()})
            curve_rows.extend({**metadata, "n_seeds": len(seeds), **row} for row in curve.to_dict("records"))
            label = reward if algorithm == "unspecified" else f"{reward} ({algorithm})"
            if sum(key[1:] == (algorithm, reward) for key in groups) > 1:
                label += f" — {experiment}"
            label += f" [n={len(seeds)}]"
            line, = ax.plot(common, means, label=label)
            if len(seeds) > 1:
                ax.fill_between(common, means - stds, means + stds, color=line.get_color(), alpha=0.16)
            ax.scatter(best_step, means.loc[best_step], color=line.get_color(), s=25)
            labels.append(label)
            values.append(float(tails.mean()))
            errors.append(float(tails.std(ddof=1)) if len(seeds) > 1 else 0.0)

        ax.set(xlabel="Training steps", ylabel="Employability (applicable jobs)", title=f"k={k} — {dataset}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
        positions = np.arange(len(labels))
        bar_ax.bar(positions, values, yerr=errors, capsize=4)
        bar_ax.set_xticks(positions, labels, rotation=30, ha="right", fontsize=8)
        bar_ax.set(ylabel="Employability (applicable jobs)", title=f"Mean of last {used_n} shared evaluations — k={k}")
        bar_ax.grid(axis="y", alpha=0.25)
        save_plot(fig, output, f"employability_{dataset}_k{k}")
        save_plot(bar_fig, output, f"tail_{dataset}_k{k}")
        print(f"[k={k}, {dataset}] {len(common)} common steps; last {used_n} used for the tail score.")

    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    summary.to_html(output / "summary.html", index=False, float_format=lambda value: f"{value:.4f}", na_rep="")
    pd.DataFrame(seed_rows).to_csv(output / "per_seed.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "curves.csv", index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Plot employability and summarize the available experiment seeds.")
    parser.add_argument("directory", type=Path, help="Experiment directory containing results_k*/seed*/ logs.")
    parser.add_argument("--output", type=Path, help="Report directory; default: DIRECTORY/report.")
    parser.add_argument("--last-n", type=int, default=50)
    parser.add_argument("--max-step", type=int, help="Optional maximum training step.")
    args = parser.parse_args()
    root = args.directory.resolve()
    output = args.output.resolve() if args.output else root / "report"
    if not root.is_dir():
        parser.error(f"Directory not found: {root}")
    if args.last_n < 1 or (args.max_step is not None and args.max_step < 1):
        parser.error("last-n and max-step must be positive.")
    output.mkdir(parents=True, exist_ok=True)
    runs, skipped = [], []
    for path in sorted(root.rglob("*.txt")):
        if FILENAME.fullmatch(path.name) is None:
            continue
        try:
            run = read_run(path, root, args.max_step)
            if run is not None:
                runs.append(run)
        except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as error:
            skipped.append({"path": str(path), "reason": str(error)})
    pd.DataFrame(skipped, columns=["path", "reason"]).to_csv(output / "skipped.csv", index=False)
    if not runs:
        parser.error(f"No valid completed logs. See {output / 'skipped.csv'}")
    build_report(runs, output, args.last_n)
    print(f"[SAVED] {output} — {len(runs)} runs included; {len(skipped)} skipped.")


if __name__ == "__main__":
    main()
