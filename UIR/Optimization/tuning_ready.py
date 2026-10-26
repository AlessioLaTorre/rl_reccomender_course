"""Optimize PPO and MaskablePPO with Optuna. See README.md for usage."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from optuna.trial import TrialState
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import is_masking_supported

from UIR.Scripts.Dataset import Dataset
from UIR.Scripts.Reinforce import Reinforce


matplotlib.use("Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = {"MaskablePPO": "ppo_mask", "PPO": "ppo"}
REWARDS = {
    "Employability": {"feature": "Employability", "method": 1},
    "UIR-gap": {"feature": "UIR", "method": 1},
    "UIR-threshold": {"feature": "UIR", "method": 0},
    "EUIR-gap": {"feature": "EUIR", "method": 1},
}
REWARD_ALIASES = {"UIR_gap": "UIR-gap", "UIR_threshold": "UIR-threshold", "EUIR": "EUIR-gap"}
K_VALUES = [2, 3, 4, 5]
DEFAULT_SEEDS = [42, 43, 44, 45, 50]
SEARCH_SPACE = {
    "n_steps": [256, 512, 1024],
    "num_minibatches": [4, 8, 16],
    "n_epochs": [4, 6, 8],
    "clip_range": [0.10, 0.30, False],
    "ent_coef": [1e-6, 0.03, True],
    "gamma": [0.90, 0.99, False],
    "gae_lambda": [0.90, 0.98, False],
    "lr_initial": [1e-4, 1e-3, True],
    "lr_final": [1e-6, 1e-4, True],
    "warmup_frac": [0.0, 0.08, False],
    "start_at": [0.50, 0.85, False],
}
# Continuous domains are [minimum, maximum, logarithmic sampling].
LOG_COLUMNS = [
    "step", "jobs", "reward", "gap", "pref", "req_levels", "req_skills",
    "skills_covered", "skills_missing", "eval_seconds",
]


def write_json(path: Path, value) -> None:
    """Replace a JSON file only after writing its complete contents."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class NoPreferencesEnv(gym.Wrapper):
    """Keep preference vectors empty and expose the native action mask."""

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.setdefault("learner", None)
        options["want"] = np.zeros(self.unwrapped.nb_skills, dtype=np.int32)
        options["avoid"] = np.zeros(self.unwrapped.nb_skills, dtype=np.int32)
        return self.env.reset(seed=seed, options=options)

    def action_masks(self):
        return self.env.get_wrapper_attr("get_action_mask")()


class CourseRecommender(Reinforce):
    """Use the project algorithms without preferences in either environment."""

    def get_model(self):
        self.train_env = NoPreferencesEnv(self.train_env)
        self.eval_env = NoPreferencesEnv(self.eval_env)
        super().get_model()
        if self.model_name == "ppo_mask":
            if not isinstance(self.model, MaskablePPO) or not is_masking_supported(self.model.get_env()):
                raise RuntimeError("MaskablePPO requires action masking during training.")


def create_recommender(dataset, config):
    return CourseRecommender(dataset=dataset, config=config)


def load_config(args, output: Path) -> dict:
    """Load the project YAML and apply the optimization settings."""
    path = Path(args.config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.update(
        nb_courses=args.nb_courses, nb_jobs=args.nb_jobs, nb_cvs=-1, max_cv_skills=15,
        level_3=True, set_dataset_seed=True, dataset_seed=42,
        use_preference=False, use_pretrained=False, save_model=False, use_standard=False,
    )
    for key in ["total_steps", "eval_freq"]:
        value = getattr(args, key)
        config[key] = int(config[key] if value is None else value)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config.setdefault("hypers", {})["device"] = device
    for name in SEARCH_SPACE:
        if name in config["hypers"]:
            convert = int if name in {"n_steps", "num_minibatches", "n_epochs"} else float
            config["hypers"][name] = convert(config["hypers"][name])
    config["hypers"]["batch_size"] = int(config["hypers"]["batch_size"])
    for key in ["taxonomy_path", "course_path", "cv_path", "job_path", "mastery_levels_path"]:
        data_path = Path(config[key])
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        config[key] = str(data_path.resolve())
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing file for {key}: {data_path}")
    config["results_path"] = str(output / "results_k")
    config["save_dir"] = str(output / "models")
    return config


def dataset_details(dataset) -> dict:
    """Record sizes and fingerprints to verify data when resuming."""
    details = {}
    for name in ["learners", "jobs", "courses"]:
        matrix = getattr(dataset, name)
        index = getattr(dataset, name + "_index")
        ids = [index[i] for i in range(len(matrix))]
        content = json.dumps(ids).encode() + matrix.tobytes()
        details[name] = {"count": len(matrix), "sha256": hashlib.sha256(content).hexdigest()}
    details["skills"] = len(dataset.skills)
    return details


def score_log(path: Path, config: dict, last_n: int) -> float:
    """Score the last N scheduled employability evaluations at the same budget."""
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(LOG_COLUMNS):
        raise ValueError(f"Invalid log format: {path}")
    frame.columns = LOG_COLUMNS
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Empty or non-finite log: {path}")
    if not frame.step.is_monotonic_increasing or frame.step.duplicated().any():
        raise ValueError(f"Invalid evaluation steps: {path}")
    if not frame.pref.eq(0).all():
        raise ValueError(f"Preferences must be disabled: {path}")
    frame = frame[frame.step <= config["total_steps"]]
    expected = np.arange(config["eval_freq"], config["total_steps"] + 1, config["eval_freq"])
    if not np.array_equal(frame.step.to_numpy(), expected) or len(frame) < last_n:
        raise ValueError(f"Missing scheduled evaluations: {path}")
    return float(frame.jobs.tail(last_n).mean())


def mean_score(scores: list) -> float:
    """Give every selected reward/k pair equal weight."""
    pairs = {}
    for row in scores:
        pairs.setdefault((row["reward"], row["k"]), []).append(row["jobs"])
    return float(np.mean([np.mean(values) for values in pairs.values()]))


def make_groups(rewards: list, k_values: list, scope: str) -> list:
    """Return each study folder and its selected reward/k pairs."""
    match scope:
        case "reward-k":
            return [
                (f"k{k}/{reward}", [(reward, k)])
                for k in k_values for reward in rewards
            ]
        case "reward":
            return [
                (reward, [(reward, k) for k in k_values])
                for reward in rewards
            ]
        case "all":
            return [("shared", [(reward, k) for k in k_values for reward in rewards])]
        case _:
            raise ValueError(f"Unknown scope: {scope}")


class ReinforceTuner:
    """Tune one study using the same candidate for all its reward/k/seed runs."""

    def __init__(self, configs, dataset, output, seeds, algorithm, scope,
                 n_trials=20, last_n=50, sampler_seed=42, pruning=False):
        self.configs = copy.deepcopy(configs)
        self.dataset = dataset
        self.output = Path(output)
        self.k_values = sorted({config["k"] for config in configs})
        self.rewards = list(dict.fromkeys(config["name_exp"] for config in configs))
        self.seeds = {k: list(seeds[k]) for k in self.k_values}
        if len({tuple(values) for values in self.seeds.values()}) != 1:
            raise ValueError("Use the same seeds, in the same order, for every k.")
        self.algorithm, self.scope = algorithm, scope
        self.n_trials, self.last_n = n_trials, last_n
        self.sampler_seed, self.pruning = sampler_seed, pruning
        if scope in {"reward-k", "k"}:
            group = self.rewards[0] if scope == "reward-k" else "shared"
            self.study_name = f"{algorithm}_{scope}_{group}_k{self.k_values[0]}"
        else:
            group = self.rewards[0] if scope == "reward" else "shared"
            self.study_name = f"{algorithm}_{scope}_{group}"

    def sample_hypers(self, trial) -> dict:
        hypers = copy.deepcopy(self.configs[0]["hypers"])
        for name, domain in SEARCH_SPACE.items():
            if name in {"n_steps", "num_minibatches", "n_epochs"}:
                hypers[name] = trial.suggest_categorical(name, domain)
            else:
                low, high, log = domain
                hypers[name] = trial.suggest_float(name, low, high, log=log)
        hypers["batch_size"] = hypers["n_steps"] // hypers["num_minibatches"]
        return hypers

    def run_seed(self, trial, base_config, seed, hypers) -> float:
        """Train one fresh model and save its configuration, status and log."""
        config = copy.deepcopy(base_config)
        config.update(hypers=copy.deepcopy(hypers), seed=seed,
                      results_path=str(self.output / f"trial{trial.number}" / "results_k"))
        prefix = f"{config['name_exp']}_k{config['k']}_seed{seed}"
        folder = Path(f"{config['results_path']}{config['k']}") / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{prefix}.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)
        status_path = folder / f"{prefix}.json"
        status = {"status": "running"}
        write_json(status_path, status)
        print(f"[RUN] {self.algorithm}, {prefix}, trial={trial.number}", flush=True)
        recommender = None
        try:
            recommender = create_recommender(self.dataset, config)
            recommender.reinforce_recommendation()
            score = score_log(folder / f"{prefix}.txt", config, self.last_n)
            status.update(status="completed", score=score,
                          actual_steps=int(recommender.model.num_timesteps))
            return score
        except BaseException as error:
            status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=f"{type(error).__name__}: {error}")
            if isinstance(error, KeyboardInterrupt):
                trial.set_user_attr("interrupted", True)
            raise
        finally:
            write_json(status_path, status)
            if recommender is not None:
                recommender.train_env.close()
                recommender.eval_env.close()

    def objective(self, trial) -> float:
        hypers = self.sample_hypers(trial)
        trial.set_user_attr("effective_hypers", hypers)
        scores = []
        shared_seeds = self.seeds[self.k_values[0]]
        for seed_count, seed in enumerate(shared_seeds, 1):
            for config in self.configs:
                score = self.run_seed(trial, config, seed, hypers)
                scores.append({"reward": config["name_exp"], "k": config["k"],
                               "seed": seed, "jobs": score})
                trial.set_user_attr("scores", scores)
            # Report after every selected pair has completed this same seed.
            trial.report(mean_score(scores), step=seed_count)
            if seed_count < len(shared_seeds) and trial.should_prune():
                raise optuna.TrialPruned(f"Stopped after {seed_count} seed rounds.")
        return mean_score(scores)

    def enqueue_config(self, study) -> None:
        """Try the YAML parameters first when they fit the search space."""
        hypers = self.configs[0]["hypers"]
        params = {name: hypers.get(name) for name in SEARCH_SPACE}
        if hypers["n_steps"] % hypers["batch_size"]:
            return
        params["num_minibatches"] = hypers["n_steps"] // hypers["batch_size"]
        for name, domain in SEARCH_SPACE.items():
            value = params[name]
            if value is None:
                return
            if name in {"n_steps", "num_minibatches", "n_epochs"}:
                if value not in domain:
                    return
            elif not domain[0] <= value <= domain[1]:
                return
        study.enqueue_trial(params)

    def export(self, study):
        """Save scores and the best parameters, without retraining a model."""
        study.trials_dataframe().to_csv(self.output / "trials.csv", index=False)
        rows = [{"trial": trial.number, "state": trial.state.name, **score}
                for trial in study.trials for score in trial.user_attrs.get("scores", [])]
        scores = pd.DataFrame(rows, columns=["trial", "state", "reward", "k", "seed", "jobs"])
        scores.to_csv(self.output / "per_seed.csv", index=False)
        scores.groupby(["trial", "state", "reward", "k"]).jobs.agg(
            ["mean", "std", "count"]).reset_index().to_csv(self.output / "per_pair.csv", index=False)
        completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
        if not completed:
            return None
        best = study.best_trial
        hypers = best.user_attrs["effective_hypers"]
        pair_scores = []
        for config in self.configs:
            values = [row["jobs"] for row in best.user_attrs["scores"]
                      if row["reward"] == config["name_exp"] and row["k"] == config["k"]]
            pair_scores.append({
                "reward": config["name_exp"], "k": config["k"], "n_seeds": len(values),
                "mean_jobs": float(np.mean(values)),
                "std_jobs": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            })
        result = {
            "schema": 4, "algorithm": self.algorithm, "scope": self.scope,
            "study": self.study_name, "rewards": self.rewards, "k_values": self.k_values,
            "trial": best.number, "value": best.value,
            "seeds": {str(k): values for k, values in self.seeds.items()},
            "last_n": self.last_n, "objective": "equal mean of per-pair mean tail employability",
            "sampled_params": best.params, "effective_hypers": hypers,
            "pair_scores": pair_scores, "scores": best.user_attrs["scores"],
            "total_steps": self.configs[0]["total_steps"], "eval_freq": self.configs[0]["eval_freq"],
        }
        write_json(self.output / "best_params.json", result)
        with (self.output / "best_hypers.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(hypers, file, sort_keys=False)
        config_dir = self.output / "best_configs"
        config_dir.mkdir(exist_ok=True)
        for base_config in self.configs:
            config = copy.deepcopy(base_config)
            reward, k = config["name_exp"], config["k"]
            config.update(hypers=copy.deepcopy(hypers), seed=self.seeds[k][0],
                          results_path=str(self.output / "best_run" / reward / "results_k"))
            with (config_dir / f"{reward}_k{k}.yaml").open("w", encoding="utf-8") as file:
                yaml.safe_dump(config, file, sort_keys=False)
            if len(self.configs) == 1:
                with (self.output / "best_config.yaml").open("w", encoding="utf-8") as file:
                    yaml.safe_dump(config, file, sort_keys=False)
        numbers, values = [trial.number for trial in completed], [trial.value for trial in completed]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(numbers, values, "o-", label="Trial")
        ax.plot(numbers, np.maximum.accumulate(values), "--", label="Best so far")
        ax.set(xlabel="Trial", ylabel="Mean applicable jobs", title=self.study_name)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            fig.savefig(self.output / f"optimization_history.{suffix}", dpi=180)
        plt.close(fig)
        return result

    def tune(self):
        self.output.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            direction="maximize", study_name=self.study_name,
            storage=f"sqlite:///{(self.output / 'optuna.sqlite3').resolve().as_posix()}",
            sampler=optuna.samplers.TPESampler(seed=self.sampler_seed, n_startup_trials=5),
            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=2)
            if self.pruning else optuna.pruners.NopPruner(), load_if_exists=True,
        )
        settings = {
            "configs": self.configs, "data": dataset_details(self.dataset),
            "seeds": {str(k): values for k, values in self.seeds.items()},
            "algorithm": self.algorithm, "scope": self.scope, "version": 4,
            "last_n": self.last_n, "sampler_seed": self.sampler_seed, "pruning": self.pruning,
            "search_space": SEARCH_SPACE, "objective": "equal mean of per-pair mean tail jobs",
            "project_code": {name: hashlib.sha256((PROJECT_ROOT / "UIR/Scripts" / name).read_bytes()).hexdigest()
                             for name in ["Dataset.py", "CourseRecEnv.py", "Reinforce.py"]},
        }
        previous = study.user_attrs.get("settings")
        if previous is not None and previous != settings:
            raise ValueError(f"Settings or project code changed for {self.study_name}. Use a new --output.")
        if previous is None and study.trials:
            raise ValueError("Existing study has no settings. Use a new --output.")
        existing_trials = len(study.trials)
        if previous is None:
            study.set_user_attr("settings", settings)
            self.enqueue_config(study)
        restarted = {trial.user_attrs.get("restart_of") for trial in study.trials}
        for trial in study.trials:
            interrupted = trial.state == TrialState.RUNNING or (
                trial.state == TrialState.FAIL and trial.user_attrs.get("interrupted"))
            if interrupted and trial.number not in restarted:
                if trial.state == TrialState.RUNNING:
                    study.tell(trial.number, state=TrialState.FAIL)
                study.enqueue_trial(trial.params, user_attrs={"restart_of": trial.number})
                print(f"[RESTART] Trial {trial.number}: same parameters, from the first run.")
        finished = sum(trial.state in {TrialState.COMPLETE, TrialState.PRUNED}
                       for trial in study.trials)
        remaining = max(0, self.n_trials - finished)
        runs = sum(len(self.seeds[config["k"]]) for config in self.configs)
        print(f"[STUDY] {self.study_name}: {finished}/{self.n_trials} trials finished.")
        print(f"[BUDGET] Up to {runs} training runs per trial.")
        try:
            if remaining:
                study.sampler = optuna.samplers.TPESampler(
                    seed=(self.sampler_seed + existing_trials) % (2**32), n_startup_trials=5)
                study.optimize(self.objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)
        finally:
            result = self.export(study)
        return result


def save_summary(output: Path, scope: str) -> None:
    """Collect the best results already available for this algorithm and scope."""
    rows = []
    for folder, _ in make_groups(list(REWARDS), K_VALUES, scope):
        path = output / folder / "best_params.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema") != 4:
            continue
        rows.append({"algorithm": result["algorithm"], "scope": scope, "study": result["study"],
                     "rewards": ", ".join(result["rewards"]),
                     "k_values": ",".join(map(str, result["k_values"])),
                     "trial": result["trial"], "jobs": result["value"], "last_n": result["last_n"],
                     "total_steps": result["total_steps"], "eval_freq": result["eval_freq"]})
    columns = ["algorithm", "scope", "study", "rewards", "k_values", "trial", "jobs",
               "last_n", "total_steps", "eval_freq"]
    pd.DataFrame(rows, columns=columns).to_csv(output / "summary.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize course recommendation with Optuna.")
    parser.add_argument("--config", default="UIR/config/run.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT.parent / "optuna_results")
    parser.add_argument("--algorithms", "--algorithm", dest="algorithms", nargs="+",
                        choices=list(ALGORITHMS), default=["MaskablePPO"])
    parser.add_argument("--scope", choices=["reward-k", "reward", "all"], default="reward-k")
    parser.add_argument("--rewards", "--reward", "--experiments", dest="rewards", nargs="+",
                        type=lambda value: REWARD_ALIASES.get(value, value),
                        choices=list(REWARDS), default=list(REWARDS))
    parser.add_argument("--k-values", nargs="+", type=int, choices=K_VALUES, default=K_VALUES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Use this same seed list for every selected k, reward and algorithm.")
    parser.add_argument("--nb-courses", type=int, default=100)
    parser.add_argument("--nb-jobs", type=int, default=100)
    parser.add_argument("--trials", type=int, default=20, help="Target COMPLETE + PRUNED trials per study.")
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--last-n", type=int, default=50)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if min(args.trials, args.last_n, args.nb_courses, args.nb_jobs) < 1:
        parser.error("trials, last_n, nb_courses and nb_jobs must be positive.")
    for name in ["algorithms", "rewards", "k_values"]:
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"Choose distinct {name}.")
    if args.seeds is not None and (len(args.seeds) != len(set(args.seeds))
                                   or min(args.seeds) < 0 or max(args.seeds) >= 2**32):
        parser.error("Choose distinct seeds between 0 and 2**32 - 1.")
    if not 0 <= args.sampler_seed < 2**32:
        parser.error("sampler-seed must be between 0 and 2**32 - 1.")
    if args.pruning and args.seeds is not None and len(args.seeds) < 3:
        parser.error("Pruning requires at least three seeds.")
    args.output = args.output.resolve()
    if args.output == PROJECT_ROOT or PROJECT_ROOT in args.output.parents:
        parser.error("Choose an output directory outside the repository.")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    config = load_config(args, args.output)
    if config["eval_freq"] < 1 or config["total_steps"] < config["eval_freq"] * args.last_n:
        raise ValueError("Require total_steps >= eval_freq * last_n, with eval_freq >= 1.")
    dataset = Dataset(config)
    print(dataset)
    if len(dataset.courses) != args.nb_courses or len(dataset.jobs) != args.nb_jobs:
        raise ValueError("Loaded dataset sizes differ from --nb-courses / --nb-jobs.")
    details = dataset_details(dataset)
    seeds = {k: list(args.seeds) for k in args.k_values}
    for algorithm in args.algorithms:
        output = args.output / algorithm / args.scope
        output.mkdir(parents=True, exist_ok=True)
        path = output / "dataset.json"
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != details:
            raise ValueError("Dataset changed. Use a new --output.")
        write_json(path, details)
        print(f"Algorithm: {algorithm}; scope: {args.scope}; output: {output}")
        for folder, pairs in make_groups(args.rewards, args.k_values, args.scope):
            configs = []
            for reward, k in pairs:
                base = copy.deepcopy(config)
                base.update(REWARDS[reward])
                base.update(model=ALGORITHMS[algorithm], name_exp=reward, k=k, seed=seeds[k][0])
                configs.append(base)
            tuner = ReinforceTuner(configs, dataset, output / folder, seeds, algorithm, args.scope,
                                   args.trials, args.last_n, args.sampler_seed, args.pruning)
            result = tuner.tune()
            if result is not None:
                print(f"[BEST] {result['study']}: {result['value']:.4f} applicable jobs.")
            save_summary(output, args.scope)


if __name__ == "__main__":
    main()
