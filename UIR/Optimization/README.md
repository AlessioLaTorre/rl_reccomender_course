# Hyperparameter optimization

Optimize PPO or MaskablePPO with Optuna.
Use the Python 3.12 environment from the [main README](../../README.md).
Run commands from the repository root, where `requirements.txt` is located.

## 1. Check that it runs

Run one short trial with one seed:

```bash
python -m UIR.Optimization.tuning_ready --rewards UIR-gap --k-values 2 --trials 1 --seeds 42 --total-steps 256 --eval-freq 128 --last-n 2 --output ../optuna_check
```

Use the full training budget for the actual search.

## 2. Choose what to optimize

| `--scope` | Result |
|---|---|
| `reward-k` | One set of hyperparameters for each reward/k pair. This is the default. |
| `reward` | One set per reward, shared across the selected k values. |
| `all` | One set shared across all selected rewards and k values. |

**One experiment:**

```bash
python -m UIR.Optimization.tuning_ready --scope reward-k --rewards UIR-gap --k-values 2 --trials 20
```

**One set per reward, using all k values:**

```bash
python -m UIR.Optimization.tuning_ready --scope reward --trials 20
```

**One set for all rewards and all k values:**

```bash
python -m UIR.Optimization.tuning_ready --scope all --trials 20
```

Defaults select all four rewards: `Employability`, `UIR-gap`, `UIR-threshold`, `EUIR-gap`,
and all k values: `2 3 4 5`. Select subsets with `--rewards` and `--k-values`.
Omit these two options with `--scope reward-k` to optimize all 16 pairs separately.

## 3. Choose masking

`MaskablePPO` is the default and uses masking during training and evaluation.
Use `--algorithm PPO` for PPO without masking.

Add `--algorithms MaskablePPO PPO` to run both. Each has separate studies and best parameters.
For example:

```bash
python -m UIR.Optimization.tuning_ready --scope all --algorithms MaskablePPO PPO --rewards UIR-gap EUIR-gap --k-values 2 3 --trials 20
```

## 4. Choose a budget

- **1 short trial:** check execution, as in step 1.
- **20 trials:** start with one reward/k pair.
- **50 trials:** a larger, focused search if time allows it.

Start with individual experiments, then try shared modes. These are starting budgets.
`--trials` applies to each study and algorithm. A complete `all` trial with all
rewards and k values requires **80 training runs per algorithm**.

Optional `--pruning` compares candidates after 2, then 4 seed rounds across all pairs.
A round trains every selected pair with the same seed.
It requires at least 3 seeds. Pruned candidates count towards `--trials`.

## Data and scoring

Defaults: **100 courses, 100 jobs**, dataset seed **42**, level-3 skills,
all learners filtered with `max_cv_skills=15`, preferences disabled, and CPU.
Change the device with `--device`.

Seeds are always the same for every k, reward and algorithm: `42 43 44 45 50`.
Use `--seeds 42 43` to replace this list for all selected configurations.
Keep data, seeds, and training budget consistent in comparisons.

`UIR/config/run.yaml` supplies training steps and evaluation frequency
(500,000 and 2,000). Override them with `--total-steps` and `--eval-freq`.

The score is mean employability over the last **50 scheduled evaluations** per run.
Seeds are averaged within each reward/k pair. All selected pairs receive equal weight.
Evaluation uses the project's learner set.

For the full dataset, add:

```text
--nb-courses 3000 --nb-jobs 3587 --output ../optuna_full
```

## Search space

Edit `SEARCH_SPACE` near the top of `tuning_ready.py` to change these ranges.

| Parameter | Values / range |
|---|---|
| `n_steps` | 256, 512, 1024 |
| `num_minibatches` | 4, 8, 16 |
| `batch_size` | `n_steps / num_minibatches` |
| `n_epochs` | 4, 6, 8 |
| Initial `clip_range` | 0.10–0.30 |
| Initial `ent_coef` | 1e-6–0.03, logarithmic |
| `gamma` | 0.90–0.99 |
| `gae_lambda` | 0.90–0.98 |
| `lr_initial` | 1e-4–1e-3, logarithmic |
| `lr_final` | 1e-6–1e-4, logarithmic |
| `warmup_frac` | 0.00–0.08 |
| `start_at` | 0.50–0.85 |

TPE uses 5 startup trials. YAML parameters are tried first if they fit this space.
The search sets `use_standard=False`; other parameters, including `max_grad_norm`
and `target_kl`, remain from the YAML. Project schedules stay active: learning-rate
warmup/decay and callback adjustments towards entropy 0.006 and clipping 0.12.

## Results and resume

Results: `../optuna_results/<algorithm>/<scope>/`. Change the root with `--output`.
Each study saves:

- `optuna.sqlite3`: resume state.
- `trial*/results_k*/seed*/`: logs, configurations, status.
- `trials.csv`, `per_seed.csv`, `per_pair.csv`: scores.
- `best_params.json`, `best_hypers.yaml`, `best_configs/`: selected parameters/configurations.
- `optimization_history.png` and `.pdf`: score plots.

The search saves parameters, not model weights. Final retraining is separate.

Rerun the same command to continue. Completed trials are kept; an interrupted
candidate restarts from its first run. Increase `--trials` to extend the total target.
For changed settings or code, use a new `--output`. Use one process per output directory.
