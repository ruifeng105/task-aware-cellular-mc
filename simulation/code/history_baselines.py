"""Causal, run-grouped baselines for application-first chemical response data.

This is a baseline harness, not the proposed novel AI controller. It requires
manually verified long-format measurements; it does not guess workbook layout.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"run_id", "cell_id", "history_id", "time_s",
            "input_concentration", "input_unit", "response", "response_unit"}
TAUS_S = (60.0, 600.0, 3600.0)


def validate_raw(df):
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.copy()
    for col in ("run_id", "cell_id", "history_id", "input_unit", "response_unit"):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identifier or unit: {col}")
        df[col] = df[col].astype(str)
    for col in ("input_unit", "response_unit"):
        if df[col].nunique() != 1:
            raise ValueError(f"Do not pool incompatible units: {col}")
    for col in ("time_s", "input_concentration", "response"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    for col in ("time_s", "input_concentration"):
        if not np.isfinite(df[col].to_numpy()).all():
            raise ValueError(f"Nonfinite {col}; verify protocol and timing")
    if (df.input_concentration < 0).any():
        raise ValueError("Negative chemical concentration")
    if np.isinf(df.response.to_numpy()).any():
        raise ValueError("Infinite response")
    if df.duplicated(["run_id", "cell_id", "time_s"]).any():
        raise ValueError("Duplicate cell timestamps")
    if "eligible" not in df:
        df["eligible"] = True
    else:
        mapping = {"true": True, "false": False, "1": True, "0": False}
        values = df.eligible.astype(str).str.strip().str.lower().map(mapping)
        if values.isna().any():
            raise ValueError("eligible must contain true/false or 1/0")
        df["eligible"] = values
    return df.sort_values(["run_id", "cell_id", "time_s"])


def prepare(df, max_gap_s):
    """Features at t use measurements at or before t, never target values.

Only transitions with unchanged recorded input are retained. This baseline
forecasts the next observation, not arbitrary unexecuted interventions.
EMA initialization is computational, not an inferred physical initial state.
"""
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("A positive finite max_gap_s is required")
    df = validate_raw(df)
    records = []
    stats = dict(raw_rows=len(df), missing_response_rows=int(df.response.isna().sum()),
                 input_change_transitions_skipped=0, gap_transitions_skipped=0)
    for (run, cell), g in df.groupby(["run_id", "cell_id"], sort=False):
        g = g.reset_index(drop=True)
        if g.history_id.nunique() != 1:
            raise ValueError("One tracked cell must have one history_id")
        eligible = g.eligible.to_numpy(bool)
        if np.any(np.diff(eligible.astype(int)) > 0):
            raise ValueError("A censored cell cannot re-enter this trajectory")
        t = g.time_s.to_numpy(float)
        c = g.input_concentration.to_numpy(float)
        y = g.response.to_numpy(float)
        ec = np.repeat(c[0], len(TAUS_S))
        ey = np.zeros(len(TAUS_S))
        has_y = False
        dose = 0.0
        for i in range(len(g)):
            dt = 0.0 if i == 0 else t[i] - t[i - 1]
            if i > 0:
                if dt <= 0:
                    raise ValueError("Timestamps must increase")
                # Piecewise-constant input, explicitly assumed between samples.
                dose += c[i - 1] * dt
                decay = np.exp(-dt / np.asarray(TAUS_S))
                ec = decay * ec + (1.0 - decay) * c[i]
            if np.isfinite(y[i]) and eligible[i]:
                if not has_y:
                    ey[:] = y[i]
                    has_y = True
                elif i > 0:
                    ey = decay * ey + (1.0 - decay) * y[i]
            if i + 1 == len(g) or not (eligible[i] and eligible[i + 1]):
                continue
            horizon = t[i + 1] - t[i]
            if horizon > max_gap_s:
                stats["gap_transitions_skipped"] += 1
                continue
            if not np.isclose(c[i], c[i + 1], rtol=0, atol=1e-12):
                stats["input_change_transitions_skipped"] += 1
                continue
            if not (np.isfinite(y[i]) and np.isfinite(y[i + 1])):
                continue
            row = dict(run_id=run, cell_id=cell, history_id=g.history_id.iloc[0],
                       time_s=t[i], horizon_s=horizon, target=y[i + 1],
                       current_y=y[i], c=c[i], c2=c[i] ** 2,
                       y2=y[i] ** 2, cy=c[i] * y[i],
                       elapsed_s=t[i] - t[0], cumulative_input=dose)
            for j, tau in enumerate(TAUS_S):
                row[f"ema_c_{int(tau)}s"] = ec[j]
                row[f"ema_y_{int(tau)}s"] = ey[j]
            records.append(row)
    out = pd.DataFrame(records)
    stats["usable_transitions"] = len(out)
    if out.empty:
        raise ValueError("No eligible transitions; inspect timing, censoring, and input")
    return out, stats


def validate_split(rows, split):
    sets = {name: set(map(str, split.get(name, []))) for name in ("train", "validation", "test")}
    if any(not v for v in sets.values()):
        raise ValueError("Explicit nonempty train/validation/test run lists required")
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Run leakage between splits")
    observed = set(rows.run_id)
    listed = set.union(*sets.values())
    if observed != listed:
        raise ValueError(f"Unassigned runs: {sorted(observed-listed)}; absent runs: {sorted(listed-observed)}")
    return {name: rows[rows.run_id.isin(ids)].copy() for name, ids in sets.items()}


def weights(rows):
    counts = rows.groupby("run_id").size()
    w = rows.run_id.map(1.0 / counts).to_numpy(float)
    return w / w.sum()


def run_metric(rows, prediction):
    err = prediction - rows.target.to_numpy(float)
    tmp = pd.DataFrame({"run_id": rows.run_id.to_numpy(), "se": err ** 2, "ae": abs(err)})
    per = tmp.groupby("run_id").agg(mse=("se", "mean"), mae=("ae", "mean"))
    return dict(run_equal_mse=float(per.mse.mean()), run_equal_mae=float(per.mae.mean()),
                n_runs=len(per), n_transitions=len(rows),
                per_run={str(k): {x: float(v) for x, v in row.items()} for k, row in per.iterrows()})


def fit_ridge(rows, columns, alpha):
    x = rows[columns].to_numpy(float)
    y = rows.target.to_numpy(float)
    w = weights(rows)
    mu = w @ x
    scale = np.sqrt(w @ ((x - mu) ** 2))
    scale[scale < 1e-12] = 1.0
    mean_y = float(w @ y)
    a = (x - mu) / scale
    aw = a * np.sqrt(w)[:, None]
    yw = (y - mean_y) * np.sqrt(w)
    beta = np.linalg.solve(aw.T @ aw + alpha * np.eye(a.shape[1]), aw.T @ yw)
    return dict(columns=columns, alpha=float(alpha), mean=mu.tolist(), scale=scale.tolist(),
                beta=beta.tolist(), intercept=mean_y)


def predict(rows, model):
    x = rows[model["columns"]].to_numpy(float)
    return ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["beta"]) + model["intercept"]


def run_baselines(rows, split):
    parts = validate_split(rows, split)
    features = {
        "input_only": ["c", "c2", "horizon_s"],
        "current_response": ["c", "c2", "current_y", "y2", "cy", "horizon_s"],
        "causal_history": ["c", "c2", "current_y", "y2", "cy", "horizon_s", "elapsed_s", "cumulative_input"]
        + [f"ema_{v}_{int(tau)}s" for v in ("c", "y") for tau in TAUS_S],
    }
    results = {"persistence": {"test": run_metric(parts["test"], parts["test"].current_y.to_numpy(float))}}
    predictions = parts["test"][["run_id", "cell_id", "history_id", "time_s", "target"]].copy()
    predictions["persistence"] = parts["test"].current_y.to_numpy(float)
    for name, columns in features.items():
        candidates = []
        for alpha in (1e-6, 1e-4, 1e-2, 1.0, 100.0):
            model = fit_ridge(parts["train"], columns, alpha)
            score = run_metric(parts["validation"], predict(parts["validation"], model))["run_equal_mse"]
            candidates.append((score, model))
        score, model = min(candidates, key=lambda item: item[0])
        yp = predict(parts["test"], model)
        results[name] = dict(validation_run_equal_mse=score, model=model, test=run_metric(parts["test"], yp))
        predictions[name] = yp
    return results, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-gap-s", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.data, dtype={"run_id": str, "cell_id": str, "history_id": str})
    rows, stats = prepare(raw, args.max_gap_s)
    split = json.loads(Path(args.split).read_text())
    results, predictions = run_baselines(rows, split)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_results.json").write_text(json.dumps({"preparation": stats, "split": split, "models": results,
        "limitations": ["Measured constant-input one-step prediction only; not policy evaluation",
                        "Units and run/date identifiers require manual verification",
                        "Rows and cells are not independent experimental replications",
                        "No controller or inferred receptor states are implemented"]}, indent=2))
    predictions.to_csv(out / "test_predictions.csv", index=False)
    print(json.dumps({"status": "complete", "usable_transitions": len(rows), "test_runs": results["persistence"]["test"]["n_runs"]}))


if __name__ == "__main__":
    main()
