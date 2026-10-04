"""Build manuscript assets from the frozen FGF2 pilot; never fit a model.

Run from anywhere. Tables and scientific plots are derived from saved results
and author-exported data. The B3 curve is a selection-exposed reference.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "simulation"
OUT = PILOT / "results/fgf2_pilot"
FIG = ROOT / "paper/figures"
TAB = ROOT / "paper/tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
results = json.loads((OUT / "calibration_results.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 180})

EXPANDED = PILOT / "results/fgf2_expanded"
expanded = json.loads((EXPANDED / "expanded_results.json").read_text(encoding="utf-8"))
protocol = json.loads((PILOT / "configs/fgf2_expanded_protocol.json").read_text(encoding="utf-8"))
names = {
    "persistence": "Persistence",
    "current_input": "Input only",
    "current_response": "Input+response (IR)",
    "current_response_clock": "IR + clock",
    "local_slope": "IR + slope",
    "causal_history": "Causal history",
    "history_clock": "History + clock",
}


def write_table(name, spec, header, body):
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [row if row.startswith(r"\midrule") else row + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", newline="\n")


def bold_min(values):
    best = min(values)
    return [(r"\textbf{%.5f}" if v == best else "%.5f") % v for v in values]


def rmse_rows(models, columns):
    """One row per model from MSE getters; the smallest RMSE per column is bold."""
    table = {m: [math.sqrt(get(m)) for get in columns] for m in models}
    cells = {m: [] for m in models}
    for j in range(len(columns)):
        for m, cell in zip(models, bold_min([table[m][j] for m in models])):
            cells[m].append(cell)
    return [names[m] + " & " + " & ".join(cells[m]) for m in models]


pilot = {int(r["horizon_min"]): r["models"] for r in results["forecasting"]}
boundary = pilot[10]["causal_history"]["test"]["time_support"]["max_train_origin_min"]
write_table("forecast_rmse", "lrrrrr",
            rf"Predictor & 2 min & 10 min & Val. & $\le${boundary:g} & $>${boundary:g}",
            rmse_rows([m for m in names if m != "history_clock"],
                      [lambda m: pilot[2][m]["test"]["run_equal_mse"],
                       lambda m: pilot[10][m]["test"]["run_equal_mse"],
                       lambda m: pilot[10][m]["validation_condition_equal_mse"],
                       lambda m: pilot[10][m]["test"]["time_support"]["within"]["run_equal_mse"],
                       lambda m: pilot[10][m]["test"]["time_support"]["beyond"]["run_equal_mse"]]))

scored = expanded["horizons"]["10"]["roles"]
roles = ("validation", "test_new_protocol", "test_new_protocol_inferred_timing", "test_new_concentration_same_session")


def wins(role, rival):
    history, other = scored[role]["causal_history"]["per_run"], scored[role][rival]["per_run"]
    return f"{sum(history[k]['mse'] < other[k]['mse'] for k in history)}/{len(history)}"


fresh = ["persistence", "current_response", "current_response_clock", "local_slope", "causal_history", "history_clock"]
body = rmse_rows(fresh, [lambda m, r=r: scored[r][m]["run_equal_mse"] for r in roles])
body += [r"\midrule",
         "History beats clock & " + " & ".join(wins(r, "current_response_clock") for r in roles),
         "History beats persistence & " + " & ".join(wins(r, "persistence") for r in roles)]
write_table("expanded_rmse", "lrrrr", r"10-min predictor & Val. & A & B & C", body)

labels = {"fgf_sus": "Sustained", "fgf_3_20": "3/20 pulses", "fgf_sp_5": "Single 5 min",
          "fgf_sp_10": "Single 10 min", "fgf_sp_60": "Single 60 min", "fgf_mixed_new": "Mixed, new conc."}
lopo = expanded["leave_one_protocol_out"]
lopo_models = ("persistence", "current_response_clock", "causal_history", "history_clock")
write_table("lopo_rmse", "lrrrr", r"Held-out protocol & Persist. & + clock & History & Hist.+clock",
            [label + " & " + " & ".join(bold_min([lopo[g][m] for m in lopo_models])) for g, label in labels.items()])

inventory = {row["condition_id"]: row for row in expanded["inventory"]}
viewed = set(protocol["roles"]["viewed_reference"])
tokens = ("0-25ng", "2-5ng", "25ng", "250ng")
body = []
for proto, label, role in (("fgf_sus", "Sustained", "Train"), ("fgf_3_20", "3/20 pulses", "Train"),
                           ("fgf_sp_5", "Single 5 min", "Val."), ("fgf_sp_10", "Single 10 min", "Test A"),
                           ("fgf_sp_60", "Single 60 min", "Test B"), ("fgf_mixed", "Mixed pulses", "Test C")):
    keys = [f"{proto}_{t}" for t in tokens]
    cells = ["--" if k not in inventory else str(inventory[k]["n_cells"]) + (r"$^\dagger$" if k in viewed else "")
             for k in keys]
    frames = next(inventory[k]["n_times"] for k in keys if k in inventory)
    body.append(f"{label} & {role} & " + " & ".join(cells) + f" & {frames}")
write_table("inventory", "llrrrrr", r"Protocol & Role & 0.25 & 2.5 & 25 & 250 & Frames", body)

body = []
for k, label in (("positive_single_lag", "Positive single lag"),
                 ("stable_filter_bank", "Stable filter bank"),
                 ("training_mean_constant", "Training-mean constant"),
                 ("published_B3_predictive_mean", "Author B3 reference")):
    vals = [results["common_window_open_loop_comparison"]["condition_rmse"][t][k] for t in ("2-5ng", "250ng")]
    body.append(label + " & " + " & ".join(f"{v:.5f}" for v in vals))
write_table("open_loop_rmse", "lrr", r"Population-mean model & 2.5 ng/ml & 250 ng/ml", body)

lines = [r"\begin{tabular}{lcrr}", r"\toprule",
         r"Dose & Matched features & Cells & $\overline{\Delta y_{+10}}$ \\", r"\midrule"]
for token, dose in (("2-5ng", "2.5"), ("250ng", "250")):
    for suffix, label in (("current_only", "$y$"), ("current_and_slope", "$y$, slope")):
        row = results["exploratory_history_matching"]["analyses"][token + "_" + suffix]
        lines.append(f"{dose} & {label} & {row['n_matched_cells']}/{row['n_eligible_cells']} & {row['mean_late_minus_early_target']:+.6f}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "matched_history.tex").write_text("\n".join(lines) + "\n", newline="\n")

curves = pd.read_csv(OUT / "input_output_test_curves.csv")
source = PILOT / "public_data/fgf2_author_repository"
prefix = source / "Inference_results/Fgf_B3/results/sus_3_20"
fig, axes = plt.subplots(2, 2, figsize=(7.08, 3.05), sharex="col",
                         gridspec_kw={"height_ratios": [3.0, 1.0]})
colors = {"measured_mean": "#222222", "positive_single_lag": "#CB7022",
          "stable_filter_bank": "#26808B", "B3": "#8663A8"}
for j, (token, dose) in enumerate((("2-5ng", 2.5), ("250ng", 250.0))):
    data = curves[curves.run_id == "fgf_mixed_" + token]
    top, bottom = axes[:, j]
    for k, label in (("measured_mean", "Measured cell mean"),
                     ("positive_single_lag", "Positive single lag"),
                     ("stable_filter_bank", "Stable filter bank")):
        top.plot(data.time_min, data[k], label=label, color=colors[k],
                 lw=1.5 if k == "measured_mean" else 1.2,
                 ls="--" if k == "stable_filter_bank" else "-")
    time = np.loadtxt(prefix / "sim_post_times.txt", delimiter=",").ravel()
    raw = np.loadtxt(prefix / f"sim_post_mixed_{token}_measurements.txt", delimiter=",")
    top.plot(time, raw.mean(axis=0), label="Author B3 reference", color=colors["B3"], lw=1.1)
    top.set_title(f"({chr(97+j)}) {dose:g} ng/ml command amplitude", loc="left")
    top.set_ylim(.995, 1.17)
    top.grid(axis="y", alpha=.16)
    if j == 0:
        top.set_ylabel("Normalized FRET ratio")
    for a, b in ((1, 4), (24, 54), (114, 119)):
        bottom.fill_between([a, b], [dose, dose], color="#52697C", alpha=.8, step="post")
    bottom.set_xlim(0, 258)
    bottom.set_ylim(0, dose * 1.12)
    bottom.set_yticks([0, dose])
    bottom.set_xlabel("Exported protocol time (min)")
    if j == 0:
        bottom.set_ylabel("Command\n(ng/ml)")
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.03), ncol=4, frameon=False)
fig.subplots_adjust(left=.075, right=.99, bottom=.14, top=.83, hspace=.18, wspace=.20)
fig.savefig(FIG / "mixed_response.pdf", bbox_inches="tight", pad_inches=.02)
fig.savefig(FIG / "mixed_response.png", bbox_inches="tight", pad_inches=.02)
plt.close(fig)

pred = pd.read_csv(OUT / "forecasts_10min.csv")
fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.65), sharex=True)
for ax, token, label in zip(axes, ("2-5ng", "250ng"), ("2.5 ng/ml", "250 ng/ml")):
    frame = pred[pred.run_id == "fgf_mixed_" + token].copy()
    frame["target_time"] = frame.time_s / 60.0 + 10.0
    avg = frame.groupby("target_time")[["target", "current_response_clock", "causal_history"]].mean()
    ax.plot(avg.index, avg.target, color="#222222", lw=1.3, label="Measured mean")
    ax.plot(avg.index, avg.current_response_clock, color="#A46F24", lw=1.0, ls="--", label="Current + clock")
    ax.plot(avg.index, avg.causal_history, color="#167B83", lw=1.1, label="Causal history")
    ax.set_title(label, loc="left", fontsize=8, pad=2)
    ax.set_ylim(.995, 1.16)
    ax.set_ylabel("FRET ratio")
    ax.grid(axis="y", alpha=.15)
    ax.set_xlim(0, 258)
axes[1].set_xlabel("Target protocol time (min)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.035), ncol=3, frameon=False, fontsize=6.6)
fig.subplots_adjust(left=.15, right=.98, bottom=.16, top=.84, hspace=.4)
fig.savefig(FIG / "forecast_mean_10min.pdf", bbox_inches="tight", pad_inches=.02)
fig.savefig(FIG / "forecast_mean_10min.png", bbox_inches="tight", pad_inches=.02)
plt.close(fig)

audit = {
    "source_result_sha256": hashlib.sha256((OUT / "calibration_results.json").read_bytes()).hexdigest(),
    "expanded_result_sha256": hashlib.sha256((EXPANDED / "expanded_results.json").read_bytes()).hexdigest(),
    "generated_tables": [p.name for p in sorted(TAB.glob("*.tex"))],
    "generated_figures": [p.name for p in sorted(FIG.glob("*.pdf"))],
    "fitted_models": False,
    "note": "Plots summarize saved pilot outputs; B3 is an original-author selection-exposed reference.",
}
(ROOT / "paper/assets_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", newline="\n")
print(json.dumps(audit, indent=2))
