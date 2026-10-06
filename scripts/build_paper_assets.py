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
    "M2_b3_feedback": "B3 + feedback (M2)",
    "M3_history_plus_b3": "History + B3 (M3)",
}
B3_RESULTS = PILOT / "results/b3"
b3_forecast = json.loads((B3_RESULTS / "forecast_results.json").read_text(encoding="utf-8"))
readiness = json.loads((B3_RESULTS / "readiness_results.json").read_text(encoding="utf-8"))
example = json.loads((B3_RESULTS / "readiness_example.json").read_text(encoding="utf-8"))
NESTED = PILOT / "results/nested"
nested = json.loads((NESTED / "nested_results.json").read_text(encoding="utf-8"))
waiting = json.loads((B3_RESULTS / "waiting_results.json").read_text(encoding="utf-8"))


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
            rmse_rows([m for m in names if m in pilot[10]],
                      [lambda m: pilot[2][m]["test"]["run_equal_mse"],
                       lambda m: pilot[10][m]["test"]["run_equal_mse"],
                       lambda m: pilot[10][m]["validation_condition_equal_mse"],
                       lambda m: pilot[10][m]["test"]["time_support"]["within"]["run_equal_mse"],
                       lambda m: pilot[10][m]["test"]["time_support"]["beyond"]["run_equal_mse"]]))

scored = expanded["horizons"]["10"]["roles"]
roles = ("validation", "test_new_protocol", "test_new_protocol_inferred_timing", "test_new_concentration_same_session")


b3_scored = b3_forecast["scores"]["10"]


def metric(role, model):
    return (b3_scored if model in ("M2_b3_feedback", "M3_history_plus_b3") else scored)[role][model]


def wins(role, model, rival):
    ours, other = metric(role, model)["per_run"], metric(role, rival)["per_run"]
    return f"{sum(ours[k]['mse'] < other[k]['mse'] for k in ours)}/{len(ours)}"


fresh = ["persistence", "current_response", "current_response_clock", "local_slope", "causal_history", "history_clock",
         "M2_b3_feedback", "M3_history_plus_b3"]
body = rmse_rows(fresh, [lambda m, r=r: metric(r, m)["run_equal_mse"] for r in roles])
body += [r"\midrule",
         "History beats clock & " + " & ".join(wins(r, "causal_history", "current_response_clock") for r in roles),
         "History beats persistence & " + " & ".join(wins(r, "causal_history", "persistence") for r in roles),
         "M3 beats M2 & " + " & ".join(wins(r, "M3_history_plus_b3", "M2_b3_feedback") for r in roles)]
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
fig.savefig(FIG / "mixed_response.pdf", bbox_inches="tight", pad_inches=.02, metadata={"CreationDate": None})
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
fig.savefig(FIG / "forecast_mean_10min.pdf", bbox_inches="tight", pad_inches=.02, metadata={"CreationDate": None})
fig.savefig(FIG / "forecast_mean_10min.png", bbox_inches="tight", pad_inches=.02)
plt.close(fig)

# In-silico readiness study on B3 (reference-palette slots 1-2 plus line style and marker).
SERIES = {"short": ("#2a78d6", "-", "o", "3-min history"), "long": ("#eb6834", "--", "s", "30-min history")}
INK, MUTED, BAND = "#0b0b0b", "#52514e", "#f0efec"
fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.1))
ax = axes[0]
waits = np.array(readiness["readiness_curves"]["waits_min"])
ax.axvspan(6, 48, color=BAND, zorder=0, lw=0)
ax.text(27, .55, "candidate\nwaits", ha="center", va="center", fontsize=6.5, color=MUTED)
for name, (color, style, marker, label) in SERIES.items():
    ax.plot(waits, readiness["readiness_curves"][name], color=color, ls=style, lw=1.3, marker=marker,
            markevery=10, ms=3.5, label=label)
ax.set(xlim=(0, 120), ylim=(-.03, 1.03), xlabel="Wait after previous command (min)", ylabel="Readiness")
ax.set_title("(a) Readiness vs wait", loc="left")
ax = axes[1]
bins = readiness["matched_reporter_readiness"]["bins"]
centre = [(b["reporter_rise_from"] + b["reporter_rise_to"]) / 2 for b in bins]
for name, (color, style, marker, label) in SERIES.items():
    ax.plot(centre, [b[name]["readiness"] for b in bins], color=color, ls=style, lw=1.3, marker=marker, ms=3.5)
ax.set(ylim=(-.03, 1.03), xlabel="Current reporter rise (FRET $-$ 1)")
ax.set_title("(b) Readiness vs current reporter", loc="left")
ax = axes[2]
curve = readiness["fixed_wait_curve"]
ax.plot([q["mean_completion_min"] for q in curve], [q["success_rate"] for q in curve], color=MUTED, lw=1.0, zorder=1)
for q in curve:
    if q["wait_min"] in (80., 100., 120.):
        ax.plot(q["mean_completion_min"], q["success_rate"], marker=".", color=MUTED, ms=5)
        offset = (-34, -6) if q["wait_min"] == 120. else (3, -8)
        ax.annotate(f"fixed {q['wait_min']:g}", (q["mean_completion_min"], q["success_rate"]), textcoords="offset points",
                    xytext=offset, fontsize=6, color=MUTED)
ax.axhline(.9, color=MUTED, lw=.6, ls=":", zorder=0)
ax.text(99.5, .915, "q", fontsize=6.5, color=MUTED)
policy_marks = {"oracle": ("*", "oracle", (5, -3)), "readiness_history": ("o", "history-aware", (-30, -12)),
                "readiness_current": ("^", "current only", (6, -3)), "threshold_reset": ("x", "half-decay reset", (-62, 4))}
for key, (marker, label, offset) in policy_marks.items():
    point = readiness["policies"][key]
    ax.plot(point["mean_completion_min"], point["success_rate"], marker=marker, color=INK, ms=5.5, ls="none", zorder=3)
    ax.annotate(label, (point["mean_completion_min"], point["success_rate"]), textcoords="offset points", xytext=offset,
                fontsize=6.5, color=INK)
ax.set(xlim=(98, 143), ylim=(-.03, 1.05), xlabel="Mean completion time (min)", ylabel="Task success rate")
ax.set_title("(c) Waiting policies", loc="left")
for a in axes:
    a.grid(axis="y", alpha=.15)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.06), ncol=2, frameon=False)
fig.subplots_adjust(left=.07, right=.99, bottom=.22, top=.82, wspace=.32)
fig.savefig(FIG / "readiness_policies.pdf", bbox_inches="tight", pad_inches=.02, metadata={"CreationDate": None})
fig.savefig(FIG / "readiness_policies.png", bbox_inches="tight", pad_inches=.02)
plt.close(fig)

# Fig. 2: readiness-aware waiting on one B3 sample (30-min history colour as in Fig. 3; policies in neutral ink).
color = SERIES[example["history"]][0]
grid, ready = np.array(example["grid_min"]), np.array(example["ready"], dtype=bool)
first_ready = grid[ready][0]
fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.45, 2.45), sharex=True, gridspec_kw=dict(height_ratios=[1.45, 1]))
for a in (top, bottom):
    a.axvspan(6, 48, color=BAND, zorder=0, lw=0)
    a.axvspan(first_ready, grid[-1] + 2, color=MUTED, alpha=.12, zorder=0, lw=0)
top.axvspan(example["reporter"]["times_min"][0], 0, color=color, alpha=.15, zorder=0, lw=0)
top.text(-15, 1.012, f"{example['command_ng_per_ml']:g} ng/ml\ncommand", ha="center", va="bottom", fontsize=6, color=MUTED)
top.text(27, 1.152, "candidate\nwaits", ha="center", va="top", fontsize=6, color=MUTED)
top.plot(example["reporter"]["times_min"], example["reporter"]["fret"], color=color, lw=1.2, label="noise-free reporter")
top.plot(grid, example["observed"], ls="none", marker="o", ms=2.2, color=color, alpha=.75, mew=0,
         label="observations (2 min)")
BRANCH = {"fixed": (MUTED, ":", ".", "fixed 48"), "current": (INK, "--", "^", "current only"),
          "history": (INK, "-", "o", "history-aware")}
for name, (ink, style, marker, label) in BRANCH.items():
    branch = example["branches"][name]
    t, fret = np.array(branch["times_min"]), np.array(branch["fret"])
    top.plot(t, fret, color=ink, ls=style, lw=1.0)
    top.plot(t[0], fret[0], marker=marker, color=ink, ms=4, ls="none", zorder=3)
    top.hlines(fret[0] + example["required_rise"], t[0], t[-1], color=ink, lw=.7, ls=style, alpha=.8)
    mark = "✓" if branch["success"] else "✗"
    x, y, align = {"fixed": (58, 1.14, "center"), "current": (82, 1.112, "right"), "history": (106, 1.14, "center")}[name]
    top.text(x, y, f"{label} {mark}", fontsize=6, color=ink, ha=align)
top.set(ylim=(.995, 1.155), ylabel="FRET ratio")
top.set_title("(a) Reporter and probe responses at three waits", loc="left")
top.legend(loc="lower left", bbox_to_anchor=(.2, 0), frameon=False, fontsize=6, handlelength=1.6, borderaxespad=.2)
bottom.plot(grid, example["probability"]["history"], color=INK, lw=1.2, label="all observations")
bottom.plot(grid, example["probability"]["current"], color=INK, lw=.8, ls="--", label="current observation")
for name, (ink, style, marker, label) in BRANCH.items():
    if name != "fixed":
        k = int(np.flatnonzero(grid == example["branches"][name]["wait_min"])[0])
        bottom.plot(grid[k], example["probability"][name][k], marker=marker, color=ink, ms=4, ls="none", zorder=3)
bottom.axhline(example["q"], color=MUTED, lw=.6, ls=":")
bottom.text(-30, example["q"] + .03, f"q = {example['q']:g}", fontsize=6, color=MUTED)
bottom.text(first_ready + 2, .08, "ready", fontsize=6, color=MUTED)
bottom.set(ylim=(-.03, 1.05), xlim=(-32, 122), xlabel="Time after the command ends (min)", ylabel="Est. success prob.")
bottom.set_title("(b) Readiness estimate and probe decisions", loc="left")
bottom.legend(loc="upper left", bbox_to_anchor=(0, .86), frameon=False, fontsize=6, handlelength=1.6, borderaxespad=.2)
for a in (top, bottom):
    a.grid(axis="y", alpha=.15)
fig.subplots_adjust(left=.15, right=.98, bottom=.14, top=.93, hspace=.32)
fig.savefig(FIG / "readiness_example.pdf", bbox_inches="tight", pad_inches=.02, metadata={"CreationDate": None})
fig.savefig(FIG / "readiness_example.png", bbox_inches="tight", pad_inches=.02)
plt.close(fig)

# Six-page revision: Table I with B3 data use, Table II core models, Fig. 2 experimental prediction, Fig. 3 waiting.
roles_b3 = nested["b3_roles"]
role_label = {"train": "Train", "validation": "Val.", "test_new_protocol": "Test A",
              "test_new_protocol_inferred_timing": "Test B", "test_new_concentration_same_session": "Test C",
              "viewed_reference": "Viewed"}
protocol_label = {"fgf_sus": "Sustained", "fgf_3_20": "3/20 pulses", "fgf_sp_5": "Single 5 min",
                  "fgf_sp_10": "Single 10 min", "fgf_sp_60": "Single 60 min", "fgf_mixed": "Mixed"}
body = []
for role in role_label:
    recs = [r for r in expanded["inventory"] if r["role"] == role and r["included"]]
    for proto in dict.fromkeys(r["condition_id"].rsplit("_", 1)[0] for r in recs):
        sub = [r for r in recs if r["condition_id"].rsplit("_", 1)[0] == proto]
        conc = {r["condition_id"].replace("fgf_", "", 1): r["concentration_ng_ml"] for r in sub}
        fit = [f"{v:g}" for k, v in conc.items() if k in roles_b3["fitted"]]
        arch = [f"{v:g}" for k, v in conc.items() if k in roles_b3["architecture_likelihood"]]
        use = f"fit {', '.join(fit)}" if fit else f"arch. {', '.join(arch)}" if arch else "--"
        body.append(f"{protocol_label[proto]} & {role_label[role]} & {', '.join(f'{v:g}' for v in conc.values())} & "
                    f"{sum(r['n_cells'] for r in sub)} & {use}")
write_table("inventory_core", "lllrl", "Protocol & Role & ng/ml & Cells & B3 use", body)

core_cols = ("validation", "test_new_protocol", "test_new_protocol_inferred_timing", "test_new_concentration_same_session")
cmp10, cmp2 = nested["main"]["10"]["comparisons"], nested["main"]["2"]["comparisons"]
core = (("P", "P: persistence"), ("C", "C: current + slope"), ("O", "O: + reporter filters"),
        ("M", "M: + B3 increment"), ("H", "H: + input history"))
getters = [lambda m, r=r: cmp10[r]["rmse"][m] for r in core_cols] + [lambda m: cmp2["test_new_protocol"]["rmse"][m]]
cells = {m: [] for m, _ in core}
for get in getters:
    for (m, _), cell in zip(core, bold_min([get(m) for m, _ in core])):
        cells[m].append(cell)
body = [label + " & " + " & ".join(cells[m]) for m, label in core] + [r"\midrule"]
for a, b in (("H", "M"), ("M", "O"), ("O", "C")):
    counts = [f"{cmp10[r][f'{a}_beats_{b}']['conditions']}/{cmp10[r]['n_conditions']}" for r in core_cols]
    counts.append(f"{cmp2['test_new_protocol'][f'{a}_beats_{b}']['conditions']}/{cmp2['test_new_protocol']['n_conditions']}")
    body.append(f"{a} beats {b} & " + " & ".join(counts))
write_table("core_rmse", "lccccc", r"Model & Val. & A & B & C & A, 2 min", body)

MODEL_STYLE = {"P": ("#52514e", ":", "."), "C": ("#0b0b0b", "--", "x"), "O": ("#1baf7a", "-", "o"),
               "M": ("#4a3aa7", "-", "s"), "H": ("#e34948", "-", "^")}
ex = nested["example"]
fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.45, 2.75), gridspec_kw=dict(height_ratios=[1.3, 1]))
t_obs, q = np.array(ex["time_min"]), np.array(ex["quartiles"])
for a, b in ex["pulses"]:
    top.axvspan(a, b, color=MUTED, alpha=.18, lw=0, zorder=0)
top.fill_between(t_obs, q[0], q[2], color=BAND, lw=0, zorder=0, label="cells, middle 50%")
top.plot(t_obs, ex["observed"], ls="none", marker="o", ms=2, color=INK, mew=0, label="example cell")
for name in ("P", "M", "H"):
    color, style, _ = MODEL_STYLE[name]
    top.plot(ex["target_time_min"], ex["forecast"][name], color=color, ls=style, lw=1.0,
             label=f"{name} (cell RMSE {ex['cell_rmse'][name]:.4f})")
top.set(xlim=(0, t_obs.max()), ylim=(.98, 1.215), xlabel="Time (min)", ylabel="FRET ratio")
top.set_title("(a) Test A, 25 ng/ml: 10-min forecasts", loc="left")
top.legend(loc="upper right", ncol=2, frameon=False, fontsize=5.5, handlelength=1.5, borderaxespad=.1, columnspacing=.8)
phases = nested["phases"]["10"]
order = [p for p in ("stimulation", "next_command", "early_washout", "late") if p in phases]
tick = {"stimulation": "stimulation", "next_command": "next command", "early_washout": "early washout", "late": "late"}
for j, name in enumerate(("P", "C", "O", "M", "H")):
    color, _, marker = MODEL_STYLE[name]
    bottom.plot([i + (j - 2) * .11 for i in range(len(order))], [phases[p]["rmse"][name] for p in order],
                ls="none", marker=marker, ms=4, color=color, mew=1, label=name)
bottom.set_xticks(range(len(order)), [f"{tick[p]}\n({phases[p]['n_conditions']} cond.)" for p in order], fontsize=6)
bottom.set(ylabel="10-min RMSE", xlim=(-.5, len(order) - .5))
bottom.set_title("(b) Test A-C by forecast phase", loc="left")
bottom.legend(loc="lower left", ncol=5, frameon=False, fontsize=6, handletextpad=.1, columnspacing=.6, borderaxespad=.1)
for a in (top, bottom):
    a.grid(axis="y", alpha=.15)
fig.subplots_adjust(left=.15, right=.98, bottom=.13, top=.94, hspace=.8)
fig.savefig(FIG / "prediction_phases.pdf", bbox_inches="tight", pad_inches=.02, metadata={"CreationDate": None})
fig.savefig(FIG / "prediction_phases.png", bbox_inches="tight", pad_inches=.02)
plt.close(fig)

POLICY = {"current": ("^", "current"), "smoothed": ("D", "smoothed"), "history": ("o", "history-aware")}
noise_main, noise_zero = waiting["noise"]["0.005"], waiting["noise"]["0"]
fixed = noise_main["evaluation"]["fixed_calibrated"]
fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.1))
ax = axes[0]
for name, (color, style, marker, label) in SERIES.items():
    ax.plot(centre, [b[name]["readiness"] for b in bins], color=color, ls=style, lw=1.3, marker=marker, ms=3.5, label=label)
ax.set(ylim=(-.03, 1.03), xlabel="Current reporter rise (FRET $-$ 1)", ylabel="Readiness")
ax.set_title("(a) Readiness vs current reporter", loc="left")
ax.legend(loc="upper right", frameon=False, fontsize=6)
ax = axes[1]
ax.axhline(fixed["mean_completion_min"], color=MUTED, lw=.8, ls="--")
ax.text(2.45, fixed["mean_completion_min"] - .4, f"fixed {noise_main['choices']['fixed']['wait_min']:g} min",
        fontsize=6, color=MUTED, ha="right", va="top")
for i, name in enumerate(POLICY):
    marker = POLICY[name][0]
    for noise, block, face, dx in (("0", noise_zero, "white", -.12), ("0.005", noise_main, INK, .12)):
        row = block["evaluation"][name]
        ax.plot(i + dx, row["mean_completion_min"], marker=marker, ms=5.5, mfc=face, mec=INK, ls="none")
        ax.annotate(f"{100 * row['success_rate']:.0f}%", (i + dx, row["mean_completion_min"]), textcoords="offset points",
                    xytext=(0, -9), ha="center", fontsize=5.5, color=INK)
ax.plot([], [], marker="o", mfc="white", mec=INK, ls="none", label="noise 0")
ax.plot([], [], marker="o", mfc=INK, mec=INK, ls="none", label="noise 0.005")
ax.set_xticks(range(len(POLICY)), [POLICY[n][1] for n in POLICY], fontsize=6)
ax.set(xlim=(-.5, 2.5), ylim=(110.3, 123), ylabel="Mean completion (min)")
ax.set_title("(b) Noise ablation (evaluation)", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(.5, .86), frameon=False, fontsize=6, handletextpad=.2)
ax = axes[2]
curve = waiting["fixed_wait_curve_eval"]
ax.plot([p["mean_completion_min"] for p in curve], [p["success_rate"] for p in curve], color=MUTED, lw=1.0, zorder=1)
ax.plot(fixed["mean_completion_min"], fixed["success_rate"], marker=".", color=MUTED, ms=7, ls="none")
ax.annotate(f"fixed {noise_main['choices']['fixed']['wait_min']:g}", (fixed["mean_completion_min"], fixed["success_rate"]),
            textcoords="offset points", xytext=(5, 4), fontsize=6, color=MUTED)
ax.axhline(.9, color=MUTED, lw=.6, ls=":", zorder=0)
marks = dict(POLICY, oracle=("*", "oracle"))
offsets = {"current": (5, -11), "smoothed": (-6, -11), "history": (-2, 7), "oracle": (5, -3)}
for name, (marker, label) in marks.items():
    row = noise_main["evaluation"][name]
    ax.plot(row["mean_completion_min"], row["success_rate"], marker=marker, color=INK, ms=5.5, ls="none", zorder=3)
    ax.annotate(label, (row["mean_completion_min"], row["success_rate"]), textcoords="offset points",
                xytext=offsets[name], fontsize=6, color=INK, ha="right" if offsets[name][0] < 0 else "left")
ax.set(xlim=(100, 143), ylim=(.6, 1.02), xlabel="Mean completion time (min)", ylabel="Task success rate")
ax.set_title("(c) Calibrated rules, noise 0.005", loc="left")
for a in axes:
    a.grid(axis="y", alpha=.15)
fig.subplots_adjust(left=.07, right=.99, bottom=.22, top=.88, wspace=.36)
fig.savefig(FIG / "readiness_waiting.pdf", bbox_inches="tight", pad_inches=.02, metadata={"CreationDate": None})
fig.savefig(FIG / "readiness_waiting.png", bbox_inches="tight", pad_inches=.02)
plt.close(fig)

audit = {
    "source_result_sha256": hashlib.sha256((OUT / "calibration_results.json").read_bytes()).hexdigest(),
    "expanded_result_sha256": hashlib.sha256((EXPANDED / "expanded_results.json").read_bytes()).hexdigest(),
    "b3_forecast_result_sha256": hashlib.sha256((B3_RESULTS / "forecast_results.json").read_bytes()).hexdigest(),
    "b3_readiness_result_sha256": hashlib.sha256((B3_RESULTS / "readiness_results.json").read_bytes()).hexdigest(),
    "b3_readiness_example_sha256": hashlib.sha256((B3_RESULTS / "readiness_example.json").read_bytes()).hexdigest(),
    "nested_result_sha256": hashlib.sha256((NESTED / "nested_results.json").read_bytes()).hexdigest(),
    "b3_waiting_result_sha256": hashlib.sha256((B3_RESULTS / "waiting_results.json").read_bytes()).hexdigest(),
    "generated_tables": [p.name for p in sorted(TAB.glob("*.tex"))],
    "generated_figures": [p.name for p in sorted(FIG.glob("*.pdf"))],
    "fitted_models": False,
    "note": "Plots summarize saved pilot outputs; B3 is an original-author selection-exposed reference.",
}
(ROOT / "paper/assets_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", newline="\n")
print(json.dumps(audit, indent=2))
