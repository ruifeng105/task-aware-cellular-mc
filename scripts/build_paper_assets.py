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
robust = json.loads((B3_RESULTS / "waiting_robustness_results.json").read_text(encoding="utf-8"))
baselines = json.loads((B3_RESULTS / "waiting_baselines_results.json").read_text(encoding="utf-8"))
stress = json.loads((B3_RESULTS / "waiting_stress_results.json").read_text(encoding="utf-8"))
noprobe = json.loads((B3_RESULTS / "noprobe_results.json").read_text(encoding="utf-8"))
arx = json.loads((NESTED / "nested_arx_results.json").read_text(encoding="utf-8"))
margin = json.loads((B3_RESULTS / "waiting_margin_results.json").read_text(encoding="utf-8"))
ar1 = json.loads((B3_RESULTS / "ar1_likelihood_results.json").read_text(encoding="utf-8"))


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
acmp10, acmp2 = arx["main"]["10"]["comparisons"], arx["main"]["2"]["comparisons"]
lagged = ("A", "A+M")
rmse_of = lambda cmp, acmp, role, m: (acmp if m in lagged else cmp)[role]["rmse"][m]
# Descriptive predictor names (revision plan item 2); code keys P/C/O/A/M/A+M/H are unchanged.
SHORT = {"P": "Persist", "C": "Current", "O": "Filter", "A": "Lag", "M": "Filter+B3", "A+M": "Lag+B3", "H": "F+B3+Input",
         "M_phase": "F+B3 (phase)"}
core = tuple((m, SHORT[m]) for m in ("P", "C", "O", "A", "M", "A+M", "H"))
getters = ([lambda m, r=r: rmse_of(cmp10, acmp10, r, m) for r in core_cols]
           + [lambda m: rmse_of(cmp2, acmp2, "test_new_protocol", m)])
cells = {m: [] for m, _ in core}
for get in getters:
    for (m, _), cell in zip(core, bold_min([get(m) for m, _ in core])):
        cells[m].append(cell)
body = [label + " & " + " & ".join(cells[m]) for m, label in core] + [r"\midrule"]
for a, b in (("O", "C"), ("M", "O"), ("A+M", "A"), ("H", "M")):
    pick = lambda cmp, acmp: acmp if (a, b) == ("A+M", "A") else cmp
    counts = [f"{pick(cmp10, acmp10)[r][f'{a}_beats_{b}']['conditions']}/{cmp10[r]['n_conditions']}" for r in core_cols]
    counts.append(f"{pick(cmp2, acmp2)['test_new_protocol'][f'{a}_beats_{b}']['conditions']}/{cmp2['test_new_protocol']['n_conditions']}")
    body.append(f"{SHORT[a]} $<$ {SHORT[b]} & " + " & ".join(counts))
write_table("core_rmse", "lccccc", r"Model & Val. & A & B & C & A, 2 min", body)

conc_of = {r["condition_id"].replace("fgf_", "", 1): r["concentration_ng_ml"] for r in expanded["inventory"]}
fold_label = {"fgf_sus": "Sustained", "fgf_3_20": "3/20 pulses", "fgf_sp_5": "Single 5 min",
              "fgf_sp_10": "Single 10 min", "fgf_sp_60": "Single 60 min", "fgf_mixed_new": "Mixed, 0.25/25"}
body = []
for group, label in fold_label.items():
    fold = nested["lopo"][group]
    use = fold["b3_use"]
    used = use["fitted"] or use["architecture_likelihood"]
    assert not used or sorted(conc_of[k] for k in used) == [2.5, 250.], used   # the caption names 2.5 and 250 ng/ml
    if use["fitted"]:
        prior = "fit"
    elif use["architecture_likelihood"]:
        prior = "arch."
    else:
        prior = r"arch.$^\circ$" if group == "fgf_mixed_new" else "none"
    values = [fold["rmse"][m] for m in ("O", "M", "H")] + [arx["lopo"][group]["rmse"][m] for m in lagged]
    best = min(values)
    body.append(f"{label} & " + " & ".join((r"\textbf{%.5f}" if v == best else "%.5f") % v for v in values) + f" & {prior}")
write_table("lopo_core", "lcccccl", r"Held-out & Filter & F+B3 & F+B3+In & Lag & Lag+B3 & B3 use", body)

r0 = robust["repeats"]["0"]["0.005"]["rules"]
rule_label = (("current", "Current obs."), ("smoothed", "Smoothed (16 min)"), ("history", "History-aware"),
              ("fixed", f"Fixed {waiting['noise']['0.005']['choices']['fixed']['wait_min']:g} min"), ("oracle", "Oracle"))
row_of = lambda block, label: label + " & " + " & ".join(
    f"{100 * block[h]['success_rate']:.1f} & {block[h]['mean_completion_min']:.1f}" for h in ("short", "long"))
body = [row_of(r0[name], label) for name, label in rule_label[:4]]
for rule, suffix in (("pooled", ""), ("per_history", r" (each $\geq$0.9)")):
    choice = baselines["primary"]["conditioned"][rule]
    body.append(row_of(choice["evaluation"], "Fixed {:g}/{:g} min".format(*choice["waits_min"]) + suffix))
for target in ("0.92",):
    cell = margin["repeats"]["0"]["0.005"]["cells"][f"coarse/{target}"]
    body.append(row_of(cell["rules"]["conditioned"], "Fixed {:g}/{:g} min".format(*cell["choices"]["conditioned"]["waits_min"])
                       + f" (cal. {target})"))
body.append(row_of(r0["oracle"], "Oracle"))
write_table("waiting_history", "lcccc", r"Rule & $S_3$ (\%) & $T_3$ (min) & $S_{30}$ (\%) & $T_{30}$ (min)", body)

sensitivity = json.loads((B3_RESULTS / "waiting_sensitivity_results.json").read_text(encoding="utf-8"))
body = []
st_ = lambda row, mark="": f"{100 * row['success_rate']:.1f} / {row['mean_completion_min']:.1f}" + mark
signed = lambda x: f"{x:+.1f}".replace("-", "$-$")
for kappa, block in sensitivity["cells"].items():
    conditioned = baselines["kappa"][kappa]
    by_history = st_(conditioned["evaluation"]["all"], "" if conditioned["target_reached"] else r"$^\ast$")
    for noise, cell in block.items():
        rules = {r: st_(cell["rules"][r]["all"], "" if r == "oracle" or cell["reached"][r] else r"$^\ast$")
                 for r in ("fixed", "history", "smoothed", "current", "oracle")}
        diff = cell["paired"]["history-smoothed"]["completion"]
        low, high = cell["bootstrap"]["history-smoothed"]["completion"]
        body.append(f"{float(kappa):.1f} & {float(noise):.3f} & {cell['choices']['fixed']['wait_min']:g} & {rules['fixed']} & "
                    f"{by_history} & {rules['history']} & {rules['smoothed']} & {rules['current']} & {rules['oracle']}"
                    f" & {signed(diff)} [{signed(low)}, {signed(high)}]")
body.append(r"\midrule")
primary_by_history = st_(baselines["primary"]["conditioned"]["pooled"]["evaluation"]["all"])
for name, label in (("ar1", r"AR(1)$^\dagger$"), ("offset", r"offset$^\dagger$")):
    e = stress["mismatch"][name]
    body.append(f"0.5 & {label} & {waiting['noise']['0.005']['choices']['fixed']['wait_min']:g} & {st_(e['fixed_calibrated'])} & "
                f"{primary_by_history} & {st_(e['history'])} & {st_(e['smoothed'])} & {st_(e['current'])} & {st_(e['oracle'])} & "
                f"{signed(e['history']['mean_completion_min'] - e['smoothed']['mean_completion_min'])}")
a0 = ar1["repeats"]["0"]
reach = lambda arm, name: "" if a0["choices"][arm][name]["target_reached"] else r"$^\ast$"
rules = {r: st_(a0["rules"][f"independent_{r}"]["all"], reach("independent", r)) for r in ("history", "smoothed", "current")}
fixed_wait = a0["choices"]["independent"]["fixed"]["wait_min"]
oracle = st_(stress["mismatch"]["ar1"]["oracle"])
smoothed_t = a0["rules"]["independent_smoothed"]["all"]["mean_completion_min"]
ar1_mark = "" if a0["choices"]["ar1"]["target_reached"] else r"$^\ast$"
body.append(f"0.5 & AR(1), recal.$^\\ddagger$ & {fixed_wait:g} & {st_(a0['rules']['fixed']['all'])} & {st_(a0['rules']['conditioned']['all'])} & "
            f"{rules['history']} & {rules['smoothed']} & {rules['current']} & {oracle} & "
            f"{signed(a0['rules']['independent_history']['all']['mean_completion_min'] - smoothed_t)}")
body.append(f"0.5 & AR(1) lik.$^\\ddagger$ & -- & -- & -- & {st_(a0['rules']['ar1_history']['all'], ar1_mark)} & "
            f"-- & -- & -- & {signed(a0['rules']['ar1_history']['all']['mean_completion_min'] - smoothed_t)}")
write_table("waiting_sensitivity", "cccccccccc",
            r"$\kappa$ & Noise & Wait & Fixed & By history & History & Smoothed & Current & Oracle & $\Delta T_{\rm H-S}$", body)

MODEL_STYLE = {"P": ("#52514e", ":", "."), "C": ("#0b0b0b", "--", "x"), "O": ("#1baf7a", "-", "o"),
               "A": ("#eda100", "-", "v"), "M": ("#4a3aa7", "-", "s"), "A+M": ("#e87ba4", "-", "P"), "H": ("#e34948", "-", "^")}
fig, ax = plt.subplots(figsize=(3.45, 1.95))
phases = nested["phases"]["10"]
order = [p for p in ("stimulation", "next_command", "early_washout", "late") if p in phases]
tick = {"stimulation": "stimulation", "next_command": "next command", "early_washout": "early washout", "late": "late"}
phase_rmse = lambda p, name: arx["phases"]["10"]["pooled"][p][name] if name in lagged else phases[p]["rmse"][name]
for j, name in enumerate(MODEL_STYLE):
    color, _, marker = MODEL_STYLE[name]
    ax.plot([i + (j - 3) * .1 for i in range(len(order))], [phase_rmse(p, name) for p in order],
            ls="none", marker=marker, ms=4, color=color, mew=1, label=SHORT[name])
ax.set_xticks(range(len(order)), [f"{tick[p]}\n({phases[p]['n_conditions']} cond.)" for p in order], fontsize=6)
ax.set(ylabel="10-min RMSE", xlim=(-.5, len(order) - .5))
ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=4, frameon=False, fontsize=6, handletextpad=.1, columnspacing=.8, borderaxespad=.1)
ax.grid(axis="y", alpha=.15)
fig.subplots_adjust(left=.15, right=.98, bottom=.2, top=.97)
fig.savefig(FIG / "prediction_phases.pdf", bbox_inches="tight", pad_inches=.02, metadata={"CreationDate": None})
fig.savefig(FIG / "prediction_phases.png", bbox_inches="tight", pad_inches=.02)
plt.close(fig)

POLICY = {"current": ("^", "current"), "smoothed": ("D", "smoothed"), "history": ("o", "history")}
noise_main, noise_zero = waiting["noise"]["0.005"], waiting["noise"]["0"]
fixed = noise_main["evaluation"]["fixed_calibrated"]
fig, axes = plt.subplots(1, 4, figsize=(7.08, 2.15))
ax = axes[0]
for name, (color, style, marker, label) in SERIES.items():
    ax.plot(centre, [b[name]["readiness"] for b in bins], color=color, ls=style, lw=1.3, marker=marker, ms=3.5, label=label)
ax.set(ylim=(-.03, 1.03), xlabel="Reporter rise (FRET $-$ 1)", ylabel="Readiness")
ax.set_title("(a) Readiness vs reporter", loc="left")
ax.legend(loc="upper right", frameon=False, fontsize=6)
ax = axes[1]
levels = list(stress["noise_grid"])
xs = range(len(levels))
ax.axhline(90, color=MUTED, lw=.6, ls=":", zorder=0)
for name, style in (("current", "--"), ("smoothed", "-.")):
    ax.plot(xs, [100 * stress["noise_grid"][n]["evaluation"][name]["success_rate"] for n in levels], color=INK, ls=style,
            lw=.9, marker=POLICY[name][0], ms=3.5, mfc="white", label=POLICY[name][1])
ax.plot(xs, [100 * stress["noise_grid"][n]["evaluation"]["history"]["success_rate"] for n in levels], color=INK, lw=1.1,
        marker="o", ms=3.5, label="history")
ax.plot(xs, [100 * stress["noise_grid"][n]["history_bandwidth_variant"]["evaluation"]["success_rate"] for n in levels],
        color=MUTED, lw=1.1, marker="o", ms=3.5, mfc="white", label="history, bw 0.005")
ax.set_xticks(list(xs), [f"{float(n):g}".replace("0.", ".") for n in levels], fontsize=6)
ax.set(ylim=(60, 100), xlabel="Observation noise s.d.", ylabel="Evaluation success (%)")
ax.set_title("(b) Success vs noise", loc="left")
ax.legend(loc="lower right", frameon=False, fontsize=5.5, handletextpad=.3, borderaxespad=.1)
ax = axes[2]
curve = waiting["fixed_wait_curve_eval"]
ax.plot([p["mean_completion_min"] for p in curve], [p["success_rate"] for p in curve], color=MUTED, lw=1.0, zorder=1)
ax.plot(fixed["mean_completion_min"], fixed["success_rate"], marker=".", color=MUTED, ms=7, ls="none")
ax.annotate(f"fixed {noise_main['choices']['fixed']['wait_min']:g}", (fixed["mean_completion_min"], fixed["success_rate"]),
            textcoords="offset points", xytext=(5, 4), fontsize=6, color=MUTED)
conditioned = baselines["primary"]["conditioned"]["pooled"]
ax.plot(conditioned["evaluation"]["all"]["mean_completion_min"], conditioned["evaluation"]["all"]["success_rate"], marker="s",
        color=MUTED, ms=4.5, ls="none", zorder=3)
ax.annotate("fixed by history", (conditioned["evaluation"]["all"]["mean_completion_min"], conditioned["evaluation"]["all"]["success_rate"]),
            xytext=(124, .76), fontsize=6, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=.6))
ax.axhline(.9, color=MUTED, lw=.6, ls=":", zorder=0)
marks = dict(POLICY, oracle=("*", "oracle"))
offsets = {"current": (5, -11), "smoothed": (-6, -11), "history": (-2, 7), "oracle": (5, -3)}
for name, (marker, label) in marks.items():
    row = noise_main["evaluation"][name]
    ax.plot(row["mean_completion_min"], row["success_rate"], marker=marker, color=INK, ms=5.5, ls="none", zorder=3)
    ax.annotate(label, (row["mean_completion_min"], row["success_rate"]), textcoords="offset points",
                xytext=offsets[name], fontsize=6, color=INK, ha="right" if offsets[name][0] < 0 else "left")
ax.set(xlim=(100, 143), ylim=(.6, 1.02), xlabel="Mean completion (min)", ylabel="Task success rate")
ax.set_title("(c) Success vs completion", loc="left")
ax = axes[3]
reps = [robust["repeats"][str(r)]["0.005"]["rules"] for r in range(len(robust["repeats"]))]
ax.axhline(0, color=MUTED, lw=.8, ls="--")
offsets = np.linspace(-.18, .18, len(reps))
by_history = baselines["repeats"]["rows"]
columns = [("by hist.", "s", [row["evaluation"]["all"] for row in by_history])]
columns += [(POLICY[n][1], POLICY[n][0], [rep[n]["all"] for rep in reps]) for n in POLICY]
for i, (label, marker, rows_) in enumerate(columns):
    delta = [row["mean_completion_min"] - rep["fixed"]["all"]["mean_completion_min"] for row, rep in zip(rows_, reps)]
    reached = sum(row["success_rate"] >= .9 for row in rows_)
    ax.plot(i + offsets, delta, marker=marker, ms=3, ls="none", mfc="white", mec=MUTED if i == 0 else INK, mew=.7)
    ax.plot([i - .25, i + .25], [np.median(delta)] * 2, color=INK, lw=1.4)
    ax.annotate(f"{reached}/{len(reps)}", (i, max(delta)), textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=5.5, color=INK)
ax.set_xticks(range(len(columns)), [c_[0] for c_ in columns], fontsize=5.5, rotation=25, ha="right")
ax.set(xlim=(-.5, len(columns) - .5), ylabel="Completion $-$ fixed wait (min)")
ax.set_title("(d) 20 splits, noise 0.005", loc="left")
for a in axes:
    a.grid(axis="y", alpha=.15)
fig.subplots_adjust(left=.06, right=.995, bottom=.22, top=.88, wspace=.45)
fig.savefig(FIG / "readiness_waiting.pdf", bbox_inches="tight", pad_inches=.02, metadata={"CreationDate": None})
fig.savefig(FIG / "readiness_waiting.png", bbox_inches="tight", pad_inches=.02)
plt.close(fig)

# Revision plan item 5: per-history reliability constraint, oracle-gap closure (Table III) and frontiers (Fig. 3).
frontier = json.loads((B3_RESULTS / "waiting_frontier_results.json").read_text(encoding="utf-8"))
MAIN_TARGET = "0.94"   # smallest pre-specified target at which every rule met 0.9 for both histories in all 20 splits
assert all(n == 20 for n in frontier["summary"]["0.005"][MAIN_TARGET]["meets_per_history"].values())
ADAPTIVE_RULES = ("current", "smoothed", "history")
RULE_LABEL = {"fixed": "Fixed per history", "current": "Current obs.", "smoothed": "Smoothed", "history": "B3 belief",
              "oracle": "Oracle"}
body = []
for noise in ("0.005", "0.01"):
    cell, summ = frontier["repeats"]["0"][noise]["cells"][MAIN_TARGET], frontier["summary"][noise][MAIN_TARGET]
    splits = [frontier["repeats"][str(r)][noise]["cells"][MAIN_TARGET]["rules"] for r in range(len(frontier["repeats"]))]
    names = ("fixed", "current", "smoothed", "history", "oracle") if noise == "0.005" else ("smoothed", "history")
    if noise != "0.005":
        body += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{Noise s.d.\ {float(noise):g} (same fixed waits)}}"]
    for name in names:
        r = cell["rules"][name]
        label = RULE_LABEL[name]
        if name == "fixed":
            label += " ({:g}/{:g})".format(*(cell["choices"]["fixed"][h]["wait_min"] for h in ("short", "long")))
        if name in ADAPTIVE_RULES:
            dt = float(np.median([row[name]["all"]["mean_completion_min"] - row["fixed"]["all"]["mean_completion_min"]
                                  for row in splits]))
            gap = 100 * summ["gap_closure"][name]["all"]["median"]
            extra = (f"{dt:+.1f} & {gap:.0f}".replace("-", "$-$") + f" & {summ['meets_per_history'][name]}/20")
        elif name == "fixed":
            extra = f"-- & -- & {summ['meets_per_history']['fixed']}/20"
        else:
            extra = "-- & 100 & --"
        body.append(f"{label} & " + " & ".join(f"{100 * r[h]['success_rate']:.1f} & {r[h]['mean_completion_min']:.1f}"
                                               for h in ("short", "long")) + f" & {extra}")
write_table("waiting_per_history", "lccccccc",
            r"Rule & $S_3$ & $T_3$ & $S_{30}$ & $T_{30}$ & $\Delta T$ & $G$ (\%) & Met", body)

# Revision plan items 6, 7, 9: noise-adjusted skill by horizon on Test A (Table II(c)).
extensions = json.loads((NESTED / "nested_extensions_results.json").read_text(encoding="utf-8"))
HORIZONS = ("2", "10", "20", "30")
role_a = lambda h: extensions["horizons"][h]["roles"]["test_new_protocol"]
signed1 = lambda x: f"{x:.1f}".replace("-", "$-$")
body = []
for key in ("O", "M", "H", "M_phase"):
    body.append(f"{SHORT[key]} & " + " & ".join(signed1(100 * role_a(h)["skill"][key]) for h in HORIZONS))
    if key == "M":
        body.append(r"\quad 95\% interval & " + " & ".join(
            "{:.0f}--{:.0f}".format(*(100 * x for x in role_a(h)["bootstrap"]["skill_M"])) for h in HORIZONS))
body.append(r"\midrule")
body.append(r"Crossing BA, Filter / F+B3 & -- & " + " & ".join(
    "{:.0f} / {:.0f}".format(100 * role_a(h)["crossing"]["O"]["balanced_accuracy"], 100 * role_a(h)["crossing"]["M"]["balanced_accuracy"])
    for h in HORIZONS[1:]))
write_table("skill_horizon", "lcccc", r"Horizon $h$ (min) & 2 & 10 & 20 & 30", body)

delay = json.loads((B3_RESULTS / "delay_map_results.json").read_text(encoding="utf-8"))


def non_dominated(points):
    keep = [p for p in points if not any(q["success_rate"] >= p["success_rate"] and q["mean_completion_min"] <= p["mean_completion_min"]
                                         and (q["success_rate"] > p["success_rate"] or q["mean_completion_min"] < p["mean_completion_min"])
                                         for q in points)]
    return sorted(keep, key=lambda p: p["mean_completion_min"])


RULE_STYLE = {"fixed": (MUTED, ":", "s", "fixed wait"), "current": (INK, "--", "^", "current"),
              "smoothed": ("#2a78d6", "-.", "D", "smoothed"), "history": ("#eb6834", "-", "o", "B3 belief")}
f0 = frontier["repeats"]["0"]["0.005"]
operating = f0["cells"][MAIN_TARGET]
fig, axes = plt.subplots(1, 4, figsize=(7.08, 2.25), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.18]))
for ax, (hist, title) in zip(axes[:2], (("short", "(a) After 3-min command"), ("long", "(b) After 30-min command"))):
    block = f0["frontier"][hist]
    for name, (color, style, marker, label) in RULE_STYLE.items():
        points = block[name]["points"]
        if name == "smoothed":
            points = [q for q in points if q["tau_min"] == operating["choices"]["smoothed"][hist]["tau_min"]]
        front = non_dominated(points)
        ax.plot([q["mean_completion_min"] for q in front], [100 * q["success_rate"] for q in front], color=color, ls=style,
                lw=1.1, label=label)
        op = operating["rules"][name][hist]
        ax.plot(op["mean_completion_min"], 100 * op["success_rate"], marker=marker, color=color, ms=4.5, ls="none",
                mfc="white", mew=1.1, zorder=3)
    oracle = block["oracle"]
    ax.plot(oracle["mean_completion_min"], 100 * oracle["success_rate"], marker="*", color=INK, ms=7, ls="none", label="oracle")
    ax.axhline(90, color=MUTED, lw=.6, ls=":", zorder=0)
    ax.set(xlim=(96, 141), ylim=(75, 101), xlabel="Mean completion (min)")
    ax.set_title(title, loc="left", fontsize=8)
axes[0].set_ylabel("Evaluation success (%)")
axes[0].legend(loc="lower right", frameon=False, fontsize=5.5, handlelength=2.0, borderaxespad=.1)
ax = axes[2]
DELAY_STYLE = {"fixed": (MUTED, ":", "s", "fixed wait"), "smoothed_slope": ("#2a78d6", "-.", "D", "smoothed+slope"),
               "b3_nowcast": (INK, "--", "v", "B3 nowcast"), "b3_predictive": ("#eb6834", "-", "o", "B3 predictive")}
ds = delay["grid"]["delays_min"]
for name, (color, style, marker, label) in DELAY_STYLE.items():
    ys = [delay["summary"][f"0.005/2/{d}"]["completion"][name]["all"]["median"] for d in ds]
    ax.plot(ds, ys, color=color, ls=style, marker=marker, ms=3.5, lw=1.1, mfc="white" if name == "fixed" else color, label=label)
ax.plot(ds, [delay["summary"][f"0.005/2/{d}"]["completion"]["oracle"]["all"]["median"] for d in ds], color=INK, lw=.6,
        ls=(0, (1, 2)), label="oracle")
ax.set(xlabel="Delay $d$ (min)", ylabel="Mean completion (min)", xticks=ds)
ax.set_title(r"(c) Delay ($\Delta$=2 min, s.d. .005)", loc="left", fontsize=8)
ax.legend(loc="upper left", frameon=False, fontsize=5.5, handlelength=2.0, borderaxespad=.1)
ax = axes[3]
rows = [(n, i) for n in delay["grid"]["noise_sd"] for i in delay["grid"]["intervals_min"]]
values = np.array([[delay["summary"][f"{n:g}/{i}/{d}"]["map_value_min"] for d in ds] for n, i in rows])
helps = np.array([[delay["summary"][f"{n:g}/{i}/{d}"]["map_class"] == "model helps" for d in ds] for n, i in rows])
limit = max(1., float(np.abs(values).max()))
image = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
for (r_, c_), v in np.ndenumerate(values):
    ax.text(c_, r_, "0.0" if abs(v) < .05 else f"{v:+.1f}".replace("-", "−"), ha="center", va="center", fontsize=5,
            color="white" if abs(v) > .6 * limit else INK, fontweight="bold" if helps[r_, c_] else "normal")
ax.set_xticks(range(len(ds)), [f"{d:g}" for d in ds], fontsize=6)
ax.set_yticks(range(len(rows)), [f"{n:g}".replace("0.", ".") + f", {i}" for n, i in rows], fontsize=5.5)
ax.set(xlabel="Delay $d$ (min)")
ax.set_ylabel(r"Noise s.d., $\Delta$ (min)", fontsize=6.5, labelpad=1)
ax.set_title(r"(d) $\Delta T$, B3 pred. $-$ smooth.+slope", loc="left", fontsize=8)
for a in axes[:3]:
    a.grid(axis="y", alpha=.15)
fig.subplots_adjust(left=.06, right=.995, bottom=.2, top=.9, wspace=.5)
fig.savefig(FIG / "waiting_frontier.pdf", bbox_inches="tight", pad_inches=.02, metadata={"CreationDate": None})
fig.savefig(FIG / "waiting_frontier.png", bbox_inches="tight", pad_inches=.02)
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
