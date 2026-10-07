"""Verify or reproduce the fixed manuscript pilot, with no new model choices."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulation/results/fgf2_pilot"
EXPANDED = ROOT / "simulation/results/fgf2_expanded"
B3 = ROOT / "simulation/results/b3"
NESTED = ROOT / "simulation/results/nested"
REPRO = ROOT / "reproduction"
CODE = ROOT / "simulation/code"
KIT = ROOT / "validation_kit/scripts"


def paper_metrics(result):
    """Extract all numerical entries underlying the four manuscript tables."""
    values = {
        "source_git_blobs_verified": result["source_git_blobs_verified"],
        "n_cells": result["n_cells"],
        "n_measurements": result["n_measurements"],
    }
    for row in result["inventory"]:
        for key in ("n_cells", "n_times", "n_measurements"):
            values[f"inventory/{row['condition_id']}/{key}"] = row[key]
    for forecast in result["forecasting"]:
        horizon = f"{forecast['horizon_min']:g}"
        for name, model in forecast["models"].items():
            base = f"forecast/{horizon}min/{name}"
            values[base + "/rmse"] = math.sqrt(model["test"]["run_equal_mse"])
            values[base + "/n_transitions"] = model["test"]["n_transitions"]
            values[base + "/validation_rmse"] = math.sqrt(model["validation_condition_equal_mse"])
            for part in ("within", "beyond"):
                support = model["test"]["time_support"][part]
                values[f"{base}/test_{part}_time_support_rmse"] = math.sqrt(support["run_equal_mse"])
                values[f"{base}/test_{part}_time_support_n_transitions"] = support["n_transitions"]
    values["open_loop/positive_single_lag/tau_min"] = result["input_output_identification"]["positive_single_lag"]["tau_min"]
    for token, row in result["washout_description"].items():
        values[f"washout/{token}/mean_y_54min"] = row["mean_y_54min"]
        values[f"washout/{token}/mean_y_66min"] = row["mean_y_66min"]
        values[f"washout/{token}/n_cells_positive_change"] = round(row["fraction_cells_with_positive_change"] * row["n_cells"])
    for condition, models in result["common_window_open_loop_comparison"]["condition_rmse"].items():
        for name, rmse in models.items():
            values[f"open_loop/{condition}/{name}/rmse"] = rmse
    for name, row in result["exploratory_history_matching"]["analyses"].items():
        for key in ("n_matched_cells", "n_eligible_cells", "mean_late_minus_early_target"):
            values[f"history_matching/{name}/{key}"] = row[key]
    return values


def expanded_metrics(result):
    """Every number of the pre-registered expanded analysis cited in the manuscript."""
    values = {"expanded/n_conditions": len(result["inventory"]),
              "expanded/n_cells": sum(row["n_cells"] for row in result["inventory"]),
              "expanded/H1": result["hypotheses"]["H1_history_beats_clock_10min_new_protocol"],
              "expanded/H2": result["hypotheses"]["H2_history_clock_beats_clock_10min_new_protocol"]}
    for horizon, analysis in result["horizons"].items():
        for name, selection in analysis["selection"].items():
            values[f"expanded/{horizon}min/{name}/validation_rmse"] = math.sqrt(selection["validation_condition_equal_mse"])
        for role, models in analysis["roles"].items():
            for name, metric in models.items():
                values[f"expanded/{horizon}min/{role}/{name}/rmse"] = math.sqrt(metric["run_equal_mse"])
                for part in ("within", "beyond"):
                    support = metric["time_support"][part]
                    if support is not None:
                        values[f"expanded/{horizon}min/{role}/{name}/{part}_time_support_rmse"] = math.sqrt(support["run_equal_mse"])
    for row in result["inventory"]:
        values[f"expanded/inventory/{row['condition_id']}/n_cells"] = row["n_cells"]
    for key, row in result["mixed_normalization"].items():
        if row["reconstructable"]:
            values[f"expanded/normalization/{key}/max_abs_error"] = row["max_abs_error"]
    for horizon, by_role in result["hypotheses"]["by_horizon"].items():
        for role, comparison in by_role.items():
            for key in ("conditions_history_beats_clock", "conditions_history_beats_persistence"):
                values[f"expanded/{horizon}min/{role}/{key}"] = comparison[key]
    for group, row in result["leave_one_protocol_out"].items():
        for name, rmse in row.items():
            if name != "conditions":
                values[f"expanded/lopo/{group}/{name}/rmse"] = rmse
    for shift, by_horizon in result["sp60_timing_sensitivity"].items():
        for horizon, models in by_horizon.items():
            for name, rmse in models.items():
                values[f"expanded/sp60_{shift}/{horizon}min/{name}/rmse"] = rmse
    return values


def b3_metrics():
    """Every B3 number cited in the manuscript: acceptance, M0-M3 forecasts, readiness study."""
    load = lambda name: json.loads((B3 / name).read_text(encoding="utf-8"))
    values = {}
    for token, row in load("verification_results.json")["acceptance"]["conditions"].items():
        values[f"b3/acceptance/{token}/max_abs_mean_difference"] = row["max_abs_mean_difference"]
        values[f"b3/acceptance/{token}/median_samplewise_rmse"] = row["median_samplewise_rmse"]
    forecast = load("forecast_results.json")
    for horizon, roles in forecast["scores"].items():
        for role, models in roles.items():
            for name, metric in models.items():
                values[f"b3/forecast/{horizon}min/{role}/{name}/rmse"] = math.sqrt(metric["run_equal_mse"])
    for horizon, roles in forecast["comparisons"]["by_horizon"].items():
        for role, comparison in roles.items():
            for key in ("M3_beats_M2", "M2_beats_M0", "M2_beats_M1"):
                values[f"b3/forecast/{horizon}min/{role}/{key}/conditions"] = comparison[key]["conditions"]
    for key in ("C1_primary_M3_beats_M2", "C2_M2_beats_M0", "C3_M2_beats_M1"):
        values[f"b3/forecast/{key}"] = forecast["comparisons"][key]
    readiness = load("readiness_results.json")
    for policy, row in readiness["policies"].items():
        for key in ("success_rate", "mean_completion_min", "mean_wait_min"):
            values[f"b3/readiness/{policy}/{key}"] = row[key]
    for policy, row in readiness["matched_reliability"].items():
        if row["matching_fixed_wait"] is not None:
            for key in ("wait_min", "success_rate", "mean_completion_min"):
                values[f"b3/readiness/matched/{policy}/{key}"] = row["matching_fixed_wait"][key]
    values["b3/readiness/S1_weighted_difference"] = readiness["matched_reporter_readiness"]["weighted_mean_difference_long_minus_short"]
    for key, value in readiness["version1_check"].items():
        values[f"b3/readiness/version1/{key}"] = value
    example = load("readiness_example.json")
    for key in ("plant", "group_size", "current_later"):
        values[f"b3/readiness/example/{key}"] = example[key]
    for name, branch in example["branches"].items():
        values[f"b3/readiness/example/{name}/wait_min"] = branch["wait_min"]
        values[f"b3/readiness/example/{name}/success"] = branch["success"]
    return values


def nested_metrics():
    """Nested P/C/O/M/H comparison, forecast phases, example cell and corrected LOPO."""
    data = json.loads((NESTED / "nested_results.json").read_text(encoding="utf-8"))
    values = {}
    for h, analysis in data["main"].items():
        for role, comp in analysis["comparisons"].items():
            for name, value in comp["rmse"].items():
                values[f"nested/{h}min/{role}/{name}/rmse"] = value
            for key, row in comp.items():
                if key.endswith(tuple(f"_beats_{b}" for b in ("M", "O", "C", "P"))):
                    values[f"nested/{h}min/{role}/{key}/conditions"] = row["conditions"]
        for phase, row in data["phases"][h].items():
            values[f"nested/{h}min/phase/{phase}/n_conditions"] = row["n_conditions"]
            for name, value in row["rmse"].items():
                values[f"nested/{h}min/phase/{phase}/{name}/rmse"] = value
    for group, fold in data["lopo"].items():
        for name, value in fold["rmse"].items():
            values[f"nested/lopo/{group}/{name}/rmse"] = value
    for key, value in data["decision"].items():
        values[f"nested/decision/{key}"] = value
    values["nested/example/cell"] = data["example"]["cell"]
    values["nested/example/n_cells"] = data["example"]["n_cells"]
    for name, value in data["example"]["cell_rmse"].items():
        values[f"nested/example/cell_rmse/{name}"] = value
    return values


def waiting_metrics():
    """Calibrated waiting study (readiness protocol v3) and its weight-degeneracy diagnostic."""
    data = json.loads((B3 / "waiting_results.json").read_text(encoding="utf-8"))
    values = {}
    for noise, block in data["noise"].items():
        for name, choice in block["choices"].items():
            for key in ("threshold", "tau_min", "wait_min", "target_reached"):
                if key in choice:
                    values[f"b3/waiting/{noise}/choice/{name}/{key}"] = choice[key]
        for policy, row in block["evaluation"].items():
            for key in ("success_rate", "mean_completion_min", "mean_wait_min"):
                values[f"b3/waiting/{noise}/{policy}/{key}"] = row[key]
    diagnostic = json.loads((B3 / "waiting_verification_results.json").read_text(encoding="utf-8"))["weight_degeneracy"]
    for noise, value in diagnostic["median_effective_sample_size"].items():
        values[f"b3/waiting/diagnostic/{noise}/median_effective_sample_size"] = value
    return values


def robustness_metrics():
    """Waiting robustness study: repeat-0 outcomes by history, paired differences and intervals, summary over repeats."""
    data = json.loads((B3 / "waiting_robustness_results.json").read_text(encoding="utf-8"))
    values = {}
    for noise, r0 in data["repeats"]["0"].items():
        for rule, parts in r0["rules"].items():
            for part, row in parts.items():
                for key, value in row.items():
                    values[f"b3/robust/{noise}/r0/{rule}/{part}/{key}"] = value
        for pair, row in r0["paired"].items():
            for key, value in row.items():
                values[f"b3/robust/{noise}/r0/paired/{pair}/{key}"] = value
        for pair, row in r0["bootstrap"].items():
            for key, (low, high) in row.items():
                values[f"b3/robust/{noise}/r0/bootstrap/{pair}/{key}/low"] = low
                values[f"b3/robust/{noise}/r0/bootstrap/{pair}/{key}/high"] = high
    for noise, block in data["summary"].items():
        for rule, parts in block["rules"].items():
            for key, row in parts["all"].items():
                for stat, value in row.items():
                    values[f"b3/robust/{noise}/summary/{rule}/{key}/{stat}"] = value
        for pair, row in block["paired"].items():
            for key, stats_ in row.items():
                for stat, value in stats_.items():
                    values[f"b3/robust/{noise}/summary/paired/{pair}/{key}/{stat}"] = value
        for key in ("history_target_reached", "history_faster_than_fixed"):
            values[f"b3/robust/{noise}/summary/{key}"] = block[key]
    return values


def sensitivity_metrics():
    """Waiting sensitivity to the task threshold and observation noise (primary split)."""
    data = json.loads((B3 / "waiting_sensitivity_results.json").read_text(encoding="utf-8"))
    values = {}
    for kappa, block in data["cells"].items():
        for noise, cell in block.items():
            prefix = f"b3/sensitivity/{kappa}/{noise}"
            values[f"{prefix}/fixed_wait_min"] = cell["choices"]["fixed"]["wait_min"]
            for rule, reached in cell["reached"].items():
                values[f"{prefix}/{rule}/target_reached"] = reached
            for rule, parts in cell["rules"].items():
                for key, value in parts["all"].items():
                    values[f"{prefix}/{rule}/{key}"] = value
            for pair, row in cell["paired"].items():
                for key, value in row.items():
                    values[f"{prefix}/paired/{pair}/{key}"] = value
            for pair, row in cell["bootstrap"].items():
                for key, (low, high) in row.items():
                    values[f"{prefix}/bootstrap/{pair}/{key}/low"] = low
                    values[f"{prefix}/bootstrap/{pair}/{key}/high"] = high
    return values


def revision3_metrics():
    """Draft-3 reviewer items: no-probe control, history-conditioned fixed waits and fine grid, noise/bandwidth
    grid and mismatch, lagged ARX baseline."""
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    values = {}
    noprobe = load(B3 / "noprobe_results.json")
    values["b3/noprobe/fraction_without_probe"] = noprobe["success_without_probe"]["fraction"]
    values["b3/noprobe/max_rise_without_probe_over_naive"] = noprobe["success_without_probe"]["max_rise_without_probe_over_naive"]
    for key, value in noprobe["agreement"].items():
        values[f"b3/noprobe/agreement/{key}"] = value
    for label, rules in noprobe["rescored"].items():
        for rule, row in rules.items():
            for key, value in row.items():
                values[f"b3/noprobe/rescored/{label}/{rule}/{key}"] = value
    baselines = load(B3 / "waiting_baselines_results.json")
    for rule, block in baselines["primary"]["conditioned"].items():
        prefix = f"b3/baselines/primary/{rule}"
        values[f"{prefix}/wait_3"], values[f"{prefix}/wait_30"] = block["waits_min"]
        for part, row in block["evaluation"].items():
            for key, value in row.items():
                values[f"{prefix}/{part}/{key}"] = value
        for pair, row in block["paired"].items():
            for key, value in row.items():
                values[f"{prefix}/paired/{pair}/{key}"] = value
        for pair, row in block["bootstrap"].items():
            for key, (low, high) in row.items():
                values[f"{prefix}/bootstrap/{pair}/{key}/low"] = low
                values[f"{prefix}/bootstrap/{pair}/{key}/high"] = high
    for key, value in baselines["repeats"]["summary"].items():
        if isinstance(value, dict):
            for stat, v in value.items():
                values[f"b3/baselines/repeats/{key}/{stat}"] = v
        else:
            values[f"b3/baselines/repeats/{key}"] = value
    for kappa, block in baselines["kappa"].items():
        for key, value in block["evaluation"]["all"].items():
            values[f"b3/baselines/kappa/{kappa}/{key}"] = value
        for noise, value in block["history_minus_conditioned"].items():
            values[f"b3/baselines/kappa/{kappa}/history_minus_conditioned/{noise}"] = value
    for noise, block in baselines["fine_grid"].items():
        for name, row in block["calibrated"].items():
            values[f"b3/baselines/fine/{noise}/{name}/threshold"] = row["threshold"]
            for key, value in row["evaluation"].items():
                values[f"b3/baselines/fine/{noise}/{name}/{key}"] = value
        for level, row in block["matched_success"].items():
            values[f"b3/baselines/fine/{noise}/matched/{level}/history_minus_smoothed"] = row["history_minus_smoothed"]
    stress = load(B3 / "waiting_stress_results.json")
    for noise, cell in stress["noise_grid"].items():
        for rule in ("current", "smoothed", "history", "fixed_calibrated"):
            for key in ("success_rate", "mean_completion_min"):
                values[f"b3/stress/{noise}/{rule}/{key}"] = cell["evaluation"][rule][key]
        variant = cell["history_bandwidth_variant"]
        values[f"b3/stress/{noise}/history_floor/target_reached"] = variant["target_reached"]
        for key, value in variant["evaluation"].items():
            values[f"b3/stress/{noise}/history_floor/{key}"] = value
        for name, value in cell["median_ess_90min"].items():
            values[f"b3/stress/{noise}/ess/{name}"] = value
    for name, block in stress["mismatch"].items():
        for rule in ("current", "smoothed", "history", "fixed_calibrated", "oracle"):
            for key in ("success_rate", "mean_completion_min"):
                values[f"b3/stress/mismatch/{name}/{rule}/{key}"] = block[rule][key]
    arx = load(NESTED / "nested_arx_results.json")
    for h, analysis in arx["main"].items():
        for role, comp in analysis["comparisons"].items():
            for name, value in comp["rmse"].items():
                values[f"nested/arx/{h}min/{role}/{name}/rmse"] = value
            for key, row in comp.items():
                if "_beats_" in key:
                    values[f"nested/arx/{h}min/{role}/{key}/conditions"] = row["conditions"]
        for name, row in analysis["selection"].items():
            values[f"nested/arx/{h}min/selection/{name}/window"] = row["window"]
    for phase, row in arx["phases"]["10"]["pooled"].items():
        for name, value in row.items():
            values[f"nested/arx/phase/{phase}/{name}/rmse"] = value
    for phase, conditions in arx["phases"]["10"]["per_condition"].items():
        for condition, row in conditions.items():
            for name, value in row["rmse"].items():
                values[f"nested/arx/phase_detail/{phase}/{condition}/{name}/rmse"] = value
    for group, fold in arx["lopo"].items():
        for name, value in fold["rmse"].items():
            values[f"nested/arx/lopo/{group}/{name}/rmse"] = value
    return values


def revision4_metrics():
    """Review of main(1): shared calibration margin and matched success, kappa-dependent no-probe control,
    covariance-aware likelihood under AR(1) noise."""
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    values = {}
    margin = load(B3 / "waiting_margin_results.json")
    for noise, repeat in margin["repeats"]["0"].items():
        for key, cell in repeat["cells"].items():
            prefix = f"b3/margin/r0/{noise}/{key}"
            values[f"{prefix}/conditioned/wait_3"], values[f"{prefix}/conditioned/wait_30"] = cell["choices"]["conditioned"]["waits_min"]
            for rule, parts in cell["rules"].items():
                for metric, value in parts["all"].items():
                    values[f"{prefix}/{rule}/{metric}"] = value
            for pair, row in cell["paired"].items():
                for metric, value in row.items():
                    values[f"{prefix}/paired/{pair}/{metric}"] = value
            for pair, row in cell["bootstrap"].items():
                for metric, (low, high) in row.items():
                    values[f"{prefix}/bootstrap/{pair}/{metric}/low"] = low
                    values[f"{prefix}/bootstrap/{pair}/{metric}/high"] = high
    for noise, cells in margin["summary"].items():
        for key, block in cells.items():
            prefix = f"b3/margin/splits/{noise}/{key}"
            for rule, count in block["reached_on_evaluation"].items():
                values[f"{prefix}/reached/{rule}"] = count
            for pair, row in block["paired"].items():
                for stat, value in row["completion"].items():
                    values[f"{prefix}/paired/{pair}/completion/{stat}"] = value
                values[f"{prefix}/paired/{pair}/success/median"] = row["success"]["median"]
            for pair, count in block["faster"].items():
                values[f"{prefix}/faster/{pair}"] = count
    for noise, levels in margin["frontier_points"].items():
        for level, row in levels.items():
            for name in ("history", "smoothed"):
                for metric in ("threshold", "success_rate", "mean_completion_min"):
                    values[f"b3/margin/frontier/{noise}/{level}/{name}/{metric}"] = row[name][metric]
            values[f"b3/margin/frontier/{noise}/{level}/history_minus_smoothed"] = row["history_minus_smoothed"]
    kappa = load(B3 / "noprobe_kappa_results.json")
    values["b3/noprobe_kappa/max_rise_without_probe_over_naive"] = kappa["max_rise_without_probe_over_naive"]
    for k, block in kappa["kappa"].items():
        states = block["states"]
        prefix = f"b3/noprobe_kappa/{k}"
        values[f"{prefix}/states"] = states["states"]
        values[f"{prefix}/without_probe/count"] = states["success_without_probe"]["count"]
        values[f"{prefix}/without_probe/fraction"] = states["success_without_probe"]["fraction"]
        for history, value in states["success_without_probe"]["per_history"].items():
            values[f"{prefix}/without_probe/{history}"] = value
        for key, value in states["agreement"].items():
            values[f"{prefix}/agreement/{key}"] = value
        for noise, rescored in block["rescored"].items():
            for label in ("original", "causal"):
                for rule, row in rescored[label].items():
                    for metric, value in row.items():
                        values[f"{prefix}/{noise}/{label}/{rule}/{metric}"] = value
            for rule, value in rescored["disagreement_at_stop"].items():
                values[f"{prefix}/{noise}/disagreement/{rule}"] = value
            values[f"{prefix}/{noise}/order_unchanged"] = rescored["order_unchanged"]
    ar1 = load(B3 / "ar1_likelihood_results.json")
    r0 = ar1["repeats"]["0"]
    for key, value in r0["estimated"].items():
        values[f"b3/ar1/r0/estimated/{key}"] = value
    for rule, parts in r0["rules"].items():
        for metric, value in parts["all"].items():
            values[f"b3/ar1/r0/{rule}/{metric}"] = value
    for arm in ("ar1", "ar1_true"):
        values[f"b3/ar1/r0/{arm}/threshold"] = r0["choices"][arm]["threshold"]
        values[f"b3/ar1/r0/{arm}/target_reached"] = r0["choices"][arm]["target_reached"]
    values["b3/ar1/r0/independent_history/target_reached"] = r0["choices"]["independent"]["history"]["target_reached"]
    for pair, row in r0["paired"].items():
        for metric, value in row.items():
            values[f"b3/ar1/r0/paired/{pair}/{metric}"] = value
    for pair, row in r0["bootstrap"].items():
        for metric, (low, high) in row.items():
            values[f"b3/ar1/r0/bootstrap/{pair}/{metric}/low"] = low
            values[f"b3/ar1/r0/bootstrap/{pair}/{metric}/high"] = high
    for name, value in r0["median_ess_90min"].items():
        values[f"b3/ar1/r0/ess/{name}"] = value
    summary = ar1["summary"]
    for rule, count in summary["reached_on_evaluation"].items():
        values[f"b3/ar1/splits/reached/{rule}"] = count
    for block in ("success", "completion", "estimated", "median_ess_90min"):
        for name, stats in summary[block].items():
            for stat, value in stats.items():
                values[f"b3/ar1/splits/{block}/{name}/{stat}"] = value
    for pair, row in summary["paired"].items():
        for metric, stats in row.items():
            for stat, value in stats.items():
                values[f"b3/ar1/splits/paired/{pair}/{metric}/{stat}"] = value
    values["b3/ar1/splits/ar1_more_reliable_than_independent"] = summary["ar1_more_reliable_than_independent"]
    rows = [ar1["repeats"][r] for r in sorted(ar1["repeats"], key=int)]
    gap = [row["rules"]["ar1_history"]["all"]["mean_completion_min"] - row["rules"]["conditioned"]["all"]["mean_completion_min"]
           for row in rows]
    for stat, value in (("median", statistics.median(gap)), ("min", min(gap)), ("max", max(gap))):
        values[f"b3/ar1/splits/ar1_minus_conditioned/completion/{stat}"] = value
    values["b3/ar1/splits/ar1_faster_than_conditioned"] = sum(g < 0 for g in gap)
    return values


def flatten(prefix, obj):
    """Numeric leaves (bool, int, float) of a saved result as prefix/key paths; None and text are skipped."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            out.update(flatten(f"{prefix}/{key}", value))
        return out
    if isinstance(obj, (list, tuple)):
        out = {}
        for i, value in enumerate(obj):
            out.update(flatten(f"{prefix}/{i}", value))
        return out
    if isinstance(obj, bool) or isinstance(obj, (int, float)):
        return {prefix: obj}
    return {}


def revision5_metrics():
    """Revision plan 2026-10-07: per-history calibration, oracle-gap closure and frontiers; delay condition map;
    longer horizons, noise-adjusted skill, crossing score, cell bootstrap and phase-gated increments; natural probes."""
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    values = {}
    frontier = load(B3 / "waiting_frontier_results.json")
    values.update(flatten("b3/frontier/summary", frontier["summary"]))
    values.update(flatten("b3/frontier/version3_gap", frontier["version3_gap_closure"]))
    values.update(flatten("b3/frontier/kappa", frontier["kappa_feasibility"]))
    for noise, row in frontier["repeats"]["0"].items():
        for target, cell in row["cells"].items():
            values.update(flatten(f"b3/frontier/r0/{noise}/{target}", {k: cell[k] for k in
                                  ("choices", "rules", "gap_closure", "paired", "bootstrap", "meets_per_history")}))
        for hist, block in row["frontier"].items():
            for name, front in block.items():
                if name == "oracle":
                    values.update(flatten(f"b3/frontier/curve/{noise}/{hist}/oracle", front))
                else:
                    values.update(flatten(f"b3/frontier/curve/{noise}/{hist}/{name}/levels", front["levels"]))
    delay = load(B3 / "delay_map_results.json")
    for key, block in delay["summary"].items():
        values.update(flatten(f"b3/delay/{key}", {k: v for k, v in block.items() if k != "map_class"}))
        values[f"b3/delay/{key}/model_helps"] = block["map_class"] == "model helps"
    extensions = load(NESTED / "nested_extensions_results.json")
    values.update(flatten("nested/ext/noise_floor", extensions["noise_floor"]))
    values.update(flatten("nested/ext/horizons", extensions["horizons"]))
    values.update(flatten("nested/ext/lopo", extensions["lopo"]))
    probes = load(B3 / "natural_probe_results.json")
    values.update(flatten("b3/natural", {k: probes[k] for k in
                                          ("noise_floor", "cells_complete", "naive_reference", "probes", "pooled_3_20")}))
    return values


def phase_detail_metrics():
    """Per-condition forecast errors by phase from the frozen nested models."""
    data = json.loads((NESTED / "phase_detail.json").read_text(encoding="utf-8"))
    values = {}
    for phase, conditions in data["phases"].items():
        for condition, row in conditions.items():
            values[f"nested/phase_detail/{phase}/{condition}/rows"] = row["rows"]
            values[f"nested/phase_detail/{phase}/{condition}/origins"] = row["origins"]
            for name, value in row["rmse"].items():
                values[f"nested/phase_detail/{phase}/{condition}/{name}/rmse"] = value
    return values


def compare_reference():
    expected = json.loads((REPRO / "reference_metrics.json").read_text(encoding="utf-8"))["metrics"]
    actual = paper_metrics(json.loads((OUT / "calibration_results.json").read_text(encoding="utf-8")))
    actual.update(expanded_metrics(json.loads((EXPANDED / "expanded_results.json").read_text(encoding="utf-8"))))
    actual.update(b3_metrics())
    actual.update(nested_metrics())
    actual.update(waiting_metrics())
    actual.update(robustness_metrics())
    actual.update(phase_detail_metrics())
    actual.update(sensitivity_metrics())
    actual.update(revision3_metrics())
    actual.update(revision4_metrics())
    actual.update(revision5_metrics())
    if actual.keys() != expected.keys():
        raise ValueError("Manuscript metric keys differ from the frozen reference.")
    max_error = 0.0
    for name, value in actual.items():
        reference = expected[name]
        max_error = max(max_error, abs(value - reference))
        if isinstance(reference, int):
            agrees = value == reference
        else:
            agrees = math.isclose(value, reference, rel_tol=1e-8, abs_tol=1e-10)
        if not agrees:
            raise ValueError(f"Reference mismatch: {name}: {value} versus {reference}")
    for path in (OUT / "verification_results.json", EXPANDED / "verification_results.json",
                 B3 / "verification_results.json", B3 / "forecast_verification_results.json",
                 B3 / "readiness_verification_results.json", B3 / "waiting_verification_results.json",
                 B3 / "waiting_robustness_verification_results.json", B3 / "waiting_sensitivity_verification_results.json",
                 B3 / "noprobe_verification_results.json", B3 / "waiting_baselines_verification_results.json",
                 B3 / "waiting_stress_verification_results.json", NESTED / "nested_arx_verification_results.json",
                 B3 / "waiting_margin_verification_results.json", B3 / "noprobe_kappa_verification_results.json",
                 B3 / "ar1_likelihood_verification_results.json", NESTED / "nested_verification_results.json",
                 B3 / "waiting_frontier_verification_results.json", B3 / "delay_map_verification_results.json",
                 NESTED / "nested_extensions_verification_results.json", B3 / "natural_probe_verification_results.json"):
        if json.loads(path.read_text(encoding="utf-8"))["status"] != "passed":
            raise ValueError(f"Verification {path.relative_to(ROOT)} did not pass.")
    return {"status": "passed", "entries_compared": len(actual),
            "max_absolute_error": max_error,
            "relative_tolerance": 1e-8, "absolute_tolerance": 1e-10}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "full"), default="verify")
    parser.add_argument("--compile-paper", action="store_true",
                        help="Also compile the manuscript; requires a LaTeX installation.")
    args = parser.parse_args()
    logs = REPRO / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "mode": args.mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(),
                        **{name: metadata.version(name) for name in
                           ("numpy", "pandas", "matplotlib", "openpyxl")}},
        "steps": [], "controller_evaluated": False,
        "b3_original_simulator_rerun": False,
        "scope": "Fixed pilot reproduction; B3 original-author exports are reused.",
    }

    def run_step(name, command, cwd=ROOT):
        log = logs / f"{name}.log"
        print(f"Running {name} ...", flush=True)
        start = time.monotonic()
        completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, check=False)
        # LF logs on every platform so packaged hashes do not depend on the OS.
        log.write_bytes(completed.stdout.replace(b"\r\n", b"\n"))
        report["steps"].append({"name": name, "return_code": completed.returncode,
                                "elapsed_seconds": round(time.monotonic() - start, 3),
                                "log": str(log.relative_to(ROOT))})
        if completed.returncode:
            raise RuntimeError(f"{name} failed; see {log.relative_to(ROOT)}")
        print(f"Passed {name}", flush=True)

    try:
        if args.mode == "full":
            run_step("fit_and_predict", [sys.executable, str(CODE / "calibrate_fgf2.py")])
            run_step("expanded_fresh_test", [sys.executable, str(CODE / "expanded_fgf2.py")])
        run_step("verify_data_and_scores", [sys.executable, str(CODE / "verify_fgf2_pipeline.py")])
        run_step("verify_expanded", [sys.executable, str(CODE / "verify_expanded_fgf2.py")])
        run_step("verify_b3_model", [sys.executable, str(CODE / "verify_b3.py")])
        if args.mode == "full":
            run_step("b3_forecast", [sys.executable, str(CODE / "b3_forecast.py")])
            run_step("b3_readiness", [sys.executable, str(CODE / "b3_readiness.py")])
            run_step("b3_readiness_example", [sys.executable, str(CODE / "b3_readiness_example.py")])
            run_step("nested_forecast", [sys.executable, str(CODE / "nested_forecast.py")])
            run_step("nested_phase_detail", [sys.executable, str(CODE / "nested_phase_detail.py")])
            run_step("nested_arx", [sys.executable, str(CODE / "nested_arx.py")])
            run_step("b3_waiting", [sys.executable, str(CODE / "b3_waiting.py")])
            run_step("b3_waiting_robustness", [sys.executable, str(CODE / "b3_waiting_robustness.py")])
            run_step("b3_waiting_sensitivity", [sys.executable, str(CODE / "b3_waiting_sensitivity.py")])
            run_step("b3_noprobe", [sys.executable, str(CODE / "b3_noprobe.py")])
            run_step("b3_waiting_baselines", [sys.executable, str(CODE / "b3_waiting_baselines.py")])
            run_step("b3_waiting_stress", [sys.executable, str(CODE / "b3_waiting_stress.py")])
            run_step("b3_waiting_margin", [sys.executable, str(CODE / "b3_waiting_margin.py")])
            run_step("b3_noprobe_kappa", [sys.executable, str(CODE / "b3_noprobe_kappa.py")])
            run_step("b3_ar1_likelihood", [sys.executable, str(CODE / "b3_ar1_likelihood.py")])
            run_step("b3_waiting_frontier", [sys.executable, str(CODE / "b3_waiting_frontier.py")])
            run_step("b3_delay_map", [sys.executable, str(CODE / "b3_delay_map.py")])
            run_step("nested_extensions", [sys.executable, str(CODE / "nested_extensions.py")])
            run_step("natural_probe", [sys.executable, str(CODE / "natural_probe.py")])
        run_step("verify_b3_forecast", [sys.executable, str(CODE / "verify_b3_forecast.py")])
        run_step("verify_b3_readiness", [sys.executable, str(CODE / "verify_b3_readiness.py")])
        run_step("verify_nested_forecast", [sys.executable, str(CODE / "verify_nested_forecast.py")])
        run_step("verify_b3_waiting", [sys.executable, str(CODE / "verify_b3_waiting.py")])
        run_step("verify_b3_waiting_robustness", [sys.executable, str(CODE / "verify_b3_waiting_robustness.py")])
        run_step("verify_b3_waiting_sensitivity", [sys.executable, str(CODE / "verify_b3_waiting_sensitivity.py")])
        run_step("verify_b3_noprobe", [sys.executable, str(CODE / "verify_b3_noprobe.py")])
        run_step("verify_b3_waiting_baselines", [sys.executable, str(CODE / "verify_b3_waiting_baselines.py")])
        run_step("verify_b3_waiting_stress", [sys.executable, str(CODE / "verify_b3_waiting_stress.py")])
        run_step("verify_nested_arx", [sys.executable, str(CODE / "verify_nested_arx.py")])
        run_step("verify_b3_waiting_margin", [sys.executable, str(CODE / "verify_b3_waiting_margin.py")])
        run_step("verify_b3_noprobe_kappa", [sys.executable, str(CODE / "verify_b3_noprobe_kappa.py")])
        run_step("verify_b3_ar1_likelihood", [sys.executable, str(CODE / "verify_b3_ar1_likelihood.py")])
        run_step("verify_b3_waiting_frontier", [sys.executable, str(CODE / "verify_b3_waiting_frontier.py")])
        run_step("verify_b3_delay_map", [sys.executable, str(CODE / "verify_b3_delay_map.py")])
        run_step("verify_nested_extensions", [sys.executable, str(CODE / "verify_nested_extensions.py")])
        run_step("verify_natural_probe", [sys.executable, str(CODE / "verify_natural_probe.py")])
        run_step("synthetic_checks", [sys.executable, str(CODE / "smoke_check.py")])
        report["manuscript_reference"] = compare_reference()
        print(f"Passed frozen manuscript comparison ({report['manuscript_reference']['entries_compared']} entries)", flush=True)
        if args.mode == "full":
            run_step("build_paper_assets", [sys.executable, str(ROOT / "scripts/build_paper_assets.py")])
            # Prospective-validation planning kit (needs SciPy); it never creates biological records.
            for name, script, options in (("kit_pilot_design", "design_candidates.py", ["--stage", "pilot"]),
                                          ("kit_main_design", "design_candidates.py", ["--stage", "main"]),
                                          ("kit_sample_size", "sample_size.py", []),
                                          ("kit_registration_audit", "audit_registration.py", []),
                                          ("kit_data_contract_audit", "audit_data_contract.py", []),
                                          ("kit_software_checks", "verify_planning_tools.py", [])):
                run_step(name, [sys.executable, str(KIT / script), *options], KIT)
        run_step("pipeline_checks", [sys.executable, str(CODE / "pipeline_checks.py")])
        if args.compile_paper:
            options = ["-interaction=nonstopmode", "-halt-on-error",
                       "-jobname=AI_MC_Cellular_Receivers_Draft", "main.tex"]
            if shutil.which("latexmk"):
                run_step("compile_paper", ["latexmk", "-pdf", *options], ROOT / "paper")
            elif shutil.which("pdflatex") and shutil.which("bibtex"):
                run_step("latex_pass_1", ["pdflatex", *options], ROOT / "paper")
                run_step("bibliography", ["bibtex", "AI_MC_Cellular_Receivers_Draft"], ROOT / "paper")
                run_step("latex_pass_2", ["pdflatex", *options], ROOT / "paper")
                run_step("latex_pass_3", ["pdflatex", *options], ROOT / "paper")
            else:
                raise RuntimeError("Install latexmk, or pdflatex and bibtex, to compile the paper.")
        report["calibration_result_sha256"] = hashlib.sha256((OUT / "calibration_results.json").read_bytes()).hexdigest()
        report["expanded_result_sha256"] = hashlib.sha256((EXPANDED / "expanded_results.json").read_bytes()).hexdigest()
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        print(f"Failed: {error}", file=sys.stderr)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (REPRO / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    if report["status"] != "passed":
        return 1
    print("Complete. Report: reproduction/run_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
