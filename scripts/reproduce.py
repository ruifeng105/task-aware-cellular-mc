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
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulation/results/fgf2_pilot"
EXPANDED = ROOT / "simulation/results/fgf2_expanded"
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


def compare_reference():
    expected = json.loads((REPRO / "reference_metrics.json").read_text(encoding="utf-8"))["metrics"]
    actual = paper_metrics(json.loads((OUT / "calibration_results.json").read_text(encoding="utf-8")))
    actual.update(expanded_metrics(json.loads((EXPANDED / "expanded_results.json").read_text(encoding="utf-8"))))
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
    for directory in (OUT, EXPANDED):
        verification = json.loads((directory / "verification_results.json").read_text(encoding="utf-8"))
        if verification["status"] != "passed":
            raise ValueError(f"Verification in {directory.relative_to(ROOT)} did not pass.")
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
