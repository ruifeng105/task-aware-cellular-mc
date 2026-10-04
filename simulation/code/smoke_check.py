"""Synthetic functional checks, explicitly not scientific validation."""
import json
import numpy as np
import pandas as pd
from history_baselines import prepare, validate_split, run_baselines


def fixture():
    rows = []
    for j, run in enumerate(("a", "b", "c", "d")):
        for cell in ("0", "1"):
            for i in range(12):
                rows.append(dict(run_id=run, cell_id=cell, history_id="fixture",
                                 time_s=i*60.0, input_concentration=float(j+1), input_unit="fixture",
                                 response=float(0.1*j+np.sin(i/4)), response_unit="fixture"))
    return pd.DataFrame(rows)


def one_cell(time_s, concentration, response):
    return pd.DataFrame(dict(run_id="r", cell_id="c", history_id="h", time_s=time_s,
                             input_concentration=concentration, input_unit="fixture",
                             response=response, response_unit="fixture"))


def check_response_ema_spans_missing_frame():
    # One out-of-focus frame at 60 s: the 120 s since the last valid
    # response must decay the EMA, not the 60 s since the previous frame.
    rows, _ = prepare(one_cell([0., 60., 120., 180.], 1.0, [0.0, np.nan, 1.0, 1.0]), 600)
    got = rows.set_index("time_s").loc[120., "ema_y_60s"]
    assert np.isclose(got, 1 - np.exp(-2.0)), got


def check_input_ema_uses_dose_hold():
    # Cumulative input holds c[i-1] over [t[i-1], t[i]); the input EMA must too.
    rows, _ = prepare(one_cell([0., 60., 120.], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]), 600)
    row = rows.set_index("time_s").loc[60.]
    assert row.cumulative_input == 0.0, row.cumulative_input
    assert np.isclose(row.ema_c_60s, 0.0), row.ema_c_60s


def main():
    check_response_ema_spans_missing_frame()
    check_input_ema_uses_dose_hold()
    df = fixture()
    base, _ = prepare(df, 90)
    changed = df.copy()
    changed.loc[changed.time_s > 240, "response"] += 100
    changed.loc[changed.time_s > 240, "input_concentration"] += 20
    altered, _ = prepare(changed, 90)
    # At t<240 even the next-input-constancy selection cannot be affected.
    cols = [c for c in base if c != "target"]
    left = base[base.time_s < 240][cols].reset_index(drop=True)
    right = altered[altered.time_s < 240][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    try:
        validate_split(base, {"train":["a","b"], "validation":["b"], "test":["c","d"]})
    except ValueError as error:
        assert "leakage" in str(error)
    else:
        raise AssertionError("Overlapping runs were accepted")
    censored = df.copy()
    censored.loc[(censored.run_id == "a") & (censored.cell_id == "0") & (censored.time_s >= 300), "response"] = np.nan
    cleaned, _ = prepare(censored, 90)
    assert not ((cleaned.run_id == "a") & (cleaned.cell_id == "0") & (cleaned.time_s >= 240)).any()
    results, _ = run_baselines(base, {"train":["a","b"], "validation":["c"], "test":["d"]})
    assert all(np.isfinite(x["test"]["run_equal_mse"]) for x in results.values())
    print(json.dumps({"fixture": "synthetic; no scientific performance claim", "checks_passed":
        ["response EMA decays across missing frames", "input EMA uses the dose hold",
         "future values do not alter past features", "overlapping runs rejected", "missing targets not relabeled", "baseline execution"]}))


if __name__ == "__main__":
    main()
