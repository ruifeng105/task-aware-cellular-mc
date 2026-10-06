# Application-first AI in molecular communication: real-data pilot

This directory is included in the merged manuscript package. From the project root, use `python scripts/reproduce.py --mode verify` or `--mode full`; see the root `README.md` and `Paper_Simulation_Map_ZH.md` for the unified workflow and manuscript correspondence.

Date: 2026-10-04. This package contains an executed identification/forecasting pilot on author-exported experimental FGF2/ERK trajectories, source provenance, a targeted novelty audit, and prospective experimental plans. It contains 606 cell trajectories (53,998 measurements; 2-minute sampling), not raw microscope images or a complete Mendeley archive. It does not contain a novel trained controller or new live laboratory experiments.

Read `AI_MC_Application_First_Review_ZH.md` first. `dataset_registry.json` distinguishes accessible records from numerically inspected files. Existing physical-channel calibration in the previous ICC2027-MBMC package remains separate.

## Reproduce the completed FGF2 pilot and the pre-registered expansion

Install `requirements.txt` (Python >= 3.10), then run from this directory:

```bash
python code/calibrate_fgf2.py
python code/verify_fgf2_pipeline.py
python code/expanded_fgf2.py
python code/verify_expanded_fgf2.py
python code/smoke_check.py
```

The expansion (2026-10-04) adds 14 previously unused author conditions from the same pinned commit (22 conditions, 1,565 cells): 0.25 and 25 ng/ml, single 10-min and 60-min pulses. `configs/fgf2_expanded_protocol.json` was frozen before any new response file was downloaded; its SHA-256 and UTC time are in `results/fgf2_expanded/protocol_freeze.json`, and `expanded_fgf2.py` refuses to run if the protocol changes. The 56 added files are listed in `public_data/fgf2_expanded_source_manifest.json` (the pilot manifest is unchanged); `code/fetch_expanded_sources.py` optionally re-downloads and re-verifies them. Roles: train sustained and 3/20 pulses at all concentrations; validate single 5-min pulses; test A single 10-min pulses (new protocol); test B single 60-min pulses (new protocol; timing inferred from the authors' B3 simulations, +/-1 min sensitivity); test C mixed pulses at 0.25/25 ng/ml (new concentrations, recorded as further channels of the inspected mixed experiment). On test A, causal history beat input/response/clock at 10 min (H1, 8.8%) and history plus clock did too (H2, 8.2%), but the gain over persistence was 0.7% (1 of 4 conditions). Leave-one-protocol-out with five training protocols favored history in all six folds.

The source is the paper-linked author repository `Mijan/LFNS_MSB`, pinned commit `5c917abda0618d75c00c9cab45f24ed893dd71f1`. All 47 saved original files verify against their Git blob hashes in `public_data/fgf2_source_manifest.json`. Experimental normalized responses and original-author simulated B3 prediction exports are separate roles. The related Mendeley record is CC BY 4.0; no additional code license was found at the repository root, and this package assigns no new license to author files.

Training uses whole sustained and 3/20 pulse condition blocks; validation uses single 5-minute pulse blocks; test uses mixed blocks. No random cell/window split is used. Independent experimental-day IDs are unavailable, so condition-equal metrics do not imply independent-day generalization. Models know the same future planned commands and use observed responses only up to the prediction time. Command concentration is not measured local concentration or molecule count.

`results/fgf2_pilot/` includes prediction CSVs, input-output curves, exploratory matched-history pairs, numerical results, source/causality checks, a scientific figure, and runtime versions. The measured-response history ridge baseline gives 10-minute test RMSE 0.026817 versus 0.029229 for current input/response/clock; no confidence interval or policy benefit is established. History scales are computational features, not biochemical rate estimates.

The B3 reference reuses the original authors' predictive exports. Those mixed/single-pulse conditions participated in their architecture selection, so this is not a fresh blind B3 test or a same-feedback controller comparison. The current pilot test has been viewed; new model choices should be validated on new independent sessions/days.

The mixed pulse annotations agree with the author XML: `[1,4)`, `[24,54)`, `[114,119)` minutes. The manuscript prose lists durations in a different order; the executable pilot follows the numerical annotations. Mixed normalization uses an original pre-export 10-30 minute baseline, verified from unnormalized exports within text rounding. Other protocols retain author preprocessing.

`novelty_audit.json` records 18 targeted neighbors and explicit body/supplement gaps. `configs/prospective_receiver_reuse.json` is a proposed experiment, with target thresholds, flow, and session IDs deliberately unfilled pending independent calibration. Prediction improvements are not proof of a novel biological mechanism, repeated-command success, or dose savings.

## B3 mechanistic reference (2026-10-05)

`code/b3_model.py` executes the original authors' B3 model by parsing `FGF_Model_B3.txt`, `initial_states_B3.txt` and `measurement_FRET_B3.txt`; the 34-parameter order of their `posterior.txt` (linear scale) comes from their `sim_post_model_summary.txt`. No equation is retyped and no parameter is refit. `code/verify_b3.py` checks sources (`public_data/fgf2_b3_source_manifest.json`), structure (15 species, 16 reactions), conservation laws, the unstimulated steady state and an acceptance gate fixed in advance: our posterior-predictive mean must match the authors' exported mixed-protocol predictions within 1e-3 (observed maximum 8.5e-6; sample-wise errors equal their 1e-4 measurement noise).

`code/b3_forecast.py` (protocol `configs/b3_forecast_protocol.json`, frozen before any B3 forecast was scored) adds a matched-feedback B3 forecaster (M2: posterior samples weighted by each cell's past reporter) and a nested history residual (M3: causal-history ridge plus the M2 increment). It is a post hoc reference analysis on inspected roles; on the 10-min-pulse protocol M3 reduced the 10-min RMSE by 8.2% relative to M2 (0.02098 vs 0.02286; M2 roughly equals persistence).

`code/b3_readiness.py` simulates task readiness and waiting policies on B3 (protocol v2 `configs/b3_readiness_protocol_v2.json`; v1 `configs/b3_readiness_protocol.json` was infeasible for every sample and is kept in `results/b3/readiness_results_v1.json`). With 25 ng/ml commands, the registered candidate waits (6-48 min) never succeeded, readiness depended almost only on the current reporter, and a history-aware policy reached 90.0% success in 113.8 min mean completion versus 121.8 min for the matching fixed wait. This is a simulation of one model structure, not evidence about measured cells. Run `python code/b3_readiness.py --reuse-tables` to recompute decisions from the saved simulation tables. `code/b3_readiness_example.py` writes the single-sample illustration of manuscript Fig. 2 (`results/b3/readiness_example.json`): among the 194 30-min-history samples in which the current-only policy probed early and failed while the history-aware policy succeeded, it takes the one with the median history-aware wait (sample 231: fixed 48 min fails, current-only 84 min fails, history-aware 96 min succeeds). No setting changes; `verify_b3_readiness.py` re-checks the selection and the values.

## Six-page revision analyses (2026-10-06)

`code/nested_forecast.py` (protocol `configs/nested_forecast_protocol.json`, frozen before running; post hoc with respect to inspected roles) compares nested predictors: P persistence; C current input, reporter, slope and known future commands; O adds causal reporter filters; M adds the matched-feedback B3 increment; H adds input filters and cumulative exposure (the feature set of M3). On test A at 10 min the RMSEs are 0.02287, 0.02442, 0.02220, 0.02120 and 0.02098: filtering reduced C by 9.1%, the B3 increment a further 4.5%, input history a further 1.0% (H beat M at only 2 of 4 concentrations). It also labels forecast phases (stimulation, next command, early washout up to 30 min after a pulse, late) and replaces the earlier leave-one-protocol-out analysis, whose penalties had been selected with the held-out protocols in the data, by nested selection: sigma on the training groups, the penalty by inner leave-one-group-out. H beat M in all six corrected folds (0.1-2.0%). The B3 data roles are read from the authors' `config_fgf_sus_3_20.xml` (fit: sus and 3/20 at 2.5/250 ng/ml; architecture likelihood: sp_5 and mixed at 2.5/250). `code/verify_nested_forecast.py` tests the phase rule at switching times, model nesting, that H equals M3, recomputes 90 scores and checks that shifting the held-out targets leaves a fold's selection and predictions unchanged.

`code/b3_waiting.py` implements readiness protocol version 3 (`configs/b3_waiting_protocol.json`): the posterior samples are split once into calibration and evaluation halves; thresholds of the current, smoothed and history-aware readiness estimators and the comparator fixed wait are chosen on calibration samples and evaluated on the others, at observation noise 0 and 0.005. At noise 0.005 the history-aware rule reached 92.6% success in 114.3 min against 121.8 min for the calibrated 100-min fixed wait (91.2%); a 16-min smoothed reporter did as well (92.3%, 113.6 min). Without noise the current reporter sufficed (93.6%) while the history weights collapsed (median effective sample size 1.1). `code/verify_b3_waiting.py` checks equivalence with the version-2 estimators, past-only use, that flipping evaluation labels leaves the calibration unchanged, the fallback rule, and recomputes every evaluation metric. Simulation of one model structure, not evidence about measured cells.

Following reviewer advice (2026-10-06), `code/b3_waiting_robustness.py` (protocol `configs/b3_waiting_robustness_protocol.json`, frozen before its only run) repeats the version-3 study over 20 calibration/evaluation splits with fresh observation noise (repeat 0 reproduces version 3), reports outcomes per history, paired per-sample differences with bootstrap intervals, and an objective-aligned threshold selection as a sensitivity analysis. At noise 0.005 the history rule was faster than the calibrated fixed wait in all 20 splits (6.3-7.8 min), the smoothed reporter faster than the history rule in all 20 (0.2-1.0 min); the fixed wait reached 97.4% success after the 3-min but 85.0% after the 30-min command on the primary split; the aligned selection matched the heuristic in all 40 split-noise cases. `code/verify_b3_waiting_robustness.py` checks repeat 0 against version 3, distinct seeds and splits, the aligned rule, leakage and one recomputed repeat. `code/nested_phase_detail.py` breaks the frozen phase scores down by condition: in the next-command window the B3 increment helped at 0.25 ng/ml but raised the error at 25 ng/ml (0.02920 to 0.03603).

## Generic biological data workflow for additional sources

1. Obtain the raw files through the cited official dataset pages. Preserve the original filenames, README, units, stimulation timestamps, cell exclusions and experiment dates. For yeast, start with Dryad `10.5061/dryad.4f4qrfjn0`; for PC-12 signaling, use Mendeley `10.17632/ccnxn84w8z.2` and the publisher's source-data archives.
2. Inspect workbook structure without guessing its semantics:

   `python code/inspect_workbooks.py /absolute/path/raw_data --output results/workbook_inventory.json`

3. Manually verify and export a long-format CSV with these columns:

   `run_id,cell_id,history_id,time_s,input_concentration,input_unit,response,response_unit,eligible`

   `run_id` must identify the largest relevant independent session/day; cells from a shared chamber/day stay in one split. `history_id` identifies a complete stimulation regimen. Time is in seconds. Keep input units explicit (nM for yeast versus ng/ml in the PC-12 paper). Response is the measured continuous DfU or FRET ratio, **not a fabricated bit label**. `eligible=false` represents terminal exclusion such as leaving G1; an internal out-of-focus observation remains eligible with a missing response. Keep missing observations missing. Do not fill cell-cycle exclusions with zero or infer receptor occupancy from fluorescence without a measurement model.
4. Fill `configs/biological_split_template.json` with actual run IDs before examining test performance. The harness requires all runs to be assigned and splits to be disjoint. Fix any normalization using only the appropriate pre-stimulation measurements or training data.
5. Execute the grouped baselines, choosing `max-gap-s` from the actual acquisition interval and allowable missed-frame rule:

   `python code/history_baselines.py --data /absolute/path/verified_long.csv --split configs/biological_split_template.json --max-gap-s 120 --output results/biological_baselines`

   The example 120 s is **not** a verified yeast frame interval. Replace it after inspection. The EMA scales 60/600/3600 s are computational baseline features, not measured reaction constants. Tune only on validation data if changing them. Since 2026-10-04 the response EMA decays by the time since the last valid (non-missing) response, and the input EMA uses the same zero-order hold of the previous input as the cumulative dose; `code/smoke_check.py` tests both.

The four comparators are persistence, instantaneous input, current input plus current measured response, and causal input/response history. They forecast the next observation during unchanged recorded stimulation. They are deliberately small baselines to assess whether a proposed more complex AI method is justified. Their output cannot evaluate unexecuted control actions or establish biological causal mechanisms. Run-equal MSE/MAE and individual run errors are reported; independent experiment counts must be used for uncertainty, not the number of windows or cells.

## Industrial data workflow

`configs/plif_run_plan.json` records the ten actual run IDs and the observed flow/location combinations from the 2026 descriptor. It specifies a frozen flow-shift test and a separate location-shift test. The light-based concentration reference has 20 Hz native measurements; 1 kHz interpolated labels do not establish 1 kHz reference accuracy. Score at directly measured PLIF timestamps, fit scaling and any lag only on training data, and expose saturation and latency separately.

RedVAPOR uses real experimentally generated gas plumes; “synthetic plume” does not mean a numerical CFD dataset. Sequentially scanned voxel maps cannot be treated as simultaneously measured time-varying concentration fields for arbitrary robot trajectories.

## Functional verification

`python code/smoke_check.py`

This uses **synthetic fixtures only** to check feature causality, grouped splitting, missing-target handling, and baseline execution. It is not a biological training result or a novelty demonstration. The recorded verification is in `results/software_verification.json`.
