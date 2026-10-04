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
