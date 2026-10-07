# Supplementary implementation details

Companion to *Task-Aware Reuse of Cellular Molecular Receivers: Response Prediction and Feedback-Driven Waiting*. These details were moved out of the six-page manuscript (revision plan item 3, 2026-10-07). Every statement refers to code and frozen protocols in this repository; numbers are those of the frozen results.

## S1. Data, normalization and roles

- All numerical files come from the paper-linked repository `Mijan/LFNS_MSB` at commit `5c917ab` and the Mendeley record `10.17632/ccnxn84w8z.2`; commands follow the authors' model XML (`calibrate_fgf2.segments`).
- Model time 0 is the stimulation onset. Per-cell normalization uses only earlier frames: a per-cell mean over raw 2–38 min in the exports, and, for the mixed protocol whose origin is raw 42 min, a per-cell median over raw 10–30 min that we reconstructed (`pipeline_checks.check_normalization_causal`). Every forecast therefore uses only information available at its origin.
- Roles were frozen before any newly used response file was downloaded (`simulation/configs/fgf2_expanded_protocol.json`). Test A (single 10-min pulse) is a new protocol; Test B's [0, 60) min timing was inferred from the authors' B3 simulations (±1 min sensitivity in `expanded_results.json`); Test C uses new concentrations (0.25, 25 ng/ml) recorded in the session of the viewed mixed blocks (2.5, 250 ng/ml) on which an earlier pilot was tested.
- No experimental-day identifiers exist; protocol generalization is therefore not cross-day generalization.

## S2. Predictors

| Name in paper | Code key | Features (all receive the known future command: exposure in 2-min bins, concentration at t+h, h) |
|---|---|---|
| Persist | `P` | none; ŷ(t+h) = y(t) |
| Current | `C` | u, u², y, y², u·y, 2-min slope (equals the pre-registered `local_slope`) |
| Filter | `O` | Current + causal reporter filters with scales ℓ ∈ {2, 10, 60} min |
| Lag | `A` | Current + reporter and command values of the previous L ∈ {5, 15, 30} frames |
| Filter+B3 | `M` | Filter + matched-feedback B3 increment δ̂ |
| Lag+B3 | `A+M` | Lag + δ̂ |
| Filter+B3+Input | `H` | Filter+B3 + causal input filters (ℓ ∈ {2, 10, 60} min) + cumulative exposure (feature set of the earlier M3) |
| Filter+B3 (phase) | `M_phase` | Filter+B3 + stimulation and next-command indicators and their products with δ̂, y and the slope |

- Lag filling: before the first frame the reporter lag repeats that frame and the command lag is the known command (zero before onset). This affects origins before 30 min when L = 15 is selected (2-min Lag, LOPO) but not the L = 5 of the main 10-min analysis (`nested_arx.py`).
- Fitting: ridge regression with equal weight per condition and standardization from training rows; penalty from 10⁻⁸ to 10⁴, the B3 likelihood scale σ from seven values and L are selected on the validation role (single 5-min pulse). In the corrected LOPO, σ is selected on the training groups and the penalty and L by inner leave-one-group-out (`nested_forecast.lopo_fold`).
- B3 increment: the authors' model files are parsed and executed (`b3_model.py`); no equation is retyped and no parameter refitted; the posterior-predictive mean matches the authors' exports within 8.5 × 10⁻⁶. The 1,000 posterior samples are weighted by each cell's past reporter (Eq. 4 of the paper), a fixed-parameter particle approximation of the B3 belief that uses the same information as the learned predictors.
- Forecast phases take the first match in the order stimulation (command on at t), next command (a pulse starts in (t, t+h]), early washout (last pulse ended ≤ 30 min before t) and late. After its warmup Test A contains only washout rows.

## S3. Waiting rules

- Estimators: *current* — Gaussian kernel over the pooled (reporter, success) states of reference receivers; *smoothed* — the same kernel on a causal exponential smoothing (scale 2–16 min) with the smoothed-noise s.d.; *history* (B3 belief) — reference receivers under the known command history weighted by all observations, likelihood s.d. max(noise, 0.001). The floor regularizes the estimator; it is not an assumed noise level. A floor of 0.005 is a pre-specified variant (`b3_waiting_stress.py`).
- References are calibration samples; on calibration samples each receiver is left out of its own reference set; evaluation samples never enter references, thresholds or waits.
- Version 3 (pooled constraint): first threshold on {0.5, …, 0.99} whose calibration success reaches q; for smoothing, the scale with the smallest calibration completion among feasible scales; fallback: largest threshold, flagged. Conditioned fixed waits: (τ₃, τ₃₀) jointly over the 61² grid pairs.
- Revision (per-history constraint, `b3_waiting_frontier.py`): one parameter per previous command minimizing that history's calibration completion subject to its calibration success ≥ target; thresholds 0.50–0.99 in steps of 0.01.
- Paired 95% bootstrap intervals resample evaluation posterior samples (2,000 resamples), keeping both histories and all rules paired.
- Causal success criterion (`b3_noprobe_kappa.py`): the probe-induced increment Δz(s; τ) = z^τ(s) − z^∅(s) replaces the rise; it agreed with the main criterion for 97.0–97.2% of 122,000 states and changed the rules' order only for two pairs within 0.25 min.
- AR(1) likelihood (`b3_ar1_likelihood.py`): covariance σ̂²ρ̂^|a−b|, i.e. whitened residuals (eᵢ − ρ̂eᵢ₋₁)/√(1−ρ̂²) of eᵢ = yᵢ − f_s(tᵢ), with ρ̂ = 0.80 from calibration residuals.

## S4. Delay condition map (`b3_delay_map.py`)

- A probe decided at observation instant t acts at t + d; observations every Δ ∈ {2, 6, 10} min; noise s.d. ∈ {0.0025, 0.005, 0.01}; d ∈ {0, 10, 20, 30, 40} min.
- B3-predictive: B3 belief weights (as the history estimator) applied to the references' success at t + d. B3-nowcast: the same weights applied to success at t.
- Smoothed + slope: exponential smoothing with a = exp(−Δ/ℓ), ℓ ∈ {2, 4, 8, 16} min, level mᵢ and increment gᵢ = mᵢ − mᵢ₋₁; a Gaussian kernel with the stationary covariance of smoothed white noise, v[[1, 1−a], [1−a, 2(1−a)]] + 0.001²I with v = σ²(1−a)/(1+a), over all reference states (both histories, all instants), evaluated by linear binning at 0.2 kernel s.d. in whitened coordinates (max. probability error < 0.01 against direct evaluation, `verify_b3_delay_map.py`).

## S5. Forecast extensions (`nested_extensions.py`)

- White-noise floor per condition: σ_ν = median|y(i+1) − 2y(i) + y(i−1)| / (0.6745√6) over all cells and frames.
- Noise-adjusted skill: 1 − mean_c(MSE − σ_ν²)/mean_c(MSE_Persist − σ_ν²).
- Half-decay crossing score: in washout rows whose excursion is still above half the running peak since the last pulse onset, balanced accuracy of the forecast's prediction that y(t+h) − 1 ≤ ½ peak.
- Cell bootstrap: cells resampled within conditions (2,000 resamples); fitted models and σ_ν fixed.

## S6. Natural probes (`natural_probe.py`)

- 25 ng/ml only. 3/20 protocol: pulses 2–6 are 3-min probes 20 min after a 3-min pulse; the naive reference is the same cell's rise to pulse 1. Mixed protocol: the 5-min pulse at 114 min is a probe 60 min after the 30-min pulse; the naive reference is the median rise of single 5-min pulse cells.
- Rise: maximum of the centred 3-frame moving average in (onset, onset + 20] minus the mean of the last two frames at or before onset; the same operator is applied to B3 trajectories at the measured frame times.
