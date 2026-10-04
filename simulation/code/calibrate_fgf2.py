"""Reproducible pilot identification on author-exported experimental ERK data.

Prediction and input-output identification only. No live controller, new
biological experiment, inferred biochemical rate constants, or policy replay.
All partitions are whole condition blocks; experimental-day IDs are unavailable.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from history_baselines import fit_ridge, predict, validate_split, run_metric

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'public_data/fgf2_author_repository'
DATA = SOURCE / 'FGF2_models/data'
OUT = ROOT / 'results/fgf2_pilot'
CONCS = {'2-5ng': 2.5, '250ng': 250.0}
PROTOCOLS = ('fgf_sus', 'fgf_3_20', 'fgf_sp_5', 'fgf_mixed')
SPLIT = {'train': ['fgf_sus_2-5ng', 'fgf_sus_250ng', 'fgf_3_20_2-5ng', 'fgf_3_20_250ng'],
         'validation': ['fgf_sp_5_2-5ng', 'fgf_sp_5_250ng'],
         'test': ['fgf_mixed_2-5ng', 'fgf_mixed_250ng']}
ALPHAS = (1e-6, 1e-4, 1e-2, 1., 100.)
HISTORY_TAUS_MIN = (2., 10., 60.)
FILTER_TAUS_MIN = (1., 2., 4., 8., 16., 32., 64., 128.)


def load_matrix(path):
    text = path.read_text()
    return np.loadtxt(path, delimiter=',' if ',' in text.splitlines()[0] else None, ndmin=2)


def segments(protocol, until=1000.):
    """Command schedule from the original B3 XML, relative to its model origin.

    These are commanded concentrations, not measured local concentrations.
    Half-open pulse intervals fix the convention at exact switching times.
    Any history before the exported origin remains unknown.
    """
    if protocol == 'fgf_sus':
        return [(0., until)]
    if protocol == 'fgf_3_20':
        return [(float(t), float(t + 3)) for t in range(0, int(until) + 1, 23)]
    if protocol == 'fgf_sp_5':
        return [(0., 5.)]
    if protocol == 'fgf_mixed':
        return [(1., 4.), (24., 54.), (114., 119.)]
    raise ValueError(protocol)


def command(t, protocol, concentration):
    t = np.asarray(t, dtype=float)
    return concentration * sum(((t >= a) & (t < b)).astype(float) for a, b in segments(protocol))


def dose(a, b, protocol, concentration):
    return concentration * sum(max(0., min(b, end) - max(a, start))
                               for start, end in segments(protocol))


def input_filter(t, protocol, tau):
    """Exact stable first-order response to a binary commanded pulse schedule."""
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)
    for a, b in segments(protocol):
        dt_on = np.maximum(t - a, 0.)
        dt_off = np.maximum(t - b, 0.)
        result += (1 - np.exp(-dt_on / tau)) - (1 - np.exp(-dt_off / tau))
    return result


def read_blocks():
    blocks = {}
    inventory = []
    for p in PROTOCOLS:
        t = load_matrix(DATA / p / 'time_trunc.txt').ravel()
        assert np.all(np.diff(t) == 2.), p
        for token, conc in CONCS.items():
            key = p + '_' + token
            y = load_matrix(DATA / p / (token + '_trunc.txt'))
            m = load_matrix(DATA / p / (token + '_mean_trunc.txt')).ravel()
            assert y.shape[1] == len(t) == len(m), key
            assert np.isfinite(y).all(), key
            blocks[key] = dict(protocol=p, token=token, concentration=conc, time=t, y=y)
            inventory.append(dict(condition_id=key, n_cells=len(y), n_times=len(t),
                                  n_measurements=int(y.size), time_min=[float(t[0]), float(t[-1])],
                                  interval_min=2., finite=True,
                                  original_mean_max_difference=float(abs(y.mean(0) - m).max()),
                                  original_mean_rms_difference=float(np.sqrt(np.mean((y.mean(0)-m)**2)))))
    return blocks, inventory


def verify_sources():
    manifest = json.loads((ROOT / 'public_data/fgf2_source_manifest.json').read_text())
    verified = 0
    for item in manifest:
        if 'path' not in item:
            continue
        raw = (SOURCE / item['path']).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        assert blob == item['blob_sha'], item['path']
        verified += 1
    return verified


def forecasting_rows(blocks, horizon_min):
    # Future inputs are a declared candidate schedule and are available equally
    # to every predictor. Future responses are targets only.
    lag = int(horizon_min / 2)
    rows = []
    for key, block in blocks.items():
        t, values, conc, protocol = block['time'], block['y'], block['concentration'], block['protocol']
        c = command(t, protocol, conc)
        history_c = np.array([conc * input_filter(t, protocol, tau) for tau in HISTORY_TAUS_MIN]).T
        for cell, y in enumerate(values):
            ema_y = np.repeat(y[0], len(HISTORY_TAUS_MIN))
            for i, now in enumerate(t):
                if i:
                    decay = np.exp(-(now - t[i-1]) / np.array(HISTORY_TAUS_MIN))
                    ema_y = decay * ema_y + (1-decay) * y[i]
                # Fixed 10-minute observed warmup, independent of target values.
                if i < 1 or now - t[0] < 10 or i + lag >= len(t):
                    continue
                row = dict(run_id=key, cell_id=str(cell), history_id=protocol,
                           time_s=now*60, time_min=now, time2=now**2,
                           horizon_s=horizon_min*60, c=c[i], c2=c[i]**2,
                           current_y=y[i], y2=y[i]**2, cy=c[i]*y[i],
                           slope=(y[i]-y[i-1])/2., target=y[i+lag],
                           past_dose=dose(0., now, protocol, conc),
                           end_c=float(command(now+horizon_min, protocol, conc)))
                for k in range(lag):
                    row[f'future_dose_{k}'] = dose(now+2*k, now+2*(k+1), protocol, conc)
                for j, tau in enumerate(HISTORY_TAUS_MIN):
                    row[f'input_filter_{tau:g}'] = history_c[i,j]
                    row[f'response_ema_{tau:g}'] = ema_y[j]
                rows.append(row)
    return pd.DataFrame(rows)


def forecasting(blocks, horizon):
    rows = forecasting_rows(blocks, horizon)
    parts = validate_split(rows, SPLIT)
    future = [f'future_dose_{k}' for k in range(int(horizon/2))] + ['end_c', 'horizon_s']
    current = ['c', 'c2', 'current_y', 'y2', 'cy'] + future
    models = {
        'current_input': ['c', 'c2'] + future,
        'current_response': current,
        'current_response_clock': current + ['time_min', 'time2'],
        'local_slope': current + ['slope'],
        'causal_history': current + ['slope', 'past_dose']
          + [f'{prefix}_{tau:g}' for prefix in ('input_filter', 'response_ema') for tau in HISTORY_TAUS_MIN],
    }
    predictions = parts['test'][['run_id','cell_id','time_s','target','current_y']].copy()
    results = {'persistence': {'test': run_metric(parts['test'], parts['test'].current_y.to_numpy())}}
    predictions['persistence'] = parts['test'].current_y.to_numpy()
    for name, features in models.items():
        candidates = []
        for alpha in ALPHAS:
            model = fit_ridge(parts['train'], features, alpha)
            score = run_metric(parts['validation'], predict(parts['validation'], model))['run_equal_mse']
            candidates.append((score, model))
        score, model = min(candidates, key=lambda x:x[0])
        yp = predict(parts['test'], model)
        results[name] = dict(model=model, validation_condition_equal_mse=score,
                            test=run_metric(parts['test'], yp))
        predictions[name] = yp
    predictions.to_csv(OUT / f'forecasts_{horizon:g}min.csv', index=False)
    return dict(horizon_min=horizon, split=SPLIT, models=results,
                information='All predictors know the future commanded waveform. Only observed responses up to t are features.',
                score_note='run_id denotes a condition block here, NOT a verified independent experiment.')


def mean_rows(blocks, taus):
    rows = []
    for key, block in blocks.items():
        t = block['time']
        for j, now in enumerate(t):
            row = dict(run_id=key, target=float(block['y'].mean(0)[j]-1.), time_min=now)
            for tau in taus:
                f = float(input_filter(now, block['protocol'], tau))
                # Dose-specific gains do not imply concentration extrapolation.
                for token in CONCS:
                    row[f'f_{token}_{tau:g}'] = f if token == block['token'] else 0.
            rows.append(row)
    return pd.DataFrame(rows)


def input_output_identification(blocks):
    rows = mean_rows(blocks, FILTER_TAUS_MIN)
    parts = validate_split(rows, SPLIT)
    cols = [c for c in rows if c.startswith('f_')]
    candidates = []
    for alpha in ALPHAS:
        model = fit_ridge(parts['train'], cols, alpha)
        loss = run_metric(parts['validation'], predict(parts['validation'], model))['run_equal_mse']
        candidates.append((loss,model))
    val, model = min(candidates, key=lambda x:x[0])
    result = {'stable_filter_bank': dict(model=model, validation_condition_equal_mse=val,
                                         test=run_metric(parts['test'], predict(parts['test'],model)))}
    curves = parts['test'][['run_id','time_min','target']].copy()
    curves['measured_mean'] = curves.pop('target')+1
    curves['stable_filter_bank'] = predict(parts['test'],model)+1
    # A nonnegative single-lag system cannot create a secondary rise after washout.
    train_keys = SPLIT['train']
    one_candidates = []
    for tau in np.geomspace(.2,200.,100):
        gains = {}
        for token in CONCS:
            numerator=denominator=0.
            for key in train_keys:
                b=blocks[key]
                if b['token']!=token:
                    continue
                f=input_filter(b['time'],b['protocol'],tau)
                y=b['y'].mean(0)-1
                numerator += float(np.mean(f*y)); denominator += float(np.mean(f*f))
            gains[token] = max(0.,numerator/denominator)
        errors=[]
        for key in SPLIT['validation']:
            b=blocks[key]
            yp=1+gains[b['token']]*input_filter(b['time'],b['protocol'],tau)
            errors.append(float(np.mean((yp-b['y'].mean(0))**2)))
        one_candidates.append((float(np.mean(errors)),float(tau),gains))
    val1,tau1,gains1=min(one_candidates,key=lambda x:x[0])
    one_preds=[]
    for key in SPLIT['test']:
        b=blocks[key]
        one_preds.extend(gains1[b['token']]*input_filter(b['time'],b['protocol'],tau1))
    result['positive_single_lag']=dict(tau_min=tau1,gains=gains1,validation_condition_equal_mse=val1,
                                      test=run_metric(parts['test'],np.array(one_preds)))
    curves['positive_single_lag']=np.array(one_preds)+1
    curves.to_csv(OUT/'input_output_test_curves.csv',index=False)
    return result,curves


def published_reference(blocks):
    prefix=SOURCE/'Inference_results/Fgf_B3/results/sus_3_20'
    t=load_matrix(prefix/'sim_post_times.txt').ravel()
    records={}; curves={}
    for token in CONCS:
        samples=load_matrix(prefix/f'sim_post_mixed_{token}_measurements.txt')
        assert samples.shape[1]==len(t)
        b=blocks['fgf_mixed_'+token]
        indices=np.searchsorted(b['time'],t)
        assert np.array_equal(b['time'][indices],t)
        measured=b['y'].mean(0)[indices]
        ym=samples.mean(0)
        records[token]=dict(n_author_exported_predictive_samples=len(samples),time_min=[float(t[0]),float(t[-1])],
                            mean_curve_rmse=float(np.sqrt(np.mean((ym-measured)**2))),
                            note='Original authors B3 predictive export reused as context; not refitted here, not a same-feedback comparison. The source study used these mixed/single-pulse conditions in architecture selection, so this reference is not a fresh blind test.')
        curves[token]=(t,ym)
    return records,curves


def washout_description(blocks):
    out={}
    for token in CONCS:
        b=blocks['fgf_mixed_'+token]
        ia=int(np.where(b['time']==54.)[0][0]); ib=int(np.where(b['time']==66.)[0][0])
        d=b['y'][:,ib]-b['y'][:,ia]
        out[token]=dict(concentration_command_at_both_times_ng_ml=0.,mean_y_54min=float(b['y'][:,ia].mean()),
                        mean_y_66min=float(b['y'][:,ib].mean()),mean_within_cell_change=float(d.mean()),
                        fraction_cells_with_positive_change=float((d>0).mean()),n_cells=len(d),
                        interpretation='Illustrative contrast chosen after waveform inspection, not preregistered. Washout behavior already reported in the source biology paper. No independent-day inference or matched-state causality.')
    return out


def matched_history_pilot(blocks):
    """Exploratory within-cell matching; future outcomes never select pairs.

    First short-pulse washout versus washout after the long second pulse.
    Both future commanded inputs are identically zero for ten minutes.
    One pair per cell is chosen by current-response distance, with a second
    analysis requiring similarity of the immediately observed slope as well.
    This is observational common-support inspection, not a causal experiment.
    """
    summaries={}; all_pairs=[]
    for token in CONCS:
        b=blocks['fgf_mixed_'+token]; t=b['time']
        early=np.flatnonzero((t>=6)&(t<=12))
        late=np.flatnonzero((t>=56)&(t<=102))
        for slope_match in (False,True):
            pairs=[]
            for cell,y in enumerate(b['y']):
                eligible=[]
                for i in early:
                    for j in late:
                        current_distance=abs(y[i]-y[j])
                        slope_distance=abs((y[i]-y[i-1])/2-(y[j]-y[j-1])/2)
                        if current_distance>.005 or (slope_match and slope_distance>.002):
                            continue
                        assert dose(t[i],t[i]+10,b['protocol'],b['concentration'])==0
                        assert dose(t[j],t[j]+10,b['protocol'],b['concentration'])==0
                        eligible.append((current_distance,slope_distance,int(i),int(j)))
                if not eligible:
                    continue
                distance,slope_distance,i,j=min(eligible)
                row=dict(condition_id='fgf_mixed_'+token,cell_row=cell,also_match_slope=slope_match,
                         early_time_min=t[i],late_time_min=t[j],early_current_y=y[i],late_current_y=y[j],
                         current_y_distance=distance,slope_distance=slope_distance,
                         early_target_10min=y[i+5],late_target_10min=y[j+5],
                         late_minus_early_target=y[j+5]-y[i+5])
                pairs.append(row);all_pairs.append(row)
            name=token+('_current_and_slope' if slope_match else '_current_only')
            summaries[name]=dict(n_matched_cells=len(pairs),n_eligible_cells=len(b['y']),
                current_y_tolerance=.005,slope_tolerance_per_min=.002 if slope_match else None,
                mean_absolute_current_y_difference=float(np.mean([p['current_y_distance'] for p in pairs])) if pairs else None,
                mean_late_minus_early_target=float(np.mean([p['late_minus_early_target'] for p in pairs])) if pairs else None,
                median_late_minus_early_target=float(np.median([p['late_minus_early_target'] for p in pairs])) if pairs else None)
    pd.DataFrame(all_pairs).to_csv(OUT/'exploratory_matched_history_pairs.csv',index=False)
    return dict(analyses=summaries,
                limitations=['Exploratory matching rule set after inspecting these data.',
                'Within-cell matching controls cell identity but not age, protocol time, unmeasured physiology, or regression-to-the-mean.',
                'Some support is sparse. No p-value or independent-day confidence interval is justified.',
                'Differences are candidates for prospective randomized preconditioning, not proof of a novel hidden mechanism.'])


def common_window_reference(curves, reference):
    result={}
    for token in CONCS:
        t,yp=reference[token]
        table=curves[curves.run_id.eq('fgf_mixed_'+token)].set_index('time_min').loc[t]
        measured=table.measured_mean.to_numpy()
        result[token]={name:float(np.sqrt(np.mean((pred-measured)**2))) for name,pred in
                       [('positive_single_lag',table.positive_single_lag.to_numpy()),
                        ('stable_filter_bank',table.stable_filter_bank.to_numpy()),
                        ('published_B3_predictive_mean',yp)]}
    return dict(time_min=[0.,180.],condition_rmse=result,
                note='Matched time window and population-mean output. B3 is an original-author fit/export, not an independently retrained model or policy result. The source study used these conditions in architecture selection; B3 is not a fresh blind benchmark.')


def draw(blocks, curves, reference):
    fig,axes=plt.subplots(2,2,figsize=(11,6.4),sharex='col',gridspec_kw={'height_ratios':[3,1]})
    for col,token in enumerate(CONCS):
        b=blocks['fgf_mixed_'+token]; now=b['time']
        selected=curves[curves.run_id.eq('fgf_mixed_'+token)]
        ax=axes[0,col]
        ax.plot(now,b['y'].mean(0),color='#151515',lw=2,label='Measured cell mean')
        ax.plot(selected.time_min,selected.stable_filter_bank,color='#2675b6',label='Fitted stable filter bank')
        ax.plot(selected.time_min,selected.positive_single_lag,color='#c66b28',label='Fitted positive single lag')
        rt,ry=reference[token]
        ax.plot(rt,ry,'--',color='#865bad',label='Published B3 predictive mean')
        ax.set_title(f'FGF2 {CONCS[token]:g} ng/ml; {len(b["y"])} cells')
        ax.set_ylabel('Normalized FRET response'); ax.grid(alpha=.15)
        input_ax=axes[1,col]
        tt=np.linspace(0,258,2581)
        input_ax.fill_between(tt,command(tt,'fgf_mixed',CONCS[token]),color='#5c947c',step='post',alpha=.75)
        input_ax.set_ylabel('Command\nng/ml'); input_ax.set_xlabel('Time from exported origin (min)');input_ax.grid(alpha=.15)
    axes[0,0].legend(loc='upper right',fontsize=8)
    fig.suptitle('Held-out mixed chemical protocol: experimental response and baseline identification',fontsize=12)
    fig.text(.5,.01,'Condition blocks are held out; independent experiment-day IDs and local delivered concentration are unavailable.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(OUT/'fgf2_calibration.png',dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    verified=verify_sources()
    blocks,inventory=read_blocks()
    io,curves=input_output_identification(blocks)
    forecasts=[forecasting(blocks,h) for h in (2.,10.)]
    refs,ref_curves=published_reference(blocks)
    washout=washout_description(blocks)
    matching=matched_history_pilot(blocks)
    comparison=common_window_reference(curves,ref_curves)
    draw(blocks,curves,ref_curves)
    result=dict(status='experimental_data_pilot_complete',source_git_blobs_verified=verified,
                inventory=inventory,n_cells=sum(x['n_cells'] for x in inventory),
                n_measurements=sum(x['n_measurements'] for x in inventory),split=SPLIT,
                forecasting=forecasts,input_output_identification=io,published_reference=refs,
                washout_description=washout,exploratory_history_matching=matching,
                common_window_open_loop_comparison=comparison,
                limitations=['Author-exported, normalized fluorescence time courses; not molecular counts.',
                'Command schedule follows source XML; local concentration/transport are not calibrated.',
                'Condition blocks are disjoint but experimental-day independence cannot be verified.',
                'Filter time scales are computational basis constants, not identifiable biochemical rates.',
                'History prediction gains do not establish a matched-state causal hidden-memory effect.',
                'No novel controller, repeated-command success, dose savings, or new live experiment was evaluated.'])
    (OUT/'calibration_results.json').write_text(json.dumps(result,indent=2))
    (ROOT/'configs/fgf2_condition_split.json').write_text(json.dumps(SPLIT,indent=2))
    summary=dict(n_cells=result['n_cells'],n_measurements=result['n_measurements'],
                 forecast_test_rmse={str(f['horizon_min']):{k:float(np.sqrt(v['test']['run_equal_mse'])) for k,v in f['models'].items()} for f in forecasts},
                 input_output_test_rmse={k:float(np.sqrt(v['test']['run_equal_mse'])) for k,v in io.items()},
                 published_reference=refs,washout=washout,matching=matching,common_window=comparison)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
