"""Planning calculations, not observed reliability or a completed power analysis."""
import argparse
import csv
import json
import math
from pathlib import Path
from scipy.stats import beta,norm

ROOT=Path(__file__).resolve().parents[1]


def lower_bound(k,n,alpha=.05):
    if not isinstance(k,int) or not isinstance(n,int) or not 0<=k<=n or n<1 or not 0<alpha<1:
        raise ValueError('Invalid independent binomial counts/confidence tail')
    return 0. if k==0 else float(beta.ppf(alpha,k,n-k+1))


def planning(alpha=.05,power=.8):
    reliability=[]
    for q in (.9,.95,.99):
        n=math.ceil(math.log(alpha)/math.log(q))
        reliability.append({'target':q,'alpha_one_sided':alpha,'n_independent_all_success_trials':n,
                            'lower_bound_if_all_success':lower_bound(n,n,alpha),
                            'assumption':'Independent Bernoulli units and zero failures; not cells in shared chambers.'})
    cost=[]
    for effect in (.2,.3,.5,.8,1.):
        n=math.ceil((norm.ppf(1-alpha)+norm.ppf(power))**2/effect**2)
        cost.append({'standardized_paired_day_effect':effect,'alpha_one_sided':alpha,'power':power,
                     'normal_approximation_independent_days':n,
                     'assumption':'Known paired-day SD and normal mean difference; optimistic planning approximation, not a verified sample size.'})
    return reliability,cost


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--alpha',type=float,default=.05)
    p.add_argument('--power',type=float,default=.8);p.add_argument('--output',type=Path,default=ROOT/'planning_outputs')
    a=p.parse_args()
    if not 0<a.alpha<.5 or not .5<a.power<1:raise ValueError('Invalid planning alpha/power')
    a.output.mkdir(parents=True,exist_ok=True);r,c=planning(a.alpha,a.power)
    for name,rows in [('zero_failure_reliability_planning',r),('paired_day_cost_planning',c)]:
        with (a.output/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    report={'scope':'Planning only. Set multiplicity-adjusted alpha and pilot variance before a fixed formal test.',
            'actual_success_trials':0,'reliability_examples':r,'paired_day_normal_approximation':c}
    (a.output/'sample_size_planning.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
