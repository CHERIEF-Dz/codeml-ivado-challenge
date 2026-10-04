"""Illustrative aggregate-score sensitivity, NOT hidden-accuracy validation.

Assumes an additive latent reference and independent Gaussian/logistic noise;
fits only public candidate features and user-reported aggregate scores. Different
valid reference mechanisms can imply different optimal allocations. No synthetic
reference labels are generated and no primary submission is changed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr, expit

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent/'results'


def main():
    candidates = pd.read_csv(ROOT/'data/candidats_evaluation.csv')
    r = candidates.cote_r_equivalent.to_numpy()
    hours = candidates.heures_travail_semaine.to_numpy()
    income = np.log(candidates.revenu_familial_estime.to_numpy())
    ids = candidates.id_candidat.to_numpy(dtype=str)
    def read(path):
        frame = pd.read_csv(ROOT/path).set_index('id_candidat')
        return frame.loc[ids,'decision_octroi'].to_numpy()
    academic = read('results_academic/predictions.csv')
    effort = read('results_effort/predictions.csv')
    neutral = read('results_improved/predictions_merit.csv')
    def rank(score):
        pred = np.zeros(len(score),dtype=int)
        pred[np.lexsort((ids,-score))[:1600]] = 1
        return pred
    def agreement(pred,p):
        return np.where(pred,p,1-p).mean()
    rows=[]
    for family,cdf in [('gaussian',ndtr),('logistic',expit)]:
        def probability(params):
            a,b,sigma=params
            score=r+a*hours+b*income
            threshold=brentq(lambda t: cdf((score-t)/sigma).mean()-.4,
                             score.min()-15*sigma,score.max()+15*sigma)
            return cdf((score-threshold)/sigma)
        # The old90% was approximate and the scorer metric was not confirmed.
        for old_assumption in [.895,.900,.905,.910,.915,.920]:
            def residual(params):
                p=probability(params)
                return [agreement(academic,p)-.9223,agreement(effort,p)-.946,
                        agreement(neutral,p)-old_assumption]
            fit=least_squares(residual,[.14,-.6,.48 if family=='gaussian' else .27],
                bounds=([-.3,-3,.01],[.6,3,2]),ftol=1e-12,gtol=1e-12,xtol=1e-12,max_nfev=1000)
            if not fit.success or np.max(np.abs(residual(fit.x)))>1e-7:
                raise RuntimeError('Sensitivity fit failed.')
            a,b,noise=fit.x
            pred=rank(r+a*hours+b*income)
            rows.append({'noise_family':family,'assumed_old_accuracy':old_assumption,
                'hours_weight':a,'log_income_weight':b,'noise_scale':noise,
                'illustrative_expected_accuracy_not_validation':agreement(pred,probability(fit.x)),
                'changed_from_effort':int(np.sum(pred!=effort))})
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/'score_sensitivity.csv',index=False)
    (OUT/'score_sensitivity_assumptions.json').write_text(json.dumps({
        'warning':'Assumption-based fits to three feedback values; not out-of-sample evidence or an accuracy ceiling.',
        'expected_reference_prevalence':.4,'academic_report':.9223,'effort_report':.946,
        'old_report':'approximately90%, metric not independently confirmed',
        'source_files':['results_academic/predictions.csv','results_effort/predictions.csv',
                        'results_improved/predictions_merit.csv']},indent=2)+'\n')


if __name__ == '__main__':
    main()
