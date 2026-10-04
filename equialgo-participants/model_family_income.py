"""Family-income policy: modest need priority among comparable merit scores.

The tested R/work-hours model is retained; an explicit negative log-income term
represents capacity for family support. Its strength is a policy setting, not a
validated estimate of the hidden-reference optimum.
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from model_improved import ROOT, group

HOURS_WEIGHT = 0.14381742782570264
NEED_WEIGHT = 0.67
INCOME_REFERENCE = 60000.0


def family_income_scores(candidates, need_weight=NEED_WEIGHT):
    required = ['id_candidat', 'cote_r_equivalent', 'heures_travail_semaine', 'revenu_familial_estime']
    if candidates[required].isna().any().any() or not candidates.id_candidat.is_unique:
        raise ValueError('Unique IDs and complete merit/income features are required.')
    values = candidates[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(need_weight) or need_weight < 0:
        raise ValueError('Features must be finite and the need weight nonnegative.')
    r_score, hours, income = values.T
    if (hours < 0).any() or (income <= 0).any():
        raise ValueError('Work hours must be nonnegative and family income positive.')
    return r_score + HOURS_WEIGHT * hours - need_weight * np.log(income / INCOME_REFERENCE)


def family_income_predictions(candidates, need_weight=NEED_WEIGHT):
    score = family_income_scores(candidates, need_weight)
    # At equal adjusted score, prefer lower income; ID resolves remaining ties.
    order = np.lexsort((candidates.id_candidat.to_numpy(dtype=str),
                       candidates.revenu_familial_estime.to_numpy(dtype=float), -score))
    predictions = np.zeros(len(candidates), dtype=int)
    predictions[order[:round(.40 * len(candidates))]] = 1
    return predictions


def describe(df, mask):
    columns = ['cote_r_equivalent', 'heures_travail_semaine', 'revenu_familial_estime']
    part = df.loc[mask, columns]
    return {'count':len(part), 'mean':{k:float(v) if len(part) else None for k,v in part.mean().items()},
            'median':{k:float(v) if len(part) else None for k,v in part.median().items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--need-weight', type=float, default=NEED_WEIGHT)
    args = parser.parse_args()
    candidates = pd.read_csv(ROOT/'data/candidats_evaluation.csv')
    pred = family_income_predictions(candidates, args.need_weight)
    if len(pred) != 4000 or pred.sum() != 1600:
        raise ValueError('The challenge requires 4,000 applicants and a budget-valid allocation.')
    old_path = ROOT/'results_effort/predictions.csv'
    old_bytes = old_path.read_bytes()
    old = pd.read_csv(old_path).set_index('id_candidat').loc[candidates.id_candidat,'decision_octroi'].to_numpy()
    incoming, outgoing = (pred==1)&(old==0), (pred==0)&(old==1)
    remote = group(candidates)
    output = ROOT/'results_family_income'
    output.mkdir(exist_ok=True)
    submission = pd.DataFrame({'id_candidat':candidates.id_candidat, 'decision_octroi':pred})
    submission.to_csv(output/'predictions.csv',index=False)
    submission.to_csv(ROOT/'predictions.csv',index=False)
    audit = {'policy':'Modest financial-need priority at comparable academic/working merit',
        'formula':f'R + {HOURS_WEIGHT} * weekly_hours - {args.need_weight} * ln(income / {INCOME_REFERENCE})',
        'hours_weight':HOURS_WEIGHT,'need_weight':args.need_weight,
        'benefit_of_halving_income_in_r_points':float(args.need_weight*np.log(2)),
        'income_weight_origin':'Policy choice, previously explored with aggregate-score sensitivity; not independently validated',
        'rows':len(pred),'selected':int(pred.sum()),'selection_rate':float(pred.mean()),
        'changed_from_tested_94_6':int(np.sum(pred!=old)),
        'incoming':describe(candidates,incoming),'outgoing':describe(candidates,outgoing),
        'previous_selected':describe(candidates,old==1),'new_selected':describe(candidates,pred==1),
        'centre_selection_rate':float(pred[~remote].mean()),'remote_selection_rate':float(pred[remote].mean()),
        'external_accuracy':None,'external_f1_macro':None,
        'baseline_file':'results_effort/predictions.csv','baseline_user_reported_accuracy':.946,
        'sha256':hashlib.sha256((ROOT/'predictions.csv').read_bytes()).hexdigest()}
    (output/'submission_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    active={'file':'predictions.csv','archive':'results_family_income/predictions.csv',
            'policy':'family_income','status':'Awaiting external evaluation',
            'sha256':audit['sha256'],'best_scored_file':'results_effort/predictions.csv'}
    (ROOT/'active_submission.json').write_text(json.dumps(active,indent=2)+'\n')
    sweep=[]
    for weight in [0.,.25,.5,.67,1.]:
        p=family_income_predictions(candidates,weight)
        sweep.append({'need_weight':weight,'selection_rate':float(p.mean()),
            'changed_from_tested_94_6':int(np.sum(p!=old)),
            'selected_mean_r':float(candidates.loc[p==1,'cote_r_equivalent'].mean()),
            'selected_median_income':float(candidates.loc[p==1,'revenu_familial_estime'].median()),
            'centre_selection_rate':float(p[~remote].mean()),'remote_selection_rate':float(p[remote].mean())})
    pd.DataFrame(sweep).to_csv(output/'policy_sensitivity.csv',index=False)
    assert old_path.read_bytes()==old_bytes
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
