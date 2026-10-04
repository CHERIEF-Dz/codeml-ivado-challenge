"""Unsupervised density audit. No candidate labels or submissions are modified.

Fit transformations and detectors on history only. DBSCAN's evaluation assignment
uses a documented nearest-core-within-eps rule; DBSCAN has no native predict().
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audit_results'
REMOTE = {'Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine'}
HOURS_WEIGHT = 0.14381742782570264


def density_features(df, view='context'):
    result = pd.DataFrame({'r_score':df.cote_r_equivalent, 'work_hours':df.heures_travail_semaine})
    if view == 'context':
        result['log_income'] = np.log1p(df.revenu_familial_estime)
        result['log_distance'] = np.log1p(df.distance_domicile_campus_km)
    elif view != 'merit':
        raise ValueError('Unknown feature view.')
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Audit features must be finite.')
    return result


def transfer_dbscan(model, x):
    """Assign new rows to the nearest training core within eps, else -1.

Also flag rows within eps of core points from multiple clusters. Such border
assignments are intrinsically ambiguous; this rule resolves them by distance.
"""
    labels = np.full(len(x), -1, dtype=int)
    ambiguous = np.zeros(len(x), dtype=bool)
    if len(model.core_sample_indices_) == 0:
        return labels, ambiguous
    core_labels = model.labels_[model.core_sample_indices_]
    neighbors = NearestNeighbors(n_neighbors=1, radius=model.eps).fit(model.components_)
    distances, indices = neighbors.kneighbors(x)
    within = distances[:,0] <= model.eps
    labels[within] = core_labels[indices[within,0]]
    if len(np.unique(core_labels)) > 1:
        for i, core_indices in enumerate(neighbors.radius_neighbors(x, return_distance=False)):
            ambiguous[i] = len(np.unique(core_labels[core_indices])) > 1
    return labels, ambiguous


def group_summary(frame, population, detector, flag):
    rows=[]
    for key in ['All','Centre','Remote']:
        mask = np.ones(len(frame),dtype=bool) if key=='All' else frame.group.eq(key).to_numpy()
        rows.append({'population':population,'detector':detector,'group':key,
                     'rows':int(mask.sum()),'flagged':int(flag[mask].sum()),
                     'flag_rate':float(flag[mask].mean())})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    # This audit describes the scored baseline, even on alternative-policy branches.
    baseline_path = ROOT/'results_effort/predictions.csv'
    original = baseline_path.read_bytes()
    history = pd.read_csv(ROOT/'data/donnees_demandes.csv')
    candidates = pd.read_csv(ROOT/'data/candidats_evaluation.csv')
    submission = pd.read_csv(baseline_path).set_index('id_candidat')
    selected = submission.loc[candidates.id_candidat,'decision_octroi'].to_numpy()
    trials=[]
    primary=None
    for view in ['context','merit']:
        scaler=StandardScaler().fit(density_features(history,view))
        x=scaler.transform(density_features(history,view))
        xc=scaler.transform(density_features(candidates,view))
        for minimum in [10,20,40]:
            distances=NearestNeighbors(n_neighbors=minimum).fit(x).kneighbors(x)[0][:,-1]
            for quantile in [.80,.90,.95]:
                eps=float(np.quantile(distances,quantile))
                db=DBSCAN(eps=eps,min_samples=minimum,n_jobs=1).fit(x)
                transferred, ambiguous=transfer_dbscan(db,xc)
                labels=db.labels_
                cluster_sizes=pd.Series(labels[labels>=0]).value_counts()
                row={'view':view,'min_samples':minimum,'neighbor_distance_quantile':quantile,'eps':eps,
                    'clusters':int(len(cluster_sizes)), 'largest_cluster_share':float(cluster_sizes.max()/len(x)) if len(cluster_sizes) else 0.,
                    'history_noise':int(np.sum(labels==-1)),'candidate_noise':int(np.sum(transferred==-1)),
                    'history_noise_rate':float(np.mean(labels==-1)),'candidate_noise_rate':float(np.mean(transferred==-1)),
                    'candidate_ambiguous_border':int(ambiguous.sum())}
                trials.append(row)
                if view=='context' and minimum==20 and quantile==.90:
                    primary=(scaler,x,xc,db,transferred,ambiguous,row)
        print(f'{view}: completed 9 DBSCAN configurations',flush=True)
    scaler,x,xc,db,candidate_labels,ambiguous,primary_parameters=primary
    isolation=IsolationForest(n_estimators=300,contamination=.05,random_state=41,n_jobs=1).fit(x)
    history_if=isolation.predict(x)==-1
    candidate_if=isolation.predict(xc)==-1
    summaries=[]
    for population,df,labels,iflag in [('history',history,db.labels_,history_if),('evaluation',candidates,candidate_labels,candidate_if)]:
        df=df.copy()
        df['group']=np.where(df.region_administrative.isin(REMOTE),'Remote','Centre')
        df['dbscan_cluster']=labels
        df['dbscan_flag']=labels==-1
        df['isolation_forest_flag']=iflag
        for detector,flag in [('DBSCAN',labels==-1),('IsolationForest',iflag)]:
            summaries.extend(group_summary(df,population,detector,flag))
        if population=='evaluation':
            score=df.cote_r_equivalent.to_numpy()+HOURS_WEIGHT*df.heures_travail_semaine.to_numpy()
            cutoff=float((score[selected==0].max()+score[selected==1].min())/2)
            df['selected']=selected
            df['effort_score']=score
            df['distance_to_cutoff']=score-cutoff
            df['near_cutoff']=np.abs(score-cutoff)<=.5
            df['dbscan_ambiguous_border']=ambiguous
            df['isolation_anomaly_score']=-isolation.score_samples(xc)
            review=df
            df.to_csv(OUT/'candidate_density_audit.csv',index=False)
        else:
            cluster_summary=df.groupby('dbscan_cluster').agg(rows=('id_candidat','size'),
                remote_share=('group',lambda g:float(g.eq('Remote').mean())),
                historical_grant_rate=('decision_octroi','mean'),r_mean=('cote_r_equivalent','mean'),
                hours_mean=('heures_travail_semaine','mean'),income_median=('revenu_familial_estime','median'),
                distance_median=('distance_domicile_campus_km','median'))
            cluster_summary.to_csv(OUT/'historical_cluster_profiles.csv')
    pd.DataFrame(trials).to_csv(OUT/'dbscan_parameter_sensitivity.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'flag_rates_by_group.csv',index=False)
    flags={'DBSCAN':candidate_labels==-1,'IsolationForest':candidate_if,
           'Both':(candidate_labels==-1)&candidate_if}
    details={}
    for name,flag in flags.items():
        details[name]={'flagged':int(flag.sum()),'selected_flagged':int(selected[flag].sum()),
            'flagged_selection_rate':float(selected[flag].mean()) if flag.any() else None,
            'near_cutoff_flagged':int(review.loc[flag,'near_cutoff'].sum()),
            'r_mean_flagged':float(review.loc[flag,'cote_r_equivalent'].mean()),
            'r_range_flagged':[float(review.loc[flag,'cote_r_equivalent'].min()),float(review.loc[flag,'cote_r_equivalent'].max())]}
    report={'purpose':'Audit only; outliers are not bad applicants or proven prediction errors.',
        'fit_population':'10,000 historical rows; labels/IDs/region/postal/program/first-generation excluded from distances',
        'features':list(density_features(history).columns),'scaling':'StandardScaler fitted on historical data only',
        'primary_rule':'Predeclared context view, min_samples20, eps=90th percentile of 20-neighbor training distances (including self)',
        'primary_dbscan':primary_parameters,'isolation_forest':'300 trees; random_state41; contamination=.05 is a chosen review rate, not an estimated anomaly prevalence',
        'group_rates':summaries,'candidate_flags':details,'candidate_near_cutoff_count':int(review.near_cutoff.sum()),
        'effort_cutoff':cutoff,'audited_submission':'results_effort/predictions.csv',
        'audited_submission_accuracy_user_reported':.946,
        'hidden_accuracy_change':None,'prediction_changes':0,
        'submission_sha256':hashlib.sha256(original).hexdigest()}
    (OUT/'density_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
    r_limits=(float(candidates.cote_r_equivalent.min())-.5,float(candidates.cote_r_equivalent.max())+.5)
    for ax,(name,flag) in zip(axes,list(flags.items())[:2]):
        ax.scatter(candidates.cote_r_equivalent[~flag],candidates.heures_travail_semaine[~flag],s=8,alpha=.2,color='slategray')
        ax.scatter(candidates.cote_r_equivalent[flag],candidates.heures_travail_semaine[flag],s=14,alpha=.65,color='darkorange',label=f'Flagged ({int(flag.sum())})')
        rr=np.linspace(*r_limits,100)
        ax.plot(rr,(cutoff-rr)/HOURS_WEIGHT,color='navy',linestyle='--',label='Award cutoff')
        ax.set(title=name,xlabel='Academic R score',ylim=(0,30),xlim=r_limits)
        ax.legend()
    axes[0].set_ylabel('Weekly work hours')
    fig.suptitle('Evaluation audit: sparse profiles are not necessarily award errors')
    fig.tight_layout()
    fig.savefig(OUT/'density_audit.png',dpi=160)
    plt.close(fig)
    assert baseline_path.read_bytes()==original
    print(json.dumps({'primary':primary_parameters,'candidate_flags':details,'near_cutoff':int(review.near_cutoff.sum())},indent=2))


if __name__=='__main__':
    main()
