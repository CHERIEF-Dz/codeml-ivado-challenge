"""Fold-safe density ablations for the historical-label model; no hidden labels.

Run using the project environment. All outputs are scratch artifacts, never
submission replacements. Cluster labels are features only in their own ablation.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_improved import features, models, NUM, CAT, group
from model_effort import effort_predictions

def matrix(df):
    return np.column_stack([df.cote_r_equivalent, df.heures_travail_semaine,
                            np.log1p(df.revenu_familial_estime),
                            np.log1p(df.distance_domicile_campus_km)])

def density(train, other):
    scaler=StandardScaler()
    z=scaler.fit_transform(matrix(train))
    zo=scaler.transform(matrix(other))
    dist=NearestNeighbors(n_neighbors=20,n_jobs=1).fit(z).kneighbors(z)[0][:,-1]
    eps=float(np.quantile(dist,.90))
    model=DBSCAN(eps=eps,min_samples=20,n_jobs=1).fit(z)
    cluster=model.labels_
    core=model.core_sample_indices_
    # DBSCAN has no predict: use the closest training core point only when
    # it is within fitted epsilon. This is an explicit audit assignment rule.
    if len(core):
        closest=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(z[core])
        d,i=closest.kneighbors(zo)
        other_cluster=np.where(d[:,0] <= eps,cluster[core[i[:,0]]],-1)
    else:
        other_cluster=np.full(len(zo),-1,dtype=int)
    iso=IsolationForest(n_estimators=300,contamination=.05,random_state=41,n_jobs=1).fit(z)
    iso_outlier=iso.predict(z)==-1
    return cluster,other_cluster,iso_outlier,eps

def estimator(cluster=False):
    if not cluster:
        return next(model for name,model in models() if name=='logistic_C1.0')
    prep=ColumnTransformer([
        ('num',StandardScaler(),NUM+['log_income','log_distance']),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT+['density_cluster'])])
    return make_pipeline(prep,LogisticRegression(C=1.,max_iter=2000,random_state=42))

def hours_weight(model):
    scales=model[0].named_transformers_['num'].scale_
    slopes=model[1].coef_[0][:len(scales)]/scales
    return float(slopes[NUM.index('heures_travail_semaine')]/slopes[NUM.index('cote_r_equivalent')])

def metrics(y,p,r,s=None):
    yp=np.asarray(y)==1
    centre_tpr=float(p[yp & ~r].mean())
    remote_tpr=float(p[yp & r].mean())
    result={'accuracy':float(accuracy_score(y,p)),
            'f1_macro':float(f1_score(y,p,average='macro')),
            'f1_positive':float(f1_score(y,p)),
            'eop_gap':abs(centre_tpr-remote_tpr),
            'centre_tpr':centre_tpr,'remote_tpr':remote_tpr,
            'selection_rate':float(np.mean(p))}
    if s is not None: result['log_loss']=float(log_loss(y,s))
    return result

def trim_info(train,mask):
    r=group(train)
    return {'n_removed':int((~mask).sum()),
            'remote_removed_rate':float((~mask)[r].mean()),
            'centre_removed_rate':float((~mask)[~r].mean())}

def main():
    (ROOT/'audit_results').mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'data/donnees_demandes.csv')
    candidates=pd.read_csv(ROOT/'data/candidats_evaluation.csv')
    best=pd.read_csv(ROOT/'results_effort/predictions.csv').set_index('id_candidat').loc[candidates.id_candidat,'decision_octroi'].to_numpy()
    strata=group(df).astype(str)+'_'+df.decision_octroi.astype(str).to_numpy()
    names=['baseline','iforest_trim5','dbscan_trim','dbscan_feature']
    prob={k:np.zeros(len(df)) for k in names}
    merit={k:np.zeros(len(df),dtype=int) for k in names}
    folds={k:[] for k in names}
    for fold,(tr,te) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=41).split(df,strata),1):
        train,test=df.iloc[tr],df.iloc[te]
        clusters,oc,iso_outliers,eps=density(train,test)
        for name in names:
            keep=~iso_outliers if name=='iforest_trim5' else clusters != -1 if name=='dbscan_trim' else np.ones(len(train),dtype=bool)
            xtrain=features(train).copy()
            xtest=features(test).copy()
            if name=='dbscan_feature':
                xtrain['density_cluster']=clusters.astype(str)
                xtest['density_cluster']=oc.astype(str)
            model=estimator(name=='dbscan_feature').fit(xtrain.loc[keep],train.decision_octroi.iloc[np.flatnonzero(keep)])
            p=model.predict_proba(xtest)[:,1]
            prob[name][te]=p
            h=hours_weight(model)
            mp=effort_predictions(test,h)
            merit[name][te]=mp
            folds[name].append({'fold':fold,'eps':eps,'n_clusters':len(set(clusters)-{-1}),
                                'hours_weight':h,**trim_info(train,keep),
                                'historical_classifier':metrics(test.decision_octroi,(p>=.5).astype(int),group(test),p),
                                'effort_policy':metrics(test.decision_octroi,mp,group(test))})
        print(json.dumps({'fold':fold,'completed':True,'eps':eps,'dbscan_noise':int((clusters==-1).sum())}),flush=True)
    result={'warning':'All CV labels are biased historical committee decisions. No hidden-reference estimate. DBSCAN/IsolationForest flags indicate low density, not erroneous records.',
            'folds':5,'seed':41,'dbscan':{'numeric':['cote_r_equivalent','heures_travail_semaine','log1p_revenu','log1p_distance'],
                                         'standardize':'training fold only','eps_rule':'90th percentile distance to 20th point, including self, in training fold','min_samples':20},
            'results':{}}
    clusters,oc,iso_outliers,eps=density(df,candidates)
    for name in names:
        keep=~iso_outliers if name=='iforest_trim5' else clusters != -1 if name=='dbscan_trim' else np.ones(len(df),dtype=bool)
        xtrain=features(df).copy()
        if name=='dbscan_feature': xtrain['density_cluster']=clusters.astype(str)
        model=estimator(name=='dbscan_feature').fit(xtrain.loc[keep],df.decision_octroi.iloc[np.flatnonzero(keep)])
        h=hours_weight(model)
        cp=effort_predictions(candidates,h)
        classifier=(prob[name]>=.5).astype(int)
        result['results'][name]={'historical_classifier_oof':metrics(df.decision_octroi,classifier,group(df),prob[name]),
            'effort_policy_oof':metrics(df.decision_octroi,merit[name],group(df)),
            'full_fit':{'hours_weight':h,'effort_decisions_changed_from_current94_6':int((cp!=best).sum()),
                        'effort_selected':int(cp.sum()),'effort_remote_rate':float(cp[group(candidates)].mean()),
                        'effort_centre_rate':float(cp[~group(candidates)].mean()),**trim_info(df,keep)},
            'fold_results':folds[name]}
        if name!='baseline':
            b=(prob['baseline']>=.5).astype(int)
            by= b == df.decision_octroi.to_numpy()
            ny=classifier == df.decision_octroi.to_numpy()
            result['results'][name]['paired_classifier_vs_baseline']={'changed':int((classifier!=b).sum()),'corrections':int((~by & ny).sum()),'regressions':int((by & ~ny).sum())}
    (ROOT/'audit_results'/'density_validation_results.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,r in result['results'].items():
        print(json.dumps({'model':name,**{k:v for k,v in r.items() if k!='fold_results'}}),flush=True)

if __name__=='__main__': main()
