"""Historical-label diagnostics; these scores do not measure the hidden reference.

Run from any directory. Performs 340 fits: nested 5x3 CV for 12 models plus
100 bootstrap fits. Results are written next to this script under results/.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
 os.environ[key]='1'
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'results'
R='cote_r_equivalent';H='heures_travail_semaine';I='revenu_familial_estime';D='distance_domicile_campus_km';F='premiere_generation_universitaire'
CAT=['programme_etudes','region_administrative','code_postal_3']
class Probit(ClassifierMixin,BaseEstimator):
 def __init__(self,alpha=0.01,max_iter=2500): self.alpha=alpha;self.max_iter=max_iter
 def fit(self,x,y):
  a=np.column_stack([np.ones(len(x)),x]);t=2*np.asarray(y)-1
  def objective(b):
   z=t*(a@b); logp=log_ndtr(z);rat=np.exp(-.5*z*z-.5*np.log(2*np.pi)-logp)
   val=-logp.sum()+.5*self.alpha*np.dot(b[1:],b[1:]);grad=a.T@(-t*rat);grad[1:]+=self.alpha*b[1:]
   return val,grad
  opt=minimize(objective,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':self.max_iter,'ftol':1e-11,'gtol':1e-6})
  if not opt.success: raise RuntimeError('Probit convergence failure: '+str(opt.message))
  self.coef_=opt.x[None,1:];self.intercept_=opt.x[:1];self.classes_=np.array([0,1]);self.n_iter_=opt.nit
  return self
 def decision_function(self,x):return x@self.coef_[0]+self.intercept_[0]
 def predict_proba(self,x):
  p=ndtr(self.decision_function(x));return np.column_stack([1-p,p])
 def predict(self,x):return (self.decision_function(x)>=0).astype(int)

def features(df):
 x=df.drop(columns=['decision_octroi','id_candidat'],errors='ignore').copy()
 x['log_income']=np.log1p(x[I]);x['log_distance']=np.log1p(x[D]);x['work_r']=(x[R]-28)*(x[H]-10)
 x['remote']=x.region_administrative.isin(['Bas-Saint-Laurent','Cote-Nord','Gaspesie-Iles-de-la-Madeleine']).astype(int)
 x['work_remote']=x[H]*x['remote'];return x

def model(name):
 link,form=name.split(':');spl=[]
 if form=='raw': lin=[R,H,I,D,F]
 elif form=='log':lin=[R,H,'log_income','log_distance',F]
 else:lin=[R,H,I,D,F,'log_income','log_distance']
 if form in ['spline_r','spline_rh','spline_all']:spl.append(R)
 if form in ['spline_h','spline_rh','spline_all']:spl.append(H)
 if form=='spline_all':spl.extend(['log_income','log_distance'])
 lin=[v for v in lin if v not in spl]
 if form=='interaction':lin+=['work_r','work_remote']
 transforms=[('num',StandardScaler(),lin),('cat',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False),CAT)]
 if spl: transforms.append(('spline',make_pipeline(SplineTransformer(n_knots=4,degree=3,include_bias=False,extrapolation='linear'),StandardScaler()),spl))
 clf=Probit() if link=='probit' else LogisticRegression(C=100,max_iter=100,solver='newton-cholesky',tol=1e-7)
 return make_pipeline(ColumnTransformer(transforms),clf)

def metric(y,p):
 return {'accuracy':float(accuracy_score(y,p>=.5)),'log_loss':float(log_loss(y,p)),'auc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))}

if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 df=pd.read_csv(ROOT/'data/donnees_demandes.csv');x=features(df);y=df.decision_octroi.to_numpy(); strata=x.region_administrative+'_'+df.decision_octroi.astype(str)
 configs=['logit:raw','logit:log','logit:mixed','probit:log','probit:mixed','logit:spline_r','logit:spline_h','probit:spline_h','logit:spline_rh','logit:spline_all','probit:spline_all','logit:interaction']
 outer=StratifiedKFold(5,shuffle=True,random_state=147)
 pred={name:np.zeros(len(df)) for name in configs};nested=np.zeros(len(df));folds=[];weights=[]
 for fold,(tr,te) in enumerate(outer.split(x,strata)):
  inner=StratifiedKFold(3,shuffle=True,random_state=247+fold);cv={name:[] for name in configs}
  for it,iv in inner.split(x.iloc[tr],strata.iloc[tr]):
   for name in configs:
    m=model(name).fit(x.iloc[tr[it]],y[tr[it]]);p=m.predict_proba(x.iloc[tr[iv]])[:,1];cv[name].append(log_loss(y[tr[iv]],p))
  win=min(cv,key=lambda n:np.mean(cv[n]));fm={'fold':fold,'inner_winner':win,'inner_losses':{n:float(np.mean(v)) for n,v in cv.items()},'outer':{}}
  for name in configs:
   m=model(name).fit(x.iloc[tr],y[tr]);p=m.predict_proba(x.iloc[te])[:,1];pred[name][te]=p;fm['outer'][name]=metric(y[te],p)
   if name==win:nested[te]=p
   if name in ['logit:mixed','probit:mixed']:
    prep=m[0];names=prep.get_feature_names_out();scale=prep.named_transformers_['num'].scale_;cols=prep.transformers_[0][2]
    rw=float(m[-1].coef_[0,np.where(names=='num__'+R)[0][0]]/scale[cols.index(R)])
    hw=float(m[-1].coef_[0,np.where(names=='num__'+H)[0][0]]/scale[cols.index(H)])
    weights.append({'fold':fold,'model':name,'r':rw,'h':hw,'hours_per_r':hw/rw})
  folds.append(fm);print('FOLD',fold,'winner',win,'loss',fm['outer'][win]['log_loss'],flush=True)
 out={'oof':{name:metric(y,p) for name,p in pred.items()},'nested_selected':metric(y,nested),'folds':folds,'weights':weights}
 rng=np.random.default_rng(341);boot=[]
 for k in range(100):
  ix=rng.choice(len(x),len(x),replace=True);m=model('logit:mixed').fit(x.iloc[ix],y[ix]);prep=m[0];names=prep.get_feature_names_out();scale=prep.named_transformers_['num'].scale_;cols=prep.transformers_[0][2]
  rw=float(m[-1].coef_[0,np.where(names=='num__'+R)[0][0]]/scale[cols.index(R)]);hw=float(m[-1].coef_[0,np.where(names=='num__'+H)[0][0]]/scale[cols.index(H)])
  boot.append(hw/rw)
 out['hours_ratio_bootstrap']={'n':len(boot),'mean':float(np.mean(boot)),'std':float(np.std(boot)),'percentiles':{str(q):float(np.percentile(boot,q)) for q in [2.5,25,50,75,97.5]}}
 pd.DataFrame({'target':y,**pred,'nested_selected':nested}).to_csv(OUT/'oof.csv',index=False)
 (OUT/'stats.json').write_text(json.dumps(out,indent=2));print(json.dumps({'oof':out['oof'],'nested_selected':out['nested_selected'],'bootstrap':out['hours_ratio_bootstrap'],'weights':weights},indent=2),flush=True)
