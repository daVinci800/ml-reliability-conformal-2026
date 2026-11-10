"""Reproduce all numerical results. Positive label 1 means malignant.

The training, probability calibration, conformal calibration and test partitions
are disjoint. No model or calibration setting is chosen using test performance.
Run from any directory: python run_experiments.py --repeats 10
"""
from pathlib import Path
import argparse, hashlib, json, platform, sys, time
from importlib.metadata import version as package_version
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, brier_score_loss, log_loss, confusion_matrix, f1_score)
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
FIG = ROOT / 'figures'

def savejson(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')

def ece(y, p, bins=10):
    """Equal-width positive-class ECE, not top-label ECE of Guo et al."""
    group = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(np.mean(group == i) * abs(y[group == i].mean() - p[group == i].mean())
        for i in range(bins) if np.any(group == i)))

def metrics(y, p):
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(auc=roc_auc_score(y,p), ap=average_precision_score(y,p),
        accuracy=accuracy_score(y,pred), balanced_accuracy=balanced_accuracy_score(y,pred),
        f1=f1_score(y,pred), sensitivity=tp/(tp+fn), specificity=tn/(tn+fp),
        brier=brier_score_loss(y,p), nll=log_loss(y,np.c_[1-p,p],labels=[0,1]),
        ece10=ece(y,p,10), ece5=ece(y,p,5), ece15=ece(y,p,15), tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def fit_sigmoid(p, y):
    """Fit p_cal=sigmoid(a*logit(p)+b), a>=0, tiny L2 regularization.

    Monotonicity preserves ranking unless a reaches zero or numeric ties occur.
    Optimized only on probability-calibration data; not temperature scaling.
    """
    z=logit(np.clip(p,1e-6,1-1e-6))
    def objective(t):
        v=t[0]*z+t[1]
        loss=np.mean(np.logaddexp(0,v)-y*v)+1e-4*(t[0]**2+t[1]**2)/2
        r=expit(v)-y
        grad=np.array([np.mean(r*z),np.mean(r)])+1e-4*t
        return loss,grad
    fit=minimize(objective,[1.,0.],jac=True,bounds=[(0,None),(None,None)],method='L-BFGS-B')
    if not fit.success: raise RuntimeError(fit.message)
    return fit.x

def transform(p, ab):
    return expit(ab[0]*logit(np.clip(p,1e-6,1-1e-6))+ab[1])

def quantile(scores, alpha):
    n=len(scores); k=int(np.ceil((n+1)*(1-alpha)))
    return float(np.sort(scores)[k-1]) if k<=n else float('inf')

def cp_fit(p,y,alpha=.1,mondrian=False):
    s=1-np.where(y==1,p,1-p)
    return np.array([quantile(s[y==c],alpha) if mondrian else quantile(s,alpha) for c in [0,1]])

def cp_sets(p,q):
    return (1-np.c_[1-p,p])<=q[None,:]

def wilson(k,n):
    z=1.959963984540054; ph=k/n; den=1+z*z/n
    mid=(ph+z*z/(2*n))/den
    rad=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return [float(mid-rad),float(mid+rad)]

def cp_metrics(y,sets):
    hit=sets[np.arange(len(y)),y]; size=sets.sum(axis=1); single=size==1
    return dict(coverage=float(hit.mean()),set_size=float(size.mean()),
        malignant_coverage=float(hit[y==1].mean()),benign_coverage=float(hit[y==0].mean()),
        empty_rate=float(np.mean(size==0)),singleton_rate=float(single.mean()),double_rate=float(np.mean(size==2)),
        singleton_error=float(np.mean(~hit[single])) if single.any() else None,
        covered=int(hit.sum()),n=len(y),coverage_ci=wilson(int(hit.sum()),len(y)))

def specifications(seed):
    return {
      'LR':(make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,random_state=seed)),
            {'logisticregression__C':[.01,.1,1.,10.]}),
      'RF':(RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=1),
            {'max_depth':[None,5],'min_samples_leaf':[1,4]}),
      'HGB':(HistGradientBoostingClassifier(max_iter=150,early_stopping=False,random_state=seed),
            {'learning_rate':[.05,.1],'max_leaf_nodes':[7,15]})}

def main(repeats):
    started=time.time(); RES.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    ds=load_breast_cancer(); X=ds.data; y=(ds.target==0).astype(int)
    frame=pd.DataFrame(X,columns=ds.feature_names);frame.insert(0,'row_id',np.arange(len(y)));frame['malignant']=y
    frame.to_csv(ROOT/'data.csv',index=False,float_format='%.12g')
    (ROOT/'DATA_DESCRIPTION.txt').write_text(ds.DESCR,encoding='utf-8')
    audit=dict(n=len(y),features=X.shape[1],malignant=int(y.sum()),benign=int((1-y).sum()),
       missing=int(np.isnan(X).sum()),duplicate_features=int(pd.DataFrame(X).duplicated().sum()),
       duplicate_labeled_rows=int(frame.drop(columns='row_id').duplicated().sum()),
       constant_features=int((np.std(X,axis=0)==0).sum()),
       sha256=hashlib.sha256((ROOT/'data.csv').read_bytes()).hexdigest(),
       label_mapping='sklearn target 0 malignant -> 1; sklearn target 1 benign -> 0')
    savejson(audit,RES/'data_audit.json')
    frame.describe().to_csv(RES/'descriptive_statistics.csv')
    rows=[]; cp_rows=[]; predictions=[]; splits=[]; tuning=[]; main_art={}
    for run in range(repeats):
        seed=2026+run; perm=np.random.default_rng(seed).permutation(len(y))
        tr,pc,cc,te=perm[:284],perm[284:369],perm[369:455],perm[455:]
        assert len(set(tr)|set(pc)|set(cc)|set(te))==len(y)
        for label,ids in [('train',tr),('prob_cal',pc),('conformal_cal',cc),('test',te)]:
            for idx in ids:splits.append(dict(seed=seed,row_id=int(idx),split=label,y=int(y[idx])))
        rng=np.random.default_rng(seed+100000); z=rng.normal(size=X[te].shape); train_sd=X[tr].std(axis=0)
        if run==0:
            savejson({s:dict(n=len(ids),malignant=int(y[ids].sum()),benign=int((1-y[ids]).sum())) for s,ids in [('train',tr),('prob_cal',pc),('conformal_cal',cc),('test',te)]},RES/'main_split_counts.json')
        for name,(model,grid) in specifications(seed).items():
            cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
            search=GridSearchCV(model,grid,cv=cv,scoring='roc_auc',n_jobs=1,refit=True)
            search.fit(X[tr],y[tr]); model=search.best_estimator_
            cvdf=pd.DataFrame(search.cv_results_);cvdf.insert(0,'seed',seed);cvdf.insert(1,'model',name)
            cvdf.to_csv(RES/f'cv_{seed}_{name}.csv',index=False)
            tuning.append(dict(seed=seed,model=name,cv_auc=search.best_score_,params=search.best_params_))
            pcraw=model.predict_proba(X[pc])[:,1];ccraw=model.predict_proba(X[cc])[:,1]
            ab=fit_sigmoid(pcraw,y[pc]); basetest=model.predict_proba(X[te])[:,1]
            for version in ['raw','sigmoid']:
                ccp=ccraw if version=='raw' else transform(ccraw,ab)
                qs={m:cp_fit(ccp,y[cc],mondrian=m=='label') for m in ['global','label']}
                for sigma in [0.,.25,.5,1.]:
                    pert=X[te] if sigma==0 else np.maximum(0,X[te]+sigma*z*train_sd)
                    p=model.predict_proba(pert)[:,1];p=p if version=='raw' else transform(p,ab)
                    rows.append(dict(seed=seed,model=name,version=version,sigma=sigma,**metrics(y[te],p)))
                    for method,q in qs.items():
                        sets=cp_sets(p,q); r=cp_metrics(y[te],sets)
                        cp_rows.append(dict(seed=seed,model=name,version=version,sigma=sigma,method=method,q_benign=float(q[0]),q_malignant=float(q[1]),**r))
                    sg=cp_sets(p,qs['global']);sl=cp_sets(p,qs['label'])
                    for j,idx in enumerate(te):
                        predictions.append(dict(seed=seed,model=name,version=version,sigma=sigma,row_id=int(idx),y=int(y[idx]),p=float(p[j]),global_benign=int(sg[j,0]),global_malignant=int(sg[j,1]),label_benign=int(sl[j,0]),label_malignant=int(sl[j,1])))
                if run==0:
                    main_art[name+'_'+version]=dict(p=(basetest if version=='raw' else transform(basetest,ab)).tolist(),y=y[te].tolist(),ids=te.tolist(),calibration_ab=ab.tolist(),conformal_q={m:q.tolist() for m,q in qs.items()})
            if run==0:
                pi=permutation_importance(model,X[te],y[te],scoring='roc_auc',n_repeats=30,random_state=2026,n_jobs=1)
                pd.DataFrame(dict(feature=ds.feature_names,mean_auc_decrease=pi.importances_mean,sd=pi.importances_std)).sort_values('mean_auc_decrease',ascending=False).to_csv(RES/f'importance_{name}.csv',index=False)
            print(f'seed={seed} model={name} cv_auc={search.best_score_:.4f} ab={ab.round(3)} elapsed={time.time()-started:.1f}s',flush=True)
        pd.DataFrame(rows).to_csv(RES/'metrics.csv',index=False)
        pd.DataFrame(cp_rows).to_csv(RES/'conformal.csv',index=False)
    pd.DataFrame(predictions).to_csv(RES/'predictions.csv',index=False,float_format='%.12g')
    pd.DataFrame(splits).to_csv(RES/'splits.csv',index=False)
    savejson(tuning,RES/'tuning.json');savejson(main_art,RES/'main_predictions.json')
    # Paired bootstrap conditions on the already fitted models and fixed test set.
    a=main_art['LR_sigmoid']; b=main_art['RF_sigmoid']; yy=np.array(a['y']);pa=np.array(a['p']);pb=np.array(b['p'])
    rng=np.random.default_rng(4242);boot=[]
    for _ in range(2000):
        ix=rng.integers(0,len(yy),len(yy));yt=yy[ix]
        if len(np.unique(yt))<2:continue
        boot.append([roc_auc_score(yt,pa[ix])-roc_auc_score(yt,pb[ix]),np.mean((yt-pa[ix])**2)-np.mean((yt-pb[ix])**2)])
    boot=np.array(boot);savejson(dict(comparison='LR_sigmoid minus RF_sigmoid; conditional paired percentile bootstrap, 2000 resamples',auc_difference=float(roc_auc_score(yy,pa)-roc_auc_score(yy,pb)),brier_difference=float(np.mean((yy-pa)**2)-np.mean((yy-pb)**2)),auc_ci=np.quantile(boot[:,0],[.025,.975]).tolist(),brier_ci=np.quantile(boot[:,1],[.025,.975]).tolist()),RES/'bootstrap.json')
    env=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn_version,seeds=list(range(2026,2026+repeats)),elapsed_seconds=time.time()-started,alpha=.1,cv='5 fold stratified within training only',selection_metric='ROC-AUC',splits=[284,85,86,114],sigmoid_penalty=1e-4,noise='max(0, X_test + sigma * train_std * Z), same Z at all levels',date='2026-11-10')
    savejson(env,RES/'environment.json')
    (ROOT/'requirements.txt').write_text('\n'.join([f'numpy=={np.__version__}',f'pandas=={pd.__version__}',f'scipy=={scipy.__version__}',f'scikit-learn=={sklearn_version}',f'matplotlib=={package_version("matplotlib")}'])+'\n',encoding='utf-8')
    print('COMPLETE',env,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=10);args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be >= 1')
    with threadpool_limits(limits=2):main(args.repeats)
