"""Independent audit of exported predictions, splits and reported metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

root=Path(__file__).resolve().parent
p=pd.read_csv(root/'results/predictions.csv')
m=pd.read_csv(root/'results/metrics.csv')
c=pd.read_csv(root/'results/conformal.csv')
s=pd.read_csv(root/'results/splits.csv')
checks=0
for seed,g in s.groupby('seed'):
    assert g.row_id.nunique()==569 and len(g)==569
    assert g.groupby('split').size().to_dict()=={'conformal_cal':86,'prob_cal':85,'test':114,'train':284}
    checks+=1
for keys,g in p.groupby(['seed','model','version','sigma']):
    seed,model,version,sigma=keys
    target=m[(m.seed==seed)&(m.model==model)&(m.version==version)&(m.sigma==sigma)].iloc[0]
    assert set(g.row_id)==set(s[(s.seed==seed)&(s.split=='test')].row_id)
    assert np.all(np.isfinite(g.p)) and np.all((g.p>=0)&(g.p<=1))
    assert abs(roc_auc_score(g.y,g.p)-target.auc)<1e-10
    assert abs(brier_score_loss(g.y,g.p)-target.brier)<1e-10
    tn,fp,fn,tp=confusion_matrix(g.y,g.p>=.5).ravel()
    assert [tn,fp,fn,tp]==[target.tn,target.fp,target.fn,target.tp]
    for method in ['global','label']:
        pred=g[[method+'_benign',method+'_malignant']].to_numpy()
        hit=pred[np.arange(len(g)),g.y.to_numpy()]
        row=c[(c.seed==seed)&(c.model==model)&(c.version==version)&(c.sigma==sigma)&(c.method==method)].iloc[0]
        assert abs(hit.mean()-row.coverage)<1e-12
        assert abs(pred.sum(axis=1).mean()-row.set_size)<1e-12
        assert abs(hit[g.y.to_numpy()==1].mean()-row.malignant_coverage)<1e-12
    checks+=1
result={'status':'PASS','group_checks':checks,'prediction_rows':len(p),'metric_rows':len(m),'conformal_rows':len(c),'split_rows':len(s),'note':'Checks exported predictions; does not constitute independent clinical validation.'}
(root/'results/verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(result)
