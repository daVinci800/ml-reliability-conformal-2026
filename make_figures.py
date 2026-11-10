"""Generate the report figures using only exported experimental results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

ROOT=Path(__file__).resolve().parent;R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':200,'axes.titlepad':10})
COL={'LR':'#1d4f91','RF':'#d17b27','HGB':'#398369'}
m=pd.read_csv(R/'metrics.csv');cp=pd.read_csv(R/'conformal.csv');pred=pd.read_csv(R/'predictions.csv');d=pd.read_csv(ROOT/'data.csv');spl=pd.read_csv(R/'splits.csv')
main=m[(m.seed==2026)&(m.sigma==0)];clean=m[m.sigma==0];cpc=cp[cp.sigma==0]
def save(name):plt.savefig(F/(name+'.png'),bbox_inches='tight');plt.close()

fig,ax=plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'width_ratios':[.75,1.25]})
ax[0].bar(['Benign','Malignant'],[357,212],color=['#6b98bd','#c77b69']);ax[0].set_ylabel('Number of samples');ax[0].set_title('WDBC: 569 samples')
for i,n in enumerate([357,212]):ax[0].text(i,n+7,str(n),ha='center')
train=d[d.row_id.isin(spl[(spl.seed==2026)&(spl.split=='train')].row_id)]
cols=['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension']
corr=train[cols].corr(method='spearman');im=ax[1].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
labels=[c.replace('mean ','').replace('fractal dimension','fractal dim.') for c in cols]
ax[1].set_xticks(range(10),labels,rotation=65,ha='right',fontsize=8);ax[1].set_yticks(range(10),labels,fontsize=8);ax[1].set_title('Training features: Spearman correlation');fig.colorbar(im,ax=ax[1],fraction=.035);fig.tight_layout();save('01_data')

fig,axs=plt.subplots(1,3,figsize=(10,3.1))
for ax,name in zip(axs,COL):
 for ver,style in [('raw','--'),('sigmoid','-')]:
  g=pred[(pred.seed==2026)&(pred.sigma==0)&(pred.model==name)&(pred.version==ver)]
  b=np.minimum((g.p.to_numpy()*5).astype(int),4);xs=[];ys=[];ns=[]
  for i in range(5):
   sel=b==i
   if sel.any():xs.append(g.p.to_numpy()[sel].mean());ys.append(g.y.to_numpy()[sel].mean());ns.append(sel.sum())
  ax.plot(xs,ys,style,marker='o',label=ver,color=COL[name],alpha=1 if ver=='sigmoid' else .55)
  for x,y,n in zip(xs,ys,ns):
   if ver=='sigmoid':ax.annotate(str(n),(x,y),xytext=(3,-12),textcoords='offset points',fontsize=7)
 ax.plot([0,1],[0,1],':',color='gray');ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03),title=name,xlabel='Mean predicted P(malignant)',ylabel='Observed malignant fraction');ax.legend(fontsize=8)
fig.tight_layout();save('02_reliability')

fig,axs=plt.subplots(1,2,figsize=(10,3.2))
for name in COL:
 g=pred[(pred.seed==2026)&(pred.sigma==0)&(pred.model==name)&(pred.version=='sigmoid')];fpr,tpr,_=roc_curve(g.y,g.p)
 auc=main[(main.model==name)&(main.version=='sigmoid')].iloc[0].auc
 axs[0].plot(fpr,tpr,label=f'{name}: AUC {auc:.4f}',color=COL[name]);
axs[0].plot([0,1],[0,1],':',color='gray');axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Fixed test set: 114 samples');axs[0].legend(fontsize=8)
for i,name in enumerate(COL):
 for j,ver in enumerate(['raw','sigmoid']):
  vals=clean[(clean.model==name)&(clean.version==ver)].brier.to_numpy();x=i*3+j
  axs[1].scatter(np.full(len(vals),x),vals,alpha=.65,color=COL[name],marker='o' if j==0 else '^');axs[1].plot([x-.25,x+.25],[vals.mean()]*2,color='black')
axs[1].set_xticks([0,1,3,4,6,7],['LR raw','LR cal','RF raw','RF cal','HGB raw','HGB cal'],rotation=35,ha='right');axs[1].set(ylabel='Brier score (lower is better)',title='Ten random splits; bars show means');fig.tight_layout();save('03_roc_repeats')

fig,axs=plt.subplots(1,2,figsize=(10,3.2))
for i,ver in enumerate(['raw','sigmoid']):
 for j,method in enumerate(['global','label']):
  g=cpc[(cpc.model=='LR')&(cpc.version==ver)&(cpc.method==method)]
  label=f'{ver} / {method}';x=i*3+j
  axs[0].scatter(np.full(len(g),x),g.coverage,alpha=.6,color=COL['LR'] if j==0 else COL['RF']);axs[0].plot([x-.25,x+.25],[g.coverage.mean()]*2,color='black')
  axs[1].scatter(g.set_size,g.malignant_coverage,label=label,marker=['o','^','s','D'][i*2+j],alpha=.7)
axs[0].axhline(.9,color='gray',ls=':');axs[0].set_xticks([0,1,3,4],['raw/global','raw/label','cal/global','cal/label'],rotation=25,ha='right');axs[0].set(ylabel='Empirical marginal coverage',title='LR, ten random splits')
axs[1].axhline(.9,color='gray',ls=':');axs[1].set(xlabel='Mean prediction set size',ylabel='Malignant-class coverage',title='Class coverage versus efficiency');axs[1].legend(fontsize=7);fig.tight_layout();save('04_conformal')

fig,axs=plt.subplots(1,2,figsize=(10,3.2))
for name in COL:
 for ax,col,df in [(axs[0],'brier',m[(m.model==name)&(m.version=='sigmoid')]),(axs[1],'coverage',cp[(cp.model==name)&(cp.version=='sigmoid')&(cp.method=='global')])]:
  g=df.groupby('sigma')[col].agg(['mean','std']);xx=g.index.to_numpy();av=g['mean'].to_numpy();sd=g['std'].to_numpy()
  ax.plot(xx,av,'o-',color=COL[name],label=name);ax.fill_between(xx,av-sd,av+sd,color=COL[name],alpha=.13)
axs[0].set(xlabel='Noise scale (training SD units)',ylabel='Brier score',title='Calibrated models; mean +/- split SD');axs[1].set(xlabel='Noise scale (training SD units)',ylabel='Empirical coverage',title='Frozen global conformal thresholds');axs[1].axhline(.9,color='gray',ls=':')
for ax in axs:ax.legend(fontsize=8)
fig.tight_layout();save('05_shift')

fig,axs=plt.subplots(1,3,figsize=(10,3.2),gridspec_kw={'width_ratios':[.8,.8,1.4]})
for ax,ver in zip(axs[:2],['raw','sigmoid']):
 row=main[(main.model=='LR')&(main.version==ver)].iloc[0];arr=np.array([[row.tn,row.fp],[row.fn,row.tp]])
 ax.imshow(arr,cmap='Blues',vmin=0,vmax=80)
 for (i,j),v in np.ndenumerate(arr):ax.text(j,i,str(int(v)),ha='center',va='center',color='white' if v>40 else 'black',fontsize=14)
 ax.set_xticks([0,1],['Benign','Malignant'],fontsize=8);ax.set_yticks([0,1],['Benign','Malignant'],fontsize=8);ax.set(xlabel='Predicted',ylabel='True',title='LR '+ver)
imp=pd.read_csv(R/'importance_LR.csv').head(8).iloc[::-1];axs[2].barh(imp.feature,imp.mean_auc_decrease,xerr=imp.sd,color=COL['LR'],alpha=.8);axs[2].set(xlabel='Permutation AUC decrease',title='LR: 30 permutations / feature');axs[2].tick_params(axis='y',labelsize=8);fig.tight_layout();save('06_errors_importance')

fig,ax=plt.subplots(figsize=(10,2.5));ax.set_axis_off()
boxes=[(.01,.55,.2,.3,'569 WDBC samples\nRandom permutation'),(.27,.55,.2,.3,'284 training\n5-fold CV + refit'),(.53,.55,.2,.3,'85 probability cal.\nMonotone sigmoid'),(.79,.55,.2,.3,'86 conformal cal.\nGlobal / label quantile')]
for x,y,w,h,t in boxes:ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='#f2f5f8',edgecolor='#445b72'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
for x in [.22,.48,.74]:ax.annotate('',(x+.04,.7),(x,.7),arrowprops={'arrowstyle':'->'})
ax.text(.5,.17,'114 untouched test samples  |  raw vs calibrated  |  clean vs perturbed  |  10 splits',ha='center',bbox={'facecolor':'white','edgecolor':'#445b72','pad':10},fontsize=10)
ax.annotate('',(.88,.31),(.88,.54),arrowprops={'arrowstyle':'->'});fig.tight_layout();save('07_pipeline')

summary={}
for name in COL:
 for ver in ['raw','sigmoid']:
  g=clean[(clean.model==name)&(clean.version==ver)];row=main[(main.model==name)&(main.version==ver)].iloc[0]
  summary[name+'_'+ver]={'main':row.to_dict(),'repeat_mean':g[['auc','ap','accuracy','sensitivity','specificity','brier','nll','ece10']].mean().to_dict(),'repeat_sd':g[['auc','accuracy','sensitivity','brier','nll','ece10']].std().to_dict()}
(R/'report_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print('Figures and summary generated.')
