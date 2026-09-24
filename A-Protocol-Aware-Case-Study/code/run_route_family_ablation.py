#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_routes import SignedLogAbsTransformer, RouteARCombinedTransformer
from parse_uci_drift import batches_to_frames, discover_batch_files, feature_columns

GRID_C=[1.0,10.0,50.0]
GRID_GAMMA=[0.001,1.0/256.0,1.0/128.0,0.01]
LABELS=[1,2,3,4,5,6]

class BlockCompTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, block_size=8, eps=1e-9): self.block_size=int(block_size); self.eps=float(eps)
    def fit(self, X, y=None):
        if X.shape[1] % self.block_size: raise ValueError('feature dimension not divisible')
        self.n_features_in_=X.shape[1]; return self
    def transform(self, X):
        n=X.shape[1]//self.block_size
        r=X.reshape(X.shape[0],n,self.block_size)
        d=np.sum(np.abs(r),axis=2,keepdims=True)+self.eps
        return (r/d).reshape(X.shape[0],X.shape[1])

def build_variant(name):
    svc=SVC(kernel='rbf')
    if name=='ss_svm_rbf': return Pipeline([('scaler',StandardScaler()),('svc',svc)])
    if name=='routea_core_svm': return Pipeline([('routea',SignedLogAbsTransformer()),('scaler',StandardScaler()),('svc',svc)])
    if name=='routear_blockcomp_svm': return Pipeline([('block',BlockCompTransformer()),('scaler',StandardScaler()),('svc',svc)])
    if name=='routear_combined_svm': return Pipeline([('routear',RouteARCombinedTransformer(block_size=8)),('scaler',StandardScaler()),('svc',svc)])
    if name=='routear_combined_svm_noscaler': return Pipeline([('routear',RouteARCombinedTransformer(block_size=8)),('svc',svc)])
    raise ValueError(name)

def f1fix(y,p): return float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0))
def f1present(y,p): return float(f1_score(y,p,labels=sorted(np.unique(y)),average='macro',zero_division=0))
def concat(frames,ids): return pd.concat([frames[i] for i in ids],ignore_index=True)
def bestkey(score,c,g): return (score,-c,-g)

def main():
    parser = argparse.ArgumentParser(description="Run the full chronological route-family ablation.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "raw" / "uci270")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "verification")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    root = args.data_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frames=batches_to_frames(discover_batch_files(root),'uci270',128); ids=sorted(frames); cols=feature_columns(128)
    names=['routea_core_svm','routear_blockcomp_svm','routear_combined_svm','routear_combined_svm_noscaler','ss_svm_rbf']
    rows=[]
    for target in [i for i in ids if i>=2]:
        train_ids=[i for i in ids if i<target]; train=concat(frames,train_ids); test=frames[target]
        if target==2:
            X=train[cols].to_numpy(float); y=train.class_label.to_numpy(int)
            Xi,Xv,yi,yv=train_test_split(X,y,test_size=.2,random_state=args.random_seed,stratify=y); nval=len(yv)
        else:
            val_id=target-1; inner=concat(frames,[i for i in ids if i<val_id]); val=frames[val_id]
            Xi=inner[cols].to_numpy(float); yi=inner.class_label.to_numpy(int); Xv=val[cols].to_numpy(float); yv=val.class_label.to_numpy(int); nval=len(yv)
        Xf=train[cols].to_numpy(float); yf=train.class_label.to_numpy(int); Xt=test[cols].to_numpy(float); yt=test.class_label.to_numpy(int)
        for name in names:
            best=None; bestp=None
            for c in GRID_C:
                for g in GRID_GAMMA:
                    m=build_variant(name); m.set_params(svc__C=c,svc__gamma=g); m.fit(Xi,yi); s=f1fix(yv,m.predict(Xv)); k=bestkey(s,c,g)
                    if best is None or k>best: best=k; bestp=(c,g)
            m=build_variant(name); m.set_params(svc__C=bestp[0],svc__gamma=bestp[1]); m.fit(Xf,yf); pred=m.predict(Xt)
            rows.append(dict(target_batch=target,model_name=name,n_test=len(yt),macro_f1_fixed6=f1fix(yt,pred),macro_f1_present=f1present(yt,pred),accuracy=accuracy_score(yt,pred),balanced_accuracy=balanced_accuracy_score(yt,pred),selected_C=bestp[0],selected_gamma=bestp[1]))
    per=pd.DataFrame(rows); per.to_csv(out/'route_family_ablation_b2_b10_per_target.csv',index=False)
    base=per[per.model_name=='ss_svm_rbf'].set_index('target_batch')
    summary=[]
    for name in names:
        s=per[per.model_name==name].set_index('target_batch').sort_index(); d=s.macro_f1_fixed6-base.macro_f1_fixed6
        summary.append(dict(model_name=name,n_targets=len(s),weighted_mean_macro_f1=float(np.average(s.macro_f1_fixed6,weights=s.n_test)),mean_macro_f1=float(s.macro_f1_fixed6.mean()),weighted_mean_macro_f1_present=float(np.average(s.macro_f1_present,weights=s.n_test)),weighted_mean_accuracy=float(np.average(s.accuracy,weights=s.n_test)),weighted_mean_delta_macro_f1_vs_ss=float(np.average(d,weights=s.n_test)),mean_delta_macro_f1_vs_ss=float(d.mean()),wins_vs_ss=int((d>0).sum()),losses_vs_ss=int((d<0).sum()),ties_vs_ss=int((d==0).sum())))
    sm=pd.DataFrame(summary); sm.to_csv(out/'route_family_ablation_b2_b10_summary.csv',index=False)
    support_rows=[]
    per_index=per.set_index(['target_batch','model_name'])
    for target in [i for i in ids if i>=2]:
        counts=frames[target].class_label.value_counts().sort_index()
        route=per_index.loc[(target,'routear_combined_svm')]
        baseline=per_index.loc[(target,'ss_svm_rbf')]
        missing=[str(label) for label in LABELS if label not in counts.index]
        support_rows.append(dict(
            target_batch=target, n_test=int(counts.sum()), n_classes_present=int(len(counts)),
            missing_labels=','.join(missing) if missing else None,
            routear_fixed6=route.macro_f1_fixed6, routear_present=route.macro_f1_present,
            ss_fixed6=baseline.macro_f1_fixed6, ss_present=baseline.macro_f1_present,
            routear_delta_fixed6=route.macro_f1_fixed6-baseline.macro_f1_fixed6,
            routear_delta_present=route.macro_f1_present-baseline.macro_f1_present,
            support_min=int(counts.min()), support_max=int(counts.max())))
    pd.DataFrame(support_rows).to_csv(out/'metric_label_support_sensitivity.csv',index=False)
    print(sm.to_string(index=False))
    print(f"Wrote per-target, summary, and class-support tables to {out}")
if __name__=='__main__': main()
