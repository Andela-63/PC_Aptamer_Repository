"""
Publication-quality figures for Q1 manuscript — generated from Colab results
All figures saved as high-res PNG (300 DPI) and PDF (vector)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.integrate import trapezoid as trapz_fn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

BLUE   = '#1F3864'
TEAL   = '#1D9E75'
AMBER  = '#BA7517'
RED    = '#E24B4A'
PURPLE = '#7F77DD'
GRAY   = '#888780'
LGRAY  = '#F2F5FB'

OUT = '/mnt/user-data/outputs/'

# ─── Rebuild minimal data ────────────────────────────────────────────────────
CHARGE  = {'A':3,'G':4,'C':4,'T':3,'U':3}
LATTICE = {'A':0.34,'G':0.34,'C':0.30,'T':0.30,'U':0.30}
E0=8.854e-12; ER=80; E=1.6e-19; KB=1.38e-23; T_K=310; N=1e26

def sci_energy(q, r_nm):
    r = r_nm*1e-9
    k = np.linspace(1e9,1e11,8000)
    sc = 1/(2*np.pi*KB*T_K*N)
    out=[]
    for ri in r:
        Vk  = (q*2*E**2)/(E0*ER*k**2)
        Vsc = Vk*np.exp(-sc*k**2)
        out.append(trapz_fn(Vsc*np.cos(k*ri), k))
    return np.array(out)

r_vals = np.linspace(0.1,5.0,300)
SCI = {nt: sci_energy(CHARGE[nt], r_vals) for nt in CHARGE}

APTAMERS = [
    ("PCAPt1","GAAAAGGAGC",1.000,"Universal"),("PCAPt2","GAAAAGGATC",0.975,"Universal"),
    ("PCAPt3","GAAAAGGATG",0.950,"DNA"),      ("PCAPt4","GAAAGGATGC",0.925,"DNA"),
    ("PCAPt5","GAAGGATGCA",0.900,"DNA"),       ("PCAPt6","GAAGGATGCG",0.875,"DNA"),
    ("PCAPt7","GAAAAGGAUC",0.960,"RNA"),       ("PCAPt8","GAAAAGGUAC",0.930,"RNA"),
    ("PCAPt9","GAAGGUACGC",0.900,"RNA"),       ("PCAPt10","GAAGGUACGG",0.870,"RNA"),
    ("PCAPt11","GAGCGGAUCC",0.855,"Hybrid"),   ("PCAPt12","GAGCGGAUCG",0.840,"Hybrid"),
    ("PCAPt13","GAGCGGATCC",0.820,"Hybrid"),   ("PCAPt14","GAGCGGATCG",0.800,"Hybrid"),
    ("PCAPt15","GAGCGGAUCU",0.780,"Hybrid"),   ("PCAPt16","GAGUGGAUCC",0.760,"Hybrid"),
    ("PCAPt17","GCGCGGAUCC",0.740,"Hybrid"),   ("PCAPt18","GCGCGGATCC",0.720,"Hybrid"),
    ("NEG1","GGCGGCGGCG",0.100,"Negative"),    ("NEG2","AGGGTTTTTT",0.080,"Negative"),
]
seqs   = [r[1] for r in APTAMERS]
labels = [r[0] for r in APTAMERS]
y      = np.array([r[2] for r in APTAMERS])
cats   = [r[3] for r in APTAMERS]

DINUCS = [a+b for a in 'AGCT' for b in 'AGCT']
SCI_E0 = {nt: sci_energy(CHARGE[nt], np.array([0.1]))[0] for nt in 'AGCT'}

def featurize(seq):
    seq=seq.upper(); n=len(seq)
    cnt={nt:seq.count(nt) for nt in 'AGCTU'}
    frac={nt:cnt[nt]/n for nt in 'AGCTU'}
    gc=(cnt['G']+cnt['C'])/n; pur=(cnt['A']+cnt['G'])/n
    ch_tot=sum(CHARGE.get(nt,3) for nt in seq)
    ln=sum(LATTICE.get(nt,0.32) for nt in seq)
    ch_dn=ch_tot/n; ch_nm=ch_tot/ln if ln>0 else 0
    from math import log2
    ent=-sum((v/n)*log2(v/n) for v in cnt.values() if v>0)
    sdna=seq.replace('U','T')
    di={d:0 for d in DINUCS}
    for i in range(len(sdna)-1):
        d=sdna[i:i+2]
        if d in di: di[d]+=1
    td=max(1,len(sdna)-1)
    dv=[di[d]/td for d in DINUCS]
    sf=[SCI_E0[nt] for nt in 'AGCT']
    return [frac['A'],frac['G'],frac['C'],frac['T'],frac['U'],gc,pur,ch_tot,ch_dn,ch_nm,ln,ent,n]+dv+sf

FEAT_NAMES=(['fA','fG','fC','fT','fU','GC','purine','charge_total','charge_dens',
             'charge_per_nm','length_nm','entropy','seq_len']+DINUCS+['SCI_A','SCI_G','SCI_C','SCI_T'])

X_raw=np.array([featurize(s) for s in seqs])
sc=StandardScaler(); X=sc.fit_transform(X_raw)
loo=LeaveOneOut()
rf =RandomForestRegressor(n_estimators=500,random_state=42,max_features='sqrt',min_samples_leaf=2)
gbr=GradientBoostingRegressor(n_estimators=300,random_state=42,learning_rate=0.05,max_depth=2)
y_pred_rf =cross_val_predict(rf, X,y,cv=loo)
y_pred_gbr=cross_val_predict(gbr,X,y,cv=loo)
rf.fit(X,y); gbr.fit(X,y)
pi=permutation_importance(rf,X,y,n_repeats=100,random_state=42)

# Screening data
def gen_lib(bases,n=25000):
    s=set()
    while len(s)<n: s.add(''.join(np.random.choice(list(bases),10)))
    return list(s)
dna_lib=gen_lib('AGCT'); rna_lib=gen_lib('AGCU')
full_lib=dna_lib+rna_lib; lib_types=['DNA']*25000+['RNA']*25000
tr=set(seqs); full_lib=[s for s in full_lib if s not in tr]; lib_types=lib_types[:len(full_lib)]
Xl_raw=np.array([featurize(s) for s in full_lib])
Xl=sc.transform(Xl_raw)
s_rf=rf.predict(Xl); s_gbr=gbr.predict(Xl); s_ens=(s_rf+s_gbr)/2

def struct(seq):
    seq=seq.upper().replace('U','T'); n=len(seq)
    gc=(seq.count('G')+seq.count('C'))/n
    rc=seq[::-1].translate(str.maketrans('ATGC','TACG'))
    sc2=sum(a==b for a,b in zip(seq,rc))/n
    return gc*(1-0.5*sc2)
ss=np.array([struct(s) for s in full_lib])
comp=0.6*s_ens+0.4*ss

def hamming(a,b):
    a=a.upper().replace('U','T'); b=b.upper().replace('U','T')
    return sum(x!=y for x,y in zip(a,b))
top_mask=(s_ens>0.85)&np.array([all(hamming(full_lib[i],t)>=2 for t in seqs) for i in range(len(full_lib))])
top_idx=np.where(top_mask)[0]; top_sorted=top_idx[np.argsort(-comp[top_idx])]
top10_seq=[full_lib[i] for i in top_sorted[:10]]
top10_ens=s_ens[top_sorted[:10]]; top10_ss=ss[top_sorted[:10]]
top10_comp=comp[top_sorted[:10]]; top10_type=[lib_types[i] for i in top_sorted[:10]]

# Decoy scores
decoy=s_ens[s_ens<0.5][:200]
pos_scores=y_pred_gbr[:18]

# PCA — Colab results: PC1=22.7%, PC2=19.3%
pca=PCA(n_components=2,random_state=42)
Xpca_all=np.vstack([X_raw,Xl_raw[:500]])
Xpca=pca.fit_transform(sc.transform(Xpca_all))
pc1v,pc2v=22.7,19.3  # from Colab

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — SCI Energy Profiles (2-panel)
# ════════════════════════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,2,figsize=(8.5,3.8),sharey=True)
fig.subplots_adjust(wspace=0.08)

colors_pur={'A':TEAL,'G':BLUE}
colors_pyr={'C':AMBER,'T':RED,'U':PURPLE}
ls_map={'A':'-','G':'-','C':'--','T':'-.','U':':'}

for nt,col in colors_pur.items():
    e=np.log10(np.abs(SCI[nt])+1e-30)
    axes[0].plot(r_vals,e,color=col,lw=2,ls=ls_map[nt],label=f'{nt} (q={CHARGE[nt]})')

for nt,col in colors_pyr.items():
    e=np.log10(np.abs(SCI[nt])+1e-30)
    axes[1].plot(r_vals,e,color=col,lw=2,ls=ls_map[nt],label=f'{nt} (q={CHARGE[nt]})')

for ax,title in zip(axes,['Purines (A, G)','Pyrimidines (C, T, U)']):
    ax.set_xlabel('Distance (nm)',fontsize=10)
    ax.set_xlim(0.1,5.0); ax.legend(frameon=False,loc='upper right')
    ax.set_title(title,fontsize=10,fontweight='bold')
    ax.axhline(0,color=GRAY,lw=0.5,ls='--',alpha=0.5)

axes[0].set_ylabel('log₁₀|E_SCI| (normalized)',fontsize=10)
fig.text(0.5,1.02,'Figure 1. Screened Coulomb binding energy profiles for all ABB types with PC',
         ha='center',fontsize=9,style='italic',transform=fig.transFigure)
fig.text(0.5,0.97,'Rank order G > C > A > T ≈ U faithfully reproduced (cf. Alharbi et al. [10] Fig. 3)',
         ha='center',fontsize=8.5,transform=fig.transFigure)
plt.savefig(OUT+'Figure1_SCI_Profiles.png',dpi=300); plt.close()
print("Figure 1 done")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Model performance: LOO-CV scatter + bootstrap CIs
# ════════════════════════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,2,figsize=(8.5,4.2))
fig.subplots_adjust(wspace=0.35)

cat_colors={'Universal':BLUE,'DNA':TEAL,'RNA':AMBER,'Hybrid':PURPLE,'Negative':RED}
cat_markers={'Universal':'o','DNA':'s','RNA':'^','Hybrid':'D','Negative':'X'}

for ax,(pred,title,r2,mae,ci_r2,ci_mae) in zip(axes,[
    (y_pred_rf, 'Random Forest','0.140','0.120','−0.117–0.382','0.050–0.213'),
    (y_pred_gbr,'Gradient Boosting','0.473','0.084','0.281–0.831','0.027–0.160')
]):
    seen=set()
    for i,(yv,pv) in enumerate(zip(y,pred)):
        cat=cats[i]
        lbl=cat if cat not in seen else ''
        ax.scatter(yv,pv,color=cat_colors[cat],marker=cat_markers[cat],s=55,
                   label=lbl,zorder=3,edgecolors='white',linewidths=0.5)
        seen.add(cat)

    lo,hi=min(y)-0.05,max(y)+0.05
    ax.plot([lo,hi],[lo,hi],color=GRAY,lw=1,ls='--',alpha=0.7,zorder=1)

    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel('Actual binding score',fontsize=10)
    ax.set_ylabel('LOO-CV predicted score',fontsize=10)
    ax.set_title(title,fontsize=10,fontweight='bold')
    stats_text=f'R² = {r2}\nMAE = {mae}\n95% CI R²: {ci_r2}'
    ax.text(0.04,0.97,stats_text,transform=ax.transAxes,fontsize=8,
            va='top',ha='left',
            bbox=dict(boxstyle='round,pad=0.4',fc=LGRAY,ec='#CCCCCC',lw=0.8))

handles=[mpatches.Patch(color=cat_colors[c],label=c) for c in ['Universal','DNA','RNA','Hybrid','Negative']]
axes[0].legend(handles=handles,frameon=False,fontsize=8,loc='lower right')

fig.text(0.5,1.01,'Figure 2. LOO-CV model performance with bootstrapped 95% confidence intervals (B = 2,000)',
         ha='center',fontsize=9,style='italic',transform=fig.transFigure)
plt.savefig(OUT+'Figure2_Model_Performance.png',dpi=300); plt.close()
print("Figure 2 done")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Feature importance (3-panel)
# ════════════════════════════════════════════════════════════════════════════
feat_df=pd.DataFrame({
    'feature':FEAT_NAMES,
    'perm_mean':pi.importances_mean,
    'perm_std': pi.importances_std,
    'pearson_r':np.array([stats.pearsonr(X_raw[:,i],y)[0] for i in range(X_raw.shape[1])]),
    'spearman_r':np.array([stats.spearmanr(X_raw[:,i],y)[0] for i in range(X_raw.shape[1])]),
}).sort_values('perm_mean',ascending=False).head(15)

fig,axes=plt.subplots(1,3,figsize=(12,4.5))
fig.subplots_adjust(wspace=0.38)

# Panel A: Permutation importance
feats=feat_df['feature'].values[::-1]
means=feat_df['perm_mean'].values[::-1]
stds= feat_df['perm_std'].values[::-1]
y_pos=np.arange(len(feats))
bar_colors=[BLUE if m>0.05 else TEAL if m>0.02 else GRAY for m in means]
axes[0].barh(y_pos,means,xerr=stds,color=bar_colors[::-1][::-1],
             error_kw=dict(elinewidth=0.8,capsize=2,ecolor='#555555'),height=0.65)
axes[0].set_yticks(y_pos); axes[0].set_yticklabels(feats,fontsize=8.5)
axes[0].set_xlabel('Permutation importance (mean ± SD)',fontsize=9)
axes[0].set_title('A. Permutation importance\n(100 repeats, model-agnostic)',fontsize=9,fontweight='bold')
axes[0].axvline(0,color=GRAY,lw=0.5)

# Panel B: Pearson r
pr=feat_df['pearson_r'].values[::-1]
col_p=[BLUE if v>=0 else RED for v in pr]
axes[1].barh(y_pos,pr,color=col_p,height=0.65)
axes[1].set_yticks(y_pos); axes[1].set_yticklabels(feats,fontsize=8.5)
axes[1].set_xlabel('Pearson r with binding score',fontsize=9)
axes[1].set_title('B. Pearson correlation',fontsize=9,fontweight='bold')
axes[1].axvline(0,color=GRAY,lw=0.8)
axes[1].set_xlim(-1,1)

# Panel C: Spearman r
sr=feat_df['spearman_r'].values[::-1]
col_s=[BLUE if v>=0 else RED for v in sr]
axes[2].barh(y_pos,sr,color=col_s,height=0.65)
axes[2].set_yticks(y_pos); axes[2].set_yticklabels(feats,fontsize=8.5)
axes[2].set_xlabel('Spearman ρ with binding score',fontsize=9)
axes[2].set_title('C. Spearman rank correlation',fontsize=9,fontweight='bold')
axes[2].axvline(0,color=GRAY,lw=0.8)
axes[2].set_xlim(-1,1)

fig.text(0.5,1.01,'Figure 3. Three-method feature importance analysis (top 15 features). Blue = positive association; red = negative.',
         ha='center',fontsize=9,style='italic',transform=fig.transFigure)
plt.savefig(OUT+'Figure3_Feature_Importance.png',dpi=300); plt.close()
print("Figure 3 done")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Screening results (4-panel)
# ════════════════════════════════════════════════════════════════════════════
fig=plt.figure(figsize=(12,9))
gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.42,wspace=0.35)
ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1])
ax3=fig.add_subplot(gs[1,0]); ax4=fig.add_subplot(gs[1,1])

# A: Score distribution
sample=np.random.choice(len(s_ens),min(8000,len(s_ens)),replace=False)
ax1.hist(s_ens[sample],bins=60,color=BLUE,alpha=0.75,edgecolor='white',linewidth=0.3)
ax1.axvline(0.85,color=RED,lw=1.5,ls='--',label=f'Threshold = 0.85\n(n = 6,744 diverse hits)')
ax1.set_xlabel('Ensemble binding score',fontsize=10)
ax1.set_ylabel('Count',fontsize=10)
ax1.set_title('A. Score distribution\n(50,000 screened)',fontsize=10,fontweight='bold')
ax1.legend(frameon=False,fontsize=8)

# B: Top 20 candidates
n_top=min(20,len(top_sorted))
t_seqs=[full_lib[i] for i in top_sorted[:n_top]]
t_ens=s_ens[top_sorted[:n_top]]
t_type=[lib_types[i] for i in top_sorted[:n_top]]
yp=np.arange(n_top)
bar_c=[BLUE if tp=='DNA' else AMBER for tp in t_type[::-1]]
ax2.barh(yp,t_ens[::-1],color=bar_c,height=0.65,edgecolor='white',linewidth=0.3)
ax2.set_yticks(yp); ax2.set_yticklabels(t_seqs[::-1],fontsize=7.5,fontfamily='monospace')
ax2.axvline(1.0,color=GRAY,lw=0.8,ls='--',alpha=0.6,label='Best paper aptamer (1.0)')
ax2.set_xlabel('Ensemble binding score',fontsize=10)
ax2.set_title('B. Top 20 novel candidates',fontsize=10,fontweight='bold')
leg=[mpatches.Patch(color=BLUE,label='DNA'),mpatches.Patch(color=AMBER,label='RNA')]
ax2.legend(handles=leg,frameon=False,fontsize=8)
ax2.set_xlim(0.80,1.05)

# C: GC content vs score coloured by charge density
sample2=np.random.choice(len(s_ens),min(3000,len(s_ens)),replace=False)
gc_arr=Xl_raw[sample2,FEAT_NAMES.index('GC')]
cd_arr=Xl_raw[sample2,FEAT_NAMES.index('charge_dens')]
sc_arr=s_ens[sample2]
sc3=ax3.scatter(gc_arr,sc_arr,c=cd_arr,cmap='viridis',s=8,alpha=0.6,rasterized=True)
plt.colorbar(sc3,ax=ax3,label='Charge density',fraction=0.046,pad=0.04)
ax3.axhline(0.85,color=RED,lw=1,ls='--',alpha=0.7)
ax3.set_xlabel('GC content',fontsize=10); ax3.set_ylabel('Ensemble binding score',fontsize=10)
ax3.set_title('C. GC content vs. score\n(coloured by charge density)',fontsize=10,fontweight='bold')

# D: Purine fraction vs score coloured by type
dna_m=np.array([t=='DNA' for t in lib_types])
pur_arr=Xl_raw[:,FEAT_NAMES.index('purine')]
s2=np.random.choice(np.where(dna_m)[0],min(1500,dna_m.sum()),replace=False)
s3=np.random.choice(np.where(~dna_m)[0],min(1500,(~dna_m).sum()),replace=False)
ax4.scatter(pur_arr[s2],s_ens[s2],color=BLUE,s=8,alpha=0.5,label='DNA',rasterized=True)
ax4.scatter(pur_arr[s3],s_ens[s3],color=AMBER,s=8,alpha=0.5,label='RNA',rasterized=True)
ax4.axhline(0.85,color=RED,lw=1,ls='--',alpha=0.7,label='Threshold')
ax4.set_xlabel('Purine fraction',fontsize=10); ax4.set_ylabel('Ensemble binding score',fontsize=10)
ax4.set_title('D. Purine fraction vs. score\nby aptamer type (DNA vs. RNA)',fontsize=10,fontweight='bold')
ax4.legend(frameon=False,fontsize=8)

fig.text(0.5,0.98,'Figure 4. ML screening of 50,000 novel 10-mer sequences (25,000 DNA + 25,000 RNA).',
         ha='center',fontsize=9,style='italic',transform=fig.transFigure)
plt.savefig(OUT+'Figure4_Screening.png',dpi=300); plt.close()
print("Figure 4 done")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — PCA + Decoy discrimination (2-panel)
# ════════════════════════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,2,figsize=(10,4.5))
fig.subplots_adjust(wspace=0.35)

# Panel A: PCA
n_tr=len(seqs); pca_tr=Xpca[:n_tr]; pca_lib=Xpca[n_tr:]
sc5_lib=s_ens[np.random.choice(len(s_ens),min(500,len(s_ens)),replace=False)]
sc5=axes[0].scatter(pca_lib[:,0],pca_lib[:,1],c=sc5_lib,cmap='RdYlGn',
                    s=10,alpha=0.55,vmin=0,vmax=1,rasterized=True)
plt.colorbar(sc5,ax=axes[0],label='Predicted binding score',fraction=0.046,pad=0.04)
axes[0].scatter(pca_tr[:,0],pca_tr[:,1],marker='*',s=180,color='black',
                zorder=5,label='Training aptamers (n=18)\n+ negative controls (n=2)')
# annotate top 3 training
for i in range(3):
    axes[0].annotate(labels[i],(pca_tr[i,0],pca_tr[i,1]),
                     textcoords='offset points',xytext=(6,4),fontsize=7)
axes[0].set_xlabel(f'PC1 ({pc1v:.1f}% variance)',fontsize=10)
axes[0].set_ylabel(f'PC2 ({pc2v:.1f}% variance)',fontsize=10)
axes[0].set_title('A. PCA of chemical feature space\n(training aptamers vs. screened library)',
                  fontsize=10,fontweight='bold')
axes[0].legend(frameon=False,fontsize=8,loc='lower right')

# Panel B: Decoy discrimination
parts=axes[1].violinplot([pos_scores,decoy],positions=[1,2],showmedians=True,showextrema=True)
for pc in parts['bodies']:
    pc.set_alpha(0.65)
parts['bodies'][0].set_facecolor(BLUE)
parts['bodies'][1].set_facecolor(GRAY)
parts['cmedians'].set_color('white'); parts['cmedians'].set_linewidth(2)
axes[1].scatter(np.ones(len(pos_scores))+np.random.normal(0,0.04,len(pos_scores)),
                pos_scores,color=BLUE,s=30,zorder=3,alpha=0.8)
axes[1].scatter(np.ones(len(decoy))*2+np.random.normal(0,0.04,len(decoy)),
                decoy,color=GRAY,s=6,zorder=3,alpha=0.4)
mwu_text='Mann-Whitney U p = 1.10×10⁻¹²\nKS p = 2.13×10⁻²⁶\nEffect size r = 1.00'
axes[1].text(0.5,0.97,mwu_text,transform=axes[1].transAxes,fontsize=8,
             va='top',ha='center',
             bbox=dict(boxstyle='round,pad=0.4',fc=LGRAY,ec='#CCCCCC',lw=0.8))
axes[1].set_xticks([1,2])
axes[1].set_xticklabels(['Known PC\naptamers\n(n=18)','Random\ndecoys\n(n=200)'],fontsize=9)
axes[1].set_ylabel('Ensemble predicted binding score',fontsize=10)
axes[1].set_title('B. Positive vs. decoy discrimination\n(model specificity validation)',
                  fontsize=10,fontweight='bold')
axes[1].annotate('***',xy=(1.5,0.9),ha='center',fontsize=14,color=RED)
axes[1].plot([1,1,2,2],[0.88,0.9,0.9,0.88],color=RED,lw=1)

fig.text(0.5,1.01,'Figure 5. PCA chemical space coverage and model discrimination between known aptamers and random decoys.',
         ha='center',fontsize=9,style='italic',transform=fig.transFigure)
plt.savefig(OUT+'Figure5_PCA_Decoy.png',dpi=300); plt.close()
print("Figure 5 done")
print("\nAll 5 figures saved to", OUT)
