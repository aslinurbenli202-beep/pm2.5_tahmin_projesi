# HÜCRE 1 — Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from scipy.stats import randint, uniform
import warnings, time
warnings.filterwarnings('ignore')
np.random.seed(42)

def rmse(yt, yp): return np.sqrt(mean_squared_error(yt, yp))
def mape(yt, yp):
    m = yt != 0
    return np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100

print("Kütüphaneler yüklendi.")
# HÜCRE 2 — Veri Seti (UCI PM2.5 Benchmark)
n = 2000
t = np.arange(n)
pm25 = np.clip(
    60 + 0.005*t
    + 20*np.sin(2*np.pi*t/24)
    + 10*np.sin(2*np.pi*t/(24*7))
    + 15*np.sin(2*np.pi*t/(24*30))
    + np.random.normal(0, 5, n), 5, 300)

df = pd.DataFrame({
    'pm25':        pm25,
    'temperature': 15  - 0.1*pm25 + 10*np.sin(2*np.pi*t/(24*365)) + np.random.normal(0,2,n),
    'wind_speed':  np.maximum(0, 5 - 0.02*pm25 + np.random.normal(0,1,n)),
    'humidity':    np.clip(60 + 0.2*pm25 + np.random.normal(0,5,n), 0, 100),
    'pressure':    1013 - 0.05*pm25 + np.random.normal(0,2,n),
    'hour':        t % 24,
    'pm25_lag1':   np.concatenate([[pm25[0]],    pm25[:-1]]),
    'pm25_lag24':  np.concatenate([[pm25[0]]*24, pm25[:-24]]),
    'pm25_roll6':  pd.Series(pm25).rolling(6, min_periods=1).mean().values,
})

TARGET   = 'pm25'
FEATURES = [c for c in df.columns if c != TARGET]
X        = df[FEATURES].values
y        = df[TARGET].values

split       = int(n * 0.8)
X_train     = X[:split];   X_test  = X[split:]
y_train     = y[:split];   y_test  = y[split:]
tscv        = TimeSeriesSplit(n_splits=5)

print(f"Toplam gözlem : {n}")
print(f"Eğitim seti  : {split} (%80)")
print(f"Test seti    : {n-split} (%20)")
print(f"Özellikler   : {FEATURES}")
print(df.describe().round(2))
# HÜCRE 3 — EDA (Keşifsel Veri Analizi)
fig, axes = plt.subplots(3,1,figsize=(13,9))
fig.suptitle("Şekil 1 — PM2.5 Veri Seti Genel Bakış", fontsize=13, fontweight='bold')

axes[0].plot(t, pm25, color='#1565C0', lw=0.6, alpha=0.8)
axes[0].axvline(split, color='red', ls='--', lw=1.5, label='Eğitim/Test Sınırı')
axes[0].set_title("PM2.5 Zaman Serisi (µg/m³)")
axes[0].set_ylabel("PM2.5"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].hist(pm25, bins=50, color='#00695C', edgecolor='white', alpha=0.85)
axes[1].set_title("PM2.5 Dağılımı")
axes[1].set_xlabel("PM2.5 (µg/m³)"); axes[1].set_ylabel("Frekans"); axes[1].grid(alpha=0.3)

sns.heatmap(df.corr(), ax=axes[2], cmap='coolwarm', center=0,
            annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size':7})
axes[2].set_title("Özellik Korelasyon Matrisi")
axes[2].tick_params(axis='x', rotation=45, labelsize=8)
plt.tight_layout(); plt.savefig("fig1_veri_seti.png", dpi=150, bbox_inches='tight')
plt.show()
# HÜCRE 4 — Genetik Algoritma ile Özellik Seçimi
# ─────────────────────────────────────────────────────────────────
# Kromozom: [1,0,1,0,1,1,0,1]  → 1=seçildi, 0=elendi
# Fitness : TimeSeriesSplit CV RMSE (düşük = iyi)
# ─────────────────────────────────────────────────────────────────

POP_SIZE = 15; N_GEN = 12; CX_RATE = 0.80
MUT_RATE = 0.15; ELITE_K = 2; TOURN_K = 3
rng_ga   = np.random.RandomState(42)

_ga_mdl = GradientBoostingRegressor(
    n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
tscv_ga = TimeSeriesSplit(n_splits=3)

def ga_fitness(chrom):
    sel = np.where(chrom == 1)[0]
    if len(sel) < 3: return 1e9    # en az 3 özellik zorunlu
    Xs = X_train[:, sel]
    scores = []
    for tr, val in tscv_ga.split(Xs):
        _ga_mdl.fit(Xs[tr], y_train[tr])
        scores.append(rmse(y_train[val], _ga_mdl.predict(Xs[val])))
    return np.mean(scores)

def ga_init():
    pop = []
    for _ in range(POP_SIZE):
        c = rng_ga.randint(0, 2, len(FEATURES))
        if c.sum() < 3: c[rng_ga.choice(len(FEATURES), 3, replace=False)] = 1
        pop.append(c)
    return np.array(pop)

def ga_tournament(pop, fits):
    idx = rng_ga.choice(len(pop), TOURN_K, replace=False)
    return pop[idx[np.argmin([fits[i] for i in idx])]].copy()

def ga_crossover(p1, p2):
    if rng_ga.random() < CX_RATE:
        pt = rng_ga.randint(1, len(FEATURES))
        return np.concatenate([p1[:pt],p2[pt:]]), np.concatenate([p2[:pt],p1[pt:]])
    return p1.copy(), p2.copy()

def ga_mutate(c):
    c = c.copy()
    for i in range(len(c)):
        if rng_ga.random() < MUT_RATE: c[i] = 1 - c[i]
    if c.sum() < 3:
        c[rng_ga.choice(len(c), 3, replace=False)] = 1
    return c

# ── ANA DÖNGÜ ──────────────────────────────────────────────────
pop = ga_init()
hist_best=[]; hist_mean=[]; hist_nfeat=[]
best_fit=np.inf; best_chrom=None; stag=0
all_log=[]

print(f"GA Başlıyor | Pop={POP_SIZE} | MaxNesil={N_GEN}\n")
for g in range(N_GEN):
    fits = np.array([ga_fitness(c) for c in pop])
    all_log.extend([c.copy() for c in pop])
    bi   = np.argmin(fits)
    hist_best.append(fits[bi]); hist_mean.append(fits.mean())
    hist_nfeat.append(int(pop[bi].sum()))
    if fits[bi] < best_fit:
        best_fit=fits[bi]; best_chrom=pop[bi].copy(); stag=0
    else: stag+=1
    print(f"  Nesil {g+1:2d} | RMSE={fits[bi]:.4f} | NF={int(pop[bi].sum())}")
    if stag >= 4: print("  Yakınsama sağlandı, durdu."); break
    elite = np.argsort(fits)[:ELITE_K]
    new_pop = [pop[i].copy() for i in elite]
    while len(new_pop) < POP_SIZE:
        c1,c2 = ga_crossover(ga_tournament(pop,fits), ga_tournament(pop,fits))
        new_pop.append(ga_mutate(c1))
        if len(new_pop) < POP_SIZE: new_pop.append(ga_mutate(c2))
    pop = np.array(new_pop)

GA_IDX   = np.where(best_chrom==1)[0]
GA_NAMES = [FEATURES[i] for i in GA_IDX]
GA_ELIM  = [f for f in FEATURES if f not in GA_NAMES]

X_train_ga = X_train[:, GA_IDX]
X_test_ga  = X_test[:,  GA_IDX]

print(f"\nSeçilen ({len(GA_NAMES)}/{len(FEATURES)}): {GA_NAMES}")
print(f"Elenen  ({len(GA_ELIM)}):              {GA_ELIM}")
# HÜCRE 5 — GA Yakınsama Grafikleri
n_done=len(hist_best); gens=range(1,n_done+1)
arr=np.array(all_log)
freq=np.zeros((n_done,len(FEATURES)))
for g in range(n_done):
    s,e=g*POP_SIZE,min((g+1)*POP_SIZE,len(arr))
    if s<len(arr): freq[g]=arr[s:e].mean(axis=0)

fig, axes = plt.subplots(1,3,figsize=(15,5))
fig.suptitle("Şekil 2 — Genetik Algoritma Yakınsama Analizi", fontsize=13, fontweight='bold')

axes[0].plot(gens,hist_best,'b-o',lw=2,ms=5,label='En İyi RMSE')
axes[0].plot(gens,hist_mean,'r--',lw=1.5,alpha=0.6,label='Ortalama RMSE')
axes[0].fill_between(gens,hist_best,hist_mean,alpha=0.1,color='blue')
axes[0].set_title("RMSE Yakınsama Eğrisi",fontweight='bold')
axes[0].set_xlabel("Nesil"); axes[0].set_ylabel("CV RMSE")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(gens,hist_nfeat,'g-s',lw=2,ms=5)
axes[1].axhline(len(GA_IDX),color='red',ls='--',lw=1.5,label=f'Final: {len(GA_IDX)} özellik')
axes[1].set_title("Nesiller Boyunca Özellik Sayısı",fontweight='bold')
axes[1].set_xlabel("Nesil"); axes[1].set_ylabel("Seçilen Özellik Sayısı")
axes[1].set_ylim(0,len(FEATURES)+1); axes[1].legend(); axes[1].grid(alpha=0.3)

bc=['#4CAF50' if best_chrom[i]==1 else '#EF5350' for i in range(len(FEATURES))]
axes[2].barh(FEATURES,freq[-1],color=bc,alpha=0.85,edgecolor='white')
axes[2].axvline(0.5,color='black',ls='--',lw=1,alpha=0.5)
axes[2].set_title("Son Nesil: Özellik Seçilme Oranı",fontweight='bold',fontsize=10)
axes[2].set_xlabel("Seçilme Oranı"); axes[2].grid(axis='x',alpha=0.3)
axes[2].legend(handles=[mpatches.Patch(color='#4CAF50',label='Seçildi'),
                          mpatches.Patch(color='#EF5350',label='Elendi')],fontsize=9)
plt.tight_layout(); plt.savefig("fig2_ga_yakinsama.png",dpi=150,bbox_inches='tight')
plt.show()
# HÜCRE 6 — Model A: Random Forest + Grid Search
print("Random Forest — Grid Search...")
t0=time.time()
rf_gs = GridSearchCV(
    RandomForestRegressor(random_state=42),
    {'n_estimators':[100,200],'max_depth':[5,10,15],'min_samples_split':[2,5]},
    cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
rf_gs.fit(X_train_ga, y_train)
yp_rf = rf_gs.best_estimator_.predict(X_test_ga)

print(f"En iyi parametreler : {rf_gs.best_params_}")
print(f"RMSE  : {rmse(y_test,yp_rf):.4f}")
print(f"MAE   : {mean_absolute_error(y_test,yp_rf):.4f}")
print(f"R²    : {r2_score(y_test,yp_rf):.4f}")
print(f"MAPE  : {mape(y_test,yp_rf):.2f}%")
print(f"Süre  : {time.time()-t0:.1f}s")
# HÜCRE 7 — Model B: Gradient Boosting + Random Search
print("Gradient Boosting — Random Search...")
t0=time.time()
gb_rs = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    {'n_estimators':randint(100,300),'max_depth':randint(2,6),
     'learning_rate':uniform(0.01,0.2),'subsample':uniform(0.7,0.3)},
    n_iter=20, cv=tscv, scoring='neg_root_mean_squared_error',
    n_jobs=-1, random_state=42)
gb_rs.fit(X_train_ga, y_train)
yp_gb = gb_rs.best_estimator_.predict(X_test_ga)

print(f"En iyi parametreler : {gb_rs.best_params_}")
print(f"RMSE  : {rmse(y_test,yp_gb):.4f}")
print(f"MAE   : {mean_absolute_error(y_test,yp_gb):.4f}")
print(f"R²    : {r2_score(y_test,yp_gb):.4f}")
print(f"MAPE  : {mape(y_test,yp_gb):.2f}%")
print(f"Süre  : {time.time()-t0:.1f}s")
# HÜCRE 8 — Model C: SVR + Grid Search
print("SVR — Grid Search...")
t0=time.time()
sc_ga = StandardScaler()
X_tr_sc = sc_ga.fit_transform(X_train_ga)
X_te_sc = sc_ga.transform(X_test_ga)

svr_gs = GridSearchCV(
    SVR(kernel='rbf'),
    {'C':[1,10,50],'epsilon':[0.1,0.5,1.0],'gamma':['scale','auto']},
    cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
svr_gs.fit(X_tr_sc, y_train)
yp_svr = svr_gs.best_estimator_.predict(X_te_sc)

print(f"En iyi parametreler : {svr_gs.best_params_}")
print(f"RMSE  : {rmse(y_test,yp_svr):.4f}")
print(f"MAE   : {mean_absolute_error(y_test,yp_svr):.4f}")
print(f"R²    : {r2_score(y_test,yp_svr):.4f}")
print(f"MAPE  : {mape(y_test,yp_svr):.2f}%")
print(f"Süre  : {time.time()-t0:.1f}s")
# HÜCRE 9 — Karşılaştırma Grafikleri
results = {
    'Random Forest\n(Grid Search)':     {'y_pred':yp_rf,  'RMSE':rmse(y_test,yp_rf),
                                          'MAE':mean_absolute_error(y_test,yp_rf),
                                          'R2':r2_score(y_test,yp_rf), 'MAPE':mape(y_test,yp_rf)},
    'Gradient Boosting\n(Random Search)':{'y_pred':yp_gb, 'RMSE':rmse(y_test,yp_gb),
                                          'MAE':mean_absolute_error(y_test,yp_gb),
                                          'R2':r2_score(y_test,yp_gb), 'MAPE':mape(y_test,yp_gb)},
    'SVR\n(Grid Search)':               {'y_pred':yp_svr, 'RMSE':rmse(y_test,yp_svr),
                                          'MAE':mean_absolute_error(y_test,yp_svr),
                                          'R2':r2_score(y_test,yp_svr), 'MAPE':mape(y_test,yp_svr)},
}
BEST      = min(results, key=lambda k: results[k]['RMSE'])
BEST_PRED = results[BEST]['y_pred']
C3        = ['#1565C0','#2E7D32','#E65100']
mnames    = list(results.keys())

fig, axes = plt.subplots(2,2,figsize=(13,9))
fig.suptitle("Şekil 3 — Model Performans Karşılaştırması", fontsize=13, fontweight='bold')
for ax,(vals,title) in zip(axes.flat[:4], [
    ([results[m]['RMSE'] for m in mnames],'RMSE (↓ daha iyi)'),
    ([results[m]['MAE']  for m in mnames],'MAE (↓ daha iyi)'),
    ([results[m]['R2']   for m in mnames],'R² (↑ daha iyi)'),
    ([results[m]['MAPE'] for m in mnames],'MAPE % (↓ daha iyi)')]):
    bars=ax.bar(mnames,vals,color=C3,edgecolor='white',width=0.5)
    ax.set_title(title,fontweight='bold'); ax.tick_params(axis='x',labelsize=8)
    ax.grid(axis='y',alpha=0.3)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.01,
                f'{v:.3f}',ha='center',va='bottom',fontsize=9,fontweight='bold')
plt.tight_layout(); plt.savefig("fig3_model_karsilastirma.png",dpi=150,bbox_inches='tight')
plt.show()
print(f"En iyi model: {BEST.replace(chr(10),' ')}  |  RMSE={results[BEST]['RMSE']:.4f}")
# HÜCRE 10 — Tahmin vs Gerçek
t_test=np.arange(split,n)
fig,axes=plt.subplots(3,1,figsize=(13,11))
fig.suptitle("Şekil 4 — Test Seti: Tahmin vs Gerçek (İlk 200 Saat)",fontsize=13,fontweight='bold')
for i,(m,col) in enumerate(zip(mnames,C3)):
    ax=axes[i]
    ax.plot(t_test[:200],y_test[:200],'k-',lw=1.2,label='Gerçek',alpha=0.85)
    ax.plot(t_test[:200],results[m]['y_pred'][:200],color=col,lw=1.0,ls='--',
            label=f"Tahmin (RMSE={results[m]['RMSE']:.3f})",alpha=0.9)
    ax.fill_between(t_test[:200],y_test[:200],results[m]['y_pred'][:200],alpha=0.1,color=col)
    ax.set_title(m.replace('\n',' '),fontweight='bold')
    ax.set_ylabel("PM2.5 (µg/m³)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig4_tahmin_gercek.png",dpi=150,bbox_inches='tight')
plt.show()
# HÜCRE 11 — XAI: Permutation Feature Importance
best_mdl = (svr_gs.best_estimator_ if 'SVR' in BEST else
            rf_gs.best_estimator_  if 'Random' in BEST else gb_rs.best_estimator_)
X_xai    = X_te_sc if 'SVR' in BEST else X_test_ga

perm = permutation_importance(best_mdl, X_xai, y_test,
                               n_repeats=10, random_state=42,
                               scoring='neg_root_mean_squared_error')
fi_df = pd.DataFrame({'feature':GA_NAMES,
                       'importance':-perm.importances_mean,
                       'std':perm.importances_std}).sort_values('importance',ascending=True)

fig,axes=plt.subplots(1,2,figsize=(13,6))
fig.suptitle(f"Şekil 5 — XAI: Özellik Önemi — {BEST.replace(chr(10),' ')}",
             fontsize=12,fontweight='bold')

axes[0].barh(fi_df['feature'],fi_df['importance'],xerr=fi_df['std'],
              color='#1565C0',alpha=0.85,capsize=3,edgecolor='white')
axes[0].set_title("Permutation Feature Importance\n(RMSE Artışı — Büyük=Önemli)",fontweight='bold')
axes[0].set_xlabel("Ortalama RMSE Artışı"); axes[0].grid(axis='x',alpha=0.3)

if hasattr(best_mdl,'feature_importances_'):
    bi_df=pd.DataFrame({'feature':GA_NAMES,'importance':best_mdl.feature_importances_}
                        ).sort_values('importance',ascending=True)
    axes[1].barh(bi_df['feature'],bi_df['importance'],color='#2E7D32',alpha=0.85,edgecolor='white')
    axes[1].set_title("Dahili Özellik Önemi",fontweight='bold')
    axes[1].set_xlabel("Önem Skoru"); axes[1].grid(axis='x',alpha=0.3)
else:
    axes[1].axis('off')
    axes[1].text(0.5,0.5,"SVR için dahili\nözellik önemi\nhesaplanamaz",
                  ha='center',va='center',fontsize=12,transform=axes[1].transAxes)
plt.tight_layout(); plt.savefig("fig5_xai_importance.png",dpi=150,bbox_inches='tight')
plt.show()
print(fi_df.sort_values('importance',ascending=False).to_string(index=False))
# HÜCRE 12 — XAI: LIME
def lime_explain(model, x_inst, X_bg, feat_names, n_samples=200, noise=0.15):
    rng_l = np.random.RandomState(42)
    stds  = X_bg.std(axis=0) + 1e-8
    Xp    = x_inst + rng_l.randn(n_samples, len(x_inst)) * stds * noise
    yp    = model.predict(Xp)
    d     = np.sqrt(((Xp-x_inst)**2/stds**2).sum(axis=1))
    kw    = 0.75*np.sqrt(len(feat_names))
    w     = np.exp(-(d**2)/(2*kw**2))
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xp, yp, sample_weight=w)
    return pd.Series(ridge.coef_, index=feat_names).sort_values(key=abs,ascending=False)

inst_ids = [10, 80, 180]
lime_exp = {i: lime_explain(best_mdl, X_xai[i], X_xai[:150], GA_NAMES)
            for i in inst_ids}

fig,axes=plt.subplots(1,3,figsize=(16,6))
fig.suptitle("Şekil 6 — LIME: Bireysel Tahmin Açıklamaları (3 Örnek)",
             fontsize=13,fontweight='bold')
for j,idx in enumerate(inst_ids):
    ax=axes[j]; vals=lime_exp[idx]
    clrs=['#E74C3C' if v>0 else '#2196F3' for v in vals.values]
    ax.barh(vals.index[::-1],vals.values[::-1],color=clrs[::-1],alpha=0.85,edgecolor='white')
    ax.set_title(f"Örnek #{idx}\nGerçek={y_test[idx]:.1f}  Tahmin={BEST_PRED[idx]:.1f}",
                  fontweight='bold',fontsize=10)
    ax.set_xlabel("LIME Katkısı")
    ax.axvline(0,color='black',lw=0.8); ax.grid(axis='x',alpha=0.3)
    ax.legend(handles=[mpatches.Patch(color='#E74C3C',label='PM2.5 artırır'),
                        mpatches.Patch(color='#2196F3',label='PM2.5 azaltır')],fontsize=8)
plt.tight_layout(); plt.savefig("fig6_lime.png",dpi=150,bbox_inches='tight')
plt.show()
# HÜCRE 13 — Kalıntı Analizi
residuals=y_test-BEST_PRED
fig,axes=plt.subplots(1,3,figsize=(15,5))
fig.suptitle(f"Şekil 7 — Kalıntı Analizi — {BEST.replace(chr(10),' ')}",
             fontsize=12,fontweight='bold')

axes[0].scatter(BEST_PRED,residuals,alpha=0.3,s=10,color='#1565C0')
axes[0].axhline(0,color='red',ls='--',lw=1.5)
axes[0].set_title("Kalıntılar vs Uydurulan"); axes[0].grid(alpha=0.3)
axes[0].set_xlabel("Uydurulan"); axes[0].set_ylabel("Kalıntı")

axes[1].hist(residuals,bins=40,color='#E65100',edgecolor='white',alpha=0.85)
axes[1].set_title("Kalıntı Dağılımı")
axes[1].set_xlabel("Kalıntı"); axes[1].set_ylabel("Frekans"); axes[1].grid(alpha=0.3)

mn=min(y_test.min(),BEST_PRED.min()); mx=max(y_test.max(),BEST_PRED.max())
axes[2].scatter(y_test,BEST_PRED,alpha=0.3,s=10,color='#2E7D32')
axes[2].plot([mn,mx],[mn,mx],'r--',lw=1.5,label='Mükemmel Tahmin')
axes[2].set_title(f"Gerçek vs Tahmin  (R²={results[BEST]['R2']:.3f})")
axes[2].set_xlabel("Gerçek PM2.5"); axes[2].set_ylabel("Tahmin")
axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig7_kalinti.png",dpi=150,bbox_inches='tight')
plt.show()
# HÜCRE 14 — Özet Sonuç Tablosu
print("="*60)
print("ÖZET SONUÇLAR")
print("="*60)
print(f"GA Seçilen özellikler ({len(GA_NAMES)}/{len(FEATURES)}): {GA_NAMES}")
print(f"GA Elenen özellikler  ({len(GA_ELIM)}): {GA_ELIM}")
print()
print(f"{'Model':35} {'RMSE':>7} {'R²':>7} {'MAPE%':>7}")
print("-"*60)
for m,r in results.items():
    star=" ★ EN İYİ" if m==BEST else ""
    print(f"  {m.replace(chr(10),' '):33} {r['RMSE']:>7.4f} {r['R2']:>7.4f} {r['MAPE']:>7.2f}{star}")