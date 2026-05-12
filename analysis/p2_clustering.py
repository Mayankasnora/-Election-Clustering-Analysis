"""
UGDSAI 29 — Part 2: PCA, Clustering, t-SNE, Swing Analysis, Plots
"""
import os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
PROC  = os.path.join(BASE, 'data', 'processed')
PLOTS = os.path.join(BASE, 'plots')

plt.rcParams.update({'figure.dpi':150,'font.family':'DejaVu Sans',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.titleweight':'bold','axes.titlesize':13,'axes.labelsize':11})

with open(os.path.join(PROC, '_state.pkl'), 'rb') as f:
    state = pickle.load(f)
wide=state['wide']; X_scaled=state['X_scaled']; feat_cols=state['feat_cols']
df=state['df']; eci=state['eci']; FEATURES=state['FEATURES']
PARTY_COLORS=state['PARTY_COLORS']; YEAR_COLORS=state['YEAR_COLORS']
CLUSTER_COLORS = ['#20A558','#FF6600','#3B82F6','#9333EA']
OPTIMAL_K = 4

# ── PCA ──
print("="*60); print("STEP 4 — PCA"); print("="*60)
pca = PCA(n_components=min(20, X_scaled.shape[1]))
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
for i,v in enumerate(explained[:10]):
    print(f"  PC{i+1}: {v:.3f} ({v*100:.1f}%) | cum: {sum(explained[:i+1])*100:.1f}%")

party_color_map = {'TMC':'#20A558','BJP':'#FF6600','LEFT':'#CC0000','INC':'#0077B5','OTHERS':'#888888'}
loadings = pd.DataFrame(pca.components_[:5].T, index=feat_cols, columns=[f'PC{i+1}' for i in range(5)])

fig, axes = plt.subplots(1,2,figsize=(14,5))
cum_var = np.cumsum(explained[:15])*100
axes[0].bar(range(1,16), explained[:15]*100, color='#3B82F6', alpha=0.7, label='Individual')
axes[0].plot(range(1,16), cum_var, 'o-', color='#EF4444', lw=2, label='Cumulative')
axes[0].axhline(80, color='grey', ls='--', lw=1, label='80% threshold')
axes[0].set_xlabel('Principal Component'); axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_title('PCA Scree Plot'); axes[0].legend(); axes[0].set_xticks(range(1,16))
colors_dom = wide['dom_2021'].map(party_color_map).fillna('#888888')
axes[1].scatter(X_pca[:,0], X_pca[:,1], c=colors_dom, alpha=0.6, s=30, edgecolors='none')
axes[1].set_xlabel(f'PC1 ({explained[0]*100:.1f}% var)'); axes[1].set_ylabel(f'PC2 ({explained[1]*100:.1f}% var)')
axes[1].set_title('PC1 vs PC2 — Dominant Party 2021')
axes[1].legend(handles=[mpatches.Patch(color=v,label=k) for k,v in party_color_map.items()], fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'01_pca.png'),bbox_inches='tight'); plt.close()
print("Saved 01_pca.png")

fig, axes = plt.subplots(1,2,figsize=(14,6))
for ax,pc,title in [(axes[0],'PC1','PC1 — Loadings (Top 12)'),(axes[1],'PC2','PC2 — Loadings (Top 12)')]:
    top = loadings[pc].abs().nlargest(12); vals = loadings.loc[top.index, pc]
    labels = [l.replace('_vote_share','_vs').replace('_percent','_%').replace('log_','ln_') for l in vals.index]
    ax.barh(range(len(vals)), vals.values, color=['#20A558' if v>0 else '#EF4444' for v in vals], alpha=0.8)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0,color='black',lw=0.8); ax.set_title(title); ax.set_xlabel('Loading')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'02_pca_loadings.png'),bbox_inches='tight'); plt.close()
print("Saved 02_pca_loadings.png")

# ── Optimal K ──
print("="*60); print("STEP 5 — OPTIMAL K"); print("="*60)
cumvar = np.cumsum(explained)
n95 = int(np.argmax(cumvar >= 0.95)) + 1 if any(cumvar >= 0.95) else len(explained)
X_pca_95 = X_pca[:, :n95]
print(f"Using {n95} PCs for clustering (covers {cumvar[n95-1]*100:.1f}% variance)")
inertias, silhouettes = [], []
for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca_95)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca_95, labels))
    print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={silhouettes[-1]:.3f}")

fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].plot(range(2,11), inertias, 'o-', color='#3B82F6', lw=2)
axes[0].axvline(OPTIMAL_K, color='#EF4444', ls='--', lw=1.5, label=f'Chosen k={OPTIMAL_K}')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].set_title('Elbow Method')
axes[0].set_xticks(range(2,11)); axes[0].legend()
best_sil_k = list(range(2,11))[np.argmax(silhouettes)]
axes[1].plot(range(2,11), silhouettes, 's-', color='#F59E0B', lw=2)
axes[1].axvline(best_sil_k, color='#EF4444', ls='--', lw=1.5, label=f'Best k={best_sil_k}')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score'); axes[1].set_title('Silhouette Score')
axes[1].set_xticks(range(2,11)); axes[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'03_elbow_silhouette.png'),bbox_inches='tight'); plt.close()
print("Saved 03_elbow_silhouette.png")

# ── KMeans ──
print("="*60); print("STEP 6 — K-MEANS"); print("="*60)
km_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=20)
wide['kmeans_cluster'] = km_final.fit_predict(X_pca_95)
sil_final = silhouette_score(X_pca_95, wide['kmeans_cluster'])
print(f"Silhouette (k={OPTIMAL_K}): {sil_final:.4f}")
print(wide['kmeans_cluster'].value_counts().sort_index())

profile_cols = [c for c in wide.columns if any(x in c for x in
    ['tmc_vote_share','bjp_vote_share','left_inc_vote_share','turnout_percent',
     'victory_margin_percent','enp','criminal_cases_2021','age_2021','log_assets_2021'])]
profiles = wide.groupby('kmeans_cluster')[profile_cols].mean().round(3)

cluster_labels = {}
for c in range(OPTIMAL_K):
    row = profiles.loc[c]
    tmc21 = row.get('tmc_vote_share_2021', 0)
    bjp21 = row.get('bjp_vote_share_2021', 0)
    left21 = row.get('left_inc_vote_share_2021', 0)
    margin21 = row.get('victory_margin_percent_2021', 0)
    if tmc21 > 0.45:   cluster_labels[c] = f'TMC Strongholds (C{c})'
    elif bjp21 > 0.40: cluster_labels[c] = f'BJP Surge (C{c})'
    elif left21 > 0.25:cluster_labels[c] = f'Left Legacy (C{c})'
    elif margin21 < 0.08: cluster_labels[c] = f'Swing/Competitive (C{c})'
    else:              cluster_labels[c] = f'Mixed Profile (C{c})'

wide['cluster_label'] = wide['kmeans_cluster'].map(cluster_labels)
print("Cluster labels:", cluster_labels)

fig, axes = plt.subplots(1,2,figsize=(14,6))
for c in range(OPTIMAL_K):
    mask = wide['kmeans_cluster']==c
    axes[0].scatter(X_pca[mask,0], X_pca[mask,1], c=CLUSTER_COLORS[c],
                    label=cluster_labels[c], alpha=0.65, s=30, edgecolors='none')
    cx,cy = X_pca[mask,0].mean(), X_pca[mask,1].mean()
    axes[0].scatter(cx, cy, marker='*', s=200, c=CLUSTER_COLORS[c], edgecolors='black', lw=1.2, zorder=5)
axes[0].set_xlabel(f'PC1 ({explained[0]*100:.1f}% var)'); axes[0].set_ylabel(f'PC2 ({explained[1]*100:.1f}% var)')
axes[0].set_title('K-Means Clusters (PC Space)'); axes[0].legend(fontsize=8)
heat_data = profiles[[c for c in profiles.columns if 'vote_share' in c]]
heat_labels = [c.replace('_vote_share','').replace('_',' ').upper() for c in heat_data.columns]
sns.heatmap(heat_data.T*100, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=[cluster_labels[i] for i in range(OPTIMAL_K)],
            yticklabels=heat_labels, ax=axes[1], linewidths=0.5, cbar_kws={'label':'Vote Share %'})
axes[1].set_title('Party Vote Share by Cluster (%)')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'04_kmeans_clusters.png'),bbox_inches='tight'); plt.close()
print("Saved 04_kmeans_clusters.png")

# ── Without PCA ──
km_raw = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=20)
wide['kmeans_nopca'] = km_raw.fit_predict(X_scaled)
sil_raw = silhouette_score(X_scaled, wide['kmeans_nopca'])
print(f"Silhouette WITHOUT PCA: {sil_raw:.4f} | WITH PCA: {sil_final:.4f}")
fig, axes = plt.subplots(1,2,figsize=(14,6))
for c in range(OPTIMAL_K):
    axes[0].scatter(X_pca[wide['kmeans_nopca']==c,0], X_pca[wide['kmeans_nopca']==c,1],
                    c=CLUSTER_COLORS[c], alpha=0.6, s=30, edgecolors='none', label=f'C{c}')
    axes[1].scatter(X_pca[wide['kmeans_cluster']==c,0], X_pca[wide['kmeans_cluster']==c,1],
                    c=CLUSTER_COLORS[c], alpha=0.6, s=30, edgecolors='none', label=cluster_labels[c])
axes[0].set_title(f'WITHOUT PCA (Sil={sil_raw:.3f})'); axes[0].legend(fontsize=9)
axes[1].set_title(f'WITH PCA (Sil={sil_final:.3f})'); axes[1].legend(fontsize=8)
for ax in axes: ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'05_pca_vs_nopca.png'),bbox_inches='tight'); plt.close()
print("Saved 05_pca_vs_nopca.png")

# ── Hierarchical ──
print("="*60); print("STEP 8 — HIERARCHICAL"); print("="*60)
Z = linkage(X_pca_95, method='ward')
fig, ax = plt.subplots(figsize=(14,6))
dendrogram(Z, ax=ax, color_threshold=0, above_threshold_color='grey', no_labels=True)
for lvl,lbl,col in [(Z[-3,2],'k=4 cut','#EF4444'),(Z[-5,2],'k=6 cut','#F59E0B')]:
    ax.axhline(y=lvl, color=col, ls='--', lw=1.5, label=f'{lbl} (d={lvl:.0f})')
ax.set_title('Ward Hierarchical Clustering Dendrogram'); ax.set_xlabel('Constituencies'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'06_dendrogram.png'),bbox_inches='tight'); plt.close()
print("Saved 06_dendrogram.png")

hc = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward')
wide['hc_cluster'] = hc.fit_predict(X_pca_95)
sil_hc = silhouette_score(X_pca_95, wide['hc_cluster'])
ari = adjusted_rand_score(wide['kmeans_cluster'], wide['hc_cluster'])
print(f"HC Silhouette: {sil_hc:.4f} | ARI (KMeans vs HC): {ari:.4f}")

# ── t-SNE ──
print("="*60); print("STEP 9 — t-SNE"); print("="*60)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, learning_rate='auto', init='pca')
X_tsne = tsne.fit_transform(X_pca_95)
fig, axes = plt.subplots(1,2,figsize=(14,6))
for c in range(OPTIMAL_K):
    mask = wide['kmeans_cluster']==c
    axes[0].scatter(X_tsne[mask,0], X_tsne[mask,1], c=CLUSTER_COLORS[c],
                    alpha=0.7, s=35, edgecolors='none', label=cluster_labels[c])
axes[0].set_title('t-SNE — K-Means Cluster'); axes[0].legend(fontsize=8)
dom26_colors = wide['dom_2026'].map(PARTY_COLORS).fillna('#888888')
axes[1].scatter(X_tsne[:,0], X_tsne[:,1], c=dom26_colors, alpha=0.7, s=35, edgecolors='none')
axes[1].set_title('t-SNE — Dominant Party 2026')
axes[1].legend(handles=[mpatches.Patch(color=v,label=k) for k,v in PARTY_COLORS.items()], fontsize=9)
for ax in axes: ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'07_tsne.png'),bbox_inches='tight'); plt.close()
print("Saved 07_tsne.png")

# ── Swing Analysis ──
print("="*60); print("STEP 10 — SWING ANALYSIS"); print("="*60)
eci_piv = eci.pivot_table(index='const_key', columns='year',
    values=['tmc_vote_share','bjp_vote_share','left_inc_vote_share','turnout_percent'])
eci_piv.columns = ['_'.join([str(c) for c in col]).strip() for col in eci_piv.columns]
eci_piv = eci_piv.reset_index()
for party,col in [('tmc','tmc_vote_share'),('bjp','bjp_vote_share'),('left','left_inc_vote_share')]:
    c16,c21,c26 = f'{col}_2016',f'{col}_2021',f'{col}_2026'
    if all(c in eci_piv.columns for c in [c16,c21,c26]):
        eci_piv[f'{party}_swing_16_21'] = eci_piv[c21] - eci_piv[c16]
        eci_piv[f'{party}_swing_21_26'] = eci_piv[c26] - eci_piv[c21]
eci_piv = eci_piv.dropna(subset=['bjp_swing_16_21','bjp_swing_21_26'])
swing_seats      = eci_piv[(eci_piv['bjp_swing_16_21']>10)&(eci_piv['bjp_swing_21_26']<-10)]
bjp_consolidated = eci_piv[(eci_piv['bjp_swing_16_21']>5)&(eci_piv['bjp_swing_21_26']>0)]
left_collapse    = eci_piv[eci_piv['left_swing_16_21']<-15]
print(f"Swing seats: {len(swing_seats)} | BJP consolidated: {len(bjp_consolidated)} | Left collapse: {len(left_collapse)}")

fig, axes = plt.subplots(1,2,figsize=(14,6))
for ax,xs,ys,title in [
    (axes[0],'tmc_swing_16_21','bjp_swing_16_21','2016→2021 Swing'),
    (axes[1],'tmc_swing_21_26','bjp_swing_21_26','2021→2026 Swing')]:
    ax.scatter(eci_piv[xs], eci_piv[ys], alpha=0.4, s=25, color='grey', edgecolors='none')
    ax.axhline(0,color='black',lw=0.8); ax.axvline(0,color='black',lw=0.8)
    ax.set_xlabel('TMC Change (pp)'); ax.set_ylabel('BJP Change (pp)'); ax.set_title(title)
    ax.text(0.02,0.97,'TMC↓ BJP↑',transform=ax.transAxes,ha='left',va='top',color='#FF6600',fontsize=9)
    ax.text(0.98,0.03,'TMC↑ BJP↓',transform=ax.transAxes,ha='right',va='bottom',color='#20A558',fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'08_swing_analysis.png'),bbox_inches='tight'); plt.close()
print("Saved 08_swing_analysis.png")

# ── Turnout ──
fig, axes = plt.subplots(1,3,figsize=(15,5))
for i,yr in enumerate([2016,2021,2026]):
    data = eci[eci['year']==yr]['turnout_percent'].dropna()
    axes[i].hist(data, bins=30, color=YEAR_COLORS[yr], alpha=0.8, edgecolor='white', lw=0.5)
    axes[i].axvline(data.median(), color='black', ls='--', lw=1.5, label=f'Median: {data.median():.1f}%')
    axes[i].set_title(f'Turnout {yr}'); axes[i].set_xlabel('Turnout (%)'); axes[i].legend()
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'09_turnout.png'),bbox_inches='tight'); plt.close()
print("Saved 09_turnout.png")

# ── Winner Profiles ──
winner_data = df[df['year'].isin([2021,2026])].merge(
    wide[['const_key','kmeans_cluster','cluster_label']], on='const_key', how='left'
).dropna(subset=['kmeans_cluster','criminal_cases','age','log_assets'])
fig, axes = plt.subplots(1,3,figsize=(15,5))
for ax,col,title,fmt in [
    (axes[0],'criminal_cases','Criminal Cases','{:.1f}'),
    (axes[1],'age','Avg Age','{:.0f}'),
    (axes[2],'log_assets','Log(Assets)','{:.2f}')]:
    grouped = winner_data.groupby('kmeans_cluster')[col].mean()
    bars = ax.bar([cluster_labels.get(int(i),f'C{i}') for i in grouped.index],
                  grouped.values, color=[CLUSTER_COLORS[int(i)] for i in grouped.index], alpha=0.8, edgecolor='white')
    for bar,val in zip(bars, grouped.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01, fmt.format(val), ha='center', va='bottom', fontsize=9)
    ax.set_title(title); ax.tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'10_winner_profiles.png'),bbox_inches='tight'); plt.close()
print("Saved 10_winner_profiles.png")

# ── Party Seats ──
party_year = eci.groupby(['year','winner_party']).size().reset_index(name='seats')
top_parties = party_year.groupby('winner_party')['seats'].sum().nlargest(5).index
party_year = party_year[party_year['winner_party'].isin(top_parties)]
fig, ax = plt.subplots(figsize=(12,6))
pivot_py = party_year.pivot(index='winner_party', columns='year', values='seats').fillna(0)
x = np.arange(len(pivot_py.index)); width = 0.25
for i,yr in enumerate([2016,2021,2026]):
    if yr in pivot_py.columns:
        ax.bar(x+i*width, pivot_py[yr], width, label=str(yr), color=YEAR_COLORS[yr], alpha=0.85, edgecolor='white')
ax.set_xticks(x+width); ax.set_xticklabels(pivot_py.index, rotation=15)
ax.set_ylabel('Seats Won'); ax.set_title('Seats Won by Party — 2016, 2021, 2026'); ax.legend(title='Year')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'11_party_seats.png'),bbox_inches='tight'); plt.close()
print("Saved 11_party_seats.png")

# ── Cluster Party Mix ──
comp = wide.groupby(['kmeans_cluster','dom_2021']).size().reset_index(name='count')
fig, axes = plt.subplots(1,OPTIMAL_K,figsize=(16,5))
for c in range(OPTIMAL_K):
    data = comp[comp['kmeans_cluster']==c].set_index('dom_2021')['count']
    colors_pie = [PARTY_COLORS.get(p,'#888888') for p in data.index]
    axes[c].pie(data.values, labels=data.index, colors=colors_pie, autopct='%1.0f%%', startangle=90, textprops={'fontsize':9})
    axes[c].set_title(cluster_labels[c], fontsize=10)
plt.suptitle('Dominant Party Mix by Cluster (2021)', fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'12_cluster_party_mix.png'),bbox_inches='tight'); plt.close()
print("Saved 12_cluster_party_mix.png")

# ── Export ──
wide.to_csv(os.path.join(PROC,'clustered_constituencies.csv'), index=False)
print(f"\nAll 12 plots saved to {PLOTS}")
print(f"Silhouette: {sil_final:.4f} | PCs for 95% var: {n95}")
print(f"Swing seats: {len(swing_seats)} | Left collapse: {len(left_collapse)}")
print("Done ✓")
