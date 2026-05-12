"""
UGDSAI 29 — Part 4: DBSCAN + GMM Clustering
Additional unsupervised techniques beyond K-Means and Hierarchical
"""
import os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(BASE, 'data', 'processed')
PLOTS = os.path.join(BASE, 'plots')

plt.rcParams.update({'figure.dpi':150,'font.family':'DejaVu Sans',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.titleweight':'bold','axes.titlesize':13,'axes.labelsize':11})

with open(os.path.join(PROC,'_state.pkl'),'rb') as f:
    state = pickle.load(f)
wide=state['wide']; X_scaled=state['X_scaled']

# Load pre-computed cluster labels from CSV
csv_clusters = pd.read_csv(os.path.join(PROC,'clustered_constituencies.csv'))
csv_clusters = csv_clusters.drop_duplicates(subset='const_key')
for col in ['kmeans_cluster','hc_cluster','cluster_label']:
    if col in csv_clusters.columns:
        wide = wide.merge(csv_clusters[['const_key', col]], on='const_key', how='left', suffixes=('','_new'))
        if col+'_new' in wide.columns:
            wide[col] = wide[col+'_new'].fillna(wide[col])
            wide.drop(columns=[col+'_new'], inplace=True)

pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
X_pca_95 = X_pca[:, :20]

CLUSTER_COLORS = ['#20A558','#FF6600','#3B82F6','#9333EA','#EF4444','#F59E0B','#06B6D4']

# ═══════════════════════════════
# 1. DBSCAN
# ═══════════════════════════════
print("="*60); print("DBSCAN CLUSTERING"); print("="*60)

# Tune eps using k-distance
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(X_pca_95)
distances, _ = nbrs.kneighbors(X_pca_95)
k_dist = np.sort(distances[:, -1])

# Try a range of eps
best_eps, best_sil, best_labels_db = 0.5, -1, None
for eps in np.arange(1.0, 6.0, 0.5):
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_pca_95)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    if n_clusters >= 2:
        sil = silhouette_score(X_pca_95[labels != -1], labels[labels != -1]) if (labels != -1).sum() > 1 else -1
        print(f"  eps={eps:.1f}: clusters={n_clusters}, noise={n_noise}, silhouette={sil:.3f}")
        if sil > best_sil:
            best_sil, best_eps, best_labels_db = sil, eps, labels

wide['dbscan_cluster'] = best_labels_db
n_db_clusters = len(set(best_labels_db)) - (1 if -1 in best_labels_db else 0)
n_noise = (best_labels_db == -1).sum()
print(f"\nBest DBSCAN: eps={best_eps}, clusters={n_db_clusters}, noise={n_noise}, silhouette={best_sil:.3f}")

# ═══════════════════════════════
# 2. GAUSSIAN MIXTURE MODEL (GMM)
# ═══════════════════════════════
print("\n"+"="*60); print("GMM CLUSTERING"); print("="*60)

bic_scores, aic_scores = [], []
for k in range(2, 9):
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gmm.fit(X_pca_95)
    bic_scores.append(gmm.bic(X_pca_95))
    aic_scores.append(gmm.aic(X_pca_95))
    print(f"  k={k}: BIC={gmm.bic(X_pca_95):.0f}, AIC={gmm.aic(X_pca_95):.0f}")

best_k_gmm = list(range(2,9))[np.argmin(bic_scores)]
gmm_final = GaussianMixture(n_components=best_k_gmm, random_state=42, n_init=10)
gmm_final.fit(X_pca_95)
wide['gmm_cluster']  = gmm_final.predict(X_pca_95)
wide['gmm_prob_max'] = gmm_final.predict_proba(X_pca_95).max(axis=1)
sil_gmm = silhouette_score(X_pca_95, wide['gmm_cluster'])
print(f"\nBest GMM k={best_k_gmm}: silhouette={sil_gmm:.4f}")

# ═══════════════════════════════
# Fig 18: DBSCAN plots
# ═══════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# k-distance plot
axes[0].plot(range(len(k_dist)), k_dist, color='#3B82F6', lw=1.5)
axes[0].axhline(best_eps, color='#EF4444', ls='--', lw=1.5, label=f'Chosen eps={best_eps}')
axes[0].set_xlabel('Points (sorted by 5-NN distance)')
axes[0].set_ylabel('5th Nearest Neighbor Distance')
axes[0].set_title('k-Distance Plot (eps selection)')
axes[0].legend()

# DBSCAN cluster scatter in PC space
unique_labels = sorted(set(best_labels_db))
color_map_db = {-1: '#CCCCCC'}
for i, lbl in enumerate([l for l in unique_labels if l != -1]):
    color_map_db[lbl] = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
colors_db = [color_map_db[l] for l in best_labels_db]
axes[1].scatter(X_pca[:,0], X_pca[:,1], c=colors_db, alpha=0.65, s=30, edgecolors='none')
handles = [mpatches.Patch(color=color_map_db[l],
           label='Noise' if l==-1 else f'Cluster {l}') for l in unique_labels]
axes[1].legend(handles=handles, fontsize=9)
axes[1].set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
axes[1].set_title(f'DBSCAN (eps={best_eps}, k={n_db_clusters} clusters, {n_noise} noise)\nSilhouette={best_sil:.3f}')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'18_dbscan.png'), bbox_inches='tight')
plt.close()
print("Saved 18_dbscan.png")

# ═══════════════════════════════
# Fig 19: GMM plots
# ═══════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18,6))

# BIC/AIC
axes[0].plot(range(2,9), bic_scores, 'o-', color='#3B82F6', lw=2, label='BIC')
axes[0].plot(range(2,9), aic_scores, 's-', color='#F59E0B', lw=2, label='AIC')
axes[0].axvline(best_k_gmm, color='#EF4444', ls='--', lw=1.5, label=f'Best k={best_k_gmm}')
axes[0].set_xlabel('Number of Components'); axes[0].set_ylabel('Score')
axes[0].set_title('GMM — BIC & AIC Model Selection'); axes[0].legend()

# GMM clusters in PC space
for c in range(best_k_gmm):
    mask = wide['gmm_cluster'] == c
    axes[1].scatter(X_pca[mask,0], X_pca[mask,1],
                    c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                    alpha=0.65, s=30, edgecolors='none', label=f'Component {c}')
axes[1].set_xlabel(f'PC1'); axes[1].set_ylabel(f'PC2')
axes[1].set_title(f'GMM Clusters (k={best_k_gmm}, Sil={sil_gmm:.3f})')
axes[1].legend(fontsize=8)

# Membership probability distribution (soft clustering)
axes[2].hist(wide['gmm_prob_max'], bins=30, color='#9333EA', alpha=0.8, edgecolor='white')
axes[2].axvline(wide['gmm_prob_max'].median(), color='black', ls='--', lw=1.5,
                label=f"Median={wide['gmm_prob_max'].median():.2f}")
axes[2].set_xlabel('Max Membership Probability'); axes[2].set_ylabel('Count')
axes[2].set_title('GMM — Soft Cluster Confidence\n(How certain is each assignment?)')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'19_gmm.png'), bbox_inches='tight')
plt.close()
print("Saved 19_gmm.png")

# ═══════════════════════════════
# Fig 20: Method Comparison
# ═══════════════════════════════
methods = ['K-Means\n(k=4)', 'Hierarchical\n(Ward,k=4)', f'GMM\n(k={best_k_gmm})', f'DBSCAN\n(eps={best_eps})']
silhouettes = []
from sklearn.metrics import silhouette_score as ss
silhouettes.append(ss(X_pca_95, wide['kmeans_cluster']))
silhouettes.append(ss(X_pca_95, wide['hc_cluster']) if 'hc_cluster' in wide.columns else 0)
silhouettes.append(sil_gmm)
silhouettes.append(best_sil)

colors_bar = ['#20A558','#FF6600','#9333EA','#3B82F6']
fig, ax = plt.subplots(figsize=(10,5))
bars = ax.bar(methods, silhouettes, color=colors_bar, alpha=0.85, edgecolor='white', width=0.5)
for bar, val in zip(bars, silhouettes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_ylabel('Silhouette Score'); ax.set_ylim(0, max(silhouettes)*1.2)
ax.set_title('Clustering Method Comparison — Silhouette Score\n(Higher = better separated clusters)')
ax.axhline(0.1, color='grey', ls='--', lw=1, label='Baseline 0.1')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'20_method_comparison.png'), bbox_inches='tight')
plt.close()
print("Saved 20_method_comparison.png")

# Save updated wide with new cluster columns
wide.to_csv(os.path.join(PROC,'clustered_constituencies.csv'), index=False)

total = len([f for f in os.listdir(PLOTS) if f.endswith('.png')])
print(f"\n✅ Part 4 complete. Total plots: {total}")
