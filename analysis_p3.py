"""
UGDSAI 29 — Part 3: Missing Analysis Items
- Census 2011 demographic integration
- Polarization analysis (urban/rural, literacy vs vote share)
- Association analysis
- District-level heatmap
"""
import os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, 'plots')
DATA  = os.path.join(BASE, 'data')

plt.rcParams.update({'figure.dpi':150,'font.family':'DejaVu Sans',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.titleweight':'bold','axes.titlesize':13,'axes.labelsize':11})

# ── Load state ──
with open(os.path.join(BASE,'_state.pkl'),'rb') as f:
    state = pickle.load(f)
wide=state['wide']; df=state['df']; eci=state['eci']
PARTY_COLORS={'TMC':'#20A558','BJP':'#FF6600','LEFT':'#CC0000','INC':'#0077B5','OTHERS':'#888888'}

# ══════════════════════════════════════════════════
# 1. CENSUS 2011 DATA (West Bengal — 19 Districts)
# ══════════════════════════════════════════════════
print("="*60); print("STEP A — CENSUS 2011 INTEGRATION"); print("="*60)

census = pd.DataFrame({
    'district': [
        'KOLKATA','NORTH TWENTY FOUR PARGANAS','SOUTH TWENTY FOUR PARGANAS',
        'HOWRAH','HOOGHLY','NADIA','MURSHIDABAD','BIRBHUM',
        'BARDDHAMAN','BANKURA','PURULIA','PASCHIM MEDINIPUR',
        'PURBA MEDINIPUR','MALDA','UTTAR DINAJPUR','DAKSHIN DINAJPUR',
        'COOCH BEHAR','JALPAIGURI','DARJEELING'
    ],
    'literacy_rate':  [87.1,82.0,78.6,82.0,82.8,75.6,67.0,70.9,75.8,72.4,65.4,79.0,87.7,62.7,60.1,75.0,75.5,75.6,79.9],
    'urban_pct':      [100.0,56.3,20.3,65.7,43.0,35.7,16.8,15.1,40.4,17.4,20.2,15.3,13.0,20.9,17.8,22.4,23.6,25.9,35.8],
    'sex_ratio':      [899,956,956,938,951,947,957,956,940,955,955,962,956,937,940,954,942,954,971],
    'pop_density':    [24306,2463,819,3300,1751,1318,1334,774,1082,523,468,601,1076,1072,905,680,752,621,585],
})

# Load myneta to get constituency→district mapping
myn = pd.read_csv(os.path.join(DATA,'myneta.csv'))
myn['const_key'] = myn['constituency'].str.upper().str.strip()
myn['district_upper'] = myn['district'].str.upper().str.strip()

# Map known district name variants
district_map = {
    'NORTH 24 PARGANAS':'NORTH TWENTY FOUR PARGANAS',
    'SOUTH 24 PARGANAS':'SOUTH TWENTY FOUR PARGANAS',
    'NORTH TWENTY FOUR PARGANAS (BARASAT)':'NORTH TWENTY FOUR PARGANAS',
    'SOUTH TWENTY FOUR PARGANAS (ALIPORE)':'SOUTH TWENTY FOUR PARGANAS',
    'BURDWAN':'BARDDHAMAN','BARDHAMAN':'BARDDHAMAN',
    'WEST MIDNAPORE':'PASCHIM MEDINIPUR','EAST MIDNAPORE':'PURBA MEDINIPUR',
    'MIDNAPORE (WEST)':'PASCHIM MEDINIPUR','MIDNAPORE (EAST)':'PURBA MEDINIPUR',
    'NORTH DINAJPUR':'UTTAR DINAJPUR','SOUTH DINAJPUR':'DAKSHIN DINAJPUR',
    'COOCHBEHAR':'COOCH BEHAR','JALPAIGURI':'JALPAIGURI',
}
myn['district_clean'] = myn['district_upper'].replace(district_map)
const_district = myn[['const_key','district_clean']].drop_duplicates('const_key')

# Join census to wide
wide2 = wide.merge(const_district, on='const_key', how='left')
wide2 = wide2.merge(census.rename(columns={'district':'district_clean'}), on='district_clean', how='left')
matched = wide2['literacy_rate'].notna().sum()
print(f"Census matched: {matched}/{len(wide2)} constituencies ({matched/len(wide2)*100:.0f}%)")

# ── Fig 13: Literacy & Urban % vs Party Vote Share ──
fig, axes = plt.subplots(2, 2, figsize=(14,10))

vote_cols = {'tmc_vote_share_2021':'TMC 2021','bjp_vote_share_2021':'BJP 2021',
             'left_inc_vote_share_2021':'Left/INC 2021','tmc_vote_share_2026':'TMC 2026'}
demo_pairs = [('literacy_rate','Literacy Rate (%)'),('urban_pct','Urban Population (%)')]

for row, (demo_col, demo_label) in enumerate(demo_pairs):
    for col_i, (vote_col, vote_label) in enumerate(list(vote_cols.items())[:2]):
        ax = axes[row][col_i]
        sub = wide2[[demo_col, vote_col, 'dom_2021']].dropna()
        colors = sub['dom_2021'].map(PARTY_COLORS).fillna('#888888')
        ax.scatter(sub[demo_col], sub[vote_col]*100, c=colors, alpha=0.5, s=25, edgecolors='none')
        # Trend line
        z = np.polyfit(sub[demo_col], sub[vote_col]*100, 1)
        p = np.poly1d(z)
        xline = np.linspace(sub[demo_col].min(), sub[demo_col].max(), 100)
        ax.plot(xline, p(xline), 'k--', lw=1.5, alpha=0.6)
        corr = sub[[demo_col, vote_col]].corr().iloc[0,1]
        ax.set_xlabel(demo_label); ax.set_ylabel(f'{vote_label} (%)')
        ax.set_title(f'{vote_label} vs {demo_label}\n(r = {corr:.3f})')
        ax.legend(handles=[mpatches.Patch(color=v,label=k) for k,v in PARTY_COLORS.items()],
                  fontsize=7, loc='upper right')

plt.suptitle('Polarization: Literacy & Urbanisation vs Vote Share', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'13_polarization_literacy_urban.png'), bbox_inches='tight')
plt.close()
print("Saved 13_polarization_literacy_urban.png")

# ── Fig 14: District-level heatmap ──
district_agg = wide2.groupby('district_clean').agg(
    tmc_2021=('tmc_vote_share_2021','mean'),
    bjp_2021=('bjp_vote_share_2021','mean'),
    left_2021=('left_inc_vote_share_2021','mean'),
    tmc_2026=('tmc_vote_share_2026','mean'),
    bjp_2026=('bjp_vote_share_2026','mean'),
    literacy=('literacy_rate','mean'),
    urban=('urban_pct','mean'),
    density=('pop_density','mean'),
).dropna(thresh=4).round(3)

district_agg[['tmc_2021','bjp_2021','left_2021','tmc_2026','bjp_2026']] *= 100

fig, ax = plt.subplots(figsize=(14, max(6, len(district_agg)*0.45)))
plot_cols = ['tmc_2021','bjp_2021','left_2021','tmc_2026','bjp_2026','literacy','urban']
col_labels = ['TMC 2021%','BJP 2021%','Left 2021%','TMC 2026%','BJP 2026%','Literacy%','Urban%']
heat_df = district_agg[plot_cols].copy()
heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min())
sns.heatmap(heat_norm, annot=heat_df.round(1), fmt='.1f', cmap='RdYlGn',
            xticklabels=col_labels, yticklabels=district_agg.index,
            ax=ax, linewidths=0.4, cbar_kws={'label':'Normalised value'})
ax.set_title('District-Level Electoral & Demographic Profile', fontsize=14, fontweight='bold')
ax.set_xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'14_district_heatmap.png'), bbox_inches='tight')
plt.close()
print("Saved 14_district_heatmap.png")

# ── Fig 15: Sex ratio & Density vs Turnout ──
fig, axes = plt.subplots(1,2,figsize=(14,5))
for ax, xcol, xlabel in [(axes[0],'sex_ratio','Sex Ratio (F per 1000 M)'),
                          (axes[1],'pop_density','Population Density (per km²)')]:
    sub = wide2[[xcol,'turnout_percent_2021']].dropna()
    ax.scatter(sub[xcol], sub['turnout_percent_2021'], alpha=0.4, s=25, color='#3B82F6', edgecolors='none')
    z = np.polyfit(sub[xcol], sub['turnout_percent_2021'], 1)
    xline = np.linspace(sub[xcol].min(), sub[xcol].max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), 'r--', lw=1.5)
    corr = sub.corr().iloc[0,1]
    ax.set_xlabel(xlabel); ax.set_ylabel('Turnout 2021 (%)')
    ax.set_title(f'Turnout vs {xlabel}\n(r = {corr:.3f})')
plt.suptitle('Demographic Drivers of Voter Turnout', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'15_turnout_demographics.png'), bbox_inches='tight')
plt.close()
print("Saved 15_turnout_demographics.png")

# ══════════════════════════════════════════════════
# 2. ASSOCIATION ANALYSIS
# ══════════════════════════════════════════════════
print("\n"+"="*60); print("STEP B — ASSOCIATION ANALYSIS"); print("="*60)

# Discretize key features into High/Medium/Low for Apriori
assoc_df = wide2[['tmc_vote_share_2021','bjp_vote_share_2021','left_inc_vote_share_2021',
                   'turnout_percent_2021','victory_margin_percent_2021',
                   'literacy_rate','urban_pct','criminal_cases_2021']].dropna().copy()

def discretize(series, col):
    q33, q67 = series.quantile(0.33), series.quantile(0.67)
    return pd.cut(series, bins=[-np.inf, q33, q67, np.inf],
                  labels=[f'{col}_LOW', f'{col}_MED', f'{col}_HIGH'])

labels_map = {
    'tmc_vote_share_2021':'TMC_VS', 'bjp_vote_share_2021':'BJP_VS',
    'left_inc_vote_share_2021':'LEFT_VS', 'turnout_percent_2021':'TURNOUT',
    'victory_margin_percent_2021':'MARGIN', 'literacy_rate':'LITERACY',
    'urban_pct':'URBAN', 'criminal_cases_2021':'CRIME',
}
transactions = []
for _, row in assoc_df.iterrows():
    items = []
    for col, lbl in labels_map.items():
        q33 = assoc_df[col].quantile(0.33)
        q67 = assoc_df[col].quantile(0.67)
        val = row[col]
        if val <= q33: items.append(f'{lbl}_LOW')
        elif val <= q67: items.append(f'{lbl}_MED')
        else: items.append(f'{lbl}_HIGH')
    transactions.append(items)

te = TransactionEncoder()
te_array = te.fit_transform(transactions)
trans_df = pd.DataFrame(te_array, columns=te.columns_)

freq_items = apriori(trans_df, min_support=0.25, use_colnames=True)
rules = association_rules(freq_items, metric='lift', min_threshold=1.2, num_itemsets=len(freq_items))
rules = rules.sort_values('lift', ascending=False)

print(f"Frequent itemsets: {len(freq_items)} | Rules: {len(rules)}")
print("\nTop 10 rules by lift:")
top10 = rules.head(10)[['antecedents','consequents','support','confidence','lift']]
for _, r in top10.iterrows():
    ant = ', '.join(list(r['antecedents']))
    con = ', '.join(list(r['consequents']))
    print(f"  {ant} → {con}  [sup={r['support']:.2f}, conf={r['confidence']:.2f}, lift={r['lift']:.2f}]")

# ── Fig 16: Association Rules scatter ──
fig, axes = plt.subplots(1,2,figsize=(14,6))

sc = axes[0].scatter(rules['support'], rules['confidence'], c=rules['lift'],
                     cmap='YlOrRd', alpha=0.7, s=50, edgecolors='none')
plt.colorbar(sc, ax=axes[0], label='Lift')
axes[0].set_xlabel('Support'); axes[0].set_ylabel('Confidence')
axes[0].set_title('Association Rules — Support vs Confidence\n(colour = Lift)')
axes[0].axhline(0.6, color='grey', ls='--', lw=1, label='Confidence=0.6')
axes[0].legend(fontsize=9)

# Top rules bar chart
top15 = rules.head(15).copy()
top15['rule'] = [f"{', '.join(list(a))} → {', '.join(list(c))}"
                 for a,c in zip(top15['antecedents'],top15['consequents'])]
top15['rule'] = top15['rule'].str[:50]
axes[1].barh(range(len(top15)), top15['lift'].values, color='#3B82F6', alpha=0.8)
axes[1].set_yticks(range(len(top15)))
axes[1].set_yticklabels(top15['rule'].values, fontsize=7)
axes[1].set_xlabel('Lift'); axes[1].set_title('Top 15 Rules by Lift')
axes[1].axvline(1.0, color='red', ls='--', lw=1)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'16_association_rules.png'), bbox_inches='tight')
plt.close()
print("Saved 16_association_rules.png")

# ── Fig 17: Feature Correlation heatmap (all engineered features) ──
corr_cols = [
    'tmc_vote_share_2021','bjp_vote_share_2021','left_inc_vote_share_2021',
    'turnout_percent_2021','victory_margin_percent_2021','enp_2021',
    'fragmentation_2021','nota_pct_2021','criminal_cases_2021',
    'literacy_rate','urban_pct','sex_ratio','pop_density'
]
avail = [c for c in corr_cols if c in wide2.columns]
corr_mat = wide2[avail].corr().round(2)
clean_labels = [c.replace('_2021','').replace('_vote_share','_VS').replace('_percent','_%')
                .replace('victory_margin','margin').replace('left_inc','LEFT') for c in avail]

fig, ax = plt.subplots(figsize=(12,10))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=clean_labels, yticklabels=clean_labels,
            mask=mask, ax=ax, linewidths=0.3, cbar_kws={'label':'Pearson r'},
            annot_kws={'size':8})
ax.set_title('Feature Correlation Matrix (Electoral + Census 2011)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'17_correlation_heatmap.png'), bbox_inches='tight')
plt.close()
print("Saved 17_correlation_heatmap.png")

print("\n✅ Part 3 complete — 5 new plots saved (13–17).")
print(f"Total plots: {len([f for f in os.listdir(PLOTS) if f.endswith('.png')])}")
