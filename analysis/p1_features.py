"""
UGDSAI 29 — Unsupervised Machine Learning
Group 5: Decoding Electoral Behavior in West Bengal
Part 1: Data Loading, Feature Engineering, Feature Matrix
"""
import os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA  = os.path.join(BASE, 'data', 'raw')
PROC  = os.path.join(BASE, 'data', 'processed')
PLOTS = os.path.join(BASE, 'plots')
os.makedirs(PROC, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'axes.titlesize': 13, 'axes.labelsize': 11,
})

PARTY_COLORS = {'TMC':'#20A558','BJP':'#FF6600','LEFT':'#CC0000','INC':'#0077B5','OTHERS':'#888888'}
YEAR_COLORS  = {2016:'#3B82F6', 2021:'#F59E0B', 2026:'#EF4444'}

print("="*60); print("STEP 1 — DATA LOADING"); print("="*60)

eci = pd.read_csv(os.path.join(DATA, 'eci_results.csv'))
myn = pd.read_csv(os.path.join(DATA, 'myneta.csv'))
eci['const_key'] = eci['constituency'].str.upper().str.strip()
myn['const_key'] = myn['constituency'].str.upper().str.strip()
print(f"ECI rows: {len(eci):,}  |  Myneta rows: {len(myn):,}")
print(f"Constituencies: {eci['const_key'].nunique()}  |  Years: {sorted(eci['year'].unique())}")

def encode_education(text):
    if pd.isna(text): return np.nan
    t = str(text).lower()
    if 'doctorate' in t or 'ph.d' in t: return 8
    if 'post graduate' in t or 'postgrad' in t or ' mba' in t or ' m.a' in t or ' m.sc' in t: return 7
    if 'graduate professional' in t: return 7
    if 'graduate' in t or ' b.a' in t or ' b.sc' in t or ' b.com' in t or 'b.tech' in t: return 6
    if '12th' in t or 'higher secondary' in t or 'h.s.' in t: return 5
    if '10th' in t or 'matricul' in t or 'secondary' in t: return 4
    if '8th' in t: return 3
    if '5th' in t: return 2
    if 'illiterate' in t: return 1
    return 5

winners = myn[myn['is_winner'] == 1].copy()
winners['edu_code'] = winners['education'].apply(encode_education)
winners = winners[['year','const_key','criminal_cases','edu_code','age','assets','liabilities']]

df = eci.merge(winners, on=['year','const_key'], how='left')
print(f"Merged shape: {df.shape}")

print("\n"+"="*60); print("STEP 2 — FEATURE ENGINEERING"); print("="*60)

df = df.sort_values(['const_key','year']).reset_index(drop=True)
for col in ['tmc_vote_share','bjp_vote_share','left_inc_vote_share','others_vote_share']:
    df[col] = df[col].fillna(0) / 100

df['enp'] = 1 / (df['tmc_vote_share']**2 + df['bjp_vote_share']**2 +
                  df['left_inc_vote_share']**2 + df['others_vote_share']**2).replace(0, np.nan)
df['fragmentation'] = 1 - (df['tmc_vote_share']**2 + df['bjp_vote_share']**2 +
                            df['left_inc_vote_share']**2 + df['others_vote_share']**2)

def dominant_party(row):
    shares = {'TMC':row['tmc_vote_share'],'BJP':row['bjp_vote_share'],
              'LEFT':row['left_inc_vote_share'],'OTHERS':row['others_vote_share']}
    return max(shares, key=shares.get)

df['dominant_party'] = df.apply(dominant_party, axis=1)
df = df.sort_values(['const_key','year'])
for col in ['tmc_vote_share','bjp_vote_share','left_inc_vote_share','turnout_percent']:
    df[f'd_{col}'] = df.groupby('const_key')[col].diff()

df['log_assets']      = np.log1p(df['assets'].fillna(0))
df['log_liabilities'] = np.log1p(df['liabilities'].fillna(0))
df['competitiveness'] = 1 - df['victory_margin_percent'].fillna(50) / 100
df['nota_pct']        = df['nota_vote_share'].fillna(0)

print("\n"+"="*60); print("STEP 3 — FEATURE MATRIX"); print("="*60)

FEATURES = ['tmc_vote_share','bjp_vote_share','left_inc_vote_share','others_vote_share',
            'turnout_percent','victory_margin_percent','candidate_count','nota_pct',
            'enp','fragmentation','competitiveness','criminal_cases','edu_code','age',
            'log_assets','log_liabilities']

ml_df = df[['year','const_key','constituency'] + FEATURES + ['dominant_party']].copy()
for col in FEATURES:
    if ml_df[col].dtype in [np.float64, np.int64]:
        ml_df[col] = ml_df[col].fillna(ml_df[col].median())

pivots = []
for feat in FEATURES:
    piv = ml_df.pivot_table(index='const_key', columns='year', values=feat)
    piv.columns = [f'{feat}_{yr}' for yr in piv.columns]
    pivots.append(piv)

wide = pd.concat(pivots, axis=1).reset_index()
wide = wide.merge(df[['const_key','constituency']].drop_duplicates(), on='const_key')
dom21 = df[df['year']==2021][['const_key','dominant_party']].rename(columns={'dominant_party':'dom_2021'})
dom26 = df[df['year']==2026][['const_key','dominant_party']].rename(columns={'dominant_party':'dom_2026'})
wide = wide.merge(dom21, on='const_key', how='left').merge(dom26, on='const_key', how='left')
wide = wide.dropna(thresh=len(wide.columns)-10)

feat_cols = [c for c in wide.columns if any(f in c for f in FEATURES)]
X_raw     = wide[feat_cols].fillna(wide[feat_cols].median())
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_raw)
print(f"Feature matrix: {X_raw.shape}")

# Save state for part 2
with open(os.path.join(PROC, '_state.pkl'), 'wb') as f:
    pickle.dump({'wide': wide, 'X_scaled': X_scaled, 'feat_cols': feat_cols,
                 'df': df, 'eci': eci, 'FEATURES': FEATURES,
                 'PARTY_COLORS': PARTY_COLORS, 'YEAR_COLORS': YEAR_COLORS}, f)
print("Part 1 complete — state saved.")
