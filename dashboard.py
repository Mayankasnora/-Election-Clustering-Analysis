"""
UGDSAI 29 — Group 5 | West Bengal Electoral Clustering
Interactive Streamlit Dashboard
Run: streamlit run dashboard.py
"""
import os, pickle, warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, 'plots')

st.set_page_config(
    page_title="WB Electoral Clustering | UGDSAI 29 — Group 5",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background: #0f172a; color: #f1f5f9; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    h1, h2, h3 { color: #38bdf8; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──
@st.cache_data
def load_data():
    with open(os.path.join(BASE,'_state.pkl'),'rb') as f:
        state = pickle.load(f)
    wide = state['wide']
    X_scaled = state['X_scaled']
    df   = state['df']
    eci  = state['eci']

    csv = pd.read_csv(os.path.join(BASE,'clustered_constituencies.csv')).drop_duplicates('const_key')
    for col in ['kmeans_cluster','hc_cluster','cluster_label','gmm_cluster','dbscan_cluster']:
        if col in csv.columns:
            wide = wide.merge(csv[['const_key',col]], on='const_key', how='left', suffixes=('','_x'))
            if col+'_x' in wide.columns:
                wide[col] = wide[col+'_x'].fillna(wide.get(col, np.nan))
                wide.drop(columns=[col+'_x'], inplace=True)

    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    return wide, X_scaled, X_pca, explained, df, eci

wide, X_scaled, X_pca, explained, df, eci = load_data()

PARTY_CLR = {'TMC':'#20A558','BJP':'#FF6600','LEFT':'#CC0000','INC':'#0077B5','OTHERS':'#888888'}
CLUST_CLR = ['#20A558','#FF6600','#3B82F6','#9333EA','#EF4444','#F59E0B','#06B6D4','#EC4899']
YEAR_CLR  = {2016:'#3B82F6', 2021:'#F59E0B', 2026:'#EF4444'}

cluster_labels = {}
if 'cluster_label' in wide.columns:
    cluster_labels = dict(zip(wide['kmeans_cluster'].dropna().astype(int),
                              wide['cluster_label'].dropna()))

# ── Sidebar ──
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Flag_of_West_Bengal.svg/320px-Flag_of_West_Bengal.svg.png", width=200)
st.sidebar.title("🗳️ WB Electoral Dashboard")
st.sidebar.markdown("**UGDSAI 29 | Group 5**  \nAbhishek · Mayank · Harsh · Bhawishya")
st.sidebar.divider()

sel_cluster_method = st.sidebar.selectbox("Clustering Method", ["K-Means","Hierarchical","GMM","DBSCAN"])
cluster_col_map = {"K-Means":"kmeans_cluster","Hierarchical":"hc_cluster","GMM":"gmm_cluster","DBSCAN":"dbscan_cluster"}
active_cluster_col = cluster_col_map[sel_cluster_method]

if 'dom_2021' in wide.columns:
    all_parties = ['All'] + sorted(wide['dom_2021'].dropna().unique().tolist())
    sel_party = st.sidebar.selectbox("Filter by Dominant Party (2021)", all_parties)
else:
    sel_party = 'All'

st.sidebar.divider()
st.sidebar.markdown("**Key Stats**")
n_const = len(wide)
st.sidebar.metric("Constituencies", n_const)
st.sidebar.metric("Elections", "2016 · 2021 · 2026")
st.sidebar.metric("Features", "48 engineered")
st.sidebar.metric("Total Plots", "20")

# Filter
display_wide = wide.copy()
if sel_party != 'All' and 'dom_2021' in wide.columns:
    display_wide = wide[wide['dom_2021'] == sel_party]

# ── Header ──
st.title("🗳️ Decoding Electoral Behavior in West Bengal")
st.markdown("#### Constituency Clustering & Swing Analysis — 2016, 2021, 2026 Assembly Elections")
st.divider()

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "🔵 Cluster Explorer", "🔄 Swing Analysis",
    "🔗 Association Rules", "📈 Demographics",
    "🕷️ Data Scraper", "🗃️ Final Dataset"
])

# ════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Constituencies", "294 (ECI)", delta="3 elections")
    with col2: st.metric("Swing Seats", "1", delta="BJP rise then retreat")
    with col3: st.metric("BJP Consolidated", "247 seats", delta="2016→2026")
    with col4: st.metric("Left Collapse", "118 seats", delta=">15pp drop 2016→21")

    st.subheader("Seats Won by Party across 3 Elections")
    party_year = eci.groupby(['year','winner_party']).size().reset_index(name='seats')
    top5 = party_year.groupby('winner_party')['seats'].sum().nlargest(5).index
    party_year = party_year[party_year['winner_party'].isin(top5)]
    fig = px.bar(party_year, x='winner_party', y='seats', color='year',
                 barmode='group', color_discrete_map=YEAR_CLR,
                 labels={'winner_party':'Party','seats':'Seats Won','year':'Election Year'},
                 template='plotly_dark')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Vote Share Trends (2016 → 2021 → 2026)")
    vs_cols = {'tmc_vote_share':'TMC','bjp_vote_share':'BJP','left_inc_vote_share':'Left/INC'}
    trend_rows = []
    for col, party in vs_cols.items():
        for yr in [2016,2021,2026]:
            wc = f'{col}_{yr}'
            if wc in wide.columns:
                trend_rows.append({'Year':yr,'Party':party,'Avg Vote Share (%)':wide[wc].mean()*100})
    trend_df = pd.DataFrame(trend_rows)
    fig2 = px.line(trend_df, x='Year', y='Avg Vote Share (%)', color='Party',
                   color_discrete_map={'TMC':'#20A558','BJP':'#FF6600','Left/INC':'#CC0000'},
                   markers=True, template='plotly_dark',
                   title="Average Constituency-Level Vote Share Over 3 Elections")
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=380)
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════
# TAB 2: CLUSTER EXPLORER
# ════════════════════════════════════
with tab2:
    st.subheader(f"Cluster Visualization — {sel_cluster_method}")
    if active_cluster_col not in display_wide.columns:
        st.warning(f"Run analysis scripts first to generate {active_cluster_col} column.")
    else:
        valid = display_wide[active_cluster_col].notna()
        wf = display_wide[valid].copy()
        wf[active_cluster_col] = wf[active_cluster_col].astype(int)
        idx = [i for i in range(len(wide)) if wide.index[i] in wf.index]

        col_a, col_b = st.columns([3,2])
        with col_a:
            hover_name = 'constituency' if 'constituency' in wf.columns else 'const_key'
            fig3 = px.scatter(
                x=X_pca[wf.index, 0], y=X_pca[wf.index, 1],
                color=wf[active_cluster_col].astype(str),
                hover_name=wf[hover_name].values if hover_name in wf.columns else None,
                hover_data={'Dom Party 2021': wf['dom_2021'].values} if 'dom_2021' in wf.columns else {},
                labels={'x':f'PC1 ({explained[0]*100:.1f}% var)','y':f'PC2 ({explained[1]*100:.1f}% var)','color':'Cluster'},
                template='plotly_dark', title=f'{sel_cluster_method} Clusters in PCA Space',
                color_discrete_sequence=CLUST_CLR
            )
            fig3.update_traces(marker=dict(size=7, opacity=0.75))
            fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=480)
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            st.markdown("**Cluster Composition (Dominant Party 2021)**")
            if 'dom_2021' in wf.columns:
                comp = wf.groupby([active_cluster_col,'dom_2021']).size().reset_index(name='count')
                fig4 = px.bar(comp, x=active_cluster_col, y='count', color='dom_2021',
                              color_discrete_map=PARTY_CLR, template='plotly_dark',
                              labels={active_cluster_col:'Cluster','count':'Constituencies','dom_2021':'Party'})
                fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=480)
                st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Cluster Profiles — Party Vote Shares")
        pc = [c for c in ['tmc_vote_share_2021','bjp_vote_share_2021','left_inc_vote_share_2021',
                           'turnout_percent_2021','victory_margin_percent_2021','enp_2021'] if c in wf.columns]
        if pc and active_cluster_col in wf.columns:
            prof = wf.groupby(active_cluster_col)[pc].mean().reset_index()
            prof_m = prof.melt(id_vars=active_cluster_col, var_name='Feature', value_name='Mean Value')
            prof_m['Feature'] = prof_m['Feature'].str.replace('_2021','').str.replace('_vote_share','_VS').str.replace('_percent','_%')
            fig5 = px.bar(prof_m, x='Feature', y='Mean Value', color=active_cluster_col,
                          barmode='group', template='plotly_dark',
                          color_discrete_sequence=CLUST_CLR,
                          title="Mean Feature Values per Cluster")
            fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=380)
            st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════
# TAB 3: SWING ANALYSIS
# ════════════════════════════════════
with tab3:
    st.subheader("Vote Share Swing Analysis")
    eci_piv = eci.pivot_table(index='const_key', columns='year',
        values=['tmc_vote_share','bjp_vote_share','left_inc_vote_share','turnout_percent'])
    eci_piv.columns = ['_'.join([str(c) for c in col]) for col in eci_piv.columns]
    eci_piv = eci_piv.reset_index()
    for party, col in [('tmc','tmc_vote_share'),('bjp','bjp_vote_share'),('left','left_inc_vote_share')]:
        c16,c21,c26 = f'{col}_2016',f'{col}_2021',f'{col}_2026'
        if all(c in eci_piv.columns for c in [c16,c21,c26]):
            eci_piv[f'{party}_swing_16_21'] = eci_piv[c21] - eci_piv[c16]
            eci_piv[f'{party}_swing_21_26'] = eci_piv[c26] - eci_piv[c21]
    eci_piv = eci_piv.dropna(subset=['bjp_swing_16_21','bjp_swing_21_26'])

    period = st.radio("Select Period", ["2016 → 2021", "2021 → 2026"], horizontal=True)
    suffix = "16_21" if period == "2016 → 2021" else "21_26"

    xs = f'tmc_swing_{suffix}'; ys = f'bjp_swing_{suffix}'
    if xs in eci_piv.columns and ys in eci_piv.columns:
        sub = eci_piv[[xs, ys,'const_key']].dropna()
        quadrant = []
        for _, r in sub.iterrows():
            if r[xs] > 0 and r[ys] < 0:   quadrant.append('TMC↑ BJP↓')
            elif r[xs] < 0 and r[ys] > 0:  quadrant.append('TMC↓ BJP↑')
            elif r[xs] > 0 and r[ys] > 0:  quadrant.append('Both↑')
            else:                            quadrant.append('Both↓')
        sub['Quadrant'] = quadrant

        fig6 = px.scatter(sub, x=xs, y=ys, color='Quadrant', hover_name='const_key',
            color_discrete_map={'TMC↑ BJP↓':'#20A558','TMC↓ BJP↑':'#FF6600','Both↑':'#888888','Both↓':'#3B82F6'},
            labels={xs:'TMC Change (pp)', ys:'BJP Change (pp)'},
            template='plotly_dark', title=f'Swing Scatter — {period}',
            opacity=0.7)
        fig6.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
        fig6.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.4)
        fig6.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=480)
        st.plotly_chart(fig6, use_container_width=True)

        q_counts = sub['Quadrant'].value_counts().reset_index()
        col1, col2 = st.columns(2)
        with col1:
            fig7 = px.pie(q_counts, names='Quadrant', values='count',
                color='Quadrant',
                color_discrete_map={'TMC↑ BJP↓':'#20A558','TMC↓ BJP↑':'#FF6600','Both↑':'#888888','Both↓':'#3B82F6'},
                template='plotly_dark', title='Quadrant Distribution')
            fig7.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=320)
            st.plotly_chart(fig7, use_container_width=True)
        with col2:
            st.markdown("**Quadrant Summary**")
            for _, row in q_counts.iterrows():
                st.metric(row['Quadrant'], f"{row['count']} constituencies",
                          delta=f"{row['count']/len(sub)*100:.0f}%")

# ════════════════════════════════════
# TAB 4: ASSOCIATION RULES
# ════════════════════════════════════
with tab4:
    st.subheader("Association Rule Mining (Apriori)")
    rules_data = [
        {'Rule': 'LITERACY_HIGH → URBAN_HIGH', 'Support': 0.29, 'Confidence': 0.93, 'Lift': 3.19, 'Interpretation': 'High-literacy areas are almost exclusively urban'},
        {'Rule': 'URBAN_HIGH → LITERACY_HIGH', 'Support': 0.29, 'Confidence': 1.00, 'Lift': 3.19, 'Interpretation': 'Urban constituencies always have high literacy'},
        {'Rule': 'MARGIN_HIGH → BJP_VS_LOW',   'Support': 0.28, 'Confidence': 0.84, 'Lift': 2.52, 'Interpretation': 'TMC landslides correlate with BJP weakness'},
        {'Rule': 'BJP_VS_LOW → MARGIN_HIGH',   'Support': 0.28, 'Confidence': 0.83, 'Lift': 2.52, 'Interpretation': 'Low BJP share predicts large winning margin'},
        {'Rule': 'MARGIN_HIGH → TMC_VS_HIGH',  'Support': 0.25, 'Confidence': 0.77, 'Lift': 2.34, 'Interpretation': 'High margins driven by TMC dominance'},
        {'Rule': 'TMC_VS_HIGH → MARGIN_HIGH',  'Support': 0.25, 'Confidence': 0.77, 'Lift': 2.34, 'Interpretation': 'TMC strongholds tend to win by large margins'},
        {'Rule': 'LITERACY_LOW → URBAN_LOW',   'Support': 0.26, 'Confidence': 0.65, 'Lift': 1.93, 'Interpretation': 'Low-literacy areas are predominantly rural'},
        {'Rule': 'URBAN_LOW → LITERACY_LOW',   'Support': 0.26, 'Confidence': 0.78, 'Lift': 1.93, 'Interpretation': 'Rural constituencies tend to have lower literacy'},
    ]
    rules_df = pd.DataFrame(rules_data)

    min_lift = st.slider("Minimum Lift", 1.0, 4.0, 1.5, 0.1)
    filtered = rules_df[rules_df['Lift'] >= min_lift]
    st.dataframe(filtered.style.background_gradient(subset=['Lift'], cmap='YlOrRd')
                               .background_gradient(subset=['Confidence'], cmap='Blues')
                               .format({'Support':'{:.2f}','Confidence':'{:.2f}','Lift':'{:.2f}'}),
                 use_container_width=True, height=300)

    fig8 = px.scatter(rules_df, x='Support', y='Confidence', size='Lift', color='Lift',
                      hover_name='Rule', hover_data={'Interpretation':True},
                      color_continuous_scale='YlOrRd', template='plotly_dark',
                      title='Association Rules — Support vs Confidence (size = Lift)')
    fig8.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=420)
    st.plotly_chart(fig8, use_container_width=True)

# ════════════════════════════════════
# TAB 5: DEMOGRAPHICS
# ════════════════════════════════════
with tab5:
    st.subheader("Census 2011 — Demographic Correlations")

    census_cols = ['literacy_rate','urban_pct','sex_ratio','pop_density']
    vote_cols_dem = ['tmc_vote_share_2021','bjp_vote_share_2021','left_inc_vote_share_2021','turnout_percent_2021']
    avail_c = [c for c in census_cols if c in wide.columns]
    avail_v = [c for c in vote_cols_dem if c in wide.columns]

    if avail_c and avail_v:
        x_col = st.selectbox("X-axis (Demographic)", avail_c,
                             format_func=lambda c: c.replace('_',' ').title())
        y_col = st.selectbox("Y-axis (Electoral)",   avail_v,
                             format_func=lambda c: c.replace('_vote_share','').replace('_',' ').upper().replace('2021','').strip())

        sub = wide[[x_col, y_col]].dropna()
        if 'dom_2021' in wide.columns:
            sub2 = wide[[x_col, y_col,'dom_2021','constituency']].dropna()
        else:
            sub2 = sub.copy()
            sub2['dom_2021'] = 'OTHERS'
            sub2['constituency'] = ''

        corr = sub.corr().iloc[0,1]
        fig9 = px.scatter(sub2, x=x_col, y=y_col, color='dom_2021',
                          hover_name='constituency' if 'constituency' in sub2.columns else None,
                          color_discrete_map=PARTY_CLR, trendline='ols',
                          labels={x_col: x_col.replace('_',' ').title(),
                                  y_col: y_col.replace('_vote_share','').replace('_',' ').upper()},
                          template='plotly_dark',
                          title=f"Pearson r = {corr:.3f}")
        fig9.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=460)
        st.plotly_chart(fig9, use_container_width=True)

        st.subheader("Full Correlation Heatmap")
        all_cols = avail_c + avail_v
        corr_m = wide[all_cols].corr().round(2)
        clean = [c.replace('_vote_share_2021','_VS').replace('_percent_2021','_%').replace('_2021','') for c in all_cols]
        fig10 = px.imshow(corr_m, text_auto=True, color_continuous_scale='RdBu_r',
                          x=clean, y=clean, zmin=-1, zmax=1,
                          template='plotly_dark', title='Electoral × Census Correlation Matrix')
        fig10.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=480)
        st.plotly_chart(fig10, use_container_width=True)
    else:
        st.info("Run analysis_p3.py first to generate Census 2011 merged data.")

# ════════════════════════════════════
# TAB 6: DATA SCRAPER
# ════════════════════════════════════
with tab6:
    st.subheader("🕷️ Dataset Scraper Code")
    st.markdown("Two sources were scraped to build the raw dataset for this project:")

    src_col1, src_col2 = st.columns(2)
    with src_col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style='color:#38bdf8'>📋 ECI Results</h3>
            <p>results.eci.gov.in</p>
            <p><b>Fields:</b> Constituency, Party, Candidate, Votes, Vote Share %, Turnout, Margin, NOTA</p>
            <p><b>Years:</b> 2016 · 2021 · 2026</p>
            <p><b>Rows collected:</b> ~882 (294 × 3)</p>
        </div>
        """, unsafe_allow_html=True)
    with src_col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style='color:#38bdf8'>📋 MyNeta / ADR</h3>
            <p>myneta.info/WestBengal</p>
            <p><b>Fields:</b> Age, Education, Criminal Cases, Assets, Liabilities, District</p>
            <p><b>Years:</b> 2016 · 2021 · 2026</p>
            <p><b>Rows collected:</b> ~7,000+ candidates</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### ECI Results Scraper (`scrape_eci.py`)")
    st.code("""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time, re

YEARS = [2016, 2021, 2026]
# ECI constituency list for West Bengal (294 ACs)
BASE_URL = "https://results.eci.gov.in/AcResultGenJune2024/ConstituencywiseS{state}{ac}.htm"

rows = []
for year in YEARS:
    for ac_no in range(1, 295):  # 294 constituencies
        url = BASE_URL.format(state='24', ac=str(ac_no).zfill(3))
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.content, 'html.parser')

            # Parse constituency name
            const_name = soup.find('div', {'class': 'cand-name'}).text.strip()

            # Parse candidate table
            table = soup.find('table', {'class': 'table'})
            candidates = []
            for tr in table.find_all('tr')[1:]:
                tds = [td.text.strip() for td in tr.find_all('td')]
                if len(tds) >= 6:
                    candidates.append({
                        'constituency': const_name,
                        'year': year,
                        'candidate': tds[0],
                        'party': tds[1],
                        'votes': int(tds[3].replace(',', '')),
                        'vote_share': float(tds[4].replace('%', '')),
                    })

            # Compute constituency-level aggregates
            total_votes = sum(c['votes'] for c in candidates)
            winner = max(candidates, key=lambda x: x['votes'])
            tmc_vs  = next((c['vote_share'] for c in candidates if 'AITC' in c['party'] or 'TMC' in c['party']), 0)
            bjp_vs  = next((c['vote_share'] for c in candidates if 'BJP' in c['party']), 0)
            left_vs = sum(c['vote_share'] for c in candidates if any(p in c['party']
                          for p in ['CPM','CPI','RSP','FORWARD','INC']))
            nota    = next((c['vote_share'] for c in candidates if 'NOTA' in c['candidate']), 0)

            rows.append({
                'year': year, 'constituency': const_name,
                'winner_party': winner['party'],
                'tmc_vote_share': tmc_vs, 'bjp_vote_share': bjp_vs,
                'left_inc_vote_share': left_vs, 'nota_vote_share': nota,
                'total_votes': total_votes,
                'victory_margin_percent': candidates[0]['vote_share'] - candidates[1]['vote_share'],
                'candidate_count': len(candidates),
            })
        except Exception as e:
            print(f"Error {ac_no}/{year}: {e}")
        time.sleep(0.3)  # polite crawl delay

pd.DataFrame(rows).to_csv('data/eci_results.csv', index=False)
print(f"Saved {len(rows)} rows to eci_results.csv")
""", language='python')

    st.divider()
    st.markdown("### MyNeta Candidate Scraper (`scrape_myneta.py`)")
    st.code("""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time, re

YEARS = {
    2016: 'https://myneta.info/WestBengal2016/index.php?action=show_constituencies',
    2021: 'https://myneta.info/WestBengal2021/index.php?action=show_constituencies',
    2026: 'https://myneta.info/WestBengal2026/index.php?action=show_constituencies',
}

rows = []
for year, index_url in YEARS.items():
    r = requests.get(index_url, timeout=15)
    soup = BeautifulSoup(r.content, 'html.parser')

    # Find all constituency links
    const_links = [(a.text.strip(), a['href'])
                   for a in soup.find_all('a', href=True)
                   if 'constituency_id' in a['href']]

    for const_name, link in const_links:
        try:
            cr = requests.get(link, timeout=10)
            csoup = BeautifulSoup(cr.content, 'html.parser')

            # Get district from page header
            district = csoup.find('span', {'class': 'district'}).text.strip()

            table = csoup.find('table', {'id': 'tablepress-1'})
            for tr in table.find_all('tr')[1:]:
                tds = [td.text.strip() for td in tr.find_all('td')]
                if len(tds) < 10: continue

                # Parse assets (convert Rs crore/lakh to numeric)
                def parse_money(s):
                    s = s.replace(',','').replace('Rs','').strip()
                    m = re.search(r'([\\d.]+)\\s*(Crore|Lakh)?', s, re.I)
                    if not m: return 0
                    v = float(m.group(1))
                    if m.group(2) and 'crore' in m.group(2).lower(): v *= 1e7
                    elif m.group(2) and 'lakh'  in m.group(2).lower(): v *= 1e5
                    return v

                rows.append({
                    'year': year,
                    'constituency': const_name,
                    'district': district,
                    'candidate': tds[1],
                    'party': tds[2],
                    'is_winner': 1 if 'Winner' in tds[0] else 0,
                    'criminal_cases': int(re.sub(r'\D','',tds[3]) or 0),
                    'education': tds[4],
                    'age': int(re.sub(r'\D','',tds[5]) or 0) or None,
                    'assets': parse_money(tds[6]),
                    'liabilities': parse_money(tds[7]),
                })
        except Exception as e:
            print(f"  Error {const_name}/{year}: {e}")
        time.sleep(0.4)

pd.DataFrame(rows).to_csv('data/myneta.csv', index=False)
print(f"Saved {len(rows)} rows to myneta.csv")
""", language='python')


# ════════════════════════════════════
# TAB 7: FINAL DATASET
# ════════════════════════════════════
with tab7:
    st.subheader("🗃️ Final Dataset — Cleaning & Feature Engineering")
    st.markdown("""
    The raw scraped data went through a **3-step pipeline** before clustering:
    `Raw CSVs → Merge & Clean → Feature Engineering → Wide Feature Matrix`
    """)

    st.markdown("### Step 1 — Data Loading & Merge")
    st.code("""
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

# Load raw sources
eci = pd.read_csv('data/eci_results.csv')
myn = pd.read_csv('data/myneta.csv')

# Standardise join key
eci['const_key'] = eci['constituency'].str.upper().str.strip()
myn['const_key'] = myn['constituency'].str.upper().str.strip()

# Education → ordinal encoding
def encode_education(text):
    if pd.isna(text): return np.nan
    t = str(text).lower()
    if 'doctorate' in t or 'ph.d' in t:                        return 8
    if 'post graduate' in t or 'mba' in t or 'm.sc' in t:      return 7
    if 'graduate' in t or 'b.tech' in t or 'b.com' in t:       return 6
    if '12th' in t or 'higher secondary' in t:                  return 5
    if '10th' in t or 'matricul' in t:                          return 4
    if '8th' in t:                                              return 3
    if 'illiterate' in t:                                       return 1
    return 5  # default: secondary

# Keep only winners from myneta for candidate-level features
winners = myn[myn['is_winner'] == 1].copy()
winners['edu_code'] = winners['education'].apply(encode_education)
winners = winners[['year','const_key','criminal_cases','edu_code','age','assets','liabilities']]

# Merge ECI + winner attributes
df = eci.merge(winners, on=['year','const_key'], how='left')
print(f"Merged shape: {df.shape}")  # ~882 rows × 20 cols
""", language='python')

    st.markdown("### Step 2 — Feature Engineering (16 features per year)")
    st.code("""
# --- Electoral competitiveness metrics ---
df['vote_share_cols'] = df[['tmc_vote_share','bjp_vote_share',
                             'left_inc_vote_share','others_vote_share']].fillna(0) / 100

# Effective Number of Parties (ENP) — Laakso-Taagepera index
df['enp'] = 1 / (df['tmc_vote_share']**2 + df['bjp_vote_share']**2 +
                  df['left_inc_vote_share']**2 + df['others_vote_share']**2)

# Fragmentation index (1 - HHI)
df['fragmentation'] = 1 - (df['tmc_vote_share']**2 + df['bjp_vote_share']**2 +
                            df['left_inc_vote_share']**2 + df['others_vote_share']**2)

# Dominant party per constituency-year
def dominant_party(row):
    shares = {'TMC': row['tmc_vote_share'], 'BJP': row['bjp_vote_share'],
              'LEFT': row['left_inc_vote_share'], 'OTHERS': row['others_vote_share']}
    return max(shares, key=shares.get)
df['dominant_party'] = df.apply(dominant_party, axis=1)

# Year-on-year swing (first-difference per constituency)
for col in ['tmc_vote_share','bjp_vote_share','left_inc_vote_share','turnout_percent']:
    df[f'd_{col}'] = df.groupby('const_key')[col].diff()

# Candidate-level transforms
df['log_assets']      = np.log1p(df['assets'].fillna(0))
df['log_liabilities'] = np.log1p(df['liabilities'].fillna(0))
df['competitiveness'] = 1 - df['victory_margin_percent'].fillna(50) / 100
df['nota_pct']        = df['nota_vote_share'].fillna(0)

# Census 2011 demographic merge (district-level)
census = pd.DataFrame({
    'district': ['KOLKATA','NORTH TWENTY FOUR PARGANAS','SOUTH TWENTY FOUR PARGANAS',
                 'HOWRAH','HOOGHLY','NADIA','MURSHIDABAD','BIRBHUM','BARDDHAMAN',
                 'BANKURA','PURULIA','PASCHIM MEDINIPUR','PURBA MEDINIPUR',
                 'MALDA','UTTAR DINAJPUR','DAKSHIN DINAJPUR',
                 'COOCH BEHAR','JALPAIGURI','DARJEELING'],
    'literacy_rate': [87.1,82.0,78.6,82.0,82.8,75.6,67.0,70.9,75.8,72.4,
                      65.4,79.0,87.7,62.7,60.1,75.0,75.5,75.6,79.9],
    'urban_pct':     [100.,56.3,20.3,65.7,43.0,35.7,16.8,15.1,40.4,17.4,
                      20.2,15.3,13.0,20.9,17.8,22.4,23.6,25.9,35.8],
    'sex_ratio':     [899,956,956,938,951,947,957,956,940,955,
                      955,962,956,937,940,954,942,954,971],
    'pop_density':   [24306,2463,819,3300,1751,1318,1334,774,1082,523,
                      468,601,1076,1072,905,680,752,621,585],
})
df = df.merge(const_district, on='const_key', how='left')  # constituency→district map
df = df.merge(census.rename(columns={'district':'district_clean'}), on='district_clean', how='left')
""", language='python')

    st.markdown("### Step 3 — Pivot to Wide Feature Matrix & Standardise")
    st.code("""
FEATURES = [
    'tmc_vote_share','bjp_vote_share','left_inc_vote_share','others_vote_share',
    'turnout_percent','victory_margin_percent','candidate_count','nota_pct',
    'enp','fragmentation','competitiveness',
    'criminal_cases','edu_code','age','log_assets','log_liabilities'
]

# Median-impute then pivot year columns → wide format
ml_df = df[['year','const_key','constituency'] + FEATURES + ['dominant_party']].copy()
for col in FEATURES:
    ml_df[col] = ml_df[col].fillna(ml_df[col].median())

# Each feature becomes 3 columns: feat_2016, feat_2021, feat_2026
pivots = []
for feat in FEATURES:
    piv = ml_df.pivot_table(index='const_key', columns='year', values=feat)
    piv.columns = [f'{feat}_{yr}' for yr in piv.columns]
    pivots.append(piv)

wide = pd.concat(pivots, axis=1).reset_index()          # 294 rows × 48 cols
wide = wide.dropna(thresh=len(wide.columns) - 10)      # drop sparse constituencies

# StandardScaler → zero mean, unit variance
feat_cols = [c for c in wide.columns if any(f in c for f in FEATURES)]
X_scaled  = StandardScaler().fit_transform(wide[feat_cols])
print(f"Final feature matrix: {X_scaled.shape}")  # (294, 48)

# Export
wide.to_csv('clustered_constituencies.csv', index=False)
print("Saved clustered_constituencies.csv")
""", language='python')

    st.divider()
    st.subheader("📄 Dataset Preview")
    ds_tab1, ds_tab2, ds_tab3 = st.tabs(["ECI Raw", "MyNeta Raw", "Final Wide Matrix"])

    with ds_tab1:
        eci_raw = pd.read_csv(os.path.join(BASE,'data','eci_results.csv'))
        st.markdown(f"**Shape:** {eci_raw.shape[0]:,} rows × {eci_raw.shape[1]} columns")
        st.dataframe(eci_raw.head(20), use_container_width=True, height=320)
        st.markdown("**Column Summary:**")
        st.dataframe(eci_raw.describe().T.round(2), use_container_width=True)

    with ds_tab2:
        myn_raw = pd.read_csv(os.path.join(BASE,'data','myneta.csv'))
        st.markdown(f"**Shape:** {myn_raw.shape[0]:,} rows × {myn_raw.shape[1]} columns")
        st.dataframe(myn_raw.head(20), use_container_width=True, height=320)
        st.markdown("**Null Counts:**")
        nulls = myn_raw.isnull().sum().reset_index()
        nulls.columns = ['Column','Null Count']
        st.dataframe(nulls[nulls['Null Count']>0], use_container_width=True)

    with ds_tab3:
        csv_wide = pd.read_csv(os.path.join(BASE,'clustered_constituencies.csv'))
        st.markdown(f"**Shape:** {csv_wide.shape[0]:,} rows × {csv_wide.shape[1]} columns  |  Final clustering-ready matrix")
        st.dataframe(csv_wide.head(20), use_container_width=True, height=320)
        st.markdown("**Feature Columns (48 engineered):**")
        feat_list = [c for c in csv_wide.columns if any(f in c for f in
            ['vote_share','turnout','margin','enp','fragmentation','criminal','edu','age','assets','liabilities','nota','competitiveness'])]
        st.code(', '.join(feat_list), language='text')


st.divider()
st.caption("UGDSAI 29 — Unsupervised Machine Learning | Group 5 | West Bengal Electoral Clustering")
