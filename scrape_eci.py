"""
UGDSAI 29 — Group 5
ECI Results Scraper — West Bengal Assembly Elections
Scrapes: results.eci.gov.in
Outputs: data/eci_results.csv
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time, re, os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
os.makedirs(DATA, exist_ok=True)

YEARS = [2016, 2021, 2026]

# ECI URL templates per year (state code 24 = West Bengal)
URL_TEMPLATES = {
    2016: "https://results.eci.gov.in/ResultAcGenMay2016/ConstituencywiseS24{ac}.htm",
    2021: "https://results.eci.gov.in/ResultAcGenMay2021/ConstituencywiseS24{ac}.htm",
    2026: "https://results.eci.gov.in/AcResultGenJune2026/ConstituencywiseS24{ac}.htm",
}

PARTY_ALIASES = {
    'TMC':  ['AITC', 'TMC', 'ALL INDIA TRINAMOOL'],
    'BJP':  ['BJP', 'BHARATIYA JANATA'],
    'LEFT': ['CPM', 'CPI', 'RSP', 'FORWARD BLOC', 'AIFB', 'INC', 'CONGRESS'],
}

def classify_party(party_name):
    p = party_name.upper()
    for key, aliases in PARTY_ALIASES.items():
        if any(a in p for a in aliases):
            return key
    return 'OTHERS'

rows = []
for year in YEARS:
    print(f"\n{'='*50}")
    print(f"Scraping ECI — {year}")
    print(f"{'='*50}")
    url_template = URL_TEMPLATES[year]

    for ac_no in range(1, 295):   # 294 Assembly Constituencies
        url = url_template.format(ac=str(ac_no).zfill(3))
        try:
            r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code != 200:
                print(f"  [{ac_no}] HTTP {r.status_code} — skipping")
                continue

            soup = BeautifulSoup(r.content, 'html.parser')

            # -- Constituency name --
            name_tag = (soup.find('div', {'class': 'cand-name'}) or
                        soup.find('h2') or
                        soup.find('title'))
            const_name = name_tag.text.strip().split('(')[0].strip() if name_tag else f"AC_{ac_no}"

            # -- Candidate table --
            table = soup.find('table', {'class': lambda c: c and 'table' in c})
            if not table:
                continue

            candidates = []
            for tr in table.find_all('tr')[1:]:
                tds = [td.get_text(strip=True) for td in tr.find_all('td')]
                if len(tds) < 5:
                    continue
                try:
                    votes = int(tds[3].replace(',', '').replace(' ', '') or 0)
                except ValueError:
                    votes = 0
                try:
                    vs = float(tds[4].replace('%', '').strip() or 0)
                except ValueError:
                    vs = 0.0
                candidates.append({
                    'candidate': tds[0],
                    'party':     tds[1],
                    'votes':     votes,
                    'vote_share': vs,
                })

            if not candidates:
                continue

            candidates.sort(key=lambda x: x['votes'], reverse=True)
            winner = candidates[0]

            # Aggregate party vote shares
            tmc_vs   = sum(c['vote_share'] for c in candidates if classify_party(c['party']) == 'TMC')
            bjp_vs   = sum(c['vote_share'] for c in candidates if classify_party(c['party']) == 'BJP')
            left_vs  = sum(c['vote_share'] for c in candidates if classify_party(c['party']) == 'LEFT')
            nota_vs  = next((c['vote_share'] for c in candidates if 'NOTA' in c['candidate'].upper()), 0)
            others_vs = max(0, 100 - tmc_vs - bjp_vs - left_vs - nota_vs)
            total_v  = sum(c['votes'] for c in candidates)
            margin   = (candidates[0]['vote_share'] - candidates[1]['vote_share']) if len(candidates) > 1 else 0

            rows.append({
                'year':                      year,
                'constituency':              const_name,
                'winner_party':              classify_party(winner['party']),
                'winner_party_raw':          winner['party'],
                'winner_candidate':          winner['candidate'],
                'tmc_vote_share':            round(tmc_vs, 2),
                'bjp_vote_share':            round(bjp_vs, 2),
                'left_inc_vote_share':       round(left_vs, 2),
                'others_vote_share':         round(others_vs, 2),
                'nota_vote_share':           round(nota_vs, 2),
                'total_votes':               total_v,
                'victory_margin_percent':    round(margin, 2),
                'candidate_count':           len(candidates),
                'turnout_percent':           None,  # filled separately if available
            })
            print(f"  [{ac_no:3d}] {const_name[:30]:30s} → {classify_party(winner['party'])}")

        except Exception as e:
            print(f"  [{ac_no}] ERROR: {e}")

        time.sleep(0.35)   # polite crawl delay

out_path = os.path.join(DATA, 'eci_results.csv')
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\n✅ Saved {len(rows)} rows → {out_path}")
