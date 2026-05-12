"""
UGDSAI 29 — Group 5
MyNeta / ADR Candidate Scraper — West Bengal Assembly Elections
Scrapes: myneta.info
Outputs: data/myneta.csv
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time, re, os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
os.makedirs(DATA, exist_ok=True)

YEARS = {
    2016: 'https://myneta.info/WestBengal2016/index.php?action=show_constituencies',
    2021: 'https://myneta.info/WestBengal2021/index.php?action=show_constituencies',
    2026: 'https://myneta.info/WestBengal2026/index.php?action=show_constituencies',
}

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def parse_money(s):
    """Convert 'Rs 1,23,456' / '2 Crore' / '50 Lakh' strings to a float (INR)."""
    if not s:
        return 0.0
    s = str(s).replace(',', '').replace('Rs', '').replace('rs', '').strip()
    m = re.search(r'([\d.]+)\s*(Crore|Lakh)?', s, re.IGNORECASE)
    if not m:
        return 0.0
    v = float(m.group(1))
    unit = (m.group(2) or '').lower()
    if unit == 'crore':
        v *= 1_00_00_000
    elif unit == 'lakh':
        v *= 1_00_000
    return round(v, 2)

def parse_int(s):
    """Extract first integer from a string, return 0 if none found."""
    digits = re.sub(r'\D', '', str(s))
    return int(digits) if digits else 0

rows = []

for year, index_url in YEARS.items():
    print(f"\n{'='*50}")
    print(f"Scraping MyNeta — {year}")
    print(f"{'='*50}")

    try:
        r = requests.get(index_url, timeout=15, headers=HEADERS)
        soup = BeautifulSoup(r.content, 'html.parser')
    except Exception as e:
        print(f"  Failed to load index: {e}")
        continue

    # Collect all constituency page links
    const_links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'constituency_id' in href or 'constituency.php' in href:
            full_url = href if href.startswith('http') else f"https://myneta.info{href}"
            const_links.append((a.get_text(strip=True), full_url))

    # De-duplicate
    seen = set()
    unique_links = []
    for name, url in const_links:
        if url not in seen:
            seen.add(url)
            unique_links.append((name, url))

    print(f"  Found {len(unique_links)} constituencies")

    for idx, (const_name, link) in enumerate(unique_links, 1):
        try:
            cr = requests.get(link, timeout=15, headers=HEADERS)
            csoup = BeautifulSoup(cr.content, 'html.parser')

            # District name
            district_tag = (csoup.find('span', {'class': 'district'}) or
                            csoup.find('td', string=re.compile('District', re.I)))
            district = district_tag.get_text(strip=True).replace('District:', '').strip() if district_tag else ''

            # Candidate table (MyNeta uses tablepress or similar)
            table = (csoup.find('table', {'id': 'tablepress-1'}) or
                     csoup.find('table', {'class': lambda c: c and 'candidate' in str(c).lower()}) or
                     csoup.find('table'))

            if not table:
                continue

            for tr in table.find_all('tr')[1:]:
                tds = [td.get_text(separator=' ', strip=True) for td in tr.find_all('td')]
                if len(tds) < 6:
                    continue

                # Typical MyNeta column order:
                # 0:Sno/Winner, 1:Candidate, 2:Party, 3:Criminal, 4:Education, 5:Age, 6:Assets, 7:Liabilities
                is_winner = 1 if ('winner' in tds[0].lower() or '✓' in tds[0] or 'W' == tds[0].strip()) else 0

                rows.append({
                    'year':            year,
                    'constituency':    const_name,
                    'district':        district,
                    'candidate':       tds[1] if len(tds) > 1 else '',
                    'party':           tds[2] if len(tds) > 2 else '',
                    'is_winner':       is_winner,
                    'criminal_cases':  parse_int(tds[3]) if len(tds) > 3 else 0,
                    'education':       tds[4]             if len(tds) > 4 else '',
                    'age':             parse_int(tds[5])  if len(tds) > 5 else None,
                    'assets':          parse_money(tds[6]) if len(tds) > 6 else 0.0,
                    'liabilities':     parse_money(tds[7]) if len(tds) > 7 else 0.0,
                })

            print(f"  [{idx:3d}/{len(unique_links)}] {const_name[:35]:35s} — district: {district}")

        except Exception as e:
            print(f"  [{idx}] ERROR {const_name}: {e}")

        time.sleep(0.4)   # polite crawl delay

out_path = os.path.join(DATA, 'myneta.csv')
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\n✅ Saved {len(rows)} rows → {out_path}")
