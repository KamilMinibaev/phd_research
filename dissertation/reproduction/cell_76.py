"""Reproduction of source notebook cell 76 (zero-based).

Historical specification, including its original sample and merge behaviour.
Use run_analysis.py for the documented 1997-2015 specification.
Only paths, import behaviour and the main guard are adapted here.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
"""
ONE-CELL SCRIPT (Jupyter-friendly): build QAROT minimal panel (if needed),
merge with your main panel (final_data.csv), run STACKED event-study via PanelOLS,
save point-estimates CSVs + plots + summary.csv.

Required:
  pip install pandas numpy matplotlib linearmodels openpyxl

Files expected (adjust paths below if needed):
  - ../data/final_data/final_data.csv
  - ../data/qarot_data/qarot_data.csv   (raw QAROT)
    OR ../data/qarot_data/qarot_panel_min.csv (already built)
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
MAIN_PANEL_CSV = str(ROOT / 'data/processed/country_panel.csv')
QAROT_RAW_CSV = str(ROOT / 'data/raw/qarot.csv')
QAROT_MIN_CSV = str(ROOT / 'data/processed/qarot_minimal.csv')
MERGED_DIR = str(ROOT / 'results/reproduction/merged')
MERGED_CSV = os.path.join(MERGED_DIR, 'merged_panel.csv')
OUTDIR = str(ROOT / 'results/reproduction/cell-76')
COHORT_MIN, COHORT_MAX = (2002, 2010)
W_PRE, W_POST = (5, 12)
BASE_K = -1
CONTROL_COLS = ['GDP', 'Political_Stability', 'Corruption', 'Fertility']

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    m = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_main_panel(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=';', engine='python', decimal=',')

def normalize_country_year(df: pd.DataFrame) -> pd.DataFrame:
    if 'Country_Code' not in df.columns and 'countrycode' in df.columns:
        df = df.rename(columns={'countrycode': 'Country_Code'})
    if 'Year' not in df.columns and 'year' in df.columns:
        df = df.rename(columns={'year': 'Year'})
    if 'Country_Code' not in df.columns or 'Year' not in df.columns:
        raise RuntimeError('Need Country_Code and Year columns after normalization.')
    df['Country_Code'] = df['Country_Code'].astype(str)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    return df

def build_qarot_min_panel_if_needed() -> pd.DataFrame:
    """
    Produce minimal QAROT panel with:
      Country_Code, Year, treat_year, PARL
    From raw QAROT (common columns: code, year, women.rep, implemented.quota / first.implement.flag).
    """
    if os.path.exists(QAROT_MIN_CSV):
        dfq = pd.read_csv(QAROT_MIN_CSV, engine='python')
        dfq = normalize_country_year(dfq)
        if 'treat_year' not in dfq.columns or 'PARL' not in dfq.columns:
            raise RuntimeError('QAROT_MIN_CSV exists but must contain treat_year and PARL.')
        dfq['treat_year'] = pd.to_numeric(dfq['treat_year'], errors='coerce').astype('Int64')
        dfq['PARL'] = pd.to_numeric(dfq['PARL'], errors='coerce')
        return dfq[['Country_Code', 'Year', 'treat_year', 'PARL']].copy()
    if not os.path.exists(QAROT_RAW_CSV):
        raise FileNotFoundError(f'Neither {QAROT_MIN_CSV} nor {QAROT_RAW_CSV} exists.')
    df = pd.read_csv(QAROT_RAW_CSV, engine='python')
    if 'code' not in df.columns or 'year' not in df.columns:
        raise RuntimeError('Raw QAROT must contain columns: code, year (at least).')
    df = df.rename(columns={'code': 'Country_Code', 'year': 'Year'})
    df = normalize_country_year(df)
    if 'women.rep' in df.columns and 'PARL' not in df.columns:
        df['PARL'] = pd.to_numeric(df['women.rep'], errors='coerce')
    elif 'PARL' in df.columns:
        df['PARL'] = pd.to_numeric(df['PARL'], errors='coerce')
    else:
        raise RuntimeError('Raw QAROT must contain women.rep (to build PARL) or PARL.')
    treat_year = None
    if 'first.implement.flag' in df.columns:
        tmp = df.copy()
        tmp['flag'] = pd.to_numeric(tmp['first.implement.flag'], errors='coerce')
        treat_year = tmp.loc[tmp['flag'] == 1].groupby('Country_Code')['Year'].min()
    elif 'implemented.quota' in df.columns:
        tmp = df.copy()
        tmp['flag'] = pd.to_numeric(tmp['implemented.quota'], errors='coerce')
        treat_year = tmp.loc[tmp['flag'] == 1].groupby('Country_Code')['Year'].min()
    elif 'implemented_reform_flag' in df.columns:
        tmp = df.copy()
        tmp['flag'] = pd.to_numeric(tmp['implemented_reform_flag'], errors='coerce')
        treat_year = tmp.loc[tmp['flag'] == 1].groupby('Country_Code')['Year'].min()
    else:
        raise RuntimeError('Raw QAROT must contain first.implement.flag or implemented.quota to build treat_year.')
    dfq = df[['Country_Code', 'Year', 'PARL']].copy()
    dfq['treat_year'] = dfq['Country_Code'].map(treat_year).astype('Int64')
    ensure_dir(os.path.dirname(QAROT_MIN_CSV))
    dfq_out = dfq[['Country_Code', 'Year', 'treat_year', 'PARL']].copy()
    dfq_out.to_csv(QAROT_MIN_CSV, index=False)
    print('Saved QAROT min panel:', QAROT_MIN_CSV)
    return dfq_out

def build_optional_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indices that don't require external downloads:
      - edu_index_z: z(Female_Adult_Literacy) (if exists)
      - health_index_z: z(Life_Expectancy) (if exists)
      - composite_hc_z: mean of available z-scores among:
            z(Human_Capital), edu_index_z, health_index_z
    """
    if 'Female_Adult_Literacy' in df.columns:
        df['edu_index_z'] = zscore(df['Female_Adult_Literacy'])
    else:
        df['edu_index_z'] = np.nan
    if 'Life_Expectancy' in df.columns:
        df['health_index_z'] = zscore(df['Life_Expectancy'])
    else:
        df['health_index_z'] = np.nan
    hc_parts = []
    if 'Human_Capital' in df.columns:
        df['human_capital_z'] = zscore(df['Human_Capital'])
        hc_parts.append(df['human_capital_z'])
    if 'edu_index_z' in df.columns:
        hc_parts.append(df['edu_index_z'])
    if 'health_index_z' in df.columns:
        hc_parts.append(df['health_index_z'])
    if hc_parts:
        df['composite_hc_z'] = pd.concat(hc_parts, axis=1).mean(axis=1, skipna=True)
    else:
        df['composite_hc_z'] = np.nan
    return df

def add_lag_controls(d: pd.DataFrame, controls: list) -> pd.DataFrame:
    d = d.sort_values(['Country_Code', 'stack', 'Year']).copy()
    for c in controls:
        if c not in d.columns:
            d[c] = np.nan
        d[f'lag1_{c}'] = d.groupby(['Country_Code', 'stack'])[c].shift(1)
    return d

def build_stacked_base_sample(df: pd.DataFrame) -> pd.DataFrame:
    cohort_years = list(range(COHORT_MIN, COHORT_MAX + 1))
    stacks = []
    for g in cohort_years:
        treated_ctys = df.loc[df['treat_year'] == g, 'Country_Code'].dropna().unique().tolist()
        if not treated_ctys:
            continue
        control_ctys = df.loc[df['treat_year'].isna() | (df['treat_year'] > g + W_POST), 'Country_Code'].dropna().unique().tolist()
        dfg = df[df['Country_Code'].isin(treated_ctys + control_ctys)].copy()
        dfg['stack'] = int(g)
        dfg['rel_year'] = dfg['Year'].astype(int) - int(g)
        dfg = dfg[(dfg['rel_year'] >= -W_PRE) & (dfg['rel_year'] <= W_POST)].copy()
        dfg['treated_stack'] = dfg['Country_Code'].isin(treated_ctys).astype(int)
        stacks.append(dfg)
    if not stacks:
        raise RuntimeError('No stacks created. Check COHORT_MIN/COHORT_MAX and treat_year coverage.')
    d = pd.concat(stacks, ignore_index=True)
    mm = d.groupby(['stack', 'rel_year'])['treated_stack'].agg(['min', 'max'])
    good = mm[(mm['min'] == 0) & (mm['max'] == 1)].index
    d = d.set_index(['stack', 'rel_year']).loc[good].reset_index()
    d['time_id'] = d['Year'].astype(int) + 10000 * d['stack'].astype(int)
    d['entity_id'] = d['Country_Code'].astype(str) + '__' + d['stack'].astype(str)
    return d

def make_event_terms(d: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    k_list = [k for k in range(-W_PRE, W_POST + 1) if k != BASE_K]
    evt_terms = []
    for k in k_list:
        term = f'evt_{k:+d}'.replace('+', 'p').replace('-', 'm')
        d[term] = ((d['treated_stack'] == 1) & (d['rel_year'] == k)).astype(int)
        evt_terms.append(term)
    lead_terms = []
    for k in range(-W_PRE, 0):
        if k == BASE_K:
            continue
        if k <= -2:
            lead_terms.append(f'evt_{k:+d}'.replace('+', 'p').replace('-', 'm'))
    return (d, evt_terms, lead_terms)

def wald_pretrend_p(res, lead_terms: list) -> float:
    if not lead_terms:
        return np.nan
    pindex = list(res.params.index)
    keep = [t for t in lead_terms if t in pindex]
    if not keep:
        return np.nan
    R = np.zeros((len(keep), len(pindex)))
    for i, t in enumerate(keep):
        R[i, pindex.index(t)] = 1.0
    w = res.wald_test(R, np.zeros(len(keep)))
    try:
        return float(w.pval)
    except Exception:
        return float(getattr(w, 'pvalue', np.nan))

def run_eventstudy_panelols(d_base: pd.DataFrame, outcome: str, evt_terms: list, lead_terms: list, controls: list):
    dd = d_base.copy()
    if outcome not in dd.columns:
        raise RuntimeError(f'Outcome missing: {outcome}')
    dd[outcome] = pd.to_numeric(dd[outcome], errors='coerce')
    dd = dd.dropna(subset=[outcome]).copy()
    dd = add_lag_controls(dd, controls)
    lag_cols = [f'lag1_{c}' for c in controls]
    dd = dd.set_index(['entity_id', 'time_id']).sort_index()
    y = dd[outcome]
    X = dd[evt_terms + lag_cols].copy()
    good_cols = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
    X = X[good_cols]
    clusters = dd.index.get_level_values(0).str.split('__').str[0]
    clusters = pd.Series(clusters.values, index=dd.index)
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type='clustered', clusters=clusters)
    rows = []
    for k in range(-W_PRE, W_POST + 1):
        if k == BASE_K:
            continue
        term = f'evt_{k:+d}'.replace('+', 'p').replace('-', 'm')
        if term in res.params.index:
            coef = float(res.params[term])
            se = float(res.std_errors[term])
            pval = float(res.pvalues[term])
        else:
            coef, se, pval = (np.nan, np.nan, np.nan)
        rows.append({'term': term, 'k': k, 'coef': coef, 'se': se, 'pval': pval})
    tab = pd.DataFrame(rows).sort_values('k')
    pre_p = wald_pretrend_p(res, lead_terms)
    return (res, tab, pre_p)

def plot_eventstudy(tab: pd.DataFrame, title: str, out_png: str):
    plt.figure(figsize=(11, 5))
    plt.axhline(0, linewidth=1)
    plt.axvline(BASE_K, linestyle='--', linewidth=1)
    plt.errorbar(tab['k'], tab['coef'], yerr=1.96 * tab['se'], fmt='o', capsize=3)
    plt.title(title)
    plt.xlabel('Event time k (years relative to quota year)')
    plt.ylabel('Effect vs baseline (k=-1)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ensure_dir(OUTDIR)
    ensure_dir(MERGED_DIR)
    df_main = read_main_panel(MAIN_PANEL_CSV)
    df_main = normalize_country_year(df_main)
    df_q = build_qarot_min_panel_if_needed()
    df_q = normalize_country_year(df_q)
    df_q['treat_year'] = pd.to_numeric(df_q['treat_year'], errors='coerce').astype('Int64')
    df_q['PARL'] = pd.to_numeric(df_q['PARL'], errors='coerce')
    df = df_main.merge(df_q[['Country_Code', 'Year', 'treat_year', 'PARL']].rename(columns={'PARL': 'PARL_qarot'}), on=['Country_Code', 'Year'], how='left')
    if 'PARL' in df.columns:
        df['PARL'] = pd.to_numeric(df['PARL'], errors='coerce')
        df['PARL'] = df['PARL'].where(df['PARL'].notna(), df['PARL_qarot'])
    else:
        df['PARL'] = df['PARL_qarot']
    df = df.drop(columns=['PARL_qarot'], errors='ignore')
    df = build_optional_indices(df)
    df.to_csv(MERGED_CSV, index=False)
    print('Saved merged panel to:', MERGED_CSV)
    d_base = build_stacked_base_sample(df)
    print('STACKED BASE SAMPLE')
    print('n_obs:', len(d_base), 'countries:', d_base['Country_Code'].nunique(), 'treated countries (in stacks):', d_base.loc[d_base['treated_stack'] == 1, 'Country_Code'].nunique())
    print('years:', int(d_base['Year'].min()), '-', int(d_base['Year'].max()))
    used = sorted(d_base['stack'].unique())
    print('cohorts used:', used[:5], '...', used[-5:])
    d_base, evt_terms, lead_terms = make_event_terms(d_base)
    outcomes = ['PARL', 'Human_Capital', 'Life_Expectancy', 'Female_Labor_Force', 'Female_Adult_Literacy', 'Fertility', 'Women_Business', 'Women_Civil_Society', 'edu_index_z', 'health_index_z', 'composite_hc_z']
    outcomes = [o for o in outcomes if o in d_base.columns]
    summaries = []
    for out in outcomes:
        try:
            res, tab, pre_p = run_eventstudy_panelols(d_base, out, evt_terms, lead_terms, CONTROL_COLS)
            tab_path = os.path.join(OUTDIR, f'eventstudy_point_{out}_w{W_PRE}_{W_POST}.csv')
            tab.to_csv(tab_path, index=False)
            png_path = os.path.join(OUTDIR, f'eventstudy_plot_{out}_w{W_PRE}_{W_POST}.png')
            avg_post_0_5 = float(tab.loc[(tab['k'] >= 0) & (tab['k'] <= 5), 'coef'].mean())
            plot_eventstudy(tab, title=f'Stacked event-study: {out} (baseline k={BASE_K} omitted) | pretrend p={pre_p:.3f}', out_png=png_path)
            summaries.append({'outcome': out, 'n_obs': int(res.nobs), 'n_entities': int(res.entity_info.total), 'pretrend_p': pre_p, 'avg_post_0..5': avg_post_0_5, 'csv': tab_path, 'png': png_path})
            print(f'[OK] {out}: pretrend_p={pre_p:.4f} avg_post_0..5={avg_post_0_5:.4f} -> {tab_path}')
        except Exception as e:
            print(f'[FAIL] {out}: {e}')
    if summaries:
        summ = pd.DataFrame(summaries).sort_values('outcome')
        summ_path = os.path.join(OUTDIR, 'summary.csv')
        summ.to_csv(summ_path, index=False)
        print('\nSaved summary to:', summ_path)
    else:
        print('\nNo outcomes estimated (all failed or missing).')
    print('All outputs in:', OUTDIR)

if __name__ == "__main__":
    main()
