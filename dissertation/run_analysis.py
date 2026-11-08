"""Stacked event-study of gender-quota implementation, 1997-2015.

Run from any directory: python dissertation/run_analysis.py --spec main
Historical notebook estimates are reproduced separately in reproduction/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype']='none'
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = ['PARL', 'Human_Capital', 'Life_Expectancy', 'Female_Adult_Literacy',
            'Fertility', 'edu_index_z', 'health_index_z', 'composite_hc_z']
SPECS = {
    'main': {'controls': [], 'post': 12},
    'adjusted': {'controls': ['GDP', 'Political_Stability', 'Corruption'], 'post': 12},
    'short-window': {'controls': [], 'post': 5},
}


def check_keys(df: pd.DataFrame, keys: list[str], label: str) -> None:
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError(f'{label}: missing or duplicate identifiers {keys}')


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f'Cannot standardise {s.name}')
    return (s - s.mean()) / sd


def load_panel() -> tuple[pd.DataFrame, dict]:
    main = pd.read_csv(ROOT/'data/processed/country_panel.csv', sep=';', decimal=',')
    raw = pd.read_csv(ROOT/'data/raw/qarot.csv')
    q = pd.read_csv(ROOT/'data/processed/qarot_minimal.csv')
    check_keys(main, ['Country_Code', 'Year'], 'Country panel')
    check_keys(raw, ['code', 'year'], 'Raw QAROT')
    check_keys(q, ['Country_Code', 'Year'], 'Minimal QAROT')
    if int(raw.year.max()) != 2015:
        raise ValueError('This specification is validated for the QAROT snapshot ending in 2015')

    # A country-level event date cannot disappear when a year is absent in QAROT.
    first = raw.loc[raw['first.implement.flag'].eq(1)].groupby('code').year.min()
    stored = q.dropna(subset=['treat_year']).groupby('Country_Code').treat_year.first()
    if not stored.eq(first.reindex(stored.index)).all():
        raise ValueError('Raw and minimal QAROT treatment dates disagree')
    known = set(raw.code.dropna())
    excluded = sorted(set(main.Country_Code) - known)
    panel = main[main.Country_Code.isin(known)].copy()
    panel = panel.merge(q[['Country_Code','Year','PARL']].rename(columns={'PARL':'PARL_qarot'}),
                        on=['Country_Code','Year'], how='left', validate='one_to_one')
    panel['PARL'] = panel.PARL.fillna(panel.PARL_qarot)
    panel = panel.drop(columns='PARL_qarot')
    panel['treat_year'] = panel.Country_Code.map(first)
    panel = panel.sort_values(['Country_Code','Year'])

    # Compute calendar lags before selecting the estimation period or removing missing outcomes.
    for col in ['GDP','Political_Stability','Corruption']:
        g = panel.groupby('Country_Code')
        panel['lag1_'+col] = g[col].shift(1).where(panel.Year.sub(g.Year.shift(1)).eq(1))
    panel = panel[panel.Year.between(1997,2015)].copy()
    panel['edu_index_z'] = zscore(panel.Female_Adult_Literacy)
    panel['health_index_z'] = zscore(panel.Life_Expectancy) - zscore(panel.Fertility)
    panel['composite_hc_z'] = panel[['edu_index_z','health_index_z']].mean(axis=1, skipna=True)
    for col in OUTCOMES:
        if np.isinf(panel[col].dropna()).any():
            raise ValueError(f'Infinite values in {col}')
    info = {'year_min':1997,'year_max':2015,'panel_rows':len(panel),
            'countries':int(panel.Country_Code.nunique()),'excluded_unknown_quota_status':excluded,
            'treatment_definition':'first.implement.flag == 1 in raw QAROT, earliest year by country',
            'standardisation':'pooled country-year mean and sample standard deviation, 1997-2015',
            'health_index':'z(Life_Expectancy) - z(Fertility)',
            'education_index':'z(Female_Adult_Literacy)',
            'composite_index':'row mean of available education and health indices',
            'duplicate_input_keys':0}
    return panel, info


def build_stacks(panel: pd.DataFrame, post: int) -> pd.DataFrame:
    stacks = []
    for cohort in range(2002,2011):
        treated = panel.treat_year.eq(cohort)
        if not treated.any():
            continue
        controls = panel.treat_year.isna() | panel.treat_year.gt(cohort+post)
        d = panel.loc[treated | controls].copy()
        d['stack'] = cohort
        d['event_time'] = d.Year - cohort
        d = d[d.event_time.between(-5,post)].copy()
        d['treated'] = d.treat_year.eq(cohort).astype(int)
        stacks.append(d)
    if not stacks:
        raise ValueError('No cohorts in the requested sample')
    return pd.concat(stacks,ignore_index=True)


def estimate(stacked: pd.DataFrame, outcome: str, controls: list[str], post: int):
    xcontrol = ['lag1_'+c for c in controls]
    d = stacked.dropna(subset=[outcome]+xcontrol).copy()
    # Check comparison support after outcome-specific missingness is removed.
    group = d.groupby(['stack','event_time']).treated
    support = group.agg(['min','max'])
    keep = support[(support['min']==0)&(support['max']==1)].index
    d = d.set_index(['stack','event_time']).loc[keep].reset_index()
    support = d.groupby('event_time').agg(rows=('Country_Code','size'),
              countries=('Country_Code','nunique'),treated_rows=('treated','sum'),
              cohorts=('stack','nunique')).reset_index()
    d['unit'] = d.Country_Code+'__'+d['stack'].astype(str)
    d['time'] = d['stack']*10000+d.Year
    check_keys(d,['unit','time'],'Stacked estimation sample')
    terms = []
    for k in range(-5,post+1):
        if k == -1:
            continue
        term = 'event_'+('m'+str(-k) if k<0 else 'p'+str(k))
        d[term] = ((d.treated==1)&(d.event_time==k)).astype(int)
        if d[term].nunique()>1:
            terms.append(term)
    d = d.set_index(['unit','time']).sort_index()
    clusters = pd.Series(d.Country_Code.to_numpy(),index=d.index)
    model = PanelOLS(d[outcome],d[terms+xcontrol],entity_effects=True,
                     time_effects=True,drop_absorbed=True,check_rank=True)
    result = model.fit(cov_type='clustered',clusters=clusters)
    rows=[]
    for k in range(-5,post+1):
        if k == -1:
            continue
        term='event_'+('m'+str(-k) if k<0 else 'p'+str(k))
        identified = term in result.params.index
        b=float(result.params[term]) if identified else np.nan
        se=float(result.std_errors[term]) if identified else np.nan
        rows.append({'k':k,'coef':b,'se':se,'pvalue':float(result.pvalues[term]) if identified else np.nan,
                     'ci_lower':b-1.96*se,'ci_upper':b+1.96*se,'identified':identified})
    table=pd.DataFrame(rows)
    leads=[t for t in result.params.index if t.startswith('event_m')]
    R=np.zeros((len(leads),len(result.params)))
    for i,t in enumerate(leads):R[i,result.params.index.get_loc(t)]=1
    pre_p=float(result.wald_test(R,np.zeros(len(leads))).pval) if leads else np.nan
    k0=table.loc[table.k.eq(0)].iloc[0]
    post_terms=['event_p'+str(k) for k in range(6)]
    post_mean=post_se=post_p=np.nan
    if all(t in result.params.index for t in post_terms):
        contrast=np.array([1/6 if t in post_terms else 0 for t in result.params.index])
        post_mean=float(contrast @ result.params.to_numpy())
        variance=float(contrast @ result.cov.to_numpy() @ contrast)
        if variance>0:
            post_se=float(np.sqrt(variance))
            post_p=float(result.wald_test(contrast[None,:],np.zeros(1)).pval)
    summary={'outcome':outcome,'n_obs':int(result.nobs),'countries':int(d.Country_Code.nunique()),
             'treated_countries':int(d.loc[d.treated.eq(1),'Country_Code'].nunique()),
             'year_min':int(d.Year.min()),'year_max':int(d.Year.max()),
             'cohorts':int(d['stack'].nunique()),'k0':float(k0.coef),'k0_se':float(k0.se),
             'k0_pvalue':float(k0.pvalue),'pretrend_p':pre_p,
             'mean_post_0_5':post_mean,'mean_post_0_5_se':post_se,
             'mean_post_0_5_pvalue':post_p,'mean_post_0_5_ci_lower':post_mean-1.96*post_se,
             'mean_post_0_5_ci_upper':post_mean+1.96*post_se}
    return result,table,support,summary


def plot(table: pd.DataFrame,outcome: str,path: Path) -> None:
    titles={'PARL':'Women in parliament (percentage points)',
            'Human_Capital':'PWT human-capital index','Life_Expectancy':'Life expectancy (years)',
            'Female_Adult_Literacy':'Female adult literacy (%)','Fertility':'Adolescent fertility',
            'edu_index_z':'Education proxy (standardised)',
            'health_index_z':'Health proxy (standardised components)',
            'composite_hc_z':'Composite education and health proxy'}
    clean=table[table.identified]
    fig,ax=plt.subplots(figsize=(9,4.6))
    ax.axhline(0,color='#8692a1',lw=1)
    ax.axvline(-0.5,color='#8692a1',ls='--',lw=1)
    ax.errorbar(clean.k,clean.coef,yerr=1.96*clean.se,fmt='o',color='#2363a4',
                capsize=3,lw=1.2,markersize=4)
    ax.scatter([-1],[0],marker='s',color='#6b7280',s=28,label='Reference year')
    ax.set(title=titles[outcome],xlabel='Years relative to first quota implementation',
           ylabel='Difference relative to k = -1')
    ax.set_xticks(sorted(set([-5,-3,-1,0]+list(range(2,int(table.k.max())+1,2)))))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    fig.savefig(path.with_suffix('.svg'),format='svg')
    plt.close(fig)


def run(spec: str) -> None:
    config=SPECS[spec]
    out=ROOT/'results/dissertation'/spec
    out.mkdir(parents=True,exist_ok=True)
    panel,info=load_panel()
    stacked=build_stacks(panel,config['post'])
    summaries=[]
    for outcome in OUTCOMES:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result,table,support,summary=estimate(stacked,outcome,config['controls'],config['post'])
        table.to_csv(out/f'{outcome}.csv',index=False)
        support.to_csv(out/f'{outcome}_support.csv',index=False)
        (out/f'{outcome}_model.txt').write_text(str(result.summary)+'\n')
        plot(table,outcome,out/f'{outcome}.png')
        summary['warnings']=' | '.join(dict.fromkeys(str(w.message).strip() for w in caught))
        summaries.append(summary)
        print(f"{spec}: {outcome}, n={summary['n_obs']}, k0={summary['k0']:.5f}, pretrend p={summary['pretrend_p']:.4f}",flush=True)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    info.update({'spec':spec,'cohort_min':2002,'cohort_max':2010,'window_pre':5,
                 'window_post':config['post'],'controls':config['controls'],
                 'fixed_effects':['country x cohort','year x cohort'],'cluster':'original country',
                 'packages':{n:importlib.metadata.version(n) for n in ['pandas','numpy','scipy','linearmodels','matplotlib']},
                 'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [ROOT/'data/processed/country_panel.csv',ROOT/'data/raw/qarot.csv',ROOT/'data/processed/qarot_minimal.csv']}})
    (out/'metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',choices=list(SPECS),default='main')
    run(parser.parse_args().spec)
