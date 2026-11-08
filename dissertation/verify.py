"""Check data invariants and reproduction against the original PARL snapshot."""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_analysis import ROOT, load_panel, build_stacks, SPECS


def verify():
    panel,_=load_panel()
    assert panel.Year.between(1997,2015).all()
    assert not panel.duplicated(['Country_Code','Year']).any()
    assert panel.groupby('Country_Code').treat_year.nunique(dropna=False).max()==1
    for spec,cfg in SPECS.items():
        stacked=build_stacks(panel,cfg['post'])
        controls=stacked.loc[stacked.treated.eq(0)]
        assert (controls.treat_year.isna() | controls.treat_year.gt(controls['stack']+cfg['post'])).all()
        assert not stacked.duplicated(['Country_Code','stack','Year']).any()
        out=ROOT/'results/dissertation'/spec
        summary=pd.read_csv(out/'summary.csv')
        assert len(summary)==8
        assert summary.year_min.ge(1997).all() and summary.year_max.le(2015).all()
        assert np.isfinite(summary[['k0','k0_se','pretrend_p']]).all().all()
        for outcome in summary.outcome:
            table=pd.read_csv(out/f'{outcome}.csv')
            assert not table.k.eq(-1).any()
            assert table.loc[table.identified,['coef','se']].notna().all().all()
            support=pd.read_csv(out/f'{outcome}_support.csv')
            assert support.treated_rows.gt(0).all()
            assert support.rows.gt(support.treated_rows).all()
    original=pd.read_csv(ROOT/'results/original-snapshots/eventstudy_point_PARL_w5_12.csv')
    rerun=pd.read_csv(ROOT/'results/reproduction/cell-76/eventstudy_point_PARL_w5_12.csv')
    assert original.k.equals(rerun.k)
    for col in ['coef','se','pval']:
        np.testing.assert_allclose(original[col],rerun[col],atol=1e-10,rtol=1e-9,equal_nan=True)
    # Recover a known effect in a noisy synthetic staggered panel.
    from run_analysis import estimate
    rng=np.random.default_rng(73)
    synthetic=[]
    for i in range(90):
        treatment=2002 if i<30 else 2005 if i<60 else np.nan
        for year in range(1997,2016):
            treated_post=np.isfinite(treatment) and year>=treatment
            synthetic.append({'Country_Code':f'C{i:03d}','Year':year,'treat_year':treatment,
                              'y':i*.3+(year-1997)*.2+3*treated_post+rng.normal(0,.15)})
    s=build_stacks(pd.DataFrame(synthetic),5)
    _,t,_,_=estimate(s,'y',[],5)
    assert t.loc[t.k.ge(0),'coef'].between(2.8,3.2).all()
    assert t.loc[t.k.le(-2),'coef'].between(-.2,.2).all()
    print('PASS: input keys, country-level treatment, cohort controls, sample bounds, outputs, exact PARL reproduction, synthetic effect recovery')


if __name__=='__main__':
    verify()
