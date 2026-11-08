"""Rebuild descriptive figures from the frozen 1997-2015 panel."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    panel=pd.read_csv(ROOT/'data/processed/country_panel.csv',sep=';',decimal=',')
    out=ROOT/'results/descriptive'
    out.mkdir(parents=True,exist_ok=True)
    for col,prefix,title in [('PARL','parl','Women in parliament (%)'),
                             ('Human_Capital','hc','PWT human-capital index')]:
        d=panel[panel.Year.between(1997,2015)].dropna(subset=[col,'Region','Country_Name'])
        counts=d.groupby('Region').Country_Name.nunique()
        d=d[d.Region.isin(counts[counts.ge(3)].index)]
        trend=d.groupby(['Year','Region'])[col].mean().unstack()
        fig,ax=plt.subplots(figsize=(11,6))
        trend.plot(ax=ax,lw=1.5)
        ax.set(title=title+' by region, 1997-2015',xlabel='Year',ylabel=title)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8,frameon=False)
        fig.tight_layout()
        fig.savefig(out/f'{prefix}_region_trends_1997_2015.svg')
        plt.close(fig)
        country=d.groupby('Country_Name')[col].mean().nlargest(15).sort_values()
        fig,ax=plt.subplots(figsize=(10,6))
        country.plot.barh(ax=ax,color='#2363a4')
        ax.set(title='Top 15 country means, 1997-2015',xlabel=title,ylabel='')
        fig.tight_layout()
        fig.savefig(out/f'{prefix}_top15_countries_mean_1997_2015.svg')
        plt.close(fig)
        means=d.groupby(['Region','Country_Name'])[col].mean().reset_index()
        order=means.groupby('Region')[col].median().sort_values().index.tolist()
        fig,ax=plt.subplots(figsize=(11,7))
        ax.boxplot([means.loc[means.Region.eq(r),col] for r in order],vert=False,
                   labels=order,patch_artist=True,boxprops={'facecolor':'#dce9f7'})
        ax.set(title='Distribution of country means, 1997-2015',xlabel=title)
        fig.tight_layout()
        fig.savefig(out/f'{prefix}_region_boxplot_country_means.svg')
        plt.close(fig)
        summary=d.groupby('Region').agg(mean=(col,'mean'),median=(col,'median'),
                     countries=('Country_Name','nunique'),observations=(col,'count'))
        summary.to_csv(out/f'{prefix}_region_summary.csv')
    print('Saved six descriptive SVG figures and two region tables')


if __name__=='__main__':
    main()
