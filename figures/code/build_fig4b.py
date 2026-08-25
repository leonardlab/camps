"""Figure 4b - plating density and activation timing.

    PYTHONPATH=. python build_fig4b.py

Significance uses the pooled-error model, refit here from the per-well data.
"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
# macOS Helvetica.ttc intermittently raises "failed to load glyph" under repeated loads,
# so register Arial by FILE and put it first. See AESTHETICS.md.
import os as _os
from matplotlib import font_manager as _fm
for _p in ["/System/Library/Fonts/Supplemental/Arial.ttf",
           "/Library/Fonts/Arial.ttf"]:
    if _os.path.exists(_p):
        try:
            _fm.fontManager.addfont(_p)
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
            break
        except Exception:
            pass
else:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle010 as F
F.apply_style()

from panel_common import *  # noqa: F401,F403
from panel_common import cells, fits, dens

# --- density-specific fit view (pf in panel_common is the PMA view)
pf = fits.copy()
pf["dl"] = pf.well.str.extract(r"^(?:PMA_)?(25E4|5E5|1E6)")[0]
pf["is_pma"] = pf.well.str.startswith("PMA")
# --- sel4b
def sel4b(act,a,dl):
    m=(pf.dataset=="density")&(pf.activation==act)&(pf.dl==dl)&(pf.is_pma==(a=="PMA"))
    return pf[m]

# --- b4b_positions
def b4b_positions(): return [i*SPACING for i in range(4)]

# --- Fig 4b column means and derived caption numbers
def colmean(act, a, dl):
    q_ = sel4b(act, a, dl); s = dens[dens.well.isin(q_.well)]
    return float(np.mean([np.mean(10**s[s.well == w].y.values) for w in q_.well]))
r24 = [colmean("24h prior", "dose", d) for d in ["25E4", "5E5", "1E6"]]
ria = [colmean("in-assay", "dose", d) for d in ["25E4", "5E5", "1E6"]]
REF = r24[1]
rel_24 = [v/REF for v in r24]; rel_ia = [v/REF for v in ria]
# --- TFOLD
TFOLD=np.mean(r24)/np.mean(ria); DSPAN=max(r24)/min(r24)

# --- DSPAN
DSPAN=max(r24)/min(r24)

# --- pooled-error model and BH significance-727]
# The within-condition variance is estimated from ALL EIGHT wells of the arm
# (three densities + PMA, four conditions) giving 4 residual df -- not from the
# two contrasted columns alone, which would give 2.
import statsmodels.formula.api as smf
def _pooled_4b():
    out = {}
    for act in ["24h prior", "in-assay"]:
        rows = []
        for a, dl in [("dose", "25E4"), ("dose", "5E5"), ("dose", "1E6"), ("PMA", "5E5")]:
            q_ = sel4b(act, a, dl)
            for w in q_.well:
                s = dens[dens.well == w]
                rows.append(dict(cond=f"{a}_{dl}", y=np.log10(np.mean(10**s.y.values))))
        d = pd.DataFrame(rows)
        m = smf.ols("y ~ C(cond)", data=d).fit()
        mu = d.groupby("cond").y.mean()
        est = float(mu["PMA_5E5"] - mu["dose_5E5"])
        se = float(np.sqrt(m.mse_resid * (1.0/2 + 1.0/2)))
        from scipy import stats as _st
        tt = est / se
        p = float(2 * _st.t.sf(abs(tt), m.df_resid))
        out[act] = dict(fold=float(10**est), p=p, sd=float(np.sqrt(m.mse_resid)),
                        df=int(m.df_resid), t=float(tt))
    ps = [out[a]["p"] for a in ["24h prior", "in-assay"]]
    order = np.argsort(ps); qv = [0.0, 0.0]
    for rank, i in enumerate(order):
        qv[i] = min(1.0, ps[i] * 2 / (rank + 1))
    if qv[order[0]] > qv[order[1]]:
        qv[order[0]] = qv[order[1]]
    for i, a in enumerate(["24h prior", "in-assay"]):
        out[a]["q"] = float(qv[i])
    return out
res = _pooled_4b()
f24, fia = res["24h prior"]["fold"], res["in-assay"]["fold"]
q24, qia = res["24h prior"]["q"], res["in-assay"]["q"]
SIG = {("Fig 4b", a, "5E5 vs 5E5+PMA"): dict(stars=stars(res[a]["q"]), p=res[a]["p"], q=res[a]["q"])
       for a in ["24h prior", "in-assay"]}

# --- HEAD4B

# --- CAP_SHARED_STAT_POOLED

# --- CAP4B

# --- build4b
def build4b(fname,dpi=300,legend_in=0.6):
    FIGSIZE=(11.0,7.4); TOP=0.462; BOTTOM=0.126; LEGY=0.600; LETY=0.984
    FIGSIZE,TOP,BOTTOM,LEGY,LETY= nocap_layout(FIGSIZE,TOP,BOTTOM,LEGY,LETY,legend_in=legend_in)
    fig=plt.figure(figsize=FIGSIZE)
    gs=fig.add_gridspec(1,2,wspace=0.10,left=0.088,right=0.990,top=TOP,bottom=BOTTOM)
    YL=(-1.86,4.62); xs=b4b_positions()
    order=[("dose","25E4"),("dose","5E5"),("dose","1E6"),("PMA","5E5")]
    qref=sel4b("24h prior","dose","5E5"); RA=arith_ref(qref,dens[dens.well.isin(qref.well)])
    for k,act in enumerate(["24h prior","in-assay"]):
        ax=fig.add_subplot(gs[0,k]); labs=[]
        ax.set_xlim(-1.02,xs[-1]+1.02); ax.set_ylim(*YL)
        for di,(a,dl) in enumerate(order):
            xc=xs[di]; q=sel4b(act,a,dl); sub=dens[dens.well.isin(q.well)]
            if a=="PMA": F.control_band(ax,xc-0.92,xc+0.96)
            draw_column(ax,xc,q,sub,width=WIDTH,ms=1.9,count_y=-1.56,pct_y=(3.66,-1.14),
                        ref=RA,pct_dx=0.34,pct_inset=0.10,rel_y=4.12,rel_fs=6.2)
            isref=(k==0 and a=="dose" and dl=="5E5")
            labs.append(dl+("\n+PMA" if a=="PMA" else "")+(REFTAG if isref else ""))
            if di: F.axis_break(ax,(xs[di-1]+xs[di])/2)
        s=SIG[("Fig 4b",act,"5E5 vs 5E5+PMA")]
        F.sig_bracket(ax,xs[1],xs[3],4.40,s["stars"],fontsize=8.4)
        ax.axhline(np.log10(F.THR_UM2),color=F.FAINT,ls=(0,(1,2.4)),lw=.7,zorder=1)
        ax.set_xticks(xs); ax.set_xticklabels(labs)
        F.log_ticks(ax,show=(k==0)); ax.tick_params(labelsize=7)
        ax.spines["bottom"].set_position(("data",-1.78))
        F.gp_axes(ax,ybounds=(-0.55,3.15),xbounds=F.band_bounds(ax,-0.55,xs[-1]+0.55),
                  title="activated 24 h before encapsulation" if k==0 else "activated in-assay")
        if k==0: ax.set_ylabel("MSD at 9 min lag (µm²)",fontsize=8)
        ax.set_xlabel("cells plated per gel",fontsize=7.5,labelpad=3)
        ax.text(-1.02,4.12,"rel. motility" if k==0 else "",ha="left",va="center",
                fontsize=5.8,color=F.MUTED,style="italic")
    fig.legend(handles=LEG([Patch(fc=F.CTRL_BAND,label="PMA positive control")]),loc="upper left",
               bbox_to_anchor=(0.088,LEGY),ncol=3,frameon=False,fontsize=6.6,
               handletextpad=0.6,columnspacing=1.6,labelspacing=0.55)
    fig.text(0.016,LETY,"b",ha="left",va="top",fontsize=12,fontweight="bold")
    F.resolve_label_overlaps(fig); fig.savefig(fname,dpi=dpi); return fig


if __name__ == "__main__":
    # --- repo output location -------------------------------------------
    # Rebuilds land in figures/rebuilt/ so they never overwrite the shipped
    # reference set in figures/output/. Diff the two to check a rebuild.
    import os as _o
    _OUT = _o.environ.get("CAMPS_FIG_OUT",
                          _o.path.join(_o.path.dirname(_o.path.abspath(__file__)),
                                       "..", "rebuilt"))
    _NOCAP = _o.path.join(_OUT, "nocaption")
    _o.makedirs(_OUT, exist_ok=True); _o.makedirs(_NOCAP, exist_ok=True)
    fig = build4b(_o.path.join(_OUT, "fig4b_density_timing.svg"), dpi=300)
    print("overlaps:", overlaps(fig))
