"""Figure 4c - acute and chronic PMA exposure.

    PYTHONPATH=. python build_fig4c.py

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
from panel_common import cells, fits, pma, pf, DOSES

# --- TIMING
TIMING={"acute":"PMA added at 0 h","chronic":"PMA added at \u221248 h"}

# --- UNITS
UNITS={"acute":"PMA (ng/mL)","chronic":"PMA (ng/mL)"}

# --- NAIVE_MARK
NAIVE_MARK="#"

# --- pooled-error model per subplot, BH across the six activated comparisons
#     [reproduces tableS2b_fig4c_pooled_error_tests.csv]
import statsmodels.formula.api as smf
from scipy import stats as _st
def _pooled_4c():
    raw = []
    for trt in ["acute", "chronic"]:
        doses = DOSES[trt]
        rows = []
        for dl in doses:
            q = pf[(pf.activation == "activated") & (pf.treatment == trt) & (pf.dl == dl)]
            for w in q.well:
                s = pma[pma.well == w]
                rows.append(dict(cond=dl, y=np.log10(np.mean(10**s.y.values))))
        d = pd.DataFrame(rows)
        m = smf.ols("y ~ C(cond)", data=d).fit()
        mu = d.groupby("cond").y.mean(); n = d.groupby("cond").y.size()
        for dl in doses[1:]:
            est = float(mu[dl] - mu[doses[0]])
            se = float(np.sqrt(m.mse_resid * (1.0/n[dl] + 1.0/n[doses[0]])))
            tt = est / se
            raw.append(dict(trt=trt, dose=dl, fold=float(10**est), t=float(tt),
                            p=float(2*_st.t.sf(abs(tt), m.df_resid)),
                            sd=float(np.sqrt(m.mse_resid)), df=int(m.df_resid)))
    ps = np.array([r["p"] for r in raw]); k = len(ps)
    order = np.argsort(ps); q = np.empty(k)
    running = 1.0
    for rank in range(k-1, -1, -1):
        i = order[rank]
        running = min(running, ps[i]*k/(rank+1))
        q[i] = min(1.0, running)
    for r, qq in zip(raw, q):
        r["q"] = float(qq)
    return raw
res4c = _pooled_4c()
SIG = {}
for r in res4c:
    pan = f"activated \u00b7 {TIMING[r['trt']]}"
    SIG[("Fig 4c", pan, f"{DOSES[r['trt']][0]} vs {r['dose']} ng/mL")] = dict(
        stars=stars(r["q"]), p=r["p"], q=r["q"])

# --- CAP_SHARED_STAT_POOLED - Fig 4c wording, six comparisons]

# --- HEAD4C

# --- CAP4C

# --- CAP4C_S

# --- build4c
def build4c(fname,dpi=300,legend_in=0.46):
    FIGSIZE=(12.6,10.2); TOP=0.606; BOTTOM=0.082; LEGY=0.700; LETY=0.988
    FIGSIZE,TOP,BOTTOM,LEGY,LETY= nocap_layout(FIGSIZE,TOP,BOTTOM,LEGY,LETY,legend_in=legend_in)
    fig=plt.figure(figsize=FIGSIZE)
    gs=fig.add_gridspec(2,2,hspace=0.52,wspace=0.085,left=0.072,right=0.992,top=TOP,bottom=BOTTOM)
    YL=(-1.86,5.30)
    for ri,act in enumerate(["activated","naive"]):
        for ci,trt in enumerate(["acute","chronic"]):
            ax=fig.add_subplot(gs[ri,ci]); doses=DOSES[trt]
            qr=pf[(pf.activation==act)&(pf.treatment==trt)&(pf.dl=="0")]
            RA=arith_ref(qr,pma[pma.well.isin(qr.well)])
            xs=[SPACING*i for i in range(len(doses))]
            ax.set_xlim(-1.02,xs[-1]+1.06); ax.set_ylim(*YL)
            for di,dl in enumerate(doses):
                q=pf[(pf.activation==act)&(pf.treatment==trt)&(pf.dl==dl)]
                draw_column(ax,xs[di],q,pma[pma.well.isin(q.well)],width=WIDTH,ms=1.8,
                            count_y=-1.56,pct_y=(3.66,-1.14),ref=RA,pct_dx=0.34,pct_inset=0.10,
                            rel_y=4.12,rel_fs=6.0)
                if di: F.axis_break(ax,(xs[di-1]+xs[di])/2)
            if act=="naive":
                ax.text(xs[0]+0.40,4.12,NAIVE_MARK,ha="left",va="center",fontsize=7.0,color=F.MUTED)
            else:
                pan=f"activated · {TIMING[trt]}"
                for j,dl in enumerate(doses[1:]):
                    s=SIG[("Fig 4c",pan,f"{doses[0]} vs {dl} ng/mL")]
                    F.sig_bracket(ax,xs[0],xs[j+1],4.42+j*0.27,s["stars"],fontsize=8.0)
            ax.axhline(np.log10(F.THR_UM2),color=F.FAINT,ls=(0,(1,2.4)),lw=.7,zorder=1)
            ax.set_xticks(xs); ax.set_xticklabels([d+(REFTAG if d=="0" else "") for d in doses])
            F.log_ticks(ax,show=(ci==0)); ax.tick_params(labelsize=7)
            ax.spines["bottom"].set_position(("data",-1.78))
            F.gp_axes(ax,ybounds=(-0.55,3.15),xbounds=F.band_bounds(ax,-0.55,xs[-1]+0.55),
                      title=f"{act} · {TIMING[trt]}")
            ax.title.set_position((0.0,1.045)); ax.title.set_ha("left")
            if ci==0:
                ax.set_ylabel("MSD at 9 min lag (µm²)",fontsize=8)
                ax.text(-1.02,4.12,"rel. motility",ha="left",va="center",fontsize=5.8,color=F.MUTED,style="italic")
            if ri==1: ax.set_xlabel(UNITS[trt],fontsize=7.5,labelpad=3)
    fig.legend(handles=LEG(),loc="upper left",bbox_to_anchor=(0.072,LEGY),ncol=3,frameon=False,
               fontsize=6.6,handletextpad=0.6,columnspacing=1.6,labelspacing=0.55)
    fig.text(0.012,LETY,"c",ha="left",va="top",fontsize=12,fontweight="bold")
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
    fig = build4c(_o.path.join(_OUT, "fig4c_acute_chronic_pma.svg"), dpi=300)
    print("overlaps:", overlaps(fig))
