"""Figure 6b - doxycycline titration of the CAMP lines.

    PYTHONPATH=. python build_fig6b.py
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
from panel_common import cells, fits, dox, dxf, CORD, DOSE_O, LBL

RUNS = ["run1", "run2"]
RUN_OFF = 1.86
DOSE_STEP = 4.05

# --- CAMP
CAMP={"239":"Ctrl-LumiScarlet","240":"CAMP-7","255":"CAMP-24","257":"CAMP-26"}

# --- dox_sub
def dox_sub(con,dl,run):
    q=dxf[(dxf.construct.astype(str)==con)&(dxf.dose_label.astype(str)==dl)&(dxf.run==run)]
    return q,dox[dox.well.isin(q.well)]

# --- well_arith_table
def well_arith_table(q,sub):
    """Per-well arithmetic mean MSD (linear um2) — the quantity relative motility is built from."""
    g=sub.well.map(repmap(q.well)).values
    w=F.well_arith_means(sub.y.values,g)
    return np.array([w[k] for k in sorted(w)],float)

# --- Fig 6b significance: run-blocked pooled OLS on the 239 PMA control
#-504; the panel annotates the runs-pooled result]
import statsmodels.formula.api as smf
def _sig6b():
    sig = {}
    recs = []
    for run in RUNS:
        for dl, lab in [("untreated", "untr"), ("PMA", "PMA")]:
            q, s = dox_sub("239", dl, run)
            for v in well_arith_table(q, s):
                recs.append(dict(run=run, arm=lab, y=np.log10(v)))
    D = pd.DataFrame(recs)
    m = smf.ols("y ~ arm + run", data=D).fit()
    est = float(m.params["arm[T.untr]"]); pv = float(m.pvalues["arm[T.untr]"])
    for r in RUNS:
        sig[("Fig 6b", f"Ctrl-LumiScarlet \u00b7 {r}", "untr. vs PMA")] = dict(
            stars=stars(pv), p=pv, q=pv, fold=float(10**(-est)))
    return sig, dict(fold=float(10**(-est)), p=pv, n_wells=len(D))
SIG, res6b = _sig6b()

# --- CAP_SHARED_STAT

# --- HEAD6B

# --- CAP6B

# --- CAP6B_S

# --- build6b
def build6b(fname,dpi=300,legend_in=0.32):
    FIGSIZE=(13.6,17.4); TOP=0.822; BOTTOM=0.042; LEGY=0.876; LETY=0.997
    FIGSIZE,TOP,BOTTOM,LEGY,LETY= nocap_layout(FIGSIZE,TOP,BOTTOM,LEGY,LETY,legend_in=legend_in)
    fig=plt.figure(figsize=FIGSIZE)
    gs=fig.add_gridspec(4,1,hspace=0.36,left=0.068,right=0.992,top=TOP,bottom=BOTTOM)
    YL=(-2.62,4.94); RUNS=["run1","run2"]; RUN_OFF=1.86; DOSE_STEP=4.05
    for k,con in enumerate(["239","240","255","257"]):
        ax=fig.add_subplot(gs[k,0]); doses=CORD[con]
        base=[di*DOSE_STEP for di in range(len(doses))]
        xt=[b+RUN_OFF/2 for b in base]
        ax.set_xlim(-1.10,base[-1]+RUN_OFF+1.14); ax.set_ylim(*YL)
        RA={}; xpos={}
        for r in RUNS:
            qr=dxf[(dxf.construct.astype(str)==con)&(dxf.dose_label.astype(str)=="untreated")&(dxf.run==r)]
            RA[r]=arith_ref(qr,dox[dox.well.isin(qr.well)]) if len(qr) else np.nan
        for di,dl in enumerate(doses):
            for rj,r in enumerate(RUNS):
                q=dxf[(dxf.construct.astype(str)==con)&(dxf.dose_label.astype(str)==dl)&(dxf.run==r)]
                if not len(q): continue
                xc=base[di]+rj*RUN_OFF; xpos[(dl,r)]=xc
                if dl=="PMA": F.control_band(ax,xc-0.86,xc+0.90)
                draw_column(ax,xc,q,dox[dox.well.isin(q.well)],width=1.46,ms=1.5,
                            count_y=-1.82,pct_y=(3.94,-1.36),ref=RA[r],
                            pct_fs=4.8,count_fs=5.0,unit_gap=0.10,
                            pct_dx=0.34,pct_inset=0.10,rel_y=4.40,rel_fs=5.6)
                ax.text(xc,-2.28,r,ha="center",va="center",fontsize=5.6,color=F.MUTED)
            if di: F.axis_break(ax,(base[di-1]+RUN_OFF+base[di])/2)
        if con=="239":
            for r in RUNS:
                if ("PMA",r) not in xpos: continue
                s=SIG[("Fig 6b",f"{CAMP[con]} · {r}","untr. vs PMA")]
                xc=xpos[("PMA",r)]
                F.sig_bracket(ax,xc-0.74,xc+0.74,4.80,s["stars"],fontsize=8.0)
                ax.text(xc+1.30,4.80,"",ha="left",va="center",fontsize=5.2,color=F.MUTED,style="italic")
        ax.axhline(np.log10(F.THR_UM2),color=F.FAINT,ls=(0,(1,2.4)),lw=.7,zorder=1)
        ax.set_xticks(xt)
        ax.set_xticklabels([DOSE_O[d]+(REFTAG if d=="untreated" else "") for d in doses])
        F.log_ticks(ax); ax.tick_params(labelsize=7)
        ax.spines["bottom"].set_position(("data",-2.46))
        F.gp_axes(ax,ybounds=(-0.55,3.15),xbounds=F.band_bounds(ax,-0.55,base[-1]+RUN_OFF+0.55),title=LBL[con])
        ax.title.set_position((0.0,1.030)); ax.title.set_ha("left")
        ax.set_ylabel("MSD at 10 min lag (µm²)",fontsize=8)
        ax.text(-1.10,4.40,"rel. motility",ha="left",va="center",fontsize=5.6,color=F.MUTED,style="italic")
        if k==3: ax.set_xlabel("doxycycline (ng/mL)",fontsize=7.5,labelpad=3)
    fig.legend(handles=LEG([Patch(fc=F.CTRL_BAND,label="PMA positive control")]),loc="upper left",
               bbox_to_anchor=(0.068,LEGY),ncol=5,frameon=False,fontsize=6.8,
               handletextpad=0.6,columnspacing=1.8,labelspacing=0.55)
    fig.text(0.012,LETY,"b",ha="left",va="top",fontsize=12,fontweight="bold")
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
    fig = build6b(_o.path.join(_OUT, "fig6b_dox_titration_gmm.svg"), dpi=300)
    print("overlaps:", overlaps(fig))
