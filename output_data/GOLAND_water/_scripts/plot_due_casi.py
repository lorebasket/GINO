#!/usr/bin/env python3
"""Report a due casi: Vg-Vf sovrapposti in profondita' + confronto UVLM-only."""
import csv, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.cm as cm
from matplotlib.lines import Line2D
OUT="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/output_data/GOLAND_water"
kn=1/0.514444; JUMP=0.20
REV=os.environ.get("GW_REV","revA")          # revD = BEM con torsione corretta (21/10/2026)
SFX="" if REV=="revA" else "_"+REV

def load(d, br=2):
    """g,f troncati al primo salto, solo punti convergenti."""
    vg=list(csv.DictReader(open(d+"/vg_data.csv",encoding="utf-8",errors="replace")))
    vf=list(csv.DictReader(open(d+"/vf_data.csv",encoding="utf-8",errors="replace")))
    V=np.array([float(x["Velocity_m_s"]) for x in vg])*kn
    def c_(r,n):
        o=[]
        for x in r:
            try:o.append(float(x[n]))
            except Exception:o.append(np.nan)
        return np.array(o)
    g=c_(vg,f"Damping_Branch_{br}"); f=c_(vf,f"Frequency_Hz_Branch_{br}")
    ok=np.array([x.get(f"Converged_Branch_{br}")=="1" for x in vg])&np.isfinite(g)&np.isfinite(f)
    idx=np.where(ok)[0]; cut=len(V)
    for a,b in zip(idx[:-1],idx[1:]):
        if f[a]>1e-9 and abs(f[b]-f[a])/f[a]>JUMP: cut=b; break
    m=ok.copy(); m[cut:]=False
    return V,g,f,m

def cross(V,g,f,m):
    ii=np.where(m)[0]
    for a,b in zip(ii[:-1],ii[1:]):
        if g[a]<0<=g[b]:
            t=-g[a]/(g[b]-g[a]); return V[a]+t*(V[b]-V[a]), f[a]
    return None,None

CASI={
 "A": dict(nome="CASO numerical A — massima sensibilità (t/c=40%, μ*=2.74, senza accoppiamento)",
           glob=OUT+"/casoA_numeric_"+REV+"/*_hc*",
           uvlm=OUT+"/casoA_numeric_"+REV+"/*UVLMpuro*",
           tag="casoA", xlim=(1.5,8.5)),
 "B": dict(nome="CASO steel B — realizzabile (t/c=20%, μ*=1.37, ψ=+0.4)",
           glob=OUT+"/casoB_numericSteel_"+REV+"/*_hc*",
           uvlm=OUT+"/casoB_numericSteel_"+REV+"/*UVLMpuro*",
           tag="casoB", xlim=(1.0,9.0)),
}

for key,C in CASI.items():
    ds=[]
    for d in glob.glob(C["glob"]):
        if not os.path.exists(d+"/vg_data.csv"): continue
        if "UVLMpuro" in d: continue   # il riferimento cieco non e' un punto di profondita
        hc=float(os.path.basename(d).split("_hc")[1].replace("p","."))
        ds.append((hc,d))
    ds.sort()
    if not ds: print(f"{key}: nessun dato"); continue
    COLS=cm.viridis(np.linspace(0,0.9,len(ds)))

    # ---- FIG 1: sovrapposizione in profondita' ----
    fig,ax=plt.subplots(1,2,figsize=(14,5.6))
    cr=[]
    for (hc,d),col in zip(ds,COLS):
        V1,g1,f1,m1=load(d,br=1)          # ramo flessionale
        ax[0].plot(V1[m1],g1[m1],"--",color=col,lw=1.1,alpha=.55)
        ax[1].plot(V1[m1],f1[m1],"--",color=col,lw=1.1,alpha=.55)
        V,g,f,m=load(d)
        ax[0].plot(V[m],g[m],color=col,lw=2.1,label=f"h/c = {hc:g}")
        ax[1].plot(V[m],f[m],color=col,lw=2.1)
        vf,ff=cross(V,g,f,m)
        if vf:
            ax[0].plot(vf,0,"o",color=col,ms=9,mec="k",zorder=6)
            ax[1].plot(vf,ff,"o",color=col,ms=9,mec="k",zorder=6)
            cr.append((hc,vf))
    ax[0].axhline(0,color="k",lw=1.3); ax[0].set_xlim(*C["xlim"]); ax[0].set_ylim(-0.30,0.25)
    ax[0].set_xlabel("V [kn]"); ax[0].set_ylabel("g = σ/ω"); ax[0].grid(alpha=.3)
    l1=ax[0].legend(fontsize=8.5,ncol=2,title="sommergenza",loc="upper left"); ax[0].add_artist(l1)
    ax[0].legend(handles=[Line2D([0],[0],color="k",lw=2.1,label="ramo 2 (2º modo) — va in flutter"),
                          Line2D([0],[0],color="k",ls="--",lw=1.1,alpha=.6,label="ramo 1 (1º modo, flessionale)")],
                 fontsize=8,loc="lower right",framealpha=.95)
    ax[0].set_title("V–g  (ramo 2, solo punti convergenti)",fontsize=10.5)
    ax[1].set_xlim(*C["xlim"]); ax[1].set_xlabel("V [kn]"); ax[1].set_ylabel("f [Hz]")
    ax[1].grid(alpha=.3); ax[1].set_title("V–f",fontsize=10.5)
    if cr:
        v=[x[1] for x in cr]; dV=100*(max(v)-min(v))/min(v)
        ax[0].annotate(f"ΔV = {dV:.1f}%   ({max(v):.2f} → {min(v):.2f} kn)",
                       (0.98,0.97),xycoords="axes fraction",ha="right",va="top",
                       fontsize=11,fontweight="bold",
                       bbox=dict(fc="lightyellow",ec="grey",boxstyle="round,pad=0.4"))
    fig.suptitle(C["nome"],fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.93])
    fig.savefig(f"{OUT}/figures/fig_{C['tag']}_profondita{SFX}.png",dpi=150)
    print(f"fig_{C['tag']}_profondita.png   crossing: {[(h,round(v,2)) for h,v in cr]}")

    # ---- FIG 2: estremi + UVLM-only ----
    uv=glob.glob(C["uvlm"])
    uv=[u for u in uv if os.path.exists(u+"/vg_data.csv")]
    fig,ax=plt.subplots(1,2,figsize=(14,5.6))
    sel=[ds[0],ds[-1]]
    _dy=[14,-20]
    for (hc,d),col,dy in zip(sel,("tab:red","tab:blue"),_dy):
        V1,g1,f1,m1=load(d,br=1)
        ax[0].plot(V1[m1],g1[m1],"--",color=col,lw=1.3,alpha=.6)
        ax[1].plot(V1[m1],f1[m1],"--",color=col,lw=1.3,alpha=.6)
        V,g,f,m=load(d)
        ax[0].plot(V[m],g[m],color=col,lw=2.4,label=f"UVLM+BEM  h/c = {hc:g}")
        ax[1].plot(V[m],f[m],color=col,lw=2.4)
        vf,ff=cross(V,g,f,m)
        if vf:
            ax[0].plot(vf,0,"o",color=col,ms=10,mec="k",zorder=6)
            ax[0].annotate(f"{vf:.2f} kn",(vf,0),textcoords="offset points",
                           xytext=(-30 if dy>0 else 6,dy),fontsize=10,color=col,fontweight="bold")
    if uv:
        V1,g1,f1,m1=load(uv[0],br=1)
        ax[0].plot(V1[m1],g1[m1],":",color="k",lw=1.3,alpha=.6)
        ax[1].plot(V1[m1],f1[m1],":",color="k",lw=1.3,alpha=.6)
        V,g,f,m=load(uv[0])
        ax[0].plot(V[m],g[m],"--",color="k",lw=2.2,label="UVLM only (cieco alla profondità)")
        ax[1].plot(V[m],f[m],"--",color="k",lw=2.2)
        vf,ff=cross(V,g,f,m)
        if vf:
            ax[0].plot(vf,0,"s",color="k",ms=10,zorder=6)
            ax[0].annotate(f"{vf:.2f} kn",(vf,0),textcoords="offset points",
                           xytext=(6,-18),fontsize=9.5,fontweight="bold")
    ax[0].axhline(0,color="k",lw=1.3); ax[0].set_ylim(-0.30,0.15)
    ax[0].set_xlim(*C["xlim"]); ax[1].set_xlim(*C["xlim"])
    ax[0].set_xlabel("V [kn]"); ax[0].set_ylabel("g = σ/ω"); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=9,loc="lower right",title="linea piena = ramo 2 · tratteggio sottile = ramo 1")
    ax[0].set_title("V–g : cosa cambia sostituendo la massa apparente UVLM col BEM",fontsize=10.5)
    ax[1].set_xlabel("V [kn]"); ax[1].set_ylabel("f [Hz]"); ax[1].grid(alpha=.3)
    ax[1].set_title("V–f",fontsize=10.5)
    fig.suptitle(C["nome"]+"  —  UVLM only vs UVLM+BEM",fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.93])
    fig.savefig(f"{OUT}/figures/fig_{C['tag']}_uvlm_vs_bem{SFX}.png",dpi=150)
    print(f"fig_{C['tag']}_uvlm_vs_bem{SFX}.png")
