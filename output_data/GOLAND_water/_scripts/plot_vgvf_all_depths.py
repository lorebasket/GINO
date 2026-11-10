#!/usr/bin/env python3
"""Vg-Vf di TUTTE le profondita' del caso B sovrapposte, filtrati (solo punti
convergenti, troncati al primo salto di ramo >20%). Compagno di fig_depth_final.png."""
import csv, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
OUT="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/output_data/GOLAND_water"
kn=1/0.514444; JUMP=0.20

cases=[]
for d in glob.glob(OUT+"/sweeps/tf04_depth_B_soft/*_hc*"):
    hc=float(os.path.basename(d).split("_hc")[1].replace("p","."))
    cases.append((hc,d))
cases.sort()
COLS=cm.viridis(np.linspace(0,0.9,len(cases)))

fig,ax=plt.subplots(1,2,figsize=(14,5.8))
cross=[]
for (hc,d),col in zip(cases,COLS):
    vg=list(csv.DictReader(open(d+"/vg_data.csv",encoding="utf-8",errors="replace")))
    vf=list(csv.DictReader(open(d+"/vf_data.csv",encoding="utf-8",errors="replace")))
    V=np.array([float(x["Velocity_m_s"]) for x in vg])*kn
    def c_(r,n):
        o=[]
        for x in r:
            try:o.append(float(x[n]))
            except Exception:o.append(np.nan)
        return np.array(o)
    for br,lw,ls,alpha in ((1,1.1,"--",0.45),(2,2.3,"-",1.0)):
        g=c_(vg,f"Damping_Branch_{br}"); f=c_(vf,f"Frequency_Hz_Branch_{br}")
        ok=np.array([x.get(f"Converged_Branch_{br}")=="1" for x in vg])&np.isfinite(g)&np.isfinite(f)
        idx=np.where(ok)[0]; cut=len(V)
        for a,b in zip(idx[:-1],idx[1:]):
            if f[a]>1e-9 and abs(f[b]-f[a])/f[a]>JUMP: cut=b; break
        m=ok.copy(); m[cut:]=False
        ax[0].plot(V[m],g[m],ls,color=col,lw=lw,alpha=alpha,
                   label=(f"h/c = {hc:g}" if br==2 else None))
        ax[1].plot(V[m],f[m],ls,color=col,lw=lw,alpha=alpha)
        if br==2:
            ii=np.where(m)[0]
            for a,b in zip(ii[:-1],ii[1:]):
                if g[a]<0<=g[b]:
                    t=-g[a]/(g[b]-g[a]); Vf=V[a]+t*(V[b]-V[a])
                    ax[0].plot(Vf,0,"o",color=col,ms=9,mec="k",mew=1.1,zorder=6)
                    ax[1].plot(Vf,f[a],"o",color=col,ms=9,mec="k",mew=1.1,zorder=6)
                    cross.append((hc,Vf,f[a])); break

ax[0].axhline(0,color="k",lw=1.4)
ax[0].set_xlim(1.5,8.5); ax[0].set_ylim(-0.55,0.35)
ax[0].set_xlabel("V [kn]"); ax[0].set_ylabel("g = σ/ω")
ax[0].set_title("V–g : il ramo 2 attraversa g=0 sempre più tardi\nman mano che il profilo si avvicina alla superficie",fontsize=10.5)
ax[0].grid(alpha=.3)
l1=ax[0].legend(fontsize=8.5,loc="upper left",ncol=2,title="sommergenza"); ax[0].add_artist(l1)
ax[0].legend(handles=[Line2D([0],[0],color="k",ls="-",lw=2.3,label="ramo 2 (2º modo) — flutter"),
                      Line2D([0],[0],color="k",ls="--",lw=1.1,alpha=.5,label="ramo 1 (1º modo) — stabile"),
                      Line2D([0],[0],color="k",ls="none",marker="o",ms=8,mec="k",label="g = 0")],
             fontsize=8,loc="lower left",bbox_to_anchor=(0.0,0.0),framealpha=.95)

ax[1].set_xlim(1.5,8.5); ax[1].set_xlabel("V [kn]"); ax[1].set_ylabel("f [Hz]")
ax[1].set_title("V–f : nessuna coalescenza, la frequenza scende liscia",fontsize=10.5)
ax[1].grid(alpha=.3)
if cross:
    C=np.array(cross)
    dV=100*(C[:,1].max()-C[:,1].min())/C[:,1].min()
    ax[0].annotate(f"ΔV = {dV:.1f}%   ({C[:,1].max():.2f} → {C[:,1].min():.2f} kn)",
                   (0.985,0.965),xycoords="axes fraction",ha="right",va="top",fontsize=11.5,
                   fontweight="bold",bbox=dict(fc="lightyellow",ec="grey",boxstyle="round,pad=0.4"))
    # banda che copre l'escursione dei crossing
    ax[0].axvspan(C[:,1].min(),C[:,1].max(),color="gold",alpha=.18,zorder=0)
    ax[1].axvspan(C[:,1].min(),C[:,1].max(),color="gold",alpha=.18,zorder=0)
fig.suptitle("Caso B — acciaio sp=0.40, $x_{EA}$=0.15, $x_{CM}$=0.75, μ*=2.74 : $V$–$g$ / $V$–$f$ a tutte le sommergenze\n"
             "(solo punti convergenti, curve troncate al primo salto di ramo)",
             fontsize=12,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.90])
fig.savefig(OUT+"/fig_vgvf_all_depths.png",dpi=150)
print("fig_vgvf_all_depths.png")
for hc,v,f0 in cross: print(f"   h/c={hc:5.2f}  {v:6.2f} kn  {f0:6.3f} Hz")
