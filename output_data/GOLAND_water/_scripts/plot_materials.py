#!/usr/bin/env python3
"""Figure materiali/spessore. I rami vengono TRONCATI al primo salto di frequenza
(>20% fra punti consecutivi convergenti): oltre quel punto il tracker p-k ha
cambiato radice e i valori non appartengono piu' allo stesso modo fisico."""
import csv, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
REPO="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
OUT=f"{REPO}/output_data/GOLAND_water"; kn=1/0.514444
JUMP=0.20

def load(d):
    vg=list(csv.DictReader(open(d+"/vg_data.csv",encoding="utf-8",errors="replace")))
    vf=list(csv.DictReader(open(d+"/vf_data.csv",encoding="utf-8",errors="replace")))
    V=np.array([float(x["Velocity_m_s"]) for x in vg])*kn
    o={}
    for br in (1,2):
        def c_(rows,n):
            r=[]
            for x in rows:
                try:r.append(float(x[n]))
                except Exception:r.append(np.nan)
            return np.array(r)
        g=c_(vg,f"Damping_Branch_{br}"); f=c_(vf,f"Frequency_Hz_Branch_{br}")
        ok=np.array([x.get(f"Converged_Branch_{br}")=="1" for x in vg])&np.isfinite(g)&np.isfinite(f)
        # ---- troncamento al primo salto ----
        idx=np.where(ok)[0]; cut=len(V); Vjump=None
        for a,b in zip(idx[:-1],idx[1:]):
            if f[a]>1e-9 and abs(f[b]-f[a])/f[a]>JUMP:
                cut=b; Vjump=V[a]; break
        m=ok.copy(); m[cut:]=False
        o[br]=(g,f,m,Vjump)
    return V,o

MAT=[("CFRP  1600", 0.168, "tab:blue",
      f"{OUT}/sweeps/working_topology_depth/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p015_gjfrac0p0037_rho1600_hc2p0"),
     ("Al    2700", 0.283, "tab:orange",
      f"{OUT}/sweeps/mass_ladder/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p0253_gjfrac0p00624_hc2p0"),
     ("acciaio 7850", 0.821, "tab:green",
      f"{OUT}/sweeps/mass_ladder/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p0736_gjfrac0p01815_rho7850_hc2p0")]
TF=[("sp=0.20", 1.37, "tab:purple",
     f"{OUT}/sweeps/tf_ladder/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p0091_gjfrac0p00226_rho7850_tf0p2_hc2p0"),
    ("sp=0.30", 2.05, "tab:red",
     f"{OUT}/sweeps/tf_ladder/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p0041_gjfrac0p00102_rho7850_tf0p3_hc2p0"),
    ("sp=0.40", 2.74, "darkred",
     f"{OUT}/sweeps/tf_ladder/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p0024_gjfrac0p00059_rho7850_tf0p4_hc2p0")]

STYLE=[Line2D([0],[0],color="k",ls="-",lw=2,
              label="— continuo: ramo 1  (1º modo, ≈4.4 Hz a secco)"),
       Line2D([0],[0],color="k",ls="--",lw=2,
              label="-- tratteggiato: ramo 2  (2º modo, ≈14 Hz a secco)"),
       Line2D([0],[0],color="k",ls="none",marker="x",ms=8,
              label="× troncamento al 1º salto di ramo (Δf>20%)")]
NOTE=("sp = spessore/corda della sezione  (sp=0.20 → sezione al 20%)\n"
      "entrambi i modi sono MISTI flessione-torsione\n"
      "(twist·b/plunge ≈ 0.34–0.42): il ramo 2 è il più torsionale")

def draw(cases, fname, title, sub, ylim, mark_cross):
    fig,ax=plt.subplots(1,2,figsize=(13,5.4))
    for nm,mus,col,d in cases:
        if not os.path.exists(d+"/vg_data.csv"): continue
        V,o=load(d)
        for br,ls in ((1,"-"),(2,"--")):
            g,f,m,Vj=o[br]
            ax[0].plot(V[m],g[m],ls,color=col,lw=1.8,
                       label=f"{nm}  (μ*={mus:.2f})" if br==1 else None)
            ax[1].plot(V[m],f[m],ls,color=col,lw=1.8)
            if Vj is not None and Vj<=40:
                ax[0].plot(Vj,g[m][-1] if m.any() else 0,"x",color=col,ms=9,mew=2.2)
                ax[1].plot(Vj,f[m][-1] if m.any() else 0,"x",color=col,ms=9,mew=2.2)
            if mark_cross and br==2:
                idx=np.where(m)[0]
                for a,b in zip(idx[:-1],idx[1:]):
                    if g[a]<0<=g[b]:
                        t=-g[a]/(g[b]-g[a]); Vf=V[a]+t*(V[b]-V[a])
                        ax[0].plot(Vf,0,"o",color=col,ms=10,mec="k",zorder=6)
                        ax[0].annotate(f"{Vf:.1f} kn",(Vf,0),textcoords="offset points",
                                       xytext=(5,9),fontsize=9.5,color=col,fontweight="bold")
                        break
    ax[0].axhline(0,color="k",lw=1.3); ax[0].set_ylim(*ylim); ax[0].set_xlim(0,40)
    ax[0].set_xlabel("V [kn]"); ax[0].set_ylabel("g"); ax[0].grid(alpha=.3)
    ax[0].set_title(sub,fontsize=10)
    l1=ax[0].legend(fontsize=9,loc="lower left"); ax[0].add_artist(l1)
    ax[0].legend(handles=STYLE,fontsize=8,loc="upper left",framealpha=.95)
    ax[1].set_xlim(0,40); ax[1].set_xlabel("V [kn]"); ax[1].set_ylabel("f [Hz]")
    ax[1].set_title("frequenze (troncate al 1º salto)",fontsize=10); ax[1].grid(alpha=.3)
    ax[1].annotate(NOTE,(0.98,0.03),xycoords="axes fraction",ha="right",va="bottom",
                   fontsize=8.2,bbox=dict(fc="lightyellow",ec="grey",alpha=.9,boxstyle="round,pad=0.35"))
    fig.suptitle(title,fontweight="bold",fontsize=12.5)
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUT}/{fname}",dpi=150); print(fname)

draw(MAT,"fig_scala_materiali.png",
     "Effetto della densità del materiale (xea=0.15 / xcm=0.75, sp=0.12)",
     "g non raggiunge mai lo zero a nessuna densità reale",(-3,0.5),False)
draw(TF,"fig_scala_spessore.png",
     "Effetto dello spessore (acciaio, f₁ tenuta a 4.42 Hz)",
     "il ramo 2 attraversa g=0 per sp ≥ 0.30",(-1.2,0.8),True)

# ---------- combinata ----------
fig,ax=plt.subplots(1,2,figsize=(14,5.6))
allc=MAT+TF; mus_all=[];gmax=[];cols=[]
for nm,mus,col,d in allc:
    if not os.path.exists(d+"/vg_data.csv"): continue
    V,o=load(d); g,f,m,Vj=o[2]
    ax[0].plot(V[m],g[m],"--",color=col,lw=2,label=f"{nm}  μ*={mus:.2f}")
    if Vj is not None and Vj<=40: ax[0].plot(Vj,g[m][-1],"x",color=col,ms=9,mew=2.2)
    idx=np.where(m)[0]
    for a,b in zip(idx[:-1],idx[1:]):
        if g[a]<0<=g[b]:
            t=-g[a]/(g[b]-g[a]); ax[0].plot(V[a]+t*(V[b]-V[a]),0,"o",color=col,ms=10,mec="k",zorder=6); break
    mus_all.append(mus); gmax.append(np.nanmax(g[m]) if m.any() else np.nan); cols.append(col)
ax[0].axhline(0,color="k",lw=1.3); ax[0].set_ylim(-2.0,0.9); ax[0].set_xlim(0,40)
ax[0].set_xlabel("V [kn]"); ax[0].set_ylabel("g  —  ramo 2 (2º modo)"); ax[0].grid(alpha=.3)
l1=ax[0].legend(fontsize=9,title="μ* crescente →",loc="lower left"); ax[0].add_artist(l1)
ax[0].legend(handles=[STYLE[1],STYLE[2],
                      Line2D([0],[0],color="k",ls="none",marker="o",ms=9,mec="k",label="○ flutter, g=0")],
             fontsize=8,loc="upper left",framealpha=.95)
ax[0].set_title("solo μ* ≥ 2 attraversa lo zero  (curve troncate al 1º salto)",fontsize=10)

ax[1].axhline(0,color="k",lw=1.2)
ax[1].scatter(mus_all,gmax,c=cols,s=150,edgecolor="k",zorder=5)
for m_,gm,(nm,_,_,_) in zip(mus_all,gmax,allc):
    ax[1].annotate(nm,(m_,gm),textcoords="offset points",xytext=(8,-3),fontsize=8.5)
ax[1].axvspan(2.0,3.0,color="lightgreen",alpha=.35)
ax[1].annotate("soglia di flutter\nμ* ≈ 2",(2.05,-0.25),fontsize=9.5,color="darkgreen")
ax[1].set_xlabel("μ* = 0.1047·ρ_s/ρ_w·(sp/0.12)"); ax[1].set_ylabel("g max sul ramo 2")
ax[1].set_xscale("log"); ax[1].grid(alpha=.3,which="both")
ax[1].set_title("il flutter compare solo sopra μ* ≈ 2",fontsize=10)
ax[1].annotate("sp = spessore/corda della sezione\n(sp=0.20 → sezione al 20%)",
               (0.02,0.97),xycoords="axes fraction",va="top",fontsize=8.5,
               bbox=dict(fc="lightyellow",ec="grey",alpha=.95,boxstyle="round,pad=0.35"))
fig.suptitle("GOLAND_water — il flutter è governato dal rapporto di massa μ*",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(f"{OUT}/fig_combinata_massa_flutter.png",dpi=150)
print("fig_combinata_massa_flutter.png")
