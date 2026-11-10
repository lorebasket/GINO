#!/usr/bin/env python3
"""A(omega) vs FREQUENZA in Hz, asse lineare, con segnate le frequenze di flutter
reali dei nostri casi. La frequenza e' quella di OSCILLAZIONE DELLA STRUTTURA nel
modo considerato: A(omega) e' la massa aggiunta che il fluido oppone a un corpo che
oscilla a quella omega (problema di radiazione Capytaine). Al flutter coincide con
la frequenza di flutter."""
import glob, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/output_data/GOLAND_water"
C=0.294; G=9.81
P={}
for d in sorted(glob.glob(f"{OUT}/bem_depth_probe_v2/hc*")):
    hc=float(os.path.basename(d)[2:].replace("p","."))
    n=sorted(glob.glob(d+"/capytaine/modal_radiation_AB*.npz"))
    if n:
        z=np.load(n[0],allow_pickle=True)
        P[hc]=(np.asarray(z["omega"],float),np.asarray(z["added_mass"],float))
hcs=sorted(P); deep=hcs[-1]
wd,Ad=P[deep]
FMAX=9.0
f=np.linspace(0.30,FMAX,220); w=2*np.pi*f

fig,ax=plt.subplots(figsize=(11.5,6.2))
import matplotlib.cm as cm
COLS=cm.viridis(np.linspace(0,.88,len(hcs)-1))
for hc,col in zip(hcs[:-1],COLS):
    ww,A=P[hc]
    a=np.interp(w,ww,A[:,0,0]); ad=np.interp(w,wd,Ad[:,0,0])
    ax.plot(f,100*(a-ad)/ad,color=col,lw=2.3,label=f"h/c = {hc:g}")
ax.axhline(0,color="k",lw=1.2)

CASES=[("B morbida\n(nostro)",2.10,"crimson"),
       ("Model2t\n(paper)",6.74,"dimgray"),
       ("A rigida\n(nostro)",8.30,"navy")]
for nm,f0,col in CASES:
    ax.axvline(f0,color=col,ls="--",lw=1.6,alpha=.85)
    ax.annotate(nm,(f0,78),color=col,fontsize=9,ha="center",fontweight="bold",
                bbox=dict(fc="white",ec=col,alpha=.9,boxstyle="round,pad=0.25"))
ax.annotate("stale-mass 10.6 Hz →\n(fuori scala)",(8.75,-46),fontsize=8.5,color="dimgray",ha="right")

ax.axvspan(0.30,1.15,color="gold",alpha=.22)
ax.annotate("A AUMENTA\n(parete rigida)",(0.62,45),fontsize=9.5,ha="center",color="#7a6000",fontweight="bold")
ax.annotate("A DIMINUISCE (pressure-release)",(4.6,-57),fontsize=9.5,ha="center",color="#444",fontweight="bold")

ax.set_xlim(0.3,FMAX); ax.set_ylim(-65,95)
ax.set_xticks(np.arange(1,int(FMAX)+1))
ax.set_xlabel("frequenza di oscillazione della struttura  $f=\\omega/2\\pi$  [Hz]",fontsize=11)
ax.set_ylabel("$\\Delta A_{00}$ rispetto ad acqua profonda  [%]",fontsize=11)
ax.set_title("Massa aggiunta vs frequenza di vibrazione dell'ala (corda 0.294 m)\n"
             "sonda v2: griglia BEM a 51 punti (0.008–39.8 Hz); qui 0.3–9 Hz dove cadono i nostri modi",
             fontsize=11.5,fontweight="bold")
ax.legend(fontsize=10,loc="center right"); ax.grid(alpha=.3)

sec=ax.secondary_xaxis("top",functions=(lambda x:(2*np.pi*x)**2/G*(0.25*C),
                                        lambda y:np.sqrt(y*G/(0.25*C))/(2*np.pi)))
sec.set_xlabel("$k_w h = \\omega^2 h/g$   (a h/c = 0.25)",fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_A_vs_frequenza.png",dpi=150)
print("fig_A_vs_frequenza.png")
