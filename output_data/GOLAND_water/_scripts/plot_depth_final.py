#!/usr/bin/env python3
"""Figura finale: V_flutter vs h/c sul caso B, 10 profondita' da 0.15 a 2."""
import csv, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/output_data/GOLAND_water"
kn=1/0.514444; C=0.294; G=9.81
rows=[]
for d in glob.glob(OUT+"/sweeps/tf04_depth_B_soft/*_hc*"):
    tag=os.path.basename(d).split("_hc")[1]
    hc=float(tag.replace("p","."))
    vg=list(csv.DictReader(open(d+"/vg_data.csv",encoding="utf-8",errors="replace")))
    vf=list(csv.DictReader(open(d+"/vf_data.csv",encoding="utf-8",errors="replace")))
    V=np.array([float(x["Velocity_m_s"]) for x in vg])*kn
    def c_(r,n):
        o=[]
        for x in r:
            try:o.append(float(x[n]))
            except Exception:o.append(np.nan)
        return np.array(o)
    g=c_(vg,"Damping_Branch_2"); f=c_(vf,"Frequency_Hz_Branch_2")
    ok=np.array([x.get("Converged_Branch_2")=="1" for x in vg])&np.isfinite(g)&np.isfinite(f)
    idx=np.where(ok)[0]
    for a,b in zip(idx[:-1],idx[1:]):
        if g[a]<0<=g[b]:
            t=-g[a]/(g[b]-g[a]); rows.append((hc,V[a]+t*(V[b]-V[a]),f[a])); break
rows.sort(); R=np.array(rows)
hc,Vf,ff=R[:,0],R[:,1],R[:,2]
dV=100*(Vf.max()-Vf.min())/Vf.min()

fig,ax=plt.subplots(1,2,figsize=(13,5.2))
ax[0].semilogx(hc,Vf,"o-",color="crimson",lw=2.2,ms=9,mec="k")
for a,b in zip(hc,Vf):
    ax[0].annotate(f"{b:.2f}",(a,b),textcoords="offset points",xytext=(0,9),fontsize=8.5,ha="center")
ax[0].axvline(1.0,color="k",ls=":",lw=1.3)
ax[0].annotate("h/c = 1",(1.05,Vf.max()-0.15),fontsize=9)
ax[0].set_xlabel("h/c  (sommergenza relativa)"); ax[0].set_ylabel("$V_{flutter}$ [kn]")
ax[0].set_title(f"$V_{{flutter}}$ vs sommergenza — ΔV = {dV:.1f}%",fontsize=11,fontweight="bold")
ax[0].grid(alpha=.3,which="both")

ax2=ax[1]; ax2.semilogx(hc,ff,"s-",color="navy",lw=2.2,ms=8,mec="k",label="$f_{flutter}$")
ax2.set_xlabel("h/c"); ax2.set_ylabel("$f_{flutter}$ [Hz]",color="navy")
ax2.tick_params(axis="y",labelcolor="navy"); ax2.grid(alpha=.3,which="both")
a3=ax2.twinx()
a3.semilogx(hc,(2*np.pi*ff)**2/G*(hc*C),"^--",color="darkgreen",lw=1.8,ms=7,label="$k_w h$")
a3.axhline(1,color="darkgreen",ls=":",lw=1.2)
a3.set_ylabel("$k_w h = \\omega^2 h/g$",color="darkgreen"); a3.tick_params(axis="y",labelcolor="darkgreen")
ax2.set_title("frequenza di flutter e parametro di superficie libera",fontsize=11)
fig.suptitle("Caso B — acciaio sp=0.40, $x_{EA}$=0.15, $x_{CM}$=0.75, μ*=2.74  (10 profondità, 90–99% convergenza)",
             fontsize=12,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(OUT+"/fig_depth_final.png",dpi=150)
print("fig_depth_final.png")
print(f"\n{'h/c':>6}{'Vf [kn]':>10}{'f [Hz]':>9}{'k_w*h':>9}")
for a,b,c in zip(hc,Vf,ff):
    print(f"{a:6.2f}{b:10.2f}{c:9.3f}{(2*np.pi*c)**2/G*(a*C):9.2f}")
print(f"\ndV_flutter = {dV:.1f}%   ({Vf.max():.2f} -> {Vf.min():.2f} kn)")
