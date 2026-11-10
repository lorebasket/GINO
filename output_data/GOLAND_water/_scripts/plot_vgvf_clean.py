#!/usr/bin/env python3
"""Vg-Vf filtrato di un singolo caso: solo punti convergenti, troncato al primo
salto di ramo (>20%). Uso: plot_vgvf_clean.py <dir_caso> [out.png]"""
import csv, sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d=sys.argv[1]; out=sys.argv[2] if len(sys.argv)>2 else d+"/vg_vf_clean.png"
kn=1/0.514444
vg=list(csv.DictReader(open(d+"/vg_data.csv",encoding="utf-8",errors="replace")))
vf=list(csv.DictReader(open(d+"/vf_data.csv",encoding="utf-8",errors="replace")))
V=np.array([float(x["Velocity_m_s"]) for x in vg])*kn
def c_(r,n):
    o=[]
    for x in r:
        try:o.append(float(x[n]))
        except Exception:o.append(np.nan)
    return np.array(o)
fig,ax=plt.subplots(1,2,figsize=(13,5))
info=[]
for br,col,nm in ((1,"tab:blue","ramo 1 (1º modo)"),(2,"crimson","ramo 2 (2º modo)")):
    g=c_(vg,f"Damping_Branch_{br}"); f=c_(vf,f"Frequency_Hz_Branch_{br}")
    ok=np.array([x.get(f"Converged_Branch_{br}")=="1" for x in vg])&np.isfinite(g)&np.isfinite(f)
    idx=np.where(ok)[0]; cut=len(V); Vj=None
    for a,b in zip(idx[:-1],idx[1:]):
        if f[a]>1e-9 and abs(f[b]-f[a])/f[a]>0.20: cut=b; Vj=V[a]; break
    m=ok.copy(); m[cut:]=False
    ax[0].plot(V[m],g[m],color=col,lw=2.2,label=nm)
    ax[1].plot(V[m],f[m],color=col,lw=2.2,label=nm)
    if Vj is not None:
        for a_ in ax: a_.axvline(Vj,color=col,ls=":",lw=1.5)
        ax[0].annotate(f"salto di ramo\n{Vj:.1f} kn",(Vj,ax[0].get_ylim()[0]),color=col,
                       fontsize=8.5,ha="center",va="bottom")
    ii=np.where(m)[0]
    for a,b in zip(ii[:-1],ii[1:]):
        if g[a]<0<=g[b]:
            t=-g[a]/(g[b]-g[a]); Vfl=V[a]+t*(V[b]-V[a])
            ax[0].plot(Vfl,0,"o",color=col,ms=12,mec="k",zorder=6)
            ax[0].annotate(f"flutter {Vfl:.2f} kn\n{f[a]:.2f} Hz",(Vfl,0),
                           textcoords="offset points",xytext=(12,14),fontsize=10,
                           color=col,fontweight="bold")
            info.append((br,Vfl,f[a])); break
ax[0].axhline(0,color="k",lw=1.3); ax[0].set_xlabel("V [kn]"); ax[0].set_ylabel("g = σ/ω")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3); ax[0].set_title("V–g (solo punti convergenti)",fontsize=11)
ax[1].set_xlabel("V [kn]"); ax[1].set_ylabel("f [Hz]"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
ax[1].set_title("V–f",fontsize=11)
fig.suptitle(os.path.basename(d).replace("GOLAND_water_",""),fontsize=10,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(out,dpi=150)
print("salvato:",out)
for br,v,f0 in info: print(f"   ramo {br}: flutter {v:.2f} kn @ {f0:.3f} Hz")
