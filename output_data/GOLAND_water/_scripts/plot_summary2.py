#!/usr/bin/env python3
import csv, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
OUT=f"{REPO}/output_data/GOLAND_water"; C=0.294; G=9.81; kn=1/0.514444

P={}
for d in sorted(glob.glob(f"{OUT}/bem_depth_probe/hc*")):
    hc=float(os.path.basename(d)[2:].replace("p","."))
    n=sorted(glob.glob(d+"/capytaine/modal_radiation_AB*.npz"))
    if n:
        z=np.load(n[0],allow_pickle=True)
        P[hc]=(np.asarray(z["omega"],float),np.asarray(z["added_mass"],float))
hcs=sorted(P); deep=hcs[-1]
A_dep={0.25:17.294,0.5:16.840,1.0:16.628,2.0:16.589}
B_dep={0.25:5.265,0.5:4.496,1.0:4.175,2.0:4.149}

fig,ax=plt.subplots(1,3,figsize=(16,5)) 
fig.suptitle("Quando la massa aggiunta dipende dalla profondità — e quando questo arriva a $V_{flutter}$",
             fontsize=13,fontweight="bold")

# (a) dA vs k_w*h, h/c come parametro
fs=np.geomspace(0.4,9,60)
for hc,col in zip(hcs[:-1],("tab:blue","tab:orange","tab:green","tab:red")):
    ww,A=P[hc]; wd,Ad=P[deep]; X=[];Y=[]
    for f in fs:
        w=2*np.pi*f
        if w<ww.min() or w>ww.max(): continue
        a=np.interp(w,ww,A[:,0,0]); ad=np.interp(w,wd,Ad[:,0,0])
        X.append(w*w/G*(hc*C)); Y.append(100*(a-ad)/ad)
    ax[0].semilogx(X,Y,"o-",ms=3,color=col,label=f"h/c = {hc:g}")
ax[0].axhline(0,color="k",lw=1)
ax[0].axvspan(0.2,0.9,color="gold",alpha=.3)
ax[0].annotate("cambio\ndi segno",(0.24,60),fontsize=9,color="#8a6d00")
ax[0].annotate("parete rigida\nA ↑",(0.045,55),fontsize=8.5,color="#444")
ax[0].annotate("pressure-release\nA ↓",(4,-40),fontsize=8.5,color="#444")
ax[0].set_xlabel("$k_w h=\\omega^2 h/g$"); ax[0].set_ylabel("$\\Delta A_{00}$ vs acqua profonda [%]")
ax[0].set_title("(a) il SEGNO è fissato da $k_w h$",fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3,which="both")

# (b) |dA| vs h/c  -> il vero cancello
med=[];mx=[]
for hc in hcs[:-1]:
    ww,A=P[hc]; wd,Ad=P[deep]; v=[]
    for f in fs:
        w=2*np.pi*f
        if w<ww.min() or w>ww.max(): continue
        a=np.interp(w,ww,A[:,0,0]); ad=np.interp(w,wd,Ad[:,0,0])
        v.append(abs(100*(a-ad)/ad))
    med.append(np.median(v)); mx.append(max(v))
ax[1].loglog(hcs[:-1],mx,"o-",color="crimson",lw=2,label="|ΔA| massimo")
ax[1].loglog(hcs[:-1],med,"s--",color="darkblue",lw=2,label="|ΔA| mediano")
ax[1].axvline(1.0,color="k",ls=":",lw=1.3)
ax[1].axvspan(1.0,4,color="lightgrey",alpha=.5)
ax[1].annotate("h/c > 1:\nnessun effetto\n(≤6%)",(1.15,20),fontsize=9)
ax[1].set_xlabel("h/c"); ax[1].set_ylabel("|ΔA| [%]")
ax[1].set_title("(b) ma il CANCELLO è h/c",fontsize=10); ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.3,which="both")

# (c) V_flutter: A vs B
for nm,D,col,f0,kwh in (("A rigida  8.3 Hz  ($k_w h$≈19)",A_dep,"tab:blue",8.3,19),
                         ("B morbida 2.1 Hz  ($k_w h$≈1.3)",B_dep,"crimson",2.1,1.3)):
    h=sorted(D); v=[D[x] for x in h]
    rel=[100*(x-v[-1])/v[-1] for x in v]
    ax[2].semilogx(h,rel,"o-",color=col,lw=2,ms=8,label=f"{nm}   ΔV={max(rel):.1f}%")
    for hh,rr,vv in zip(h,rel,v):
        ax[2].annotate(f"{vv:.2f} kn",(hh,rr),textcoords="offset points",xytext=(0,8),fontsize=8,ha="center",color=col)
ax[2].axhline(0,color="k",lw=1)
ax[2].set_xlabel("h/c"); ax[2].set_ylabel("$V_{flutter}$ rispetto a h/c=2  [%]")
ax[2].set_title("(c) stessa struttura, solo rigidezza diversa",fontsize=10)
ax[2].legend(fontsize=8.5); ax[2].grid(alpha=.3,which="both"); ax[2].set_ylim(-2,32)

fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig(f"{OUT}/bem_depth_vs_flutter_summary.png",dpi=150)
print("bem_depth_vs_flutter_summary.png aggiornata")
