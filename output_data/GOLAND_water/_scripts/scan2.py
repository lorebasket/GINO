import glob,re,os,sys,numpy as np
R="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/"
n=4
for d in sorted(glob.glob(R+"output_data/GOLAND_water/runs/"+sys.argv[1])):
    tag=re.sub(r".*_s(\d)","s\\1",os.path.basename(d)).split("_xea")[0]+"_"+d.split("_")[-1]
    if not os.path.exists(d+"/vg_data.csv"): print(tag,"NO DATA");continue
    log=open(d+"/flutter.log",errors="ignore").read()
    fq=re.findall(r"Mode \d: ([\d.]+)",log)[:4]
    tab=re.search(r"modal GAF table (.*?\.npz)",log).group(1)
    tab=tab if tab[0]=="/" else R+tab
    t=np.load(tab);k=t["k_list"];Q=t["Q"];m=k>=.58*k[-1]
    A0=np.array([[.5*997*np.polyfit(k[m]**2,Q[m,i,j].real,1)[0] for j in range(n)] for i in range(n)]);A0=.5*(A0+A0.T)
    h=np.load(max(glob.glob(d+"/capytaine/modal_radiation_AB_depth_*.npz"),key=os.path.getmtime),allow_pickle=True);i=np.argmin(abs(h["omega"]-20));A=h["added_mass"][i][:n,:n]
    bad=[(a,b) for a in range(n) for b in range(a+1,n) if np.sign(A[a,b])!=np.sign(A0[a,b]) and abs(A0[a,b])>.05]
    vg=np.genfromtxt(d+"/vg_data.csv",delimiter=",",names=True);vf=np.genfromtxt(d+"/vf_data.csv",delimiter=",",names=True);V=vg["Velocity_m_s"]
    out=[]
    for b in (1,2):
        g=vg[f"Damping_Branch_{b}"];f=vf[f"Frequency_Hz_Branch_{b}"];c=vg[f"Converged_Branch_{b}"]==1
        gmax=np.max(np.where(c&(f<30),g,-9))
        for j in range(len(V)-1):
            if c[j] and c[j+1] and g[j]<0<=g[j+1] and abs(f[j+1]-f[j])<.25*f[j] and f[j]<30 and V[j+1]>1:
                out.append((b,round(V[j]/.5144,2),round(f[j],2),round(g[j],3),round(g[j+1],3)))
        out.append(("gmax",b,round(float(gmax),3)))
    print(tag,"f",fq,"signbad",bad,"\n   ",out)
