import sys,glob,os,json,numpy as np
sys.path.insert(0,sys.argv[1]+"/_scripts")
src=open(sys.argv[1]+"/_scripts/plot_due_casi.py").read().split("CASI={")[0]
exec(src)
out={}
for k,gl,ug in [("A",OUT+"/casoA_numeric_"+REV+"/*_hc*",OUT+"/casoA_numeric_"+REV+"/*UVLMpuro*"),
                ("B",OUT+"/casoB_numericSteel_"+REV+"/*_hc*",OUT+"/casoB_numericSteel_"+REV+"/*UVLMpuro*")]:
    rows=[]
    for d in sorted(glob.glob(gl)):
        if "UVLMpuro" in d or not os.path.exists(d+"/vg_data.csv"): continue
        hc=float(os.path.basename(d).split("_hc")[1].replace("p","."))
        V,g,f,m=load(d); vf,ff=cross(V,g,f,m)
        V1,g1,f1,m1=load(d,1)
        i=np.where(m)[0][0]; j=np.where(m1)[0][0]
        rows.append(dict(hc=hc,V=vf,f=ff,f1w=f1[j],f2w=f[i]))
    rows.sort(key=lambda r:r["hc"])
    u=[x for x in glob.glob(ug) if os.path.exists(x+"/vg_data.csv")]
    V,g,f,m=load(u[0]); vu,fu=cross(V,g,f,m)
    out[k]=dict(rows=rows,uvlm=dict(V=vu,f=fu,dir=os.path.basename(u[0])))
    for r in rows: print(k,r)
    print(k,"UVLM",vu,fu)
json.dump(out,open(sys.argv[2],"w"),indent=1,default=float)
