#!/bin/bash
# Test O (T1.7) su E3: parte circolatoria di Q scalata di f(h/c) (immagine phi = 0, fattore_immagine.json).
# Riferimento depth-blind: scan_delta.txt dei run di rev E (f = 1). Uscite in questa cartella.
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/output_data/GOLAND_water"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/home/lorenzo/anaconda3/envs/GINO/bin/python3; G=_scripts/gwscan.py; O="$PWD/testO"
R=runs/GOLAND_water_c0p294_s1p029_xea0p15_xcm0p75_eifrac0p24_gjfrac0p030_rho7850_mesh28x32
F=$(python3 -c "import json;print(' '.join('%g:%.4f:%.4f'%(r['hc'],r['f2d'],r['f3d']) for r in json.load(open('$O/fattore_immagine.json')) if r['hc']!=0.75))")
for t in $F; do
  IFS=: read h f2 f3 <<< "$t"; ht=$(python3 -c "print(('%.1f' % $h).replace('.','p'))")
  d=$(ls -d ${R}_delta_psip0p4_*al0p5_gax100_hc$ht | head -1)
  for k in 2d:$f2 3d:$f3; do
    tag=${k%%:*}; f=${k#*:}
    ($P $G "$d" delta --vmin-kn 36 --vmax-kn 64 --dv 0.05 --circ-factor $f --out "$O"/scan_hc${ht}_$tag.txt 2>&1 | tail -1 | sed "s|^|hc$ht $tag f=$f |") &
  done
done
d=$(ls -d ${R}_delta_psip0p4_*al0p5_gax100_hc2p0 | head -1)
($P $G "$d" delta --vmin-kn 36 --vmax-kn 64 --dv 0.05 --circ-factor 1.000001 --out "$O"/scan_hc2p0_identita.txt 2>&1 | tail -1 | sed "s|^|hc2p0 identita |") &
wait; echo fine
