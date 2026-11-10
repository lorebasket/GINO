#!/bin/bash
# Test P, caso B (06/11/2026): tabelle M16 con ala immagine, somma alla tabella fine di rev E e scansioni delta.
# Una tabella alla volta, nice 19, 1 thread; aspetta >= 5 GB liberi prima di ogni tabella (PC condiviso con la mesh).
O="$(cd "$(dirname "$0")" && pwd)"
ROOT="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
SP=/home/lorenzo/anaconda3/envs/sharpy/bin/python; P=/home/lorenzo/anaconda3/envs/GINO/bin/python3
FINE="$ROOT/FLUID/PanelAero/Qjj/Quvlm_GOLAND_water_casoB_psiSHARPy_fineM32_krylov.npz"
N=GOLAND_water/casoB_numericSteel_revD/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p00087_gjfrac0p000217_rho7850_mesh24x32_delta_psip0p4_gafovr_al0p5_tf0p2_hc2p0
RB="$ROOT/output_data/GOLAND_water/casoB_numericSteel_revE/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p00087_gjfrac0p000217_rho7850_mesh24x32_delta_psip0p4_gafovr_al0p5_tf0p2_hc"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
read MU I11 I22 I33 EIS GJS <<<$(python3 -c "
import math
c=.294;t=.2*c;a=c/2;b=t/2;m=7850*.685*t*c;e=.6*c;ic=m*c*c/12
print(m,ic+m*e*e,.1*ic,.9*ic+m*e*e,.00087*200e9*math.pi/4*a*b**3/9.77221e6,.000217*200e9/2.6*math.pi*a**3*b**3/(a*a+b*b)/.987581e6)")
attendi_ram(){ while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 5 ]; do sleep 60; done; }
gaf(){ # <profondita' m> <nome>
  [ -s "$O/$2.npz" ] && return
  attendi_ram
  (cd "$ROOT/FLUID/sharpy" && PYTHONPATH=. nice -n 19 /usr/bin/time -v $SP gino_bem/precompute_quvlm_goland.py --grid fine \
    --v-ref 40 --num-modes-gino 4 --gino-name "$N" --krylov --main-ea .15 --main-cg .75 --main-chord .294 \
    --wing-semispan .882 --theta-scale .147 --sharpy-ei-scale $EIS --sharpy-gj-scale $GJS --sharpy-mu $MU \
    --sharpy-i11 $I11 --sharpy-i22 $I22 --sharpy-i33 $I33 --sharpy-alpha .5 --sharpy-psi .4 \
    --image-depth $1 --out "$O/$2.npz" > "$O/$2.log" 2>&1)
  echo "$(date +%T) $2 rc=$?"
}
scan(){ # <hc> <gaf|""> <uscita>
  nice -n 19 $P "$ROOT/output_data/GOLAND_water/_scripts/gwscan.py" "$RB$1" delta --vmin-kn 7.5 --vmax-kn 12 --dv 0.02 \
    ${2:+--gaf "$2"} --out "$O/$3" 2>&1 | tail -1 | sed "s|^|$3 |"
}
# 0) identita': scansione di rev E rifatta con gwscan e la tabella fine originale
scan 0p4 "" scan_ident_hc0p4.txt
# 1) tabelle M16: immagine lontana (riferimento) e quattro sommergenze
gaf 14.7 m16_img_h14p7
for x in "2p0 0.588" "1p0 0.294" "0p5 0.147" "0p4 0.1176"; do set -- $x; gaf $2 m16_img_hc$1; done
# 2) somma alla tabella fine e scansioni
for h in 2p0 1p0 0p5 0p4; do
  $SP "$O/../somma_delta.py" "$FINE" "$O/m16_img_h14p7.npz" "$O/m16_img_hc$h.npz" "$O/Q_img_hc$h.npz" | sed "s|^|hc$h |"
  scan $h "$O/Q_img_hc$h.npz" scan_img_hc$h.txt
done
echo "$(date +%T) fine"
