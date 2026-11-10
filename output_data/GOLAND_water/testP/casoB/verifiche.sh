#!/bin/bash
# Caso B: (1) scansione 2-7.6 kn senza immagine (tabella di rev E) a h/c 0.4; (2) correzione immagine su M8 a h/c 0.4.
O="$(cd "$(dirname "$0")" && pwd)"
ROOT="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
SP=/home/lorenzo/anaconda3/envs/sharpy/bin/python; P=/home/lorenzo/anaconda3/envs/GINO/bin/python3
FINE="$ROOT/FLUID/PanelAero/Qjj/Quvlm_GOLAND_water_casoB_psiSHARPy_fineM32_krylov.npz"
RB="$ROOT/output_data/GOLAND_water/casoB_numericSteel_revE/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p00087_gjfrac0p000217_rho7850_mesh24x32_delta_psip0p4_gafovr_al0p5_tf0p2_hc"
N=GOLAND_water/casoB_numericSteel_revD/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p00087_gjfrac0p000217_rho7850_mesh24x32_delta_psip0p4_gafovr_al0p5_tf0p2_hc2p0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
read MU I11 I22 I33 EIS GJS <<<$(python3 -c "
import math
c=.294;t=.2*c;a=c/2;b=t/2;m=7850*.685*t*c;e=.6*c;ic=m*c*c/12
print(m,ic+m*e*e,.1*ic,.9*ic+m*e*e,.00087*200e9*math.pi/4*a*b**3/9.77221e6,.000217*200e9/2.6*math.pi*a**3*b**3/(a*a+b*b)/.987581e6)")
scan(){ nice -n 19 $P "$ROOT/output_data/GOLAND_water/_scripts/gwscan.py" "$RB$1" delta --vmin-kn 2 --vmax-kn 7.6 --dv 0.02 ${2:+--gaf "$2"} --out "$O/$3" 2>&1 | tail -1 | sed "s|^|$3 |"; }
scan 0p4 "" scan_noimg_basse_hc0p4.txt
for x in "14.7 m8_img_h14p7" "0.1176 m8_img_hc0p4"; do set -- $x
  (cd "$ROOT/FLUID/sharpy" && PYTHONPATH=. nice -n 19 $SP gino_bem/precompute_quvlm_goland.py --grid default \
    --v-ref 40 --num-modes-gino 4 --gino-name "$N" --krylov --main-ea .15 --main-cg .75 --main-chord .294 \
    --wing-semispan .882 --theta-scale .147 --sharpy-ei-scale $EIS --sharpy-gj-scale $GJS --sharpy-mu $MU \
    --sharpy-i11 $I11 --sharpy-i22 $I22 --sharpy-i33 $I33 --sharpy-alpha .5 --sharpy-psi .4 \
    --image-depth $1 --out "$O/$2.npz" > "$O/$2.log" 2>&1); echo "$2 rc=$?"; done
$SP "$O/../somma_delta.py" "$FINE" "$O/m8_img_h14p7.npz" "$O/m8_img_hc0p4.npz" "$O/Q_img_m8_hc0p4.npz"
scan 0p4 "$O/Q_img_m8_hc0p4.npz" scan_img_m8_basse_hc0p4.txt
