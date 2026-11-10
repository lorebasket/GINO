#!/bin/bash
# Test P (05/11/2026): GAF UVLM di E3 con l'ala immagine della superficie libera (phi = 0) a +2h.
# Uso: gaf_img.sh <hc_run: 2p0|1p0|0p5|0p4> <h [m], 0 = senza immagine> <nome uscita> [griglia, default fineM32]
O="$(cd "$(dirname "$0")" && pwd)"
R=GOLAND_water_c0p294_s1p029_xea0p15_xcm0p75_eifrac0p24_gjfrac0p030_rho7850_mesh28x32_delta_psip0p4
case $1 in 2p0) T=${R}_al0p5_gax100_hc2p0;; *) T=${R}_gafovr_al0p5_gax100_hc$1;; esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/FLUID/sharpy"
PYTHONPATH=. /usr/bin/time -v /home/lorenzo/anaconda3/envs/sharpy/bin/python gino_bem/precompute_quvlm_goland.py \
  --grid ${4:-fineM32} --v-ref 40 --num-modes-gino 4 --gino-name "GOLAND_water/runs/$T" --krylov \
  --main-ea .15 --main-cg .75 --main-chord .294 --wing-semispan 1.029 --theta-scale .147 \
  --sharpy-ei-scale 0.0031128013396362793 --sharpy-gj-scale 0.005839274421070091 \
  --sharpy-mu 55.774557720000004 --sharpy-i11 2.137278820848091 --sharpy-i22 0.040174413925715996 \
  --sharpy-i33 2.0971044069223748 --sharpy-alpha .5 --sharpy-psi .4 --k-scale 1.0 \
  --image-depth $2 --out "$O/$3.npz" > "$O/$3.log" 2>&1
echo "$3 rc=$?"
