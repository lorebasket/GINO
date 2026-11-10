#!/bin/bash
# Test P: tabella fine + effetto immagine (M16, riferimento = immagine lontana a h = 14.7 m) e scansione delta
O="$(cd "$(dirname "$0")" && pwd)"
SP=/home/lorenzo/anaconda3/envs/sharpy/bin/python; P=/home/lorenzo/anaconda3/envs/GINO/bin/python3
G="$O/../_scripts/gwscan.py"
FINE="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/FLUID/PanelAero/Qjj/Quvlm_GOLAND_water_c0p294_s1p029_xea0p15_xcm0p75_eifrac0p24_gjfrac0p030_rho7850_mesh28x32_delta_psip0p4_al0p5_gax100_hc2p0_fineM32_krylov.npz"
R="$O/../runs/GOLAND_water_c0p294_s1p029_xea0p15_xcm0p75_eifrac0p24_gjfrac0p030_rho7850_mesh28x32_delta_psip0p4"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for h in 2p0 1p0 0p5 0p4; do
  $SP "$O/somma_delta.py" "$FINE" "$O/m16_img_h14p7.npz" "$O/m16_img_hc$h.npz" "$O/Q_img_hc$h.npz" | sed "s|^|hc$h |"
done
for h in 2p0 1p0 0p5 0p4; do
  case $h in 2p0) D=${R}_al0p5_gax100_hc2p0;; *) D=${R}_gafovr_al0p5_gax100_hc$h;; esac
  ($P "$G" "$D" delta --vmin-kn 36 --vmax-kn 64 --dv 0.05 --gaf "$O/Q_img_hc$h.npz" --out "$O/scan_img_hc$h.txt" 2>&1 | tail -1 | sed "s|^|hc$h |") &
done; wait
