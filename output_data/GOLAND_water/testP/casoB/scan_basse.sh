#!/bin/bash
# Caso B con immagine: V_f sotto 7.5 kn a h/c <= 1, scansioni 2-7.6 kn (una alla volta, nice 19)
O="$(cd "$(dirname "$0")" && pwd)"
ROOT="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
RB="$ROOT/output_data/GOLAND_water/casoB_numericSteel_revE/GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p00087_gjfrac0p000217_rho7850_mesh24x32_delta_psip0p4_gafovr_al0p5_tf0p2_hc"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for h in 1p0 0p5 0p4; do
  nice -n 19 /home/lorenzo/anaconda3/envs/GINO/bin/python3 "$ROOT/output_data/GOLAND_water/_scripts/gwscan.py" "$RB$h" delta \
    --vmin-kn 2 --vmax-kn 7.6 --dv 0.02 --gaf "$O/Q_img_hc$h.npz" --out "$O/scan_img_basse_hc$h.txt" 2>&1 | tail -1 | sed "s|^|hc$h |"
done
