#!/bin/bash
# FIRST physical flutter of the study: steel, tf=0.40 (mu*=2.74), xea=0.15/xcm=0.75.
# Torsion branch crosses g=0 at 16.6 kn, 94% converged, 0.6% max freq jump, no
# divergence. Two depth sweeps:
#   A) as-is          -> flutter ~8.3 Hz, k_w*h ~ 19 at h/c=0.25 (deep, expect small dV)
#   B) stiffness /16  -> flutter ~2.1 Hz, k_w*h ~ 1.3 at h/c=0.25 (the sensitive band)
# B is the real test of whether the +89%/-49% added-mass swing reaches V_flutter.
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_RHO_S=7850 GOLAND_WATER_E_MOD=200e9 GOLAND_WATER_NU=0.30
export GOLAND_WATER_XEA=0.15 GOLAND_WATER_XCM=0.75
export GOLAND_WATER_THICKNESS_FACTOR=0.40 GOLAND_WATER_SHARPY_ALPHA=0.0
sweep () { # eif gjf vmax label
  for hc in 0.25 0.5 1.0 2.0; do
    echo "########## $4  h/c = $hc ##########"
    GOLAND_WATER_EI_FRAC=$1 GOLAND_WATER_GJ_FRAC=$2 GOLAND_WATER_DEPTH_CHORD=$hc \
      $PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 \
      --grid fineM32 --v-min 1.0 --v-max $3 --n-v 110 2>&1 \
      | grep -E "dry f|^ *[0-9],|Branch,|FAILED|Total"
    echo
  done
}
sweep 0.0024   0.00059    45  "A stiff (~8 Hz)"
sweep 0.00015  0.0000369  15  "B soft  (~2 Hz)"
echo "TFDEPTH COMPLETE"
