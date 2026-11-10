#!/bin/bash
# The topology that actually produced clean flutter (100% converged, 0.8% max freq
# jump, monotonic): xea=0.15 (FORWARD of the AC at 0.175 -> no divergence),
# xcm=0.75 (e/c=0.60 -> strong inertial coupling). No coalescence involved: g goes
# positive with the frequency flat, which is the real criterion.
# Rebuilt here with PHYSICAL CFRP mass and stiffness scaled so the wet frequency
# lands at k_w*h ~ 1, instead of the stale-mass 10.6 Hz that sat in the deep plateau.
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_RHO_S=1600 GOLAND_WATER_E_MOD=70e9 GOLAND_WATER_NU=0.30
export GOLAND_WATER_XEA=0.15 GOLAND_WATER_XCM=0.75
export GOLAND_WATER_EI_FRAC=0.015 GOLAND_WATER_GJ_FRAC=0.0037
export GOLAND_WATER_SHARPY_ALPHA=0.0
for hc in 0.25 0.5 1.0 2.0; do
  echo "########## h/c = $hc ##########"
  GOLAND_WATER_DEPTH_CHORD=$hc \
    $PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 \
    --grid fineM32 --v-min 2.0 --v-max 45 --n-v 110 2>&1 | grep -E "dry f|^ *[0-9],|Branch,|FAILED|Total"
  echo
done
echo "WORKINGTOPO COMPLETE"
