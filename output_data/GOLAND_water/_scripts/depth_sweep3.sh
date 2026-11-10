#!/bin/bash
# Corrected design: EA essentially ON the aerodynamic centre (xea=0.18 vs AC 0.175)
# so static divergence is pushed to ~38 kn and flutter is driven by MASS unbalance
# (CG at 0.60c, well AFT of EA). Wet torsion ~2.2 Hz -> sits in the band where the
# Capytaine probe measured 34-46% added-mass spread with submergence.
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_RHO_S=1600 GOLAND_WATER_E_MOD=70e9 GOLAND_WATER_NU=0.30
export GOLAND_WATER_XEA=0.18 GOLAND_WATER_XCM=0.60
export GOLAND_WATER_EI_FRAC=0.0031 GOLAND_WATER_GJ_FRAC=0.0015
export GOLAND_WATER_SHARPY_ALPHA=0.0
for hc in 0.25 0.5 1.0 2.0; do
  echo "########## h/c = $hc ##########"
  GOLAND_WATER_DEPTH_CHORD=$hc \
    $PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 \
    --grid fineM32 --v-min 2.0 --v-max 40 --n-v 96 2>&1 | tail -30
  echo
done
echo "DEPTHSWEEP3 COMPLETE"
