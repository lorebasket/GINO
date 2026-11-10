#!/bin/bash
# Same proven topology (xea=0.15 fwd of AC, xcm=0.75). Walk mu* UP from CFRP toward
# the value that fluttered, using REAL densities, with stiffness scaled by the same
# factor so the frequency (hence k_w*h) stays put. Find the lightest foil that still
# crosses g=0 -- that is the best achievable BEM sensitivity with real flutter.
#   CFRP 1600 -> mu*=0.168 (no flutter, g=-0.17)
#   Al   2700 -> mu*=0.283
#   steel7850 -> mu*=0.821
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_XEA=0.15 GOLAND_WATER_XCM=0.75
export GOLAND_WATER_SHARPY_ALPHA=0.0 GOLAND_WATER_DEPTH_CHORD=2.0
# ei/gj scaled by rho/1600 to hold frequency fixed
run () { # rho  E  nu  eif  gjf
  echo "########## rho_s=$1  ei=$4 gj=$5 ##########"
  GOLAND_WATER_RHO_S=$1 GOLAND_WATER_E_MOD=$2 GOLAND_WATER_NU=$3 \
  GOLAND_WATER_EI_FRAC=$4 GOLAND_WATER_GJ_FRAC=$5 \
    $PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 \
    --grid fineM32 --v-min 2.0 --v-max 45 --n-v 110 2>&1 | grep -E "dry f|^ *[0-9],|Branch,|FAILED|Total"
  echo
}
run 2700 69e9  0.33 0.0253 0.00624
run 7850 200e9 0.30 0.0736 0.01815
echo "MASSLADDER COMPLETE"
