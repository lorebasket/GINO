#!/bin/bash
# thickness_factor is the ONLY lever on mu* besides density (chord/span cancel).
# Steel + thick section walks mu* 0.82 -> 2.74, toward the 4.55 that fluttered,
# while stiffness is re-scaled each step to hold f1 = 4.42 Hz so the wet frequency
# (and hence k_w*h) stays in the BEM-sensitive band. Deep water only: this asks one
# question -- does g cross zero at any realizable mu*?
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_RHO_S=7850 GOLAND_WATER_E_MOD=200e9 GOLAND_WATER_NU=0.30
export GOLAND_WATER_XEA=0.15 GOLAND_WATER_XCM=0.75
export GOLAND_WATER_SHARPY_ALPHA=0.0 GOLAND_WATER_DEPTH_CHORD=2.0
run () { # tf eif gjf
  echo "########## tf=$1 (mu*=$4) ei=$2 gj=$3 ##########"
  GOLAND_WATER_THICKNESS_FACTOR=$1 GOLAND_WATER_EI_FRAC=$2 GOLAND_WATER_GJ_FRAC=$3 \
    $PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 \
    --grid fineM32 --v-min 2.0 --v-max 45 --n-v 110 2>&1 | grep -E "dry f|^ *[0-9],|Branch,|FAILED|Total"
  echo
}
run 0.20 0.0091 0.00226 1.37
run 0.30 0.0041 0.00102 2.05
run 0.40 0.0024 0.00059 2.74
echo "TFLADDER COMPLETE"
