#!/bin/bash
# e/c = 0.10 (xea 0.21 / xcm 0.31) gives a genuine bending/torsion pair at ratio 1.43
# -- the coalescence condition the only-ever-fluttering case had at 1.11, and which
# e/c = 0.47 made unreachable. Stiffness scaled along the constant-ratio line so the
# WET frequencies sweep k_w*h through 1 at h/c = 0.25, where dA/dh is largest.
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_RHO_S=1600 GOLAND_WATER_E_MOD=70e9 GOLAND_WATER_NU=0.30
export GOLAND_WATER_XEA=0.21 GOLAND_WATER_XCM=0.31
export GOLAND_WATER_SHARPY_ALPHA=0.0 GOLAND_WATER_DEPTH_CHORD=2.0
set -- "0.030 0.0008" "0.015 0.0004" "0.0075 0.0002"
for pair in "$@"; do
  set -- $pair; eif=$1; gjf=$2
  echo "########## ei_frac=$eif gj_frac=$gjf ##########"
  GOLAND_WATER_EI_FRAC=$eif GOLAND_WATER_GJ_FRAC=$gjf \
    $PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 \
    --grid fineM32 --v-min 2.0 --v-max 40 --n-v 96 2>&1 | grep -E "dry f|^ *[0-9],|Branch,|FAILED|Total"
  echo
done
echo "DESIGNFINAL COMPLETE"
