#!/bin/bash
# Keep e/c = 0.10 (ratio 1.43, the coalescence condition) and move xea toward the AC,
# which is the one divergence lever left: V_div ~ 1/sqrt(xea - 0.175).
# xea 0.21 -> 27.5 kn, 0.19 -> ~42 kn, 0.18 -> ~73 kn. Flutter is driven by the
# INERTIAL coupling (CG 0.10c aft of EA), which survives xea -> AC, so V_flutter
# should stay put while V_div runs away above it.
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_RHO_S=1600 GOLAND_WATER_E_MOD=70e9 GOLAND_WATER_NU=0.30
export GOLAND_WATER_SHARPY_ALPHA=0.0 GOLAND_WATER_DEPTH_CHORD=2.0
for pair in "0.18 0.28" "0.19 0.29"; do
  set -- $pair; xea=$1; xcm=$2
  for sp in "0.030 0.0008" "0.015 0.0004"; do
    set -- $sp; eif=$1; gjf=$2
    echo "########## xea=$xea xcm=$xcm ei=$eif gj=$gjf ##########"
    GOLAND_WATER_XEA=$xea GOLAND_WATER_XCM=$xcm \
    GOLAND_WATER_EI_FRAC=$eif GOLAND_WATER_GJ_FRAC=$gjf \
      $PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 \
      --grid fineM32 --v-min 2.0 --v-max 60 --n-v 116 2>&1 | grep -E "dry f|^ *[0-9],|Branch,|FAILED|Total"
    echo
  done
done
echo "XEASWEEP2 COMPLETE"
