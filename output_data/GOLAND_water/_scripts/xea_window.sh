#!/bin/bash
# xea=0.18 gave no flutter (EA on the AC -> no aero moment about the EA; g stuck at
# the -0.0177 structural floor). xea=0.69 gave divergence at 17.9 kn before flutter.
# Scan the window between them. Also raise EI to bring BENDING up toward torsion
# (freq ratio was 3.8; flutter wants ~2) -- EI does not affect divergence, so it is
# the free lever. Deep water only (h/c=2) to pick the design; depth sweep afterwards.
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_RHO_S=1600 GOLAND_WATER_E_MOD=70e9 GOLAND_WATER_NU=0.30
export GOLAND_WATER_GJ_FRAC=0.0015 GOLAND_WATER_XCM=0.65
export GOLAND_WATER_SHARPY_ALPHA=0.0 GOLAND_WATER_DEPTH_CHORD=2.0
for xea in 0.25 0.35; do for eif in 0.008 0.014; do
  echo "########## xea=$xea ei_frac=$eif ##########"
  GOLAND_WATER_XEA=$xea GOLAND_WATER_EI_FRAC=$eif \
    $PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 \
    --grid fineM32 --v-min 2.0 --v-max 40 --n-v 96 2>&1 | grep -E "dry f|^ *[0-9],|Branch,|FAILED|Total"
  echo
done; done
echo "XEAWINDOW COMPLETE"
