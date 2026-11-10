#!/bin/bash
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_RHO_S=1600 GOLAND_WATER_E_MOD=70e9 GOLAND_WATER_NU=0.30
export GOLAND_WATER_XEA=0.25 GOLAND_WATER_XCM=0.65
export GOLAND_WATER_EI_FRAC=0.008 GOLAND_WATER_GJ_FRAC=0.0015
export GOLAND_WATER_SHARPY_ALPHA=0.0 GOLAND_WATER_DEPTH_CHORD=2.0
$PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 \
  --grid fineM32 --v-min 2.0 --v-max 40 --n-v 96 2>&1 | tail -25
echo "OMEGAFIX TEST COMPLETE"
