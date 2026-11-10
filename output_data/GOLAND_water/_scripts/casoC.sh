#!/bin/bash
# Caso C (02/11/2026): caso B con t/c 12 %, semi-apertura/corda 7.0 (2.058 m), EI x50, GJ x25 per riavere f simili a B.
# Uso: casoC.sh run <hc> | puro <hc>
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_RHO_S=${RHO:-7850} GOLAND_WATER_E_MOD=200e9 GOLAND_WATER_NU=0.30 GOLAND_WATER_XEA=0.15 GOLAND_WATER_XCM=0.75
export GOLAND_WATER_MESH_NS=${NS:-56} GOLAND_WATER_MESH_NC=32 GOLAND_WATER_THICKNESS_FACTOR=0.12 GOLAND_WATER_EI_FRAC=${EIF:-0.0435} GOLAND_WATER_GJ_FRAC=${GJF:-0.005425} GOLAND_WATER_PSI=${PSI:-0.4} GOLAND_WATER_GAX_RATIO=100 GOLAND_WATER_SHARPY_ALPHA=0.5 GOLAND_WATER_SHARPY_VREF=${VREF:-40}
R="examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span ${SPAN:-2.058} --grid fineM32 --v-min 0.3 --v-max ${VMAX:-20} --n-v ${NV:-110}"
case $1 in
 run) env GOLAND_WATER_PK_BEM_MODE=delta GOLAND_WATER_DEPTH_CHORD=$2 ${3:+"GOLAND_WATER_QUVLM_NPZ=$3"} $PY $R ;;
 puro) env GOLAND_WATER_PK_CAPYTAINE=0 GOLAND_WATER_B2_REMOVAL=0 GOLAND_WATER_DEPTH_CHORD=$2 "GOLAND_WATER_QUVLM_NPZ=$3" $PY $R ;;
esac
