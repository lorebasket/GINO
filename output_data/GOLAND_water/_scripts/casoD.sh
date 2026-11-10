#!/bin/bash
# Caso D (03/11/2026): caso C AR14 (semi-apertura/corda 14) scalato geometricamente; CHORD e SPAN da ambiente.
# Uso: casoD.sh run <hc> | puro <hc>
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
export GOLAND_WATER_RHO_S=${RHO:-7850} GOLAND_WATER_E_MOD=200e9 GOLAND_WATER_NU=0.30 GOLAND_WATER_XEA=0.15 GOLAND_WATER_XCM=${XCM:-0.75}
export GOLAND_WATER_MESH_NS=${NS:-112} GOLAND_WATER_MESH_NC=32 GOLAND_WATER_THICKNESS_FACTOR=0.12 GOLAND_WATER_EI_FRAC=${EIF:-0.0435} GOLAND_WATER_GJ_FRAC=${GJF:-0.005425} GOLAND_WATER_PSI=${PSI:-0.4} GOLAND_WATER_GAX_RATIO=100 GOLAND_WATER_SHARPY_ALPHA=0.5 GOLAND_WATER_SHARPY_VREF=${VREF:-10}
R="examples/GOLAND_water/run_custom_geometry_flutter.py --chord ${CHORD:-0.2058} --span ${SPAN:-2.8812} --grid fineM32 --v-min 0.3 --v-max ${VMAX:-20} --n-v ${NV:-110}"
case $1 in
 run) env GOLAND_WATER_PK_BEM_MODE=delta GOLAND_WATER_DEPTH_CHORD=$2 ${3:+"GOLAND_WATER_QUVLM_NPZ=$3"} $PY $R ;;
 puro) env GOLAND_WATER_PK_CAPYTAINE=0 GOLAND_WATER_B2_REMOVAL=0 GOLAND_WATER_DEPTH_CHORD=$2 "GOLAND_WATER_QUVLM_NPZ=$3" $PY $R ;;
esac
