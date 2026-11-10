#!/bin/bash
# Casi A e B con mesh BEM 24x32 (convergenza superficie libera, 14/10/2026); GAF UVLM riusata.
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3
Q=FLUID/PanelAero/Qjj
export GOLAND_WATER_RHO_S=7850 GOLAND_WATER_E_MOD=200e9 GOLAND_WATER_NU=0.30 GOLAND_WATER_XEA=0.15 GOLAND_WATER_XCM=0.75
export GOLAND_WATER_MESH_NS=24 GOLAND_WATER_MESH_NC=32
run () { # caso hc
  if [ $1 = A ]; then
    E="GOLAND_WATER_THICKNESS_FACTOR=0.40 GOLAND_WATER_EI_FRAC=0.00015 GOLAND_WATER_GJ_FRAC=0.0000369"
    T=GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p00015_gjfrac0p0000369_rho7850_tf0p4_hc$3; V="--v-min 1.0 --v-max 15"
  else
    E="GOLAND_WATER_THICKNESS_FACTOR=0.20 GOLAND_WATER_EI_FRAC=0.00087 GOLAND_WATER_GJ_FRAC=0.000217 GOLAND_WATER_PSI=0.4 GOLAND_WATER_SHARPY_ALPHA=0.5"
    T=GOLAND_water_c0p294_s0p882_xea0p15_xcm0p75_eifrac0p00087_gjfrac0p000217_rho7850_psip0p4_al0p5_tf0p2_hc$3; V="--v-min 0.5 --v-max 20"
  fi
  env $E $EXTRA GOLAND_WATER_DEPTH_CHORD=$2 GOLAND_WATER_QUVLM_NPZ=$Q/Quvlm_${T}_fineM32_krylov.npz \
    $PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 --grid fineM32 $V --n-v 110 \
    > output_data/GOLAND_water/freq_coverage/run_$1_$3${TAGLOG}.log 2>&1
  echo "fatto $1 $3"
}
run "$@"
