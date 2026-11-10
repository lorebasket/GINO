#!/bin/bash
# Caso B rev E (02/11/2026): GAF con psi anche in SHARPy; h/c 2 e UVLM-only di controllo.
# Uso: casoB_revE.sh gaf | run <hc> <hctag> | puro
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI"
Q=$PWD/FLUID/PanelAero/Qjj/${QNAME:-Quvlm_GOLAND_water_casoB_psiSHARPy_fineM32_krylov.npz}
PY=/home/lorenzo/anaconda3/envs/GINO/bin/python3; SP=/home/lorenzo/anaconda3/envs/sharpy/bin/python
export GOLAND_WATER_RHO_S=7850 GOLAND_WATER_E_MOD=200e9 GOLAND_WATER_NU=0.30 GOLAND_WATER_XEA=0.15 GOLAND_WATER_XCM=0.75
export GOLAND_WATER_MESH_NS=24 GOLAND_WATER_MESH_NC=32 GOLAND_WATER_THICKNESS_FACTOR=0.20 GOLAND_WATER_EI_FRAC=0.00087 GOLAND_WATER_GJ_FRAC=0.000217 GOLAND_WATER_PSI=0.4 GOLAND_WATER_SHARPY_ALPHA=0.5
G=output_data/GOLAND_water; mkdir -p $G/casoB_revE
run(){ env "$@" "GOLAND_WATER_QUVLM_NPZ=$Q" $PY examples/GOLAND_water/run_custom_geometry_flutter.py --chord 0.294 --span 0.882 --grid fineM32 --v-min ${VMIN:-0.5} --v-max ${VMAX:-20} --n-v ${NV:-110}; }
case $1 in
 gaf) N=GOLAND_water/casoB_numericSteel_revD/$(basename $(ls -d $G/casoB_numericSteel_revD/*delta*hc2p0))
  read MU I11 I22 I33 EIS GJS <<<$(python3 -c "
import math
c=.294;t=.2*c;a=c/2;b=t/2;m=7850*.685*t*c;e=.6*c;ic=m*c*c/12
print(m,ic+m*e*e,.1*ic,.9*ic+m*e*e,.00087*200e9*math.pi/4*a*b**3/9.77221e6,.000217*200e9/2.6*math.pi*a**3*b**3/(a*a+b*b)/.987581e6)")
  cd FLUID/sharpy; PYTHONPATH=. $SP gino_bem/precompute_quvlm_goland.py --grid fineM32 --v-ref 40 --num-modes-gino 4 --gino-name "$N" --krylov --main-ea .15 --main-cg .75 --main-chord .294 --wing-semispan .882 --theta-scale .147 --out "$Q" --sharpy-ei-scale $EIS --sharpy-gj-scale $GJS --sharpy-mu $MU --sharpy-i11 $I11 --sharpy-i22 $I22 --sharpy-i33 $I33 --sharpy-alpha .5 --sharpy-psi .4 ;;
 puro) run GOLAND_WATER_PK_CAPYTAINE=0 GOLAND_WATER_B2_REMOVAL=0 GOLAND_WATER_DEPTH_CHORD=${2:-2.0} ;;
 run) run GOLAND_WATER_PK_BEM_MODE=delta GOLAND_WATER_DEPTH_CHORD=$2 ${3:+GOLAND_WATER_PK_TRACKING=$3} ;;
esac
