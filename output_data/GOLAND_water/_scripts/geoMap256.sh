# Mappa S(h/c, k_w h): caso E3 in scala geometrica (stesso acciaio, stesse frazioni EI/GJ -> V_f invariata, k_w h ~ 1/s)
cd "/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/output_data/GOLAND_water"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NS=28 EIF=0.24 GJF=0.030 GOLAND_WATER_NUM_MODES_FLUTTER=4 VREF=40 VMAX=40 NV=200
QD="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/FLUID/PanelAero/Qjj"
P=/home/lorenzo/anaconda3/envs/GINO/bin/python3; G="/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/output_data/GOLAND_water/_scripts/gwscan.py"
mkdir -p freq_coverage/mappa
for s in 256; do
  c=$(python3 -c "print('%g'%(0.294*$s))"); sp=$(python3 -c "print('%g'%(1.029*$s))")
  CHORD=$c SPAN=$sp bash _scripts/casoD.sh run 2.0 > freq_coverage/mappa/s${s}_d2.log 2>&1; echo "s$s gaf $?"
  ct=$(echo $c | sed 's/\./p/'); st=$(echo $sp | sed 's/\./p/')
  q=$(ls "$QD"/Quvlm_GOLAND_water_c${ct}_s${st}_*eifrac0p24_*hc2p0_fineM32_krylov.npz | head -1); echo "  $q"
  for h in 1.0 0.5 0.4; do (CHORD=$c SPAN=$sp bash _scripts/casoD.sh run $h "$q" > freq_coverage/mappa/s${s}_d$h.log 2>&1; echo "  s$s hc$h $?") & done
  (CHORD=$c SPAN=$sp bash _scripts/casoD.sh puro 2.0 "$q" > freq_coverage/mappa/s${s}_puro.log 2>&1; echo "  s$s puro $?") &
  wait
  for d in runs/GOLAND_water_c${ct}_s${st}_*eifrac0p24_*; do
    st_=delta; case $d in *UVLMpuro*) st_=uvlm;; esac
    ($P "$G" "$d" $st_ --vmin-kn 15 --vmax-kn 75 --dv 0.1 2>&1 | tail -1 | sed "s|^|  s$s ${d##*_} |") &
  done; wait
done; echo all
