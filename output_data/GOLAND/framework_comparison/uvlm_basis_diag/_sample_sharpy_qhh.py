
import sys
from pathlib import Path
import numpy as np
REPO = Path('/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI')
sys.path.insert(0, str(REPO / "sharpy"))
from gino_bem.run_goland_water_rho import _build_case
import sharpy.sharpy_main

V_ref = 165.0
k_probe = np.array([0.0, 0.2, 0.42, 0.8, 1.5], float)
route = REPO / "sharpy" / "gino_bem"
ws, *_ = _build_case(
    rho=1.02, u_inf=V_ref, alpha_deg=2.0, M=16, N=32, M_star_fact=10,
    num_modes=8, v_min=V_ref, v_max=V_ref, n_vel=1, velocity_list=None,
    route=route, case_tag="qhh_probe", use_krylov=True, newmark_damp=0.5e-4,
)
ws.config["SHARPy"]["flow"] = [
    "BeamLoader", "AerogridLoader", "StaticCoupled", "Modal", "LinearAssembler",
]
ws.config["StaticCoupled"]["structural_solver_settings"]["gravity_on"] = "off"
ws.config["LinearAssembler"]["linear_system_settings"]["beam_settings"]["gravity"] = "off"
ws.config.write()
data = sharpy.sharpy_main.main(["", ws.route + ws.case_name + ".sharpy"])
ls = data.linear.linear_system
ls.update(V_ref)
facts = ls.uvlm.sys.ScalingFacts
t_ref = float(facts["length"]) / V_ref
dimfac = float(facts["length"]) ** 2

def _var_slice(varlist, name):
    for v in varlist:
        if v.name == name:
            return slice(v.first_position, v.end_position)
    raise RuntimeError(name)

sl_q = _var_slice(ls.uvlm.ss.input_variables, "q")
sl_qd = _var_slice(ls.uvlm.ss.input_variables, "q_dot")
sl_Q = _var_slice(ls.uvlm.ss.output_variables, "Q")
w_s = k_probe * V_ref * t_ref
H = ls.uvlm.ss.freqresp(w_s)
nm = int(ls.beam.sys.num_modes)
Q = np.zeros((len(k_probe), nm, nm), complex)
for i, wsc in enumerate(w_s):
    q_sh = H[sl_Q, sl_q, i] + 1j * wsc * H[sl_Q, sl_qd, i]
    Q[i] = np.conj(dimfac * q_sh)[:nm, :nm]
out = Path('/media/lorenzo/Seagate Por1/recupero dati Linux/lorebasket/FSI/output_data/GOLAND/framework_comparison/uvlm_basis_diag/sharpy_live_qhh.npz')
np.savez(out, k_list=k_probe, Q=Q, nm=nm, V_ref_m_s=V_ref,
         omega_rad=np.asarray(ls.beam.sys.freq_natural)[:nm]*0+np.sqrt(
             np.diag(ls.beam.sys.K[:nm,:nm])) if False else np.asarray(ls.beam.sys.freq_natural)[:nm])
# freq_natural may already be rad/s or Hz — store raw
freq = np.asarray(ls.beam.sys.freq_natural, float).ravel()[:nm]
np.savez(out, k_list=k_probe, Q=Q, nm=nm, V_ref_m_s=V_ref, freq_natural_raw=freq,
         n_states=ls.uvlm.ss.states)
print("saved", out, "Q shape", Q.shape)
for i,k in enumerate(k_probe):
    print(f"k={k:.3f} Q00={Q[i,0,0]:.4g} Q01={Q[i,0,1]:.4g} Q11={Q[i,1,1]:.4g}")
