"""Griglia UVLM REALE del T semplificato (vertici zeta di SHARPy) -> npz, per i grafici di controllo.
Carica solo BeamLoader + AerogridLoader (niente UVLM lineare).
    [TFS_*=...] conda run -n sharpy python sharpy/gino_bem/dump_tfoil_simple_grid.py <out.npz>
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sharpy.sharpy_main
from gino_bem.tfoil_simple import TfoilSimple, GRIDS, G

grid = sys.argv[2] if len(sys.argv) > 2 else "n40c12e16"
M, N, Ms, Ne = GRIDS[grid]
route = Path(__file__).resolve().parent / "cases"; route.mkdir(exist_ok=True)
tag = "tfoil_simple_grid_" + G.tag().replace(".", "p")
ws = TfoilSimple(M=M, N=N, Mstar_fact=Ms, u_inf=1.0, N_elev=Ne, route=str(route) + "/", case_name=tag)
ws.clean_test_files(); ws.update_derived_params(); ws.set_default_config_dict()
ws.generate_aero_file(); ws.generate_fem_file()
ws.config["SHARPy"]["flow"] = ["BeamLoader", "AerogridLoader"]
ws.config["SHARPy"]["write_screen"] = "off"
ws.config["AerogridLoader"] = {"unsteady": "off", "aligned_grid": "on", "mstar": 1,
                               "freestream_dir": ["1", "0", "0"], "wake_shape_generator": "StraightWake",
                               "wake_shape_generator_input": {"u_inf": 1.0, "u_inf_direction": ["1", "0", "0"],
                                                              "dt": ws.main_chord / M}}
ws.config.write()
data = sharpy.sharpy_main.main(["", str(route / (tag + ".sharpy"))])
# zeta e' nel frame inerziale G (ruotato dal quaternione del roll): lo si riporta nel frame A
# del corpo, quello dei nodi della trave e delle GAF
import sharpy.utils.algebra as algebra
C_GA = algebra.quat2rotation(np.asarray(data.structure.timestep_info[0].quat))
zeta = [np.einsum("ij,jmn->imn", C_GA.T, np.asarray(z)) for z in data.aero.timestep_info[0].zeta]
pos_A = np.asarray(data.structure.timestep_info[0].pos)                  # nodi della trave in A, per controllo
np.savez(sys.argv[1], n_surf=len(zeta), **{"zeta%d" % i: z for i, z in enumerate(zeta)},
         beam=np.column_stack([ws.x, ws.y, ws.z]), pos_A=pos_A, quat=np.asarray(data.structure.timestep_info[0].quat), aero_node=ws.aero_node, wl=G.WL, z_root=G.Z_ROOT)
print("superfici", len(zeta), [z.shape for z in zeta], "nodi bagnati del timone", int(ws.aero_node[:ws.n_rudder_node].sum()),
      "/", ws.n_rudder_node)
