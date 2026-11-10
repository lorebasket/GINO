"""Superficie immagine della superficie libera (phi = 0, alto Froude) per la GAF UVLM di SHARPy.

Un'ala sotto la superficie libera a profondita' h, con Fn_h = U/sqrt(g h) >> 1 e omega^2 h/g >> 1,
vede la superficie come un piano a perturbazione di potenziale nulla. Al primo ordine (ala quasi piana)
la condizione e' soddisfatta esattamente da una copia dell'ala TRASLATA di +2h, con la stessa cinematica
(stessa traslazione verticale, stessa rotazione, stessa incidenza): per simmetria la sua circolazione
risulta concorde con quella dell'ala vera e il potenziale si annulla sul piano a quota h.

Implementazione nel caso SHARPy a due semiali (GolandControlSurface):

* ogni elemento di trave viene duplicato sugli stessi nodi, con massa e rigidezza dimezzate in entrambe
  le copie (la trave assemblata e' identica); la copia porta la superficie immagine, con la linea media
  del profilo spostata di 2h/c;
* il guadagno trave -> griglia della superficie immagine viene sostituito con quello della superficie vera
  (senza, la copia ruotando attorno all'asse elastico trasla di 2h*theta in corda);
* le forze della superficie immagine non arrivano alla trave: escluse dalla soluzione statica, dal
  guadagno griglia -> trave e dai termini di rigidezza aerodinamica (Kss).

Le uscite Q restano quindi le forze generalizzate della sola ala vera, calcolate con l'immagine presente.
"""
from __future__ import annotations

import h5py
import numpy as np


def add_image_surfaces(case_base: str, depth: float) -> list[int]:
    """Duplica elementi e superfici in <case_base>.fem.h5 / .aero.h5; ritorna gli indici delle superfici immagine."""
    with h5py.File(case_base + ".fem.h5", "r+") as f:
        ne = int(f["num_elem"][()])
        X = f["coordinates"][()]
        conn = f["connectivities"][()]
        fd = f["frame_of_reference_delta"][()]
        for key in ("connectivities", "frame_of_reference_delta", "structural_twist"):
            d = f[key][()]
            del f[key]
            f[key] = np.concatenate([d, d])
        for key, off in (("elem_stiffness", 0), ("elem_mass", 0)):
            d = f[key][()]
            del f[key]
            f[key] = np.concatenate([d, d + off])
        bn = f["beam_number"][()]
        del f["beam_number"]
        f["beam_number"] = np.concatenate([bn, bn + bn.max() + 1])
        for key in ("stiffness_db", "mass_db"):
            d = 0.5 * f[key][()]
            del f[key]
            f[key] = d
        del f["num_elem"]
        f["num_elem"] = 2 * ne
    # verso dello z locale (z_b = x_b ^ y_b) di ogni elemento in terna A: la linea media dell'immagine
    # va spostata verso +z_A su entrambe le semiali
    zsign = np.zeros(ne)
    for e in range(ne):
        xb = X[conn[e, 1]] - X[conn[e, 0]]
        zsign[e] = np.sign(np.cross(xb / np.linalg.norm(xb), fd[e, 0])[2])
    if np.any(zsign == 0):
        raise RuntimeError("terna locale degli elementi non definita")
    with h5py.File(case_base + ".aero.h5", "r+") as f:
        chord = f["chord"][()]
        ad = f["airfoil_distribution"][()]
        n_af = len(f["airfoils"])
        sd = f["surface_distribution"][()]
        n_surf = int(sd.max()) + 1
        # un profilo immagine per ogni nodo-elemento (la corda e il verso di z possono variare)
        new_ad = np.zeros_like(ad)
        k = n_af
        for e in range(ne):
            for j in range(ad.shape[1]):
                af = f["airfoils"][str(ad[e, j])][()].copy()
                af[:, 1] += zsign[e] * 2.0 * depth / chord[e, j]
                f["airfoils"][str(k)] = af
                new_ad[e, j] = k
                k += 1
        for key in ("chord", "elastic_axis", "twist", "control_surface"):
            d = f[key][()]
            del f[key]
            f[key] = np.concatenate([d, d])
        del f["airfoil_distribution"]
        f["airfoil_distribution"] = np.concatenate([ad, new_ad])
        del f["surface_distribution"]
        f["surface_distribution"] = np.concatenate([sd, sd + n_surf])
        sm = f["surface_m"][()]
        del f["surface_m"]
        f["surface_m"] = np.concatenate([sm, sm])
    return list(range(n_surf, 2 * n_surf))


def install_patches(image_surfs: list[int]) -> None:
    """Esclude le forze dell'immagine dalla trave e lega la sua cinematica a quella dell'ala vera."""
    import sharpy.aero.utils.mapping as mapping
    from sharpy.linear.assembler import linearaeroelastic as lae

    n_real = min(image_surfs)
    orig_map = mapping.aero2struct_force_mapping

    def map_no_image(aero_forces, *a, **kw):
        forces = [np.zeros_like(F) if i in image_surfs else F for i, F in enumerate(aero_forces)]
        return orig_map(forces, *a, **kw)

    mapping.aero2struct_force_mapping = map_no_image

    orig_gains = lae.LinearAeroelastic.get_gebm2uvlm_gains

    def gains_image(self, data):
        ts = self.uvlm.tsaero0
        saved = [ts.forces[i].copy() for i in image_surfs]
        for i in image_surfs:
            ts.forces[i][:] = 0.0
        orig_gains(self, data)
        for i, F in zip(image_surfs, saved):
            ts.forces[i][:] = F
        KK = self.uvlm.sys.MS.KKzeta
        for i in image_surfs:
            r = i - n_real
            if KK[i] != KK[r]:
                raise RuntimeError("superficie immagine con pannellatura diversa da quella vera")
            si = slice(3 * sum(KK[:i]), 3 * sum(KK[:i + 1]))
            sr = slice(3 * sum(KK[:r]), 3 * sum(KK[:r + 1]))
            for name in ("Kdisp", "Kdisp_vel", "Kvel_disp", "Kvel_vel"):
                K = getattr(self, name)
                K[si, :] = K[sr, :]
            self.Kforces[:, si] = 0.0
        print(f"[image] guadagni: superfici immagine {image_surfs} legate a {[i - n_real for i in image_surfs]}, "
              "forze immagine escluse", flush=True)

    lae.LinearAeroelastic.get_gebm2uvlm_gains = gains_image
