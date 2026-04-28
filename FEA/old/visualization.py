import numpy as np
from matplotlib import cm   
import matplotlib.pyplot as plt
import pandas as pd
import time
from .old import Bauchau_stiffness_matrix_assembly as Bauchau
#from beam_properties import dof_per_node

def plot_undeformed_shape(nodes):

    x_orig = np.array([node["position"][0] for node in nodes])
    y_orig = np.array([node["position"][1] for node in nodes])
    z_orig = np.array([node["position"][2] for node in nodes])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_orig, y_orig, z_orig, 'b-', linewidth=2, label='Undeformed shape')
    ax.scatter(x_orig, y_orig, z_orig, c='r', marker='o')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Undeformed Shape of the Beam')
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()

def plot_deformed_shape(nodes, displacements, section_breaks, scale_factor=1):
    """
    Plot the deformed shape of the beam
    """

    # Extract original positions
    x_orig = np.array([node["position"][0] for node in nodes])
    y_orig = np.array([node["position"][1] for node in nodes])
    z_orig = np.array([node["position"][2] for node in nodes])
    
    # Extract displacements
    x_disp = np.array([disp["x"] for disp in displacements])
    y_disp = np.array([disp["y"] for disp in displacements])
    z_disp = np.array([disp["z"] for disp in displacements])
    
    # Calculate deformed positions
    x_def = x_orig + scale_factor * x_disp
    y_def = y_orig + scale_factor * y_disp
    z_def = z_orig + scale_factor * z_disp
    
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.plot(x_orig*1000, x_disp*1000, label='X displacement')
    plt.plot(x_orig*1000, y_disp*1000, label='Y displacement')
    plt.plot(x_orig*1000, z_disp*1000, label='Z displacement')
    plt.ylabel('Tip displacement (mm)')
    plt.xlabel('X position (mm)')
    plt.title('Displacement along the beam')
    plt.grid(True)
    plt.legend()
    
    ## == plotting options for composite box beam == #
    #bending_slope = np.array([disp["rz"] for disp in displacements])*10000000  #hide for steel beam
    #plt.figure(figsize=(12, 6))
    #plt.plot(x_orig, bending_slope, label='Bending slope (rz)')
    #plt.xlabel('X position (m)')
    #plt.ylabel('Bending slope (rad)')
    #plt.title('Bending slope along the beam')
    #plt.grid(True)
    #plt.legend()
    
    plt.show()

def plot_all_stresses(nodes, stresses):
    """
    Plot all six strain components along the beam in subplots.

    Args:
        nodes: List of node dictionaries with positions
        strains: List of 6-component strain vectors per element
    """
    x_pos = [0.5 * (nodes[e]["position"][0] + nodes[e+1]["position"][0])
             for e in range(len(stresses))]

    labels = ["Axial εx", "Shear γxy", "Shear γxz", "Curvature κx", "Curvature κy", "Curvature κz"]

    plt.figure(figsize=(12, 12))
    for i in range(6):
        component = [stresses[i] for stresses in stresses]
        plt.subplot(3, 2, i + 1)
        plt.plot(x_pos, component, 'b-o')
        plt.xlabel('X position (m)')
        plt.ylabel(labels[i])
        plt.title(labels[i])
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle('Stresses Components Along the Beam', fontsize=16, y=1.02)
    plt.show()



def classify_modes(mode_shapes, n_nodes):
    """
    Classify modes based on dominant displacement/rotation DOFs.

    Args:
        mode_shapes: Mode shapes from modal analysis (total_dof × num_modes)
        n_nodes: Number of nodes

    Returns:
        List of tuples: (mode_number, dominant_mode_type, DOF_contributions_dict)
    """
    mode_types = []
    for i in range(mode_shapes.shape[1]):
        mode_shape = mode_shapes[:, i]
        energies = compute_mode_strain_energy_contributions(K_global, mode_shape, n_nodes)
        print(f"  Mode      {i+1}:")
        print(f"  Axial:    {energies['Axial']:.6e}")
        print(f"  Torsion:  {energies['Torsion']:.6e}")
        print(f"  Bending Y:{energies['Bending Y']:.6e}")
        print(f"  Bending Z:{energies['Bending Z']:.6e}")
        print(f"  Total:    {energies['Total']:.6e}")

        # Optional: classify by highest energy
        dominant_mode = max(energies.items(), key=lambda x: x[1] if x[0] != "Total" else -1)[0]
        print(f"→ Dominant deformation: {dominant_mode}\n")
    
    return mode_types


### mode shapes plotting
from typing import Iterable, Optional, Tuple

def _dof_index(node_id: int, comp: str, dof_per_node: int = 6) -> int:
    """
    Return the global DOF index for a given node (0-based) and component.
    comp in {'ux','uy','uz','rx','ry','rz'} for dof_per_node=6.
    """
    comp_map = {'ux':0, 'uy':1, 'uz':2, 'rx':3, 'ry':4, 'rz':5}
    base = node_id * dof_per_node
    return base + comp_map[comp]

def _extract_component_along_span(mode_vec_full: np.ndarray,
                                  n_nodes: int,
                                  comp: str = 'uy',
                                  dof_per_node: int = 6) -> np.ndarray:
    """
    Pull the translational/rotational component (e.g., 'uy') per node from a full-DOF mode vector.
    Returns shape (n_nodes,)
    """
    vals = np.zeros(n_nodes)
    for i in range(n_nodes):
        vals[i] = mode_vec_full[_dof_index(i, comp, dof_per_node)]
    return vals

def _nice_scale(y: np.ndarray) -> float:
    """Auto scaling to make shapes visible but not cartoonishly huge."""
    if not np.any(np.isfinite(y)):
        return 1.0
    rng = np.nanmax(np.abs(y))
    return 0.0 if rng == 0 else 0.2 / rng  # scale so max deflection ~ 0.2 units

# ---------- 1) mode shape plots ----------

def plot_mode_shapes(nodes: np.ndarray,
                     mode_shapes: np.ndarray,
                     frequencies_hz: Iterable[float],
                     dof_per_node: int = 6,
                     which_modes: Optional[Iterable[int]] = None,
                     components: Tuple[str, ...] = ('uy','uz'),
                     amplify: Optional[float] = None,
                     plot_3d: bool = False,
                     titlesuffix: str = "",
                     tight_layout: bool = True,
                     show=True):
                    
    nodes = np.asarray(nodes)
    if nodes.ndim == 1:
        # if only x is given
        nodes = np.column_stack([nodes, np.zeros_like(nodes), np.zeros_like(nodes)])

    n_nodes = nodes.shape[0]
    x = nodes[:, 0]
    n_dof, n_modes = mode_shapes.shape

    if which_modes is None:
        which_modes = list(range(1, n_modes+1))
    else:
        which_modes = list(which_modes)

    # 2D plots: each selected mode gets a figure with subplots per component
    for m in which_modes:
        idx = m - 1
        phi = mode_shapes[:, idx]
        fig, axs = plt.subplots(len(components), 1, figsize=(8, 3.0*len(components)))
        if not isinstance(axs, (list, np.ndarray)):
            axs = [axs]

        # scaling
        scales = {}
        for comp in components:
            y = _extract_component_along_span(phi, n_nodes, comp=comp, dof_per_node=dof_per_node)
            scales[comp] = _nice_scale(y) if amplify is None else amplify

        for ax, comp in zip(axs, components):
            y = _extract_component_along_span(phi, n_nodes, comp=comp, dof_per_node=dof_per_node)
            y_plot = y * scales[comp]
            ax.plot(x, np.zeros_like(x), linestyle='--', linewidth=1)
            ax.plot(x, y_plot, linewidth=2)
            ax.set_xlabel('x (span)')
            ax.set_ylabel(f'{comp} (scaled)')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Mode {m} — {frequencies_hz[idx]:.3f} Hz  {titlesuffix}".strip())

        if tight_layout:
            plt.tight_layout()

        # Optional simple 3D view (centerline with uy/uz deflection)
        if plot_3d:
            try:
                from mpl_toolkits.mplot3d import Axes3D  # noqa
                fig3d = plt.figure(figsize=(6, 4))
                ax3d = fig3d.add_subplot(111, projection='3d')
                uy = _extract_component_along_span(phi, n_nodes, 'uy', dof_per_node)
                uz = _extract_component_along_span(phi, n_nodes, 'uz', dof_per_node)
                s_uy = _nice_scale(uy) if amplify is None else amplify
                s_uz = _nice_scale(uz) if amplify is None else amplify
                ax3d.plot(xs=x, ys=uy*s_uy, zs=uz*s_uz, linewidth=2)
                ax3d.plot(xs=x, ys=np.zeros_like(x), zs=np.zeros_like(x), linestyle='--', linewidth=1)
                ax3d.set_xlabel('x'); ax3d.set_ylabel('uy'); ax3d.set_zlabel('uz')
                ax3d.set_title(f"Mode {m} — 3D view")
            except Exception as e:
                print(f"[plot_mode_shapes] 3D plot skipped: {e}")
        
        plt.show()


#### FRF plotting

def _mass_normalize_modes(Phi_ff: np.ndarray, M_ff: np.ndarray) -> np.ndarray:
    """
    Mass-normalize columns of Phi_ff so that Phi^T M Phi = I.
    """
    PhiM = Phi_ff.T @ M_ff
    norms = np.sqrt(np.sum(PhiM * Phi_ff.T, axis=1))  # diag entries of Phi^T M Phi
    norms[norms == 0] = 1.0
    return Phi_ff / norms

def plot_frf_modal(frequencies_hz: np.ndarray,
                   Phi_ff: np.ndarray,
                   omegas_rad: np.ndarray,
                   M_ff: np.ndarray,
                   dof_in_ff: int,
                   dof_out_ff: int,
                   zeta: float = 0.01,
                   per_mode_zeta: Optional[np.ndarray] = None,
                   quantity: str = "receptance",
                   show_peaks_at: Optional[Iterable[float]] = None,
                   title: str = "Modal FRF"):

    omega = 2*np.pi*frequencies_hz
    n_modes = Phi_ff.shape[1]
    PhiN = _mass_normalize_modes(Phi_ff, M_ff)  # mass-normalize
    # modal numerators for SISO (out,in)
    num = PhiN[dof_out_ff, :] * PhiN[dof_in_ff, :]

    if per_mode_zeta is None:
        per_mode_zeta = np.full(n_modes, zeta)

    H = np.zeros_like(omega, dtype=complex)
    for r in range(n_modes):
        wr = omegas_rad[r]
        zr = per_mode_zeta[r]
        denom = (-omega**2 + 2j*zr*wr*omega + wr**2)
        H += num[r] / denom

    if quantity == "mobility":
        H = 1j*omega*H
    elif quantity == "accelerance":
        H = -(omega**2)*H

    mag = np.abs(H)
    phase = np.angle(H, deg=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    ax1.plot(frequencies_hz, mag, linewidth=2)
    ax1.set_ylabel('|H|')
    ax1.grid(True, alpha=0.3)
    ax2.plot(frequencies_hz, phase, linewidth=1.5)
    ax2.set_ylabel('Phase (deg)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.grid(True, alpha=0.3)
    ax1.set_title(title)

    # Mark natural frequencies if supplied
    if show_peaks_at is not None:
        for f in show_peaks_at:
            ax1.axvline(f, linestyle='--', alpha=0.4)
            ax2.axvline(f, linestyle='--', alpha=0.4)

    plt.tight_layout()

def plot_frf_direct(frequencies_hz: np.ndarray,
                    K_ff: np.ndarray,
                    M_ff: np.ndarray,
                    dof_in_ff: int,
                    dof_out_ff: int,
                    C_ff: Optional[np.ndarray] = None,
                    quantity: str = "receptance",
                    title: str = "Direct FRF"):

    omega = 2*np.pi*frequencies_hz
    n = K_ff.shape[0]
    e_in = np.zeros(n); e_in[dof_in_ff] = 1.0

    H = np.zeros_like(omega, dtype=complex)
    # Prefer scipy if available; fallback to numpy
    try:
        from scipy.linalg import solve
    except ImportError:
        solve = None

    for i, w in enumerate(omega):
        A = K_ff - (w**2)*M_ff
        if C_ff is not None:
            A = A + 1j*w*C_ff
        if solve:
            x = solve(A, e_in, assume_a='gen')
        else:
            x = np.linalg.solve(A, e_in)
        H[i] = x[dof_out_ff]

    if quantity == "mobility":
        H = 1j*omega*H
    elif quantity == "accelerance":
        H = -(omega**2)*H

    mag = np.abs(H)
    phase = np.angle(H, deg=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    ax1.plot(frequencies_hz, mag, linewidth=2)
    ax1.set_ylabel('|H|')
    ax1.grid(True, alpha=0.3)
    ax2.plot(frequencies_hz, phase, linewidth=1.5)
    ax2.set_ylabel('Phase (deg)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.grid(True, alpha=0.3)
    ax1.set_title(title)
    plt.tight_layout()


def excel_results(Matrix, M_added, C_added, M_global, C_global, result, frequencies, wet_freqs, n_nodes, nspan, nchord, blade_name, damp_ratios, eigvals):
    start_time = time.time()

    Ma = result["Ma"]
    f = result["f"]
    k = result["k"]
    V_0 = result["V_0"]
    alpha = result["alpha"]  # back-calculate angle of attack
    M_added = result["M_added"]
    C_added = result["C_added"]
    term1_mass = result["term1_mass"]
    term2_mass = result["term2_mass"]
    term1_damp = result["term1_damp"]
    term2_damp = result["term2_damp"]
    Z = result["Z"]
    Qjj = result["Qjj"]
    panel_area = result["panel_area"]
    panel_normals = result["panel_normals"]
    aerogrid = result["aerogrid"]

    M_w_rigidbody = Matrix.T @ M_added @ Matrix
    C_w_rigidbody = Matrix.T @ C_added @ Matrix

    Ms_rigidbody = Matrix.T @ M_global @ Matrix
    Cs_rigidbody = Matrix.T @ C_global @ Matrix
    #print(f"Rigid body added_mass: {M_w_rigidbody} ")
    #print(f"Rigid body added_damping: {C_w_rigidbody} ")

    term1_mass_rigidbody = Matrix.T @ term1_mass @ Matrix
    term2_mass_rigidbody = Matrix.T @ term2_mass @ Matrix
    term1_damp_rigidbody = Matrix.T @ term1_damp @ Matrix
    term2_damp_rigidbody = Matrix.T @ term2_damp @ Matrix


    ## ==== STORE RESULTS INTO CSV FILE ==== ##
    # Save only first N frequencies if desired
    num_modes_to_store = min(5, len(wet_freqs))
    dry_freqs_to_save = frequencies[:num_modes_to_store]
    wet_freqs_to_save = wet_freqs[:num_modes_to_store]
    damp_ratios_to_save = damp_ratios[:num_modes_to_store]
    real_eigvals_to_save = np.real(eigvals[:num_modes_to_store])

    output_data = []

    # Create dictionary row
    row = {
        "nodes": n_nodes,
        "nspan": nspan,
        "nchord": nchord,
        "alpha": alpha,
        "V_0": V_0,
        "Ma": Ma,
        "f": f,
        "k": k,
        #"Z": Z,

        "Ms_xx": Ms_rigidbody[0,0],
        "Ms_yy": Ms_rigidbody[1,1],
        "Ms_zz": Ms_rigidbody[2,2],
        #"Ms_phix": Ms_rigidbody[3,3],
        #"Ms_phiy": Ms_rigidbody[4,4],
        #"Ms_phiz": Ms_rigidbody[5,5],
        #
        #"Cs_xx": Ms_rigidbody[0,0],
        #"Cs_yy": Ms_rigidbody[1,1],
        #"Cs_zz": Ms_rigidbody[2,2],
        #"Cs_phix": Ms_rigidbody[3,3],
        #"Cs_phiy": Ms_rigidbody[4,4],
        #"Cs_phiz": Ms_rigidbody[5,5],

        "Mw_xx": M_w_rigidbody[0, 0],
        "Mw_yy": M_w_rigidbody[1, 1],
        "Mw_zz": M_w_rigidbody[2, 2],
        #"Mw_phix": M_w_rigidbody[3,  3],
        #"Mw_phiy": M_w_rigidbody[4, 4],
        #"Mw_phiz": M_w_rigidbody[5, 5],

        "Cw_xx": C_w_rigidbody[0, 0],
        "Cw_yy": C_w_rigidbody[1, 1],
        "Cw_zz": C_w_rigidbody[2, 2],
        #"Cw_phix": C_w_rigidbody[3, 3],
        #"Cw_phiy": C_w_rigidbody[4, 4],
        #"Cw_phiz": C_w_rigidbody[5, 5],

        "term1_mass_xx": term1_mass_rigidbody[0, 0],
        "term1_mass_yy": term1_mass_rigidbody[1, 1],
        "term1_mass_zz": term1_mass_rigidbody[2, 2],

        "term2_mass_xx": term2_mass_rigidbody[0, 0],
        "term2_mass_yy": term2_mass_rigidbody[1, 1],
        "term2_mass_zz": term2_mass_rigidbody[2, 2],

        "term1_damp_xx": term1_damp_rigidbody[0, 0],
        "term1_damp_yy": term1_damp_rigidbody[1, 1],
        "term1_damp_zz": term1_damp_rigidbody[2, 2],

        "term2_damp_xx": term2_damp_rigidbody[0, 0],
        "term2_damp_yy": term2_damp_rigidbody[1, 1],
        "term2_damp_zz": term2_damp_rigidbody[2, 2],
    }

    # Add frequency entries
    for i in range(num_modes_to_store):
        row[f"dry_freq{i+1}"] = dry_freqs_to_save[i]
        row[f"wet_freq{i+1}"] = wet_freqs_to_save[i]
        row[f"damp_ratios{i+1}"] = damp_ratios_to_save[i]
        row[f"real_eigvals{i+1}"] = real_eigvals_to_save[i]

    output_data.append(row)

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(output_data)
    csv_output_path = "/home/lorebasket/FSI/output_summary" + blade_name + ".csv"
    df.to_csv(csv_output_path, index=False)
    print(f"\n✅ Summary saved to {csv_output_path}")

    print(f"Fluid results printed in {time.time() - start_time:.3f} seconds")
