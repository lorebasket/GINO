import time
import csv


def main(start_time=None):

    import sys
    import numpy as np
    from pathlib import Path
    from precompute_qjj import precompute_qjj_grid, check_existing_qjj_files
    from precompute_qjj_vlm import precompute_qjj_vlm
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import math

    FSI_path = '/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI'

    sys.path.extend([
        FSI_path,
        FSI_path + '/PanelAero',
        FSI_path + '/PanelAero/panelaero_utl',
        FSI_path + '/PanelAero/panelaero_utl/old',
        FSI_path + '/SONATA/wing01',
        FSI_path + '/SONATA/arm01'
    ])

    from rotate_aerogrid import _rotation
    import build_aeromodel
    from CAERO1_generator import format_caero1_card
    import rotate_aerogrid
    from rotate_aerogrid_z import rotate_aerogrid_z
    #import LE_curve_points, TE_curve_points
    import build_aeromodel_crvs
    from scipy.interpolate import interp1d
    from Hydroelastic_analysis_workflow.post_processing import plot_aero_beam_model

    # --- AEROGRID SETUP --- #
    # Multiple blade names: an aerogrid is built for every blade in this list.
    blade_names = ['tnz_boot', 'tnz_foil_dx', 'tnz_foil_sx']
    fluid = 'water'
    nspan = [16]  # number of spanwise panels
    AR = [0.5]    # aspect ratio
    DLM = True
    VLM = False
    dihedral_angle = 20.0  # degrees, applied to tnz_foil_dx only (which is the only one with a non-zero dihedral in the original geometry)
    # Chordwise offset applied to tnz_foil_dx and tnz_foil_sx aerogrids.
    # Positive = forward (+X, toward leading edge); negative = aft (−X).
    # Must match the foil_chordwise_offset used in structural_model.py.
    foil_chordwise_offset = 0.1    # [m] if 
    # Choose DLM integration method: 'parabolic' (Rodden 1971/72) or 'quartic' (Rodden 1998)
    # Set to 'quartic' to use the quartic integration approximation.
    DLM_method = 'quartic'

    # We'll compute nchord for each combination of nspan and AR inside the loop
    # nchord = nspan / AR  (computed per iteration)

    plot = True

    # angle of attack
    attack_angle = [0.0]  # [0, 10] # °grad
    alpha_r = np.deg2rad(attack_angle[0])  # angle of attack in radians

    c_sound = {'air': 332.5, 'water': 1484.0}  # speed of sound of air at 2000m altitude

    # --- k / V LIST  --- #
    k_list = np.round(np.concatenate([np.linspace(0.001, 1, 10), np.linspace(1, 4, 20), np.linspace(4, 20, 20)]), 3)
    V_low = np.linspace(5, 10, 5); V_flutter = np.linspace(10, 30, 20); V_high = np.linspace(30, 45, 15)
    V_list = np.concatenate([V_low, V_flutter, V_high])

    Ma_list = V_list / c_sound[fluid]

    # aerogrid_dict key: (blade_name, n_span, n_chord_count)
    aerogrid_dict = {}

    # Accumulate diagnostic rows for CSV export
    csv_rows = []

    # ------------------------------------------------------------------ #
    # PHASE 1 — BUILD ALL AEROGRIDS
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("PHASE 1: Building all aerogrids")
    print(f"  Blades  : {blade_names}")
    print(f"  nspan   : {nspan}")
    print(f"  AR      : {AR}")
    print(f"{'='*60}\n")

    for blade_name in blade_names:
        for n_span in nspan:
            for aspect_ratio in AR:
                # Compute chordwise panels from aspect ratio
                n_chord_count = int(n_span * aspect_ratio)

                print(f"\nBuilding aerogrid for blade='{blade_name}'  nspan={n_span}  nchord={n_chord_count}")

                # ------------------------------------------------------------------ #
                # Method 2: curve-based aerogrid (LE/TE curve points from SONATA)
                # ------------------------------------------------------------------ #
                import importlib
                import importlib.util

                def _import_submodule(blade, submodule_name):
                    """Import LE_curve_points or TE_curve_points from the blade's dir."""
                    gr_dir = Path(FSI_path) / 'SONATA' / '8_tnz' / blade
                    mod_file = gr_dir / f'{submodule_name}.py'

                    spec = importlib.util.spec_from_file_location(
                        f"SONATA.8_tnz.{blade}.{submodule_name}", mod_file
                    )
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod

                def _find_points(module, candidates):
                    for cand in candidates:
                        for attr in dir(module):
                            if attr == cand or attr.startswith(cand):
                                val = getattr(module, attr)
                                try:
                                    arr = np.asarray(val)
                                    if arr.ndim == 2 and arr.shape[1] == 3:
                                        return arr
                                except Exception:
                                    continue
                    raise AttributeError(
                        f"No matching attribute found among {candidates} in module {module.__name__}. "
                        f"Available attributes: {[a for a in dir(module) if not a.startswith('_')]}"
                    )

                le_mod = _import_submodule(blade_name, 'LE_curve_points')
                te_mod = _import_submodule(blade_name, 'TE_curve_points')

                le_candidates = [f'le_points_{blade_name}']
                te_candidates = [f'te_points_{blade_name}']

                le_pts = _find_points(le_mod, le_candidates)
                te_pts = _find_points(te_mod, te_candidates)

                # Convert from mm to meters (always, for all blades)
                le_pts = np.asarray(le_pts, dtype=float) / 1000.0
                te_pts = np.asarray(te_pts, dtype=float) / 1000.0

                def _apply_dihedral(pts, angle_deg, pivot):
                    """
                    Rotate the blade spanwise direction by `angle_deg` around the X-axis,
                    keeping the given `pivot` point (3-element array [x, y, z]) fixed in space.

                    All points are translated so that `pivot` is at the origin, rotated
                    in the Y-Z plane, then translated back:
                      dy = y - pivot_y
                      dz = z - pivot_z
                      y_new = pivot_y + dy * cos(θ) - dz * sin(θ)
                      z_new = pivot_z + dy * sin(θ) + dz * cos(θ)
                    """
                    pts = pts.copy()
                    theta   = np.deg2rad(angle_deg)
                    pivot_y = float(pivot[1])
                    pivot_z = float(pivot[2])
                    for row in pts:
                        dy     = row[1] - pivot_y
                        dz     = row[2] - pivot_z
                        row[1] = pivot_y + dy * np.cos(theta) - dz * np.sin(theta)
                        row[2] = pivot_z + dy * np.sin(theta) + dz * np.cos(theta)
                    return pts

                if dihedral_angle != 0.0 and blade_name in ('tnz_foil_dx'):
                    # Root of foil_dx is the first LE point (index 0)
                    pivot_dx = le_pts[0]
                    le_pts = _apply_dihedral(le_pts, -dihedral_angle, pivot_dx)
                    te_pts = _apply_dihedral(te_pts, -dihedral_angle, pivot_dx)

                if dihedral_angle != 0.0 and blade_name in ('tnz_foil_sx'):
                    # Root of foil_sx is the second LE point (index 1)
                    pivot_sx = le_pts[1]
                    le_pts = _apply_dihedral(le_pts, +dihedral_angle, pivot_sx)
                    te_pts = _apply_dihedral(te_pts, +dihedral_angle, pivot_sx)

                # Apply chordwise offset to foil aerogrids (must match structural_model.py).
                # Both LE and TE points are shifted uniformly in X; Y and Z are unchanged.
                if abs(foil_chordwise_offset) > 1e-9 and blade_name in ('tnz_foil_dx', 'tnz_foil_sx'):
                    le_pts = le_pts.copy()
                    te_pts = te_pts.copy()
                    le_pts[:, 0] += foil_chordwise_offset
                    te_pts[:, 0] += foil_chordwise_offset
                    print(f"  Chordwise offset {foil_chordwise_offset:+.4f} m applied to {blade_name} aerogrid (X shift)")

                # Create interpolation functions
                def make_curve_callable(points):
                    n = len(points)
                    kind = 'cubic' if n >= 4 else ('quadratic' if n >= 3 else 'linear')
                    u = np.linspace(0, 1, n)
                    fx = interp1d(u, points[:, 0], kind=kind, fill_value="extrapolate")
                    fy = interp1d(u, points[:, 1], kind=kind, fill_value="extrapolate")
                    fz = interp1d(u, points[:, 2], kind=kind, fill_value="extrapolate")
                    return lambda t: np.array([fx(t), fy(t), fz(t)])

                le_curve = make_curve_callable(le_pts)
                te_curve = make_curve_callable(te_pts)

                # Build aerogrid
                aero = build_aeromodel_crvs.AeroGridFromCurves(le_curve, te_curve, n_span=n_span, n_chord=n_chord_count)
                aerogrid = aero.build_aerogrid(eid_start=1000)

                print(f"  Aerodynamic grid created: {aerogrid['n']} panels")

                # Store in dictionary
                aerogrid_dict[(blade_name, n_span, n_chord_count)] = aerogrid

                # --- 3D plot ---
                if plot:
                    us = np.linspace(0.0, 1.0, 300)
                    le_sample = np.array([le_curve(u) for u in us])
                    te_sample = np.array([te_curve(u) for u in us])

                    grid_ids = aerogrid['cornerpoint_grids'][:, 0]
                    grid_pts = aerogrid['cornerpoint_grids'][:, 1:4]
                    panels   = aerogrid['cornerpoint_panels']

                    fig = plt.figure(figsize=(9, 6))
                    ax = fig.add_subplot(111, projection='3d')

                    ax.scatter(grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2], s=8, c='k', alpha=0.6)

                    for panel in panels:
                        coords = []
                        for pid in panel:
                            idx = np.where(grid_ids == pid)[0]
                            if idx.size == 0:
                                continue
                            coords.append(grid_pts[idx[0]])
                        if len(coords) >= 3:
                            coords = np.vstack(coords + [coords[0]])
                            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='gray', linewidth=0.8, alpha=0.8)

                    ax.plot(le_sample[:, 0], le_sample[:, 1], le_sample[:, 2], color='red',  linewidth=2.0, label='LE curve')
                    ax.plot(te_sample[:, 0], te_sample[:, 1], te_sample[:, 2], color='blue', linewidth=2.0, label='TE curve')

                    panel_centers = aerogrid['offset_k']
                    panel_normals = aerogrid['N']
                    ax.quiver(panel_centers[:, 0], panel_centers[:, 1], panel_centers[:, 2],
                              panel_normals[:, 0], panel_normals[:, 1], panel_normals[:, 2],
                              length=0.1 * np.mean(aerogrid['l']), normalize=True, color='green', label='Normals')

                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'{blade_name}  (n_span={n_span}, n_chord={n_chord_count})')
                    ax.legend()
                    plt.tight_layout()

                    try:
                        all_x = np.concatenate([grid_pts[:, 0], le_sample[:, 0], te_sample[:, 0]])
                        all_y = np.concatenate([grid_pts[:, 1], le_sample[:, 1], te_sample[:, 1]])
                        all_z = np.concatenate([grid_pts[:, 2], le_sample[:, 2], te_sample[:, 2]])

                        x_min, x_max = float(all_x.min()), float(all_x.max())
                        y_min, y_max = float(all_y.min()), float(all_y.max())
                        z_min, z_max = float(all_z.min()), float(all_z.max())

                        x_width = x_max - x_min
                        z_center = 0.5 * (z_max + z_min)
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                        ax.set_zlim(z_center - 0.5 * x_width, z_center + 0.5 * x_width)
                    except Exception as _e:
                        print(f"Could not set equal z/x width for plot: {_e}")

                    try:
                        plot_out_dir = Path(
                            f"/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/Qjj/qjj_precomputed"
                            f"/{blade_name}_{fluid}_alpha{attack_angle[0]}_nspan{n_span}_nchord{n_chord_count}_{DLM_method}"
                        )
                        plot_out_dir.mkdir(parents=True, exist_ok=True)
                        plot_file = plot_out_dir / 'aerogrid.png'
                        fig.savefig(plot_file, dpi=200)
                        print(f"  Saved aerogrid plot to {plot_file}")
                    except Exception as _e:
                        print(f"  Warning: could not save aerogrid plot: {_e}")

                    plt.show()

    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE — {len(aerogrid_dict)} aerogrid(s) built:")
    for key in aerogrid_dict:
        b, ns, nc = key
        print(f"  blade={b:20s}  nspan={ns}  nchord={nc}  panels={aerogrid_dict[key]['n']}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # PHASE 2 — MERGE ALL AEROGRIDS INTO ONE COMBINED AEROGRID
    # ------------------------------------------------------------------ #
    # The DLM operates on a single aerogrid dict.  We concatenate every
    # per-panel and per-grid array from all individual grids, re-numbering
    # IDs so they are globally unique, and offsetting the DOF index sets
    # (set_l / set_k / set_j) so they cover non-overlapping DOF ranges.
    # The resulting combined aerogrid is structurally identical to a single-
    # blade aerogrid and can be passed directly to precompute_qjj_grid.

    print(f"\n{'='*60}")
    print("PHASE 2: Merging all aerogrids into one combined aerogrid")
    print(f"{'='*60}\n")

    def merge_aerogrids(grid_dict):
        """
        Concatenate multiple aerogrid dicts into a single combined aerogrid.
        All panel IDs and corner-point IDs are made globally unique.
        DOF index sets (set_l, set_k, set_j) are offset so they don't overlap.
        """
        grids = list(grid_dict.values())

        # ---- scalar / simple-concat arrays ----
        # Running offsets to ensure uniqueness
        panel_id_offset    = 0
        grid_id_offset     = 0
        dof_offset         = 0   # for set_l / set_k / set_j

        merged = {
            'ID':               [],
            'l':                [],
            'A':                [],
            'N':                [],
            'offset_l':         [],
            'offset_k':         [],
            'offset_j':         [],
            'offset_P1':        [],
            'offset_P3':        [],
            'r':                [],
            'set_l':            [],
            'set_k':            [],
            'set_j':            [],
            'CD':               [],
            'CP':               [],
            'cornerpoint_panels': [],
            'cornerpoint_grids':  [],
        }

        for g in grids:
            n_panels = int(g['n'])
            n_grids  = g['cornerpoint_grids'].shape[0]

            # Panel IDs — offset to avoid collisions
            merged['ID'].append(g['ID'] + panel_id_offset)

            # Per-panel scalar/vector arrays — just concatenate
            merged['l'].append(g['l'])
            merged['A'].append(g['A'])
            merged['N'].append(g['N'])
            merged['offset_l'].append(g['offset_l'])
            merged['offset_k'].append(g['offset_k'])
            merged['offset_j'].append(g['offset_j'])
            merged['offset_P1'].append(g['offset_P1'])
            merged['offset_P3'].append(g['offset_P3'])
            merged['r'].append(g['r'])
            merged['CD'].append(g['CD'])
            merged['CP'].append(g['CP'])

            # DOF index sets — offset by dof_offset so they span unique ranges
            merged['set_l'].append(g['set_l'] + dof_offset)
            merged['set_k'].append(g['set_k'] + dof_offset)
            merged['set_j'].append(g['set_j'] + dof_offset)

            # cornerpoint_panels: shift the corner-point IDs by grid_id_offset
            shifted_cp_panels = g['cornerpoint_panels'] + grid_id_offset
            merged['cornerpoint_panels'].append(shifted_cp_panels)

            # cornerpoint_grids: first column is grid ID, shift it
            cpg = g['cornerpoint_grids'].copy()
            cpg[:, 0] = cpg[:, 0] + grid_id_offset
            merged['cornerpoint_grids'].append(cpg)

            # Advance offsets for next sub-grid
            panel_id_offset += n_panels
            grid_id_offset  += n_grids
            dof_offset      += n_panels * 6   # set_* has shape (n_panels, 6)

        # Stack everything
        combined = {
            'ID':               np.concatenate(merged['ID']),
            'l':                np.concatenate(merged['l']),
            'A':                np.concatenate(merged['A']),
            'N':                np.vstack(merged['N']),
            'offset_l':         np.vstack(merged['offset_l']),
            'offset_k':         np.vstack(merged['offset_k']),
            'offset_j':         np.vstack(merged['offset_j']),
            'offset_P1':        np.vstack(merged['offset_P1']),
            'offset_P3':        np.vstack(merged['offset_P3']),
            'r':                np.vstack(merged['r']),
            'set_l':            np.vstack(merged['set_l']),
            'set_k':            np.vstack(merged['set_k']),
            'set_j':            np.vstack(merged['set_j']),
            'CD':               np.concatenate(merged['CD']),
            'CP':               np.concatenate(merged['CP']),
            'cornerpoint_panels': np.vstack(merged['cornerpoint_panels']),
            'cornerpoint_grids':  np.vstack(merged['cornerpoint_grids']),
            'n':                int(sum(int(g['n']) for g in grids)),
            'coord_desc':       'bodyfixed',
        }
        return combined

    aerogrid_combined = merge_aerogrids(aerogrid_dict)

    print(f"Combined aerogrid:")
    print(f"  Sub-grids merged : {len(aerogrid_dict)}")
    print(f"  Total panels     : {aerogrid_combined['n']}")
    for key, g in aerogrid_dict.items():
        b, ns, nc = key
        print(f"    {b:20s}  nspan={ns}  nchord={nc}  panels={g['n']}")
    print()

    # ------------------------------------------------------------------ #
    # PHASE 2b — PLOT THE MERGED AEROGRID
    # ------------------------------------------------------------------ #
    if plot:
        print("Plotting merged aerogrid …")

        # Colour palette — one colour per sub-grid
        _palette = [
            '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
        ]

        fig_m = plt.figure(figsize=(11, 7))
        ax_m  = fig_m.add_subplot(111, projection='3d')

        all_pts_x, all_pts_y, all_pts_z = [], [], []

        for sub_idx, (sub_key, sub_g) in enumerate(aerogrid_dict.items()):
            b_name, ns, nc = sub_key
            colour = _palette[sub_idx % len(_palette)]

            sub_grid_ids = sub_g['cornerpoint_grids'][:, 0]
            sub_grid_pts = sub_g['cornerpoint_grids'][:, 1:4]
            sub_panels   = sub_g['cornerpoint_panels']

            # Scatter corner points
            ax_m.scatter(sub_grid_pts[:, 0], sub_grid_pts[:, 1], sub_grid_pts[:, 2],
                         s=6, c=colour, alpha=0.7, zorder=3)

            # Draw panel edges
            for panel in sub_panels:
                coords = []
                for pid in panel:
                    idx = np.where(sub_grid_ids == pid)[0]
                    if idx.size == 0:
                        continue
                    coords.append(sub_grid_pts[idx[0]])
                if len(coords) >= 3:
                    coords_loop = np.vstack(coords + [coords[0]])
                    ax_m.plot(coords_loop[:, 0], coords_loop[:, 1], coords_loop[:, 2],
                              color=colour, linewidth=0.8, alpha=0.85)

            # Panel normals
            panel_centers = sub_g['offset_k']
            panel_normals = sub_g['N']
            mean_chord    = float(np.abs(sub_g['l']).mean())
            ax_m.quiver(panel_centers[:, 0], panel_centers[:, 1], panel_centers[:, 2],
                        panel_normals[:, 0], panel_normals[:, 1], panel_normals[:, 2],
                        length=0.1 * mean_chord, normalize=True,
                        color=colour, alpha=0.6)

            all_pts_x.append(sub_grid_pts[:, 0])
            all_pts_y.append(sub_grid_pts[:, 1])
            all_pts_z.append(sub_grid_pts[:, 2])

            # Legend proxy
            ax_m.plot([], [], [], color=colour, linewidth=2,
                      label=f'{b_name}  (ns={ns}, nc={nc}, panels={sub_g["n"]})')

        ax_m.set_xlabel('X')
        ax_m.set_ylabel('Y')
        ax_m.set_zlabel('Z')
        ax_m.set_title(f'Merged aerogrid — {len(aerogrid_dict)} sub-grid(s)  |  total panels: {aerogrid_combined["n"]}')
        ax_m.legend(fontsize=8)
        plt.tight_layout()

        # Equal-ish axis scaling
        try:
            all_x = np.concatenate(all_pts_x)
            all_y = np.concatenate(all_pts_y)
            all_z = np.concatenate(all_pts_z)
            x_rng = all_x.max() - all_x.min()
            y_rng = all_y.max() - all_y.min()
            z_rng = all_z.max() - all_z.min()
            max_rng = max(x_rng, y_rng, z_rng)
            mx = 0.5 * (all_x.max() + all_x.min())
            my = 0.5 * (all_y.max() + all_y.min())
            mz = 0.5 * (all_z.max() + all_z.min())
            ax_m.set_xlim(mx - max_rng / 2, mx + max_rng / 2)
            ax_m.set_ylim(my - max_rng / 2, my + max_rng / 2)
            ax_m.set_zlim(mz - max_rng / 2, mz + max_rng / 2)
        except Exception as _e:
            print(f"  Warning: could not set equal axis limits for merged plot: {_e}")

        # Save
        try:
            merged_plot_dir = Path(
                "/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/Qjj/qjj_precomputed"
            )
            merged_plot_dir.mkdir(parents=True, exist_ok=True)
            merged_plot_file = merged_plot_dir / f"aerogrid_merged_{'_'.join(blade_names)}.png"
            fig_m.savefig(merged_plot_file, dpi=200)
            print(f"  Saved merged aerogrid plot to {merged_plot_file}")
        except Exception as _e:
            print(f"  Warning: could not save merged aerogrid plot: {_e}")

        plt.show()

    # ------------------------------------------------------------------ #
    # PHASE 3 — COMPUTE Qjj ON THE COMBINED AEROGRID (single call)
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("PHASE 3: Computing Qjj for the combined multi-aerogrid")
    print(f"  Total panels     : {aerogrid_combined['n']}")
    print(f"  k values         : {len(k_list)}")
    print(f"  Ma values        : {len(Ma_list)}")
    print(f"  Total Qjj comput.: {len(k_list) * len(Ma_list)}")
    print(f"{'='*60}\n")

    from pathlib import Path as _Path

    # Build a descriptive directory name that lists all blades
    blades_tag = '_'.join(blade_names)
    # Use the nspan/nchord from the first (and typically only) grid configuration
    first_key = next(iter(aerogrid_dict))
    _, n_span_first, n_chord_first = first_key

    out_dir = (
        f"/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/Qjj/qjj_precomputed"
        f"/MULTI_{blades_tag}_{fluid}_alpha{attack_angle[0]}_nspan{n_span_first}_nchord{n_chord_first}_{DLM_method}_leoffset{-0.2+foil_chordwise_offset:.1f}"
    )
    out_dir_vlm = (
        f"/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/Qjj/qjj_precomputed"
        f"/vjj_MULTI_{blades_tag}_{fluid}_alpha{attack_angle[0]}_nspan{n_span_first}_nchord{n_chord_first}_{DLM_method}_leoffset{-0.2+foil_chordwise_offset:.1f}"
    )

    _Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Save the combined aerogrid for downstream use
    try:
        aerogrid_file = _Path(out_dir) / 'aerogrid_combined.npz'
        np.savez_compressed(aerogrid_file, **aerogrid_combined)
        print(f"  Saved combined aerogrid to {aerogrid_file}")
    except Exception as _e:
        print(f"  Warning: could not save combined aerogrid: {_e}")

    t_dlm_start = time.time()
    if DLM:
        precompute_qjj_grid(FSI_path, aerogrid_combined, k_list, Ma_list, out_dir,
                            dtype=np.float32, resume=True, verify_existing=True,
                            dlm_method=DLM_method)
    t_dlm_elapsed = time.time() - t_dlm_start

    t_vlm_start = time.time()
    if VLM:
        precompute_qjj_vlm(FSI_path, aerogrid_combined, Ma_list, out_dir_vlm,
                           dtype=np.float32, verbose=True, resume=True, verify_existing=False)
    t_vlm_elapsed = time.time() - t_vlm_start

    # ------------------------------------------------------------------ #
    # POST-COMPUTATION DIAGNOSTICS (on combined aerogrid)
    # ------------------------------------------------------------------ #
    aerogrid = aerogrid_combined   # alias for the diagnostics block below

    chord_lengths = np.abs(aerogrid['l'])
    span_lengths  = np.linalg.norm(aerogrid['r'], axis=1)
    panel_AR      = span_lengths / chord_lengths

    print(f"\n{'='*60}")
    print(f"PANEL ASPECT RATIOS  (combined multi-aerogrid)")
    print(f"  Number of panels   : {aerogrid['n']}")
    print(f"  Chord length  - min: {chord_lengths.min():.4f} m   max: {chord_lengths.max():.4f} m   mean: {chord_lengths.mean():.4f} m")
    print(f"  Span  length  - min: {span_lengths.min():.4f} m   max: {span_lengths.max():.4f} m   mean: {span_lengths.mean():.4f} m")
    print(f"  Panel AR      - min: {panel_AR.min():.4f}         max: {panel_AR.max():.4f}         mean: {panel_AR.mean():.4f}")
    print(f"{'='*60}\n")

    V_probe = 25.0  # m/s
    k_probe_indices = list(range(len(k_list)))

    c_sound_local = c_sound[fluid]
    Ma_probe = V_probe / c_sound_local
    i_Ma_probe = int(np.argmin(np.abs(Ma_list - Ma_probe)))
    Ma_actual  = Ma_list[i_Ma_probe]

    print(f"\n{'='*60}")
    print(f"Qjj NORMS  (combined multi-aerogrid)")
    print(f"  Probe speed V = {V_probe} m/s  ->  Ma = {Ma_probe:.6f}  (nearest stored Ma[{i_Ma_probe}] = {Ma_actual:.6f})")
    print(f"  {'k_index':>8}  {'k':>10}  {'||Qjj|| (Frobenius)':>22}  {'||Re(Qjj)||':>14}  {'||Im(Qjj)||':>14}")
    print(f"  {'-'*75}")

    norms_fro = []; norms_re = []; norms_im = []
    n_ok = 0; n_missing = 0; n_error = 0

    for i_k_probe in k_probe_indices:
        k_probe_val = k_list[i_k_probe]
        fname_base  = _Path(out_dir) / f"k_{i_k_probe:03d}__Ma_{i_Ma_probe:04d}"
        file_real   = str(fname_base) + "_real.npy"
        file_imag   = str(fname_base) + "_imag.npy"
        try:
            Qr = np.load(file_real).astype(np.complex128)
            Qi = np.load(file_imag).astype(np.complex128)
            Qjj_probe = Qr + 1j * Qi
            norm_full = float(np.linalg.norm(Qjj_probe, 'fro'))
            norm_re   = float(np.linalg.norm(Qr, 'fro'))
            norm_im   = float(np.linalg.norm(Qi, 'fro'))
            print(f"  {i_k_probe:>8d}  {k_probe_val:>10.4f}  {norm_full:>22.6f}  {norm_re:>14.6f}  {norm_im:>14.6f}")
            norms_fro.append(norm_full)
            norms_re.append(norm_re)
            norms_im.append(norm_im)
            n_ok += 1
        except FileNotFoundError:
            print(f"  {i_k_probe:>8d}  {k_probe_val:>10.4f}  {'FILE NOT FOUND':>22}")
            n_missing += 1
        except Exception as _qe:
            print(f"  {i_k_probe:>8d}  {k_probe_val:>10.4f}  {'ERROR: ' + str(_qe):>22}")
            n_error += 1

    def _agg(vals, fn): return round(fn(vals), 8) if vals else ''
    row = {
        'blade_name':             f"MULTI: {blades_tag}",
        'fluid':                  fluid,
        'DLM_method':             DLM_method,
        'n_total_cases':          len(k_list) * len(Ma_list),
        'n_panels':               int(aerogrid['n']),
        'nspan':                  n_span_first,
        'nchord':                 n_chord_first,
        'panel_AR_min':           float(panel_AR.min()),
        'panel_AR_max':           float(panel_AR.max()),
        'panel_AR_mean':          float(panel_AR.mean()),
        'V_probe_m_s':            V_probe,
        'Ma_probe':               round(float(Ma_probe), 8),
        'Ma_nearest':             round(float(Ma_actual), 8),
        'n_k_ok':                 n_ok,
        'n_k_missing':            n_missing,
        'n_k_error':              n_error,
        'norm_Qjj_fro_min':       _agg(norms_fro, min),
        'norm_Qjj_fro_max':       _agg(norms_fro, max),
        'norm_Qjj_fro_mean':      _agg(norms_fro, lambda v: sum(v) / len(v)),
        'norm_Re_Qjj_fro_mean':   _agg(norms_re, lambda v: sum(v) / len(v)),
        'norm_Im_Qjj_fro_mean':   _agg(norms_im, lambda v: sum(v) / len(v)),
        'dlm_compute_time_s':     round(t_dlm_elapsed, 3),
        'vlm_compute_time_s':     round(t_vlm_elapsed, 3),
        'aerogrid_compute_time_s': round(t_dlm_elapsed + t_vlm_elapsed, 3),
    }
    csv_rows.append(row)

    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # WRITE CSV SUMMARY
    # ------------------------------------------------------------------ #
    total_time = (time.time() - start_time) if start_time is not None else float('nan')

    csv_fieldnames = [
        'blade_name', 'fluid', 'DLM_method',
        'n_total_cases', 'n_panels', 'nspan', 'nchord',
        'panel_AR_min', 'panel_AR_max', 'panel_AR_mean',
        'V_probe_m_s', 'Ma_probe', 'Ma_nearest',
        'n_k_ok', 'n_k_missing', 'n_k_error',
        'norm_Qjj_fro_min', 'norm_Qjj_fro_max', 'norm_Qjj_fro_mean',
        'norm_Re_Qjj_fro_mean', 'norm_Im_Qjj_fro_mean',
        'dlm_compute_time_s', 'vlm_compute_time_s', 'aerogrid_compute_time_s',
        'total_execution_time_s',
    ]

    for row in csv_rows:
        row['total_execution_time_s'] = round(total_time, 2)

    # Place the diagnostics CSV one level above the individual output dirs
    csv_out_path = _Path(
        "/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/Qjj/qjj_precomputed"
    ) / 'executer_multi_aerogrid_diagnostics.csv'
    with open(csv_out_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDiagnostics CSV saved to: {csv_out_path}")


def play_completion_sound():
    try:
        import os
        os.system('echo -n "\a"')  # ASCII bell character
    except Exception as e:
        print(f"Could not play sound: {e}")


if __name__ == "__main__":
    start_time = time.time()
    try:
        main(start_time=start_time)
    finally:
        play_completion_sound()
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
