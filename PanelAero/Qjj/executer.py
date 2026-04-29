import time
import csv


def main(start_time=None):

    import sys
    import os
    import numpy as np
    from pathlib import Path
    from precompute_qjj import precompute_qjj_grid, check_existing_qjj_files
    from precompute_qjj_vlm import precompute_qjj_vlm
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import math

    FSI_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sys.path.extend([
        FSI_path,
        FSI_path + '/PanelAero',
        FSI_path + '/PanelAero/panelaero_utl',
        FSI_path + '/SONATA/wing01',
        FSI_path + '/SONATA/arm01'
    ])

    #from rotate_aerogrid import _rotation
    import rotate_aerogrid
    import build_aeromodel
    from CAERO1_generator import format_caero1_card

    import build_aeromodel_crvs
    from scipy.interpolate import interp1d
    from Hydroelastic_analysis_workflow.post_processing import plot_aero_beam_model

    # --- AEROGRID SETUP --- #
    blade_name = 'wing01' # 'hollowell', 'wing01', 'GOLAND'
    fluid = 'water'
    nspan = [4] # number of spanwise panels
    AR = [0.5]  # aspect ratio
    DLM = True
    VLM = False
    # Choose DLM integration method: 'parabolic' (Rodden 1971/72) or 'quartic' (Rodden 1998)
    # Set to 'quartic' to use the quartic integration approximation.
    DLM_method = 'quartic'

    plot = True

    # angle of attack
    attack_angle = [0.0] #[0, 10] # °grad
    alpha_r = np.deg2rad(attack_angle[0]) # angle of attack in radians
    
    c_sound = {'air': 332.5, 'water':1484.0} # speed of sound of air at 2000m altitude

    # --- LIST  --- #
    k_list = np.round(np.concatenate([np.linspace(0.001, 1, 10), np.linspace(1, 4, 10), np.linspace(4, 30, 80)]), 3)
    V_list = np.linspace(5, 55, 50)
    Ma_list = V_list / c_sound[fluid]
        
    # ===== Method ===== #
    # 1 for CAERO1 flat panel, 2 for curve-based grid builder
    # wing01 is a curve-based grid, needs to be treated differently
    if blade_name == 'wing01':
        method = 2
    else:
        method = 1

    # Accumulate diagnostic rows for CSV export
    csv_rows = []
    aerogrid_dict = {}

    for n_span in nspan:
        for aspect_ratio in AR:
            # Compute chordwise panels from aspect ratio
            n_chord_count = int(n_span * aspect_ratio)

            print("Calculating aerogrid Qjj with:")
            print(f"     spanwise panels: {n_span}")
            print(f"     aspect ratio: {aspect_ratio}")
            print(f"     chordwise panels: {n_chord_count}")

            if method == 1:
                
                if blade_name == 'GOLAND':
                    # GOLAND wing configuration
                    chord_root = 1.8288; chord_tip = 1.8288
                    span = 6.096  # m
                    beam_length = span
                    sh_offset = 0.0 # as a fraction of chord, e.g. 0.25 for quarter-chord, 0.5 for mid-chord
                
                    # CAERO1 card points
                    caero_id = 1; pid = 1; igid = 1
                
                    # Elastic axis reference (at 0.33*chord from LE)
                    x1 = -sh_offset * chord_root; x4 = -sh_offset * chord_tip
                    y1 = -0; y4 = beam_length
                    z1 = 0; z4 = 0
                
                elif blade_name == 'hollowell':
                    # Hollowell configuration
                    chord_root = 0.076; chord_tip = 0.076
                    span = 0.305  # m
                    beam_length = span
                
                    # CAERO1 card points
                    caero_id = 1; pid = 1; igid = 1
                
                    # Center the aerogrid: LE at -chord/2, TE at +chord/2
                    # This matches SONATA's centered airfoil definition
                    x1 = -0.038; x4 = -0.038  # LE at -chord/2 = -0.076/2
                    y1 = 0; y4 = beam_length
                    z1 = 0; z4 = 0

                elif blade_name == 'BAH144B':
                    # Hollowell configuration
                    chord_root = 0.052592; chord_tip = 0.052592
                    span = 0.1403374314  # m
                    beam_length = span
                
                    # CAERO1 card points
                    caero_id = 1; pid = 1; igid = 1
                
                    # Center the aerogrid: LE at -chord/2, TE at +chord/2
                    # This matches SONATA's centered airfoil definition
                    x1 = 0; x4 = +0.037603  # LE at -chord/2 = -0.076/2
                    y1 = 0; y4 = beam_length
                    z1 = 0; z4 = 0

                elif blade_name == '1x1grid':
                    # Hollowell configuration
                    chord_root = 1; chord_tip = 1
                    span = 1  # m
                    beam_length = span
                
                    # CAERO1 card points
                    caero_id = 1; pid = 1; igid = 1
                
                    # Center the aerogrid: LE at -chord/2, TE at +chord/2
                    # This matches SONATA's centered airfoil definition
                    x1 = -0.5; x4 = -0.5  # LE at -chord/2 = -0.076/2
                    y1 = 0; y4 = beam_length
                    z1 = 0; z4 = 0

                elif blade_name == 'grid_conv':
                    # Hollowell configuration
                    chord_root = 0.5; chord_tip = 0.5
                    span = 3  # m
                    beam_length = span
                
                    # CAERO1 card points
                    caero_id = 1; pid = 1; igid = 1
                
                    # Center the aerogrid: LE at -chord/2, TE at +chord/2
                    # This matches SONATA's centered airfoil definition
                    x1 = 0; x4 = 0  # LE at -chord/2 = -0.076/2
                    y1 = 0; y4 = beam_length
                    z1 = 0; z4 = 0
                
                elif blade_name == 'NACA0003':
                    # NACA0003 airfoil configuration
                    chord_root = 0.095; chord_tip = 0.095
                    span = 0.15  # m
                    beam_length = span
                
                    # CAERO1 card points
                    caero_id = 1; pid = 1; igid = 1
                
                    x1 = 0.0; x4 = 0.0
                    y1 = 0; y4 = beam_length
                    z1 = 0; z4 = 0

                if blade_name == 'ABRAMSON1965':
                    # GOLAND wing configuration
                    chord_root = 0.3048; chord_tip = 0.3048
                    span = 0.762  # m
                    beam_length = span
                    xc_offset = 0.512 # as a fraction of chord, e.g. 0.25 for quarter-chord, 0.5 for mid-chord
                
                    # CAERO1 card points
                    caero_id = 1; pid = 1; igid = 1
                
                    # Elastic axis reference (at 0.33*chord from LE)
                    x1 = -xc_offset * chord_root; x4 = -xc_offset * chord_tip
                    y1 = -0; y4 = beam_length
                    z1 = 0; z4 = 0

                    plot = True


                else:
                    raise ValueError(f"Unknown blade_name: {blade_name}")       
                
                study_case = f"{blade_name}_{fluid}_alpha{attack_angle[0]}_nspan{n_span}_nchord{n_chord_count}"
                output_dir = FSI_path + '/PanelAero/Qjj/CAERO1_cards'
                wing = output_dir + '/' + study_case + '.CAERO1' # Read CAERO1 card

                ## ==== Generate CAERO1 card blade (flat panel) ==== ##
                print(f"DEBUG: Calling format_caero1_card with:")
                print(f"  x1={x1}, y1={y1}, z1={z1}")
                print(f"  x4={x4}, y4={y4}, z4={z4}")
                print(f"  chord_root={chord_root}, chord_tip={chord_tip}")
                format_caero1_card(wing, caero_id, pid, igid, n_span, n_chord_count, x1, y1, z1, x4, y4, z4, chord_root, chord_tip)
                builder = build_aeromodel.AeroModel(wing)
                aerogrid = builder.build_aerogrid()
                aerogrid = builder.aerogrid

                aerogrid = rotate_aerogrid._rotation(aerogrid, alpha_r, axis='y') # rotate aerogrid

                aerogrid_dict[(n_span, n_chord_count)] = aerogrid
                print(f"Generated aerogrid for n_span={n_span}, n_chord={n_chord_count}")

                # Optional: plot each aerogrid
                if plot == True:
                    plot_aero_beam_model(aerogrid, beam_model=None)
            
            if method == 2:
                # original curve points
                # Dynamically import the LE/TE sub-modules from the blade's TELE_coords directory
                # and locate the point arrays within them.
                import importlib
                import importlib.util

                def _import_submodule(blade, submodule_name):
                    """Import LE_curve_points or TE_curve_points from the blade's dir."""
                    
                    if blade == 'tnz_arm':
                        gr_dir = Path(FSI_path) / 'SONATA' / 'ETNZ' / blade
                        mod_file = gr_dir / f'{submodule_name}.py'

                        spec = importlib.util.spec_from_file_location(
                            f"SONATA.ETNZ.{blade}.{submodule_name}", mod_file
                        )
                    else:
                        gr_dir = Path(FSI_path) / 'SONATA' / blade / 'TELE_coords'
                        mod_file = gr_dir / f'{submodule_name}.py'

                        spec = importlib.util.spec_from_file_location(
                            f"SONATA.{blade}.TELE_coords.{submodule_name}", mod_file
                        )
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod

                def _find_points(module, candidates):
                    for cand in candidates:
                        for attr in dir(module):
                            if attr == cand or attr.startswith(cand):
                                val = getattr(module, attr)
                                # accept array-like values
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

                le_candidates = ['le_points', f'le_points_{blade_name}', 'LE_points', 'le_pts', 'le']
                te_candidates = ['te_points', f'te_points_{blade_name}', 'TE_points', 'te_pts', 'te']

                le_pts = _find_points(le_mod, le_candidates)
                te_pts = _find_points(te_mod, te_candidates)

                # Convert from mm to meters
                le_pts = le_pts / 1000.0
                te_pts = te_pts / 1000.0
                
                # Create interpolation functions
                # Use cubic if >= 4 points, quadratic if 3, linear if 2
                def make_curve_callable(points):
                    n = len(points)
                    kind = 'cubic' if n >= 4 else ('quadratic' if n >= 3 else 'linear')
                    u = np.linspace(0, 1, n)
                    fx = interp1d(u, points[:, 0], kind=kind, fill_value="extrapolate")
                    fy = interp1d(u, points[:, 1], kind=kind, fill_value="extrapolate")
                    fz = interp1d(u, points[:, 2], kind=kind, fill_value="extrapolate")
                    return lambda t: np.array([fx(t), fy(t), fz(t)])

                le_curve_orig = make_curve_callable(le_pts)
                te_curve_orig = make_curve_callable(te_pts)

                def _rotate_points_z(pts, angle_deg, center=(0.0, 0.0, 0.0)):
                    pts = np.asarray(pts)
                    theta = np.deg2rad(angle_deg)
                    c = math.cos(theta)
                    s = math.sin(theta)
                    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                    return ((pts - center) @ R.T) + center

                def _rotate_points_y(pts, angle_deg, center=(0.0, 0.0, 0.0)):
                    pts = np.asarray(pts)
                    theta = np.deg2rad(angle_deg)
                    c = math.cos(theta)
                    s = math.sin(theta)
                    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
                    return ((pts - center) @ R.T) + center

                # Create rotated wrappers for the curves: rotate by -90 deg about Z around absolute origin
                def _make_rotated_curve(orig_func, angle_deg, y=None, z=None):
                    if z:
                        def wrapped(u):
                            p = np.array(orig_func(u), dtype=float)
                            return _rotate_points_z(p, angle_deg, center=(0.0, 0.0, 0.0))
                    if y:
                        def wrapped(u):
                            p = np.array(orig_func(u), dtype=float)
                            return _rotate_points_y(p, angle_deg, center=(0.0, 0.0, 0.0))
                    return wrapped
                
                # Use original curves without rotation
                le_curve = le_curve_orig
                te_curve = te_curve_orig

                # Build aerogrid (use the loop n_span, n_chord values)
                aero = build_aeromodel_crvs.AeroGridFromCurves(le_curve, te_curve, n_span=n_span, n_chord=n_chord_count)
                aerogrid = aero.build_aerogrid(eid_start=1000)

                print("Aerodynamic grid created successfully.")

                # store aerogrid in dictionary so downstream code (precompute) runs as for method==1
                aerogrid_dict[(n_span, n_chord_count)] = aerogrid

                # --- 3D plotting of aerogrid and curves ---
                if plot == True:
                    plot_aerogrid(FSI_path, aerogrid, le_curve, te_curve, n_span, n_chord_count, blade_name, fluid, attack_angle)


    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total grids to process: {len(aerogrid_dict)}")
    print(f"  k values: {len(k_list)}")
    print(f"  Mach values: {len(Ma_list)}")
    print(f"  Computations per grid: {len(k_list) * len(Ma_list)}")
    print(f"  Total Qjj computations: {len(aerogrid_dict) * len(k_list) * len(Ma_list)}")
    print(f"{'='*60}\n")

    output_dir = Path("qjj_precomputed")
    output_dir.mkdir(exist_ok=True)

    for (n_span, n_chord_count), aerogrid in aerogrid_dict.items():

        print(f"Saving aerogrid at path: {output_dir}")
        
        out_dir = FSI_path + f"/PanelAero/Qjj/qjj_precomputed/{blade_name}_{fluid}_alpha{attack_angle[0]}_nspan{n_span}_nchord{n_chord_count}_{DLM_method}"
        out_dir_vlm = FSI_path + f"/PanelAero/Qjj/qjj_precomputed/vjj_{blade_name}_{fluid}_alpha{attack_angle[0]}_nspan{n_span}_nchord{n_chord_count}_klist_{DLM_method}"
        
        print(f"Precomputing Qjj for n_span={n_span}, n_chord={n_chord_count}")
        # ensure output directory exists (precompute may expect it)
        from pathlib import Path as _Path
        _Path(out_dir).mkdir(parents=True, exist_ok=True)

        t_dlm_start = time.time()
        if DLM == True:
            precompute_qjj_grid(FSI_path, aerogrid, k_list, Ma_list, out_dir, dtype=np.float32, resume=True, verify_existing=True, dlm_method=DLM_method)
        t_dlm_elapsed = time.time() - t_dlm_start

        t_vlm_start = time.time()
        if VLM == True:
            precompute_qjj_vlm(FSI_path, aerogrid, Ma_list, out_dir_vlm, dtype=np.float32, verbose=True, resume=True, verify_existing=False)
        t_vlm_elapsed = time.time() - t_vlm_start

        # Save the aerogrid alongside the precomputed Qjj files so downstream
        # flutter analysis can quickly load and plot the geometry.
        try:
            aerogrid_file = _Path(out_dir) / 'aerogrid.npz'
            # Use savez_compressed to store arrays efficiently. aerogrid is a dict of arrays.
            np.savez_compressed(aerogrid_file, **aerogrid)
            print(f"Saved aerogrid to {aerogrid_file}")
        except Exception as _e:
            print(f"Warning: could not save aerogrid to {out_dir}: {_e}")
        #precompute_qjj_vlm(FSI_path, aerogrid, Ma_list, out_dir_vlm, dtype=np.float32, verbose=True, resume=True, verify_existing=False)

        # ------------------------------------------------------------------ #
        # POST-COMPUTATION DIAGNOSTICS
        # ------------------------------------------------------------------ #

        # --- 1. Panel aspect ratios ---
        # l  : chordwise length of each panel (scalar, x-component of mean chord vector)
        # r  : spanwise vector P1->P3 of each panel
        # A  : panel area
        chord_lengths = np.abs(aerogrid['l'])                    # chordwise size  [n_panels]
        span_lengths  = np.linalg.norm(aerogrid['r'], axis=1)   # spanwise size   [n_panels]
        panel_AR      = span_lengths / chord_lengths             # AR per panel

        print(f"\n{'='*60}")
        print(f"PANEL ASPECT RATIOS  (n_span={n_span}, n_chord={n_chord_count})")
        print(f"  Number of panels   : {aerogrid['n']}")
        print(f"  Chord length  - min: {chord_lengths.min():.4f} m   max: {chord_lengths.max():.4f} m   mean: {chord_lengths.mean():.4f} m")
        print(f"  Span  length  - min: {span_lengths.min():.4f} m   max: {span_lengths.max():.4f} m   mean: {span_lengths.mean():.4f} m")
        print(f"  Panel AR      - min: {panel_AR.min():.4f}         max: {panel_AR.max():.4f}         mean: {panel_AR.mean():.4f}")
        print(f"{'='*60}\n")

        # --- 2. Qjj norms at all k values for a specific speed ---
        # Choose speed and three reduced-frequency samples spread across k_list
        V_probe = 25.0   # m/s  <-- adjust to a speed of interest
        k_probe_indices = list(range(len(k_list)))   # all k indices

        # Find the Mach index closest to V_probe
        c_sound_local = c_sound[fluid]
        Ma_probe = V_probe / c_sound_local
        i_Ma_probe = int(np.argmin(np.abs(Ma_list - Ma_probe)))
        Ma_actual  = Ma_list[i_Ma_probe]

        print(f"\n{'='*60}")
        print(f"Qjj NORMS  (n_span={n_span}, n_chord={n_chord_count})")
        print(f"  Probe speed V = {V_probe} m/s  ->  Ma = {Ma_probe:.6f}  (nearest stored Ma[{i_Ma_probe}] = {Ma_actual:.6f})")
        print(f"  {'k_index':>8}  {'k':>10}  {'||Qjj|| (Frobenius)':>22}  {'||Re(Qjj)||':>14}  {'||Im(Qjj)||':>14}")
        print(f"  {'-'*75}")

        # Collect Qjj norms across all k values for this aerogrid
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

        # Aggregate norms into a single row for this aerogrid
        def _agg(vals, fn): return round(fn(vals), 8) if vals else ''
        row = {
            'blade_name':             blade_name,
            'fluid':                  fluid,
            'DLM_method':             DLM_method,
            'n_total_cases':          len(aerogrid_dict) * len(k_list) * len(Ma_list),
            'n_panels':               int(aerogrid['n']),
            'nspan':                  n_span,
            'nchord':                 n_chord_count,
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

    # Attach total execution time to every row
    for row in csv_rows:
        row['total_execution_time_s'] = round(total_time, 2)

    csv_out_path = _Path(out_dir).parent / 'executer_diagnostics.csv'
    with open(csv_out_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDiagnostics CSV saved to: {csv_out_path}")


def plot_aerogrid(FSI_path, aerogrid, le_curve, te_curve, n_span, n_chord_count, blade_name, fluid, attack_angle):
    import numpy as np
    import matplotlib.pyplot as plt

    us = np.linspace(0.0, 1.0, 300)
    # These are the curves as used by the grid builder
    le_sample = np.array([le_curve(u) for u in us])
    te_sample = np.array([te_curve(u) for u in us])

    # Extract grid points and panel connectivity
    grid_ids = aerogrid['cornerpoint_grids'][:, 0]
    grid_pts = aerogrid['cornerpoint_grids'][:, 1:4]
    panels = aerogrid['cornerpoint_panels']

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot corner points
    ax.scatter(grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2], s=8, c='k', alpha=0.6)

    # Plot panel edges
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

    # Plot LE and TE curves
    ax.plot(le_sample[:, 0], le_sample[:, 1], le_sample[:, 2], color='red', linewidth=2.0, label='LE curve')
    ax.plot(te_sample[:, 0], te_sample[:, 1], te_sample[:, 2], color='blue', linewidth=2.0, label='TE curve')

    # Plot panel normals
    panel_centers = aerogrid['offset_k']
    panel_normals = aerogrid['N']
    ax.quiver(panel_centers[:, 0], panel_centers[:, 1], panel_centers[:, 2],
                panel_normals[:, 0], panel_normals[:, 1], panel_normals[:, 2],
                length=0.1 * np.mean(aerogrid['l']), normalize=True, color='green', label='Normals')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Aerogrid (n_span={n_span}, n_chord={n_chord_count})')
    ax.legend()
    plt.tight_layout()
    # Make Z view width equal to X view width
    try:
        import pathlib as _pathlib
        # collect plotted data ranges
        all_x = np.concatenate([grid_pts[:, 0], le_sample[:, 0], te_sample[:, 0]])
        all_y = np.concatenate([grid_pts[:, 1], le_sample[:, 1], te_sample[:, 1]])
        all_z = np.concatenate([grid_pts[:, 2], le_sample[:, 2], te_sample[:, 2]])

        x_min, x_max = float(all_x.min()), float(all_x.max())
        y_min, y_max = float(all_y.min()), float(all_y.max())
        z_min, z_max = float(all_z.min()), float(all_z.max())

        x_width = x_max - x_min
        z_center = 0.5 * (z_max + z_min)
        z_min_new = z_center - 0.5 * x_width
        z_max_new = z_center + 0.5 * x_width

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min_new, z_max_new)
    
    except Exception as _e:
        print(f"Could not set equal z/x width for plot: {_e}")

    # Save plot to the qjj_precomputed folder for this study case
    try:
        plot_out_dir = Path(FSI_path + "/PanelAero/Qjj/qjj_precomputed/{blade_name}_{fluid}_alpha{attack_angle[0]}_nspan{n_span}_nchord{n_chord_count}_klist_new")
        plot_out_dir.mkdir(parents=True, exist_ok=True)
        plot_file = plot_out_dir / 'aerogrid.png'
        fig.savefig(plot_file, dpi=200)
        print(f"Saved aerogrid plot to {plot_file}")
    except Exception as _e:
        print(f"Warning: could not save aerogrid plot: {_e}")

    plt.show()


def play_completion_sound():
    try:
        import os
        os.system('echo -n "\a"')  # ASCII bell character
    except Exception as e:
        print(f"Could not play sound: {e}")


if __name__ == "__main__":
    #from precompute_qjj import precompute_qjj_grid, check_existing_qjj_files
    #check_existing_qjj_files("/media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/PanelAero/Qjj/qjj_precomputed/GOLAND_air_alpha2_nspan50_nchord30_klist_new")
    start_time = time.time()
    try:
        main(start_time=start_time)
    finally:
        play_completion_sound()
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
