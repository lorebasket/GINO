# FSI/examples/config.py
#
# Central configuration module.
# - GLOBAL_FLAGS  : default parameters shared by all cases.
# - AnalysisConfig: class that merges global defaults with case-specific kwargs.
# - get_config    : dynamically loads a per-case config from
#                   FSI/examples/<case_name>/config_<case_name>.py

import numpy as np
import sys
import os
import importlib.util

# Derive the FSI root: FSI/examples/config.py → FSI/
FSI_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DATA_ROOT = os.path.join(FSI_path, 'output_data')
QJJ_path = os.path.join(FSI_path, 'FLUID', 'PanelAero', 'Qjj', 'qjj_precomputed')


def case_output_dir(case_name: str) -> str:
    """Per-case output root: output_data/<case_name>/"""
    return os.path.join(OUTPUT_DATA_ROOT, case_name)

sys.path.append(os.path.join(FSI_path, 'STRUCTURE', 'SONATA', '6_AGARD445.6', 'csv_export'))

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL FLAGS — interchangeable defaults shared across all analysis cases.
# Each case file only needs to specify values that DIFFER from these.
# ══════════════════════════════════════════════════════════════════════════════
GLOBAL_FLAGS = {

    # ── Beam model ────────────────────────────────────────────────────────────
    #'n_elements':                       200,
    #'nspan':                            32,
    #'nchord':                           16,

    # ── Modal analysis ────────────────────────────────────────────────────────
    'num_modes_egv':                    4, # number of computed modes dry analysis
    'num_modes_flutter_egv':            2,  # number of modes used in flutter analysis

    # ── Fluid properties ──────────────────────────────────────────────────────
    'c_sound':  {'air': 332.5, 'water': 1484.0},   # [m/s]  at ~2000 m
    'rho_f':    {'air': 1.02,  'water': 997},       # [kg/m³]

    # ── Output & saving ───────────────────────────────────────────────────────
    'save_plots':                       True,
    'output_dir':                       OUTPUT_DATA_ROOT,  # resolved to OUTPUT_DATA_ROOT/<case> in AnalysisConfig
    # DLM ΔCp snapshots during flutter V-sweep (5 points: start, end, 3 interior).
    # Keep this disabled by default to avoid heavy per-step Cp image dumps.
    'plot_cp_sweep_snapshots':          False,
    'plot_cp_sweep_n_snapshots':        5,
    'plot_cp_sweep_mode_slot':          0,   # tracked-mode index for converged k = ω/V
    'plot_cp_sweep_modal_shape':        True,  # also ΔCp with wj=(ik·Z+Z_qs)@φ (flutter physics)
    'save_global_matrices':             False,
    'save_matrices':                    False,
    'save_modal_data':                  True,

    # ── Visualization ─────────────────────────────────────────────────────────
    'plot_mode_shapes':                 False,
    'plot_normalized_mode_shapes':      False,  # extra mode-shapes plot, each mode scaled to unit max(|transl|, b*|rot|)
    'plot_aerobeam':                    False,
    'plot_full_wing':                   False,
    'plot_chordwise_strip':             False,
    'plot_deformed_modes':              False,
    'plot_displacements_by_beam':       False,
    'animate_modes':                    False,
    'animation_frames':                 20,
    'mode_scale_factor':                0.5,
    'classify_modes':                   False,   # analyses modes energy distribution
    'show_eigen_plot':                  True,
    'plot_pk_wet_eigenvectors':         True,  # legacy flag (untracked plot always runs if raw_results exist)
    'plot_pk_converged_mac':            True,  # alias: plot_converged_mac (PK / Roger / RFA+Capytaine)
    'plot_converged_mac':               True,  # MAC of converged eigenvectors vs prev V (+ PK first iter)
    'show_pk_converged_mac_plot':       False,  # display converged_mac.png interactively
    'show_converged_mac_plot':          False,
    'plot_orphan_wet_root_on_vgvf':     False,  # overlay MAC-free lowest-untracked / spectrum-index branch on V-g/V-f
    'orphan_conditional_mac':           False,  # after min span of continuous free track, MAC-lock orphan branch
    'orphan_mac_min_span_m_s':          5.0,    # [m/s] free-track duration before MAC activates
    'orphan_mac_max_freq_jump_frac':    0.30,   # max relative |Δω|/ω between steps for continuity
    'orphan_mac_append_vgvf_branch':    True,   # add orphan as Branch 3 in V-g/V-f outputs
    'plot_modal_aero_bend_pitch_diagnostic': False,  # 2×2 Q_modal bend–pitch block + Z_qs/(ikZ) ratio vs V
    'plot_DLM_participants':            True,  # C_hat, K_hat, C_eff, K_eff vs V (Roger RFA / _build_A_aug)

    # ── Static hydroelastic pass (parallel to flutter sweep) ─────────────────
    # VLM (vjj_dir) only; w_ref=0 by default (incidence already in aerogrid).
    'static_hydroelastic_pass':         False,
    'static_hydroelastic_w_ref_perturbation_deg': 0.0,  # extra uniform panel downwash [deg], not in lattice
    'plot_static_hydroelastic_sweep':   False,

    'plot_stiffness_damping_contributions': True,  # K/C struct vs aero vs eff (PK or RFA-PK)
    'show_stiffness_damping_contributions_plot': False,  # display the M/K/C contribution figure interactively
    'plot_added_mass_capytaine_vs_strip': False,  # RFA-PK only: compare Capytaine AM with strip-theory AM
    'plot_abramson_imq_comparison':      False,  # ABRAMSON1965: Im(Q_DLM) vs measured curve-3 at same k

    # ── Force computation ─────────────────────────────────────────────────────
    'compute_tip_forces':               False,
    'track_tip_forces_during_flutter':  False,

    # ── Beam reference geometry (beam_geometry / beam_properties) ─────────────
    # Span along +Y; elastic axis at (beam_axis_x_offset, 0, beam_axis_z_offset) at root.
    # sweep_angle_deg: planform sweep in X–Y [deg] (0 → straight beam).
    # Section K/M are in local frame (v=beam, u=chord); per-element T6 maps them to global.
    # section_axes_along_global_span: keep u=+X, v=+Y, w=+Z even when nodes follow sweep.
    # structural_model_type: None (auto) | 'custom' → _build_structural_model_custom in structural_model.py
    'structural_model_type':            None,
    'flip_mass_cg_coupling_sign':       False,  # Test only: invert both M[2,4] and M[1,5] CG–SC coupling signs
    'flip_mass_cg_coupling_24_sign':    False,  # Test only: invert M[2,4] (w↔θy) only; M[1,5] unchanged
    'sweep_angle_deg':                  0.0,
    'section_axes_along_global_span':   False,
    'beam_axis_x_offset':               0.0,   # [m] chordwise EA position at root
    'beam_axis_z_offset':               0.0,   # [m] vertical EA position at root
    # decouple_xea_from_aerogrid: vary structural xea_factor while keeping a precomputed aerogrid
    # fixed at aerogrid_xea_factor (LE at -aerogrid_xea_factor * chord). Beam nodes shift by
    # (xea_factor - aerogrid_xea_factor) * chord; section M/K and coupling use xea_factor.
    'decouple_xea_from_aerogrid':       False,
    'aerogrid_xea_factor':              None,  # reference LE→SC fraction for aerogrid / CAERO1

    # ── Post-build dihedral (post_dihedral_utils) ───────────────────────────────
    # When dihedral_angle != 0: rotate beam (nodes, K/M, modes) and/or DLM aerogrid about +X.
    'dihedral_rotate_beam':             True,
    'dihedral_rotate_aerogrid':         True,

    # ── Structural physics ────────────────────────────────────────────────────
    'geometric_nonlinearity':           False,
    'include_gravity':                  False,
    'gravity_acc':                      np.array([0, 0, -9.81]),  # [m/s²]
    'constrained_dofs':                 list(range(6)),

    # ── Fluid-at-rest (wet modes) ─────────────────────────────────────────────
    'fluid_at_rest':                    False,
    'modal_capytaine_dofs':             False,
    
    # ── Capytaine radiation ────────────────────────────────────────────────────
    'rigid_body_motion':                False,  # Capytaine radiation (run_modal_radiation.py): True → 6 rigid-body DOFs; False → modal / --modes face fields.
    'rigid_body_rotation_center':       None,  # Optional (x, y, z) [m] for rigid-body rotations; if None, Capytaine uses mesh area centroid of face centers.
    'capytaine_phi_normalization':      'geometric',  # max|phi|→1 for BEM; A,B saved in mass-normalized modal basis
    # main.py: reuse Capytaine NPZ when source inputs match; recompute when stale/missing.
    'capytaine_auto_recalc':            True,
    'capytaine_force_recalc':           False,

    # ── FLUTTER analysis method ───────────────────────────────────────────────
    # [GINO_NEW FIX] default to pure P-K (iterates the true Qjj) instead of Roger RFA-PK.
    # The Roger fit with the fixed 4 lag poles [0.5,1,1.5,2] produced spurious unstable poles in
    # water (fit err ~226%). P-K is robust. Set pk_method=False + RFA_PK_method=True to restore old.
    'pk_method':                        True,   # P-K method aeroelastic analysis
    'roger_fit':                        False,  # False: Roger RFA hydroelastic analysis
    'RFA_PK_method':                    False, # Roger RFA + PK method aeroelastic analysis
    
    # [GINO_NEW FIX] aero conventions for the P-K path (see aero_system.make_Qg_func):
    'aero_real_flip':                   False,  # negate Re(Q); stiffness / K_aero sign from Qjj
    'aero_imag_flip':                   True,  # negate Im(Q); damping / C_aero sign from Qjj
    # [aero_b2_removal] P-K only: cancel the DLM's own apparent mass (-k^2*B2) from Re(Q) so the
    # explicit added mass in M_hat (strip/Capytaine) is the sole non-circulatory inertia (no double
    # count). Mirrors GINO_NEW's hydroelastic correction. Pair with added_mass_strip_theory=True.
    'aero_b2_removal':                  False,

    # ── P-K method — flags ────────────────────────────────────────────────────
    # True: interpolate precomputed Capytaine A(ω), B(ω) at ω=k·V each PK iteration
    # (requires capytaine_results_dir + modal_radiation npz). Incompatible with strip AM in M_hat.
    'pk_capytaine_matrices':            False,
    # PK+Capytaine hydro coupling: 'full' replaces apparent mass via aero_b2_removal + A_capy;
    # 'delta' keeps UVLM GAF intact and injects M_hydro = A_capy(ω) - A_uvlm (SHARPy parity).
    'pk_bem_mode':                      'full',
    'pk_bem_delta_grid':                None,   # None → case default (e.g. n30c12 for Model2t)
    'pk_bem_delta_npz':                 None,   # None → FLUID/capytaine/<case>/bem_delta/...
    'pk_bem_delta_force_recalc':        False,
    'pk_bem_delta_v_probe':             None,   # None → case default probe speeds [m/s]
    'pk_bem_delta_strip_a':             None,   # EA in semichords aft of midchord
    'pk_bem_delta_strip_b':             None,   # semichord [m]; None → case default
    'pk_bem_delta_v_max_kn':            None,   # probe-case velocity list extent
    'pk_bem_delta_rayleigh':            True,
    # Post-process V-g/V-f plots: also save stabilized-band figures for V >= threshold.
    'vgvf_plot_V_min_kn':               None,
    'vgvf_plot_V_min_m_s':              None,
    'mac_matching':                     True,
    'last_converged_mode_matching':     True,
    'k_guess':                          0.001,
    
    # If True, reset k to k_guess at the start of every PK iteration (j >= 1), not only at j=0
    'pk_use_k_guess_each_iter':         False,
    'dimensionless_vgvf_results':       True,
    'plot_log_decrement_vg':            False,  # also save vg_lambda_vf_combined.png (V–δ Abramson, V–g, V–f; PK or RFA)
    'v_knots':                          False,  # plot/export airspeed in knots (stored SI in CSV)
    'vf_hertz':                         False,  # plot/export frequency in Hz (stored rad/s in CSV)

    # ── Roger RFA ─────────────────────────────────────────────────────────────
    # 'panelaero_legacy': Roger fit / PK EOM in k_pa=ω/V (pre-2026 behaviour).
    # 'paper_zurich': Roger fit / PK EOM in k*=ωb/V (Kier eq. 16–21); Qjj load still k_pa.
    'rfa_roger_convention':             'panelaero_legacy',

    'k_list':                           np.linspace(0.001, 50, 100),
    'n_lag':                            2,
    'blag':                             np.array([0.5, 1.0]),
    'rfa_blag_dimensionless':           False, # True if rfa_roger_convention: 'paper_zurich', False if 'panelaero_legacy'
    'rfa_blag_ref_length':              'semichord',
    'rfa_blag_auto':                    False,  # True: derive beta poles from rfa_blag_V_flutter_kn + f_ref
    'rfa_blag_V_flutter_kn':            83.0,  # target flutter speed [kn] for auto pole placement
    'rfa_blag_f_ref_hz':                (2.0, 14.0, 27.0, 37.0),  # reference dry mode frequencies [Hz]
    
    'rfa_adaptive_k_list':              False, # False: fit Q(k) on fixed config.k_list / config.blag at every V (smoother V–g–f)
    'rfa_adaptive_blag':                False, # False: fit Q(k) on fixed config.k_list / config.blag at every V (smoother V–g–f)
    'rfa_use_precomputed_k_nodes':      True,  # Use exact k nodes from Qjj index (lower interpolation noise in Roger fit)
    'qjj_ma_interp_mode':               'auto', # 'bilinear' | 'nearest' | 'auto' (auto -> nearest for very low Mach)
    'qjj_ma_low_threshold':             0.05,  # Mach threshold used by qjj_ma_interp_mode='auto'
    
    # MAC on [q; q_dot] only — lag states must not drive mode tracking
    'rfa_mac_structural_dof':           True, # MAC on [q; q_dot] only — lag states must not drive mode tracking
    # MAC reference for RFA tracking / branch guard / reacquire:
    # 'previous_velocity' → wet eigenvector at V-1 (mode shapes may evolve);
    # 'last_valid' → latest converged step before mode_lost gaps.
    'rfa_mac_reference':                'previous_velocity',
    'rfa_apply_b2_mass':                False, # False = hydroelastic correction: AM in M_hat (strip/Capytaine), Roger without B2
    'rfa_k_fit_min':                    0.05, # Exclude k → 0 from Roger LSQ (dominates error, not flutter physics)
    'rfa_k_fit_max':                    40.0, # Upper k cap for Roger LSQ (reduces far-field overfit)
    'rfa_k_fit_adaptive_band':          True, # Narrow k_fit to tracked ω_prev/V band each step
    'rfa_k_fit_band_lo_frac':           0.35,
    'rfa_k_fit_band_hi_frac':           3.5,
    'rfa_weight_fit_k':                 True, # Weight Roger fit toward k ≈ ω_tracked/V at each airspeed step
    'rfa_weight_fit_sigma':             0.6,
    'rfa_report_tracked_fit_error':     True,
    'rfa_predict_and_select':           True, # PK-style pool reorder from V step 2 (allows branch crossing)
    'rfa_min_root_sep_rad':             1.0,  # Min |Δω| between branches at same V
    'rfa_branch_guard_enable':          True,  # Roger/RFA+Capytaine: mark mode_lost on incoherent branch jump
    'rfa_branch_guard_max_rel_jump':    0.30,  # Max |ω-ω_ref|/ω_ref before declaring mode_lost
    'rfa_branch_guard_min_mac_prev':    0.70,  # Extra guard when |Δω|/ω_ref large but MAC_prev is low
    'rfa_branch_guard_max_struct_frac': 1.35,  # Max ω/ω_struct,dry per branch (orphan high-ω capture)
    'rfa_branch_guard_min_sep_frac':    0.08,  # Min normalized separation between adjacent tracked branches
    'rfa_branch_guard_check_neighbor_collapse': True,  # Mark lost if a previously separated branch collapses onto neighbor
    'rfa_branch_reacquire_enable':        True,  # After mode_lost, search pool near predicted ω(V) to reacquire branch
    'rfa_branch_reacquire_rel_band':      0.20,  # Initial |Δω|/ω_pred search band for reacquisition
    'rfa_branch_reacquire_rel_band_growth': 0.04,  # Widen band per consecutive lost step
    'rfa_branch_reacquire_rel_band_max':  0.35,  # Cap search band (avoid orphan-pool false positives)
    'rfa_branch_reacquire_min_mac':       0.65,  # Min MAC vs last valid eigenvector to accept reacquisition
    'rfa_branch_reacquire_min_partner_sep_frac': 0.08,  # Reacquired ω must stay separated from partner branch
    'rfa_branch_reacquire_prefer_partner_streak': 1,  # After N lost steps, centre search on partner coalescence
    'rfa_pool_gamma_abs_max':           5.0,  # Drop only strongly aperiodic lag artefacts (not g>0 flutter)
    'rfa_pool_omega_max_frac':          1.35, # Drop roots above frac × max tracked ω (dry or wet)
    'rfa_pool_omega_min_frac':          0.35,
    'rfa_pool_coalesce_sep_rad':        2.0,  # Skip wet-band tightening when branches coalesce
    'capytaine_asymptotic_omega_frac':  0.5, # Capytaine asymptotic band for RFA_PK_method (ω >= om_min + frac*(om_max-om_min))
    'capytaine_asymptotic_omega_min':   np.nan,
    # RFA+Capytaine: embed A_∞, B_∞ only on these modal DOF indices (rest = 0).
    # None → all dry structural modes 0..num_modes_egv-1 (from initial modal analysis).
    'capytaine_apply_mode_indices':     None,
    # Row/col indices inside the Capytaine NPZ matrices (default 0..n_capy-1).
    'capytaine_source_mode_indices':    None,
    # RFA+Capytaine: per-velocity structural block at wet-pool DOF indices.
    # False → legacy constant embedding on all dry modes (0..num_modes_egv-1).
    'capytaine_structural_pool_embedding': False,
    # Only if capytaine_structural_pool_embedding=True:
    # True  → pool map updated each V step (bootstrap + MAC tracking).
    # False → static block on modes_to_analyze dry DOF indices.
    'capytaine_dynamic_physical_mask':  True,
    'aero_im_Q_scale':                  1.0, # Scale Im(Q) only in C_eff / C_aero assembly (Roger RFA fit uses unscaled Q_modal)
    'aero_im_Q_scale_lags':             False, # Also scale Roger lag-state aerodynamic forcing q_dyn*Blag by aero_im_Q_scale.
    'empirical_fluid_damping':          False, # Empirical fluid damping (tank / gaps / unmodeled dissipation; not Capytaine radiation)
    'empirical_fluid_model':            'abramson_delta', # 'abramson_delta' | 'constant_kg_s' | 'velocity_linear' | 'delta_at_omega'
    'empirical_fluid_delta_add_s':      4.0,   # extra δ [s⁻¹] in Abramson Fig. 5 gap → C_ii=2δM_ii
    'empirical_fluid_C_kg_s':           15.0,  # diagonal [kg/s] for model='constant_kg_s'
    'empirical_fluid_C_per_ms':         0.0,   # add C_ii += value * V [kg/s per m/s]
    'empirical_fluid_omega_ref_rad_s':  70.0,  # for model='delta_at_omega'
    
    # ── P-K solver — iteration & mode-matching parameters ────────────────────
    # When True, PK uses Newmark-beta discrete Ass (SHARPy lingebm.newmark_ss) instead
    # of the continuous companion Ac.  Structural C_hat is ignored; damping enters
    # only via newmark_damp / newmark_dt (set structural_damping_model='none').
    'pk_discrete_newmark':              False,
    # When True (with NPZ built via --sharpy-structural-basis), UVLM PK uses
    # SHARPy modal M,K,ω from the Quvlm table instead of GINO FEA reduction.
    'quvlm_sharpy_structural_basis':    False,
    'pk_tol':                           1e-2, # Tolerance for PK iteration convergence
    'pk_fXK0':                          0.618, # Factor for XK0 update in PK iteration
    'pk_fRLX':                          0.7, # Factor for RLX update in PK iteration
    'pk_freq_margin':                   0.05, # Frequency margin for PK iteration
    'pk_perturb_k':                     1, # Perturbation factor for k in PK iteration
    'pk_max_iter':                      150, # Maximum number of iterations for PK iteration
    'pk_w_freq':                        0.8, # Weight for frequency in PK iteration
    'pk_w_mac':                         0.6, # Weight for MAC in PK iteration
    'pk_skip_inter_mode_on_perfect_mac': True, # Skip inter-mode frequency separation when MAC pre-selector = 1.0 (crossed modes)
    'pk_predict_and_select':             True, # Yuan & Zhang (2023): linear p extrapolation reorder between airspeed steps
    'pk_min_root_sep_rad':                1.0, # Min |Δω| [rad/s] between modes at same V (allows crossing, blocks coalescence)
    'pk_omega_phys_min_frac':             0.45, # Drop wet roots below frac×min(ω_n) in MAC / untracked plot

    # ── Roger RFA — MAC mode-tracking parameters ──────────────────────────────
    'rfa_freq_margin':                  0.20,
    'rfa_w_freq':                       0.8,
    'rfa_w_mac':                        0.6,
    # Fit Roger on modes_to_analyze only when fewer modes are tracked than num_modes_egv.
    'rfa_fit_tracked_modes_only':       True,
    # Drop wet roots below min(ω_tracked)*rfa_omega_phys_min_frac (lag/spurious roots).
    'rfa_omega_phys_min_frac':          0.45,

    # ── Structural damping ────────────────────────────────────────────────────
    # 'rayleigh' → αM+βK from rayleigh_mode_ids / rayleigh_target_zetas
    # 'newmark'  → mass-proportional C = (2·newmark_damp/dt)·M (SHARPy Newmark-β, C_str=0)
    # 'none'     → zero structural damping
    'structural_damping_model':         'rayleigh',
    'structural_eta':                   0.0,   # isteretico K(1+i eta), tutti i modi (aero_system.modal_aero_mck)
    'newmark_damp':                     0.5e-4,   # SHARPy Goland notebook default
    # Physical Newmark/UVLM step [s]. Goland fine grid: c/(M·U_ref)=1.8288/(16·165).
    # Do not copy SHARPy's 0.1143 here — that value is scaled (nondim) time.
    'newmark_dt':                       1.8288 / (16.0 * 165.0),

    # ── Rayleigh damping ──────────────────────────────────────────────────────
    # One ζ per dry mode used in the fit (length = num_modes_flutter_egv unless overridden).
    # With >2 modes, α and β are chosen by least squares (exact match only for 2 modes).
    'rayleigh_mode_ids':                None,  # None → 0 .. num_modes_flutter_egv-1
    'rayleigh_target_zetas':            (0.0, 0.0),  # broadcast to all modes when shorter

    # ── Circulatory components ──────────────────────────────────────────────────
    'added_mass_strip_theory':          False, # True → lumped strip AM in M_hat; use added_mass_strip_scale for magnitude
    'added_mass_strip_scale':           1.0,   # multiplier on strip added-mass matrix (1.0 = full strip theory)
    'added_mass_strip_mode_scales':     None,  # optional per-mode diagonal scales on M_hat_added (S M S)
    'M_hat_added_override':             None,  # optional (n_modes, n_modes) added to M_hat after modal reduction
    # PK Pietro/flutter_tools parity (test100): 2-mode slice, Qg at (s=0,j=0), frequency tracking
    'pk_modal_dof_indices':             None,  # e.g. [0, 1] → slice M,C,K,Q before PK
    'pk_qg_eval_mode':                  'default',  # 'default' | 'frozen_j0' (always evaluate Qg at s=0,j=0)
    'pk_tracking_policy':               'paper',  # 'paper' (MAC + initial lock) | 'frequency' (Pietro track_modes)
    'pk_mac_reference':                 'previous_velocity',  # MAC ref: 'previous_velocity' | 'previous_iteration'
    'skip_post_dihedral_coupling':      False,  # True → coupling on reference-frame modes (Pietro test100 parity)
  # RFA_PK / PK+Capytaine: 'modal' = BEM face fields; 'stripstyle' = rigid A_∞ lumped like strip → Φᵀ M_add Φ
    'capytaine_added_mass_source':      'modal',
    # RFA_PK modal source: True → evaluate Capytaine A,B at each mode's structural
    # natural frequency ω_n (off-diagonals at √(ω_i·ω_j)), capturing the actual
    # low-ω radiation damping. False → high-ω asymptotic-tail mean (B≈0 for low-freq modes).
    'capytaine_matrices_at_mode_freq':  True,
    # Primary flutter point: 'min_V' (first g=0 on any branch) or 'freq_band' (coupled-mode band)
    'flutter_primary_selection':          'min_V',
    'flutter_freq_band_hz':             None,  # e.g. (14.0, 22.0) for Abramson bend–torsion
    'flutter_freq_target_hz':           None,  # in-band: prefer crossing closest to this f [Hz]
    'flutter_V_target_kn':              48.1,  # for freq_band_best selection
    # Abramson measured 3-D Q (curve 3) instead of DLM Q_modal for Roger RFA fit
    'aero_use_abramson_measured_q':   False,
    'aero_hybrid_dlm_re_measured_im': False,  # Re(Q)_DLM + Im(Q)_measured on bend–pitch block
    'aero_hybrid_im_weight':          1.0,   # 0=DLM Im, 1=measured Im (blend)
    'aero_measured_q_cal_inv_k':      1.0,   # U = V/(bω) for |Q_DLM|/|H| calibration
    'aero_measured_q_im_flip':        True,  # flip Im(H) to match PK/C_eff sign convention
    'aero_measured_q_im_scale':       1.0,   # scale Im(H) after optional flip (tune 0.2–1.0)
    'resting_fluid_analysis':           False, # Resting fluid analysis (FLUIDREST ANALYSIS)
    'resting_ref_V':                    0.01,   # [m/s] pseudo resting-fluid ref from first flutter point
    'plot_flow_added_damping_benchmark': False,  # Lamoureux Fig. 4: Ω, ζ_added vs U_R (NACA0003 only)
    'plot_vg_vf':                         True,  # plot the V-g and V-f plots

    # ── Velocity sweep (fallback — most cases override this) ─────────────────
    'V_list': np.concatenate([
        np.linspace(5,  12,  2),
        np.linspace(12, 20,  8),
        np.linspace(20, 45, 25),
    ]),

    # ── Aerodynamic panel cards ───────────────────────────────────────────────
    'caero_card_dir': os.path.join(FSI_path, 'FLUID', 'PanelAero', 'panelaero_utl', 'CAERO1_cards'),
    # Flat CAERO1 panel (Qjj/executer.py): set per case in config_<case>.py
    #   caero_x1, caero_y1, caero_z1  — corner 1 (root, chordwise origin)
    #   caero_x4, caero_y4, caero_z4  — corner 4 (tip, chordwise origin)
    #   caero_id, caero_pid, caero_igid — optional (default 1)

    # ── Aerodynamic solver ────────────────────────────────────────────────────
    'aero_source':                      'panelaero',
    'dlm_method':                       'quartic',
}


# ══════════════════════════════════════════════════════════════════════════════

class AnalysisConfig:
    def __init__(self, name, **kwargs):
        self.name = name
        self.paths = {
            'FSI':       FSI_path,
            'SONATA':    os.path.join(FSI_path, 'STRUCTURE', 'SONATA'),
            'PanelAero': os.path.join(FSI_path, 'FLUID', 'PanelAero'),
            'FEA':       os.path.join(FSI_path, 'STRUCTURE', 'FEA'),
        }

        # Apply global defaults first, then case-specific overrides
        for key, value in GLOBAL_FLAGS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if int(self.num_modes_egv) < 1:
            raise ValueError(f"num_modes_egv must be >= 1, got {self.num_modes_egv}")
        if int(self.num_modes_flutter_egv) < 1:
            raise ValueError(f"num_modes_flutter_egv must be >= 1, got {self.num_modes_flutter_egv}")
        if int(self.num_modes_flutter_egv) > int(self.num_modes_egv):
            raise ValueError(
                "num_modes_flutter_egv cannot exceed num_modes_egv: "
                f"{self.num_modes_flutter_egv} > {self.num_modes_egv}"
            )

        # All simulation outputs under output_data/<case_name>/
        if self.output_dir == OUTPUT_DATA_ROOT:
            self.output_dir = case_output_dir(name)
        if 'modal_dir' not in kwargs:
            self.modal_dir = self.output_dir

        # Auto-derive modes lists from num_modes_flutter_egv unless explicitly overridden
        if 'modes_to_analyze' not in kwargs:
            self.modes_to_analyze = list(range(self.num_modes_flutter_egv))
        if 'modes_to_visualize' not in kwargs:
            self.modes_to_visualize = list(range(self.num_modes_flutter_egv))

        self._finalize_rayleigh_damping(kwargs)

    def _finalize_rayleigh_damping(self, kwargs):
        """Align Rayleigh mode ids and ζ targets with num_modes_flutter_egv."""
        n = self.num_modes_flutter_egv

        if self.rayleigh_mode_ids is None or 'rayleigh_mode_ids' not in kwargs:
            self.rayleigh_mode_ids = tuple(range(n))

        mode_ids = tuple(int(i) for i in self.rayleigh_mode_ids)
        zetas = tuple(float(z) for z in self.rayleigh_target_zetas)

        if len(zetas) == 1:
            zetas = (zetas[0],) * len(mode_ids)
        elif len(zetas) == 2 and len(mode_ids) != 2:
            if zetas[0] == zetas[1]:
                zetas = (zetas[0],) * len(mode_ids)
            else:
                zetas = tuple(np.linspace(zetas[0], zetas[1], len(mode_ids)).tolist())
        elif len(zetas) != len(mode_ids):
            raise ValueError(
                f"rayleigh_target_zetas length ({len(zetas)}) must match "
                f"rayleigh_mode_ids length ({len(mode_ids)}); "
                f"or pass one value / two equal values to broadcast"
            )

        self.rayleigh_mode_ids = mode_ids
        self.rayleigh_target_zetas = zetas


# ══════════════════════════════════════════════════════════════════════════════

def get_config(name):
    """
    Load and return the AnalysisConfig for a named case.

    Looks for:  FSI/examples/<name>/config_<name>.py
    That file must define:  get_config(AnalysisConfig, FSI_path, QJJ_path)
    """
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    case_file    = os.path.join(examples_dir, name, f'config_{name}.py')

    if not os.path.exists(case_file):
        available = [
            d for d in os.listdir(examples_dir)
            if os.path.isdir(os.path.join(examples_dir, d))
            and os.path.exists(os.path.join(examples_dir, d, f'config_{d}.py'))
        ]
        raise ValueError(
            f"Configuration '{name}' not found.\n"
            f"Expected file: {case_file}\n"
            f"Available cases: {sorted(available)}"
        )

    spec = importlib.util.spec_from_file_location(f'examples.{name}.config_{name}', case_file)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod.get_config(AnalysisConfig, FSI_path, QJJ_path)
