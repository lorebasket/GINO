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
QJJ_path = os.path.join(FSI_path, 'FLUID', 'PanelAero', 'Qjj', 'qjj_precomputed')

sys.path.append(os.path.join(FSI_path, 'STRUCTURE', 'SONATA', '6_AGARD445.6', 'csv_export'))

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL FLAGS — interchangeable defaults shared across all analysis cases.
# Each case file only needs to specify values that DIFFER from these.
# ══════════════════════════════════════════════════════════════════════════════
GLOBAL_FLAGS = {

    # ── Beam model ────────────────────────────────────────────────────────────
    'n_elements':                       100,
    'nspan':                            18,
    'nchord':                           9,

    # ── Modal analysis ────────────────────────────────────────────────────────
    'num_modes_egv':                    2, # number of computed modes dry analysis
    'num_modes_flutter_egv':            2,  # number of modes used in flutter analysis

    # ── Fluid properties ──────────────────────────────────────────────────────
    'c_sound':  {'air': 332.5, 'water': 1484.0},   # [m/s]  at ~2000 m
    'rho_f':    {'air': 1.02,  'water': 997},       # [kg/m³]

    # ── Output & saving ───────────────────────────────────────────────────────
    'save_plots':                       True,
    'output_dir':                       os.path.join(FSI_path, 'output_data'),
    'save_global_matrices':             False,
    'save_matrices':                    False,
    'save_modal_data':                  True,

    # ── Visualization ─────────────────────────────────────────────────────────
    'plot_mode_shapes':                 False,
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
    'plot_pk_wet_eigenvectors':         True,  # ω(V), g(V) for wet roots not tracked by PK
    'plot_DLM_participants':            True,  # C_hat, K_hat, C_eff, K_eff vs V (Roger RFA / _build_A_aug)
    'plot_stiffness_damping_contributions': True,  # K/C struct vs aero vs eff (PK or RFA-PK)
    'show_stiffness_damping_contributions_plot': False,  # display the M/K/C contribution figure interactively
    'plot_added_mass_capytaine_vs_strip': False,  # RFA-PK only: compare Capytaine AM with strip-theory AM

    # ── Force computation ─────────────────────────────────────────────────────
    'compute_tip_forces':               False,
    'track_tip_forces_during_flutter':  False,

    # ── Post-build pitch (post_pitch_utils) ─────────────────────────────────────
    # When pitch != 0: rotate beam (nodes, K/M, modes) and/or DLM aerogrid about +X.
    'pitch_rotate_beam':                True,
    'pitch_rotate_aerogrid':            True,

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

    # ── FLUTTER analysis method ───────────────────────────────────────────────
    'pk_method':                        False,   # P-K method aeroelastic analysis
    'roger_fit':                        False,  # Roger RFA hydroelastic analysis
    'RFA_PK_method':                    True,

    # ── P-K method — flags ────────────────────────────────────────────────────
    'mac_matching':                     True,
    'last_converged_mode_matching':     True,
    'k_guess':                          0.001,
    
    # If True, reset k to k_guess at the start of every PK iteration (j >= 1), not only at j=0
    'pk_use_k_guess_each_iter':         False,
    'dimensionless_vgvf_results':       True,
    'plot_log_decrement_vg':            False,  # also save vg_lambda_vf_combined.png (V–Λ, V–g, V–f; PK or RFA)
    'v_knots':                          False,  # plot/export airspeed in knots (stored SI in CSV)
    'vf_hertz':                         False,  # plot/export frequency in Hz (stored rad/s in CSV)

    # ── Roger RFA ─────────────────────────────────────────────────────────────
    'k_list':                           np.linspace(0.001, 50, 100),
    'n_lag':                            4,
    'blag':                             np.array([0.5, 1.0, 1.5, 2.0]),
    'rfa_blag_dimensionless':            False,
    'rfa_blag_ref_length':               'semichord',
    
    'rfa_adaptive_k_list':              False, # False: fit Q(k) on fixed config.k_list / config.blag at every V (smoother V–g–f)
    'rfa_adaptive_blag':                False, # False: fit Q(k) on fixed config.k_list / config.blag at every V (smoother V–g–f)
    
    # MAC on [q; q_dot] only — lag states must not drive mode tracking
    'rfa_mac_structural_dof':           True, # MAC on [q; q_dot] only — lag states must not drive mode tracking
    'rfa_apply_b2_mass':                False, # False = hydroelastic correction: AM in M_hat (strip/Capytaine), Roger without B2
    'rfa_k_fit_min':                    0.05, # Exclude k → 0 from Roger LSQ (dominates error, not flutter physics)
    'rfa_weight_fit_k':                 True, # Weight Roger fit toward k ≈ ω_n/V at each airspeed step
    'rfa_weight_fit_sigma':             0.6,
    'capytaine_asymptotic_omega_frac':  0.5, # Capytaine asymptotic band for RFA_PK_method (ω >= om_min + frac*(om_max-om_min))
    'capytaine_asymptotic_omega_min':   np.nan,
    'aero_im_Q_scale':                  1.0, # Scale Im(Q) only in C_eff / C_aero assembly (Roger RFA fit uses unscaled Q_modal)
    'aero_im_Q_scale_lags':             False, # Also scale Roger lag-state aerodynamic forcing q_dyn*Blag by aero_im_Q_scale.
    'empirical_fluid_damping':          False, # Empirical fluid damping (tank / gaps / unmodeled dissipation; not Capytaine radiation)
    'empirical_fluid_model':            'abramson_delta', # 'abramson_delta' | 'constant_kg_s' | 'velocity_linear' | 'delta_at_omega'
    'empirical_fluid_delta_add_s':      4.0,   # extra δ [s⁻¹] in Abramson Fig. 5 gap → C_ii=2δM_ii
    'empirical_fluid_C_kg_s':           15.0,  # diagonal [kg/s] for model='constant_kg_s'
    'empirical_fluid_C_per_ms':         0.0,   # add C_ii += value * V [kg/s per m/s]
    'empirical_fluid_omega_ref_rad_s':  70.0,  # for model='delta_at_omega'
    
    # ── P-K solver — iteration & mode-matching parameters ────────────────────
    'pk_tol':                           1e-2, # Tolerance for PK iteration convergence
    'pk_fXK0':                          0.618, # Factor for XK0 update in PK iteration
    'pk_fRLX':                          0.8, # Factor for RLX update in PK iteration
    'pk_freq_margin':                   0.05, # Frequency margin for PK iteration
    'pk_perturb_k':                     1, # Perturbation factor for k in PK iteration
    'pk_max_iter':                      150, # Maximum number of iterations for PK iteration
    'pk_w_freq':                        0.8, # Weight for frequency in PK iteration
    'pk_w_mac':                         0.6, # Weight for MAC in PK iteration
    'pk_skip_inter_mode_on_perfect_mac': True, # Skip inter-mode frequency separation when MAC pre-selector = 1.0 (crossed modes)
    'pk_predict_and_select':              True, # Yuan & Zhang (2023): linear p extrapolation reorder between airspeed steps
    'pk_min_root_sep_rad':                1.0, # Min |Δω| [rad/s] between modes at same V (allows crossing, blocks coalescence)

    # ── Roger RFA — MAC mode-tracking parameters ──────────────────────────────
    'rfa_freq_margin':                  0.20,
    'rfa_w_freq':                       0.8,
    'rfa_w_mac':                        0.6,

    # ── Rayleigh damping ──────────────────────────────────────────────────────
    'rayleigh_mode_ids':                (0, 1),
    'rayleigh_target_zetas':            (0.0, 0.0),

    # ── Circulatory components ──────────────────────────────────────────────────
    'added_mass_strip_theory':          False,
    'resting_fluid_analysis':           False,

    # ── Velocity sweep (fallback — most cases override this) ─────────────────
    'V_list': np.concatenate([
        np.linspace(5,  12,  2),
        np.linspace(12, 20,  8),
        np.linspace(20, 45, 25),
    ]),

    # ── Aerodynamic panel cards ───────────────────────────────────────────────
    'caero_card_dir': os.path.join(FSI_path, 'FLUID', 'PanelAero', 'panelaero_utl', 'CAERO1_cards'),

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

        # Auto-derive modes lists from num_modes_flutter_egv unless explicitly overridden
        if 'modes_to_analyze' not in kwargs:
            self.modes_to_analyze = list(range(self.num_modes_flutter_egv))
        if 'modes_to_visualize' not in kwargs:
            self.modes_to_visualize = list(range(self.num_modes_flutter_egv))


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
