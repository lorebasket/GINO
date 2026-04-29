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
    'n_elements':                       400,
    'nspan':                            20,
    'nchord':                           10,

    # ── Modal analysis ────────────────────────────────────────────────────────
    'num_modes_egv':                    5, # number of computed modes dry analysis
    'num_modes_flutter_egv':            4,  # number of modes used in flutter analysis

    # ── Fluid properties ──────────────────────────────────────────────────────
    'c_sound':  {'air': 332.5, 'water': 1484.0},   # [m/s]  at ~2000 m
    'rho_f':    {'air': 1.02,  'water': 997},       # [kg/m³]

    # ── Output & saving ───────────────────────────────────────────────────────
    'save_plots':                       True,
    'output_dir':                       os.path.join(FSI_path, 'output_data'),
    'save_global_matrices':             False,
    'save_matrices':                    False,
    'save_modal_data':                  False,

    # ── Visualization ─────────────────────────────────────────────────────────
    'plot_mode_shapes':                 True,
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

    # ── Force computation ─────────────────────────────────────────────────────
    'compute_tip_forces':               False,
    'track_tip_forces_during_flutter':  False,

    # ── Structural physics ────────────────────────────────────────────────────
    'geometric_nonlinearity':           False,
    'include_gravity':                  False,
    'gravity_acc':                      np.array([0, 0, -9.81]),  # [m/s²]
    'constrained_dofs':                 list(range(6)),

    # ── Fluid-at-rest (wet modes) ─────────────────────────────────────────────
    'fluid_at_rest':                    False,
    'modal_capytaine_dofs':             False,

    # ── FLUTTER analysis method ───────────────────────────────────────────────
    'pk_method':                        True,   # P-K method aeroelastic analysis
    'roger_fit':                        False,  # Roger RFA hydroelastic analysis

    # ── P-K method — flags ────────────────────────────────────────────────────
    'mac_matching':                     False,
    'last_converged_mode_matching':     False,
    'k_guess':                          0.001,

    # ── Roger RFA ─────────────────────────────────────────────────────────────
    'k_list':                           np.linspace(0.001, 50, 100),
    'n_lag':                            3,
    'blag':                             np.linspace(0.3, 1, 3),

    # ── P-K solver — iteration & mode-matching parameters ────────────────────
    'pk_tol':                           1e-2,
    'pk_fXK0':                          0.618,
    'pk_fRLX':                          0.6,
    'pk_freq_margin':                   0.05,
    'pk_perturb_k':                     1,
    'pk_max_iter':                      150,
    'pk_w_freq':                        0.8,
    'pk_w_mac':                         0.6,

    # ── Roger RFA — MAC mode-tracking parameters ──────────────────────────────
    'rfa_freq_margin':                  0.15,
    'rfa_w_freq':                       0.8,
    'rfa_w_mac':                        0.6,

    # ── Rayleigh damping ──────────────────────────────────────────────────────
    'rayleigh_mode_ids':                (0, 1),
    'rayleigh_target_zetas':            (0.0, 0.0),

    # ── Added mass ────────────────────────────────────────────────────────────
    # False = aeroelastic analysis, True = hydroelastic analysis
    'added_mass_strip_theory':          False,

    # ── Hybrid non-circulatory operator ──────────────────────────────────────
    'hybrid_nc_operator':               False,
    'capytaine_data_dir':               os.path.join(FSI_path, 'aeroelastic_coupling', 'hydrodynamics', 'capytaine_matrices'),
    'capytaine_multibody':              False,
    'capytaine_singlebody':             False,
    'free_surface_elevation':           0.0,

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
