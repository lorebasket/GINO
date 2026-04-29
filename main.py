# /media/lorenzo/Seagate Por/recupero dati Linux/lorebasket/FSI/main.py

import sys
import os
from datetime import datetime

# Add necessary paths to sys.path to ensure all modules can be found
# Use the directory of the current script to make paths relative and portable
FSI_path = os.path.dirname(os.path.abspath(__file__))

sys.path.extend([
    FSI_path,
    os.path.join(FSI_path, 'PanelAero'),           # exposes panelaero_utl as a package
    os.path.join(FSI_path, 'FEA'),
    os.path.join(FSI_path, 'FEA', 'fea_utl'),
])


class Logger:
    """Class to redirect stdout to both terminal and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


from examples import config
from Hydroelastic_analysis_workflow import (
    structural_model,
    structural_analysis,
    aerodynamic_model,
    aero_structural_coupling,
    flutter_solver,
    post_processing,
)
from FEA.fea_utl import force_analysis

from FEA.fea_utl import mode_shape_analysis
from panelaero_utl import plotting_git
import numpy as np


def run_flutter_analysis_single(case_name, alpha_deg, param_overrides=None):
    """
    Run flutter analysis for a single angle of attack.
    
    Args:
        case_name: Name of the case (e.g., 'GOLAND')
        alpha_deg: Single angle of attack in degrees
        param_overrides: Dict of parameter overrides (GJ, EIxx, rayleigh_target_zetas, xea_factor, etc.)
        
    Returns:
        flutter_results: FlutterResults object
        analysis_config: Configuration used for this analysis
    """
    ## == 1. LOAD CONFIGURATION == ##
    analysis_config = config.get_config(case_name)
    
    # Override alpha to be a single value
    analysis_config.alpha_deg = alpha_deg
    analysis_config.alpha_r = (np.deg2rad(alpha_deg),)
    
    # Apply parameter overrides if provided
    if param_overrides:
        for param_name, param_value in param_overrides.items():
            setattr(analysis_config, param_name, param_value)
            print(f"  Override: {param_name} = {param_value}")
    
    # Update paths for this specific alpha
    QJJ_path_local = os.path.join(FSI_path, 'PanelAero', 'Qjj', 'qjj_precomputed')
    fluid = analysis_config.fluid
    nspan = analysis_config.nspan
    nchord = analysis_config.nchord
    xea_factor = analysis_config.xea_factor  # Get potentially overridden value
    
    # Format alpha to match folder naming
    # Keep decimals for 0.0 and 0.05, but use integer for 2
    if isinstance(alpha_deg, float) and alpha_deg == int(alpha_deg) and alpha_deg != 0.0:
        # Integer value but not zero (e.g., 2.0 -> "2")
        alpha_str = str(int(alpha_deg))
    else:
        # Keep as is for 0.0, 0.05, etc.
        alpha_str = str(alpha_deg)
    
    # Check if xea_factor was overridden (for elastic axis sweep)
    # If so, use the _shoff{xea_factor} naming convention
    if param_overrides and 'xea_factor' in param_overrides:
        dlm_method = analysis_config.dlm_method
        base_name = f"{case_name}_{fluid}_alpha{alpha_str}_nspan{nspan}_nchord{nchord}_klist_{dlm_method}_shoff{xea_factor}"
        vjj_base_name = f"vjj_{case_name}_{fluid}_alpha{alpha_str}_nspan{nspan}_nchord{nchord}_klist_{dlm_method}"
    
        analysis_config.qjj_dir = os.path.join(QJJ_path_local, base_name)
        analysis_config.vjj_dir = os.path.join(QJJ_path_local, vjj_base_name)
        analysis_config.aerogrid_path = os.path.join(analysis_config.qjj_dir, "aerogrid.npz")
    
    print(f"\nLoaded configuration for: {analysis_config.name}, α = {alpha_deg}°")
    if param_overrides and 'xea_factor' in param_overrides:
        print(f"  Aerogrid path: .../{base_name}/")
    
    # Check if aerogrid exists
    if not os.path.exists(analysis_config.aerogrid_path):
        print(f"⚠ WARNING: Aerogrid not found at {analysis_config.aerogrid_path}")
        print(f"   Skipping α = {alpha_deg}°")
        return None, None

    ## == 2. Build Structural Model == ##
    beam_model = structural_model.build(analysis_config)
    print("Structural model created.")

    ## == 3. Perform Structural Analysis (Dry Run) == ##
    structural_results = structural_analysis.run_dry_analysis(beam_model, analysis_config)

    #breakpoint()

    ## == 3b. Perform Fluidrest Analysis (if enabled) == ##
    fluidrest_results = None
    if analysis_config.fluid_at_rest:
        fluidrest_results = structural_analysis.run_fluidrest_analysis(structural_results, analysis_config)

    # Compute reaction forces and displacements if enabled in config (for tnz_multibody)
    if analysis_config.compute_tip_forces:
        print("\n" + "="*70)
        print("COMPUTING ARM REACTION FORCES AND DISPLACEMENTS")
        print("="*70)
        
        constrained_dofs = analysis_config.constrained_dofs
        
        # Compute and print reaction forces at root support
        reaction_data = force_analysis.print_reaction_forces(
            structural_results.u_full,
            structural_results.K_global,
            constrained_dofs,
            beam_model,
            config=analysis_config
        )
        
        # Compute and print tip displacement
        tip_data = force_analysis.print_tip_displacement(
            structural_results.u_full,
            beam_model,
            config=analysis_config
        )
        
        # Save displacements along entire span to CSV
        try:
            output_dir = analysis_config.output_dir
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f'{case_name}_displacements_along_span.csv')
            force_analysis.save_displacements_to_csv(
                structural_results.u_full,
                beam_model,
                csv_path
            )

            force_analysis.plot_displacement_distribution(
                structural_results.u_full,
                beam_model,
                output_dir=output_dir
            )

            if analysis_config.plot_displacements_by_beam:
                print("\nPlotting displacements by individual beams...")
                force_analysis.plot_displacements_by_beam(
                    structural_results.u_full,
                    beam_model,
                    output_dir=output_dir
                )
        except Exception as e:
            print(f"Warning: Could not save/plot displacement distribution: {e}")

    #breakpoint()
    if analysis_config.plot_mode_shapes:
        print("\nPlotting mode shapes...")
        mode_shape_analysis.plot_shapes(
            beam_model=beam_model,
            n_elements_or_none=None,
            dry_vectors=structural_results.dry_eigenvectors_full,
            num_modes=len(structural_results.dry_frequencies),
            title_suffix=f"α={alpha_deg}°"
        )

    # 4. Build Aerodynamic Model
    aerogrid, sharpy_data = aerodynamic_model.build(
        analysis_config.aerogrid_path,
        aero_source=analysis_config.aero_source
    )
    
    # Plot the aerogrid
    try:
        plots = plotting_git.DetailedPlots(aerogrid)
        plots.plot_aerogrid(title=f"Aerodynamic Grid - {case_name} α={alpha_deg}°")
    except Exception as e:
        print(f"Warning: Could not plot aerogrid: {e}")

    ## PLOT AEROGRID over BEAM_MODEL
    if analysis_config.plot_aerobeam:
        post_processing.plot_aero_beam_model(aerogrid, beam_model, analysis_config)
    
    # 5. Compute Aero-Structural Coupling
    coupling_results = aero_structural_coupling.calculate_spline_matrix(beam_model, aerogrid, analysis_config)
    
    # PLOT DEFORMED MODE SHAPES ON AEROGRID (like a vibrating plate)
    if analysis_config.plot_deformed_modes:
        post_processing.plot_deformed_mode_shapes(
            aerogrid,
            beam_model,
            structural_results,
            coupling_results,
            modes_to_plot=analysis_config.modes_to_visualize,
            scale_factor=analysis_config.mode_scale_factor,
            config=analysis_config,
            animate=analysis_config.animate_modes,
            n_frames=analysis_config.animation_frames
        )

    # Compute fluid at rest frequencies
    #----------------------------

    # 6. Solve for Flutter
    flutter_results = flutter_solver.solve(
        analysis_config,
        structural_results,
        coupling_results,
        aerogrid=aerogrid
    )
    
    # 8. Compute Tip Forces During Flutter (after convergence at each speed)
    if analysis_config.compute_tip_forces and analysis_config.track_tip_forces_during_flutter:
        print("\n" + "="*70)
        print("COMPUTING TIP FORCES DURING FLUTTER SWEEP")
        print("(Forces include aerodynamic loads at each velocity)")
        print("="*70)
        
        # We'll store forces at selected velocities
        # Note: This requires solving for displacements at each velocity using the converged modal amplitudes
        # For now, we'll compute at a subset of velocities
        try:
            output_dir = analysis_config.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Placeholder: collect forces at convergence points
            # This would need modal displacement recovery from flutter_results
            forces_flutter_file = os.path.join(output_dir, f'{case_name}_tip_forces_flutter_sweep.csv')
            print(f"✓ Tip forces tracking during flutter enabled (output: {forces_flutter_file})")
            print("  Note: Full implementation requires modal displacement recovery from each velocity step")
        except Exception as e:
            print(f"Warning: Could not initialize flutter force tracking: {e}")
    
    # 7. Post-process and Visualize
    post_processing.plot_vg_vf(flutter_results, analysis_config)
    # Eigenvalue trajectory plot only available for PK method (not for Roger RFA)
    if flutter_results.raw_results is not None:
        post_processing.plot_eigenvalue_trajectory(flutter_results, flutter_results.raw_results, analysis_config)
    else:
        print("\nNote: Eigenvalue trajectory plot not available for Roger RFA method (uses direct eigenvalue sweep)")
    # Stiffness/damping contributions only available for PK method
    if flutter_results.pk_solver is not None:
        post_processing.plot_stiffness_damping_contributions(pk_solver=flutter_results.pk_solver, config=analysis_config)
    else:
        print("Note: Stiffness/damping contributions plot not available for Roger RFA method")
    
    print(f"✓ Flutter analysis completed for α = {alpha_deg}°")
    if flutter_results.flutter_speed is not None:
        print(f"  Flutter speed: {flutter_results.flutter_speed:.2f} m/s")
    
    return flutter_results, analysis_config


if __name__ == "__main__":
    import argparse
    import time

    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Run flutter analysis for a given case.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python main.py GOLAND
  python main.py tnz_multibody
        """
    )
    parser.add_argument(
        "case_name",
        nargs="?",
        default="GOLAND",
        help="Analysis case name (e.g. GOLAND, tnz_multibody, grid_conv). Default: GOLAND",
    )

    args = parser.parse_args()
    case_name = args.case_name

    # Setup logging to file
    log_dir = os.path.join(FSI_path, 'output_plots')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f'flutter_analysis_{case_name}_{timestamp}.log')

    logger = Logger(log_filename)
    sys.stdout = logger

    print(f"\n{'='*70}")
    print(f"Running Analysis for: {case_name}")
    print(f"Log file: {log_filename}")
    print(f"{'='*70}\n")

    try:
        # Load config to get the alpha_deg (single value expected for direct run)
        _analysis_config = config.get_config(case_name)
        alpha_deg = _analysis_config.alpha_deg
        if isinstance(alpha_deg, (list, tuple, np.ndarray)):
            alpha_deg = alpha_deg[0]

        run_flutter_analysis_single(case_name, alpha_deg)
    
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\nLog saved to: {log_filename}")


    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")