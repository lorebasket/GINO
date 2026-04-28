# FSI/Hydroelastic_analysis_workflow/aerodynamic_model.py

import numpy as np
from pathlib import Path


def _load_aerogrid_from_npz(npz_path):
    """Load aerogrid from PanelAero .npz file."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Aerogrid file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as d:
        aerogrid = {k: d[k] for k in d.files}

    return aerogrid


def build(aerogrid_file=None, aero_source='panelaero'):
    """
    Build or load aerodynamic grid.
    
    Parameters
    ----------
    aerogrid_file : str or Path, optional
        Path to aerogrid file (.npz for PanelAero, directory for SHARPy)
    aero_source : str
        Aerodynamic source: 'panelaero' or 'sharpy'
        
    Returns
    -------
    aerogrid : dict
        Aerogrid dictionary
    sharpy_data : object or None
        SHARPy data structure (only if aero_source='sharpy')
    """
    print(f"\n--- Loading Aerodynamic Grid ---")
    print(f"  Source: {aero_source}")
    if aerogrid_file:
        print(f"  File: {aerogrid_file}")

    # PanelAero workflow (original)
    if aero_source.lower() == 'panelaero':
        if aerogrid_file is not None:
            aerogrid_path = Path(aerogrid_file)
            if aerogrid_path.exists():
                print(f"Loading PanelAero aerogrid from {aerogrid_path}")
                try:
                    aerogrid = _load_aerogrid_from_npz(aerogrid_path)
                    print(f"  Aerogrid loaded: {aerogrid['n']} panels")
                    return aerogrid, None
                except Exception as e:
                    print(f"Failed to load aerogrid from {aerogrid_path}: {e}")
                    raise
            else:
                raise FileNotFoundError(f"Aerogrid file does not exist: {aerogrid_path}")
        else:
            raise ValueError("No aerogrid file provided for PanelAero source.")

    
    else:
        raise ValueError(f"Unknown aero_source: {aero_source}. Use 'panelaero' or 'sharpy'.")
