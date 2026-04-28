"""
Diagnostic script to analyze mode classification issues.
Shows detailed participation ratios for the first N modes.
"""

import numpy as np
import sys

# Import the fixed mode classifier
from mode_classifier import ModeClassifier, _participation_ratios, DOF_GROUPS


def diagnose_mode_classifier(eigenvectors, eigenvalues, n_nodes, freq_type='lambda', 
                              threshold_dominant=0.50, threshold_secondary=0.10):
    """
    Diagnose mode classification and print detailed participation ratios.
    
    Parameters
    ----------
    eigenvectors : ndarray
        Shape (n_dofs, n_modes)
    eigenvalues : ndarray
        Shape (n_modes,)
    n_nodes : int
        Number of nodes
    freq_type : str
        'lambda', 'omega', or 'hz'
    threshold_dominant : float
        Dominance threshold
    threshold_secondary : float
        Secondary threshold for coupled modes
    """
    
    clf = ModeClassifier(
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        n_nodes=n_nodes,
        freq_type=freq_type,
        threshold_dominant=threshold_dominant,
        threshold_secondary=threshold_secondary
    )
    
    results = clf.classify()
    
    print("\n" + "="*90)
    print("  DETAILED MODE PARTICIPATION ANALYSIS")
    print("="*90)
    print(f"Dominance threshold: {threshold_dominant*100:.0f}%")
    print(f"Secondary threshold: {threshold_secondary*100:.0f}%")
    print("="*90)
    
    for r in results:
        mode_num = r['Mode']
        freq = r['f [Hz]']
        label = r['Nature']
        
        print(f"\nMode {mode_num} | {freq:.2f} Hz | Classification: {label}")
        print("-" * 90)
        
        # Get raw participation ratios
        phi = clf.Phi[:, mode_num - 1]
        ratios = _participation_ratios(phi, n_nodes)
        
        # Print bar chart
        max_ratio = max(ratios.values())
        for group_name in ["Axial", "Bending XY", "Bending XZ", "Torsion"]:
            ratio = ratios[group_name]
            pct = ratio * 100
            bar_len = int(ratio * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            
            # Highlight secondary contributions
            marker = ""
            if ratio >= threshold_dominant:
                marker = " ◄─ DOMINANT"
            elif ratio >= threshold_secondary:
                marker = " ◄─ SECONDARY (coupled)"
            
            print(f"  {group_name:15s} │ {bar} │ {pct:5.1f}%{marker}")
    
    print("\n" + "="*90 + "\n")
    
    return clf


if __name__ == "__main__":
    print("This module provides diagnostic functionality for mode classification.")
    print("Import and use diagnose_mode_classifier() with your data.")
