import pandas as pd
import numpy as np

# Load the CSV files
stiff_df = pd.read_csv('m1_anbax_beam_properties_stiff_matrices.csv', header=None, skiprows=1)
mass_df = pd.read_csv('m1_anbax_beam_properties_mass_matrices.csv', header=None, skiprows=1)

# Function to parse the 6x6 matrix from the flattened row
# Assuming the CSV stores the matrix row-by-row or flattened.
# Based on standard formats, usually 36 columns or 21 (symmetric).
# Let's inspect the first row.

def extract_properties(row_stiff, row_mass):
    # Parse Stiffness (Assuming 6x6 flattened or specific columns)
    # If flattened 6x6: S11, S12... S16, S21...
    # We look for the diagonal terms.
    # Note: VABS ordering is typically Extension(1), Shear(2,3), Torsion(4), Bending(5,6)
    
    # Just extracting the raw values to identify magnitudes
    stiff_vals = row_stiff.values.flatten()
    # Filter for non-NaN
    stiff_vals = stiff_vals[~np.isnan(stiff_vals)]
    
    # Identify diagonal terms (Extension, Torsion, Bending) by magnitude
    # Extension (S11) ~ 1e7 - 1e8
    # Bending Soft (Flat) ~ 0.3
    # Bending Stiff (Chord) ~ 3000
    # Torsion (Open section) ~ Low
    # Torsion (Closed section) ~ High
    
    print("--- Stiffness Matrix Diagonals (Estimated) ---")
    # Assuming standard 6x6 flattening, diagonals are indices 0, 7, 14, 21, 28, 35
    if len(stiff_vals) >= 36:
        diags = [stiff_vals[0], stiff_vals[7], stiff_vals[14], stiff_vals[21], stiff_vals[28], stiff_vals[35]]
        print(f"S11 (Ext): {diags[0]:.2e}")
        print(f"S22 (ShearY): {diags[1]:.2e}")
        print(f"S33 (ShearZ): {diags[2]:.2e}")
        print(f"S44 (Torsion): {diags[3]:.2e}")
        print(f"S55 (Bend1): {diags[4]:.2e}")
        print(f"S66 (Bend2): {diags[5]:.2e}")
        
        # Heuristic check
        EI_found = min(diags[4], diags[5])
        print(f"\n>> Detected Soft Bending Stiffness (EI): {EI_found:.4f} N*m^2")
    else:
        print("Could not parse 6x6 matrix standardly. Dumping raw vals:", stiff_vals[:10])
        EI_found = 0

    # Mass Properties
    mass_vals = row_mass.values.flatten()
    mass_vals = mass_vals[~np.isnan(mass_vals)]
    # M11 is usually mass per length mu
    mu_found = mass_vals[0]
    print(f"\n>> Detected Mass per Length (mu): {mu_found:.4f} kg/m")
    
    return EI_found, mu_found

print("Analyzing Section 1 (Root)...")
EI, mu = extract_properties(stiff_df.iloc[0], mass_df.iloc[0])

# Theoretical Calculation
if EI > 0 and mu > 0:
    L = 0.305
    f_theoretical = (3.52 / (2 * np.pi * L**2)) * np.sqrt(EI / mu)
    print(f"\n--- Theoretical Frequency Check ---")
    print(f"Length: {L} m")
    print(f"Calculated f1: {f_theoretical:.2f} Hz")
    
    # Check for 'Box' vs 'Plate'
    # Plate EI ~ 0.32
    # Box (0.8mm gap) EI ~ 5.4
    if 4.0 < EI < 7.0:
        print("\n[DIAGNOSIS]: The stiffness matches a HOLLOW BOX (Gap effect).")
        print("The solver created a tube instead of a collapsed plate.")
    elif 0.2 < EI < 0.5:
        print("\n[DIAGNOSIS]: The stiffness matches a FLAT PLATE.")
        print("The cross-section is correct. The error is in the beam analysis settings.")
    else:
        print(f"\n[DIAGNOSIS]: Stiffness is {EI:.4f}. This is unexpected.")