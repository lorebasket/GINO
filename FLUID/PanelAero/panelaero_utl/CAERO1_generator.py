import os

def format_caero1_card(wing, caero_id, pid, igid, nspan, nchord, x1, y1, z1, x4, y4, z4, chord_root, chord_tip):
    """
    Parameters:
    - caero_id, pid, igid: Integer IDs
    - nspan, nchord: Number of spanwise and chordwise elements
    - x1, y1, z1: Leading edge root coordinates
    - x4, y4, z4: Leading edge tip coordinates
    - span, chord: Total span and chord
    """
    line1 = f"CAERO1{caero_id:>10}{pid:>8}{igid:>8}{nspan:>8}{nchord:>8}{'':>23}+"
    # Use proper 8-character fields with adequate precision for small values
    line2 = f"+{x1:>15.3f}{y1:>8.3f}{z1:>8.3f}{chord_root:>8.3f}{x4:>8.3f}{y4:>8.3f}{z4:>8.3f}{chord_tip:>8.3f}"
    
    caero_lines = [line1, line2]

    # Create output filename and ensure its directory exists
    os.makedirs(os.path.dirname(wing), exist_ok=True)

    # Write to file
    with open(wing, "w") as f:
        f.write("$ CAERO1 card for simple wing\n")
        f.write("$ \n")
        f.write("$------><------><------><------><------><------><------><------><------>\n")
        for line in caero_lines:
            f.write(line + "\n") 

    print(f"CAERO1 card saved in {wing}")