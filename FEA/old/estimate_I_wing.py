
def estimate_Iy_Iz_thin_wing(chord, thickness, shape_factor=1.0):
    
    b = chord
    h = thickness
    Iy = shape_factor * b * h**3 / 12.0
    Iz = shape_factor * h * b**3 / 12.0
    
    return Iy, Iz