import numpy as np

def compute_polar_moment_of_inertia(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    # Ensure the polygon is closed
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # Area using the shoelace formula
    A = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])

    # Centroid (Cx, Cy)
    Cx = (1/(6*A)) * np.sum((x[:-1] + x[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1]))
    Cy = (1/(6*A)) * np.sum((y[:-1] + y[1:]) * (x[:-1]*y[1:] - x[1:]*y[:-1]))

    # Shift coordinates to centroid
    x_c = x - Cx
    y_c = y - Cy

    # Ix and Iy using the polygon formula
    Ix = (1/12) * np.sum((y_c[:-1]**2 + y_c[:-1]*y_c[1:] + y_c[1:]**2) * 
                         (x_c[:-1]*y_c[1:] - x_c[1:]*y_c[:-1]))
    Iy = (1/12) * np.sum((x_c[:-1]**2 + x_c[:-1]*x_c[1:] + x_c[1:]**2) * 
                         (x_c[:-1]*y_c[1:] - x_c[1:]*y_c[:-1]))

    # Polar moment of inertia about the centroid
    J = abs(Ix + Iy)
    return J

# NACA0015 airfoil coordinates
x_coords = [1.0, 1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.0125, 0.0, 0.0125, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.0]
y_coords = [0.0, 0.00158, 0.01008, 0.0181, 0.03279, 0.0458, 0.05704, 0.06617, 0.07254, 0.07502, 0.07427, 0.07172, 0.06682, 0.05853, 0.0525, 0.04443, 0.03268, 0.02367, 0.0, -0.02367, -0.03268, -0.04443, -0.0525, -0.05853, -0.06682, -0.07172, -0.07427, -0.07502, -0.07254, -0.06617, -0.05704, -0.0458, -0.03279, -0.0181, -0.01008, -0.00158, 0.0]

# Compute J
J = compute_polar_moment_of_inertia(x_coords, y_coords)
print(f"Polar moment of inertia of NACA0015 (unit chord): J = {J:.6e} m⁴")
