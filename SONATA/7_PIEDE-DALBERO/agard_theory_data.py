import numpy as np

# definition of measured sectional properties defined
# within NASA documents of AGARD 445.6  

semispan_ft = 2.5
semispan_m  = semispan_ft * 0.3048          # 0.762 m

length = np.sqrt(2 * semispan_m**2)         # 1.078 m
pitch  = 45

eta_span = semispan_m

# Flexural stiffness
EI_lb_in2 = np.matrix([[0.165, 18.5],
                      [0.21, 15.0],
                      [0.31, 14.5],
                      [0.40, 11.0],
                      [0.49, 9.9],
                      [0.58, 9.0],
                      [0.67, 9.0],
                      [0.75, 7.05],
                      [0.85, 4.2]])

#EI_Nm2 = EI_lb_in2[:,1] * (1e4) * (4.4482216152605 / (0.0254**2))

# Torsional stiffness
GJ_lb_in2 = np.matrix([[0.12, 155],
                      [0.21, 90],
                      [0.31, 65],
                      [0.4, 44],
                      [0.49, 43],
                      [0.58, 37],
                      [0.67, 35],
                      [0.75, 30],
                      [0.85, 28]])

# 1 lb = 4.4482216152605 N
# 1 in² = 0.0254² m²
LB_IN2_TO_NM2 = 4.4482216152605 * (0.0254**2)  # ≈ 2.873e-3

EI_Nm2 = np.asarray(EI_lb_in2[:,1]).astype(float) * 1e4 * LB_IN2_TO_NM2
GJ_Nm2 = np.asarray(GJ_lb_in2[:,1]).astype(float) * 1e4 * LB_IN2_TO_NM2