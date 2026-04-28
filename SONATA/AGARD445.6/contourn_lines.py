import matplotlib.pyplot as plt
import numpy as np

x = [0.0, 0.005, 0.0075, 0.0125, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0]
z = [0.0, 0.00304, 0.00368, 0.00469, 0.00647, 0.00875, 0.01059, 0.01213, 0.01459, 0.01645, 0.01788, 0.01892, 0.01962, 0.01997, 0.01996, 0.01954, 0.01868, 0.01743, 0.01586, 0.01402, 0.01195, 0.00967, 0.00729, 0.00490, 0.00250, 0.00009, 0.0]

plt.figure()
plt.plot(x, z)
plt.plot(x[0], z[0], 'ro')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Contourn lines - LE to TE NACA ')
plt.show()

x_te_le = x[::-1]
z_te_le = z[::-1]


plt.figure()
plt.plot(x_te_le, z_te_le)
plt.plot(x_te_le[0], z_te_le[0], 'ro')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Contourn lines - TE to LE')
plt.show()

z_negative = np.array(z)
z_negative = -z_negative

x_yaml = np.concatenate((x_te_le, x, np.array([x_te_le[0]])))
z_yaml = np.concatenate((z_te_le, z_negative, np.array([z_negative[0]])))

plt.figure()
plt.plot(x_yaml, z_yaml)
plt.plot(x_yaml[0], z_yaml[0], 'ro')
plt.plot(x_yaml[-1], z_yaml[-1], 'go')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Contourn lines - LE to TE NACA ')
plt.show()

# print entries in yaml format
print("x: " + ", ".join(map(str, x_yaml)))
print("y: " + ", ".join(map(str, z_yaml)))