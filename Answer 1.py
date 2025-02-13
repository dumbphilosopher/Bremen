import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 1.0
A = 0.1
k = 50.0 / L**2
gamma = 7.28e-5  # Surface tension (N/mm)

# Discretize the domain
num_points = 201  # Increased for better resolution
x = np.linspace(0, L, num_points)
u = x - L / 2

# Calculate f(x), f'(x), and f''(x)
f = A * np.exp(-k * u**2)
f_prime = -2 * k * u * f
f_double_prime = 2 * k * f * (2 * k * u**2 - 1)

# Calculate curvature
kappa = f_double_prime / (1 + f_prime**2)**(3/2)

# Calculate pressure difference
delta_P = gamma * kappa

# Calculate the normal vector
normal_x = -f_prime / np.sqrt(1 + f_prime**2)
normal_y = 1 / np.sqrt(1 + f_prime**2)

# Calculate the force components (per unit area)
force_x = delta_P * normal_x
force_y = delta_P * normal_y

#Print force at a couple of points to inspect
print(f"Force at x=0: ({force_x[0]:.4e}, {force_y[0]:.4e})")
print(f"Force at x=L/2: ({force_x[num_points//2]:.4e}, {force_y[num_points//2]:.4e})")
print(f"Force at x=L: ({force_x[-1]:.4e}, {force_y[-1]:.4e})")

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(x, f)
plt.title("Interface Shape (f(x))")
plt.xlabel("x")
plt.ylabel("f(x)")

plt.subplot(2, 2, 2)
plt.plot(x, kappa)
plt.title("Curvature (κ(x))")
plt.xlabel("x")
plt.ylabel("κ(x)")

plt.subplot(2, 2, 3)
plt.plot(x, delta_P)
plt.title("Pressure Difference (ΔP(x))")
plt.xlabel("x")
plt.ylabel("ΔP(x)")

plt.subplot(2, 2, 4)
plt.quiver(x, f, force_x, force_y, scale=5)  # Use quiver for vector field
plt.title("Force per Unit Area (F(x))")
plt.xlabel("x")
plt.ylabel("y")
#Adjust y scale:
plt.ylim(-0.1*A, 1.1 * A)


plt.tight_layout()
plt.show()