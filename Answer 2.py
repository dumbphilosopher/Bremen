import numpy as np
import matplotlib.pyplot as plt

def interface_shape(x, A=0.1, L=1.0, k=50.0):
    """Calculate the initial interface position"""
    k = k/L**2
    return A * np.exp(-k * (x - L/2)**2)

class InterfaceCalculator:
    def __init__(self, nx, L=1.0):
        self.nx = nx
        self.L = L
        self.x = np.linspace(0, L, nx)
        self.dx = L/(nx-1)
        
    def finite_difference_2nd(self, f):
        """Second-order central finite difference method"""
        f_xx = np.zeros_like(f)
        f_xx[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2])/self.dx**2
        f_xx[0] = (f[2] - 2*f[1] + f[0])/self.dx**2
        f_xx[-1] = (f[-1] - 2*f[-2] + f[-3])/self.dx**2
        return f_xx
    
    def finite_difference_4th(self, f):
        """Fourth-order central finite difference method"""
        f_xx = np.zeros_like(f)
        f_xx[2:-2] = (-f[4:] + 16*f[3:-1] - 30*f[2:-2] + 16*f[1:-3] - f[:-4])/(12*self.dx**2)
        f_xx[0:2] = self.finite_difference_2nd(f)[0:2]
        f_xx[-2:] = self.finite_difference_2nd(f)[-2:]
        return f_xx
    
    def spectral_method(self, f):
        """Spectral method using FFT"""
        k = 2*np.pi*np.fft.fftfreq(self.nx, self.dx)
        f_hat = np.fft.fft(f)
        f_xx_hat = -(k**2)*f_hat
        f_xx = np.real(np.fft.ifft(f_xx_hat))
        return f_xx
    
    def calculate_curvature(self, f, method='fd2'):
        """Calculate curvature using specified method"""
        if method == 'fd2':
            f_xx = self.finite_difference_2nd(f)
        elif method == 'fd4':
            f_xx = self.finite_difference_4th(f)
        elif method == 'spectral':
            f_xx = self.spectral_method(f)
        else:
            raise ValueError("Unknown method")
            
        f_x = np.gradient(f, self.dx)
        return f_xx / (1 + f_x**2)**(3/2)

    def calculate_forces(self, method='spectral'):
        """Calculate forces using specified discretization method"""
        sigma = 7.2e-5  # Surface tension
        rho = 9.98e-7    # Density
        g = 9810      # Gravity
        
        f = interface_shape(self.x)
        kappa = self.calculate_curvature(f, method)
        
        F_laplace = sigma * kappa
        
        return F_laplace, f

# Create calculator instance
nx = 1000
calculator = InterfaceCalculator(nx)

# Calculate using different methods
methods = {
    'fd2': 'Second-Order Finite Difference',
    'fd4': 'Fourth-Order Finite Difference',
    'spectral': 'Spectral Method'
}

# Calculate and display maximum differences between methods
print("\nMaximum differences in total force calculations:")
reference_method = 'spectral'  # Using spectral method as reference
_, _, F_ref, _ = calculator.calculate_forces(reference_method)

for method in methods.keys():
    if method != reference_method:
        _, _, F_method, _ = calculator.calculate_forces(method)
        max_diff = np.max(np.abs(F_method - F_ref))
        print(f"{methods[method]} vs {methods[reference_method]}: {max_diff:.5e} N/mm²")
        # Find the maximum value of F_method for *this* method
        current_max_F = np.max(F_method)
        print(f"Maximum F for {methods[method]}: {current_max_F:.5e} N/mm²")
        current_max_ref = np.max(F_ref)
        print(f"Maximum F for {methods[reference_method]}: {current_max_ref:.5e} N/mm²")
        


# Create comparison plots
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(15, 12))

# # Plot 1: Interface Shape
# ax1 = plt.subplot(3, 1, 1)
f = interface_shape(calculator.x)
# ax1.plot(calculator.x, f, 'k-', linewidth=1, label='Interface')
# ax1.set_title('Initial Interface Shape', fontsize=12)
# ax1.set_xlabel('x (mm)')
# ax1.set_ylabel('Height (mm)')
# ax1.grid(True)
# ax1.legend()

# Plot 2: Curvature Comparison
ax2 = plt.subplot(2, 1, 1)
for method, name in methods.items():
    kappa = calculator.calculate_curvature(f, method)
    ax2.plot(calculator.x, kappa, label=name, linewidth=1)
ax2.set_title('Curvature Comparison Between Methods', fontsize=12)
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('Curvature (1/mm)')
ax2.grid(True)
ax2.legend()

# Plot 3: Total Force Comparison
ax3 = plt.subplot(2, 1, 2)
for method, name in methods.items():
    _, _, F_total, _ = calculator.calculate_forces(method)
    ax3.plot(calculator.x, F_total, label=name, linewidth=1)
ax3.set_title('Total Force Comparison Between Methods', fontsize=12)
ax3.set_xlabel('x (mm)')
ax3.set_ylabel('Force per unit area (N/mm²)')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()


        #print(f"{methods[method]} = {F_method} and {methods[reference_method]} = {F_ref}") 
