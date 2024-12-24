import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
k_base = 0.4  # Base thermal conductivity [W/m·K]
c_base = 2980  # Base specific heat capacity [J/kg·K]
rho = 985  # Tissue density [kg/m³]
w_b_base = 0.02  # Base blood perfusion rate [1/s]
rho_b = 1060  # Blood density [kg/m³]
c_b = 3610  # Specific heat capacity of blood [J/kg·K]
T_a = 38  # Arterial blood temperature [°C]
Q_m = 368.1  # Metabolic heat generation [W/m³]
L = 0.025  # Depth of the tissue domain [m]
Nx = 60  # Number of spatial nodes
Nt = 1000  # Number of time steps
dx = L / (Nx - 1)  # Spatial step size
dt = 0.01  # Time step size
sigma = 0.002  # Gaussian source spread [m]
Q_ext_peak = 300 / (L * 0.4)  # Peak external heat source [W/m³]
x_grid = np.linspace(0, L, Nx)

# Gaussian heat source
def gaussian_heat_source(x, x0, sigma):
    return 0 #Q_ext_peak * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

# Temperature-dependent properties
def k(T):
    g = 0.025
    return k_base * (1 + g * (T - T_a))

def c(T):
    b = 0.01
    return c_base * (1 + b * (T - T_a))

def w_b(T):
    e = 0.01
    return w_b_base * (1 + e * (T - T_a))

# Initial conditions
T = np.ones(Nx) * T_a
T[-1] = 37  # Right boundary Dirichlet condition

# FDM with Newton Linearization
def solve_fdm():
    T_fdm = T.copy()
    for n in range(Nt):
        T_new = T_fdm.copy()
        for _ in range(100):  # Newton iteration
            T_old = T_new.copy()
            residual = np.zeros_like(T_fdm)
            jacobian = np.zeros((Nx, Nx))
            for i in range(1, Nx - 1):
                k1 = (k(T_new[i + 1]) + k(T_new[i])) / 2
                k2 = (k(T_new[i]) + k(T_new[i - 1])) / 2
                perfusion = w_b(T_new[i]) * rho_b * c_b * (T_new[i] - T_a)
                Q_ext = gaussian_heat_source(x_grid[i], L / 2, sigma)
                c_i = c(T_new[i])

                # Residual
                residual[i] = (
                    rho * c_i * (T_new[i] - T_fdm[i]) / dt
                    - (k1 * (T_new[i + 1] - T_new[i]) / dx - k2 * (T_new[i] - T_new[i - 1]) / dx) / dx
                    + perfusion - Q_m - Q_ext
                )

                # Jacobian
                jacobian[i, i - 1] = -k2 / dx**2
                jacobian[i, i] = rho * c_i / dt + (k1 + k2) / dx**2 + rho_b * c_b * w_b_base
                jacobian[i, i + 1] = -k1 / dx**2

            # Solve linear system
            delta_T = np.linalg.solve(jacobian[1:-1, 1:-1], -residual[1:-1])
            T_new[1:-1] += delta_T

            # Convergence check
            if np.linalg.norm(delta_T, ord=np.inf) < 1e-6:
                break
        T_fdm = T_new.copy()
    average_heat_fdm = np.mean(T_fdm)  # Average temperature
    print(f"Average heat (FDM): {average_heat_fdm:.2f} °C")
    return T_fdm

# FVM with Picard Linearization
def solve_fvm():
    T_fvm = T.copy()
    for n in range(Nt):
        T_new = T_fvm.copy()
        for _ in range(100):  # Picard iteration
            T_old = T_new.copy()
            for i in range(1, Nx - 1):
                k_w = k((T_fvm[i - 1] + T_fvm[i]) / 2)
                k_e = k((T_fvm[i] + T_fvm[i + 1]) / 2)
                perfusion = w_b(T_fvm[i]) * rho_b * c_b * (T_fvm[i] - T_a)
                Q_ext = 0 #gaussian_heat_source(x_grid[i], L / 2, sigma)
                c_i = c(T_fvm[i])

                # Discretization
                T_new[i] = (
                    T_fvm[i]
                    + dt / (rho * c_i * dx**2) * (k_w * (T_fvm[i - 1] - T_fvm[i]) + k_e * (T_fvm[i + 1] - T_fvm[i]))
                    - dt * perfusion / (rho * c_i)
                    + dt * (Q_m + Q_ext) / (rho * c_i)
                )
            if np.linalg.norm(T_new - T_old, ord=np.inf) < 1e-6:
                break
        T_fvm = T_new.copy()
    average_heat_fvm = np.mean(T_fvm)  # Average temperature
    print(f"Average heat (FVM): {average_heat_fvm:.2f} °C")
    return T_fvm

# Solve with both methods
T_fdm = solve_fdm()
T_fvm = solve_fvm()

# Plot results
plt.plot(x_grid, T_fdm, label="FDM Solution")
plt.plot(x_grid, T_fvm, label="FVM Solution", linestyle="dashdot")
plt.xlabel("Distance (m)")
plt.ylabel("Temperature (°C)")
plt.title("Bioheat Equation Solution (FDM vs FVM)")
plt.legend()
plt.grid()
plt.show()
