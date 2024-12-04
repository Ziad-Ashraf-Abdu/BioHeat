import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
k_base = 0.4  # Thermal conductivity [W/m·K]
c_base = 2980.0  # Specific heat capacity [J/kg·K]
rho = 985.0  # Density [kg/m³]
w_b_base = 0.02  # Blood perfusion rate [1/s]
rho_b = 1060.0  # Blood density [kg/m³]
c_b = 3610.0  # Blood specific heat capacity [J/kg·K]
T_a = 38.0  # Arterial blood temperature [°C]
Q_m = 368.1  # Metabolic heat generation [W/m³]
L = 0.025  # Depth of the tissue domain [m]
T_inf = 37.0  # Initial and ambient temperature [°C]
Nx = 60  # Number of spatial nodes
Nt = 1000  # Number of time steps
dx = L / (Nx - 1)  # Spatial step size
dt = 0.005  # Time step size (reduced for stability)

# Gaussian external heat source parameters
total_power = 300.0
x0 = L / 2
sigma = 0.002

# Create spatial grid
x = np.linspace(0, L, Nx)

# Initialize temperature array
temperature = np.ones(Nx) * T_inf


# Define the Gaussian heat source function
def calculate_heat_source(x):
    normalization_factor = total_power / (np.sqrt(2 * np.pi) * sigma)
    return normalization_factor * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


# Time-stepping loop
for step in range(Nt):
    Q_ext = calculate_heat_source(x)

    # Update temperature using finite difference method
    temperature_new = np.copy(temperature)

    for i in range(1, Nx - 1):
        d2T_dx2 = (temperature[i + 1] - 2 * temperature[i] + temperature[i - 1]) / dx ** 2

        blood_perfusion_term = w_b_base * rho_b * c_b * (T_a - temperature[i])
        metabolic_term = Q_m

        # Update temperature with stability checks
        temperature_new[i] += dt * (
                    k_base * d2T_dx2 + Q_ext[i] + metabolic_term + blood_perfusion_term / (rho * c_base))

        # Clamp values to prevent NaN or extreme temperatures
        temperature_new[i] = np.clip(temperature_new[i], T_inf - 10, T_a + 10)

    temperature = temperature_new

    if step % (Nt // 100) == 0:
        avg_temp = temperature.mean()
        print(f"Step {step}, Average Temperature: {avg_temp:.2f} °C")

# Final average temperature at the last time step
avg_temp_final = temperature.mean()
print(f"Final Average Temperature: {avg_temp_final:.2f} °C")

# Plot the final temperature profile across the tissue
plt.figure(figsize=(10, 6))
plt.plot(x, temperature, label=f"Power={total_power} W", color="blue", marker="o", linestyle="--")
plt.xlabel("Distance (m)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Profile with Gaussian Heat Source")
plt.legend()
plt.grid()
plt.show()