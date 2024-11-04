import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd  # Import pandas for handling dataframes

# Function to generate heat input based on type
def heat_input(t, heat_type, amplitude, frequency):
    if heat_type == "sinusoid":
        return amplitude * np.sin(2 * np.pi * t / frequency) if frequency != 0 else 0
    elif heat_type == "constant":
        return amplitude
    else:
        return 0

# Function to calculate Caputo-Fabrizio fractional derivative
def caputo_fabrizio_derivative(T_history, alpha, dt):
    n = len(T_history)
    sum_term = 0.0  # Initialize as a float to ensure scalar results
    for j in range(1, n):
        exp_term = np.exp(-alpha * (n - j) * dt / (1 - alpha))
        sum_term += (T_history[j] - T_history[j - 1]) * exp_term
    return (1 / (1 - alpha)) * sum_term  # Ensure this returns a scalar

# Load realistic parameters from Excel file
df = pd.read_excel('realistic_dummy_bioheat_data.xlsx')

# Prepare to collect overall heat values
overall_heat_data = []

# Loop through each individual and solve the bioheat equation
for index, person in df.iterrows():
    # Extract parameters from DataFrame
    rho = person['rho']
    c = person['c']
    k = person['k']
    rho_b = person['rho_b']
    c_b = person['c_b']
    omega_b = person['omega_b']
    T_b = person['T_b']
    Q_m = person['Q_m']
    T0 = person['T0']
    L = person['L']
    t_end = person['t_end']
    dt = person['dt']
    heat_type = person['heat_type']
    amplitude = person['amplitude']
    frequency = person['frequency']
    alpha = person['alpha']  # Fractional order

    # Simulation parameters
    Nx = 50  # Number of spatial points
    dx = L / Nx  # Spatial step size
    Nt = int(t_end / dt)  # Number of time steps

    # Modified thermal diffusivity with fractional order
    K = k * dt ** (alpha - 1)

    # Boundary conditions
    T_left = T0
    T_right = T0

    # Initial temperature distribution
    T = T0 * np.ones(Nx)
    T_new = np.copy(T)

    # Store temperature history for fractional derivative calculation
    T_history = []

    # Open CSV file for writing temperature distribution
    with open(f'temperature_distribution_person_{index + 1}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (s)', 'Temperature at center', 'Average Temperature'])

        # Time loop
        for n in range(Nt):
            # Update history
            T_history.append(np.copy(T))

            # Apply boundary conditions
            T_new[0] = T_left
            T_new[-1] = T_right

            # Update temperature for internal points
            for i in range(1, Nx - 1):
                # Spatial finite difference
                conduction = K * (T[i + 1] - 2 * T[i] + T[i - 1]) / (dx ** 2)

                # Fractional time derivative term
                if n > 0:
                    fractional_term = caputo_fabrizio_derivative(T_history[:n + 1], alpha, dt)
                    # Ensure fractional_term is a scalar
                    fractional_term = np.sum(fractional_term)  # Use np.sum to get a single scalar value
                else:
                    fractional_term = 0

                # Perfusion, metabolic, and external heat source
                perfusion = rho_b * c_b * omega_b * (T_b - T[i]) / (rho * c)
                metabolic = Q_m / (rho * c)
                heat_ext = heat_input(n * dt, heat_type, amplitude, frequency)

                # Ensure all terms are scalars
                T_new[i] = T[i] + dt * (conduction + perfusion + metabolic + heat_ext + fractional_term)
                T_new[i] = np.clip(T_new[i], 35, 40)

            # Update temperature for next time step
            T[:] = T_new[:]

            # Calculate average temperature and store in overall heat data
            average_temperature = np.mean(T)
            overall_heat_data.append(average_temperature)  # Store average temperature

            # Write to CSV file
            writer.writerow([n * dt, T[Nx // 2], average_temperature])

    # Plot results for each person
    plt.figure()
    plt.plot(np.linspace(0, L, Nx), T, label=f'Time = {t_end}s for Person {index + 1}')
    plt.xlabel('Position along tissue (m)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Distribution for Person {index + 1} with Sub-Diffusion (α={alpha})')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Ask user to navigate or exit
    navigate = input("Press Enter to see the next person's graph or type 'exit' to stop: ")
    if navigate.lower() == 'exit':
        break
