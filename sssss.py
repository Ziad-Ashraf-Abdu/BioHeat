import numpy as np
import pandas as pd

# Function to generate realistic dummy data for bioheat simulation
def generate_realistic_dummy_data(num_samples):
    data = []
    for i in range(num_samples):
        sample = {
            'ID': i + 1,
            'rho': np.random.uniform(980, 1100),    # Density (kg/m^3) for biological tissues
            'c': np.random.uniform(3500, 4000),      # Specific heat capacity (J/kg·K) for muscle tissue
            'k': np.random.uniform(0.4, 0.6),         # Thermal conductivity (W/m·K) for muscle
            'rho_b': 1060,                            # Blood density (kg/m^3) (fixed for blood)
            'c_b': 3600,                             # Blood specific heat capacity (J/kg·K) (fixed for blood)
            'omega_b': np.random.uniform(0.002, 0.005), # Perfusion rate (m/s)
            'T_b': np.random.uniform(36.5, 38.5),    # Blood temperature (°C)
            'Q_m': np.random.uniform(250, 600),       # Metabolic heat production (W/m^3)
            'T0': 37,                                  # Initial temperature (°C)
            'L': np.random.uniform(0.05, 0.2),         # Length of tissue (m), realistic range
            't_end': 300,                              # Total simulation time (s)
            'dt': 1,                                   # Time step (s)
            'heat_type': np.random.choice(['sinusoid', 'constant']), # Type of heat input
            'amplitude': np.random.uniform(0.01, 0.1), # Amplitude of heat input (realistic range)
            'frequency': np.random.randint(30, 120),  # Frequency of heat input (s)
            'alpha': np.random.uniform(0.8, 0.99)      # Fractional order for sub-diffusion
        }
        data.append(sample)
    return data

# Generate realistic dummy data
num_samples = 10  # Specify the number of samples you want
realistic_dummy_data = generate_realistic_dummy_data(num_samples)

# Convert the dummy data to a DataFrame
df = pd.DataFrame(realistic_dummy_data)

# Save the DataFrame to an Excel file
output_file = 'realistic_dummy_bioheat_data.xlsx'
df.to_excel(output_file, index=False)
print(f"Realistic dummy data saved to {output_file}")
