import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fenics import *

# Bioheat equation setup and solution using FEniCS
def solve_bioheat_equation():
    # Create a unit square mesh for the domain
    mesh = UnitSquareMesh(50, 50)  # Adjust mesh resolution as needed

    # Define the function space
    V = FunctionSpace(mesh, 'P', 1)  # Linear elements

    # Define boundary conditions (example: constant temperature at the boundaries)
    u_D = Expression('300 + 50*sin(3.14159*x[0])*sin(3.14159*x[1])', degree=2)
    bc = DirichletBC(V, u_D, 'on_boundary')

    # Define the trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define the parameters for the Bioheat equation (simplified model)
    k = 0.5  # thermal conductivity
    rho = 1000  # density
    cp = 3000  # specific heat capacity
    Q = 0.01  # internal heat generation rate

    # Define the weak form of the Bioheat equation
    a = rho*cp*dot(grad(u), grad(v))*dx
    L = Q*v*dx

    # Solve the equation
    u = Function(V)
    solve(a == L, u, bc)

    return u

# Load the results from the CSV file for comparison
def load_csv_data(csv_file):
    # Load the CSV data containing results (e.g., temperature)
    csv_data = pd.read_csv(csv_file)
    return csv_data['Temperature'].values  # Assuming the CSV has a 'Temperature' column

# Compare the COMSOL/FEniCS results with CSV data
def compare_results(fenics_results, csv_file):
    # Load the results from CSV
    csv_data = load_csv_data(csv_file)

    # Interpolate FEniCS results to a numpy array for comparison
    fenics_results_array = fenics_results.vector().get_local()

    # Calculate the difference/error between FEniCS and CSV results
    difference = fenics_results_array - csv_data
    mse = np.mean(np.square(difference))  # Mean squared error for comparison

    # Print out the error metrics
    print(f"Mean Squared Error (MSE): {mse}")

    # Visualize the comparison
    plt.figure(figsize=(10, 5))
    plt.plot(fenics_results_array, label='FEniCS Results', color='blue')
    plt.plot(csv_data, label='CSV Results', color='red', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.legend()
    plt.title('FEniCS Bioheat Equation vs CSV Data')
    plt.show()

# Main function to run the Bioheat equation solution and comparison
def main():
    # Solve the Bioheat equation using FEniCS
    fenics_results = solve_bioheat_equation()

    # Compare the results with those from the CSV file
    csv_file = 'results_comparison.csv'  # Replace with your CSV file path
    compare_results(fenics_results, csv_file)

# Run the main function
if __name__ == "__main__":
    main()
