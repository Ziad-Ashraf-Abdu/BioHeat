import numpy as np
import matplotlib.pyplot as plt


class BioheatSimulation:
    def __init__(self, method="FDM", Nx=60, Nt=100):
        """
        Initialize the Bioheat Simulation parameters.

        Parameters:
            method (str): Choose between "FDM" or "FVM".
            Nx (int): Number of spatial nodes.
            Nt (int): Number of time steps.
        """
        self.method = method
        self.Nx = Nx
        self.Nt = Nt
        self.avg_temp = None

    def run_simulation(self):
        """
        Run the bioheat simulation and compute the average temperature.

        Returns:
            float: The average temperature at the final step.
        """
        # Constants and parameters
        k_base = 0.4  # W/m·K
        c_base = 2980  # J/kg·K
        rho = 985  # kg/m³
        w_b_base = 0.02  # 1/s
        rho_b = 1060  # kg/m³
        c_b = 3610  # J/kg·K
        T_a = 38  # °C
        Q_m = 368.1  # W/m³
        L = 0.025  # m
        A = 0.4  # m²
        T_right = 37  # °C
        h_conv = 3.3  # W/m²·K
        T_inf = 37  # °C
        dx = L / (self.Nx - 1)
        dt = 0.01  # s

        # Gaussian external heat source parameters
        total_power = 500  # W #Adjustable
        V = L * A  # m³
        Q_max = total_power / V  # W/m³
        x0 = L / 2  # Center of the Gaussian source
        sigma = 0.002  # m

        # Spatial domain
        x = np.linspace(0, L, self.Nx)

        # Temperature-dependent properties
        def k_temp(T):
            return k_base * (1 + 0.01 * (T - T_a))

        def c_temp(T):
            return c_base * (1 + 0.01 * (T - T_a))

        def wb_temp(T):
            return w_b_base * (1 + 0.01 * (T - T_a))

        # Define Gaussian external heat source
        def Q_ext(x_pos):
            normalization_factor = total_power / (np.sqrt(2 * np.pi) * sigma * A)
            return normalization_factor * np.exp(-((x_pos - x0) ** 2) / (2 * sigma ** 2))

        # Stability check
        alpha = k_base / (rho * c_base)
        stability_limit = dx ** 2 / (2 * alpha)
        if dt > stability_limit:
            raise ValueError(f"Time step size {dt} exceeds stability limit {stability_limit:.5e}")

        # Initialize temperature profiles for FDM and FVM
        T_FDM = np.ones(self.Nx) * T_a
        T_FVM = np.ones(self.Nx) * T_a

        if self.method == "FDM":
            for n in range(self.Nt):
                T_new_FDM = T_FDM.copy()
                for i in range(1, self.Nx - 1):
                    c_local = c_temp(T_FDM[i])
                    k_left = k_temp(T_FDM[i - 1])
                    k_right = k_temp(T_FDM[i + 1])
                    wb_local = wb_temp(T_FDM[i])

                    flux_left = k_left * (T_FDM[i - 1] - T_FDM[i]) / dx
                    flux_right = k_right * (T_FDM[i + 1] - T_FDM[i]) / dx
                    perfusion = wb_local * rho_b * c_b * (T_a - T_FDM[i])
                    heat_source = Q_m + Q_ext(x[i])

                    T_new_FDM[i] += dt / (rho * c_local) * (flux_right - flux_left + perfusion + heat_source)

                # Boundary conditions
                T_new_FDM[0] += dt * (h_conv * (T_inf - T_FDM[0]) + Q_ext(x[0])) / (rho * c_temp(T_FDM[0]) * dx)
                T_new_FDM[-1] = T_right
                T_FDM = T_new_FDM

            self.avg_temp = np.mean(T_FDM)
            print(self.avg_temp)

        elif self.method == "FVM":
            for n in range(self.Nt):
                T_new_FVM = T_FVM.copy()
                for i in range(1, self.Nx - 1):
                    c_local = c_temp(T_FVM[i])
                    k_left = k_temp(T_FVM[i - 1])
                    k_right = k_temp(T_FVM[i + 1])
                    wb_local = wb_temp(T_FVM[i])

                    flux_left = (k_left + k_temp(T_FVM[i])) / 2 * (T_FVM[i] - T_FVM[i - 1]) / dx
                    flux_right = (k_right + k_temp(T_FVM[i])) / 2 * (T_FVM[i + 1] - T_FVM[i]) / dx
                    perfusion = wb_local * rho_b * c_b * (T_a - T_FVM[i])
                    heat_source = Q_m + Q_ext(x[i])

                    T_new_FVM[i] += dt / (rho * c_local) * (flux_right - flux_left + perfusion + heat_source)

                # Boundary conditions
                T_new_FVM[0] += dt * (h_conv * (T_inf - T_FVM[0]) + Q_ext(x[0])) / (rho * c_temp(T_FVM[0]) * dx)
                T_new_FVM[-1] = T_right
                T_FVM = T_new_FVM

            self.avg_temp = np.mean(T_FVM)

        else:
            raise ValueError("Invalid method. Choose 'FDM' or 'FVM'.")

        return self.avg_temp

    def __call__(self):
        """
        Make the class callable to directly run the simulation and return the average temperature.

        Returns:
            float: Average temperature at the final step.
        """
        return self.run_simulation()
