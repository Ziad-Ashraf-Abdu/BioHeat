from Bioheat import BioheatSimulation
from TherapeuticSession import TherapeuticSession

if __name__ == "__main__":
    # Step 1: Run the Bioheat Simulation
    simulation = BioheatSimulation(method="FDM")
    avg_temp = simulation()  # Callable instance
    print(f"Average Temperature (FDM): {avg_temp:.2f} °C")
# Example usage
session = TherapeuticSession(
    avg_temp=avg_temp,
    session_duration=60,
    target_temp=43,
    therapeutic_range=(41, 45),
    limit_temp=50,
    k=0.25)
assessment = session()
print("\nTherapeutic Session Assessment:")
for key, value in assessment.items():
    print(f"{key}: {value}")
