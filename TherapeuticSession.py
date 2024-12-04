import numpy as np


class TherapeuticSession:
    def __init__(self, avg_temp, session_duration, target_temp=43, therapeutic_range=(41, 45), limit_temp=50, k=0.25):
        """
        Initialize the therapeutic session assessment.

        Parameters:
            avg_temp (float): Average tissue temperature [°C].
            session_duration (float): Duration of the session [minutes].
            target_temp (float): Ideal therapeutic temperature [°C].
            therapeutic_range (tuple): Range of temperatures considered therapeutic (e.g., 41°C to 45°C).
            limit_temp (float): Maximum allowed temperature to prevent damage [°C].
            k (float): Arrhenius constant for CEM43 calculation.
        """
        self.avg_temp = avg_temp
        self.session_duration = session_duration
        self.target_temp = target_temp
        self.therapeutic_range = therapeutic_range
        self.limit_temp = limit_temp
        self.k = k

    def calculate_cem43(self):
        """
        Calculate the cumulative equivalent minutes at 43°C (CEM43) based on average temperature.

        Returns:
            float: CEM43 value.
        """
        # Calculate the thermal dose using CEM43 formula
        return self.session_duration * np.exp(self.k * (self.avg_temp - self.target_temp))

    def assess_session(self):
        """
        Assess the therapeutic session based on average temperature, CEM43, and limits.

        Returns:
            dict: A dictionary summarizing the session assessment.
        """
        # Calculate CEM43 (Cumulative Equivalent Minutes at 43°C)
        cem43 = self.calculate_cem43()

        # Check if the session was in the effective therapeutic range
        if self.therapeutic_range[0] <= self.avg_temp <= self.therapeutic_range[1]:
            effective = True
        else:
            effective = False

        # Calculate percentage effectiveness based on time spent in therapeutic range
        # Here, we're assuming effectiveness is proportional to the time spent in the therapeutic range
        if effective:
            percentage_effectiveness = 100
        else:
            # If the average temperature is outside the therapeutic range, reduce effectiveness based on how far it's off
            if self.avg_temp < self.therapeutic_range[0]:
                percentage_effectiveness = max(0, (self.avg_temp - (self.therapeutic_range[0] - 5)) / 5) * 100
            elif self.avg_temp > self.therapeutic_range[1]:
                percentage_effectiveness = max(0, ((self.therapeutic_range[1] + 5) - self.avg_temp) / 5) * 100
            else:
                percentage_effectiveness = 0

        # Safety check: The temperature should be below the limit to avoid tissue damage
        safe = self.avg_temp <= self.limit_temp

        # Session effectiveness: Consider effective if percentage effectiveness is high enough (e.g., > 80%)
        is_effective = effective and percentage_effectiveness >= 80

        # Returning assessment results
        return {
            "average_temperature": round(self.avg_temp, 2),
            "session_duration_minutes": self.session_duration,
            "CEM43": round(cem43, 2),
            "is_safe": safe,
            "is_effective": is_effective,
            "percentage_effectiveness": round(percentage_effectiveness, 2),
            "recommendation": self.generate_recommendation(safe, is_effective)
        }

    def generate_recommendation(self, safe, is_effective):
        """
        Provide recommendations based on safety and effectiveness.

        Returns:
            str: Recommendation message.
        """
        if not safe:
            return "Reduce heat application to prevent tissue damage."
        elif not is_effective:
            return "Adjust heat source to stay within the effective therapeutic range."
        return "Session is safe and effective."

    def __call__(self):
        """
        Make the class callable to directly assess the session.

        Returns:
            dict: Assessment results.
        """
        return self.assess_session()
