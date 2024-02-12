import math

cents = 70.3600736895
reference_frequency = 440  # A440 reference frequency

pitch_frequency = reference_frequency * (2 ** (cents / 1200))
print("Pitch frequency in Hz:", pitch_frequency)