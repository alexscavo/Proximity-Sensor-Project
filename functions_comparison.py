import matplotlib.pyplot as plt
import numpy as np

# Define range of radar point counts
n_radar = np.arange(0, 50)

# === Sigmoid parameters ===
a = 1.5  # controls sharpness
b = 10.0  # midpoint (radar dominance starts here)
sigmoid_weight = 1 / (1 + np.exp(-a * (n_radar - b)))

# === Rational weighting ===
k = 5.0  # radar influence constant
rational_weight = n_radar / (n_radar + k)
rational_weight_1 = n_radar / (n_radar + 10.0)

# === Power law ===
k = 9
p = 1.2  # try 0.5 to 0.8
power_weight = (n_radar / (n_radar + k))**p

# === Plot ===
plt.figure(figsize=(8, 5))
plt.plot(n_radar, sigmoid_weight, label='Sigmoid weighting', linewidth=2)
plt.plot(n_radar, rational_weight, label='Rational weighting (n / (n + 5))', linewidth=2, linestyle='--')
plt.plot(n_radar, rational_weight_1, label='Rational weighting (n / (n + 10))', linewidth=2, linestyle='--')
plt.plot(n_radar, power_weight, label='Power law', linewidth=2, linestyle='-.')

plt.xlabel("Number of radar points")
plt.ylabel("Radar velocity weight α")
plt.title("Radar Contribution Weighting Functions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
