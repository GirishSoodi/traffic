# compare_control_modes.py

import pandas as pd
import matplotlib.pyplot as plt

fixed_file = "traffic_metrics_fixed.xlsx"
rl_file = "traffic_metrics_rl.xlsx"

# Load control metrics
fixed_ctrl = pd.read_excel(fixed_file, sheet_name="Control_Metrics")
rl_ctrl = pd.read_excel(rl_file, sheet_name="Control_Metrics")

fixed_row = fixed_ctrl.iloc[0]
rl_row = rl_ctrl.iloc[0]

comparison = pd.DataFrame({
    "Metric": [
        "Average Waiting Time (sec)",
        "Average Queue Length (vehicles)",
        "Number of Stops per Vehicle",
        "Throughput (vehicles/hour)"
    ],
    "FIXED": [
        fixed_row["Average Waiting Time (sec)"],
        fixed_row["Average Queue Length (vehicles)"],
        fixed_row["Number of Stops per Vehicle"],
        fixed_row["Throughput (vehicles/hour)"],
    ],
    "RL": [
        rl_row["Average Waiting Time (sec)"],
        rl_row["Average Queue Length (vehicles)"],
        rl_row["Number of Stops per Vehicle"],
        rl_row["Throughput (vehicles/hour)"],
    ]
})

print("\n=== RL vs FIXED Comparison ===")
print(comparison)

# Save to Excel
comparison.to_excel("traffic_metrics_comparison.xlsx", index=False)
print("\nComparison saved to traffic_metrics_comparison.xlsx")

# --- Visualization ---
metrics = comparison["Metric"].tolist()
x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar([i - width/2 for i in x], comparison["FIXED"], width, label="FIXED")
plt.bar([i + width/2 for i in x], comparison["RL"], width, label="RL")

plt.xticks(list(x), metrics, rotation=20)
plt.ylabel("Value")
plt.title("Traffic Control Performance: RL vs FIXED")
plt.legend()
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()
