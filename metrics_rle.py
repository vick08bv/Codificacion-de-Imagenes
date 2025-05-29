import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Clases utilizadas
metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compression_metrics")
selected_indices = [1, 3, 5, 7, 9]
all_classes = sorted([f.replace(".csv", "") for f in os.listdir(metrics_dir) if f.startswith("Rle_")])
selected_classes = [all_classes[i] for i in selected_indices]

# Métricas utilizadas en la compresión RLE
metrics_names = ["total_runs", "average_length", "variance_length", "bits_per_pixel", "entropy", "compression_time"]
class_metrics = {name: [] for name in metrics_names}

for cls in selected_classes:
    path = os.path.join(metrics_dir, f"{cls}.csv")
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

        # Promedios
        for metric in metrics_names:
            values = [float(row[metric]) for row in rows]
            avg = np.mean(values)
            class_metrics[metric].append(avg)

# Visualización
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Comparación de métricas promedio por clase (RLE)", fontsize=16)

for idx, metric in enumerate(metrics_names):
    ax = axes[idx // 3, idx % 3]
    ax.bar(selected_classes, class_metrics[metric], color='skyblue')
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("Clase")
    ax.set_ylabel("Promedio")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_rle.png")
plt.savefig(output_path, dpi=300)
plt.close()
