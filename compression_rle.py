import os
import cv2
import csv
import numpy as np
from time import time
from compression import rle_encode, entropy_rle, bpp_rle

# Rutas de las imágenes y los archivos de destino
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, "images_binaries")
metrics_dir = os.path.join(current_dir, "compression_metrics")
os.makedirs(metrics_dir, exist_ok=True)

# Clases seleccionadas
selected_indices = [1, 3, 5, 7, 9]
all_classes = sorted(os.listdir(images_dir))
selected_classes = [all_classes[i] for i in selected_indices]

for cls in selected_classes:

    class_path = os.path.join(images_dir, cls)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpeg",))]

    csv_path = os.path.join(metrics_dir, f"Rle_{cls}.csv")

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "total_runs", "average_length", "variance_length", "bits_per_pixel", "entropy", "compression_time"])

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_binary = (image // 255).astype(np.uint8)

            start = time()
            rle = rle_encode(image_binary)
            end = time()
            compression_time = end - start

            # Métricas relacionadas a la distribución de runs
            total_runs = len(rle)
            lengths = [length for _, length in rle]
            average_length = np.mean(lengths)
            variance_length = np.var(lengths)

            # Bits por pixel
            bpp = bpp_rle(rle, image_binary.size)
            # Entropía
            entropy = entropy_rle(lengths)

            writer.writerow([
                image_file,
                total_runs,
                f"{average_length:.4f}",
                f"{variance_length:.4f}",
                f"{bpp:.4f}",
                f"{entropy:.4f}",
                f"{compression_time:.6f}"
            ])
