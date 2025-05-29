import os
import cv2
import csv
from time import time
from compression import huffman_encode, bpp_huffman, entropy_huffman

# Rutas de las imágenes y los archivos de destino
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, "images_processed")
metrics_dir = os.path.join(current_dir, "compression_metrics")
os.makedirs(metrics_dir, exist_ok=True)

# Clases seleccionadas
selected_indices = [1, 3, 5, 7, 9]
all_classes = sorted(os.listdir(images_dir))
selected_classes = [all_classes[i] for i in selected_indices]

for cls in selected_classes:

    class_path = os.path.join(images_dir, cls)
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpeg",))]

    csv_path = os.path.join(metrics_dir, f"Huffman_{cls}.csv")

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_name", "number_symbols", "average_length", "variance_length", "bits_per_pixel", "entropy", "compression_time"])

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_flattened = image.flatten()

            start = time()
            encoded_bits, frequencies, codes = huffman_encode(image_flattened)
            end = time()
            compression_time = end - start

            # Métricas relacionadas a la estructura del árbol de Huffman
            number_symbols = len(frequencies)
            total_symbols = sum(frequencies.values())
            average_length = sum(len(codes[sym]) * freq for sym, freq in frequencies.items()) / total_symbols
            variance_length = sum((len(codes[sym]) - average_length) ** 2 * freq for sym, freq in frequencies.items()) / total_symbols

            # Bits por pixel
            bpp = bpp_huffman(encoded_bits, image_flattened)
            # Entropía
            entropy = entropy_huffman(frequencies)

            writer.writerow([
                image_file,
                number_symbols,
                f"{average_length:.4f}",
                f"{variance_length:.4f}",
                f"{bpp:.4f}",
                f"{entropy:.4f}",
                f"{compression_time:.6f}"
            ])
