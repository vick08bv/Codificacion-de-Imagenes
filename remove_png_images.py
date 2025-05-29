import os
import shutil

# Directorio de imágenes y carpeta destino
source_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(source_dir, "images")
destination_dir = os.path.dirname(os.path.abspath(__file__))
destination_dir = os.path.join(destination_dir, "images_png")

# Crear la carpeta de destino si no existe
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Extensiones a mover
extensions = (".png", ".mpo", ".bmp")

# Leer y mover archivos
for filename in os.listdir(source_dir):
    if filename.lower().endswith(extensions):
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)

        # Mover archivo
        shutil.move(source_path, destination_path)
        print(f"Movido: {filename}")
