import os
import shutil

# Máximo permitido de ejemplos por categoría
max_examples = 4999

# Directorio de imágenes
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, "images")
unused_dir = os.path.join(current_dir, "images_unused")

# Crear la carpeta de destino si no existe
if not os.path.exists(unused_dir):
    os.makedirs(unused_dir)

# Quita el exceso de imágenes por categoría
for image_name in os.listdir(images_dir):
    name, number = os.path.splitext(image_name)[0].split("_")
    if int(number) > max_examples:
        source_path = os.path.join(images_dir, image_name)
        destination_path = os.path.join(unused_dir, image_name)

        # Mover archivo
        shutil.move(source_path, destination_path)
