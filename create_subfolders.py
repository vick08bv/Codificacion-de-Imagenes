import os
import shutil

# Ruta de la carpeta donde están las imágenes procesadas
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, "images_processed")

# Lectura de la lista de categorías
mineral_list = []
with open("list_selected_classes (full_names).txt") as file:
    lines = file.readlines()
    for line in lines:
        mineral_list.append(line.strip().split(",")[0])

categories = len(mineral_list)
mineral_list = sorted(mineral_list)
mineral_dict = {mineral_list[i]: i for i in range(categories)}
mineral_folder = [mineral_list[i] + "_" + str(i) for i in range(categories)]

# Crea la estructura de subdirectorios
for i in range(categories):
    # Crea el nombre de a subcarpeta
    new_folder_name = mineral_folder[i]
    new_folder_path = os.path.join(input_dir, new_folder_name)

    # Crea la nueva carpeta si no existe
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"Creada la carpeta: {new_folder_path}")

# Mover las imágenes que están directamente en images_processed
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    if os.path.isfile(img_path):
        label = img_name.split("_")[0]
        folder_name = mineral_folder[mineral_dict[label]]
        # Mueve la imagen a la carpeta correspondiente
        dst_path = os.path.join(input_dir, folder_name, img_name)
        shutil.move(img_path, dst_path)
