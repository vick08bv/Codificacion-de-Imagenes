import os
import re
import requests
from PIL import Image
from io import BytesIO

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


# Descarga las imágenes con las etiquetas como nombres
def download_image(image_url, image_label, download_folder, labels_dict):
    try:
        # Solicitud
        response = requests.get(image_url, headers=headers, stream=True)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        # Crea el nombre del archivo según su etiqueta
        img_extension = img.format.lower()
        file_name = f"{image_label}_{labels_dict[image_label]}.{img_extension}"
        file_path = os.path.join(download_folder, file_name)

        # Guarda la imagen en el directorio especificado
        img.save(file_path)
        # Pasa a la siguiente imagen
        labels_dict[image_label] += 1

    except Exception as e:
        print(f"Error descargando la imagen desde {image_url}: {e}")

# Lectura de la lista de clases seleccionadas
labels = dict()
with open("list_selected_classes (full_names).txt") as file:
    lines = file.readlines()
    for line in lines:
        label = line.strip()
        match = re.search(r"\(Var: ([^\)]+)\)", label)
        if match:
            label = match.group(1)
        labels[label] = 0

# Directorio de descarga
current_dir = os.path.dirname(os.path.abspath(__file__))
download_dir = os.path.join(current_dir, "images")

# Crea la carpeta destino
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Guarda las imágenes
with open("filtered_url_list.csv") as file:
    lines = file.readlines()
    for line in lines:
        url, label = line.strip().split(",")
        download_image(url, label, download_dir, labels)
