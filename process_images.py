import cv2
import os
import process_pipeline as pp

# Carpetas de imágenes
current_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(current_dir, "images")
output_dir = os.path.join(current_dir, "images_processed")

# Crea la carpeta de destino si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

n = 1
# Lee y procesa todas las imágenes
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)

    try:
        # Carga la imagen
        image = cv2.imread(input_path)

        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {filename}")

        # Procesa la imagen
        processed = pp.process_image(image)

        # Guarda la imagen procesada en la carpeta de salida
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed)

        n += 1
        if n % 500 == 0:
            print(f"Procesadas {n} imágenes")

    except Exception as e:
        # Captura cualquier error y muestra el nombre del archivo que lo causó
        print(f"Error procesando la imagen {filename}: {str(e)}")
