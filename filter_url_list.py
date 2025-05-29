import re

data = dict()
filtered_data = dict()
selected_labels = []

# Lectura de la lista de url
with open("url_list.csv") as file:
    lines = file.readlines()
    for line in lines:
        entry = line.split(",")
        url, label = entry[0].strip(), entry[1].strip()
        # Escritura en el diccionario
        data[url] = label

# Lectura de las categorias elegidas
with open("list_selected_classes (full_names).txt") as file:
    lines = file.readlines()
    for line in lines:
        label = line.strip()
        selected_labels.append(label)

# Filtrando
for url, label in data.items():
    if label in selected_labels:
        # Se simplifica el label
        expression = r"\(Var: ([^\)]+)\)"
        match = re.search(expression, label)
        if match:
            filtered_data[url] = match.group(1)
        else:
            filtered_data[url] = label

# Se guardan las url filtradas
with open("filtered_url_list.csv", 'w') as file:
    for url, label in filtered_data.items():
        file.write(f"{url},{label}\n")
