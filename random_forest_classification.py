import os
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Clases seleccionadas
metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compression_metrics")
selected_indices = [1, 3, 5, 7, 9]

# Nombres de las clases seleccionadas
all_rle_classes = sorted([f.replace("Rle_", "").replace(".csv", "") for f in os.listdir(metrics_dir) if f.startswith("Rle_")])
selected_classes = [all_rle_classes[i] for i in selected_indices]

# Columnas a extraer
rle_cols = ["image_name", "total_runs", "average_length", "variance_length"]
huff_cols = ["image_name", "number_symbols", "average_length", "entropy"]

# Nombres de las variables combinadas
output_columns = [
    "class", "image_name",
    "rle_total_runs", "rle_avg_run_length", "rle_var_run_length",
    "huff_num_symbols", "huff_avg_code_length", "huff_entropy"
]

# Combinamos los datos de ambos métodos de compresión
combined_data = []
for cls in selected_classes:
    with open(os.path.join(metrics_dir, f"Rle_{cls}.csv"), newline="") as rle_file:
        rle_reader = {row["image_name"]: row for row in csv.DictReader(rle_file)}
    with open(os.path.join(metrics_dir, f"Huffman_{cls}.csv"), newline="") as huff_file:
        huff_reader = csv.DictReader(huff_file)
        for row in huff_reader:
            img_name = row["image_name"]
            if img_name in rle_reader:
                rle_row = rle_reader[img_name]
                combined_data.append([
                    cls, img_name,
                    float(rle_row["total_runs"]),
                    float(rle_row["average_length"]),
                    float(rle_row["variance_length"]),
                    float(row["number_symbols"]),
                    float(row["average_length"]),
                    float(row["entropy"])
                ])

# Creación de un data frame para la clasificación
df = pd.DataFrame(combined_data, columns=output_columns)
X = df.iloc[:, 2:].values
y = df["class"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Modelo de Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Reporte de clasificación
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("Random Forest:\n", report)

# Se guarda el resultado del reporte
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("random_forest_report.csv", index=True)

# Matriz de confusión obtenida
conf_matrix = confusion_matrix(y_test, y_pred)
conf_df = pd.DataFrame(conf_matrix, index=le.classes_, columns=le.classes_)
conf_df.to_csv("random_forest_confusion_matrix.csv")
print("\nMatriz de confusión:\n", conf_df)
