import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt

# Selección de clases
selected_indices = [1, 3, 5, 7, 9]
num_categories = len(selected_indices)

images_dir = os.path.join(os.path.dirname(os.getcwd()), "Clasificador-Minerales", "images_processed")
all_classes = sorted(os.listdir(images_dir))
selected_classes = [all_classes[i] for i in selected_indices]

print(f"{num_categories} clases seleccionadas:")
for c in selected_classes:
    print(c.split("_")[0])

# Generador de imágenes con aumento de datos
img_height = 96
img_width = 96
val_split=0.15
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=val_split,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generador del conjunto de entrenamiento
train_generator = train_datagen.flow_from_directory(
    images_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=selected_classes,
    shuffle=True
)

# Generador del conjunto de validación
validation_generator = train_datagen.flow_from_directory(
    images_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=selected_classes,
    shuffle=False
)

# Agrega repetición y precarga
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_categories), dtype=tf.float32)
    )
)

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_categories), dtype=tf.float32)
    )
)

train_dataset = train_dataset.repeat().prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.repeat().prefetch(tf.data.AUTOTUNE)

# Parámetros usados para entrenar

# Tamaño del filtro
kernel_size = (3, 3)

# Funciones de activación
layer_activation = "relu"
output_activation = "softmax"

# Tamaño después de convolución
padding = "same"

# Reducción
pool_size = (2, 2)

# Optimizador
optimizer = "adam"

# Medición del rendimiento
loss = "categorical_crossentropy"
metrics = ["accuracy"]

# Duración del entrenamiento
epochs = 15

# Ajusta dinámicamente la tasa de entrenamiento cuando no hay mejora
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.8,
    patience=2,
    min_lr=5e-6
)

callbacks = [reduce_lr]

# Modelo 1

# Arquitectura de modelo
model_1 = models.Sequential([

    # Entrada
    layers.Input(shape=(img_height, img_width, 3)),

    # Capa 1
    layers.Conv2D(32, kernel_size, activation=layer_activation, padding=padding),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Capa 2
    layers.Conv2D(64, kernel_size, activation=layer_activation, padding=padding),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Capa 3
    layers.Conv2D(128, kernel_size, activation=layer_activation, padding=padding),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Aplanamiento global
    layers.GlobalAveragePooling2D(),

    # Capa densa 1
    layers.Dense(64, activation=layer_activation),
    layers.BatchNormalization(),

    # Capa densa 2
    layers.Dense(32, activation=layer_activation),
    layers.BatchNormalization(),

    # Capa de salida
    layers.Dense(num_categories, activation=output_activation)
])

# Compilación del modelo
model_1.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Resumen del modelo
model_1.summary()

# Entrenamiento
history_1 = model_1.fit(
    train_dataset,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_dataset,
    validation_steps=validation_generator.samples // batch_size,
    steps_per_epoch=train_generator.samples // batch_size,
)

# Modelo 2

# Arquitectura de modelo
model_2 = models.Sequential([

    # Entrada
    layers.Input(shape=(img_height, img_width, 3)),

    # Capa 1
    layers.Conv2D(16, kernel_size, activation=layer_activation, padding=padding),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Capa 2
    layers.Conv2D(32, kernel_size, activation=layer_activation, padding=padding),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Capa 3
    layers.Conv2D(64, kernel_size, activation=layer_activation, padding=padding),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Capa 4
    layers.Conv2D(128, kernel_size, activation=layer_activation, padding=padding),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Aplanamiento global
    layers.GlobalAveragePooling2D(),

    # Capa densa
    layers.Dense(64, activation=layer_activation),
    layers.BatchNormalization(),

    # Capa de salida
    layers.Dense(num_categories, activation=output_activation)
])

# Compilación del modelo
model_2.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Resumen del modelo
model_2.summary()

# Entrenamiento
history_2 = model_2.fit(
    train_dataset,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_dataset,
    validation_steps=validation_generator.samples // batch_size,
    steps_per_epoch=train_generator.samples // batch_size,
)

# Arquitectura de modelo
model_3 = models.Sequential([

    # Entrada
    layers.Input(shape=(img_height, img_width, 3)),

    # Capa 1
    layers.Conv2D(32, (5, 5), activation=layer_activation, padding=padding),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Capa 2
    layers.Conv2D(64, (5, 5), activation=layer_activation, padding=padding),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Capa 3
    layers.Conv2D(128, kernel_size, activation=layer_activation, padding=padding),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Aplanamiento global
    layers.GlobalAveragePooling2D(),

    # Capa densa
    layers.Dense(256, activation=layer_activation),
    layers.BatchNormalization(),

    # Capa de salida
    layers.Dense(num_categories, activation=output_activation)
])

# Compilación del modelo
model_3.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Resumen del modelo
model_3.summary()

# Entrenamiento
history_3 = model_3.fit(
    train_dataset,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_dataset,
    validation_steps=validation_generator.samples // batch_size,
    steps_per_epoch=train_generator.samples // batch_size,
)

# Extraer métricas de los tres modelos
train_loss_1 = history_1.history['loss']
val_loss_1 = history_1.history['val_loss']
train_acc_1 = history_1.history['accuracy']
val_acc_1 = history_1.history['val_accuracy']

train_loss_2 = history_2.history['loss']
val_loss_2 = history_2.history['val_loss']
train_acc_2 = history_2.history['accuracy']
val_acc_2 = history_2.history['val_accuracy']

train_loss_3 = history_3.history['loss']
val_loss_3 = history_3.history['val_loss']
train_acc_3 = history_3.history['accuracy']
val_acc_3 = history_3.history['val_accuracy']

# Crear la figura con dos subgráficas
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Gráfica de pérdida
axs[0].plot(range(epochs), train_loss_1, label='Modelo 1 - Entrenamiento', marker='o')
axs[0].plot(range(epochs), val_loss_1, label='Modelo 1 - Validación', marker='o')
axs[0].plot(range(epochs), train_loss_2, label='Modelo 2 - Entrenamiento', marker='o')
axs[0].plot(range(epochs), val_loss_2, label='Modelo 2 - Validación', marker='o')
axs[0].plot(range(epochs), train_loss_3, label='Modelo 3 - Entrenamiento', marker='o')
axs[0].plot(range(epochs), val_loss_3, label='Modelo 3 - Validación', marker='o')
axs[0].set_title('Pérdida a través de las épocas')
axs[0].set_xlabel('Épocas')
axs[0].set_ylabel('Pérdida')
axs[0].legend()
axs[0].grid()

# Gráfica de precisión
axs[1].plot(range(epochs), train_acc_1, label='Modelo 1 - Entrenamiento', marker='o')
axs[1].plot(range(epochs), val_acc_1, label='Modelo 1 - Validación', marker='o')
axs[1].plot(range(epochs), train_acc_2, label='Modelo 2 - Entrenamiento', marker='o')
axs[1].plot(range(epochs), val_acc_2, label='Modelo 2 - Validación', marker='o')
axs[1].plot(range(epochs), train_acc_3, label='Modelo 3 - Entrenamiento', marker='o')
axs[1].plot(range(epochs), val_acc_3, label='Modelo 3 - Validación', marker='o')
axs[1].set_title('Precisión a través de las épocas')
axs[1].set_xlabel('Épocas')
axs[1].set_ylabel('Precisión')
axs[1].legend()
axs[1].grid()

# Mostrar las gráficas
plt.tight_layout()
plt.savefig("testing_new_models.png", dpi=300)
plt.show()
