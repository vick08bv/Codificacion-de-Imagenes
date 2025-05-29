import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

# Selección de clases
selected_indices = [1, 3, 5, 7, 9]
num_categories = len(selected_indices)

images_dir = os.path.join(os.path.dirname(os.getcwd()), "Clasificador-Minerales", "images_processed")
all_classes = sorted(os.listdir(images_dir))
selected_classes = [all_classes[i] for i in selected_indices]

# Generador de imágenes con aumento de datos
img_height = 96
img_width = 96
val_split = 0.15
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

# Tasa de aprendizaje
learning_rate = 0.001

# Optimizador
optimizer = Adam(learning_rate=learning_rate)

# Medición del rendimiento
loss = "categorical_crossentropy"
metrics = ["accuracy"]

# Duración del entrenamiento
epochs = 90

# Ajusta dinámicamente la tasa de entrenamiento cuando no hay mejora
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.8,
    patience=3,
    min_lr=5e-6
)


# Detención temprana del entrenamiento
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=9,
    restore_best_weights=True
)

callbacks = [reduce_lr, early_stopping]

# Arquitectura de modelo
model = models.Sequential([

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
    layers.Conv2D(64, kernel_size, activation=layer_activation, padding=padding,
                  kernel_regularizer=regularizers.l2(0.0019)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Capa 4
    layers.Conv2D(128, kernel_size, activation=layer_activation, padding=padding,
                  kernel_regularizer=regularizers.l2(0.0011)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size),

    # Aplanamiento global
    layers.GlobalAveragePooling2D(),

    # Capa densa
    layers.Dense(64, activation=layer_activation),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Capa de salida
    layers.Dense(num_categories, activation=output_activation)
])

# Compilación del modelo
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Resumen del modelo
model.summary()

# Entrenamiento
history = model.fit(
    train_dataset,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_dataset,
    validation_steps=validation_generator.samples // batch_size,
    steps_per_epoch=train_generator.samples // batch_size,
)

# Guarda el modelo
model.save("trained_new_model.keras")

# Extrae métricas del modelo
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Rango de epocas entrenadas
range_epochs = range(1, len(train_loss) + 1)

# Crea la figura con dos subgráficas
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Gráfica de pérdida
axs[0].plot(range_epochs, train_loss, label='Entrenamiento', marker='o')
axs[0].plot(range_epochs, val_loss, label='Validación', marker='o')
axs[0].set_title('Pérdida a través de las épocas')
axs[0].set_xlabel('Épocas')
axs[0].set_ylabel('Pérdida')
axs[0].legend()
axs[0].grid()

# Gráfica de precisión
axs[1].plot(range_epochs, train_acc, label='Entrenamiento', marker='o')
axs[1].plot(range_epochs, val_acc, label='Validación', marker='o')
axs[1].set_title('Precisión a través de las épocas')
axs[1].set_xlabel('Épocas')
axs[1].set_ylabel('Precisión')
axs[1].legend()
axs[1].grid()

# Muestra las gráficas
plt.tight_layout()
plt.savefig("training_new_model.png", dpi=300)
plt.show()
