import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from keras_tuner import HyperModel, RandomSearch

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

train_dataset = train_dataset.repeat()
validation_dataset = validation_dataset.repeat()

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

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
epochs = 10

# Ajusta dinámicamente la tasa de entrenamiento cuando no hay mejora
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=5e-6
)

callbacks = [reduce_lr]


# Definición del modelo con Keras Tuner para optimización de hiperparámetros
class MyHyperModel(HyperModel):
    def build(self, hp):
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
                          kernel_regularizer=regularizers.l2(
                              hp.Float('l2_regularization_c3', min_value=0.001, max_value=0.01, sampling="linear"))),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size),

            # Capa 4
            layers.Conv2D(128, kernel_size, activation=layer_activation, padding=padding,
                          kernel_regularizer=regularizers.l2(
                              hp.Float('l2_regularization_c4', min_value=0.001, max_value=0.01, sampling="linear"))),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size),

            # Aplanamiento global
            layers.GlobalAveragePooling2D(),

            # Capa densa
            layers.Dense(64, activation=layer_activation),
            layers.BatchNormalization(),
            layers.Dropout(hp.Choice('dropout_rate', [0.3, 0.4, 0.5])),

            # Capa de salida
            layers.Dense(num_categories, activation=output_activation)

        ])

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model


# Modelo para ajustar
hypermodel = MyHyperModel()

# Realiza la búsqueda aleatoria de hiperparámetros
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='keras_new_tuning',
    project_name='hyperparameter_tuning'
)

# Realiza la búsqueda de hiperparámetros
tuner.search(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    validation_steps=validation_generator.samples // batch_size,
    steps_per_epoch=train_generator.samples // batch_size,
)

# Muestra los resultados de todas las combinaciones de hiperparámetros
for trial in tuner.oracle.trials.values():
    trial_hps = trial.hyperparameters.values
    val_accuracy = trial.score
    print(f"Hiperparámetros: {trial_hps}")
    print(f"Precisión en validación: {val_accuracy}")
    print("---------------")
