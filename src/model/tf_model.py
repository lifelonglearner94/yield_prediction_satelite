import tensorflow as tf

def create_regression_model_sequence(input_shape=(10, 224, 224, 3), use_augmentation=True):
    """
    Erzeugt ein CNN-Modell für Regressionsaufgaben mit optionaler On-the-fly Datenaugmentation.

    Args:
        input_shape (tuple): Die Form des Eingabebatches (Sequenzlänge, Höhe, Breite, Kanäle).
        use_augmentation (bool): Wenn True, wird eine Augmentierungsschicht vor den Convolutional-Blöcken eingesetzt.

    Returns:
        tf.keras.Model: Das erstellte Modell.
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Optional: On-the-fly Datenaugmentation
    if use_augmentation:
        # Da der Input eine Sequenz von Bildern ist, wickeln wir die Augmentierung in eine TimeDistributed-Schicht.
        augmentation_layer = tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2)
            ])
        )
        x = augmentation_layer(inputs)
    else:
        x = inputs

    # Hilfsfunktion für Convolutional-Blöcke
    def conv_block(filters):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2))
        ])

    # Convolutional-Blöcke, jeweils über TimeDistributed angewendet
    x = tf.keras.layers.TimeDistributed(conv_block(32))(x)
    x = tf.keras.layers.TimeDistributed(conv_block(64))(x)
    x = tf.keras.layers.TimeDistributed(conv_block(128))(x)
    x = tf.keras.layers.TimeDistributed(conv_block(256))(x)

    # GlobalAveragePooling für jedes Bild in der Sequenz
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)

    # Bidirektionale LSTM-Schichten zur Verarbeitung der Bildsequenz
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)

    # Dense-Schichten mit BatchNormalization und Dropout
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def compile_model(model, learning_rate=0.005):
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate),
                  loss='mse',
                  metrics=['mae'])
    return model


def preprocess_sequence_for_inference(image_paths, img_size=(224, 224)):
    """
    Lädt und verarbeitet eine Sequenz von Bildpfaden und gibt einen Tensor
    mit der Form (1, 10, img_size[0], img_size[1], 3) zurück.
    """
    def load_and_preprocess(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # oder decode_png, je nach Bildformat
        image = tf.image.resize(image, img_size)
        image = image / 255.0  # Normalisierung
        return image

    # Wende load_and_preprocess auf jedes Bild in der Liste an
    images = tf.map_fn(load_and_preprocess, image_paths, dtype=tf.float32)
    # Erweitere die Dimension, um einen Batch von 1 zu simulieren
    images = tf.expand_dims(images, axis=0)  # shape: (1, 10, Höhe, Breite, 3)
    return images


if __name__ == "__main__":
    image_paths_sequence = [
    "data/germany_satelite_images/Bayern/2019/Bayern_2019-03-01_true_color.png",
    "data/germany_satelite_images/Bayern/2019/Bayern_2019-07-01_true_color.png",
    # ... (insgesamt 10 Pfade)
    ]
    input_tensor = preprocess_sequence_for_inference(image_paths_sequence)
    # prediction = model.predict(input_tensor)
    # print("Prediction:", prediction)
