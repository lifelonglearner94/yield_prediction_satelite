from zenml.pipelines import pipeline
from zenml.steps import step

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Importiere den Dataset-Builder aus dem Data Processing Modul
from src.data_processing.dataset_builder import build_dataset, read_yield_data, build_lists_of_image_paths_and_labels
from src.data_processing.preprocess_data import preprocess_numeric
# Importiere die Modellfunktionen aus dem Modell Modul
from src.model.tf_model import create_regression_model_sequence, compile_model
from src.data_processing.sentinel_api import SentinelHubImageDownloader

import matplotlib.pyplot as plt

import os

@step(enable_cache=False)
def data_ingestion() -> dict:
    """
    Simuliert den Datenimport.
    Hier lädst du idealerweise deine Bildpfade und die zugehörigen Labels,
    beispielsweise aus einer CSV-Datei.
    """
    csv_path = os.path.join("data", "raw", "41215-0010_de.csv")

    # Values are now tons per hectare for each year
    raw_yield_dict = read_yield_data(csv_path, min_year=2017, max_year=2023) # keys: Bundesländer, values: List of total yield for the years 2012 to 2023

    # Remove empty columns and impute where some missing
    preprocessed_yield_dict = preprocess_numeric(raw_yield_dict)

    list_of_years = [t[0] for t in preprocessed_yield_dict[next(iter(preprocessed_yield_dict))]]

    # Download all images if not already on disk
    downloader = SentinelHubImageDownloader(states=list(preprocessed_yield_dict.keys()), years=list_of_years, num_images_per_year=10)
    downloader.download_images()
    image_paths_dict = downloader.image_paths

    # Prepare data for tensorflow
    raw_labeled_data = build_lists_of_image_paths_and_labels(image_paths_dict, preprocessed_yield_dict)

    return raw_labeled_data



@step
def train_model(data: dict, epochs=30, early_stopping_patience=8) -> dict:
    """
    Erzeugt und trainiert das TensorFlow-Modell für die Regression mit einem Validierungsset.
    Inkludiert Early Stopping, um das Training zu beenden, wenn sich der Validierungsfehler nicht mehr verbessert.
    """
    # Train/Test-Split
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
        data["image_paths"], data["labels"], test_size=0.2, random_state=42
    )

    # Baue separate TensorFlow-Datasets für Training und Validierung
    train_dataset = build_dataset(train_image_paths, train_labels, batch_size=1)
    val_dataset = build_dataset(val_image_paths, val_labels, batch_size=1)

    # Erstelle das Modell
    model = create_regression_model_sequence(input_shape=(10, 224, 224, 3), use_augmentation=False)
    model = compile_model(model, learning_rate=0.005)

    # Erstelle EarlyStopping Callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',         # Metrik, die überwacht wird
        patience=early_stopping_patience,  # Anzahl der Epochen, die auf Verbesserung gewartet wird
        min_delta=0.001,            # Minimale Verbesserung, die als signifikant gilt
        restore_best_weights=True,  # Nach dem Stoppen, verwende die besten Gewichte
        verbose=1
    )

    # Trainiere mit Validierungsset und EarlyStopping Callback
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    def plot_loss():
        # Visualisierung des Trainingsverlaufs
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['loss'], label='Trainingsverlust')
        plt.title('Trainingsverlauf')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()  # Alternativ: plt.savefig("training_verlauf.png")

        plt.figure(figsize=(10, 8))
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validierungsverlust')
        plt.title('Validierungsverlauf')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()  # Alternativ: plt.savefig("training_verlauf.png")

    plot_loss()

    # Ausgabe des letzten Validierungsfehlers (oder Trainingsfehlers, falls kein val_loss vorhanden)
    mse = history.history['val_loss'][-1] if 'val_loss' in history.history else history.history['loss'][-1]
    print(f"Training beendet. Validierungs-MSE: {mse}")

    return {"model": model, "val_mse": mse}


@pipeline
def ertragsprognose_pipeline():
    data = data_ingestion()
    train_model(data)

if __name__ == "__main__":
    ertragsprognose_pipeline()

    # TODO try with cloud free pictures
