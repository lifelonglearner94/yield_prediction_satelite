import os
import tensorflow as tf
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split

# Importiere den Dataset-Builder aus dem Data Processing Modul
from src.data_processing.dataset_builder import build_dataset, read_yield_data, build_lists_of_image_paths_and_labels
from src.data_processing.preprocess_data import preprocess_numeric
# Importiere die Modellfunktionen aus dem Modell Modul
from src.model.tf_model import create_regression_model_sequence, compile_model
from src.data_processing.sentinel_api import SentinelHubImageDownloader


class HyperparameterTuner:
    """
    Class for hyperparameter/model tuning.

    This tuner iterates over combinations of:
      - batch_size (e.g. 1 and 2)
      - learning_rate (e.g. 0.0001, 0.001, 0.01)
      - optimizer (Adam, SGD, RMSprop)
      - use_augmentation (True, False)

    For each combination it trains the model (with early stopping) and records
    the best validation MAE. At the end, all results are saved into a pandas DataFrame,
    which is sorted by the best validation MAE (lowest first) and the top 10 combinations
    are printed.
    """

    def __init__(self,
                 batch_sizes=[1],
                 learning_rates=[0.005, 0.01],
                 optimizers=["sgd"],
                 epochs=90,
                 early_stopping_patience=8,
                 input_shape=(10, 224, 224, 3),
                 use_augmentation_options=[True, False]):
        self.batch_sizes = batch_sizes
        self.learning_rates = learning_rates
        self.optimizers = optimizers
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.input_shape = input_shape
        self.use_augmentation_options = use_augmentation_options  # New tuning parameter
        self.results = []  # to store tuning results

        # Load your data once; the data_ingestion function is expected to return a dict
        # with at least "image_paths" and "labels"
        print("Loading data using data_ingestion() ...")
        self.data = self.data_ingestion()

    def data_ingestion(self) -> dict:
        """
        Simuliert den Datenimport.
        Hier lädst du idealerweise deine Bildpfade und die zugehörigen Labels,
        beispielsweise aus einer CSV-Datei.
        """
        csv_path = os.path.join("data", "raw", "41215-0010_de.csv")

        # Values are now tons per hectare for each year
        raw_yield_dict = read_yield_data(csv_path, min_year=2017, max_year=2023)  # keys: Bundesländer, values: List of total yield for the years 2012 to 2023

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

    def get_optimizer(self, optimizer_name, learning_rate):
        """Return a TensorFlow optimizer instance based on the optimizer name and learning rate."""
        optimizer_name = optimizer_name.lower()
        if optimizer_name == "adam":
            return tf.keras.optimizers.Adam(learning_rate)
        elif optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate)
        elif optimizer_name == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def run(self):
        """Run hyperparameter tuning over all combinations and output the sorted results."""
        # Loop over all combinations including the use_augmentation option.
        for batch_size, lr, opt_name, use_aug in product(
            self.batch_sizes,
            self.learning_rates,
            self.optimizers,
            self.use_augmentation_options
        ):
            print("\n======================================")
            print(f"Training with batch_size={batch_size}, learning_rate={lr}, optimizer={opt_name}, use_augmentation={use_aug}")
            print("======================================")

            # Perform a train/validation split
            train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
                self.data["image_paths"], self.data["labels"], test_size=0.2, random_state=42
            )

            # Build TensorFlow datasets using your build_dataset() function.
            train_dataset = build_dataset(train_image_paths, train_labels, batch_size=batch_size)
            val_dataset = build_dataset(val_image_paths, val_labels, batch_size=batch_size)

            # Create the model using your existing function.
            # Pass the current augmentation option (use_aug) to the model.
            model = create_regression_model_sequence(input_shape=self.input_shape, use_augmentation=use_aug)

            # Get the optimizer instance.
            optimizer = self.get_optimizer(opt_name, lr)

            # Compile the model.
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            # Create EarlyStopping callback.
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1
            )

            # Train the model.
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.epochs,
                callbacks=[early_stopping],
                verbose=1
            )

            # Extract the best validation MAE from training history.
            # Here we assume that 'val_mae' is recorded.
            if 'val_mae' in history.history:
                best_val_mae = min(history.history['val_mae'])
            else:
                best_val_mae = None  # Or handle accordingly if the metric is missing

            # Record the results.
            result = {
                "batch_size": batch_size,
                "learning_rate": lr,
                "optimizer": opt_name,
                "use_augmentation": use_aug,  # Record the augmentation option
                "best_val_mae": best_val_mae,
                "epochs_trained": len(history.history['loss'])
            }
            self.results.append(result)

            # <-- NEW CODE: Print current top 10 hyperparameter combinations so far
            results_so_far = pd.DataFrame(self.results)
            results_so_far_sorted = results_so_far.sort_values(by="best_val_mae", ascending=True)
            print("\nCurrent top 10 hyperparameter combinations so far:")
            print(results_so_far_sorted.head(10))
            # <-- END NEW CODE

        # Create a DataFrame with the results.
        results_df = pd.DataFrame(self.results)
        # Sort by best_val_mae in ascending order (lowest MAE first).
        results_df_sorted = results_df.sort_values(by="best_val_mae", ascending=True)

        print("\nFinal Top 10 hyperparameter combinations (sorted by best_val_mae):")
        print(results_df_sorted.head(10))
        return results_df_sorted

if __name__ == "__main__":
    tuner = HyperparameterTuner()
    tuner.run()
