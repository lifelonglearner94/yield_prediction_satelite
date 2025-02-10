import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import helper functions from your project modules.
from src.data_processing.dataset_builder import build_dataset, read_yield_data, build_lists_of_image_paths_and_labels
from src.data_processing.preprocess_data import preprocess_numeric
from src.data_processing.sentinel_api import SentinelHubImageDownloader

class GeneticArchitectureSearch:
    """
    Genetic algorithm to search for the optimal model architecture.

    The search space for each candidate (individual) includes:
      - use_augmentation: Whether to use on-the-fly augmentation.
      - num_conv_blocks: Number of convolutional blocks (2 to 4).
      - filters: A list (per conv block) of filter counts (chosen from 32, 64, 128, 256).
      - num_lstm_layers: Either 1 or 2 bidirectional LSTM layers.
      - lstm_units: Number of units in the LSTM layers (64, 128, or 256).
      - num_dense_layers: Either 1 or 2 fully connected (dense) layers.
      - dense_units: A list (per dense layer) of units (chosen from 128, 256, 512).
      - dropout_rate: Dropout rate after each dense layer (0.2, 0.3, or 0.5).

    Each candidate is trained (with early stopping) on your satellite image data and evaluated
    using the best validation MAE. After each generation the best individuals are selected,
    crossed over and mutated to form the next generation.
    """
    def __init__(self,
                 population_size=10,
                 generations=5,
                 mutation_rate=0.1,
                 epochs=20,
                 early_stopping_patience=3,
                 input_shape=(10, 224, 224, 3)):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.input_shape = input_shape
        self.results = []  # Store generation-wise best individuals

        # Load data once (similar to your HyperparameterTuner).
        print("Loading data using data_ingestion() ...")
        self.data = self.data_ingestion()
        # Generate initial population.
        self.population = self.generate_initial_population()

    def data_ingestion(self) -> dict:
        """
        Simulates data import: loads image paths and labels.
        Uses yield CSV data and downloads images if not present.
        """
        csv_path = os.path.join("data", "raw", "41215-0010_de.csv")
        raw_yield_dict = read_yield_data(csv_path, min_year=2017, max_year=2023)
        preprocessed_yield_dict = preprocess_numeric(raw_yield_dict)
        list_of_years = [t[0] for t in preprocessed_yield_dict[next(iter(preprocessed_yield_dict))]]

        downloader = SentinelHubImageDownloader(states=list(preprocessed_yield_dict.keys()),
                                                years=list_of_years,
                                                num_images_per_year=10)
        downloader.download_images()
        image_paths_dict = downloader.image_paths

        raw_labeled_data = build_lists_of_image_paths_and_labels(image_paths_dict, preprocessed_yield_dict)
        return raw_labeled_data

    def generate_initial_population(self):
        """Generates a list of individuals (candidate architectures) randomly."""
        population = []
        for _ in range(self.population_size):
            individual = self.random_individual()
            population.append(individual)
        return population

    def random_individual(self):
        """Randomly creates an individual representing a model architecture."""
        individual = {}
        individual['use_augmentation'] = random.choice([True, False])
        individual['num_conv_blocks'] = random.choice([2, 3, 4])
        # For each conv block, choose a filter count from these options.
        individual['filters'] = [random.choice([32, 64, 128, 256]) for _ in range(individual['num_conv_blocks'])]
        individual['num_lstm_layers'] = random.choice([1, 2])
        individual['lstm_units'] = random.choice([64, 128, 256])
        individual['num_dense_layers'] = random.choice([1, 2])
        individual['dense_units'] = [random.choice([128, 256, 512]) for _ in range(individual['num_dense_layers'])]
        individual['dropout_rate'] = random.choice([0.2, 0.3, 0.5])
        return individual

    def build_model_from_individual(self, individual):
        """
        Creates a TensorFlow Keras model based on the individual's genome.
        The model processes a sequence of images using TimeDistributed convolutional
        blocks, bidirectional LSTM layers, and dense layers for regression.
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs

        # Optional on-the-fly augmentation.
        if individual['use_augmentation']:
            augmentation_layer = tf.keras.layers.TimeDistributed(
                tf.keras.Sequential([
                    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                    tf.keras.layers.RandomRotation(0.2),
                    tf.keras.layers.RandomZoom(0.2)
                ])
            )
            x = augmentation_layer(x)

        # Helper function for a convolutional block.
        def conv_block(filters):
            return tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2))
            ])

        # Apply the convolutional blocks (wrapped in TimeDistributed).
        for filt in individual['filters']:
            x = tf.keras.layers.TimeDistributed(conv_block(filt))(x)

        # Global Average Pooling on each image.
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)

        # LSTM layers to process the sequence.
        if individual['num_lstm_layers'] == 1:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(individual['lstm_units']))(x)
        else:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(individual['lstm_units'], return_sequences=True))(x)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(individual['lstm_units']))(x)

        # Dense layers with BatchNormalization and Dropout.
        for units in individual['dense_units']:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(individual['dropout_rate'])(x)

        outputs = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, outputs)
        return model

    def evaluate_individual(self, individual):
        """
        Evaluates an individual by training the model on training data and
        returning the best validation MAE achieved.
        """
        # Split data (using a fixed random state for reproducibility).
        train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
            self.data["image_paths"], self.data["labels"], test_size=0.2, random_state=42
        )
        train_dataset = build_dataset(train_image_paths, train_labels, batch_size=1)
        val_dataset = build_dataset(val_image_paths, val_labels, batch_size=1)

        # Build and compile the model.
        model = self.build_model_from_individual(individual)
        # Here we use a fixed optimizer and learning rate.
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=[early_stopping],
            verbose=0
        )

        # Use the best validation MAE as the fitness value.
        if 'val_mae' in history.history:
            best_val_mae = min(history.history['val_mae'])
        else:
            best_val_mae = float('inf')
        return best_val_mae

    def selection(self, evaluated_population):
        """
        Selects the top 50% individuals (lower validation MAE is better).
        evaluated_population is a list of tuples: (individual, fitness)
        """
        evaluated_population.sort(key=lambda x: x[1])
        selected = evaluated_population[:max(1, len(evaluated_population) // 2)]
        return [ind for ind, fitness in selected]

    def crossover(self, parent1, parent2):
        """
        Performs a simple gene-level crossover between two parents.
        For each gene, randomly selects the gene from one of the parents.
        """
        child = {}
        for key in parent1.keys():
            if isinstance(parent1[key], list):
                # For list genes, perform element-wise crossover.
                child[key] = []
                for gene1, gene2 in zip(parent1[key], parent2[key]):
                    child[key].append(random.choice([gene1, gene2]))
            else:
                child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def mutate(self, individual):
        """
        With probability equal to mutation_rate, randomly mutates each gene.
        """
        if random.random() < self.mutation_rate:
            individual['use_augmentation'] = not individual['use_augmentation']
        if random.random() < self.mutation_rate:
            individual['num_conv_blocks'] = random.choice([2, 3, 4])
            # Update the filters list based on the new number of conv blocks.
            individual['filters'] = [random.choice([32, 64, 128, 256]) for _ in range(individual['num_conv_blocks'])]
        if random.random() < self.mutation_rate:
            individual['num_lstm_layers'] = random.choice([1, 2])
        if random.random() < self.mutation_rate:
            individual['lstm_units'] = random.choice([64, 128, 256])
        if random.random() < self.mutation_rate:
            individual['num_dense_layers'] = random.choice([1, 2])
            individual['dense_units'] = [random.choice([128, 256, 512]) for _ in range(individual['num_dense_layers'])]
        if random.random() < self.mutation_rate:
            individual['dropout_rate'] = random.choice([0.2, 0.3, 0.5])
        return individual

    def run(self):
        """
        Runs the genetic algorithm over the specified number of generations.
        In each generation, each individual is evaluated, the best are selected,
        and new offspring are generated via crossover and mutation.
        """
        best_overall = None
        best_overall_fitness = float('inf')

        for gen in range(self.generations):
            print(f"\nGeneration {gen + 1}/{self.generations}")
            evaluated_population = []
            # Evaluate all individuals.
            for individual in self.population:
                fitness = self.evaluate_individual(individual)
                evaluated_population.append((individual, fitness))
                print(f"Individual: {individual} -> Fitness (best val MAE): {fitness}")

            # Sort by fitness (lower MAE is better).
            evaluated_population.sort(key=lambda x: x[1])
            best_gen = evaluated_population[0]
            print(f"Best individual in Generation {gen + 1}: {best_gen[0]} with fitness {best_gen[1]}")

            if best_gen[1] < best_overall_fitness:
                best_overall = best_gen[0]
                best_overall_fitness = best_gen[1]

            self.results.append({
                "generation": gen + 1,
                "best_individual": best_gen[0],
                "fitness": best_gen[1]
            })

            # Selection: choose the top half of the individuals.
            selected = self.selection(evaluated_population)

            # Generate new population via crossover and mutation.
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population

        print("\nFinal Best Architecture:")
        print(best_overall)
        print(f"With Fitness (best val MAE): {best_overall_fitness}")
        return best_overall, best_overall_fitness

if __name__ == "__main__":
    # Instantiate and run the genetic architecture search.
    gas = GeneticArchitectureSearch(population_size=10, generations=5, mutation_rate=0.1)
    best_arch, best_fitness = gas.run()
