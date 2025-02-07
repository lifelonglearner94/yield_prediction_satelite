```markdown
# Yield Prediction from Satellite Images

This project implements a machine learning pipeline that leverages satellite images (captured until July) to predict crop yield in tons per hectare. The model achieves a Mean Absolute Error (MAE) of about **10 t/ha**. The pipeline is built using [ZenML](https://zenml.io/) for orchestrating the workflow and [TensorFlow](https://www.tensorflow.org/) for model development.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Pipeline Overview](#pipeline-overview)
  - [Data Ingestion Step](#data-ingestion-step)
  - [Model Training Step](#model-training-step)
- [Code Snippet](#code-snippet)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

This project demonstrates how to predict agricultural yield using satellite images. The pipeline performs the following tasks:

- **Data Ingestion**: Loads and preprocesses yield data from CSV files, downloads satellite images via the Sentinel Hub API, and prepares data for TensorFlow.
- **Model Training**: Splits data into training and validation sets, constructs and compiles a regression model, trains the model, and visualizes the training progress.

---

## Project Structure

```
├── data
│   └── raw
│       └── 41215-0010_de.csv      # CSV file containing yield data
├── src
│   ├── data_processing
│   │   ├── dataset_builder.py     # Functions for building datasets and reading yield data
│   │   ├── preprocess_data.py     # Functions for cleaning and preprocessing yield data
│   │   └── sentinel_api.py        # Module to download satellite images from Sentinel Hub
│   └── model
│       └── tf_model.py            # Functions to create and compile the TensorFlow regression model
└── pipeline.py                    # ZenML pipeline definition that ties the steps together
```

---

## Requirements

- **Python** 3.8 or higher
- **TensorFlow**
- **ZenML**
- **scikit-learn**
- **NumPy**
- **Matplotlib**

Install additional dependencies as listed in the `requirements.txt` file.

---

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install the Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Pipeline Overview

The pipeline consists of two main steps:

### Data Ingestion Step

- **CSV Data Loading**: Reads the yield data from a CSV file (located in `data/raw/`).
- **Preprocessing**: Cleans and imputes missing values in the yield data.
- **Satellite Image Download**: Uses `SentinelHubImageDownloader` to download images if they are not already available locally.
- **Data Preparation**: Prepares lists of image paths and corresponding yield labels for further processing.

### Model Training Step

- **Data Splitting**: Divides the data into training and validation sets using `train_test_split`.
- **Dataset Creation**: Builds TensorFlow datasets from image paths and yield labels.
- **Model Construction**: Creates a regression model with the specified input shape.
- **Compilation & Training**: Compiles the model with a set learning rate and trains it, while logging the training progress.
- **Visualization**: Plots training and validation loss curves to visualize performance.


---

## Results

- **Prediction Target**: Yield in tons per hectare.
- **Input Data**: Satellite images captured until July.
- **Performance**: Achieved an MAE of approximately **10 t/ha**.

---

## Future Work

- **Image Quality**: Explore using cloud-free satellite images to potentially enhance prediction accuracy.
- **Model Optimization**: Experiment with different model architectures and hyperparameters.
- **Dataset Expansion**: Incorporate additional regions and more historical data to improve model robustness.

---
