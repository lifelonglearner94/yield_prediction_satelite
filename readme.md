# Yield Prediction from Satellite Images

Welcome to the **Yield Prediction** project! This innovative pipeline leverages satellite imagery and state-of-the-art machine learning to predict crop yields in tons per hectare. Building on our success in achieving a Mean Absolute Error (MAE) of around **10 t/ha**, we've recently added some cool enhancements that improve both data quality and model performance.

---

## What's New?

- **Cloud Masked Image Processing**:
  We now download cloud-masked satellite images and apply a cloud removal process. This ensures that our input images are clearer and more reliable, leading to better prediction quality.

- **Hyperparameter Tuning**:
  After extensive experimentation, our hyperparameter tuning revealed that the **SGD optimizer** with a learning rate of **0.005** delivers the best performance for our model.

---

## Table of Contents

- [Overview](#overview)
- [What's New?](#whats-new)
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

This project demonstrates how to predict agricultural yields using satellite images. Our end-to-end pipeline:
- **Ingests** yield data and satellite imagery (now enhanced with cloud masking and removal),
- **Trains** a robust regression model using advanced hyperparameter tuning,
- **Visualizes** the training progress and performance.

Recent updates have focused on improving image quality and fine-tuning model parameters for optimal results.

---

## Requirements

- **Python** 3.8 or higher
- **TensorFlow**
- **ZenML**
- **scikit-learn**
- **NumPy**
- **Matplotlib**

> **Note:** Additional dependencies are listed in the `requirements.txt` file.

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

Our pipeline consists of two main stages:

### Data Ingestion Step

- **CSV Data Loading**:
  Reads yield data from CSV files (located in `data/raw/`).

- **Preprocessing**:
  Cleans the data and imputes missing values.

- **Satellite Image Download**:
  Uses the `SentinelHubImageDownloader` to fetch images.
  **New:** Downloads cloud-masked images to ensure higher data quality.

- **Cloud Removal**:
  Applies a processing step to remove cloud cover, resulting in clearer imagery.

- **Data Preparation**:
  Organizes image paths and corresponding yield labels for further processing.

### Model Training Step

- **Data Splitting**:
  Divides the dataset into training and validation sets using `train_test_split`.

- **Dataset Creation**:
  Constructs TensorFlow datasets from the image paths and yield labels.

- **Model Construction**:
  Builds a regression model configured to the input image shape.

- **Hyperparameter Tuning**:
  After experimenting with various settings, we found that using the **SGD optimizer** with a learning rate of **0.005** provides optimal results.

- **Compilation & Training**:
  Compiles and trains the model while tracking the training progress.

- **Visualization**:
  Plots training and validation loss curves to monitor performance improvements.

---

## Code Snippet

Below is an example snippet that highlights the hyperparameter tuning with the SGD optimizer:

```python
# Example: Model compilation and training with tuned hyperparameters
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005),
              loss='mean_absolute_error')

history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=50)
```

---

## Results

- **Prediction Target**:
  Yield in tons per hectare.

- **Input Data**:
  Satellite images captured until July, now enhanced with cloud removal.

- **Performance**:
  Achieved an MAE of approximately **8.2 t/ha** through refined data processing and hyperparameter tuning.

---

## Future Work

- **Advanced Image Processing**:
  Further refine cloud detection and removal methods to boost image quality.

- **Enhanced Model Architectures**:
  Explore new architectures and additional hyperparameter strategies to push prediction accuracy further.

- **Dataset Expansion**:
  Incorporate additional regions and historical data to improve model robustness.

- **Real-Time Prediction**:
  Develop capabilities for real-time data ingestion and yield prediction.

---

Enjoy exploring the project, and feel free to contribute or reach out with any questions or ideas!

Happy Coding!
