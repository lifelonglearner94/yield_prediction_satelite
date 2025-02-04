# tests/test_train_model.py
import numpy as np
import pytest
import tensorflow as tf

# Beispiel: Importiere deine Trainingsfunktion
# Wir gehen davon aus, dass train_model eine Funktion ist, die ein Dictionary zurückgibt
from model.pipeline import train_model

@pytest.fixture
def dummy_data():
    # Erstelle Dummy-Daten: Eingaben und Zielwerte (hier simple lineare Beziehung)
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    return {"data": X, "labels": y}

def test_train_model_creates_model(dummy_data):
    result = train_model(dummy_data)
    model = result.get("model", None)
    # Überprüfe, ob ein Modell erstellt wurde
    assert model is not None, "Das trainierte Modell sollte nicht None sein"
    # Teste, ob das Modell zumindest eine Vorhersage treffen kann
    prediction = model.predict(np.array([[4.0]]))
    assert prediction.shape == (1, 1), "Die Vorhersage sollte die Form (1, 1) haben"

def test_train_model_metric_range(dummy_data):
    result = train_model(dummy_data)
    mse = result.get("mse", None)
    # Überprüfe, ob der MSE berechnet wurde
    assert mse is not None, "MSE muss berechnet und zurückgegeben werden"
    # Beispiel: Wir erwarten einen MSE unter 1.0 (Passe den Schwellenwert ggf. an)
    assert mse < 1.0, f"Der MSE ({mse}) liegt außerhalb des akzeptablen Bereichs"
