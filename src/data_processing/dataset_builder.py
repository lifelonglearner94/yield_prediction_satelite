import csv


def build_lists_of_image_paths_and_labels(image_paths_dict, preprocessed_yield_dict):
    list_of_all_image_paths_sorted = []
    list_of_all_yield_values_sorted = []

    assert len(list_of_all_image_paths_sorted) == len(list_of_all_yield_values_sorted)
    return {"image_paths": list_of_all_image_paths_sorted, "labels": list_of_all_yield_values_sorted}


def parse_image(image_path, label, img_size=(224, 224)):
    import tensorflow as tf
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # oder decode_png, je nach Bildformat
    image = tf.image.resize(image, img_size)
    image = image / 255.0  # Normalisierung
    return image, label

def build_dataset(image_paths, labels, batch_size=32, shuffle=True):
    import tensorflow as tf
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def read_yield_data(csv_path):
    """
    Liest die CSV-Datei ein und erstellt ein Dictionary,
    in dem die Keys die Bundesländernamen sind und die Values Listen von Tupeln (Jahr, Erntemenge in t) enthalten.
    """
    bundeslaender = []         # Liste der Bundesländernamen (wie in der Header-Zeile gefunden)
    bundeslaender_indices = [] # Die Spaltenindizes, in denen die Erntemengen für die jeweiligen Bundesländer stehen
    data = {}                  # Das Dictionary, das zurückgegeben wird

    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        rows = list(reader)

    # Schritt 1: Finde die Header-Zeile, die die Bundesländernamen enthält.
    header_row = None
    for row in rows:
        if 'Baden-Württemberg' in row:
            header_row = row
            break

    if header_row is None:
        raise ValueError("Header-Zeile mit Bundesländernamen wurde nicht gefunden.")

    # Schritt 2: Nehme an, dass die Namen in ungeraden Spalten stehen (z. B. Index 1, 3, 5, …)
    for idx, val in enumerate(header_row):
        val = val.strip()
        if idx % 2 == 1 and val != "":
            bundeslaender.append(val)
            bundeslaender_indices.append(idx)
            data[val] = []  # Initialisiere den Dictionary-Eintrag

    # Schritt 3: Iteriere über die Zeilen, die mit einem Jahr beginnen.
    for row in rows:
        # Prüfe, ob das erste Element eine 4-stellige Zahl (Jahr) ist.
        if row and row[0].isdigit() and len(row[0]) == 4:
            jahr = int(row[0])  # Konvertiere das Jahr in eine Ganzzahl

            # Für jedes Bundesland: Hole den entsprechenden Wert aus der entsprechenden Spalte.
            for bl, col_idx in zip(bundeslaender, bundeslaender_indices):
                # Falls die Spalte im aktuellen Row vorhanden ist:
                if col_idx < len(row):
                    value_str = row[col_idx].strip()
                    # Beispielwerte: "305414,0" oder "." (fehlender Wert)
                    if value_str in [".", ""]:
                        value = None
                    else:
                        # Ersetze das Komma durch einen Punkt und konvertiere in float
                        try:
                            value = float(value_str.replace(',', '.'))
                        except ValueError:
                            value = None

                    # Speichere das Jahr zusammen mit dem Wert als Tupel
                    data[bl].append((jahr, value))

    return data

if __name__ == "__main__":
    # Beispielhafte Nutzung:
    csv_path = "data/raw/yield_data_germany_years.csv"
    yield_dict = read_yield_data(csv_path)
    from pprint import pprint

    from src.data_processing.preprocess_data import preprocess_numeric
    pprint(preprocess_numeric(yield_dict))
