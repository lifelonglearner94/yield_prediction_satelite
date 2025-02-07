import csv


def build_lists_of_image_paths_and_labels(image_paths_dict, preprocessed_yield_dict):
    list_of_all_image_groups = []  # Jede Gruppe ist eine Liste von 10 Bildpfaden
    list_of_all_yield_values = []

    for state in image_paths_dict:
        for year_str in image_paths_dict[state]:
            image_paths = image_paths_dict[state][year_str]  # Liste von Bildpfaden für ein Jahr
            year = int(year_str)

            # Suche nach dem entsprechenden Ertragswert
            yield_value = next((value for y, value in preprocessed_yield_dict[state] if y == year), None)

            if yield_value is not None:
                if len(image_paths) != 10:
                    print(f"Warnung: Für {state} im Jahr {year} wurden nicht 10 Bilder gefunden, sondern {len(image_paths)}.")
                    # Je nach Bedarf kannst du hier entweder fortfahren oder diesen Fall extra behandeln.
                list_of_all_image_groups.append(image_paths)
                list_of_all_yield_values.append(yield_value)
            else:
                print(f"Warnung: Kein Ertragswert für {state} im Jahr {year} gefunden.")

    assert len(list_of_all_image_groups) == len(list_of_all_yield_values), \
        "Die Listen der Bildgruppen und Ertragswerte haben unterschiedliche Längen."

    return {"image_paths": list_of_all_image_groups, "labels": list_of_all_yield_values}



def parse_image_sequence(image_paths, label, img_size=(224, 224)):
    import tensorflow as tf

    def load_and_preprocess(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Falls PNG: tf.image.decode_png
        image = tf.image.resize(image, img_size)
        image = image / 255.0  # Normalisierung
        return image

    # image_paths ist hier ein Tensor mit shape (10,), d.h. 10 Bildpfade
    images = tf.map_fn(load_and_preprocess, image_paths, fn_output_signature=tf.float32)
    # Jetzt hat "images" die Form (10, img_size[0], img_size[1], 3)
    return images, label

def build_dataset(image_groups, labels, batch_size=32, shuffle=True):
    import tensorflow as tf

    # image_groups ist eine Liste von Listen (jede innere Liste enthält 10 Bildpfade)
    dataset = tf.data.Dataset.from_tensor_slices((image_groups, labels))
    dataset = dataset.map(parse_image_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_groups))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


import csv

def read_yield_data(csv_path, min_year=None, max_year=None):
    """
    Liest die CSV-Datei ein und erstellt ein Dictionary,
    in dem die Keys die Bundesländernamen sind und die Values
    Listen von Tupeln (Jahr, Ertrag in t/ha) enthalten.

    Für jedes Bundesland werden aus den beiden Spalten
      - Anbaufläche Gemüse (ha) und
      - Erntemenge Gemüse (t)
    der Quotient (t/ha) berechnet.

    Optional:
      - min_year: Das kleinste Jahr, das gespeichert werden soll.
      - max_year: Das größte Jahr, das gespeichert werden soll.
    """
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        rows = list(reader)

    # Schritt 1: Finde die Zeile, in der mindestens "Baden-Württemberg" vorkommt.
    # Diese Zeile (row) enthält laut CSV-Dokument die Bundesländernamen in den relevanten Spalten.
    header_row = None
    for row in rows:
        if 'Baden-Württemberg' in row:
            header_row = row
            break

    if header_row is None:
        raise ValueError("Header-Zeile mit Bundesländernamen wurde nicht gefunden.")

    # Schritt 2: Baue ein Mapping auf: Für jedes Bundesland (aus header_row)
    # nehmen wir an, dass der Name in Spalte (4*k + 1) steht,
    # wobei k=0,1,2,…;
    # in den Datenzeilen steht dann in derselben Spalte der
    # 'Anbaufläche Gemüse'-Wert und in Spalte (4*k + 3) der 'Erntemenge Gemüse'-Wert.
    state_to_cols = {}
    k = 0
    while (4 * k + 1) < len(header_row):
        state = header_row[4 * k + 1].strip()
        if state:
            # Die Spalte für Anbaufläche Gemüse (ha) ist 4*k + 1,
            # und für Erntemenge Gemüse (t) ist 4*k + 3.
            state_to_cols[state] = (4 * k + 1, 4 * k + 3)
        k += 1

    # Initialisiere das Rückgabe-Dictionary
    data = {state: [] for state in state_to_cols.keys()}

    # Schritt 3: Iteriere über alle Zeilen, die mit einem Jahr (4-stellig) beginnen.
    for row in rows:
        if row and row[0].isdigit() and len(row[0]) == 4:
            jahr = int(row[0])
            # Überspringe Jahre, die nicht im gewünschten Bereich liegen.
            if min_year is not None and jahr < min_year:
                continue
            if max_year is not None and jahr > max_year:
                continue

            # Für jedes Bundesland:
            for state, (area_idx, yield_idx) in state_to_cols.items():
                # Prüfe, ob die Indizes in der aktuellen Zeile existieren.
                if area_idx < len(row) and yield_idx < len(row):
                    area_str = row[area_idx].strip()
                    yield_str = row[yield_idx].strip()

                    # Fehlende oder fehlerhafte Werte werden als None interpretiert.
                    if area_str in ["", ".", "e"] or yield_str in ["", ".", "e"]:
                        wert = None
                    else:
                        try:
                            # Ersetze Komma durch Punkt und wandle in float um.
                            area_val = float(area_str.replace(',', '.'))
                            yield_val = float(yield_str.replace(',', '.'))
                            # Bei 0-Anbaufläche nicht rechnen (Division durch 0 vermeiden)
                            if area_val == 0:
                                wert = None
                            else:
                                wert = yield_val / area_val
                        except ValueError:
                            wert = None

                    # Speichere das Jahr und den (möglicherweise berechneten) t/ha-Wert.
                    data[state].append((jahr, wert))

    return data


if __name__ == "__main__":
    # Beispielhafte Nutzung:
    csv_path = "data/raw/41215-0010_de.csv"
    yield_dict = read_yield_data(csv_path)
    from pprint import pprint

    from src.data_processing.preprocess_data import preprocess_numeric
    pprint(preprocess_numeric(yield_dict))
