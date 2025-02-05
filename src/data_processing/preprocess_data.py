import pandas as pd
import numpy as np

def preprocess_numeric(data):
    processed_data = {}

    for region, values in data.items():
        years, numbers = zip(*values)  # Jahreszahlen und Werte trennen
        df = pd.DataFrame({'Years': years, 'Values': numbers})

        # Überprüfe, ob alle Werte fehlen, und entferne das Bundesland
        if df['Values'].isnull().all():
            continue

        # Mittelwert der benachbarten Werte berechnen
        for i in range(len(df)):
            if pd.isnull(df['Values'][i]):
                # Finde die benachbarten Werte (vorher und nachher)
                prev_value = df['Values'][i-1] if i > 0 else np.nan
                next_value = df['Values'][i+1] if i < len(df)-1 else np.nan

                # Berechne den Mittelwert der benachbarten Werte (ignoriere NaNs)
                valid_values = [v for v in [prev_value, next_value] if not pd.isnull(v)]

                if valid_values:
                    df['Values'][i] = np.mean(valid_values)

        # Ursprüngliches Format wiederherstellen
        processed_data[region] = list(zip(df['Years'], df['Values']))

    return processed_data
