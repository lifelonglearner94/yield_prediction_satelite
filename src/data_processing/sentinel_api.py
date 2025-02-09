import os
import json
from datetime import datetime, timedelta
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    MimeType,
    BBox,
    CRS,
    DataCollection,
    bbox_to_dimensions,
)
from PIL import Image
from dotenv import load_dotenv
import cv2
import numpy as np

class SentinelHubImageDownloader:
    def __init__(self,
                 output_folder=os.path.join("data", "germany_satelite_images_wo_cl"),
                 states_bbox=None,
                 years=None,
                 states=None,
                 num_images_per_year=10,
                 eval_script=None):
        """
        Initialisiert den Downloader.

        Args:
            output_folder (str): Ordner, in dem die heruntergeladenen Bilder gespeichert werden.
            states_bbox (dict): Optional ein Dictionary mit Bounding Boxes für die Bundesländer.
            years (list): Liste der Jahre, für die Bilder heruntergeladen werden sollen.
            states (list): Optional eine Liste von Bundesländern; falls angegeben,
                           werden nur diese verarbeitet.
            num_images_per_year (int): Anzahl der Bilder pro Jahr.
            eval_script (str): Optionaler Evalscript-String.
        """
        load_dotenv()
        self.num_images = num_images_per_year
        # Konfiguriere Sentinel Hub mit den in .env geladenen Zugangsdaten.
        self.config = SHConfig()
        if not self.config.sh_client_id or not self.config.sh_client_secret:
            raise ValueError("Bitte gib deine Sentinel Hub Zugangsdaten (Client ID/Secret) in deiner .env oder über SHConfig an.")

        # Standard-Bounding-Boxes für die Bundesländer, falls keiner übergeben wird
        default_states_bbox = {
            "Baden-Württemberg": (7.5113934084, 47.5338000528, 10.4918239143, 49.7913749328),
            "Bayern": (9.766846, 47.332047, 14.095459, 50.576029),
            "Brandenburg": (11.315918, 51.491543, 14.897461, 53.441526),
            "Hamburg": (9.593811, 53.390740, 10.405426, 53.742970),
            "Hessen": (7.591553, 49.439524, 10.404053, 51.719582),
            "Mecklenburg-Vorpommern": (10.585327, 53.067741, 14.749146, 54.587682),
            "Niedersachsen": (6.536865, 51.747520, 11.579590, 53.864998),
            "Nordrhein-Westfalen": (5.938110, 50.464345, 9.179077, 52.498914),
            "Rheinland-Pfalz": (6.020508, 48.990947, 8.668213, 50.897539),
            "Saarland": (6.358337, 49.125998, 7.391052, 49.646211),
            "Sachsen": (11.876221, 50.264625, 15.073242, 51.631153),
            "Sachsen-Anhalt": (10.513916, 50.944560, 12.568359, 53.100022),
            "Schleswig-Holstein": (7.910156, 53.327341, 11.211548, 54.961168),
            "Thüringen": (9.497681, 50.222579, 12.617798, 51.522036)
        }
        self.states_bbox = states_bbox or default_states_bbox

        # Falls eine Liste von Bundesländern übergeben wurde, filtere das Dictionary.
        if states is not None:
            self.states_bbox = {state: bbox for state, bbox in self.states_bbox.items() if state in states}

        # Standard-Jahre, falls keine übergeben wurden.
        self.years = years or [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Datei, in der das Dictionary mit den Bildpfaden gespeichert wird.
        self.json_filepath = os.path.join("data", "raw", "image_paths.json")
        # Lade bestehendes JSON, falls vorhanden, ansonsten initialisiere ein leeres Dictionary.
        if os.path.exists(self.json_filepath):
            try:
                with open(self.json_filepath, "r") as f:
                    self.image_paths = json.load(f)
                print(f"Loaded existing image paths from {self.json_filepath}")
            except Exception as e:
                print(f"Error loading JSON file: {e}")
                self.image_paths = {}
        else:
            self.image_paths = {}

        # Evalscript für True-Color-Bilder (Skalierung in 0-255)
        self.evalscript_true_color = eval_script or """//VERSION=3
            function setup() {
                return {
                    input: ["B02", "B03", "B04", "SCL"],
                    output: { bands: 3 }
                };
            }

            function evaluatePixel(sample) {
                // Falls SCL (Scene Classification) auf Wolken/Konfidenz-Klassen (z.B. 8,9,10) hinweist,
                // wird ein roter Pixel (zur späteren Erkennung) zurückgegeben.
                if ([8, 9, 10].includes(sample.SCL)) {
                    return [1, 0, 0];
                    // Alternativ könntest du hier auch [255, 0, 0] zurückgeben, wenn du helles Rot möchtest.
                } else {
                    return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
                }
            }
            """

    @staticmethod
    def auto_red_mask(image_bgr):
        """
        Erzeugt eine Binärmaske, in der Bereiche mit sehr rotem Farbton (hier als Wolkenmarker genutzt)
        den Wert 255 erhalten.

        Args:
            image_bgr: Eingabebild im BGR-Format.

        Returns:
            mask: 8-Bit Maske, in der die roten Bereiche (potenzielle Wolken) als 255 markiert sind.
        """
        # In den HSV-Farbraum konvertieren
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Schwellwerte für "sehr rot" (untere und obere Rottöne)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        return mask

    @staticmethod
    def inpaint_with_opencv(image_bgr, mask, inpaint_radius=3, method=cv2.INPAINT_TELEA):
        """
        Führt Inpainting mithilfe von OpenCV durch, um die Bereiche zu ersetzen, die in der Maske markiert sind.

        Args:
            image_bgr: Originalbild im BGR-Format.
            mask: 8-Bit Maske, bei der 255 die zu ersetzenden Bereiche markiert.
            inpaint_radius: Radius für den Inpainting-Algorithmus.
            method: cv2.INPAINT_TELEA oder cv2.INPAINT_NS.

        Returns:
            inpainted_bgr: Das inpaintete Bild im BGR-Format.
        """
        return cv2.inpaint(image_bgr, mask, inpaint_radius, method)

    def download_images(self):
        # Durchlaufe die (gegebenenfalls gefilterten) Bundesländer
        for state, bbox_coords in self.states_bbox.items():
            state_folder = os.path.join(self.output_folder, state)
            os.makedirs(state_folder, exist_ok=True)

            # Stelle sicher, dass das Bundesland im image_paths-Dictionary existiert.
            if state not in self.image_paths:
                self.image_paths[state] = {}

            bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
            # Starte mit einer initialen Auflösung (in Metern)
            resolution = 30
            size = bbox_to_dimensions(bbox, resolution=resolution)
            width, height = size

            # Falls Breite oder Höhe 2500px überschreiten, passe die Auflösung an.
            if width > 2500 or height > 2500:
                scale_factor = max(width / 2500, height / 2500)
                resolution = resolution * scale_factor
                size = bbox_to_dimensions(bbox, resolution=resolution)
                print(f"Adjusted resolution to {resolution} m for {state} to meet API limits. New dimensions: {size}")

            # Schleife über die Jahre
            for year in self.years:
                year_str = str(year)
                year_folder = os.path.join(state_folder, year_str)
                os.makedirs(year_folder, exist_ok=True)

                if year_str not in self.image_paths[state]:
                    self.image_paths[state][year_str] = []

                # Definiere den Zeitraum: 1. März bis 1. Juli des Jahres
                period_start = datetime(year, 3, 1)
                period_end = datetime(year, 7, 1)
                total_days = (period_end - period_start).days

                # Generiere gleichmäßig verteilte Zieltermine innerhalb des Zeitraums.
                for i in range(self.num_images):
                    fraction = i / (self.num_images - 1) if self.num_images > 1 else 0
                    target_date = period_start + timedelta(days=fraction * total_days)
                    # Setze die Uhrzeit auf 12:00 Uhr
                    target_date = target_date.replace(hour=12, minute=0, second=0, microsecond=0)
                    date_str = target_date.strftime("%Y-%m-%d")
                    filename = f"{state}_{date_str}_true_color.png"
                    filepath = os.path.join(year_folder, filename)

                    # Falls die Datei bereits existiert, überspringe den Download.
                    if os.path.exists(filepath):
                        if filepath not in self.image_paths[state][year_str]:
                            self.image_paths[state][year_str].append(filepath)
                        print(f"Image already exists for {state} on {date_str}. Skipping download.")
                        continue

                    # Definiere ein Zeitfenster um den Zieltermin (±7 Tage).
                    time_from = (target_date - timedelta(days=7)).isoformat()
                    time_to   = (target_date + timedelta(days=7)).isoformat()
                    time_interval = (time_from, time_to)

                    # Baue die Request für Sentinel-2 L2A True-Color-Daten.
                    request = SentinelHubRequest(
                        evalscript=self.evalscript_true_color,
                        input_data=[
                            SentinelHubRequest.input_data(
                                data_collection=DataCollection.SENTINEL2_L2A,
                                time_interval=time_interval,
                                mosaicking_order="leastCC"
                            )
                        ],
                        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                        bbox=bbox,
                        size=size,
                        config=self.config
                    )

                    try:
                        images = request.get_data()
                        if images:
                            image = images[0]  # Verwende das erste (oder einzige) Bild

                            # --- Wolkenentfernung mittels Inpainting ---
                            # Da OpenCV standardmäßig mit BGR arbeitet, konvertiere das Bild:
                            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            # Erstelle eine Maske der roten Bereiche (die Wolken markieren sollen):
                            mask = SentinelHubImageDownloader.auto_red_mask(image_bgr)
                            # Führe das Inpainting durch, um die als Wolken markierten Bereiche zu ersetzen:
                            inpainted_bgr = SentinelHubImageDownloader.inpaint_with_opencv(
                                image_bgr, mask, inpaint_radius=3, method=cv2.INPAINT_TELEA)
                            # Konvertiere zurück ins RGB-Format:
                            inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
                            # -------------------------------------------------

                            im = Image.fromarray(inpainted_rgb)
                            im.save(filepath)
                            print(f"Saved cloud-free image for {state} on {date_str} to {filepath}")
                            # Füge den Pfad zum Dictionary hinzu.
                            self.image_paths[state][year_str].append(filepath)
                        else:
                            print(f"No image returned for {state} on {target_date.isoformat()}")
                    except Exception as e:
                        print(f"Error processing {state} on {target_date.isoformat()}: {e}")

        # Speichere das aktualisierte Dictionary als JSON in den Ordner data/raw.
        raw_folder = os.path.join("data", "raw")
        os.makedirs(raw_folder, exist_ok=True)
        try:
            with open(self.json_filepath, "w") as f:
                json.dump(self.image_paths, f, indent=4)
            print(f"Image paths dictionary saved to {self.json_filepath}")
        except Exception as e:
            print(f"Error saving image paths dictionary: {e}")

if __name__ == "__main__":
    # Beispiel: Download für Bayern und Thüringen für 2020 mit 10 Bildern pro Jahr.
    downloader = SentinelHubImageDownloader(states=["Bayern"],
                                            years=[2020],
                                            num_images_per_year=10)
    downloader.download_images()
