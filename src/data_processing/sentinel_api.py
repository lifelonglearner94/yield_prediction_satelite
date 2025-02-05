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

class SentinelHubImageDownloader:
    def __init__(self,
                 output_folder=os.path.join("data", "germany_satelite_images"),
                 states_bbox=None,
                 years=None,
                 states=None):
        """
        Initializes the downloader.

        Args:
            output_folder (str): Folder where downloaded images will be saved.
            states_bbox (dict): Optional dictionary of state bounding boxes.
            years (list): List of years for which images will be downloaded.
            states (list): Optional list of state names to process. If provided,
                           only these states will be downloaded.
        """
        load_dotenv()

        # Configure Sentinel Hub with credentials loaded from .env
        self.config = SHConfig()
        if not self.config.sh_client_id or not self.config.sh_client_secret:
            raise ValueError("Please provide your Sentinel Hub credentials (client ID/secret) in your .env or via SHConfig.")

        # Default state bounding boxes if none provided
        default_states_bbox = {
            "Baden-Württemberg": (7.5113934084, 47.5338000528, 10.4918239143, 49.7913749328),
            "Bayern": (9.8778443239, 50.2042330625, 12.6531964048, 51.6490678544),
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

        # If a list of states is provided, filter the dictionary accordingly.
        if states is not None:
            self.states_bbox = {state: bbox for state, bbox in self.states_bbox.items() if state in states}

        # Default years if not provided
        self.years = years or [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        # File where the image paths dictionary is stored
        self.json_filepath = os.path.join("data", "raw", "image_paths.json")
        # Load existing JSON if available; otherwise, initialize an empty dictionary.
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

        # Evalscript for true color image (RGB)
        self.evalscript_true_color = """
        //VERSION=3
        function setup() {
          return {
            input: ["B02", "B03", "B04"],
            output: { bands: 3 }
          };
        }
        function evaluatePixel(sample) {
          return [sample.B04, sample.B03, sample.B02];
        }
        """

    def download_images(self):
        # Loop over the (possibly filtered) states
        for state, bbox_coords in self.states_bbox.items():
            state_folder = os.path.join(self.output_folder, state)
            os.makedirs(state_folder, exist_ok=True)

            # Ensure state exists in the image_paths dictionary
            if state not in self.image_paths:
                self.image_paths[state] = {}

            bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
            resolution = 30  # desired resolution (in meters)
            size = bbox_to_dimensions(bbox, resolution=resolution)

            # Loop over each year
            for year in self.years:
                year_str = str(year)
                if year_str not in self.image_paths[state]:
                    self.image_paths[state][year_str] = []
                # Define the period: March 1 to July 31 of the given year
                period_start = datetime(year, 3, 1)
                period_end = datetime(year, 7, 31)
                total_days = (period_end - period_start).days

                num_images = 25  # we want 25 pictures per year
                # Generate 25 target dates evenly spaced within the period.
                for i in range(num_images):
                    fraction = i / (num_images - 1) if num_images > 1 else 0
                    target_date = period_start + timedelta(days=fraction * total_days)
                    # Force the time to noon (12:00)
                    target_date = target_date.replace(hour=12, minute=0, second=0, microsecond=0)
                    date_str = target_date.strftime("%Y-%m-%d")
                    filename = f"{state}_{date_str}_true_color.png"
                    filepath = os.path.join(state_folder, filename)

                    # Check if the file already exists on disk.
                    if os.path.exists(filepath):
                        if filepath not in self.image_paths[state][year_str]:
                            self.image_paths[state][year_str].append(filepath)
                        print(f"Image already exists for {state} on {date_str}. Skipping download.")
                        continue  # Skip download if the image exists

                    # Define a narrow time window around noon (±30 minutes)
                    time_from = (target_date - timedelta(minutes=30)).isoformat()
                    time_to   = (target_date + timedelta(minutes=30)).isoformat()
                    time_interval = (time_from, time_to)

                    # Build the request using Sentinel-2 L2A data for true color imagery.
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
                            image = images[0]  # get the first (or only) returned image
                            im = Image.fromarray(image)
                            im.save(filepath)
                            print(f"Saved image for {state} on {date_str} to {filepath}")
                            # Add the filepath to the dictionary for this state and year.
                            self.image_paths[state][year_str].append(filepath)
                        else:
                            print(f"No image returned for {state} on {target_date.isoformat()}")
                    except Exception as e:
                        print(f"Error processing {state} on {target_date.isoformat()}: {e}")

        # Save the updated dictionary as JSON in the data/raw folder.
        raw_folder = os.path.join("data", "raw")
        os.makedirs(raw_folder, exist_ok=True)
        try:
            with open(self.json_filepath, "w") as f:
                json.dump(self.image_paths, f, indent=4)
            print(f"Image paths dictionary saved to {self.json_filepath}")
        except Exception as e:
            print(f"Error saving image paths dictionary: {e}")

# Example usage:
# To process only "Bayern" and "Thüringen":
# downloader = SentinelHubImageDownloader(states=["Bayern", "Thüringen"])
# downloader.download_images()
