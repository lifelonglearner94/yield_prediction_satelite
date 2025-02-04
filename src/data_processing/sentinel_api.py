import os
# import datetime
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


# --- Set up your Sentinel Hub configuration ---
load_dotenv()
config = SHConfig()
# Make sure to set your client ID and client secret either here or via environment variables
if not config.sh_client_id or not config.sh_client_secret:
    print("Warning: Please provide your Sentinel Hub credentials (client ID/secret) in the SHConfig.")

# --- Define an evalscript for a true color image ---
# This script returns Sentinel-2 L2A bands B04 (red), B03 (green), B02 (blue)
evalscript_true_color = """
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

# --- Define a dictionary of German states with approximate bounding boxes ---
# Coordinates are given as (min_lon, min_lat, max_lon, max_lat)
states_bbox = {
    "BadenWürttemberg": (7.5, 47.3, 10.5, 49.0),
    "Bavaria": (10.0, 47.0, 13.0, 50.0),
    "Berlin": (13.1, 52.3, 13.7, 52.7),
    "Hamburg": (9.8, 53.3, 10.3, 53.8),
    "NorthRhineWestphalia": (6.0, 50.0, 8.5, 52.5),
    # Add additional states as needed...
}

# --- Define the list of years (or time intervals) of interest ---
years = [2019, 2020, 2021, 2022]

# --- Output folder to save images ---
output_folder = "germany_satelite_images"
os.makedirs(output_folder, exist_ok=True)

# --- Loop over each state and year to request and save the image ---
for state, bbox_coords in states_bbox.items():
    # Create a bounding box object for the state (in WGS84)
    bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
    # Set the desired resolution (e.g., 30 m per pixel)
    resolution = 30
    size = bbox_to_dimensions(bbox, resolution=resolution)

    for year in years:
        # Define a time interval for the request.
        # For example, here we choose mid-July (often with fewer clouds).
        time_interval = (f"{year}-07-15", f"{year}-07-25")

        # Build the Sentinel Hub Process API request
        request = SentinelHubRequest(
            evalscript=evalscript_true_color,
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
            config=config
        )

        try:
            images = request.get_data()
            if images:
                image = images[0]  # The request returns a list (even if one image is available)
                # Save the image to disk with a filename that includes the state name and year
                filename = f"{state}_{year}_true_color.png"
                filepath = os.path.join(output_folder, filename)
                im = Image.fromarray(image)
                im.save(filepath)
                print(f"Saved image for {state} in {year} to {filepath}")
            else:
                print(f"No image returned for {state} in {year}.")
        except Exception as e:
            print(f"Error processing {state} in {year}: {e}")
