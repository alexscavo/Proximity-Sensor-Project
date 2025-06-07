import urllib.request
import zipfile
import os

# URLs from the MAN GitHub
urls = [
    "https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/mini/man-truckscenes_metadata_v1.0-mini.zip",
    "https://man-truckscenes.s3.eu-central-1.amazonaws.com/release/mini/man-truckscenes_sensordata_v1.0-mini.zip"
]

download_dir = "data/raw_downloads/"
extract_to = "data/mini_dataset/"

os.makedirs(download_dir, exist_ok=True)
os.makedirs(extract_to, exist_ok=True)

# Download and unzip
for url in urls:
    filename = url.split("/")[-1]
    filepath = os.path.join(download_dir, filename)

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)

    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

print("Done.")
