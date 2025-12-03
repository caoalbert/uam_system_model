import os
import urllib.request

DATA_URL = "https://storage.googleapis.com/airport-data-lax/LAX_ind.csv"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "LAX_ind.csv")

JFK_DATA_URL = "https://storage.googleapis.com/airport-data-lax/JFK_ind.csv"
JFK_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "JFK_ind.csv")


def ensure_data():
    if not os.path.exists(DATA_PATH):
        print(f"LAX data not found at {DATA_PATH}. Downloading from GCS...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Download complete.")

    if not os.path.exists(JFK_DATA_PATH):
        print(f"JFK data not found at {JFK_DATA_PATH}. Downloading from GCS...")
        os.makedirs(os.path.dirname(JFK_DATA_PATH), exist_ok=True)
        urllib.request.urlretrieve(JFK_DATA_URL, JFK_DATA_PATH)
        print("Download complete.")


ensure_data()
