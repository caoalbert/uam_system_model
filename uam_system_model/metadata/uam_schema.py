import json
from pathlib import Path

class UAMSchema:
    def __init__(self, config_path=f"{Path(__file__).resolve().parent}/replica_metadata.json"):
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.mapping = config['mapping']

        # Dynamically create attributes for the mapped (clean) names
        # This prevents typos and hardcoding later in the script
        for raw_name, clean_name in self.mapping.items():
            # Use upper case for constants (e.g., TRIP_ID)
            setattr(self, clean_name.upper(), clean_name)