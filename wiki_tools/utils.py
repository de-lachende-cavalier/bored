import json
import os


def load_checkpoint(filepath):
    """Load existing entities from a JSON checkpoint file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as infile:
            return json.load(infile)
    return {}


def save_checkpoint(filepath, data):
    """Save entities to a JSON checkpoint file."""
    with open(filepath, "w") as outfile:
        json.dump(data, outfile)
