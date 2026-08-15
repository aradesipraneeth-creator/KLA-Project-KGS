"""
Experiment Tracking & Logging Utilities.
"""

import os
import csv
import json
import time


class CSVLogger:
    """
    Appends epoch or step metrics to a CSV file.
    """
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            with open(filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row_dict):
        with open(self.filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            # Filter row to fieldnames
            filtered = {k: row_dict.get(k, "") for k in self.fieldnames}
            writer.writerow(filtered)


def save_summary_json(filepath, summary_dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
