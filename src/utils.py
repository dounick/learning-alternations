"""Some general purpose utility functions."""

import csv
import json
import unicodedata
from collections import defaultdict


def read_file(path):
    return [
        unicodedata.normalize("NFKD", i.strip())
        for i in open(path, encoding="utf-8").readlines()
        if i.strip() != ""
    ]


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def read_csv_dict(path):
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            data.append(line)
    return data


def write_csv_dict(data, path):
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        for line in data:
            writer.writerow(line)
