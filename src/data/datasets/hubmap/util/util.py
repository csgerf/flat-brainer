from typing import Any, Dict, List

import pandas as pd
import json


LABEL_TYPES = ['glomerulus', 'blood_vessel', 'unsure']


def read_raw_train_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    return df


def read_raw_test_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    return df


def read_json_poly_list(train_json_file: str) -> List[Dict[str, Any]]:
    with open(train_json_file, 'r') as json_file:
        json_labels = [json.loads(line) for line in json_file]
    return json_labels
