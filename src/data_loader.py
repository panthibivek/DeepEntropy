
import os
import json
import pathlib
from datasets import load_dataset, load_from_disk


def load_NMR_dataset(realtive_path : str = "data/NMR_disorder_dataset/2024-05-09", filter : str = "unfiltered"):
    parent_dir = pathlib.Path(__file__).parent.parent
    path_to_data = parent_dir / realtive_path

    with open(path_to_data / (filter + ".json"), 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def load_DisProt_dataset(realtive_path : str = "data/DisProt/DisProt.json"):
    parent_dir = pathlib.Path(__file__).parent.parent
    path_to_data = parent_dir / realtive_path

    with open(path_to_data, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data["data"]


def load_softDis_dataset(realtive_path : str = "data/SoftDis"):
    parent_dir = pathlib.Path(__file__).parent.parent
    path_to_data = parent_dir / realtive_path

    if not path_to_data.is_dir():
        dataset = load_dataset("CQSB/SoftDis")
        dataset.save_to_disk(path_to_data)
    else:
        dataset = load_from_disk(path_to_data)
    return dataset['train']



if __name__=="__main__":
    NMR = load_NMR_dataset()
    print(len(NMR))

    disProt = load_DisProt_dataset()
    print(len(disProt))

    softDis = load_softDis_dataset()
    print(len(softDis))
