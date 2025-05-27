
import os
import sys
import json
import pathlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_NMR_dataset, load_softDis_dataset

def add_unitprot_ids_NMR(destination_filepath: str = "data/alphafold2/NMR/NMR_with_ids.json"):
    NMR = load_NMR_dataset()

    parent_dir = pathlib.Path(__file__).parent.parent
    path_to_unitprot_id = parent_dir / "data/unitprot_ids/uniprot_sprot_id_maps.json"
    with open(path_to_unitprot_id, "r") as file:
        unitprot_id_list = json.load(file)

    seq_to_accession = {entry["sequence"]: entry["accession"] for entry in unitprot_id_list}

    results = []
    for data_ele in NMR:
        accession = seq_to_accession.get(data_ele["seq"])
        if accession:
            data_ele["accession"] = accession
            results.append(data_ele)

    path_to_data = parent_dir / destination_filepath
    with open(path_to_data, "w") as file:
        json.dump(results, file)

    print("Done :)")
    return results


def add_unitprot_ids_softDis(destination_filepath: str = "data/alphafold2/softDis/softDis_with_ids.json"):
    softDis = load_softDis_dataset()

    parent_dir = pathlib.Path(__file__).parent.parent
    path_to_unitprot_id = parent_dir / "data/unitprot_ids/uniprot_sprot_id_maps.json"
    with open(path_to_unitprot_id, "r") as file:
        unitprot_id_list = json.load(file)

    seq_to_accession = {entry["sequence"]: entry["accession"] for entry in unitprot_id_list}

    results = []
    for data_ele in softDis:
        accession = seq_to_accession.get(data_ele["sequence"])
        if accession:
            data_ele["accession"] = accession
            results.append(data_ele)

    path_to_data = parent_dir / destination_filepath
    with open(path_to_data, "w") as file:
        json.dump(results, file)

    print("Done :)")
    return results

if __name__ == "__main__":
    results = add_unitprot_ids_NMR()
    print(len(results))

    results = add_unitprot_ids_softDis()
    print(len(results))
