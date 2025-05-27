
import os
import re
import sys
import copy
import json
import glob
import pathlib
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from typing import List, Dict, Any
from Bio.SeqRecord import SeqRecord
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_NMR_dataset, load_DisProt_dataset, load_softDis_dataset


def get_plddt_from_alphafold_db(accession: str):
    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v2.pdb"
    pdb_txt = requests.get(pdb_url).text.splitlines()

    plddt = []
    for line in pdb_txt:
        if line.startswith("ATOM"):
            b = float(line[60:66].strip())
            if not plddt or len(plddt) < int(line[22:26].strip()):
                plddt.append(b)
    return plddt

def get_plddt_disProt_batch(max_workers=10, filepath: str = "data/alphafold2/disProt/disProt_plddt.json"):
    disProt = load_DisProt_dataset()
    disProt_copy = copy.deepcopy(disProt)
    accession_to_item = {item['acc']: item for item in disProt_copy}

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_accession = {
            executor.submit(get_plddt_from_alphafold_db, item['acc']): item['acc']
            for item in disProt_copy
        }

        for future in as_completed(future_to_accession):
            accession = future_to_accession[future]
            try:
                plddt = future.result()
                if plddt == []:
                    raise ValueError("Empty plddt!")
                item = accession_to_item[accession]
                item['plddt'] = plddt
                results.append(item)
            except Exception as e:
                print(f"Error processing {accession}: {e}")
    
    parent_dir = pathlib.Path(__file__).parent.parent
    path_to_data = parent_dir / filepath
    with open(path_to_data, "w") as file:
        json.dump(results, file)
    
    print(f"Count: {len(results)}")
    print("Done :)")
    return results


def get_plddt_NMR_batch(
        max_workers=10, 
        data_filepath: str = "data/alphafold2/NMR/NMR_with_ids.json", 
        destination_filepath: str = "data/alphafold2/NMR/NMR_plddt.json"
    ):
    
    parent_dir = pathlib.Path(__file__).parent.parent

    with open(parent_dir / data_filepath, 'r') as file:
        NMR_data_with_ids = json.load(file)

    accession_to_item = {item['accession']: item for item in NMR_data_with_ids}

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_accession = {
            executor.submit(get_plddt_from_alphafold_db, item['accession']): item['accession']
            for item in NMR_data_with_ids
        }

        for future in as_completed(future_to_accession):
            accession = future_to_accession[future]
            try:
                plddt = future.result()
                if plddt == []:
                    raise ValueError("Empty plddt!")
                item = accession_to_item[accession]
                item['plddt'] = plddt
                results.append(item)
            except Exception as e:
                print(f"Error processing {accession}: {e}")
    
    path_to_data = parent_dir / destination_filepath
    with open(path_to_data, "w") as file:
        json.dump(results, file)
    
    print(f"Count: {len(results)}")
    print("Done :)")
    return results

def get_plddt_softDis_batch(
        max_workers=10, 
        data_filepath: str = "data/alphafold2/softDis/softDis_with_ids.json", 
        destination_filepath: str = "data/alphafold2/softDis/softDis_plddt.json"
    ):
    
    parent_dir = pathlib.Path(__file__).parent.parent

    with open(parent_dir / data_filepath, 'r') as file:
        softDis_data_with_ids = json.load(file)

    accession_to_item = {item['accession']: item for item in softDis_data_with_ids}

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_accession = {
            executor.submit(get_plddt_from_alphafold_db, item['accession']): item['accession']
            for item in softDis_data_with_ids
        }

        for future in as_completed(future_to_accession):
            accession = future_to_accession[future]
            try:
                plddt = future.result()
                if plddt == []:
                    raise ValueError("Empty plddt!")
                item = accession_to_item[accession]
                item['plddt'] = plddt
                results.append(item)
            except Exception as e:
                print(f"Error processing {accession}: {e}")
    
    path_to_data = parent_dir / destination_filepath
    with open(path_to_data, "w") as file:
        json.dump(results, file)
    
    print(f"Count: {len(results)}")
    print("Done :)")
    return results

def convert_to_fasta(sequences : List[Dict[str, Any]], filepath : str):
    """
    sequences: 
        [
            {
                "seq": <protein sequences>,
                "id": <unique ID>
            },
        ]
    """
    if not pathlib.Path.is_file(pathlib.Path(filepath)):
        parent_dir = pathlib.Path(__file__).parent.parent
        path_to_data = parent_dir / filepath
        os.makedirs(pathlib.Path(path_to_data).parent, exist_ok=True)

        records = []
        for i, seq_json in enumerate(sequences, start=1):
            record = SeqRecord(Seq(seq_json["seq"]), id=seq_json["id"], description="")
            records.append(record)

        with open(path_to_data, "w") as output_handle:
            SeqIO.write(records, output_handle, "fasta")

def extract_plddt_scores(source_dirpath: str, destination_filepath: str):
    parent_dir = pathlib.Path(__file__).parent.parent
    path_to_data = parent_dir / source_dirpath
    path_to_destination_data = parent_dir / destination_filepath

    files = glob.glob(str(path_to_data) + '/*.json')

    content = []
    for file in files:
        filepath = os.path.basename(file)
        if "score" in filepath:
            match = re.match(r'^(.*?)_score', filepath)
            seq_id = match.group(1)
            
            with open(file, 'r') as file:
                seq_json = json.load(file)

            seq_dict = {
                "id": seq_id,
                "plddt": seq_json["plddt"]
            }
            content.append(seq_dict)
    
    with open(path_to_destination_data, 'w') as file:
        json.dump(content, file)
    print(f"Total number of samples: {len(content)}")    
    
if __name__=="__main__":
    # # NMR
    # NMR = load_NMR_dataset()
    # sequences = [{"id": obj["ID"], "seq": obj["seq"]} for obj in NMR]
    # convert_to_fasta(
    #     sequences=sequences,
    #     filepath="data/alphafold2/NMR/unfiltered.fasta"
    # )

    # # disProt
    # disProt = load_DisProt_dataset()
    # sequences = [{"id": obj["disprot_id"], "seq": obj["sequence"]} for obj in disProt]
    # convert_to_fasta(
    #     sequences=sequences,
    #     filepath="data/alphafold2/disProt/disProt.fasta"
    # )

    # # softDis
    # softDis = load_softDis_dataset()
    # sequences = [{"id": obj["id"], "seq": obj["sequence"]} for obj in softDis]
    # convert_to_fasta(
    #     sequences=sequences,
    #     filepath="data/alphafold2/softDis/softDis.fasta"
    # )

    # # extracting plddt scores from the colab result dir
    # extract_plddt_scores(
    #     source_dirpath="data/alphafold2/result_NMR_unfiltered", 
    #     destination_filepath="data/alphafold2/NMR/unfiltered_plddt.json"
    # )

    # # extracting intersection plddt scores from the colab result dir
    # extract_plddt_scores(
    #     source_dirpath="data/alphafold2/result_intersection", 
    #     destination_filepath="data/alphafold2/intersection/intersection_plddt.json"
    # )

    # # using alphafold db for disprot
    # # uses parallel processing with threads 
    # get_plddt_disProt_batch()
    # get_plddt_NMR_batch()
    # get_plddt_softDis_batch()

    pass
