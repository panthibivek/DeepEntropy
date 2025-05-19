
import os
import re
import sys
import json
import glob
import pathlib
from Bio import SeqIO
from Bio.Seq import Seq
from typing import List, Dict, Any
from Bio.SeqRecord import SeqRecord

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_NMR_dataset, load_DisProt_dataset, load_softDis_dataset


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
    # NMR
    NMR = load_NMR_dataset()
    sequences = [{"id": obj["ID"], "seq": obj["seq"]} for obj in NMR]
    convert_to_fasta(
        sequences=sequences,
        filepath="data/alphafold2/NMR/unfiltered.fasta"
    )

    # disProt
    disProt = load_DisProt_dataset()
    sequences = [{"id": obj["disprot_id"], "seq": obj["sequence"]} for obj in disProt]
    convert_to_fasta(
        sequences=sequences,
        filepath="data/alphafold2/disProt/disProt.fasta"
    )

    # softDis
    softDis = load_softDis_dataset()
    sequences = [{"id": obj["id"], "seq": obj["sequence"]} for obj in softDis]
    convert_to_fasta(
        sequences=sequences,
        filepath="data/alphafold2/softDis/softDis.fasta"
    )

    # extracting plddt scores from the colab result dir
    extract_plddt_scores(
        source_dirpath="data/alphafold2/result_NMR_unfiltered", 
        destination_filepath="data/alphafold2/NMR/unfiltered_plddt.json"
    )
