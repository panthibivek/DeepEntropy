
import os
import sys
import json
import pathlib
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_NMR_dataset, load_DisProt_dataset, load_softDis_dataset


def get_common_protein_seq(save_dirpath : str = "data/intersection/", force_redo_ : bool = False):
    parent_dir = pathlib.Path(__file__).parent.parent
    path_to_data = parent_dir / save_dirpath
    os.makedirs(path_to_data, exist_ok=True)

    if not (path_to_data / "intersection.json").is_file() or force_redo_:
        NMR = load_NMR_dataset()
        disProt = load_DisProt_dataset()
        softDis = load_softDis_dataset()

        NMR_prot_seqs = [item["seq"] for item in NMR]
        disProt_prot_seqs = [item["sequence"] for item in disProt]
        softDis_prot_seqs = [item["sequence"] for item in softDis]

        NMR_prot_seqs = set(NMR_prot_seqs)
        disProt_prot_seqs = set(disProt_prot_seqs)
        softDis_prot_seqs = set(softDis_prot_seqs)
        intersection_seq = NMR_prot_seqs & disProt_prot_seqs & softDis_prot_seqs

        intersection_seq_json = {
            "common_sequences": list(intersection_seq)
        }
        with open(path_to_data / "intersection.json", "w") as file:
            json.dump(intersection_seq_json, file)

    else:
        with open(path_to_data / "intersection.json", "r") as file:
            intersection_seq_json = json.load(file)

        intersection_seq = intersection_seq_json["common_sequences"]

    print(f"Total commom seqs. : {len(intersection_seq)}") 
    return intersection_seq


def convert_to_fasta(sequences : list, save_dirpath : str = "data/intersection/"):
    parent_dir = pathlib.Path(__file__).parent.parent
    path_to_data = parent_dir / save_dirpath
    os.makedirs(path_to_data, exist_ok=True)

    records = []
    for i, seq in enumerate(sequences, start=1):
        record = SeqRecord(Seq(seq), id=f"intersection_protein_seq_{i}", description="")
        records.append(record)

    with open(path_to_data / "intersection.fasta", "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")


if __name__=="__main__":
    sequences = get_common_protein_seq()
    convert_to_fasta(sequences)