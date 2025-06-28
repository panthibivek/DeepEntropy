
import os
import sys
import torch
import pickle
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_NMR_dataset

if __name__=="__main__":
    n_bootstrap = 1000
    nmr_data = torch.load(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data/embeddings/nmr_merged.pt")
    softDis_data = torch.load(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data/embeddings/softdis_merged.pt")
    disprot_data = torch.load(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data/embeddings/disprot_merged.pt")

    data_nmr = load_NMR_dataset(filter="moderate")
    moderate_nmr_ids = [ele["ID"] for ele in data_nmr]

    nmr_embeddings = []
    nmr_plddt = []
    nmr_gscores = []
    nmr_gscores_masks = []
    for nmr_ele in nmr_data:
        if nmr_ele["ID"] in moderate_nmr_ids:
            emb = nmr_ele["embedding"].numpy()  # shape: (L, 1024)
            plddt = np.array(nmr_ele["plddt"], dtype=np.float32)  # shape: (L,)
            
            raw_gscores = nmr_ele["gscores"]
            mask = np.array([v is not None for v in raw_gscores], dtype=bool)
            cleaned_gscores = np.array([0.0 if v is None else v for v in raw_gscores], dtype=np.float32)

            if emb.shape[0] != plddt.shape[0] or emb.shape[0] != len(cleaned_gscores):
                print(f"Unmatching dimension!. Embedding dim: {emb.shape[0]} plddt dim: {plddt.shape[0]}")
                continue

            if sum(mask)/len(mask) > 0.8:
                nmr_embeddings.append(emb)
                nmr_plddt.append(plddt)
                nmr_gscores.append(cleaned_gscores)
                nmr_gscores_masks.append(mask)

    disprot_embeddings = []
    disprot_plddt = []
    disprot_disorder_values = []
    for idx, disprot_ele in enumerate(disprot_data):
        try:
            emb = disprot_ele["embedding"].numpy()  # shape: (L, 1024)
            plddt = np.array(disprot_ele["plddt"], dtype=np.float32)  # shape: (L,)
            disorder_value = np.array([disprot_ele["disorder_content"]], dtype=np.float32)  # scalar

            # Shape checks
            if emb.shape[0] != plddt.shape[0]:
                print(f"[{idx}] Shape mismatch: embedding L={emb.shape[0]}, plddt L={plddt.shape[0]}")
                continue
            if emb.shape[1] != 1024:
                print(f"[{idx}] Invalid embedding dim: got {emb.shape[1]}, expected 1024")
                continue
            
            disprot_embeddings.append(emb)
            disprot_plddt.append(plddt)
            disprot_disorder_values.append(disorder_value)

        except Exception as e:
            print(f"[{idx}] Error loading sample: {e}")


    softDis_embeddings = []
    softDis_plddt = []
    softDis_disorder_values = []
    for idx, softDis_ele in enumerate(softDis_data):
        try:
            emb = softDis_ele["embedding"].numpy()  # shape: (L, 1024)
            plddt = np.array(softDis_ele["plddt"], dtype=np.float32)  # shape: (L,)
            disorder = np.array(softDis_ele["soft_disorder_frequency"], dtype=np.float32)  # shape: (L,)

            # Shape checks
            if emb.shape[1] != 1024:
                print(f"[{idx}] Invalid embedding dimension: got {emb.shape[1]}, expected 1024")
                continue
            if emb.shape[0] != plddt.shape[0] or emb.shape[0] != disorder.shape[0]:
                print(f"[{idx}] Length mismatch: embedding L={emb.shape[0]}, plddt L={plddt.shape[0]}, disorder L={disorder.shape[0]}")
                continue

            softDis_embeddings.append(emb)
            softDis_plddt.append(plddt)
            softDis_disorder_values.append(disorder)

        except Exception as e:
            print(f"[{idx}] Error loading sample: {e}")



    disprot_masks = [np.ones_like(plddt, dtype=bool) for plddt in disprot_plddt]
    softDis_masks = [np.ones_like(plddt, dtype=bool) for plddt in softDis_plddt]
    combined_data = []

    for idx in range(len(nmr_embeddings)):
        combined_data.append((
            nmr_embeddings[idx],
            nmr_plddt[idx] / 100,
            nmr_gscores[idx],
            nmr_gscores_masks[idx],
            "g_scores"
        ))

    for idx in range(len(disprot_embeddings)):
        combined_data.append((
            disprot_embeddings[idx],
            disprot_plddt[idx] / 100,
            disprot_disorder_values[idx],
            disprot_masks[idx],
            "disprot_disorder"
        ))

    for idx in range(len(softDis_embeddings)):
        combined_data.append((
            softDis_embeddings[idx],
            softDis_plddt[idx] / 100,
            softDis_disorder_values[idx],
            softDis_masks[idx],
            "softdis_disorder"
        ))

    random.seed(42)
    random.shuffle(combined_data)

    labels = [entry[-1] for entry in combined_data]

    train_val, test_data = train_test_split(
        combined_data,
        test_size=0.10,
        random_state=42,
        stratify=labels
    )

    train_data, val_data = train_test_split(
        train_val,
        test_size=2/9,
        random_state=42,
        stratify=[entry[-1] for entry in train_val]
    )

    print(f"total: {len(combined_data)}")
    print(f"train: {len(train_data)} ({len(train_data)/len(combined_data):.2%})")
    print(f"val: {len(val_data)} ({len(val_data)/len(combined_data):.2%})")
    print(f"test: {len(test_data)} ({len(test_data)/len(combined_data):.2%})")


    task_data_dict = defaultdict(list)
    for entry in test_data:
        task_data_dict[entry[-1]].append(entry)


    # Generate and save bootstrap samples
    for data_type in task_data_dict.keys():
        dir_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / f"data/bootstrap_test_sets/{data_type}"
        os.makedirs(dir_path, exist_ok=True)

        N = len(task_data_dict[data_type])
        for i in range(n_bootstrap):
            sample_indices = random.choices(range(N), k=N)
            with open(dir_path / f"indices_{i}.pkl", "wb") as f:
                pickle.dump(sample_indices, f)

