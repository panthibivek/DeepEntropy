import torch
from data_loader import load_NMR_dataset, load_DisProt_dataset, load_softDis_dataset
from Prot5 import Prot5Embedder
import os
import pathlib
from typing import List, Dict, Any
import math


class ProteinEmbeddingPipeline:
    def __init__(self, batch_size=4):
        self.embedder = Prot5Embedder(batch_size=batch_size)
        self.temp_dir = pathlib.Path(__file__).parent.parent / "temp_embeddings"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def validate_embeddings(self, sequences: List[str], embeddings: List[torch.Tensor]) -> bool:
        """Validate that embeddings match sequence lengths"""
        for seq, emb in zip(sequences, embeddings):
            if len(seq) != emb.shape[0]:
                print(f"\nWarning: Embedding shape mismatch!")
                print(f"Sequence length: {len(seq)}")
                print(f"Embedding shape: {emb.shape}")
                print(f"First 50 chars of sequence: {seq[:50]}...")
                return False
        return True

    def save_chunk_embeddings(self, protein_data: List[Dict[str, Any]], output_dir: pathlib.Path, chunk_idx: int, is_nmr: bool = False):
        """Helper function to save a chunk of protein data to a file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"chunk_{chunk_idx}.pt"
        
        # Validate embeddings before saving
        sequences = [item['seq'] if is_nmr else item['sequence'] for item in protein_data]
        embeddings = [item['embedding'] for item in protein_data]
        
        if not self.validate_embeddings(sequences, embeddings):
            print("Error: Embeddings do not match sequence lengths. Regenerating embeddings...")
            new_embeddings = self.embedder.get_embeddings(sequences)
            
            # Update protein data with correct embeddings
            for item, new_emb in zip(protein_data, new_embeddings):
                item['embedding'] = new_emb
        
        # Save with memory efficiency in mind
        try:
            torch.save(protein_data, output_file)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM occurs, try to save CPU tensors
                for item in protein_data:
                    if torch.is_tensor(item['embedding']):
                        item['embedding'] = item['embedding'].cpu()
                torch.save(protein_data, output_file)
            else:
                raise e
                
        print(f"Saved chunk {chunk_idx} with {len(protein_data)} proteins to {output_file}")
        return output_file

    def get_embeddings_with_cache(self, sequences: List[str], cache_file: pathlib.Path):
        """Get embeddings with caching to avoid regeneration"""
        if cache_file.exists():
            print(f"Loading embeddings from cache: {cache_file}")
            try:
                embeddings = torch.load(cache_file)
                # Validate cached embeddings
                if self.validate_embeddings(sequences, embeddings):
                    return embeddings
                else:
                    print("Cached embeddings are invalid. Regenerating...")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # If OOM occurs during loading, try to load on CPU
                    embeddings = torch.load(cache_file, map_location='cpu')
                    if self.validate_embeddings(sequences, embeddings):
                        return embeddings
                    else:
                        print("Cached embeddings are invalid. Regenerating...")
                else:
                    raise e
        
        print("Generating new embeddings...")
        embeddings = self.embedder.get_embeddings(sequences)
        
        # Validate before caching
        if not self.validate_embeddings(sequences, embeddings):
            raise ValueError("Generated embeddings do not match sequence lengths!")
        
        # Save to cache immediately
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(embeddings, cache_file)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM occurs during saving, move tensors to CPU
                embeddings = [emb.cpu() for emb in embeddings]
                torch.save(embeddings, cache_file)
            else:
                raise e
                
        print(f"Cached embeddings to: {cache_file}")
        return embeddings

    def process_dataset_in_chunks(self, 
                                data: List[Dict[str, Any]], 
                                sequences: List[str],
                                output_dir: str,
                                dataset_name: str,
                                chunk_size: int = 1000, is_nmr: bool = False):
        """Process any dataset in chunks and save embeddings"""
        # Create output directory
        parent_dir = pathlib.Path(__file__).parent.parent
        output_path = parent_dir / output_dir / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate number of chunks
        num_chunks = math.ceil(len(sequences) / chunk_size)
        print(f"Processing {len(sequences)} sequences in {num_chunks} chunks of size {chunk_size}")

        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(sequences))
            
            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (sequences {start_idx} to {end_idx})")
            
            # Get sequences and data for this chunk
            chunk_sequences = sequences[start_idx:end_idx]
            chunk_data = data[start_idx:end_idx]
            
            try:
                # Get embeddings with caching
                cache_file = self.temp_dir / dataset_name / f"chunk_{chunk_idx}.pt"
                embeddings = self.get_embeddings_with_cache(chunk_sequences, cache_file)
                
                # Create list of dictionaries containing metadata and embeddings
                protein_data = [
                    {**item, 'embedding': embedding}
                    for item, embedding in zip(chunk_data, embeddings)
                ]
                
                # Save this chunk
                self.save_chunk_embeddings(protein_data, output_path, chunk_idx, is_nmr)
                
                # Clear CUDA cache after each chunk
                if torch.cuda.is_available():
                    print("cuda available")
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error processing chunk {chunk_idx}, trying to free memory...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Reduce embedder batch size and retry
                    self.embedder.batch_size = max(1, self.embedder.batch_size // 2)
                    print(f"Reduced batch size to {self.embedder.batch_size}")
                    # Retry this chunk
                    chunk_idx -= 1
                    continue
                else:
                    raise e

        print(f"\nFinished processing {dataset_name} dataset. Embeddings saved in {output_path}")
        return output_path

    def process_NMR_dataset(self, 
                          relative_path="data/NMR_disorder_dataset/2024-05-09", 
                          filter="unfiltered",
                          output_dir="embeddings",
                          chunk_size=100, is_nmr=True):
        """Process NMR dataset in chunks"""
        data = load_NMR_dataset(relative_path, filter)
        sequences = [item['seq'] for item in data]
        return self.process_dataset_in_chunks(data, sequences, output_dir, "nmr", chunk_size, is_nmr)

    def process_DisProt_dataset(self, 
                              relative_path="data/DisProt/DisProt.json",
                              output_dir="embeddings",
                              chunk_size=100, is_nmr=False):
        """Process DisProt dataset in chunks"""
        data = load_DisProt_dataset(relative_path)
        sequences = [item['sequence'] for item in data]
        return self.process_dataset_in_chunks(data, sequences, output_dir, "disprot", chunk_size, is_nmr)

    def process_softDis_dataset(self, 
                              relative_path="data/SoftDis",
                              output_dir="embeddings",
                              chunk_size=100, is_nmr=False):
        """Process SoftDis dataset in chunks"""
        dataset = load_softDis_dataset(relative_path)
        sequences = dataset['sequence']
        
        # Convert dataset to list of dictionaries
        data = []
        for i in range(len(dataset)):
            protein_dict = {}
            for feature in dataset.features:
                protein_dict[feature] = dataset[feature][i]
            data.append(protein_dict)
        
        return self.process_dataset_in_chunks(data, sequences, output_dir, "softdis", chunk_size, is_nmr)


if __name__ == "__main__":
    # Start with a larger batch size for RTX 3060 6GB
    pipeline = ProteinEmbeddingPipeline(batch_size=4)
    
    # Use larger chunk size since we have caching
    chunk_size = 10  # Increased from 200 to process more sequences at once
    
    # # Process NMR dataset in chunks
    # print("Processing NMR dataset...")
    # nmr_output_dir = pipeline.process_NMR_dataset(chunk_size=chunk_size, is_nmr=True)
    
    # # Load and verify a chunk
    # chunk_file = next((nmr_output_dir / "chunk_0.pt").parent.glob("*.pt"))
    # chunk_data = torch.load("./embeddings/nmr/chunk_0.pt")
    # print(f"\nExample from NMR chunk {chunk_file.name}:")
    # print(f"Number of proteins in chunk: {len(chunk_data)}")
    # print(f"Example embedding shape: {chunk_data[0]['embedding'].shape}")
    
    # Process DisProt dataset in chunks
    print("\nProcessing DisProt dataset...")
    disprot_output_dir = pipeline.process_DisProt_dataset(chunk_size=chunk_size)
    
    # Load and verify a chunk
    chunk_file = next((disprot_output_dir / "chunk_0.pt").parent.glob("*.pt"))
    chunk_data = torch.load(chunk_file)
    print(f"\nExample from DisProt chunk {chunk_file.name}:")
    print(f"Number of proteins in chunk: {len(chunk_data)}")
    print(f"Example embedding shape: {chunk_data[0]['sequence']}")
    print(f"Example embedding shape: {chunk_data[0]['embedding'].shape}")
    
    # # Process SoftDis dataset in chunks
    # print("\nProcessing SoftDis dataset...")
    # softdis_output_dir = pipeline.process_softDis_dataset(chunk_size=chunk_size)
    
    # # Load and verify a chunk
    # chunk_file = next((softdis_output_dir / "chunk_0.pt").parent.glob("*.pt"))
    # chunk_data = torch.load(chunk_file)
    # print(f"\nExample from SoftDis chunk {chunk_file.name}:")
    # print(f"Number of proteins in chunk: {len(chunk_data)}")
    # print(f"Example embedding shape: {chunk_data[0]['embedding'].shape}")
    
    # print("\nAll datasets processed successfully!")
    # print("\nOutput directories:")
    # print(f"NMR: {nmr_output_dir}")
    # print(f"DisProt: {disprot_output_dir}")
    # print(f"SoftDis: {softdis_output_dir}") 