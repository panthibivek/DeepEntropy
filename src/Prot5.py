from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import os


# Set environment variable for expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class Prot5Embedder:
    def __init__(self, model_name="Rostlab/prot_t5_xl_half_uniref50-enc", device="cuda" if torch.cuda.is_available() else "cpu", batch_size=4):
        print(f"Using device: {device}")
        self.model = T5EncoderModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_cache=True  # Enable cache since we're only doing inference
        )
        
        # Disable gradient checkpointing since we're not training
        self.model.gradient_checkpointing_disable()
        
        # Enable memory efficient attention
        if hasattr(self.model.config, 'use_memory_efficient_attention'):
            self.model.config.use_memory_efficient_attention = True
            
        # Set model to evaluation mode
        self.model.eval()

        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.device = device
        self.batch_size = batch_size
        self.model.to(device)

    def preprocess_sequences(self, sequences):
        """Preprocess protein sequences by replacing rare/ambiguous amino acids with X and adding spaces"""
        return [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

    def process_batch(self, sequences):
        """Process a single batch of sequences"""
        processed_seqs = self.preprocess_sequences(sequences)
        
        # Batch encode all sequences together
        ids = self.tokenizer.batch_encode_plus(processed_seqs, add_special_tokens=True, 
                                            padding="longest", return_tensors="pt")
        input_ids = ids['input_ids'].to(self.device)
        attention_mask = ids['attention_mask'].to(self.device)
        
        # Generate embeddings for the entire batch
        with torch.no_grad():
            try:
                embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    if self.batch_size > 1:
                        print(f"OOM error, reducing batch size from {self.batch_size} to {self.batch_size // 2}")
                        self.batch_size = self.batch_size // 2
                        return self.process_batch(sequences)
                    else:
                        raise e
                else:
                    raise e
        
        # Extract embeddings for each sequence in the batch
        embeddings = []
        hidden_states = embedding_repr.last_hidden_state  # Shape: (batch_size, max_seq_len, hidden_dim)
        
        for i, seq in enumerate(processed_seqs):
            # Calculate actual sequence length (excluding padding)
            actual_length = len(sequences[i])
            
            # Extract embedding for this protein: [batch_idx, :actual_sequence_length, :]
            protein_emb = hidden_states[i, :actual_length]
            embeddings.append(protein_emb.cpu())
        
        return embeddings

    def get_embeddings(self, sequences, per_protein=False):
        """Get embeddings for a list of protein sequences in batches"""
        all_embeddings = []
        
        # Process sequences in batches
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i + self.batch_size]
            batch_embeddings = self.process_batch(batch_sequences)
            all_embeddings.extend(batch_embeddings)
            
            # Print progress
            if (i + 1) % (self.batch_size * 10) == 0:
                print(f"Processed {i + 1}/{len(sequences)} sequences")
        
        if per_protein:
            # For per-protein embeddings, average over sequence length
            all_embeddings = [emb.mean(dim=0) for emb in all_embeddings]
            return torch.stack(all_embeddings)
        
        return all_embeddings

    def get_embedding(self, sequence, per_protein=False):
        """Get embedding for a single protein sequence"""
        embeddings = self.get_embeddings([sequence], per_protein=per_protein)
        return embeddings[0]

