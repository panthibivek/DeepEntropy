from transformers import T5Tokenizer, T5EncoderModel
import torch


model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model.eval()

sequence = "MALWMRLLPLLALLALWGPDPAAA..."

# Tokenize
inputs = tokenizer(
    sequence,
    return_tensors="pt",
    add_special_tokens=True
)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

embedding = embeddings[0].mean(dim=0)

print("Per-protein embedding shape:", embedding.shape)
