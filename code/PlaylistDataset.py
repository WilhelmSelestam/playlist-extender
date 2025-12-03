import torch
from torch.utils.data import Dataset, DataLoader

class PlaylistDataset(Dataset):
    def __init__(self, processed_file_path):
        self.data = torch.load(processed_file_path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        # Get the full row: [SOS, Song1, Song2, Song3, EOS, PAD...]
        full_seq = self.data[idx]
        
        # For "Next Song Prediction" (Causal Modeling), we shift the data.
        # Input:  [SOS, Song1, Song2]
        # Target: [Song1, Song2, EOS]
        
        x = full_seq[:-1] # Everything except the last token
        y = full_seq[1:]  # Everything except the first token
        
        return x, y





# dataset = PlaylistDataset("processed_dataset.pt")

# # num_workers=0 is usually fastest because data is already in RAM
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# # 3. Simulate a Training Loop
# print("\n--- Starting Training Loop Simulation ---")
# for batch_idx, (inputs, targets) in enumerate(dataloader):
    
#     print(f"Batch {batch_idx}:")
#     print(f"Input Shape: {inputs.shape}")   # Should be [32, 49] (if max_len was 50)
#     print(f"Target Shape: {targets.shape}") # Should be [32, 49]
    
#     # Example: Feed 'inputs' to model, calculate loss vs 'targets'
#     # logits = model(inputs)
#     # loss = cross_entropy(logits, targets)
    
#     break # Just show one batch


# import torch
# import torch.nn as nn

# # CONFIGURATION
# vocab_size = 229878  # Number of songs in vocab.json
# embedding_dim = 256

# # THE LAYER
# # This creates a matrix of size 50,000 x 256 filled with random numbers
# embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# # --- HOW IT WORKS ---

# # 1. Input: A batch of Song IDs (integers)
# # Let's say we have a playlist of 3 songs with IDs [10, 42, 5]
# input_ids = torch.LongTensor([10, 42, 5])

# # 2. Forward Pass
# # The layer looks up row 10, row 42, and row 5 in the matrix
# vectors = embedding_layer(input_ids)

# print(vectors.shape) 
# # Output: torch.Size([3, 256])
# # We now have 3 dense vectors, each with 256 coordinates.