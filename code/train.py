import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

from MusicTransformer import MusicTransformer
from torch.utils.data import DataLoader
from PlaylistDataset import PlaylistDataset


CONFIG = {
    "vocab_file": "./vocab.json",
    "data_file": "./processed_dataset.pt",
    "save_model_path": "./music_transformer_v1.pth",
    "batch_size": 32,
    "d_model": 256,
    "nhead": 8, # Attention heads
    "num_layers": 4, # Transformer blocks
    "dropout": 0.1,
    "lr": 0.0005, # Learning Rate)
    "epochs": 5, # How many times to see the whole dataset
    "log_interval": 100, # Print loss every N batches
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def train():
    print(f"--- Training Configuration ---")
    print(f"Device: {CONFIG['device']}")
    

    print("Loading vocabulary...")
    with open(CONFIG['vocab_file'], 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"Vocabulary Size: {vocab_size}")

    print("Loading dataset...")
    dataset = PlaylistDataset(CONFIG['data_file'])
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=0,  # Set to 2 or 4 if on Linux/Mac for faster loading
        pin_memory=True # Speeds up transfer to GPU
    )

    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("Starting training...")
    model.train()

    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])

            optimizer.zero_grad()

            logits = model(inputs)

            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update the progress bar description with current loss
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

    print(f"Training complete. Saving model to {CONFIG['save_model_path']}...")
    torch.save(model.state_dict(), CONFIG['save_model_path'])
    print("Done.")

if __name__ == "__main__":
    train()








# import torch.optim as optim
# import torch
# from MusicTransformer import MusicTransformer
# import torch.nn as nn
# from PlaylistDataset import PlaylistDataset
# from torch.utils.data import DataLoader


# DATA_DIR = "C:/Users/seles/playlist-extender/dataset/data"
# VOCAB_FILE = "./vocab.json"

# VOCAB_SIZE = 229878
# D_MODEL = 256
# NHEAD = 8 # Number of attention heads
# NUM_LAYERS = 4 # Number of transformer blocks
# LR = 0.001 # Learning Rate
# EPOCHS = 3
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset = PlaylistDataset("./processed_dataset.pt")
# # num_workers=0 is usually fastest because data is already in RAM
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)


# # Initialize Model
# model = MusicTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS)
# model = model.to(DEVICE)

# # Define Optimizer and Loss
# optimizer = optim.AdamW(model.parameters(), lr=LR)
# criterion = nn.CrossEntropyLoss(ignore_index=0) # We ignore the <PAD> token (ID 0)

# print(f"Training on {DEVICE}...")

# # Training Loop
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
    
#     for batch_idx, (inputs, targets) in enumerate(dataloader):
#         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
#         # Zero gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         logits = model(inputs) 
#         # logits shape: [Batch, Seq_Len, Vocab]
        
#         # Reshape for CrossEntropyLoss
#         # Flatten the batch and sequence dimensions
#         # logits -> [Batch * Seq_Len, Vocab]
#         # targets -> [Batch * Seq_Len]
#         loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        
#         # Backward pass
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
        
#         if batch_idx % 100 == 0:
#             print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

#     print(f"Epoch {epoch} Complete. Average Loss: {total_loss / len(dataloader):.4f}")

# # 5. Save the trained model
# torch.save(model.state_dict(), "music_transformer.pth")
# print("Model saved!")