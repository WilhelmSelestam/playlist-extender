
# Data loading
# streaming
# shuffeling

# Tokenization
# songs -> numbers

# Embedding
# tokens -> vectors?, matrices?, tensors?, instantiated as noise, learned during training
# position

# Attention?????
# idk
# multi headed?

# Transformer?????

# Training
# backpropogation?

from SongTokenizer import SongTokenizer
import json
from stream_dataset import stream_dataset



DATA_DIR = "C:/Users/seles/playlist-extender/dataset/data"
VOCAB_FILE = "./vocab.json"



# dataset_generator = stream_dataset(DATA_DIR)
# tokenizer = SongTokenizer(min_freq=20)
# tokenizer.fit(dataset_generator)
# tokenizer.save(VOCAB_FILE)


# f = open('./dataset/data/mpd.slice.0-999.json')
# js = f.read()
# f.close()
# data = json.loads(js)
  
# data = data['playlists']


# tokenizer = SongTokenizer()
# tokenizer.load(VOCAB_FILE)

# input_sequence = tokenizer.encode(data[0], max_length=60)

# print(f"Encoded Sequence: {input_sequence}")
# print(tokenizer.length())



import torch.optim as optim
import torch
from MusicTransformer import MusicTransformer
import torch.nn as nn
from PlaylistDataset import PlaylistDataset
from torch.utils.data import Dataset, DataLoader

VOCAB_SIZE = 229878
D_MODEL = 256
NHEAD = 8 # Number of attention heads
NUM_LAYERS = 4 # Number of transformer blocks
LR = 0.001 # Learning Rate
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = PlaylistDataset("processed_dataset.pt")
# num_workers=0 is usually fastest because data is already in RAM
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)


# Initialize Model
model = MusicTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS)
model = model.to(DEVICE)

# Define Optimizer and Loss
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0) # We ignore the <PAD> token (ID 0)

print(f"Training on {DEVICE}...")

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs) 
        # logits shape: [Batch, Seq_Len, Vocab]
        
        # Reshape for CrossEntropyLoss
        # Flatten the batch and sequence dimensions
        # logits -> [Batch * Seq_Len, Vocab]
        # targets -> [Batch * Seq_Len]
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} Complete. Average Loss: {total_loss / len(dataloader):.4f}")

# 5. Save the trained model
torch.save(model.state_dict(), "music_transformer.pth")
print("Model saved!")