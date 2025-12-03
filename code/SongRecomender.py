# import torch
# import torch.nn.functional as F

# class SongRecomender:
#     def __init__(self, model, tokenizer, device="cpu"):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = device
#         self.model.eval()
#         self.model.to(device)

#     def suggest_next_songs(self, current_playlist_uris, top_k=50, temperature=1.0):
#         """
#         Args:
#             current_playlist_uris: List of strings (track URIs).
#             top_k: Pool size for sampling.
#             temperature: Higher (1.2) = More creative/random. Lower (0.8) = More conservative/predictable.
#         """
#         # 1. Tokenize Input
#         # We use a simple encode, no padding needed for inference usually
#         # but we must respect the context length of the model (e.g. 50)
#         token_ids = [self.tokenizer.vocab.get(uri, self.tokenizer.vocab["<UNK>"]) for uri in current_playlist_uris]
        
#         # Add SOS if starting fresh, otherwise just raw sequence
#         if not token_ids:
#             token_ids = [self.tokenizer.vocab["<SOS>"]]
            
#         # Truncate if too long (Model can only see last N songs)
#         max_context = 50 
#         if len(token_ids) > max_context:
#             token_ids = token_ids[-max_context:]

#         # Convert to Tensor
#         input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)

#         # 2. Model Prediction
#         with torch.no_grad():
#             logits = self.model(input_tensor)
        
#         # We only care about the prediction for the LAST song in the sequence
#         # Shape: [1, Vocab_Size]
#         last_token_logits = logits[0, -1, :]
        
#         # 3. Apply Temperature
#         # Dividing by temp:
#         #   > 1.0 flattens distribution (increases diversity)
#         #   < 1.0 sharpens distribution (increases confidence)
#         scaled_logits = last_token_logits / temperature

#         # 4. Top-K Sampling
#         # Get top k values and their indices
#         probs = F.softmax(scaled_logits, dim=-1)
#         top_probs, top_indices = torch.topk(probs, top_k)
        
#         # Sample from the filtered distribution
#         # multinomial returns the index *within* top_probs (0 to k-1)
#         next_token_index = torch.multinomial(top_probs, 1).item()
        
#         # Get the actual Vocabulary ID
#         predicted_id = top_indices[next_token_index].item()
        
#         # 5. Decode back to URI
#         predicted_uri = self.tokenizer.reverse_vocab.get(predicted_id, "Unknown_URI")
        
#         return predicted_uri, probs[predicted_id].item()

# # --- USAGE EXAMPLE ---

# # 1. Setup (assuming objects from previous steps exist)
# dj = SongRecomender(model, tokenizer, device="cuda")

# # 2. Existing Playlist (User Input)
# user_playlist = [
#     "spotify:track:0UaMYEvWZi0ZqiDOoHU3YI", # Lose Control
#     "spotify:track:6I9VzXrHxO9rA9A5euc8Ak"  # Toxic
# ]

# # 3. Predict the 3rd song
# next_song, confidence = dj.suggest_next_songs(user_playlist, top_k=20, temperature=0.9)

# print(f"Based on {len(user_playlist)} songs...")
# print(f"I recommend: {next_song}")
# print(f"Confidence: {confidence:.4f}")






import torch
import torch.nn.functional as F
import json
from MusicTransformer import MusicTransformer

MODEL_PATH = "music_transformer_v1.pth"
VOCAB_PATH = "vocab.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading vocabulary...")
with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)
reverse_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

model = MusicTransformer(
    vocab_size=vocab_size,
    d_model=256,
    nhead=8,
    num_layers=4,
    dropout=0.1
)

print("Loading model weights...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval() # Switches off Dropout layers

def recommend_next_song(current_playlist_uris, top_k=50, temperature=1.0):
    input_ids = [vocab.get(uri, vocab["<UNK>"]) for uri in current_playlist_uris]
    
    if not input_ids:
        input_ids = [vocab["<SOS>"]]
        
    if len(input_ids) > 50:
        input_ids = input_ids[-50:]

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
    
    next_token_logits = logits[0, -1, :]
    
    next_token_logits = next_token_logits / temperature

    probs = F.softmax(next_token_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)
    
    next_index_in_top_k = torch.multinomial(top_probs, 1).item()
    
    predicted_id = top_indices[next_index_in_top_k].item()
    
    predicted_uri = reverse_vocab.get(predicted_id, "Unknown_URI")
    
    return predicted_uri

user_playlist = [
    "spotify:track:0UaMYEvWZi0ZqiDOoHU3YI", # Lose Control
    "spotify:track:6I9VzXrHxO9rA9A5euc8Ak"  # Toxic
]

print("\n--- Input Playlist ---")
print(user_playlist)

print("\n--- Generating Recommendations ---")

for _ in range(5):
    next_song = recommend_next_song(user_playlist, top_k=20, temperature=0.9)
    print(f"Recommended: {next_song}")
    
    user_playlist.append(next_song)