import json
from collections import Counter
from tqdm import tqdm
import os
import glob

class SongTokenizer:
  def __init__(self, min_freq=5):
    self.min_freq = min_freq
    self.vocab = {}
    self.reverse_vocab = {}
    self.special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    
    for i, token in enumerate(self.special_tokens):
      self.vocab[token] = i
      self.reverse_vocab[i] = token
          
  def fit(self, playlists):
      track_counts = Counter()
      
      for playlist in tqdm(playlists):
        for track in playlist['tracks']:
            track_counts[track['track_uri']] += 1
              
      #print(f"Total unique tracks found: {len(track_counts)}")
      
      # Assign IDs to tracks that meet the frequency threshold
      current_id = len(self.vocab)
      for track_uri, count in track_counts.items():
          if count >= self.min_freq:
              self.vocab[track_uri] = current_id
              self.reverse_vocab[current_id] = track_uri
              current_id += 1
              
      #print(f"Final Vocabulary Size: {len(self.vocab)}")
      #print(f"Tracks converted to <UNK>: {len(track_counts) - (len(self.vocab) - 4)}")
      
  def save(self, filepath):
        print(f"Saving vocabulary to {filepath}...")
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f)
        print("Saved.")
      
  def load(self, filepath):
        print(f"Loading vocabulary from {filepath}...")
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
        
        # Rebuild the reverse lookup (ID -> URI)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Loaded {len(self.vocab)} tokens.")

  def encode(self, playlist_tracks, max_length=50):
      # Start with SOS
      token_ids = [self.vocab["<SOS>"]]
      
      for track in playlist_tracks['tracks']:
          uri = track['track_uri']
          # Get ID or return <UNK> ID
          token_ids.append(self.vocab.get(uri, self.vocab["<UNK>"]))
          
      # Append EOS
      token_ids.append(self.vocab["<EOS>"])
      
      # Truncate or Pad
      if len(token_ids) > max_length:
          token_ids = token_ids[:max_length]
      else:
          # Pad with <PAD> token
          padding = [self.vocab["<PAD>"]] * (max_length - len(token_ids))
          token_ids.extend(padding)
          
      return token_ids

  def decode(self, token_ids):
      uris = []
      for tid in token_ids:
          if tid in self.reverse_vocab:
              uris.append(self.reverse_vocab[tid])
          else:
              uris.append("Unknown_ID")
      return uris
    
  def length(self):
    return len(self.vocab)


