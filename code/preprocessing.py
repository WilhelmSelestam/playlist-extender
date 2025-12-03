import torch
import json
import os
from tqdm import tqdm
from SongTokenizer import SongTokenizer
from stream_dataset import stream_dataset

def preprocess_and_save(data_path, vocab_path, output_file, max_len=50):
    
    tokenizer = SongTokenizer()
    tokenizer.load(vocab_path)
    
    generator = stream_dataset(data_path)
    
    encoded_playlists = []
    
    print("Tokenizing all playlists...")
    for playlist in tqdm(generator):
        ids = tokenizer.encode(playlist, max_length=max_len)
        encoded_playlists.append(ids)
        
    data_tensor = torch.tensor(encoded_playlists, dtype=torch.long)
    
    torch.save(data_tensor, output_file)
    #print("Size: ", os.path.getsize(output_file) / (1024*1024), "MB")








preprocess_and_save(
    data_path="C:/Users/seles/playlist-extender/dataset/data",
    vocab_path="./vocab.json",
    output_file="./processed_dataset.pt",
    max_len=50
)