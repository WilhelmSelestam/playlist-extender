import torch
import torch.nn as nn
import math

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, max_len=50, dropout=0.1):
        super(MusicTransformer, self).__init__()

        self.song_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # src shape: [Batch_Size, Seq_Len]
        batch_size, seq_len = src.shape
        
        # Create position indices: [0, 1, 2, ... seq_len-1]
        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        
        # Combine Song Embedding + Position Embedding
        # We scale song_embedding by sqrt(d_model) - a standard Transformer trick
        x = self.song_embedding(src) * math.sqrt(self.d_model) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Create the mask so the model doesn't look ahead
        mask = self.generate_square_subsequent_mask(seq_len, src.device)
        
        # Pass through Transformer
        # is_causal=True is optimized for this in newer PyTorch versions
        output = self.transformer_encoder(x, mask=mask, is_causal=True)
        
        # Project to vocabulary size
        logits = self.fc_out(output)
        
        # Output shape: [Batch_Size, Seq_Len, Vocab_Size]
        return logits