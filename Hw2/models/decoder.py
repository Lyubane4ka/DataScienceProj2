import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        return self.pos_embedding(positions)


class TransformerDecoderModel(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, forward_expansion, dropout, max_len):
        super(TransformerDecoderModel, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=num_heads,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_target_mask(self, tgt):
        tgt_len = tgt.size(1)
        mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        return mask.to(tgt.device)

    def forward(self, captions, memory):
        tgt_emb = self.dropout(self.word_embedding(captions) + self.positional_encoding(captions))
        tgt_mask = self.generate_target_mask(captions)
        out = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(out)
