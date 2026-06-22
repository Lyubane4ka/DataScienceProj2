import torch
import torch.nn as nn
from models.encoder import CNNEncoder
from models.decoder import TransformerDecoderModel

class ImageCaptioner(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, forward_expansion, dropout, max_len=100):
        super(ImageCaptioner, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = TransformerDecoderModel(
            embed_size, vocab_size, num_heads, num_layers, forward_expansion, dropout, max_len
        )

    def forward(self, images, captions):
        memory = self.encoder(images)
        output = self.decoder(captions, memory)
        return output
