import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder
from .generator import Generator
from .tce_loss import TCELoss

class EmbeddingModel(nn.Module):
    def __init__(self, encoder, decoder, generator, criterion):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.criterion = criterion

    @classmethod
    def build(cls, args, dictionary):
        # Load dictionary
        # Create Enc, Dec, Generator, Loss
        embed_tokens = nn.Embedding(len(dictionary), args.encoder_embedding_dim)
        encoder = Encoder(args, embed_tokens, dictionary)
        decoder = Decoder(args, embed_tokens, dictionary)
        generator = Generator(args, dictionary)
        criterion = TCELoss(dictionary)
        return cls(encoder, decoder, generator, criterion)

    def get_generator_output(self, _input):
        encoder_output = self.encoder(_input["srcs"], _input["src_lens"])
        decoder_output = self.decoder(_input["tgt_langs"], _input["tgts"], encoder_output)
        generator_output = self.generator(decoder_output)
        return generator_output

    def forward(self, _input):
        generator_output = self.get_generator_output(_input)
        shifted_gen_outputs = generator_output[:-1, :, :]
        shifted_tgts = _input["tgts"][:, 1:]
        # print(shifted_tgts.size(), shifted_gen_outputs.size())
        loss_output = self.criterion(shifted_gen_outputs, shifted_tgts)
        return loss_output
