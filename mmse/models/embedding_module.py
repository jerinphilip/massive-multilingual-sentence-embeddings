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

    def encode(self, _input):
        encoder_output = self.encoder(_input["srcs"], _input["src_lens"])
        encoder_outs = encoder_output['encoder_outs']

        # Max pooling to match Artetxe et. Al
        context, _ = torch.max(encoder_outs, dim=0)
        context = torch.nn.functional.normalize(context, dim=1, p=2)
        return context

    def get_generator_output(self, _input):
        encoder_output = self.encoder(_input["srcs"], _input["src_lens"])
        decoder_output = self.decoder(_input["tgt_langs"], _input["tgts"], encoder_output)
        generator_output = self.generator(decoder_output)
        return generator_output

    def forward(self, _input):
        # Compute loss
        generator_output = self.get_generator_output(_input)
        shifted_gen_outputs = generator_output[:-1, :, :]
        shifted_tgts = _input["tgts"][:, 1:]
        loss_output = self.criterion(shifted_gen_outputs, shifted_tgts)

        seq_len, batch_size, _ = shifted_gen_outputs.size()
        return loss_output, _input["tgt_lens"].detach().sum().item(), batch_size
