import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, args, embed_tokens, dictionary):
        super().__init__()
        self.args = args
        self.embed_tokens = embed_tokens
        self.dictionary = dictionary
        self.lstm = nn.LSTM(
                args.encoder_embedding_dim, args.encoder_hidden_size, 
                num_layers=args.encoder_num_layers, 
                bidirectional=args.encoder_bidirectional
        )

    def forward(self, sequence, sequence_lengths):
        #TODO(jerin): Add dropout
        x = self.embed_tokens(sequence)
        batch_size, seq_len, _ = x.size()
        x = x.transpose(0, 1) 

        scale = 2 if self.args.encoder_bidirectional else 1
        state_size = (
            scale * self.args.encoder_num_layers, 
            batch_size, self.args.encoder_hidden_size
        )

        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)

        packed_x = nn.utils.rnn.pack_padded_sequence(x, 
                sequence_lengths.data.tolist())
        
        packed_outs, (h_final, c_final) = self.lstm(packed_x, (h0, c0))

        x, _  = nn.utils.rnn.pad_packed_sequence(
                packed_outs, padding_value=self.dictionary.pad()
        )

        if self.args.encoder_bidirectional:
            def combine_bidir(outs):
                out = (outs.view(self.args.encoder_num_layers, 2, batch_size, -1)
                        .transpose(1, 2).contiguous())
                return out.view(self.args.encoder_num_layers, batch_size, -1)

            h_final = combine_bidir(h_final)
            c_final = combine_bidir(c_final)

        return {
            "encoder_outs": x,
            "encoder_hiddens": h_final, 
            "encoder_cells": c_final
        }


class Decoder(nn.Module):
    def __init__(self, args, embed_tokens, dictionary):
        super().__init__()
        self.args = args
        self.embed_tokens = embed_tokens
        self.dictionary = dictionary

        scale = 2 if self.args.encoder_bidirectional else 1
        self.input_size = (
                args.decoder_embedding_dim +  # token
                args.decoder_embedding_dim +  # language
                scale*args.encoder_hidden_size
        )

        self.lstm = nn.LSTM(
                self.input_size, args.decoder_hidden_size, 
                num_layers=args.decoder_num_layers, bidirectional=False
        )


    def forward(self, tgt_langs, prev_output_tokens, encoder_dict):
        encoder_outs = encoder_dict["encoder_outs"]
        encoder_hiddens = encoder_dict["encoder_hiddens"]
        encoder_cells = encoder_dict["encoder_cells"]
        srclen, batch_size, _ = encoder_outs.size()
        # print(encoder_outs.size())

        # T x B x H
        context = encoder_outs[-1, :, :]
        batch_size, seqlen = prev_output_tokens.size()
        x = self.embed_tokens(prev_output_tokens)
        lang_embed = self.embed_tokens(tgt_langs)
        # TODO(jerin): Dropout

        # B x T x H -> T x B x H
        x = x.transpose(0, 1)

        h0 = encoder_hiddens
        c0 = encoder_cells
        outs = []
        for j in range(seqlen):
            # Take only final hidden.
            # Concatenate with language idx_embedding
            # Concatenate with current_token_embeddding
            # print(x[j, :, :].size())
            # print(lang_embed.size())
            # print(context.size())
            decoder_input = torch.cat([ 
                x[j, :, :], lang_embed, context
            ], dim=1)

            # T=1 x B x H
            decoder_input = decoder_input.unsqueeze(0)

            # TODO(jerin): These might require projections
            decoder_outs, (h_final, c_final) = self.lstm(decoder_input, (h0, c0))
            outs.append(decoder_outs)
            h0 = h_final
            c0 = c_final

        # T x B x H 
        x = torch.stack(outs, dim=0)
        return x


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
        generator = None
        criterion = None
        return cls(encoder, decoder, generator, criterion)

    def forward(self, _input):
        encoder_output = self.encoder(_input["srcs"], _input["src_lens"])
        decoder_output = self.decoder(_input["tgt_langs"], _input["tgts"], encoder_output)
        return decoder_output
