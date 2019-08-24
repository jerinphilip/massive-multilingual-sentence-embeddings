import torch
from torch import nn
import torch.nn.functional as F

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
        x = F.dropout(x, p=self.args.dropout, training=self.training)
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

        x = F.dropout(x, p=self.args.dropout, training=self.training)

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


