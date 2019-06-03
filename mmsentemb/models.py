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
        x = x.transpose(0, 1) 
        batch_size, seq_len, _ = embedded.size()

        scale = 2 if self.args.encoder_bidirectional else 1
        state_size = (
                    scale * args.encoder_num_layers, 
                    batch_size, args.encoder_hidden_size
        )
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)

        packed_x = nn.utils.rnn.pack_padded_sequence(x, sequence_lengths.data.tolist())
        packed_outs, (h_final, c_final) = lstm(
                packed_x, (h0, c0)
        )

        x, _  = nn.utils.rnn.pack_padded_sequence(
                packed_outs, padding_value=self.dictionary.pad()
        )

        if self.args.bidirectional:
            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            h_final = combine_bidir(h_final)
            c_final = combine_bidir(c_final)

        return {
            "encoder_out": (x, h_final, c_final)
        }


class Decoder(nn.Module):
    def __init__(self, args, embed_tokens, dictionary):
        super().__init__()
        self.args = args
        self.embed_tokens = embed_tokens
        self.dictionary = dictionary
        self.lstm = nn.LSTM(
                args.decoder_embedding_dim, args.decoder_hidden_size, 
                num_layers=args.decoder_num_layers, bidirectional=False
        )


    def forward(self, source_languages, prev_output_tokens, encoder_out):
        encoder_outs, encoder_hiddens, encoder_cells = encoder_outs
        srclen = encoder_outs.size()

        # B x T x H
        context = encoder_outs[:, -1, :]
        # Discarding fairseq's incremental stuff.
        batch_size, seqlen = prev_output_tokens.size()
        x = self.embed_tokens(prev_output_tokens)
        lang_embed = self.embed_tokens(source_languages)
        # TODO(jerin): Dropout

        # B x T x H -> T x B x H
        x = x.transpose(0, 1)

        h0 = encoder_hiddens
        c0 = encoder_cells
        for j in range(seqlen):
            # Take only final hidden.
            # Concatenate with language idx_embedding
            # Concatenate with current_token_embeddding
            decoder_input = torch.cat([
                    x[j, :, :],
                    context,
                    lang_embed
                ], dim=1)

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
    def build(cls, args):
        # Load dictionary

        # Create Enc, Dec, Generator, Loss
        return cls(encoder, decoder, generator, criterion)
