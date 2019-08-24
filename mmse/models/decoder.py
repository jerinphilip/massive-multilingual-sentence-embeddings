import torch
from torch import nn
import torch.nn.functional as F

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
        x = F.dropout(x, p=self.args.dropout, training=self.training)
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
            # This may require some check
            decoder_input = torch.cat([ 
                x[j, :, :], lang_embed, context
            ], dim=1)

            # T=1 x B x H
            decoder_input = decoder_input.unsqueeze(0)

            # TODO(jerin): These might require projections
            decoder_outs, (h_final, c_final) = self.lstm(decoder_input, (h0, c0))
            decoder_outs  = F.dropout(decoder_outs, p=self.args.dropout, training=self.training)

            outs.append(decoder_outs)
            h0 = h_final
            c0 = c_final

        # T x B x H 
        x = torch.cat(outs, dim=0)
        return x


