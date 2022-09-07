import torch
import torch.nn as nn

from thumt.models.transformer import Transformer, TransformerDecoder, TransformerDecoderLayer, AttentionSubLayer, \
    FFNSubLayer
from thumt.modules.layer_norm import LayerNorm

import thumt.utils


class STACLTransformerDecoderLayer(TransformerDecoderLayer):

    def __init__(self, params, name="layer"):
        super(TransformerDecoderLayer, self).__init__(name=name)

        with thumt.utils.scope(name):
            self.self_attention = AttentionSubLayer(params,
                                                    name="self_attention")
            self.encdec_attention = AttentionSubLayer(params,
                                                      name="encdec_attention")
            self.feed_forward = FFNSubLayer(params)

    def __call__(self, x, attn_bias, encdec_bias, memory, state=None):
        x = self.self_attention(x, attn_bias, state=state)
        # waitK policy
        if len(memory) == 1:
            # Full sent
            x = self.encdec_attention(x, encdec_bias, memory[0])
        else:
            # Wait-k policy
            cross_attn_outputs = []
            for i in range(x.shape[1]):
                q = x[:, i:i + 1, :]
                if i >= len(memory):
                    e = memory[-1]
                else:
                    e = memory[i]
                cross_attn_outputs.append(
                    self.encdec_attention(q, encdec_bias[:, :, i:i + 1, :e.shape[1]], e))
            x = torch.cat(cross_attn_outputs, dim=1)

        x = self.feed_forward(x)
        return x


class STACLTransformerDecoder(TransformerDecoder):

    def __init__(self, params, name="stacl_decoder"):
        super(STACLTransformerDecoder, self).__init__(params=params, name=name)

        self.normalization = params.normalization

        with thumt.utils.scope(name):
            self.layers = nn.ModuleList([
                STACLTransformerDecoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_decoder_layers)])

            if self.normalization == "before":
                self.layer_norm = LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None


class STACLTransformer(Transformer):
    def __init__(self, params, name="stacl_transformer"):
        super(STACLTransformer, self).__init__(params=params, name=name)
        self.waitk = params.waitk

        with thumt.utils.scope(name):
            self.decoder = STACLTransformerDecoder(params)

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)

        # inputs:[batch_size,length,hidden_size(embedding):512]
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias

        # encoding:positional_encoding
        inputs = nn.functional.dropout(self.encoding(inputs), self.dropout,
                                       self.training)

        enc_attn_bias = enc_attn_bias.to(inputs)

        src_max_len = inputs.shape[1]

        # waitk policy
        if self.waitk == -1 or self.waitk >= src_max_len:
            encoder_outputs = [self.encoder(inputs, enc_attn_bias)]
        else:
            encoder_outputs = []
            for i in range(self.waitk, src_max_len + 1):
                encoder_outputs.append(self.encoder(inputs[:, :i, :], enc_attn_bias[:, :, :, :i]))

        # encoder_output: list
        state["encoder_output"] = encoder_outputs
        state["enc_attn_bias"] = enc_attn_bias

        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])



        targets = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)

        decoder_input = torch.cat(
            [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
             targets[:, 1:, :]], dim=1)
        decoder_input = nn.functional.dropout(self.encoding(decoder_input),
                                              self.dropout, self.training)

        dec_enc_attn_bias = torch.tile(enc_attn_bias, [1, 1, decoder_input.shape[1], 1])

        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(targets)

        if mode == "infer":
            decoder_input = decoder_input[:, -1:, :]
            dec_attn_bias = dec_attn_bias[:, :, -1:, :]

        decoder_output = self.decoder(decoder_input, dec_attn_bias,
                                      dec_enc_attn_bias, encoder_output, state)

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1)

        return logits, state