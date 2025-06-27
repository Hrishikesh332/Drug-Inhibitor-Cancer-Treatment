"""
This is the copied architecture with changes applied for LXT congruent with the approach provided here:
https://github.com/rachtibat/LRP-eXplains-Transformers/issues/32

As in the orihinal code and for ease of understanding changes applied are denoted with 
### <------------------------------------------- LXT
"""

import copy
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat, device
from functools import partial
from lxt.efficient.rules import divide_gradient, identity_rule_implicit, stop_gradient
from lxt.efficient.patches import patch_method, replace_module, dropout_forward

import trans_synergy
from trans_synergy.models.trans_synergy import attention_model

setting = trans_synergy.settings.get()
use_cuda = torch.cuda.is_available()

attnLRP = {
    torch.nn.Dropout: partial(patch_method, dropout_forward), # we patch dropout just in case if the user sets the model to train() mode
    attention_model: partial(replace_module, sys.modules[__name__]),
}


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        std = x.std(dim=-1, keepdim=True) + self.eps
        mean = x.mean(dim=-1, keepdim=True)
        y = (x - mean) / stop_gradient(std) ### <------------------------------------------- LXT
        y *= self.alpha
        y += self.bias
        return y


def attention(q, k, v, d_k = 1, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) 
    divide_gradient(scores, 2) ### <------------------------------------------- LXT
    scores = scores / math.sqrt(d_k) ### <------------------------------------------- LXT

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    output = divide_gradient(output, 2) ### <------------------------------------------- LXT
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=200, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.norm = Norm(d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x, low_dim = False):
        hidden_states = self.linear_1(x)
        x = identity_rule_implicit(F.relu, hidden_states) ### <------------------------------------------- LXT
        x = x if low_dim else self.norm(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class OutputFeedForward(nn.Module):

    def __init__(self, H, W, d_layers = None, dropout=0.1):

        super().__init__()

        self.d_layers = [512, 1] if d_layers is None else d_layers
        self.linear_1 = nn.Linear(H*W, self.d_layers[0])
        self.norm_1 = Norm(H*W)
        self.n_layers = len(self.d_layers)
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(self.n_layers))
        self.dropouts[0] = nn.Dropout(p=0.2)
        self.linear_layers = nn.ModuleList(nn.Linear(d_layers[i-1], d_layers[i]) for i in range(1, self.n_layers))
        self.norms = nn.ModuleList(Norm(d_layers[i-1]) for i in range(1, self.n_layers))

    def forward(self, x, low_dim = False):
        ### test whether the norm layers are needed
        # if low_dim:
        #     x = self.norm_1(x)
        x = self.dropouts[0](x)
        x = self.linear_1(x)
        for i in range(self.n_layers-1):
            x = identity_rule_implicit(F.relu, x) ### <------------------------------------------- LXT
            if not low_dim:
                x = self.norms[i](x)
            x = self.dropouts[i+1](x)
            x = self.linear_layers[i](x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, low_dim = False):

        x2 = x if low_dim else self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = x if low_dim else self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2, low_dim = low_dim))
        return x

class OutputAttentionLayer(nn.Module):

    def __init__(self, src_d_model, trg_d_model):
        ## norm(src) + norm(trg) + linear + attn
        super().__init__()
        self.src_norm = Norm(src_d_model)
        self.trg_norm = Norm(trg_d_model)
        self.src_linear = nn.Linear(src_d_model, src_d_model)
        self.trg_linear = nn.Linear(trg_d_model, trg_d_model)

    def forward(self, src, trg):
        output = attention(src, trg, trg)
        return output

class MulAttentionLayer(nn.Module):

    def __init__(self, src_d_model, trg_d_model):

        super().__init__()
        self.context = nn.Parameter(torch.FloatTensor(src_d_model, 1))
        self.src_norm = Norm(src_d_model)
        self.trg_norm = Norm(trg_d_model)
        self.src_linear = nn.Linear(src_d_model, src_d_model)
        self.trg_linear = nn.Linear(trg_d_model, trg_d_model)

    def forward(self, src, trg):
        src = self.src_linear(self.src_norm(src))
        src = identity_rule_implicit(torch.tanh, src) #### <------------------------------------------- LXT
        trg = self.trg_linear(self.trg_norm(trg))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask=None, trg_mask=None, low_dim = False):
        x2 = x if low_dim else self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = x if low_dim else self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = x if low_dim else self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2, low_dim = low_dim))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None, low_dim=False):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask, low_dim=low_dim)
        return x if low_dim else self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask=None, trg_mask=None, low_dim=False):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask, low_dim=low_dim)
        return x if low_dim else self.norm(x)


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(self.d_model, N, heads, dropout)
        self.decoder = Decoder(self.d_model, N, heads, dropout)

    def forward(self, src, trg, src_mask=None, trg_mask=None, low_dim=False):
        e_outputs = self.encoder(src, src_mask, low_dim=low_dim)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask, low_dim=low_dim)
        flat_d_output = d_output.contiguous().view(
            -1, d_output.size(-2) * d_output.size(-1)
        )
        return flat_d_output

class TransposeMultiTransformers(nn.Module):

    def __init__(
        self,
        d_input_list,
        d_model_list,
        n_feature_type_list,
        N,
        heads,
        dropout,
        masks=None,
        linear_only=False,
    ):
        super().__init__()

        assert len(d_input_list) == len(n_feature_type_list) and len(
            d_input_list
        ) == len(d_model_list), "claimed inconsistent number of transformers"
        self.linear_only = linear_only
        self.linear_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(len(d_input_list)):

            num_of_linear_module = (
                setting.n_feature_type[i] if setting.one_linear_per_dim else 1
            )

            for j in range(num_of_linear_module):
                self.linear_layers.append(
                        nn.Linear(d_input_list[i], d_model_list[i])
                )
                self.norms.append(Norm(d_model_list[i]))
                self.dropouts.append(nn.Dropout(p=dropout))

        self.transformer_list = nn.ModuleList()
        self.n_feature_type_list = n_feature_type_list
        for i in range(len(d_input_list)):
            self.transformer_list.append(
                Transformer(
                    n_feature_type_list[i] * setting.d_model_i, N, heads, dropout
                )
            )

    def forward(
        self, src_list, trg_list=None, src_mask=None, trg_mask=None, low_dim=False
    ):

        assert len(src_list) == len(
            self.transformer_list
        ), "inputs length is not same with input length for model"
        src_list_linear = []
        trg_list_linear = []
        cur_linear = 0
        for i in range(len(self.transformer_list)):

            src_list_dim = []
            trg_list_dim = []
            for j in range(src_list[i].size(1)):
                cur_src_dim = src_list[i][:, j : j + 1, :]
                cur_trg_dim = trg_list[i][:, j : j + 1, :]
                cur_src_processed_dim = self.linear_layers[cur_linear](cur_src_dim)
                cur_src_processed_dim = identity_rule_implicit( ### <------------------------------------------- LXT
                    F.relu, cur_src_processed_dim
                )
                cur_src_processed_dim = self.dropouts[cur_linear](
                    cur_src_processed_dim
                )
                cur_trg_processed_dim = self.linear_layers[cur_linear](cur_trg_dim)
                cur_trg_processed_dim = identity_rule_implicit( ### <------------------------------------------- LXT
                    F.relu, cur_trg_processed_dim
                )
                cur_trg_processed_dim = self.dropouts[cur_linear](
                    cur_trg_processed_dim
                )
                src_list_dim.append(
                    cur_src_processed_dim.contiguous().view(
                        [-1, setting.d_model_i, setting.d_model_j]
                    )
                )
                trg_list_dim.append(
                    cur_trg_processed_dim.contiguous().view(
                        [-1, setting.d_model_i, setting.d_model_j]
                    )
                )
                cur_linear += 1
            src_list_linear.append(cat(tuple(src_list_dim), dim=1))
            trg_list_linear.append(cat(tuple(trg_list_dim), dim=1))

        output_list = []
        for i in range(len(self.transformer_list)):
            src_list_linear[i] = torch.transpose(src_list_linear[i], -1, -2)
            if self.linear_only:
                batch_size = src_list_linear[i].size(0)
                output_list.append(src_list_linear[i].contiguous().view(batch_size, -1))
            else:
                trg_list_linear[i] = torch.transpose(trg_list_linear[i], -1, -2)
                output_list.append(
                    self.transformer_list[i](
                        src_list_linear[i], trg_list_linear[i], low_dim=low_dim
                    )
                )
        return output_list


class TransposeMultiTransformersPlusLinear(TransposeMultiTransformers):
    # copied
    def __init__(
        self,
        d_input_list,
        d_model_list,
        n_feature_type_list,
        N,
        heads,
        dropout,
        masks=None,
        linear_only=False,
        drugs_on_the_side=False,
        classifier=False,
    ):

        self.device1 = device("cuda") if use_cuda else device("cpu")
        self.device2 = device("cuda") if use_cuda else device("cpu")
        super().__init__(
            d_input_list,
            d_model_list,
            n_feature_type_list,
            N,
            heads,
            dropout,
            masks=masks,
            linear_only=linear_only,
        )
        out_input_length = sum(
            [d_model_list[i] * n_feature_type_list[i] for i in range(len(d_model_list))]
        )
        if drugs_on_the_side:
            self.drugs_on_the_side = drugs_on_the_side
            out_input_length += 2 * setting.drug_emb_dim
        self.out = OutputFeedForward(
            out_input_length, 1, d_layers=setting.output_FF_layers, dropout=dropout
        )
        self.linear_only = linear_only
        self.classifier = classifier

    def forward(
        self, *src_list, drugs=None, src_mask=None, trg_mask=None, low_dim=True
    ):

        input_src_list = src_list
        input_trg_list = src_list[::]
        output_list = super().forward(input_src_list, input_trg_list, low_dim=low_dim)

        cat_output = cat(tuple(output_list), dim=1)
        output = self.out(cat_output)
        if self.classifier:
            output = F.softmax(output, dim=-1)
        return output
    
    