import torch.nn as nn
from src.Sublayers import FeedForward, MultiHeadAttention, Norm, attention
import torch


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

        #src = self.src_linear(self.src_norm(src))
        #trg = self.trg_linear(self.trg_norm(trg))
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
        src = torch.tanh(self.src_linear(self.src_norm(src)))
        trg = self.trg_linear(self.trg_norm(trg))
        transfered_src = torch.matmul(src, self.context)

# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
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
