import torch.nn as nn
from torch import cat
import torch
import torch.nn.functional as F
from src.Layers import EncoderLayer, DecoderLayer
from src.Sublayers import Norm, OutputFeedForward
import copy
from src import setting
from src import use_cuda, device2
from src.CustomizedLinear import CustomizedLinear
from neural_fingerprint import NeuralFingerprint
from torch import device
import pandas as pd


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

    def forward(self, trg, e_outputs, src_mask=None, trg_mask=None, low_dim = False):
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

        # self.expand_dim_linear = nn.Linear(d_model, 16)
        # self.attn = MultiheadAttention(16, num_heads = heads, dropout = dropout)
        # self.shrink_dim_linear = nn.Linear(16, d_model)

    def forward(self, src, trg, src_mask=None, trg_mask=None, low_dim = False):
        # src = self.expand_dim_linear(src)
        # trg = self.expand_dim_linear(trg)
        e_outputs = self.encoder(src, src_mask, low_dim = low_dim)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask, low_dim=low_dim)
        # d_output, _ = self.attn(src, trg, trg)
        # d_output = self.shrink_dim_linear(d_output)
        flat_d_output = d_output.contiguous().view(-1, d_output.size(-2)*d_output.size(-1))
        return flat_d_output

class FlexibleTransformer(Transformer):

    def __init__(self, inputs_lengths, d_model, n_feature_type_list, N, heads, dropout):
        super().__init__(d_model, N, heads, dropout)
        self.final_inputs = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(inputs_lengths)):
            self.linear_layers.append(nn.Linear(inputs_lengths[i], d_model))
        out_input_length = d_model * sum(n_feature_type_list)
        self.out = OutputFeedForward(out_input_length, 1, d_layers=setting.output_FF_layers, dropout=dropout)

    def forward(self, src_list, trg_list, src_mask=None, trg_mask=None, low_dim = False):

        assert len(self.linear_layers) == len(src_list), "Features and sources length are different"
        final_srcs = []
        final_trgs = []
        for i in range(len(self.linear_layers)):
            final_srcs.append(self.linear_layers[i](src_list[i]))
            final_trgs.append(self.linear_layers[i](trg_list[i]))
        final_src = cat(tuple(final_srcs), 1)
        final_trg = cat(tuple(final_trgs), 1)
        flat_d_output = super().forward(final_src, final_trg)
        output = self.out(flat_d_output)
        return output

class TransposeMultiTransformers(nn.Module):

    def __init__(self,  d_input_list, d_model_list, n_feature_type_list, N, heads, dropout, masks = None, linear_only = False):
        super().__init__()

        assert len(d_input_list) == len(n_feature_type_list) and len(d_input_list) == len(d_model_list),\
            "claimed inconsistent number of transformers"
        self.linear_only = linear_only
        self.linear_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(len(d_input_list)):

            num_of_linear_module = setting.n_feature_type[i] if setting.one_linear_per_dim else 1

            for j in range(num_of_linear_module):
                self.linear_layers.append(CustomizedLinear(masks[i])) if masks[i] is not None else self.linear_layers.append(nn.Linear(d_input_list[i], d_model_list[i]))
                self.norms.append(Norm(d_model_list[i]))
                self.dropouts.append(nn.Dropout(p=dropout))

        self.transformer_list = nn.ModuleList()
        self.n_feature_type_list = n_feature_type_list
        for i in range(len(d_input_list)):
            self.transformer_list.append(Transformer(n_feature_type_list[i] * setting.d_model_i, N, heads, dropout))

    def forward(self, src_list, trg_list=None, src_mask=None, trg_mask=None, low_dim = False):

        # x = F.relu(self.linear_1(x))
        # x = x if low_dim else self.norm(x)
        # x = self.dropout(x)
        assert len(src_list) == len(self.transformer_list), "inputs length is not same with input length for model"
        src_list_linear = []
        trg_list_linear = []
        cur_linear = 0
        for i in range(len(self.transformer_list)):

            src_list_dim = []
            trg_list_dim = []
            for j in range(src_list[i].size(1)):
                cur_src_dim = src_list[i][:,j:j+1,:]
                cur_trg_dim = trg_list[i][:,j:j+1,:]
                cur_src_processed_dim = self.dropouts[cur_linear](F.relu(self.linear_layers[cur_linear](cur_src_dim)))
                cur_trg_processed_dim = self.dropouts[cur_linear](F.relu(self.linear_layers[cur_linear](cur_trg_dim)))
                src_list_dim.append(cur_src_processed_dim.contiguous().view([-1, setting.d_model_i, setting.d_model_j]))
                trg_list_dim.append(cur_trg_processed_dim.contiguous().view([-1, setting.d_model_i, setting.d_model_j]))
                cur_linear += 1
            src_list_linear.append(cat(tuple(src_list_dim), dim = 1))
            trg_list_linear.append(cat(tuple(trg_list_dim), dim = 1))

        output_list = []
        for i in range(len(self.transformer_list)):
            src_list_linear[i] = torch.transpose(src_list_linear[i], -1, -2)
            if self.linear_only:
                batch_size = src_list_linear[i].size(0)
                output_list.append(src_list_linear[i].contiguous().view(batch_size, -1))
            else:
                trg_list_linear[i] = torch.transpose(trg_list_linear[i], -1, -2)
                output_list.append(self.transformer_list[i](src_list_linear[i], trg_list_linear[i], low_dim=low_dim))
        return output_list

class ChemFP(nn.Module):

    feature_map = None
    def __init__(self, device):
        super().__init__()
        if self.feature_map is None:
            self.feature_map = pd.read_csv(setting.chemfp_drug_feature_file, index_col = 0)
        linear_layers = []
        pre_unit_dim = self.feature_map.shape[1]
        for i, hidden_unit in enumerate(setting.chem_linear_layers):
            linear_layers.append(nn.Linear(pre_unit_dim, hidden_unit))
            linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Dropout(p=setting.attention_dropout))
            pre_unit_dim = hidden_unit
        linear_layers.append(nn.Linear(pre_unit_dim, setting.drug_emb_dim))
        linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Dropout(p=setting.attention_dropout))
        self.linears = nn.Sequential(*linear_layers)
        self.device = device

    def forward(self, drug_names):
        input_feature = torch.from_numpy(self.feature_map.loc[list(drug_names)].values).float().to(self.device)
        return self.linears(input_feature)

class TransposeMultiTransformersPlusLinear(TransposeMultiTransformers):

    def __init__(self, d_input_list, d_model_list, n_feature_type_list, N, heads, dropout, masks=None, linear_only = False, drugs_on_the_side = False, classifier = False):

        self.device1 = device('cuda:0')
        self.device2 = device('cuda:1')
        super().__init__(d_input_list, d_model_list, n_feature_type_list, N, heads, dropout, masks=masks, linear_only = linear_only)
        out_input_length = sum([d_model_list[i] * n_feature_type_list[i] for i in range(len(d_model_list))])
        if drugs_on_the_side:
            self.drugs_on_the_side = drugs_on_the_side
            out_input_length += 2 * setting.drug_emb_dim
        self.out = OutputFeedForward(out_input_length, 1, d_layers=setting.output_FF_layers, dropout=dropout)
        self.linear_only = linear_only
        self.classifier = classifier
        if setting.neural_fp:
            self.drug_fp_a = NeuralFingerprint(setting.drug_input_dim['atom'], setting.drug_input_dim['bond'],
                                               setting.conv_size, setting.drug_emb_dim, setting.degree, device=self.device1)
            self.drug_fp_b = NeuralFingerprint(setting.drug_input_dim['atom'], setting.drug_input_dim['bond'],
                                               setting.conv_size, setting.drug_emb_dim, setting.degree, device=self.device1)
        else:
            self.drug_fp_a = ChemFP(device=self.device1)
            self.drug_fp_b = ChemFP(device=self.device1)

    def forward(self, *src_list, drugs = None, src_mask=None, trg_mask=None, low_dim = True):

        input_src_list = src_list
        input_trg_list = src_list[::]
        output_list = super().forward(input_src_list, input_trg_list, low_dim=low_dim)

        if drugs is not None and self.drugs_on_the_side:
            sub_drugs_a, sub_drugs_b = drugs[0], drugs[1]
            drug_a_embed = self.drug_fp_a(sub_drugs_a)
            drug_b_embed = self.drug_fp_b(sub_drugs_b)
            if setting.neural_fp:
                drug_a_embed = torch.sum(drug_a_embed, dim = 1)
                drug_b_embed = torch.sum(drug_b_embed, dim = 1)
            output_list += [drug_a_embed, drug_b_embed]

        cat_output = cat(tuple(output_list), dim=1)
        output = self.out(cat_output)
        if self.classifier:
            # output = F.log_softmax(output, dim = -1)
            output = F.softmax(output, dim=-1)
        return output

class LastFC(nn.Module):

    def __init__(self, d_model_list, dropout, input_len = None, classifier = False):

        super(LastFC, self).__init__()
        self.hidden_size = 100
        if input_len is None:
            input_len = 3 * len(setting.catoutput_intput_type) * setting.d_model + 2 * sum(list(
                setting.dir_input_type.values()))
        self.out = OutputFeedForward(input_len,
                                     1, d_layers=setting.output_FF_layers, dropout=dropout)
        self.classifier = classifier

    def forward(self, input):

        #cat_input = cat(tuple(input), dim=1)
        if isinstance(input, list) and len(input) == 1:
            input = input[0]
        bs = input.size(0)
        attn_output = input.contiguous().view(bs, -1)
        output = self.out(attn_output, low_dim = True)
        if self.classifier:
            # output = F.log_softmax(output, dim = -1)
            # output = F.softmax(output, dim = -1)
            output = F.relu(output)
        return output

def get_retrain_model(classifier = False):

    if not isinstance(setting.d_model, list):
        d_models = [setting.d_model] * 3 * len(setting.catoutput_intput_type)
    else:
        d_models = setting.d_model

    model = LastFC(d_models, setting.attention_dropout)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if use_cuda:
        model.to(device2)

    return model

def get_model(inputs_lengths):
    assert setting.d_model % setting.attention_heads == 0
    assert setting.attention_dropout < 1

    #model = TransformerPlusLinear(setting.d_input, setting.d_model, setting.n_layers, setting.attention_heads, setting.attention_dropout)
    model = FlexibleTransformer(inputs_lengths, setting.d_model, setting.n_layers,
                                setting.attention_heads, setting.attention_dropout)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def get_multi_models(inputs_lengths, input_masks = None, drugs_on_the_side = False, classifier = False):

    if not isinstance(setting.d_model, list):
        d_models = [setting.d_model] * len(inputs_lengths)
    else:
        d_models = setting.d_model

    if not isinstance(setting.n_feature_type, list):
        n_feature_types = [setting.n_feature_type] * len(inputs_lengths)
    else:
        n_feature_types = setting.n_feature_type

    for d_model in d_models:
        assert d_model % setting.attention_heads == 0
    assert setting.attention_dropout < 1

    if not isinstance(input_masks, list):
        input_masks = [input_masks] * len(inputs_lengths)
    else:
        input_masks = input_masks

    final_inputs_lengths = [inputs_lengths[i]//n_feature_types[i] for i in range(len(inputs_lengths))]
    model = TransposeMultiTransformersPlusLinear(final_inputs_lengths, d_models, n_feature_types, setting.n_layers,
                                                 setting.attention_heads, setting.attention_dropout,
                                                 input_masks, linear_only=False,
                                                 drugs_on_the_side = drugs_on_the_side, classifier=classifier)


    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
