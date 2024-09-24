import pandas
import json
import os
from tqdm import tqdm
from chemistry_utils import canonical_smiles, calculate_reaction_and_product_fps
import torch
import numpy as np
import random



def clk_x(x):
    return x if x == '' else canonical_smiles(x)


def parse_uspto_condition_data(data_path, verbose=True):
    with open(data_path) as Fin:
        raw_info = json.load(Fin)
    cats = set()
    sov1s = set()
    sov2s = set()
    reg1s = set()
    reg2s = set()
    iterx = tqdm(raw_info) if verbose else raw_info
    for i, element in enumerate(iterx):
        cat = clk_x(element['new']['catalyst'])
        sov1 = clk_x(element['new']['solvent1'])
        sov2 = clk_x(element['new']['solvent2'])
        reg1 = clk_x(element['new']['reagent1'])
        reg2 = clk_x(element['new']['reagent2'])
        cats.add(cat)
        sov1s.add(sov1)
        sov2s.add(sov2)
        reg1s.add(reg1)
        reg2s.add(reg2)

    all_data = {'train_data': [], 'val_data': [], 'test_data': []}
    cat2idx = {k: idx for idx, k in enumerate(cats)}
    sov12idx = {k: idx for idx, k in enumerate(sov1s)}
    sov22idx = {k: idx for idx, k in enumerate(sov2s)}
    cat2idx = {k: idx for idx, k in enumerate(cats)}
    reg12idx = {k: idx for idx, k in enumerate(reg1s)}
    reg22idx = {k: idx for idx, k in enumerate(reg2s)}


    iterx = tqdm(raw_info) if verbose else raw_info
    for i, element in enumerate(iterx):
        rxn_type = element['dataset']
        labels = [
            cat2idx[clk_x(element['new']['catalyst'])],
            sov12idx[clk_x(element['new']['solvent1'])],
            sov22idx[clk_x(element['new']['solvent2'])],
            reg12idx[clk_x(element['new']['reagent1'])],
            reg22idx[clk_x(element['new']['reagent2'])]
        ]

        reactants, products = element['new']['canonical_rxn'].split(">>")
        reac_FP, prod_FP = calculate_reaction_and_product_fps(reactants,products)
        reac_FP = torch.tensor(reac_FP, dtype=torch.float32)
        prod_FP = torch.tensor(prod_FP, dtype=torch.float32)


        
        this_line = {
            'canonical_rxn': element['new']['canonical_rxn'],
            'react_fp': reac_FP,
            'prod_fp': prod_FP,
            'label': labels,
            'mapped_rxn': element['new']['mapped_rxn']
        }
        all_data[f'{rxn_type}_data'].append(this_line)

    return all_data,  cat2idx, sov12idx, sov22idx, reg12idx, reg22idx



def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def correct_trans_output(trans_pred, end_idx, pad_idx):
    batch_size, max_len = trans_pred.shape
    device = trans_pred.device
    x_range = torch.arange(0, max_len, 1).unsqueeze(0)
    x_range = x_range.repeat(batch_size, 1).to(device)

    y_cand = (torch.ones_like(trans_pred).long() * max_len + 12).to(device)
    y_cand[trans_pred == end_idx] = x_range[trans_pred == end_idx]
    min_result = torch.min(y_cand, dim=-1, keepdim=True)
    end_pos = min_result.values
    trans_pred[x_range > end_pos] = pad_idx
    return trans_pred


def data_eval_trans(trans_pred, trans_lb, return_tensor=False):
    batch_size, max_len = trans_pred.shape
    line_acc = torch.sum(trans_pred == trans_lb, dim=-1) == max_len
    line_acc = line_acc.cpu()
    return line_acc if return_tensor else (line_acc.sum().item(), batch_size)


def generate_square_subsequent_mask(sz, device='cpu'):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = (mask == 0).to(device)
    return mask


def generate_tgt_mask(tgt, pad_idx, device='cpu'):
    siz = tgt.shape[1]
    tgt_pad_mask = (tgt == pad_idx).to(device)
    tgt_sub_mask = generate_square_subsequent_mask(siz, device)
    return tgt_pad_mask, tgt_sub_mask


def check_early_stop(*args):
    answer = True
    for x in args:
        answer &= all(t <= x[0] for t in x[1:])
    return answer


def convert_log_into_label(logits, mod='sigmoid'):
    if mod == 'sigmoid':
        pred = torch.zeros_like(logits)
        pred[logits >= 0] = 1
        pred[logits < 0] = 0
    elif mod == 'softmax':
        preds = [torch.argmax(logit, dim=-1) for logit in logits]
        pred = torch.stack(preds, dim=-1)
    else:
        raise NotImplementedError(f'Invalid mode {mod}')
    return pred
