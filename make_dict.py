import pickle
import pandas
import json
import os
from tqdm import tqdm
from chemistry_utils import canonical_smiles, calculate_reaction_and_product_fps
import torch
import numpy as np


data_path = 'data/clean_results.json'
output_path = 'reaction_fingerprints.pkl'
def parse_uspto_condition_data(data_path, output_path, verbose=True):
    with open(data_path) as Fin:
        raw_info = json.load(Fin)
    data_dict = {}
    iterx = tqdm(raw_info) if verbose else raw_info
    for i, element in enumerate(iterx):
        smiles = element['new']['canonical_rxn']
        #print(smiles)

        reactants, products = smiles.split(">>")
        reac_FP, prod_FP = calculate_reaction_and_product_fps(reactants,products)
        reac_FP = torch.tensor(reac_FP, dtype=torch.float32)
        prod_FP = torch.tensor(prod_FP, dtype=torch.float32)


        #print(prod_FP.shape)
        #print(combined_FP.shape)
        data_dict[smiles] = {'reac':reac_FP,'prod':prod_FP}
    with open(output_path, 'wb') as fout:
        pickle.dump(data_dict, fout)



parse_uspto_condition_data(data_path, output_path)
