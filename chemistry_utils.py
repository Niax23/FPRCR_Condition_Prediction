from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ExplicitBitVect
from rdkit.DataStructs import ConvertToNumpyArray
import numpy as np


def canonical_smiles(x):
    mol = Chem.MolFromSmiles(x)
    return x if mol is None else Chem.MolToSmiles(mol)


def get_morgan_fingerprint(smiles, radius=2, n_bits=16384):

    # print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits)
        return fingerprint
    print('wrong')
    return ExplicitBitVect(n_bits)


def calculate_reaction_and_product_fps(reac, prod):

    reactant_fp = get_morgan_fingerprint(reac)
    # print(reactant_fp)

    product_fp = get_morgan_fingerprint(prod)

    reaction_fp = product_fp ^ reactant_fp

    reac_array = np.zeros((16384,), dtype=int)
    ConvertToNumpyArray(reaction_fp, reac_array)
    prod_array = np.zeros((16384,), dtype=int)
    ConvertToNumpyArray(product_fp, prod_array)

    return reaction_fp, product_fp
