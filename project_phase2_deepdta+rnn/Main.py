import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys

import json
import numpy as np
import pandas as pd
import pickle
import math

from dta_models import DTAModel

data_filename = sys.argv[1]#'C:\\Users\\Aryan\\Downloads\\MainTestInp.csv'
output_file = sys.argv[2] #'predictions2.txt'
model_file = sys.argv[3] #'C:\\Users\\Aryan\\Downloads\\model41.pt'
class DTADataset(Dataset):
  def __init__(self, X_ligand, X_prot, y):
    self.X_ligand = torch.tensor(X_ligand)
    self.X_prot = torch.tensor(X_prot)
    self.y = torch.tensor(y)
  def __getitem__(self, index):
     if torch.is_tensor(index):
            index = index.tolist()
     return  self.X_ligand[index], self.X_prot[index], self.y[index]
  def __len__(self):
    return self.X_ligand.shape[0]

def calculate_predicted(model):
   model.eval()
   res = torch.tensor([]).to(device)

   with torch.no_grad():
      for i, data in enumerate(dataloader):
         drug, prot, _ = data[0].long().to(device), data[1].long().to(device),data[1].long().to(device)
         o = model(drug, prot)
         res = torch.cat([res, o], 0)

   return res

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
				"U": 19, "T": 20, "W": 21,
				"V": 22, "Y": 23, "X": 24,
				"Z": 25 }

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def encode_data(data, max_length, mapping):
    res = np.zeros((len(data), max_length))

    for i, d in enumerate(data):
        print(d)
        for w in range(min(max_length, len(d))):
            res[i][w] = mapping[d[w]]

    return res


def encode_one_hot(data, max_length, mapping):
    charlen = len(mapping)
    res = np.zeros((len(data), max_length, charlen))
    for i, d in enumerate(data.keys()):
        print(d)
        for w in range(min(max_length, len(data[d]))):
            res[i][w][mapping[data[d][w]]] = 1
    return res


max_length_lig = 85
max_length_prot = 1200


df = pd.read_csv(data_filename, delimiter='\t', header = None)

proteins = list(df[:][0])
ligands = list(df[:][1])


protein_data = encode_data(proteins, max_length_prot, CHARPROTSET)
ligand_data = encode_data(ligands, max_length_lig, CHARISOSMISET)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

model = DTAModel(85, 1200)
model.to(device)
model.load_state_dict(torch.load(model_file,  map_location=torch.device('cpu')))
model.eval()

drug_prots_dataset = DTADataset(ligand_data, protein_data, protein_data)
dataloader =  DataLoader(dataset = drug_prots_dataset,
                          batch_size = 256,

                     shuffle=True)
predictions = calculate_predicted(model).tolist()

predictions = {'preds': predictions}
print('done computing, saving now')

with open(output_file, 'w') as f:
    json.dump(predictions, f)
a = 0