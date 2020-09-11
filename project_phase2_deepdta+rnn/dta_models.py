import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
import pandas as pd
import pickle
import math

import pandas as pd


class DrugProtLSTM(nn.Module):
    def __init__(self, hidden_dim, charlen, embedding_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(charlen, embedding_dim)

        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, 1, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_e = self.embedding(x)

        lstm_out, h = self.LSTM(x_e)

        x = h[0][-1].view(-1, self.hidden_dim)

        x = self.fc1(x)
        x = self.relu(x)

        return x


class DrugModel(nn.Module):
    def __init__(self, charlen, embedding_dim, num_filters, druglen):
        super().__init__()
        self.embedding = nn.Embedding(charlen, embedding_dim)

        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=4)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=6)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size=8)

        """self.fc1 = nn.Linear(((signal_len - 15) // 3) * num_filters * 3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)"""

        self.maxpooling = nn.MaxPool1d(druglen - 15)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_e = self.embedding(x)
        x = torch.transpose(x_e, 1, 2)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)
        y = x

        x = self.maxpooling(x)

        """x = torch.flatten(x, 1)
  
        x = self.fc1(x)
        x = self.relu(x)
  
        x = F.dropout(x, p=0.1, training=self.training)
  
        x = self.fc2(x)
        x = self.relu(x)
  
        x = F.dropout(x, p = 0.1, training=self.training)
  
        x = self.fc3(x)
        x = self.relu(x)
  
        x = self.fc4(x)"""

        return x


class DrugProtModel(nn.Module):
    def __init__(self, charlen, embedding_dim, num_filters):
        super().__init__()
        self.embedding = nn.Embedding(charlen, embedding_dim)

        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=4)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=6)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size=8)

        """self.fc1 = nn.Linear(((signal_len - 15) // 3) * num_filters * 3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)"""

        self.maxpooling = nn.MaxPool1d(3)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_e = self.embedding(x)
        x = torch.transpose(x_e, 1, 2)

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.maxpooling(x)

        """x = torch.flatten(x, 1)
  
        x = self.fc1(x)
        x = self.relu(x)
  
        x = F.dropout(x, p=0.1, training=self.training)
  
        x = self.fc2(x)
        x = self.relu(x)
  
        x = F.dropout(x, p = 0.1, training=self.training)
  
        x = self.fc3(x)
        x = self.relu(x)
  
        x = self.fc4(x)"""

        return x


class DTAModel(nn.Module):
    def __init__(self, druglen, protlen):
        super().__init__()
        self.drugmodel = DrugModel(62, 128, 32, druglen)
        # self.protmodel = DrugProtModel(25, 128, 32)
        self.lstm_model = DrugProtLSTM(256, 25, 128)
        self.fc1 = nn.Linear(96 * (((druglen - 15) // 3) + ((protlen - 15) // 3)), 1024)
        self.fc_lstm = nn.Linear(96 + 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, drug, prot):
        drug_rep = self.drugmodel(drug)
        drug_rep = drug_rep.squeeze(2)
        # prot_rep = self.protmodel(prot)
        prot_rep = self.lstm_model(prot)

        x = torch.cat([drug_rep, prot_rep], 1)

        x = torch.flatten(x, 1)

        # x = self.fc1(x)
        x = self.fc_lstm(x)
        x = self.relu(x)

        x = F.dropout(x, p=0.2, training=self.training)

        x = self.fc2(x)
        x = self.relu(x)

        x = F.dropout(x, p=0.2, training=self.training)

        x = self.fc3(x)
        x = self.relu(x)

        x = F.dropout(x, p=0.1, training=self.training)

        x = self.fc4(x)

        return x
