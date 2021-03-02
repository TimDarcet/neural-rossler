import torch
import torch.nn as nn


class FC_predictor(nn.Module):
    def __init__(self, n_hidden=2, hidden_size=5):


