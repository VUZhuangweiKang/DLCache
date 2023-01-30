import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, ReLU

class Model(nn.Module):
    def __init__(self, input_features, 
                 hidden_size=128, 
                 num_layers=2, 
                 dropout_rate=0.3):
        super().__init__()
        self.lstm = LSTM(input_size=input_features, 
                         hidden_size=hidden_size, 
                         num_layers=num_layers,
                         dropout=dropout_rate,
                         batch_first=True)
        self.linear = Linear(hidden_size, input_features)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.linear(hn[0])
        return out

    def save(self, model_path):
        torch.save(self, model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        self.load_state_dict(model)
        self.eval()
        return model