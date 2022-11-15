import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, ReLU

class Model(nn.Module):
    def __init__(self, input_features, dropout_rate=0.3):
        super().__init__()
        self.layers = [
            LSTM(input_features, 34),
            ReLU(),
            LSTM(34, 68),
            ReLU(),
            LSTM(68, 128),
            ReLU(),
            LSTM(128, 68),
            ReLU()
        ]
        self.last_lstm = LSTM(68, 34)
        self.output = Linear(34, 1)
            
        
    def forward(self, x):
        for layer in self.layers[:-3]:
            x = layer(x)
        x = self.last_lstm(x)[-1]
        x = ReLU(x)
        return self.output(x)
    
    def save(self, model_path):
        torch.save(self, model_path)
        
    def load(self, model_path):
        model = torch.load(model_path)
        self.load_state_dict(model)
        self.eval()
        return model
    
# n_features = 1
# n_steps = 10