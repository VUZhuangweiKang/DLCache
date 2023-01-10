import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_features, hiddens, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        
        h = input_features
        for hidden in hiddens:
            self.layers.extend([
                nn.BatchNorm1d(h),
                nn.Linear(in_features=h, out_features=hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ])
            h =  hidden
        
        self.layers.extend([
            nn.Linear(in_features=h, out_features=1),
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def save(self, model_path):
        torch.save(self, model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        self.load_state_dict(model)
        self.eval()
        return model
      
            
        
        