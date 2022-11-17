import torch
import torch.nn as nn


class ReviewClassifier(nn.Module):
    def __init__(self, num_features):
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, 
                             out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out

    def save(self, model_path):
        torch.save(self, model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        self.load_state_dict(model)
        self.eval()
        return model