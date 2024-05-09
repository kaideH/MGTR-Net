import torch
import torch.nn as nn
import torch.nn.init as init


class Network(torch.nn.Module):
    def __init__(self, in_feature=2048, num_layers=3, num_classes=7):
        super().__init__()
        self.lstm = nn.LSTM(in_feature, 512, num_layers=num_layers, batch_first=True, dropout=0.3)

        self.head = nn.Linear(512, num_classes)

        init.xavier_normal(self.lstm.all_weights[0][0])
        init.xavier_normal(self.lstm.all_weights[0][1])

    def forward(self, x): # input x = [B, T, D]
        self.lstm.flatten_parameters()
        x, hx = self.lstm(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = Network(in_feature=2048, num_layers=3, num_classes=7)

    x = torch.randn(2, 100, 2048)
    print(x.shape)
    
    out = model(x)
    print(out.shape)
