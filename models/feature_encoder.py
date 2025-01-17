import math
import torch
import torch.nn as nn
import torchvision


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        self.pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(100000.0) / d_model))
        
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        
    def forward(self, frame_indexs):
        pe = self.pe[frame_indexs, :]
        return pe


class GTM_GCN(torch.nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super().__init__()

        from torch_geometric.nn import GCNConv # 放在开头import的话无法指定gpu
        self.conv1 = GCNConv(in_features, mid_features)
        self.conv2 = GCNConv(mid_features, mid_features)
        self.conv3 = GCNConv(mid_features, out_features)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight)
        return x


class Network(torch.nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # visual backbone
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.backbone = nn.DataParallel(self.backbone)

        # graph-based temporal modeling
        self.PositionalEncoding = PositionalEncoding(2048, 100000)
        self.gnn = GTM_GCN(2048, 1024, 1024)
        
        # cls head
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x, edge_index, frame_index): # input x = [B*T, C, H, W]
        # frame-level spatial features
        s_features = self.backbone(x) # s_features = [B*T, D]

        # clip-level temporal features
        pe = self.PositionalEncoding(frame_index).cuda()
        x = s_features + pe
        t_features = self.gnn(x, edge_index) # t_features = [B*T, D]

        # cls head
        x = self.head(t_features) # x = [B*T, C]

        return x, s_features, t_features
    
    def gtm_forward(self, s_features, edge_index, frame_index):
        
        # clip-level temporal features
        pe = self.PositionalEncoding(frame_index).cuda()
        x = s_features + pe
        t_features = self.gnn(x, edge_index) # t_features = [B*T, D]

        # cls head
        x = self.head(t_features) # x = [B*T, C]

        return x, t_features