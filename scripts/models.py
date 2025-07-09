import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GINConv, GATConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import global_mean_pool, global_add_pool
import numpy as np                                             
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, dim_in: int, dim_h: int, dim_out: int, num_layers: int, projection: bool = False):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers
        self.gcn_convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.gcn_convs.append(GCNConv(dim_in, dim_h))
        self.norms.append(BatchNorm1d(dim_h))
        
        # Additional layers - modify last layer based on projection
        for i in range(num_layers - 1):
            if i == num_layers - 2 and not projection:
                # Last layer outputs dim_out when no projection
                self.gcn_convs.append(GCNConv(dim_h, dim_out))
                self.norms.append(BatchNorm1d(dim_out))
            else:
                self.gcn_convs.append(GCNConv(dim_h, dim_h))
                self.norms.append(BatchNorm1d(dim_h))

        if projection:
            self.output_proj = nn.Sequential(
                Linear(dim_h, dim_out),
                BatchNorm1d(dim_out),
                nn.ReLU()
            )
        else:
            self.output_proj = None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        device = x.device
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        h = x
        for i in range(self.num_layers):
            h = self.gcn_convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            
        h = global_add_pool(h, batch)

        if self.output_proj is not None:
            h = self.output_proj(h)

        return h

class GraphSAGE(torch.nn.Module):
    def __init__(self, dim_in: int, dim_h: int, dim_out: int, num_layers: int, projection: bool = False):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers
        self.sage_convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.sage_convs.append(SAGEConv(dim_in, dim_h))
        self.norms.append(BatchNorm1d(dim_h))
        
        # Additional layers
        for i in range(num_layers - 1):
            if i == num_layers - 2 and not projection:
                # Last layer outputs dim_out when no projection
                self.sage_convs.append(SAGEConv(dim_h, dim_out))
                self.norms.append(BatchNorm1d(dim_out))
            else:
                self.sage_convs.append(SAGEConv(dim_h, dim_h))
                self.norms.append(BatchNorm1d(dim_h))

        if projection:
            self.output_proj = nn.Sequential(
                Linear(dim_h, dim_out),
                BatchNorm1d(dim_out),
                nn.ReLU()
            )
        else:
            self.output_proj = None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        device = x.device
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        h = x
        for i in range(self.num_layers):
            h = self.sage_convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            
        h = global_add_pool(h, batch)

        if self.output_proj is not None:
            h = self.output_proj(h)

        return h
    
class GIN(torch.nn.Module):
    def __init__(self, dim_in: int, dim_h: int, dim_out: int, num_layers: int, projection: bool = False):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers
        self.gin_convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.gin_convs.append(GINConv(Sequential(Linear(dim_in, dim_h), ReLU(), Linear(dim_h, dim_h))))
        self.norms.append(BatchNorm1d(dim_h))
        
        # Additional layers
        for i in range(num_layers - 1):
            if i == num_layers - 2 and not projection:
                # Last layer outputs dim_out when no projection
                self.gin_convs.append(GINConv(Sequential(Linear(dim_h, dim_out), ReLU(), Linear(dim_out, dim_out))))
                self.norms.append(BatchNorm1d(dim_out))
            else:
                # Other layers maintain the same hidden dimension
                self.gin_convs.append(GINConv(Sequential(Linear(dim_h, dim_h), ReLU(), Linear(dim_h, dim_h))))
                self.norms.append(BatchNorm1d(dim_h))

        if projection:
            self.output_proj = nn.Sequential(
                Linear(dim_h, dim_out),
                BatchNorm1d(dim_out),
                nn.ReLU()
            )
        else:
            self.output_proj = None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        device = x.device
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        h = x
        for i in range(self.num_layers):
            h = self.gin_convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            
        h = global_add_pool(h, batch)

        if self.output_proj is not None:
            h = self.output_proj(h)

        return h

class GraphClassifier(torch.nn.Module):
    def __init__(self, backbone_dim_in: int, backbone_dim_h: int, backbone_dim_out: int, 
                 backbone_num_layers: int, num_classes: int, backbone_type: str = 'sage',
                 projection: bool = False, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        
        # Choose backbone based on type
        if backbone_type.lower() == 'sage':
            self.backbone = GraphSAGE(
                dim_in=backbone_dim_in,
                dim_h=backbone_dim_h,
                dim_out=backbone_dim_out,
                num_layers=backbone_num_layers,
                projection=projection
            )
        elif backbone_type.lower() == 'gcn':
            self.backbone = GCN(
                dim_in=backbone_dim_in,
                dim_h=backbone_dim_h,
                dim_out=backbone_dim_out,
                num_layers=backbone_num_layers,
                projection=projection
            )
        elif backbone_type.lower() == 'gin':
            self.backbone = GIN(
                dim_in=backbone_dim_in,
                dim_h=backbone_dim_h,
                dim_out=backbone_dim_out,
                num_layers=backbone_num_layers,
                projection=projection
            )
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        self.classifier = nn.Sequential(
            Linear(backbone_dim_out, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        h = self.backbone(data)
        out = self.classifier(h)
        return out
    
    def get_embeddings(self, data):
        h = self.backbone(data)
        return h
    
class GraphSAGELayer(nn.Module):
    """使用PyTorch Geometric的GraphSAGE層 (without pooling)"""
    def __init__(self, dim_in: int, dim_h: int, dim_o:int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers
        self.sage_convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.sage_convs.append(SAGEConv(dim_in, dim_h))
        self.norms.append(BatchNorm1d(dim_h))
        
        # Additional layers
        for _ in range(num_layers - 2):
            self.sage_convs.append(SAGEConv(dim_h, dim_h))
            self.norms.append(BatchNorm1d(dim_h))
        
        if num_layers > 1:
            # Final layer
            self.sage_convs.append(SAGEConv(dim_h, dim_o))
            self.norms.append(BatchNorm1d(dim_o))

    def forward(self, x, edge_index):
        device = x.device
        edge_index = edge_index.to(device)
        
        h = x
        for i in range(self.num_layers):
            h = self.sage_convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)

        return h
    
class GATLayer(nn.Module):
    """使用PyTorch Geometric的GAT層 (without pooling)"""
    def __init__(self, dim_in: int, dim_h: int, dim_o: int, num_layers: int, heads=8, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers
        self.gat_convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer (with multiple attention heads)
        self.gat_convs.append(GATConv(dim_in, dim_h // heads, heads=heads, dropout=dropout))
        self.norms.append(BatchNorm1d(dim_h))
        
        # Additional layers
        for _ in range(num_layers - 2):
            self.gat_convs.append(GATConv(dim_h, dim_h // heads, heads=heads, dropout=dropout))
            self.norms.append(BatchNorm1d(dim_h))
        
        if num_layers > 1:
            # Final layer (typically with 1 head for output)
            self.gat_convs.append(GATConv(dim_h, dim_o, heads=1, concat=False, dropout=dropout))
            self.norms.append(BatchNorm1d(dim_o))

    def forward(self, x, edge_index):
        device = x.device
        edge_index = edge_index.to(device)
        
        h = x
        for i in range(self.num_layers):
            h = self.gat_convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.elu(h)  # ELU activation commonly used with GAT

        return h
    
class GINLayer(nn.Module):
    """使用PyTorch Geometric的GIN層 (without pooling)"""
    def __init__(self, dim_in: int, dim_h: int, dim_o: int, num_layers: int, eps=0.0, train_eps=True):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers
        self.gin_convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.mlps = torch.nn.ModuleList()
        
        # First layer
        # GIN requires an MLP for each layer
        self.mlps.append(nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.BatchNorm1d(dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_h)
        ))
        self.gin_convs.append(GINConv(self.mlps[0], eps=eps, train_eps=train_eps))
        self.norms.append(BatchNorm1d(dim_h))
        
        # Additional layers
        for i in range(num_layers - 2):
            self.mlps.append(nn.Sequential(
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h)
            ))
            self.gin_convs.append(GINConv(self.mlps[i+1], eps=eps, train_eps=train_eps))
            self.norms.append(BatchNorm1d(dim_h))
        
        if num_layers > 1:
            # Final layer
            self.mlps.append(nn.Sequential(
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_o)
            ))
            self.gin_convs.append(GINConv(self.mlps[-1], eps=eps, train_eps=train_eps))
            self.norms.append(BatchNorm1d(dim_o))

    def forward(self, x, edge_index):
        device = x.device
        edge_index = edge_index.to(device)
        
        h = x
        for i in range(self.num_layers):
            h = self.gin_convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)

        return h
    
class GCNLayer(nn.Module):
    """使用PyTorch Geometric的GCN層"""
    def __init__(self, dim_in: int, dim_h: int, dim_o: int, num_layers: int, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GNN layers
        self.gcn_convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.gcn_convs.append(GCNConv(dim_in, dim_h, improved=True))
        self.norms.append(BatchNorm1d(dim_h))
        
        # Additional layers
        for _ in range(num_layers - 2):
            self.gcn_convs.append(GCNConv(dim_h, dim_h, improved=True))
            self.norms.append(BatchNorm1d(dim_h))
        
        if num_layers > 1:
            # Final layer
            self.gcn_convs.append(GCNConv(dim_h, dim_o, improved=True))
            self.norms.append(BatchNorm1d(dim_o))

    def forward(self, x, edge_index):
        device = x.device
        edge_index = edge_index.to(device)
        
        h = x
        for i in range(self.num_layers):
            h = self.gcn_convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h
    
class CNNLayer(nn.Module):
    """使用PyTorch的CNN層"""
    def __init__(self, dim_in: int, dim_h: int, dim_o: int, num_layers: int, 
                 kernel_size=3, padding=1, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # CNN layers
        self.conv_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        # First layer
        self.conv_layers.append(nn.Conv2d(dim_in, dim_h, kernel_size=kernel_size, 
                                         padding=padding))
        self.norms.append(nn.BatchNorm2d(dim_h))
        
        # Additional layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(nn.Conv2d(dim_h, dim_h, kernel_size=kernel_size, 
                                             padding=padding))
            self.norms.append(nn.BatchNorm2d(dim_h))
        
        if num_layers > 1:
            # Final layer
            self.conv_layers.append(nn.Conv2d(dim_h, dim_o, kernel_size=kernel_size, 
                                             padding=padding))
            self.norms.append(nn.BatchNorm2d(dim_o))

    def forward(self, x):
        h = x
        for i in range(self.num_layers):
            h = self.conv_layers[i](h)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h

class GraphRelationNetwork(nn.Module):
    """基於GraphSAGE的關係網絡"""
    def __init__(self, input_dim, hidden_dim, out_dim, layer_num, model_type='GraphSAGE'):
        super(GraphRelationNetwork, self).__init__()
        if model_type == 'GraphSAGE':
            self.block = GraphSAGELayer(input_dim, hidden_dim, out_dim, num_layers=layer_num)
        elif model_type == 'GCN':
            self.block = GCNLayer(input_dim, hidden_dim, out_dim, num_layers=layer_num)
        elif model_type == 'GIN':
            self.block = GINLayer(input_dim, hidden_dim, out_dim, num_layers=layer_num)
        elif model_type == 'GAT':
            self.block = GATLayer(input_dim, hidden_dim, out_dim, num_layers=layer_num)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        h = self.block(x, edge_index)
        h = global_add_pool(h, batch)
        return self.fc(h)
    
class MLPRelationModule(nn.Module):
    """MLP-based Relation Module for Relation Network"""
    def __init__(self, dim_in: int, dropout: float = 0.2):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_in // 2, dim_in // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_in // 4, 1)
        )
    
    def forward(self, x):
        return self.fc(x)