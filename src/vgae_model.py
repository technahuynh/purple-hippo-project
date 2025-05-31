import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import math

# Import the custom convolution layers
from src.conv import GCNConv, GINConv

class GraphEncoder(nn.Module):
    """Graph encoder for VGAE"""
    def __init__(self, input_dim, hidden_dim, latent_dim, gnn_type='gcn', dropout=0.1):
        super(GraphEncoder, self).__init__()
        self.gnn_type = gnn_type
        
        if gnn_type == 'gcn':
            self.conv1 = GCNConv(hidden_dim)
            self.conv2 = GCNConv(hidden_dim)
        elif gnn_type == 'gin':
            self.conv1 = GINConv(hidden_dim)
            self.conv2 = GINConv(hidden_dim)
        
        # Project to latent space
        if gnn_type == 'gcn':
            self.conv_mu = GCNConv(latent_dim)
            self.conv_logvar = GCNConv(latent_dim)
        elif gnn_type == 'gin':
            self.conv_mu = GINConv(latent_dim)
            self.conv_logvar = GINConv(latent_dim)
        
        # Input projection layer to match hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        # Project input to hidden dimension
        h = self.input_proj(x)
        
        # First layer
        h = self.conv1(h, edge_index, edge_attr)
        h = self.batch_norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Second layer
        h = self.conv2(h, edge_index, edge_attr)
        h = self.batch_norm2(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Latent parameters
        mu = self.conv_mu(h, edge_index, edge_attr)
        logvar = self.conv_logvar(h, edge_index, edge_attr)
        
        return mu, logvar

class GraphDecoder(nn.Module):
    """Graph decoder for VGAE"""
    def __init__(self, latent_dim):
        super(GraphDecoder, self).__init__()
        self.latent_dim = latent_dim
        
    def forward(self, z):
        # Inner product decoder for adjacency matrix reconstruction
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj

class GraphClassifier(nn.Module):
    """Graph-level classifier"""
    def __init__(self, latent_dim, hidden_dim, num_classes, pooling='mean', dropout=0.3):
        super(GraphClassifier, self).__init__()
        self.pooling = pooling
        
        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, z, batch):
        if self.pooling == 'mean':
            graph_repr = global_mean_pool(z, batch)
        elif self.pooling == 'sum':
            graph_repr = global_add_pool(z, batch)
        elif self.pooling == 'max':
            graph_repr = global_max_pool(z, batch)
        elif self.pooling == 'attention':
            # Attention-based pooling
            att_weights = self.attention(z)
            att_weights = F.softmax(att_weights, dim=0)
            graph_repr = global_add_pool(z * att_weights, batch)
        
        return self.classifier(graph_repr)

class VGAE_Classifier(nn.Module):
    """Combined Variational Graph Autoencoder and Classifier"""
    def __init__(self, input_dim=1, hidden_dim=128, latent_dim=64, num_classes=6, 
             gnn_type='gcn', pooling='attention', use_edge_attr=True, dropout=0.1):
        super(VGAE_Classifier, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_edge_attr = use_edge_attr
        
        # Node feature encoder (if nodes have no features, use embeddings)
        self.node_encoder = nn.Embedding(1, input_dim)
        
        # VGAE components - custom convolutions handle edge attributes internally
        self.encoder = GraphEncoder(input_dim, hidden_dim, latent_dim, gnn_type, dropout) 
        self.decoder = GraphDecoder(latent_dim)

        # Classifier
        self.classifier = GraphClassifier(latent_dim, hidden_dim, num_classes, pooling, dropout)
    
        # Noise-aware components
        self.noise_detector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, data, return_latent=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Create node embeddings if x contains indices
        if x.dtype == torch.long:
            x = self.node_encoder(x)
            while x.dim() > 2:
                x = x.squeeze(-1)
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        
        # Encode to latent space - edge attributes are handled by custom convolutions
        mu, logvar = self.encoder(x, edge_index, edge_attr, batch)
        z = self.reparameterize(mu, logvar)
        
        # Classification
        pred = self.classifier(z, batch)
        
        # Noise confidence (higher = more confident the sample is clean)
        noise_conf = self.noise_detector(global_mean_pool(z, batch))
        
        if return_latent:
            return pred, mu, logvar, z, noise_conf
        
        return pred, mu, logvar, noise_conf
    
    def reconstruct(self, data):
        """Reconstruct adjacency matrix"""
        with torch.no_grad():
            pred, mu, logvar, z, _ = self.forward(data, return_latent=True)
            
            # Get unique graphs in batch
            batch_size = data.batch.max().item() + 1
            reconstructions = []
            
            for i in range(batch_size):
                mask = data.batch == i
                z_graph = z[mask]
                recon_adj = self.decoder(z_graph)
                reconstructions.append(recon_adj)
            
            return reconstructions

class EnhancedVGAE(VGAE_Classifier):
    """Enhanced VGAE with additional regularization and noise handling"""
    def __init__(self, *args, **kwargs):
        super(EnhancedVGAE, self).__init__(*args, **kwargs)
        
        # Additional components for noise robustness
        self.prototype_layer = nn.Parameter(
            torch.randn(kwargs.get('num_classes', 6), kwargs.get('latent_dim', 64))
        )
        nn.init.xavier_uniform_(self.prototype_layer)
        
        # Temperature for prototype-based classification
        self.temperature = nn.Parameter(torch.ones(1))
        
    def prototype_loss(self, z, batch, targets):
        """Prototype-based loss for noise robustness"""
        graph_repr = global_mean_pool(z, batch)
        
        # Compute distances to prototypes
        distances = torch.cdist(graph_repr, self.prototype_layer)
        logits = -distances / self.temperature
        
        return F.cross_entropy(logits, targets)
    
    def forward(self, data, use_prototypes=False):
        pred, mu, logvar, noise_conf = super().forward(data)
        
        if use_prototypes:
            _, _, _, z, _ = super().forward(data, return_latent=True)
            proto_logits = self.prototype_loss(z, data.batch, torch.zeros(data.batch.max().item() + 1, dtype=torch.long, device=data.batch.device))
            return pred, mu, logvar, noise_conf, proto_logits
        
        return pred, mu, logvar, noise_conf