import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import math

class GraphEncoder(nn.Module):
    """Graph encoder for VGAE"""
    def __init__(self, input_dim, hidden_dim, latent_dim, gnn_type='gcn', dropout=0.1):
        super(GraphEncoder, self).__init__()
        self.gnn_type = gnn_type
        
        if gnn_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == 'gin':
            self.conv1 = GINConv(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            self.conv2 = GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        # First layer
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        
        # Second layer
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)
        
        # Latent parameters
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        
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
        
        # Edge feature encoder
        if use_edge_attr:
            self.edge_encoder = nn.Linear(7, hidden_dim)  # Assuming 7-dim edge features
        
        # VGAE components
        encoder_input_dim = input_dim + (hidden_dim if use_edge_attr else 0)
        self.encoder = GraphEncoder(encoder_input_dim, hidden_dim, latent_dim, gnn_type, dropout) 
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
    
    def encode_edges(self, edge_attr, edge_index, num_nodes):
        """Encode edge features into node features"""
        if not self.use_edge_attr or edge_attr is None:
            return torch.zeros(num_nodes, 0, device=edge_index.device)
            
        edge_emb = self.edge_encoder(edge_attr)
        
        # Aggregate edge features to nodes
        node_edge_features = torch.zeros(num_nodes, edge_emb.size(1), device=edge_index.device)
        node_edge_features = node_edge_features.scatter_add(0, edge_index[0].unsqueeze(1).expand(-1, edge_emb.size(1)), edge_emb)
        node_edge_features = node_edge_features.scatter_add(0, edge_index[1].unsqueeze(1).expand(-1, edge_emb.size(1)), edge_emb)
        
        return node_edge_features
    
    def forward(self, data, return_latent=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_nodes = x.size(0)
        
        # Create node embeddings if x contains indices
        if x.dtype == torch.long:
            x = self.node_encoder(x)
            while x.dim() > 2:
                x = x.squeeze(-1)
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        
        # Encode edge features
        edge_features = self.encode_edges(edge_attr, edge_index, num_nodes)
        
        # Combine node and edge features
        if edge_features.size(1) > 0:
            x = torch.cat([x, edge_features], dim=1)
        
        # Encode to latent space
        mu, logvar = self.encoder(x, edge_index, batch)
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