import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoisyCrossEntropyLoss(nn.Module):
    """Enhanced Noisy Cross Entropy Loss with adaptive noise detection"""
    def __init__(self, noise_prob=0.2, num_classes=6, alpha=0.1, beta=1.0):
        super(NoisyCrossEntropyLoss, self).__init__()
        self.noise_prob = noise_prob
        self.num_classes = num_classes
        self.alpha = alpha  # Weight for regularization
        self.beta = beta    # Weight for confidence penalty
        
    def forward(self, outputs, targets, epoch=0):
        # Standard cross entropy
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        
        # Get prediction probabilities
        probs = F.softmax(outputs, dim=1)
        pred_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
        
        # Confidence-based weighting (lower weight for low-confidence predictions)
        confidence_weights = torch.clamp(pred_probs, min=0.01)
        
        # Adaptive noise handling based on training progress
        if epoch < 10:
            # Early training: use standard CE with confidence weighting
            weighted_loss = ce_loss * confidence_weights
        else:
            # Later training: apply noise-robust modifications
            # Symmetric label smoothing
            smooth_targets = torch.zeros_like(probs)
            smooth_targets.fill_(self.noise_prob / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.noise_prob)
            
            smooth_loss = -torch.sum(smooth_targets * torch.log(probs + 1e-8), dim=1)
            
            # Combine losses based on confidence
            weighted_loss = confidence_weights * ce_loss + (1 - confidence_weights) * smooth_loss
            
        # Add entropy regularization to prevent overconfident predictions
        entropy_reg = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        total_loss = weighted_loss - self.alpha * entropy_reg
        
        return total_loss.mean()

class AdaptiveNoisyLoss(nn.Module):
    """Adaptive loss that detects noise type and adjusts accordingly"""
    def __init__(self, num_classes=6, warmup_epochs=10):
        super(AdaptiveNoisyLoss, self).__init__()
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs
        self.noise_detection_buffer = []
        self.detected_noise_type = 'unknown'
        
    def detect_noise_pattern(self, outputs, targets):
        """Detect if noise is symmetric or asymmetric based on loss patterns"""
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            pred_classes = torch.argmax(probs, dim=1)
            
            # Calculate per-class error rates
            class_errors = {}
            for c in range(self.num_classes):
                mask = (targets == c)
                if mask.sum() > 0:
                    error_rate = (pred_classes[mask] != c).float().mean().item()
                    class_errors[c] = error_rate
            
            # If error rates are similar across classes, likely symmetric noise
            if len(class_errors) > 1:
                error_values = list(class_errors.values())
                error_std = np.std(error_values)
                error_mean = np.mean(error_values)
                
                if error_std / (error_mean + 1e-8) < 0.3:  # Low coefficient of variation
                    return 'symmetric'
                else:
                    return 'asymmetric'
        return 'unknown'
    
    def forward(self, outputs, targets, epoch=0):
        # Detect noise pattern during warmup
        if epoch < self.warmup_epochs:
            noise_type = self.detect_noise_pattern(outputs, targets)
            self.noise_detection_buffer.append(noise_type)
            
            if len(self.noise_detection_buffer) >= 5:
                # Determine predominant noise type
                from collections import Counter
                noise_counts = Counter(self.noise_detection_buffer[-5:])
                self.detected_noise_type = noise_counts.most_common(1)[0][0]
        
        # Apply appropriate loss based on detected noise type
        if self.detected_noise_type == 'symmetric':
            return self._symmetric_noise_loss(outputs, targets)
        elif self.detected_noise_type == 'asymmetric':
            return self._asymmetric_noise_loss(outputs, targets)
        else:
            # Default to robust loss during detection phase
            return self._robust_loss(outputs, targets)
    
    def _symmetric_noise_loss(self, outputs, targets):
        """Loss function optimized for symmetric noise"""
        # Label smoothing for symmetric noise
        probs = F.softmax(outputs, dim=1)
        smooth_targets = torch.zeros_like(probs)
        smooth_targets.fill_(0.1 / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 0.9)
        
        loss = -torch.sum(smooth_targets * torch.log(probs + 1e-8), dim=1)
        return loss.mean()
    
    def _asymmetric_noise_loss(self, outputs, targets):
        """Loss function optimized for asymmetric noise"""
        # Use focal loss variant for asymmetric noise
        probs = F.softmax(outputs, dim=1)
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        
        # Get probabilities for correct class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze()
        
        # Focal loss with dynamic focusing
        alpha = 0.25
        gamma = 2.0
        focal_weight = alpha * (1 - pt) ** gamma
        
        return (focal_weight * ce_loss).mean()
    
    def _robust_loss(self, outputs, targets):
        """Robust loss for unknown noise patterns"""
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        
        # Truncated loss - ignore samples with very high loss (likely mislabeled)
        sorted_loss, _ = torch.sort(ce_loss)
        truncate_idx = int(0.8 * len(sorted_loss))  # Keep 80% of samples
        truncated_loss = sorted_loss[:truncate_idx]
        
        return truncated_loss.mean()

class VAEReconstructionLoss(nn.Module):
    """Reconstruction loss for Variational Autoencoder component"""
    def __init__(self, beta=1.0):
        super(VAEReconstructionLoss, self).__init__()
        self.beta = beta
        
    def forward(self, recon_adj, orig_adj, mu, logvar):
        # Reconstruction loss (binary cross entropy for adjacency matrix)
        recon_loss = F.binary_cross_entropy_with_logits(recon_adj, orig_adj, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss