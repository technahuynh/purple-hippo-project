import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import argparse
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns

# Load utility functions from cloned repository
from src.loadData import GraphDataset
from src.utils import set_seed

# Import the new models and losses
from src.enhanced_loss import NoisyCrossEntropyLoss, AdaptiveNoisyLoss, VAEReconstructionLoss
from src.vgae_model import VGAE_Classifier, EnhancedVGAE

# Set the random seed
set_seed()

def add_zeros(data):
    """Add zero node features if they don't exist"""
    if not hasattr(data, 'x') or data.x is None:
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def compute_metrics(predictions, targets, num_classes=6):
    """Compute comprehensive metrics including F1 scores"""
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    # Accuracy
    accuracy = np.mean(predictions == targets)
    
    # F1 scores
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(targets, predictions, average='micro', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    f1_per_class = f1_score(targets, predictions, average=None, labels=range(num_classes), zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class
    }

def train_vgae(data_loader, model, optimizer, criterion, device, 
               current_epoch, alpha=1.0, beta=0.1, gamma=0.01):
    """Enhanced training function for VGAE with multiple loss components"""
    model.train()
    total_loss = 0
    total_class_loss = 0
    total_recon_loss = 0
    total_noise_loss = 0
    
    all_predictions = []
    all_targets = []
    
    for batch_idx, data in enumerate(tqdm(data_loader, desc="Training", unit="batch")):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, EnhancedVGAE):
            pred, mu, logvar, noise_conf = model(data, use_prototypes=False)
        else:
            pred, mu, logvar, noise_conf = model(data)
        
        # Classification loss with noise adaptation
        if hasattr(criterion, 'forward') and 'epoch' in criterion.forward.__code__.co_varnames:
            class_loss = criterion(pred, data.y, epoch=current_epoch)
        else:
            class_loss = criterion(pred, data.y)
        
        # Reconstruction loss (simplified - using latent space similarity)
        batch_size = data.batch.max().item() + 1
        recon_loss = 0
        
        for i in range(batch_size):
            mask = data.batch == i
            mu_graph = mu[mask]
            logvar_graph = logvar[mask]
            
            # Simple reconstruction loss based on latent consistency
            if len(mu_graph) > 1:
                # Pairwise similarity in latent space should match graph structure
                latent_sim = torch.mm(mu_graph, mu_graph.t())
                # This is a simplified version - in practice you'd reconstruct actual adjacency
                recon_loss += torch.mean((latent_sim - torch.eye(len(mu_graph), device=device))**2)
        
        recon_loss = recon_loss / batch_size if batch_size > 0 else 0
        
        # KL divergence loss
        if mu.numel() > 0 and logvar.numel() > 0:
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            kl_loss = 0
        
        # Noise confidence regularization
        noise_reg = torch.mean(torch.abs(noise_conf - 0.5))  # Encourage confident noise detection
        
        # Combined loss
        total_batch_loss = (alpha * class_loss + 
                           beta * (recon_loss + kl_loss) + 
                           gamma * noise_reg)
        
        total_batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_batch_loss.item()
        total_class_loss += class_loss.item() if isinstance(class_loss, torch.Tensor) else class_loss
        total_recon_loss += (recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss)
        total_noise_loss += noise_reg.item()
        
        # Collect predictions for metrics
        pred_classes = torch.argmax(pred, dim=1)
        all_predictions.extend(pred_classes.cpu().numpy())
        all_targets.extend(data.y.cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    
    avg_loss = total_loss / len(data_loader)
    avg_class_loss = total_class_loss / len(data_loader)
    avg_recon_loss = total_recon_loss / len(data_loader)
    avg_noise_loss = total_noise_loss / len(data_loader)
    
    return avg_loss, avg_class_loss, avg_recon_loss, avg_noise_loss, metrics

def evaluate_vgae(data_loader, model, device, calculate_metrics=True):
    """Enhanced evaluation function with comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_confidence = []
    total_loss = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", unit="batch"):
            data = data.to(device)
            
            # Forward pass
            if isinstance(model, EnhancedVGAE):
                pred, mu, logvar, noise_conf = model(data, use_prototypes=False)
            else:
                pred, mu, logvar, noise_conf = model(data)
            
            pred_classes = torch.argmax(pred, dim=1)
            all_predictions.extend(pred_classes.cpu().numpy())
            
            if calculate_metrics:
                all_targets.extend(data.y.cpu().numpy())
                total_loss += criterion(pred, data.y).item()
                
            # Store confidence scores
            pred_probs = torch.softmax(pred, dim=1)
            max_probs = torch.max(pred_probs, dim=1)[0]
            all_confidence.extend(max_probs.cpu().numpy())
    
    if calculate_metrics:
        metrics = compute_metrics(all_predictions, all_targets)
        avg_loss = total_loss / len(data_loader)
        return avg_loss, metrics, all_predictions, all_confidence
    
    return all_predictions, all_confidence

def save_predictions(predictions, test_path, confidence_scores=None):
    """Save predictions with confidence scores"""
    script_dir = os.getcwd() 
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_data = {
        "id": test_graph_ids,
        "pred": predictions
    }
    
    if confidence_scores is not None:
        output_data["confidence"] = confidence_scores
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def plot_training_progress(train_metrics, val_metrics, output_dir):
    """Plot comprehensive training progress"""
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss plot
    axes[0, 0].plot(epochs, train_metrics['loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, val_metrics['loss'], label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    
    # F1 Macro
    axes[0, 1].plot(epochs, train_metrics['f1_macro'], label='Train F1 Macro', color='blue')
    axes[0, 1].plot(epochs, val_metrics['f1_macro'], label='Val F1 Macro', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Macro')
    axes[0, 1].set_title('F1 Macro Score')
    axes[0, 1].legend()
    
    # Accuracy
    axes[0, 2].plot(epochs, train_metrics['accuracy'], label='Train Accuracy', color='blue')
    axes[0, 2].plot(epochs, val_metrics['accuracy'], label='Val Accuracy', color='red')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Accuracy')
    axes[0, 2].legend()
    
    # F1 Weighted
    axes[1, 0].plot(epochs, train_metrics['f1_weighted'], label='Train F1 Weighted', color='blue')
    axes[1, 0].plot(epochs, val_metrics['f1_weighted'], label='Val F1 Weighted', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Weighted')
    axes[1, 0].set_title('F1 Weighted Score')
    axes[1, 0].legend()
    
    # Component losses (if available)
    if 'class_loss' in train_metrics:
        axes[1, 1].plot(epochs, train_metrics['class_loss'], label='Classification Loss', color='blue')
        axes[1, 1].plot(epochs, train_metrics['recon_loss'], label='Reconstruction Loss', color='green')
        axes[1, 1].plot(epochs, train_metrics['noise_loss'], label='Noise Regularization', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Components')
        axes[1, 1].set_title('Loss Components')
        axes[1, 1].legend()
    
    # F1 per class (latest epoch)
    if 'f1_per_class' in val_metrics and len(val_metrics['f1_per_class']) > 0:
        latest_f1_per_class = val_metrics['f1_per_class'][-1]
        axes[1, 2].bar(range(len(latest_f1_per_class)), latest_f1_per_class)
        axes[1, 2].set_xlabel('Class')
        axes[1, 2].set_ylabel('F1 Score')
        axes[1, 2].set_title('F1 Score per Class (Latest Epoch)')
        axes[1, 2].set_xticks(range(len(latest_f1_per_class)))
    
    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))

    # Initialize VGAE model
    if args.model_type == 'vgae':
        model = VGAE_Classifier(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_classes=6,
            gnn_type=args.gnn_type,
            pooling=args.pooling,
            use_edge_attr=args.use_edge_attr
        ).to(device)
    elif args.model_type == 'enhanced_vgae':
        model = EnhancedVGAE(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_classes=6,
            gnn_type=args.gnn_type,
            pooling=args.pooling,
            use_edge_attr=args.use_edge_attr
        ).to(device)
    else:
        raise ValueError(f'Invalid model type: {args.model_type}')

    print(f"Model initialized: {args.model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    # Initialize loss function
    if args.loss_type == 'noisy_ce':
        criterion = NoisyCrossEntropyLoss(
            noise_prob=args.noise_prob,
            num_classes=6,
            alpha=args.alpha,
            beta=args.beta
        )
    elif args.loss_type == 'adaptive':
        criterion = AdaptiveNoisyLoss(num_classes=6, warmup_epochs=args.warmup_epochs)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print(f"Loss function: {args.loss_type}")

    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO, 
        format='%(asctime)s - %(message)s',
        filemode='w'
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well

    # Define checkpoint path
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded best model from {checkpoint_path}")

    # Training phase
    if args.train_path:
        print("Starting training phase...")
        
        # Create train and validation datasets
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        # Training loop variables
        num_epochs = args.epochs
        best_val_f1 = 0.0   
        
        # Initialize metric tracking
        train_metrics = {
            'loss': [], 'class_loss': [], 'recon_loss': [], 'noise_loss': [],
            'accuracy': [], 'f1_macro': [], 'f1_weighted': [], 'f1_per_class': []
        }
        val_metrics = {
            'loss': [], 'accuracy': [], 'f1_macro': [], 'f1_weighted': [], 'f1_per_class': []
        }

        # Start training
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_class_loss, train_recon_loss, train_noise_loss, train_metric = train_vgae(
                train_loader, model, optimizer, criterion, device, epoch,
                alpha=args.loss_alpha, beta=args.loss_beta, gamma=args.loss_gamma
            )
            
            # Validation
            val_loss, val_metric, val_predictions, val_confidence = evaluate_vgae(
                val_loader, model, device, calculate_metrics=True
            )

            # Store metrics
            train_metrics['loss'].append(train_loss)
            train_metrics['class_loss'].append(train_class_loss)
            train_metrics['recon_loss'].append(train_recon_loss)
            train_metrics['noise_loss'].append(train_noise_loss)
            train_metrics['accuracy'].append(train_metric['accuracy'])
            train_metrics['f1_macro'].append(train_metric['f1_macro'])
            train_metrics['f1_weighted'].append(train_metric['f1_weighted'])
            train_metrics['f1_per_class'].append(train_metric['f1_per_class'])
            
            val_metrics['loss'].append(val_loss)
            val_metrics['accuracy'].append(val_metric['accuracy'])
            val_metrics['f1_macro'].append(val_metric['f1_macro'])
            val_metrics['f1_weighted'].append(val_metric['f1_weighted'])
            val_metrics['f1_per_class'].append(val_metric['f1_per_class'])

            # Logging
            log_msg = (f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train F1: {train_metric['f1_macro']:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val F1: {val_metric['f1_macro']:.4f}, "
                      f"Val Acc: {val_metric['accuracy']:.4f}")
            print(log_msg)
            logging.info(log_msg)

            # Learning rate scheduling
            scheduler.step(val_metric['f1_macro'])
            
            # Save best model based on F1 score
            if val_metric['f1_macro'] > best_val_f1:
                best_val_f1 = val_metric['f1_macro']
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated! F1: {best_val_f1:.4f}")
                logging.info(f"Best model saved with F1: {best_val_f1:.4f}")

            # Early stopping
            if epoch > args.patience and all(
                val_metrics['f1_macro'][-1] <= val_metrics['f1_macro'][-(i+1)] 
                for i in range(1, min(args.patience, len(val_metrics['f1_macro'])))
            ):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Plot training progress
        plot_training_progress(train_metrics, val_metrics, os.path.join(logs_folder, "plots"))
        
        # Final validation evaluation for confusion matrix
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        val_loss, val_metric, val_predictions, val_confidence = evaluate_vgae(
            val_loader, model, device, calculate_metrics=True
        )
        
        # Get validation targets
        val_targets = []
        for data in val_loader:
            val_targets.extend(data.y.numpy())
        
        # Plot confusion matrix
        plot_confusion_matrix(val_targets, val_predictions, 
                            os.path.join(logs_folder, "plots"), 
                            "Validation Confusion Matrix")
        
        # Print final classification report
        print("\nFinal Validation Results:")
        print(classification_report(val_targets, val_predictions, target_names=[f"Class {i}" for i in range(6)]))
        logging.info(f"Final validation F1 macro: {val_metric['f1_macro']:.4f}")
    
    # Test phase
    print("\nStarting test phase...")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Test samples: {len(test_dataset)}")

    # Load best model and generate predictions
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_predictions, test_confidence = evaluate_vgae(test_loader, model, device, calculate_metrics=False)
    
    # Save predictions
    save_predictions(test_predictions, args.test_path, test_confidence)
    print("Training and evaluation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate VGAE models on graph datasets with noisy labels.")
    
    # Data arguments
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='vgae', choices=['vgae', 'enhanced_vgae'],
                        help='Model type: vgae or enhanced_vgae (default: vgae)')
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension (default: 1)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension (default: 256)')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension (default: 128)')
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gin'],
                        help='GNN type: gcn or gin (default: gcn)')
    parser.add_argument('--pooling', type=str, default='attention', choices=['mean', 'sum', 'max', 'attention'],
                        help='Graph pooling method (default: attention)')
    parser.add_argument('--use_edge_attr', action='store_true', default=True,
                        help='Use edge attributes (default: True)')
    
    # Training arguments
    parser.add_argument('--device', type=int, default=0, help='Which GPU to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (default: 1e-5)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (default: 20)')
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='noisy_ce', choices=['ce', 'noisy_ce', 'adaptive'],
                        help='Loss function type (default: noisy_ce)')
    parser.add_argument('--noise_prob', type=float, default=0.2, help='Noise probability (default: 0.2)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for NoisyCE loss (default: 0.1)')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for NoisyCE loss (default: 1.0)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs for adaptive loss (default: 10)')
    
    # Loss weighting for VGAE
    parser.add_argument('--loss_alpha', type=float, default=1.0, help='Weight for classification loss (default: 1.0)')
    parser.add_argument('--loss_beta', type=float, default=0.1, help='Weight for reconstruction loss (default: 0.1)')
    parser.add_argument('--loss_gamma', type=float, default=0.01, help='Weight for noise regularization (default: 0.01)')
    
    args = parser.parse_args()
    main(args)