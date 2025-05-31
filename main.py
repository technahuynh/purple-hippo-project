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
from sklearn.metrics import f1_score

# Load utility functions from cloned repository
from src.loadData import GraphDataset
from src.utils import set_seed

# Import the new models and losses
from src.enhanced_loss import NoisyCrossEntropyLoss, AdaptiveNoisyLoss
from src.vgae_model import VGAE_Classifier, EnhancedVGAE

# Set the random seed
set_seed()

def add_zeros(data):
    """Add zero node features if they don't exist"""
    if not hasattr(data, 'x') or data.x is None:
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def compute_basic_metrics(predictions, targets):
    """Compute only essential metrics: accuracy and F1 macro"""
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    accuracy = np.mean(predictions == targets)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    
    return accuracy, f1_macro

def train_vgae(data_loader, model, optimizer, criterion, device, current_epoch):
    """Simplified training function for VGAE"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    for data in tqdm(data_loader, desc="Training", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, EnhancedVGAE):
            pred, mu, logvar, noise_conf = model(data, use_prototypes=False)
        else:
            pred, mu, logvar, noise_conf = model(data)
        
        # Classification loss
        if hasattr(criterion, 'forward') and 'epoch' in criterion.forward.__code__.co_varnames:
            loss = criterion(pred, data.y, epoch=current_epoch)
        else:
            loss = criterion(pred, data.y)
        
        # Add simple KL divergence for VGAE
        if mu.numel() > 0 and logvar.numel() > 0:
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss + 0.01 * kl_loss  # Small weight for KL term
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for metrics
        pred_classes = torch.argmax(pred, dim=1)
        all_predictions.extend(pred_classes.cpu().numpy())
        all_targets.extend(data.y.cpu().numpy())
    
    accuracy, f1_macro = compute_basic_metrics(all_predictions, all_targets)
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy, f1_macro

def evaluate_vgae(data_loader, model, device, calculate_metrics=True):
    """Simplified evaluation function"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_confidence = []
    total_loss = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", unit="batch"):
            data = data.to(device)
            
            if isinstance(model, EnhancedVGAE):
                pred, _, _, _ = model(data, use_prototypes=False)
            else:
                pred, _, _, _ = model(data)
            
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
        accuracy, f1_macro = compute_basic_metrics(all_predictions, all_targets)
        avg_loss = total_loss / len(data_loader)
        return avg_loss, accuracy, f1_macro, all_predictions, all_confidence
    
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

def plot_training_progress(train_loss, val_loss, train_f1, val_f1, output_dir):
    """Plot essential training progress"""
    epochs = range(1, len(train_loss) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(epochs, train_loss, label='Train Loss', color='blue')
    ax1.plot(epochs, val_loss, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # F1 plot
    ax2.plot(epochs, train_f1, label='Train F1', color='blue')
    ax2.plot(epochs, val_f1, label='Val F1', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Macro')
    ax2.set_title('F1 Macro Score')
    ax2.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
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

    print(f"Model: {args.model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    logging.getLogger().addHandler(logging.StreamHandler())

    # Define checkpoint path
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")

    # Training phase
    if args.train_path:
        print("Starting training...")
        
        # Create datasets
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Training tracking
        best_val_f1 = 0.0
        train_losses, val_losses = [], []
        train_f1s, val_f1s = [], []

        # Training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train
            train_loss, train_acc, train_f1 = train_vgae(
                train_loader, model, optimizer, criterion, device, epoch
            )
            
            # Validate
            val_loss, val_acc, val_f1, _, _ = evaluate_vgae(
                val_loader, model, device, calculate_metrics=True
            )

            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

            # Log progress
            print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
            logging.info(f"Epoch {epoch + 1} - Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

            # Update scheduler
            scheduler.step(val_f1)
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model saved! F1: {best_val_f1:.4f}")

            # Early stopping
            if epoch > args.patience and all(
                val_f1s[-1] <= val_f1s[-(i+1)] 
                for i in range(1, min(args.patience, len(val_f1s)))
            ):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Plot training progress
        plot_training_progress(train_losses, val_losses, train_f1s, val_f1s, 
                             os.path.join(logs_folder, "plots"))
        
        print(f"Training completed. Best Val F1: {best_val_f1:.4f}")
    
    # Test phase
    print("\nGenerating test predictions...")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_predictions, test_confidence = evaluate_vgae(test_loader, model, device, calculate_metrics=False)
    
    save_predictions(test_predictions, args.test_path, test_confidence)
    print("Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlined VGAE training with minimal metrics")
    
    # Data arguments
    parser.add_argument("--train_path", type=str, default=None, help="Path to training dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test dataset")
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='vgae', choices=['vgae', 'enhanced_vgae'])
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gin'])
    parser.add_argument('--pooling', type=str, default='attention', choices=['mean', 'sum', 'max', 'attention'])
    parser.add_argument('--use_edge_attr', action='store_true', default=True)
    
    # Training arguments
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=20)
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='noisy_ce', choices=['ce', 'noisy_ce', 'adaptive'])
    parser.add_argument('--noise_prob', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    
    args = parser.parse_args()
    main(args)