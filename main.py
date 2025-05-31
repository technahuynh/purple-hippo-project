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

def train_vgae(data_loader, model, optimizer, criterion, device, current_epoch):
    """Simplified training function for VGAE"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0    
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
            # Clamp logvar to prevent exp() from exploding
            logvar = torch.clamp(logvar, min=-20, max=10)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = torch.clamp(kl_loss, max=100)  # Prevent KL from becoming too large
            loss = loss + 0.01 * kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pred_classes = torch.argmax(pred, dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}_ncod.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return avg_loss, accuracy 

def evaluate_vgae(data_loader, model, device, calculate_metrics=True):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", unit="batch"):
            data = data.to(device)
            if isinstance(model, EnhancedVGAE):
                pred, _, _, _ = model(data, use_prototypes=False)
            else:
                pred, _, _, _ = model(data)
            pred_classes = torch.argmax(pred, dim=1)
            predictions.extend(pred_classes.cpu().numpy())
            if calculate_metrics:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                targets.extend(data.y.cpu().numpy())
                total_loss += criterion(pred, data.y).item()

    if calculate_metrics:
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        return avg_loss, accuracy, f1_macro

    return predictions

def save_predictions(predictions, test_path):
    script_dir = os.getcwd() 
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    script_dir = os.getcwd() 
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3

    # Initialize VGAE model
    if args.model_type == 'vgae':
        model = VGAE_Classifier(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_classes=6,
            gnn_type=args.gnn_type,
            pooling=args.pooling,
            use_edge_attr=args.use_edge_attr,
            dropout=args.drop_ratio
        ).to(device)
    elif args.model_type == 'enhanced_vgae':
        model = EnhancedVGAE(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_classes=6,
            gnn_type=args.gnn_type,
            pooling=args.pooling,
            use_edge_attr=args.use_edge_attr,
            dropout=args.drop_ratio
        ).to(device)
    else:
        raise ValueError(f'Invalid model type: {args.model_type}')

    print(f"Model: {args.model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    
    # Initialize early stop
    patience = args.patience  # epochs to wait after no improvement
    delta = 0.01  # minimum change in the monitored metric
    best_val_loss = float("inf")  # best validation loss to compare against
    no_improvement_count = 0  # count of epochs with no improvement
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)

    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    # Define checkpoint path
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)
    num_epochs = args.epochs
    if num_checkpoints > 1:
        checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
    else:
        checkpoint_intervals = [num_epochs]

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
        train_accuracies, val_f1s = [], []

        # Training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train
            # train_loss, train_acc, train_f1 = 
            train_loss, train_acc = train_vgae(train_loader, model, optimizer, criterion, device, epoch)
            # Validate
            val_loss, val_acc, val_f1 = evaluate_vgae(val_loader, model, device, calculate_metrics=True)

            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_f1s.append(val_f1)

            # Log progress
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1 {val_f1:.4f}")
            if hasattr(logging, 'info'):
                logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1 {val_f1:.4f}")

            # Update scheduler
            scheduler.step(val_f1)
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}. F1: {best_val_f1:.4f}")
            # Save checkpoints
            if (epoch + 1) in checkpoint_intervals:
                checkpoint_file = f"{checkpoint_path}_epoch_{epoch + 1}.pth"
                torch.save(model.state_dict(), checkpoint_file)
                print(f"Checkpoint saved at {checkpoint_file}")
            # Early stopping
            early_stopping.check_early_stop(val_loss)
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch}")
                break

        # Plot training progress
        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))
        plot_training_progress(val_losses, val_f1s, os.path.join(logs_folder, "plotsVal"))
        print(f"Training completed. Best Val F1: {best_val_f1:.4f}")
    
    # Test phase
    print("\nGenerating test predictions...")
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_predictions = evaluate_vgae(test_loader, model, device, calculate_metrics=False)
    save_predictions(test_predictions, args.test_path)

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
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='Dropout ratio')
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gin'])
    parser.add_argument('--pooling', type=str, default='attention', choices=['mean', 'sum', 'max', 'attention'])
    parser.add_argument('--use_edge_attr', action='store_true', default=True)
    
    # Training arguments
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=20)
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='noisy_ce', choices=['ce', 'noisy_ce', 'adaptive'])
    parser.add_argument('--noise_prob', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    
    args = parser.parse_args()
    main(args)