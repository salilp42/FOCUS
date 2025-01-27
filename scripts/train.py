"""
Training script for FocusNet models.

Example usage:
    python train.py --dataset pathmnist --epochs 100 --batch_size 32
"""

import os
import sys
import argparse
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import medmnist
from medmnist import INFO

from focusnet import EnhancedFocusNet2D, EnhancedFocusNet3D
from focusnet.utils import FocalLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Train FocusNet on MedMNIST")
    parser.add_argument("--dataset", type=str, required=True,
                      help="MedMNIST dataset name")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                      help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="results",
                      help="Output directory")
    return parser.parse_args()

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        if isinstance(output, tuple):
            output = output[0]
            
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({'loss': total_loss/(batch_idx+1),
                         'acc': 100.*correct/total})
    
    return total_loss/len(dataloader), 100.*correct/total

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataloader."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if isinstance(output, tuple):
                output = output[0]
                
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
    return total_loss/len(dataloader), 100.*correct/total

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.dataset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset info
    info = INFO[args.dataset]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    # Load dataset
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', download=True)
    val_dataset = DataClass(split='val', download=True)
    test_dataset = DataClass(split='test', download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    
    # Create model
    is_3d = '3d' in args.dataset.lower()
    ModelClass = EnhancedFocusNet3D if is_3d else EnhancedFocusNet2D
    
    model = ModelClass(
        in_channels=n_channels,
        num_classes=n_classes,
        is_multilabel=(task == 'multi-label')
    ).to(device)
    
    # Setup training
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.epochs
        )
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f'\nEpoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(output_dir, 'best_model.pth'))
    
    # Final test
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'\nFinal Test Accuracy: {test_acc:.2f}%')

if __name__ == '__main__':
    main()
