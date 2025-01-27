"""
Visualization script for FocusNet models.

Example usage:
    python visualize.py --model_path results/pathmnist_latest/best_model.pth --dataset pathmnist
"""

import os
import argparse
import torch
import medmnist
from medmnist import INFO

from focusnet import EnhancedFocusNet2D, EnhancedFocusNet3D
from focusnet.utils import plot_attention_maps, measure_receptive_field

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize FocusNet attention")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to trained model")
    parser.add_argument("--dataset", type=str, required=True,
                      help="MedMNIST dataset name")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                      help="Output directory")
    parser.add_argument("--n_samples", type=int, default=4,
                      help="Number of samples to visualize")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataset info
    info = INFO[args.dataset]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    # Load dataset
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(split='test', download=True)
    
    # Create and load model
    is_3d = '3d' in args.dataset.lower()
    ModelClass = EnhancedFocusNet3D if is_3d else EnhancedFocusNet2D
    
    model = ModelClass(
        in_channels=n_channels,
        num_classes=n_classes,
        is_multilabel=(info['task'] == 'multi-label')
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Plot attention maps
    print("Generating attention visualizations...")
    plot_attention_maps(
        model, test_dataset,
        args.output_dir,
        n_samples=args.n_samples
    )
    
    # Analyze receptive field
    print("Analyzing receptive field...")
    rf_data = measure_receptive_field(model, device, is_3d)
    
    # Save results
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
