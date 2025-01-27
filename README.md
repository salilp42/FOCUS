# SaccadeNet: Bio-Inspired Visual Attention for Medical Image Analysis

Official implementation of "SaccadeNet: A Bio-Inspired Deep Learning Architecture for Medical Image Analysis" (Nature, 2025).

## Overview

SaccadeNet is a novel deep learning architecture that incorporates biological vision principles:
- M/P/K pathways mimicking the human visual system
- Saccadic attention mechanism
- Predictive coding with GRU-based integration
- Support for both 2D and 3D medical imaging data

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from saccadenet import EnhancedFocusNet2D

# Create model
model = EnhancedFocusNet2D(
    in_channels=1,
    num_classes=10,
    num_saccades=2
)

# Train on MedMNIST
python scripts/train.py --dataset pathmnist --epochs 100
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{saccadenet2025,
  title={SaccadeNet: A Bio-Inspired Deep Learning Architecture for Medical Image Analysis},
  author={[Author list]},
  journal={Nature},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
