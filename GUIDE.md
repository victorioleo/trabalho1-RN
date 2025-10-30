# CIFAR-100 Training Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/victorioleo/trabalho1-RN.git
cd trabalho1-RN

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
# Test label mappings
python test_labels.py

# Check model architectures
python example.py

# Compare different backbones
python check_backbones.py
```

### 3. Train Models

#### Train 20 Superclasses Model
```bash
python train.py --model_type coarse --epochs 100
```

#### Train 100 Classes Model
```bash
python train.py --model_type fine --epochs 100
```

#### Train Multi-head Model
```bash
python train.py --model_type multihead --epochs 100
```

## Understanding the Models

### Model 1: Coarse (20 Superclasses)

This model classifies CIFAR-100 images into 20 broad categories:
- aquatic mammals, fish, flowers, food containers, etc.
- Simpler task, higher expected accuracy
- Output: 20 logits (one per superclass)

### Model 2: Fine (100 Classes)

This model classifies CIFAR-100 images into 100 specific categories:
- apple, beaver, bicycle, boy, etc.
- More challenging task
- Output: 100 logits (one per class)

### Model 3: Multi-head

This model performs both classifications simultaneously:
- Two separate classification heads
- Shared feature extractor (backbone)
- Combined loss: `loss_total = loss_coarse + loss_fine`
- Outputs: 20 logits + 100 logits

**Advantages:**
- The model learns to recognize both general and specific features
- The coarse task can help regularize the fine task
- More efficient than training two separate models

## Training Details

### Default Hyperparameters

- **Backbone**: mobilenetv3_small_100 (~1.5M parameters)
- **Optimizer**: SGD with momentum=0.9
- **Learning Rate**: 0.1 (with cosine annealing)
- **Weight Decay**: 5e-4
- **Batch Size**: 128
- **Epochs**: 100

### Data Augmentation

Training uses:
- Random crop (32x32 with padding=4)
- Random horizontal flip
- Normalization with CIFAR-100 mean/std

### Customization

```bash
# Use a larger backbone
python train.py --model_type multihead --backbone mobilenetv3_large_100

# Adjust hyperparameters
python train.py --model_type fine --epochs 200 --lr 0.05 --batch_size 64

# Use pretrained weights (if available)
python train.py --model_type coarse --pretrained
```

## Model Outputs

### Coarse Model
```python
import torch
from models import create_model

model = create_model('coarse')
image = torch.randn(1, 3, 32, 32)
output = model(image)  # Shape: [1, 20]
predicted_superclass = output.argmax(1)
```

### Fine Model
```python
model = create_model('fine')
image = torch.randn(1, 3, 32, 32)
output = model(image)  # Shape: [1, 100]
predicted_class = output.argmax(1)
```

### Multi-head Model
```python
model = create_model('multihead')
image = torch.randn(1, 3, 32, 32)
coarse_out, fine_out = model(image)  # Shapes: [1, 20], [1, 100]
predicted_superclass = coarse_out.argmax(1)
predicted_class = fine_out.argmax(1)
```

## Expected Performance

Approximate accuracy ranges (vary by backbone and training):

| Model Type | Expected Test Accuracy |
|------------|------------------------|
| Coarse (20 classes) | 65-75% |
| Fine (100 classes) | 55-65% |
| Multi-head (coarse) | 65-75% |
| Multi-head (fine) | 55-65% |

Note: These are estimates. Actual performance depends on:
- Chosen backbone architecture
- Training hyperparameters
- Data augmentation
- Training duration

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python train.py --model_type multihead --batch_size 64
```

### Slow Training

- Reduce number of workers: `--num_workers 2`
- Use smaller batch size
- Reduce number of epochs for testing

### Dataset Download Issues

If automatic download fails:
1. Manually download CIFAR-100 from https://www.cs.toronto.edu/~kriz/cifar.html
2. Extract to `./data/cifar-100-python/`
3. Run training again

## File Structure

```
trabalho1-RN/
├── README.md              # Main documentation
├── GUIDE.md              # This file - detailed guide
├── requirements.txt      # Dependencies
├── .gitignore           # Git ignore rules
├── cifar100_labels.py   # Label mappings
├── dataset.py           # Dataset utilities
├── models.py            # Model definitions
├── train.py             # Training script
├── config.py            # Configuration examples
├── example.py           # Usage examples
├── test_labels.py       # Label mapping tests
├── check_backbones.py   # Architecture comparison
├── data/                # Dataset (auto-downloaded)
└── checkpoints/         # Saved models
```

## Advanced Usage

### Loading Saved Models

```python
import torch
from models import create_model

# Create model
model = create_model('multihead', model_name='mobilenetv3_small_100')

# Load checkpoint
checkpoint_path = 'checkpoints/multihead_mobilenetv3_small_100_best.pth'
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Use for inference
```

### Custom Training Loop

See `train.py` for the complete training implementation. Key functions:
- `train_epoch_coarse()` - Training loop for coarse model
- `train_epoch_fine()` - Training loop for fine model
- `train_epoch_multihead()` - Training loop for multi-head model

### Trying Different Architectures

```bash
# List available timm models (many options!)
python -c "import timm; print(timm.list_models())"

# Check if a model fits the 10M parameter limit
python -c "
from models import create_model, count_parameters
model = create_model('multihead', model_name='YOUR_MODEL_NAME')
print(f'Parameters: {count_parameters(model):,}')
"
```

## Citation

If you use this code for academic work, please cite:

```
CIFAR-100 Training Framework
Trabalho 1 - Redes Neurais
FACOM/UFMS, 2024
```

## License

Educational use - FACOM/UFMS
