# SML-X-ray: Siamese Meta-Learning for X-ray Analysis

A PyTorch implementation of a meta-learning framework that combines Siamese networks with Model-Agnostic Meta-Learning (MAML) for few-shot medical image analysis on chest X-rays.

## Overview

This project implements a novel approach to medical image analysis by combining:

- **Siamese Networks** for learning similarity representations in chest X-ray images
- **Meta-Learning (MAML)** for rapid adaptation to new diseases/conditions with limited data
- **CheXpert Dataset** integration for multi-label chest pathology classification

The system is designed to address the challenge of classifying rare diseases in medical imaging where labeled data is scarce, making it particularly useful for few-shot learning scenarios in healthcare.

## Architecture

### Core Components

1. **Siamese Network** (`models/siamese.py`)

   - Shared CNN backbone (ResNet18/50, DenseNet121)
   - Contrastive loss for similarity learning
   - Normalized embeddings for robust distance computation
2. **Meta-Learner** (`models/meta_learner.py`)

   - MAML implementation for few-shot adaptation
   - Inner-loop optimization for task-specific fine-tuning
   - Support for first-order approximation (FOMAML)
3. **Data Pipeline** (`data/`)

   - CheXpert dataset integration
   - Cached preprocessing for faster training
   - Balanced sampling strategies
   - Memory-mapped data loading for large datasets

## Requirements

```bash
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.2
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.4
higher>=0.2.1
tqdm>=4.56.0
pyyaml>=5.4.1
Pillow>=8.1.0
learn2learn>=0.1.5
streamlit  # For web application
```

## Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd SML-X-ray
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Download CheXpert Dataset:**
   - Download from [CheXpert official website](https://stanfordmlgroup.github.io/competitions/chexpert/)
   - Update dataset paths in `config/config.yaml`

## Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  dataset_path: /path/to/chexpert/dataset
  batch_size: 32
  image_size: [224, 224]
  use_cache: true
  
siamese:
  backbone: resnet50
  embedding_dim: 256
  margin: 1.0
  
meta:
  algorithm: maml
  inner_lr: 0.01
  meta_lr: 0.001
  adapt_steps: 5
  shots: 7

training:
  siamese_epochs: 50
  meta_episodes: 5000
  save_dir: checkpoints
```

## Usage

### Training

**Train both models sequentially:**

```bash
python main.py --mode all
```

**Train Siamese network only:**

```bash
python main.py --mode train_siamese
```

**Train meta-learner only (requires pretrained Siamese model):**

```bash
python main.py --mode train_meta
```

### Evaluation

**Evaluate both models:**

```bash
python main.py --mode evaluate
```

### Web Application

Launch the interactive Streamlit web interface:

```bash
streamlit run finalApp.py
```

The web app provides:

- Image upload and similarity comparison
- Real-time disease classification
- Visualization of embeddings and predictions
- Model performance metrics

### Individual Scripts

**Inference on single images:**

```bash
python inference.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

**Generate predictions:**

```bash
python predict.py --input path/to/test/images --output predictions.csv
```

## Project Structure

```
├── main.py                    # Main training and evaluation pipeline
├── finalApp.py               # Streamlit web application
├── inference.py              # Single image inference
├── predict.py                # Batch prediction script
├── requirements.txt          # Package dependencies
├── config/
│   └── config.yaml          # Configuration file
├── data/                    # Data processing modules
│   ├── dataloader.py       # Dataset classes and loaders
│   ├── preprocessed_cache.py # Caching utilities
│   └── memory_mapped.py    # Memory-efficient data loading
├── models/                  # Model architectures
│   ├── siamese.py          # Siamese network implementation
│   └── meta_learner.py     # MAML implementation
├── training/                # Training scripts
│   ├── train_siamese.py    # Siamese network training
│   └── train_meta.py       # Meta-learner training
├── evaluation/              # Evaluation utilities
│   └── evaluate.py         # Model evaluation functions
├── losses/                  # Loss functions
│   └── contrastive_loss.py # Contrastive loss implementation
├── utils/                   # Utility functions
│   ├── visualization.py    # Plotting and visualization
│   └── pt_to_npy.py        # Format conversion utilities
└── results/                 # Generated results and plots
    ├── confusion_matrix_*.png
    └── overall_performance.png
```

## Supported Pathologies

The model is trained to classify the following chest pathologies from the CheXpert dataset:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Enlarged Cardiomediastinum
- Fracture
- Lung Lesion
- Lung Opacity
- No Finding
- Pleural Effusion
- Pleural Other
- Pneumonia
- Pneumothorax
- Support Devices

## Performance Metrics

The system evaluates performance using:

- **Similarity Accuracy**: For image pair classification
- **Area Under Curve (AUC)**: For individual pathology classification
- **Average Precision (AP)**: For multi-label classification
- **Confusion Matrices**: For detailed per-class analysis

## Key Features

### Meta-Learning Capabilities

- **Few-shot Learning**: Rapidly adapt to new diseases with minimal examples
- **Task-Agnostic**: Framework can be applied to different medical imaging tasks
- **Gradient-Based**: Uses MAML for efficient meta-optimization

### Robust Architecture

- **Multiple Backbones**: Support for ResNet and DenseNet architectures
- **Caching System**: Efficient data loading and preprocessing
- **Memory Optimization**: Memory-mapped datasets for large-scale training

### Interactive Interface

- **Web Application**: User-friendly Streamlit interface
- **Real-time Inference**: Immediate results for uploaded images
- **Visualization Tools**: Built-in plotting and analysis capabilities

## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory**

   - Reduce batch size in `config.yaml`
   - Use gradient checkpointing
   - Enable mixed precision training
2. **Dataset Path Issues**

   - Verify paths in `config.yaml`
   - Ensure CheXpert dataset is properly downloaded
   - Check file permissions
3. **Model Loading Errors**

   - Ensure model checkpoints exist in `checkpoints/`
   - Verify model configuration matches training setup

## References

- **MAML**: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- **CheXpert**: [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels](https://arxiv.org/abs/1901.07031)
- **Siamese Networks**: [Siamese Neural Networks for One-shot Image Recognition](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

For more detailed information about specific components, please refer to the documentation in each module's docstrings.
