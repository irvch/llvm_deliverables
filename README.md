# BERT and Quaternion Transformer Models for Sentiment Analysis

This repository contains implementations comparing different transformer architectures for financial sentiment analysis, including a standard BERT baseline and novel quaternion-based transformer models.

## Overview

The project evaluates three different approaches to financial sentiment analysis:

1. **Pretrained BERT** - Fine-tuned BERT-base-uncased model
2. **Partial Quatformer** - Hybrid model with quaternion attention and standard feed-forward layers
3. **Full Quatformer** - Complete quaternion-based transformer with quaternion feed-forward networks

## Dataset

The models are trained and evaluated on the **Financial Sentiment Analysis** dataset from HuggingFace (`sjyuxyz/financial-sentiment-analysis`) with three classes:
- Neutral (0)
- Positive (1)  
- Negative (2)

### Dataset Statistics
- Training set: 80,029 samples
- Validation set: 10,004 samples
- Test set: 10,004 samples

## Model Architecture

### Quaternion Transformers

The quaternion-based models introduce several novel components:

- **Quaternion Algebra**: Represents each vector as a quaternion with real (r), i, j, k components
- **Quaternion Attention**: Multi-head self-attention using Hamilton product for quaternion computations
- **Quaternion Feed-Forward Networks**: Optional quaternion transformations in the FFN layers

#### Key Components

1. **Quaternion Class**: Implements quaternion operations including Hamilton product, normalization, and conjugation
2. **QuaternionTransformation**: Linear transformations in quaternion space
3. **QuaternionSelfAttention**: Self-attention mechanism using quaternion mathematics
4. **MultiHeadQuaternionSelfAttention**: Multi-head variant of quaternion attention

Key features:
- More efficient representation with 4-component quaternions
- Novel attention mechanism using quaternion mathematics
- Maintains comparable performance to BERT with different parameter efficiency

## Results

| Model | Parameters | Test Accuracy | Training Time | F1-Score | Notes |
|-------|------------|---------------|---------------|----------|-------|
| Pretrained BERT | ~109M | 91.48% | ~21.5 min | 0.9146 | Baseline model |
| Full Quatformer | ~27M | 88.58% | ~96.9 min | 0.8880 | Complete quaternion implementation |
| Partial Quatformer | ~70M | 87.81% | ~79.0 min | 0.8780 | Quaternion attention + standard FFN |

### Per-Class Performance (BERT)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| Neutral | 0.9054 | 0.8687 | 0.8867 | 2,094 |
| Positive | 0.9189 | 0.9426 | 0.9306 | 5,038 |
| Negative | 0.9140 | 0.8997 | 0.9068 | 2,872 |

## Key Findings

1. **Parameter Efficiency**: The full quatformer achieves competitive performance with ~75% fewer parameters than BERT
2. **Computational Trade-off**: While more parameter-efficient, quaternion operations increase training time due to complex Hamilton product computations
3. **Architecture Impact**: The quaternion attention mechanism maintains strong performance even with standard FFN layers
4. **Memory Efficiency**: Quaternion models require less memory despite longer training times


### Running the Models

1. **BERT Baseline**:
```python
# Open and run llvm_bert_benchmark.ipynb
# The notebook will automatically:
# - Download the dataset
# - Preprocess the text data
# - Train the BERT model
# - Evaluate on test set
```

2. **Quaternion Models**:
```python
# Open llvm_final_quatformer_text.ipynb
# Toggle between full/partial quatformer by commenting/uncommenting 
# the FFN implementation in TransformerBlockQuaternions class

# For Full Quatformer (use quaternion FFN):
# self.ffnn = nn.Sequential(
#     QuaternionTransformation(dmodel, ffnn_hidden_size),
#     nn.ReLU(),
#     nn.Dropout(dropout),
#     QuaternionTransformation(ffnn_hidden_size, dmodel)
# )

# For Partial Quatformer (use standard FFN):
# self.ffnn = nn.Sequential(
#     nn.Linear(dmodel, ffnn_hidden_size),
#     nn.ReLU(),
#     nn.Dropout(dropout),
#     nn.Linear(ffnn_hidden_size, dmodel)
# )
```

### Notebook Contents

#### `llvm_bert_benchmark.ipynb`
- TextPreprocessor class for data cleaning and tokenization
- BERT model loading and fine-tuning
- Training and evaluation loops
- Performance metrics and confusion matrix

#### `llvm_final_quatformer_text.ipynb`
- Quaternion mathematics implementation
- Quaternion attention mechanisms
- Custom transformer blocks
- Learnable positional encoding
- Comprehensive training pipeline

## Implementation Details

### Text Preprocessing
- HTML tag removal using BeautifulSoup
- Lowercase conversion
- Punctuation removal
- BERT tokenization with max length 50 tokens
- Vectorized operations for efficiency

### Model Configuration

#### BERT Model
- Model: `bert-base-uncased`
- Hidden size: 768
- Number of layers: 12
- Attention heads: 12

#### Quaternion Models
- Input dimension: 256 (using `google/bert_uncased_L-4_H-256_A-4`)
- Number of transformer blocks: 27
- Attention heads: 8
- Hidden dimension: 4096
- Dropout: 0.1

### Training Configuration
- Batch size: 64
- Learning rate: 2e-5
- Epochs: 8
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Pooling: Max pooling for sequence aggregation
