# DistilBERT Medical Text Summarization Model

## Overview
This model implements extractive summarization for MIMIC-III discharge summaries using DistilBERT (Bidirectional Encoder Representations from Transformers for Biomedical Text Mining). The model classifies sentences as "important" or "not important" based on medical features engineered from the data pipeline.

## Architecture

### Model Choice Rationale
- **DistilBERT**: DistilBERT (67MB) - faster training, similar architecture
- **Task**: Binary classification (important vs. not important sentences)

### How It Works
```
Medical Text → DistilBERT Encoder → Importance Classification → Extract Top Sentences → Summary
                      ↑
                47 Features Guide Training
```

## Data Integration

### Input Data
The model uses two processed datasets from our project data pipeline:
1. **mimic_features_advanced.csv**: 47 engineered features including:
   - Clinical indicators: `urgency_indicator`, `abnormal_lab_ratio`
   - Text metrics: `complexity_score`, `kw_medications`, `kw_symptoms`
   - Demographics: One-hot encoded age, gender, ethnicity

2. **processed_discharge_summaries.csv**: Cleaned medical text with:
   - Extracted sections (diagnosis, medications, follow-up)
   - Expanded medical abbreviations
   - Cleaned and normalized text

### Label Generation Strategy
Importance scores are calculated using domain knowledge:
```python
importance_score = (urgency_indicator * 0.4 + 
                   abnormal_lab_ratio * 0.3 + 
                   complexity_score * 0.3)
```
- High urgency → Higher importance weight (40%)
- Abnormal labs → Medical attention needed (30%)
- Complexity → Detailed documentation required (30%)

## Training Process

### Data Split
- Training: 4,080 samples (80%)
- Testing: 1,020 samples (20%)
- Stratified by importance labels to maintain balance

### Hyperparameters
```python
batch_size: 8 (reduce to 4 for memory constraints)
max_length: 512 tokens
epochs: 3
learning_rate: 2e-5 (default)
warmup_steps: 500
weight_decay: 0.01
```

### Training Arguments
- Evaluation Strategy: Per epoch
- Save Strategy: Best model based on eval loss
- Mixed Precision: Disabled for stability

## Key Differences from Original Notebook

### 1. Data Pipeline Integration
**Original**: 
```python
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
```
**My Implementation**:
```python
df_features = pd.read_csv('mimic_features_advanced.csv')
df_text = pd.read_csv('processed_discharge_summaries.csv')
df_complete = pd.merge(df_features, df_text, on='hadm_id')
```

### 2. Task Transformation
**Original**: Language modeling (predict next token)
```python
def group_texts(examples):
    # Concatenate and split into fixed blocks
    return {"input_ids": chunks, "labels": chunks}  # Same as input
```
**My Implementation**: Classification (identify important sentences)
```python
def prepare_summarization_data(df):
    # Create importance labels from medical features
    texts = df['cleaned_text']
    labels = (importance_score > median).astype(int)  # Binary
    return texts, labels
```

### 3. Model Architecture
**Original**: GPT-2 for generation
```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Causal language modeling head
```
**My Implementation**: DistilBERT for classification
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "DistilBERT",
    num_labels=2  # Binary classification
)
```

### 4. Feature Utilization
**Original**: No feature engineering, raw text only
**My Implementation**: 47 features guide importance scoring
- Medical keywords influence sentence selection
- Urgency indicators prioritize critical information
- Demographic features enable bias detection

## Running the Model

### Prerequisites
```bash
pip install transformers torch pandas scikit-learn mlflow
```


### Expected Output
```
Starting DistilBERT training script...
Loading data from pipeline...
Training samples: 4080, Test samples: 1020
Loading DistilBERT...
Starting training...
Epoch 1/3: loss=0.693, eval_loss=0.685
Epoch 2/3: loss=0.672, eval_loss=0.668
Epoch 3/3: loss=0.651, eval_loss=0.656
Model saved to ../models/biobert_summarizer
```
![ML FLow Output](./results/mlFlow_DistilBERT.png)

## Model Evaluation

### Metrics Tracked (via MLflow)
- Training/Evaluation Loss
- Accuracy on importance classification
- F1 Score for important sentence identification
- ROUGE scores for summary quality (future)

### Bias Analysis
The model uses demographic features to check for fairness:
```python
# Evaluate performance across groups
for demographic in ['gender', 'ethnicity', 'age_group']:
    group_performance = evaluate_by_group(predictions, demographic)
```

## Deployment

### Model Artifacts
- Saved to: `biobert_summarizer.py`
- MLflow Registry: `DistilBERT-Medical-Summarizer`
- Format: PyTorch state dict + tokenizer


