# Fake News Detection using BERT + GPT-2

This project implements a hybrid deep learning model for fake news detection
by combining BERT and GPT-2 representations.

## Model Architecture
- BERT (bert-base-uncased) for contextual sentence understanding
- GPT-2 for generative-style semantic representation
- Concatenated embeddings → Dense layers → Binary classifier

## Datasets Used
- IFND (Indian Fake News Dataset)
- Fake News Incidents India (Excel dataset)

Note: Datasets are not included due to licensing restrictions.

## Features
- Binary classification: TRUE vs FAKE
- Hybrid transformer architecture
- Extensive evaluation:
  - Accuracy
  - Precision / Recall / F1
  - Confusion Matrix
- Custom news testing

## Tech Stack
- Python
- TensorFlow / Keras
- HuggingFace Transformers
- Scikit-learn
- Pandas / NumPy

## How to Run
1. Place datasets inside `data/`
2. Update dataset paths in `src/train_hybrid_model.py`
3. Run:
   ```bash
   python src/train_hybrid_model.py
