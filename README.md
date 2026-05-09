# DA6401 Assignment 3 — Neural Machine Translation with Transformers

Implementation of the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017) for German → English translation using the Multi30k dataset.

## Project Structure

```
├── dataset.py        # Data loading, vocab building, tokenization
├── model.py          # Full Transformer architecture
├── lr_scheduler.py   # Noam learning rate scheduler
├── train.py          # Training loop, BLEU evaluation, checkpointing
├── experiments.py    # W&B ablation experiments
└── requirements.txt  # Dependencies
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m spacy download de_core_news_sm
```

## Training

```bash
python3 train.py
```

Trains for 20 epochs with Noam scheduler, label smoothing ε=0.1, and saves the best checkpoint by validation BLEU.

## Inference

```python
from model import Transformer

model = Transformer()   # downloads weights from Drive automatically
model.eval()
print(model.infer("Ein Mann sitzt auf einer Bank."))
# → "a man is sitting on a bench ."
```

The model weights are downloaded automatically from Google Drive via gdown inside `Transformer.__init__()`.

## Model Architecture

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| Encoder/Decoder layers (N) | 3 |
| Attention heads | 8 |
| d_ff | 512 |
| Dropout | 0.1 |
| Optimizer | Adam (β1=0.9, β2=0.98, ε=1e-9) |
| Warmup steps | 4000 |
| Label smoothing | 0.1 |
| Batch size | 128 |
| Epochs | 20 |

## Results

| Metric | Score |
|---|---|
| Validation BLEU | 36.54 |
| Test BLEU | 37.54 |

## Experiments (W&B Report)

Five ablation studies logged to Weights & Biases:

1. **Noam vs Fixed LR** :- Noam scheduler vs constant 1e-4 LR
2. **Scaling Factor** :- With vs without 1/√dk in attention, with gradient norm tracking
3. **Attention Head Visualisation** :- Heatmaps of last encoder layer attention heads
4. **Positional Encoding** :- Sinusoidal vs learned positional embeddings
5. **Label Smoothing** :- ε=0.1 vs ε=0.0, with prediction confidence tracking

## Dataset

Multi30k: 29,000 training / 1,014 validation / 1,000 test German-English sentence pairs.
Source: https://huggingface.co/datasets/bentrevett/multi30k

## Reference

Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf