import math
import copy
import os
import gdown
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# special token indices — defined here to avoid top-level dataset import
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    return torch.matmul(attn_w, V), attn_w


def make_src_mask(src, pad_idx=PAD_IDX):
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad_idx=PAD_IDX):
    tgt_len  = tgt.size(1)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal   = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
    return pad_mask | causal


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def split_heads(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int   = 18669,
        tgt_vocab_size: int   = 9797,
        d_model:        int   = 256,
        N:              int   = 3,
        num_heads:      int   = 8,
        d_ff:           int   = 512,
        dropout:        float = 0.1,
        max_len:        int   = 256,
        checkpoint_path: str  = "best_model.pt",
        gdrive_id:       str  = "1CtH2Ac29z0yifIay04HKuF_9BV4k6VpW",
    ):
        super().__init__()
        self.d_model        = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc   = PositionalEncoding(d_model, dropout, max_len)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder     = Encoder(enc_layer, N)
        self.decoder     = Decoder(dec_layer, N)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()
        self._load_vocab_and_tokenizers()

        # download weights from drive if not already present
        if checkpoint_path is not None:
            if gdrive_id is not None and not os.path.exists(checkpoint_path):
                gdown.download(id=gdrive_id, output=checkpoint_path, quiet=False)
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                self.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded weights from {checkpoint_path}")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _load_vocab_and_tokenizers(self):
        import subprocess, sys
        # install missing packages if autograder env doesn't have them
        try:
            import datasets
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "-q"], check=True)
        try:
            self.de_nlp = spacy.load("de_core_news_sm")
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"], check=True)
            self.de_nlp = spacy.load("de_core_news_sm")
        try:
            self.en_nlp = spacy.load("en_core_web_sm")
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.en_nlp = spacy.load("en_core_web_sm")
        from dataset import Multi30kDataset
        train_ds = Multi30kDataset(split='train')
        train_ds.build_vocab()
        self.src_vocab = train_ds.src_vocab
        self.tgt_vocab = train_ds.tgt_vocab
        self.tgt_itos  = train_ds.tgt_itos

    def encode(self, src, src_mask):
        return self.encoder(
            self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model)), src_mask
        )

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.output_proj(self.decoder(
            self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model)),
            memory, src_mask, tgt_mask
        ))

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        self.eval()
        device = next(self.parameters()).device
        tokens  = [tok.text.lower() for tok in self.de_nlp.tokenizer(src_sentence)]
        src_ids = [SOS_IDX] + [self.src_vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
        src     = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            memory = self.encode(src, make_src_mask(src).to(device))
            ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
            for _ in range(100):
                tgt_mask = make_tgt_mask(ys).to(device)
                out      = self.decode(memory, make_src_mask(src).to(device), ys, tgt_mask)
                next_tok = out[:, -1, :].argmax(dim=-1).item()
                ys = torch.cat([ys, torch.tensor([[next_tok]], device=device)], dim=1)
                if next_tok == EOS_IDX:
                    break

        special = {PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX}
        tokens = [self.tgt_itos[i] for i in ys[0].tolist() if i not in special]
        import re
        text = " ".join(tokens)
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        text = re.sub(r" n't", "n't", text)
        text = re.sub(r" 's", "'s", text)
        text = re.sub(r" 'm", "'m", text)
        text = re.sub(r" 're", "'re", text)
        text = re.sub(r" 've", "'ve", text)
        text = re.sub(r" 'll", "'ll", text)
        return text
