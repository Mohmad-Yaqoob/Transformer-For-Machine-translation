import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import wandb
import sacrebleu

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from lr_scheduler import NoamScheduler


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        # logits: [N, vocab_size], target: [N]
        log_probs = F.log_softmax(logits, dim=-1)

        # smooth target distribution
        smooth_val = self.smoothing / (self.vocab_size - 2)  # -2 for true token and pad
        with torch.no_grad():
            dist = torch.full_like(log_probs, smooth_val)
            dist[:, self.pad_idx] = 0.0
            dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # mask pad positions — they don't contribute to loss
        pad_mask = (target == self.pad_idx)
        loss = -(dist * log_probs).sum(dim=-1)
        loss = loss.masked_fill(pad_mask, 0.0)
        return loss.sum() / (~pad_mask).sum().clamp(min=1)


def run_epoch(data_iter, model, loss_fn, optimizer, scheduler=None,
              epoch_num=0, is_train=True, device="cpu"):
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in  = tgt[:, :-1]  # decoder input  — drop last token
            tgt_out = tgt[:, 1:]   # target labels  — drop <sos>

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_in)

            logits = model(src, tgt_in, src_mask, tgt_mask)

            # flatten for loss
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(B * T))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            non_pad = (tgt_out != PAD_IDX).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device="cpu"):
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = out[:, -1, :].argmax(dim=-1).item()
            ys = torch.cat([ys, torch.tensor([[next_tok]], device=device)], dim=1)
            if next_tok == end_symbol:
                break
    return ys


def evaluate_bleu(model, test_dataloader, tgt_itos, device="cpu", max_len=100):
    model.eval()
    hypotheses = []
    references = []
    special    = {PAD_IDX, SOS_IDX, EOS_IDX}

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            src_mask = make_src_mask(src)

            pred_ids = greedy_decode(model, src, src_mask, max_len,
                                     SOS_IDX, EOS_IDX, device)
            pred_tokens = [tgt_itos[i] for i in pred_ids[0].tolist() if i not in special]
            ref_tokens  = [tgt_itos[i] for i in tgt[0].tolist()      if i not in special]

            hypotheses.append(" ".join(pred_tokens))
            references.append(" ".join(ref_tokens))

    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score


def save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt"):
    torch.save({
        "epoch": epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model":   model.d_model,
            "N":         len(model.encoder.layers),
            "num_heads": model.encoder.layers[0].self_attn.num_heads,
            "d_ff":      model.encoder.layers[0].ffn.linear1.out_features,
            "dropout":   0.1,
        }
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


def run_training_experiment(
    d_model      = 256,
    N            = 3,
    num_heads    = 8,
    d_ff         = 512,
    dropout      = 0.1,
    batch_size   = 64,
    num_epochs   = 15,
    warmup_steps = 4000,
    smoothing    = 0.1,
    device_str   = None,
):
    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(project="da6401-a3", config={
        "d_model": d_model, "N": N, "num_heads": num_heads,
        "d_ff": d_ff, "dropout": dropout, "batch_size": batch_size,
        "num_epochs": num_epochs, "warmup_steps": warmup_steps,
        "smoothing": smoothing,
    })

    train_loader, val_loader, test_loader, vocab_info = get_dataloaders(batch_size)
    src_vocab_size = len(vocab_info["src_vocab"])
    tgt_vocab_size = len(vocab_info["tgt_vocab"])
    tgt_itos       = vocab_info["tgt_itos"]

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model, N=N, num_heads=num_heads,
        d_ff=d_ff, dropout=dropout,
        checkpoint_path=None,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)
    loss_fn   = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                               epoch_num=epoch, is_train=True, device=device)
        val_loss   = run_epoch(val_loader, model, loss_fn, None, None,
                               epoch_num=epoch, is_train=False, device=device)

        val_bleu = evaluate_bleu(model, val_loader, tgt_itos, device)

        print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_bleu={val_bleu:.2f}")

        wandb.log({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_bleu":   val_bleu,
            "lr":         optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, "best_model.pt")
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})")

    # final test BLEU on best checkpoint
    load_checkpoint("best_model.pt", model)
    test_bleu = evaluate_bleu(model, test_loader, tgt_itos, device)
    print(f"\nTest BLEU: {test_bleu:.2f}")
    wandb.log({"test_bleu": test_bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()