import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from lr_scheduler import NoamScheduler
from train import LabelSmoothingLoss, run_epoch, evaluate_bleu, save_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(src_vocab_size, tgt_vocab_size):
    return Transformer(
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        d_model=256, N=3, num_heads=8, d_ff=512, dropout=0.1,
        checkpoint_path=None,
    ).to(DEVICE)


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys).to(device)
            out      = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = out[:, -1, :].argmax(dim=-1).item()
            ys = torch.cat([ys, torch.tensor([[next_tok]], dtype=torch.long, device=device)], dim=1)
            if next_tok == end_symbol:
                break
    return ys


def evaluate_bleu_local(model, dataloader, tgt_itos, device, max_len=100):
    import sacrebleu
    model.eval()
    hyps, refs = [], []
    special = {PAD_IDX, SOS_IDX, EOS_IDX}
    with torch.no_grad():
        for src, tgt in dataloader:
            for i in range(src.size(0)):
                s  = src[i].unsqueeze(0).to(device)
                sm = make_src_mask(s).to(device)
                pred = greedy_decode(model, s, sm, max_len, SOS_IDX, EOS_IDX, device)
                hyps.append(" ".join(tgt_itos[j] for j in pred[0].tolist() if j not in special))
                refs.append(" ".join(tgt_itos[j] for j in tgt[i].tolist()  if j not in special))
    return sacrebleu.corpus_bleu(hyps, [refs], force=True).score


def run_epoch_local(loader, model, loss_fn, optimizer, scheduler, is_train, device):
    model.train() if is_train else model.eval()
    total_loss, total_tok = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for src, tgt in loader:
            src = src.to(device); tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]; tgt_out = tgt[:, 1:]
            sm = make_src_mask(src).to(device)
            tm = make_tgt_mask(tgt_in).to(device)
            logits  = model(src, tgt_in, sm, tm)
            B, T, V = logits.shape
            loss    = loss_fn(logits.reshape(B*T, V), tgt_out.reshape(B*T))
            if is_train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler: scheduler.step()
            non_pad      = (tgt_out != PAD_IDX).sum().item()
            total_loss  += loss.item() * non_pad
            total_tok   += non_pad
    return total_loss / max(total_tok, 1)


def train_loop(model, train_loader, val_loader, tgt_itos, tgt_vocab_size,
               num_epochs=15, fixed_lr=None, run_name="run", log_grad_norms=False):
    if fixed_lr is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=fixed_lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
    loss_fn = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, 0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_tok = 0.0, 0
        for src, tgt in train_loader:
            src = src.to(DEVICE); tgt = tgt.to(DEVICE)
            tgt_in = tgt[:, :-1]; tgt_out = tgt[:, 1:]
            sm = make_src_mask(src).to(DEVICE)
            tm = make_tgt_mask(tgt_in).to(DEVICE)
            logits  = model(src, tgt_in, sm, tm)
            B, T, V = logits.shape
            loss    = loss_fn(logits.reshape(B*T, V), tgt_out.reshape(B*T))
            if log_grad_norms:
                optimizer.zero_grad(); loss.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None and ('W_q' in n or 'W_k' in n):
                        wandb.log({f"grad_norm/{n}": p.grad.norm().item()})
            else:
                optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            non_pad      = (tgt_out != PAD_IDX).sum().item()
            total_loss  += loss.item() * non_pad
            total_tok   += non_pad

        train_loss = total_loss / max(total_tok, 1)
        val_bleu   = evaluate_bleu_local(model, val_loader, tgt_itos, DEVICE)
        lr_now     = optimizer.param_groups[0]["lr"]
        print(f"  [{run_name}] epoch {epoch+1:02d} | loss={train_loss:.4f} | bleu={val_bleu:.2f}")
        wandb.log({"epoch": epoch+1, "train_loss": train_loss,
                   "val_bleu": val_bleu, "lr": lr_now})


# ── Experiment 1: Noam vs Fixed LR ──────────────────────────────────

def experiment1():
    print("\n=== Exp 1: Noam vs Fixed LR ===")
    train_loader, val_loader, _, vocab_info = get_dataloaders(batch_size=128)
    src_vocab_size = len(vocab_info["src_vocab"])
    tgt_vocab_size = len(vocab_info["tgt_vocab"])
    tgt_itos       = vocab_info["tgt_itos"]

    for mode in ["noam", "fixed"]:
        wandb.init(project="da6401-a3", name=f"exp1_{mode}_lr",
                   group="exp1_scheduler", reinit=True)
        model = build_model(src_vocab_size, tgt_vocab_size)
        train_loop(model, train_loader, val_loader, tgt_itos, tgt_vocab_size,
                   num_epochs=15, fixed_lr=(1e-4 if mode == "fixed" else None),
                   run_name=f"exp1_{mode}")
        wandb.finish()


# ── Experiment 2: Scaling factor ablation ───────────────────────────

class UnscaledMHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.num_heads = num_heads; self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_w = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        out    = torch.matmul(attn_w, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


def experiment2():
    print("\n=== Exp 2: Scaling Factor ===")
    train_loader, val_loader, _, vocab_info = get_dataloaders(batch_size=128)
    src_vocab_size = len(vocab_info["src_vocab"])
    tgt_vocab_size = len(vocab_info["tgt_vocab"])
    tgt_itos       = vocab_info["tgt_itos"]

    for use_scale in [True, False]:
        name = "with_scale" if use_scale else "no_scale"
        wandb.init(project="da6401-a3", name=f"exp2_{name}",
                   group="exp2_scaling", reinit=True)
        model = build_model(src_vocab_size, tgt_vocab_size)
        if not use_scale:
            for enc_layer in model.encoder.layers:
                enc_layer.self_attn = UnscaledMHA(256, 8).to(DEVICE)
            for dec_layer in model.decoder.layers:
                dec_layer.self_attn  = UnscaledMHA(256, 8).to(DEVICE)
                dec_layer.cross_attn = UnscaledMHA(256, 8).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
        loss_fn   = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, 0.1)

        for epoch in range(10):
            model.train()
            total_loss, total_tok = 0.0, 0
            grad_norms_q, grad_norms_k = [], []
            for src, tgt in train_loader:
                src = src.to(DEVICE); tgt = tgt.to(DEVICE)
                tgt_in = tgt[:, :-1]; tgt_out = tgt[:, 1:]
                sm = make_src_mask(src).to(DEVICE)
                tm = make_tgt_mask(tgt_in).to(DEVICE)
                logits  = model(src, tgt_in, sm, tm)
                B, T, V = logits.shape
                loss    = loss_fn(logits.reshape(B*T, V), tgt_out.reshape(B*T))
                optimizer.zero_grad(); loss.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        if 'W_q' in n: grad_norms_q.append(p.grad.norm().item())
                        if 'W_k' in n: grad_norms_k.append(p.grad.norm().item())
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                non_pad      = (tgt_out != PAD_IDX).sum().item()
                total_loss  += loss.item() * non_pad
                total_tok   += non_pad

            train_loss = total_loss / max(total_tok, 1)
            val_bleu   = evaluate_bleu_local(model, val_loader, tgt_itos, DEVICE)
            avg_q = sum(grad_norms_q) / max(len(grad_norms_q), 1)
            avg_k = sum(grad_norms_k) / max(len(grad_norms_k), 1)
            print(f"  [{name}] epoch {epoch+1:02d} | loss={train_loss:.4f} | bleu={val_bleu:.2f} | grad_Q={avg_q:.4f} | grad_K={avg_k:.4f}")
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_bleu": val_bleu,
                       "avg_grad_norm_Q": avg_q, "avg_grad_norm_K": avg_k})
        wandb.finish()


# ── Experiment 3: Attention heatmaps ────────────────────────────────

def experiment3():
    print("\n=== Exp 3: Attention Heatmaps ===")
    train_loader, val_loader, _, vocab_info = get_dataloaders(batch_size=128)
    src_vocab_size = len(vocab_info["src_vocab"])
    tgt_vocab_size = len(vocab_info["tgt_vocab"])
    tgt_itos       = vocab_info["tgt_itos"]
    src_itos       = vocab_info["src_itos"]

    wandb.init(project="da6401-a3", name="exp3_attn_heads",
               group="exp3_heads", reinit=True)
    model     = build_model(src_vocab_size, tgt_vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, 256, 4000)
    loss_fn   = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX)

    for epoch in range(15):
        run_epoch_local(train_loader, model, loss_fn, optimizer, scheduler, True, DEVICE)
        val_bleu = evaluate_bleu_local(model, val_loader, tgt_itos, DEVICE)
        wandb.log({"epoch": epoch+1, "val_bleu": val_bleu})
        print(f"  epoch {epoch+1} val_bleu={val_bleu:.2f}")

    model.eval()
    src_batch, _ = next(iter(val_loader))
    src_single   = src_batch[:1].to(DEVICE)
    src_tokens   = [src_itos.get(i.item(), "<unk>") for i in src_single[0]
                    if i.item() not in {PAD_IDX, SOS_IDX, EOS_IDX}]

    attn_store = {}
    def hook_fn(module, inp, out):
        q, k = inp[0], inp[1]
        Q = module.split_heads(module.W_q(q))
        K = module.split_heads(module.W_k(k))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(module.d_k)
        attn_store['w'] = F.softmax(scores, dim=-1).detach().cpu()

    handle = model.encoder.layers[-1].self_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        model.encode(src_single, make_src_mask(src_single).to(DEVICE))
    handle.remove()

    attn_w    = attn_store['w'][0]
    num_heads = attn_w.shape[0]
    seq_len   = len(src_tokens)

    fig, axes = plt.subplots(2, num_heads // 2, figsize=(20, 8))
    for h, ax in enumerate(axes.flatten()):
        data = attn_w[h, :seq_len, :seq_len].numpy()
        ax.imshow(data, cmap='Blues')
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(src_tokens, fontsize=7)
        ax.set_title(f"Head {h+1}", fontsize=9)

    plt.suptitle("Last Encoder Layer — All Attention Heads")
    plt.tight_layout()
    plt.savefig("attention_heads.png", dpi=100)
    wandb.log({"attention_heatmaps": wandb.Image("attention_heads.png")})
    wandb.finish()


# ── Experiment 4: Sinusoidal vs Learned PE ──────────────────────────

class LearnedPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(positions))


def experiment4():
    print("\n=== Exp 4: Sinusoidal vs Learned PE ===")
    train_loader, val_loader, _, vocab_info = get_dataloaders(batch_size=128)
    src_vocab_size = len(vocab_info["src_vocab"])
    tgt_vocab_size = len(vocab_info["tgt_vocab"])
    tgt_itos       = vocab_info["tgt_itos"]

    for pe_type in ["sinusoidal", "learned"]:
        wandb.init(project="da6401-a3", name=f"exp4_{pe_type}",
                   group="exp4_pos_enc", reinit=True)
        model = build_model(src_vocab_size, tgt_vocab_size)
        if pe_type == "learned":
            model.pos_enc = LearnedPE(256, max_len=256).to(DEVICE)
        train_loop(model, train_loader, val_loader, tgt_itos, tgt_vocab_size,
                   num_epochs=15, run_name=f"exp4_{pe_type}")
        wandb.finish()


# ── Experiment 5: Label Smoothing ────────────────────────────────────

def experiment5():
    print("\n=== Exp 5: Label Smoothing ===")
    train_loader, val_loader, _, vocab_info = get_dataloaders(batch_size=128)
    src_vocab_size = len(vocab_info["src_vocab"])
    tgt_vocab_size = len(vocab_info["tgt_vocab"])
    tgt_itos       = vocab_info["tgt_itos"]

    for eps in [0.1, 0.0]:
        name = f"eps_{str(eps).replace('.','')}"
        wandb.init(project="da6401-a3", name=f"exp5_{name}",
                   group="exp5_smoothing", reinit=True)
        model     = build_model(src_vocab_size, tgt_vocab_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, 256, 4000)
        loss_fn   = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=eps)

        for epoch in range(15):
            model.train()
            total_loss, total_tok = 0.0, 0
            conf_sum, conf_count  = 0.0, 0

            for src, tgt in train_loader:
                src = src.to(DEVICE); tgt = tgt.to(DEVICE)
                tgt_in = tgt[:, :-1]; tgt_out = tgt[:, 1:]
                sm = make_src_mask(src).to(DEVICE)
                tm = make_tgt_mask(tgt_in).to(DEVICE)
                logits  = model(src, tgt_in, sm, tm)
                B, T, V = logits.shape
                flat_l  = logits.reshape(B*T, V)
                flat_t  = tgt_out.reshape(B*T)
                loss    = loss_fn(flat_l, flat_t)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()

                with torch.no_grad():
                    probs   = F.softmax(flat_l, dim=-1)
                    non_pad = flat_t != PAD_IDX
                    conf    = probs[non_pad].gather(1, flat_t[non_pad].unsqueeze(1)).mean()
                    conf_sum += conf.item(); conf_count += 1

                non_pad_n   = (tgt_out != PAD_IDX).sum().item()
                total_loss += loss.item() * non_pad_n
                total_tok  += non_pad_n

            train_loss = total_loss / max(total_tok, 1)
            val_bleu   = evaluate_bleu_local(model, val_loader, tgt_itos, DEVICE)
            avg_conf   = conf_sum / max(conf_count, 1)
            print(f"  [eps={eps}] epoch {epoch+1:02d} | loss={train_loss:.4f} | bleu={val_bleu:.2f} | conf={avg_conf:.4f}")
            wandb.log({"epoch": epoch+1, "train_loss": train_loss,
                       "val_bleu": val_bleu, "prediction_confidence": avg_conf})
        wandb.finish()


if __name__ == "__main__":
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else "all"
    if exp in ("1", "all"): experiment1()
    if exp in ("2", "all"): experiment2()
    if exp in ("3", "all"): experiment3()
    if exp in ("4", "all"): experiment4()
    if exp in ("5", "all"): experiment5()