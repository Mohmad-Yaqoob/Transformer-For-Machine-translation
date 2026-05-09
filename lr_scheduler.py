import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def _get_lr_scale(self):
        step = self.last_epoch + 1
        return (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)

    def get_lr(self):
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


def get_lr_history(d_model, warmup_steps, total_steps):
    dummy = torch.nn.Linear(1, 1)
    optimizer = optim.Adam(dummy.parameters(), lr=1.0)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return history


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL      = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS  = 20000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    print(f"Peak LR         : {max(lrs):.6f}  at step {lrs.index(max(lrs))}")
    print(f"LR at step 1    : {lrs[0]:.8f}")
    print(f"LR at step 4000 : {lrs[3999]:.6f}")
    print(f"LR at step 8000 : {lrs[7999]:.6f}")

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("noam_lr_plot.png")
    print("Plot saved to noam_lr_plot.png")