import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 0

# ---- Steps 1-2: load, scale to 0-1, split 200 / 798 / 799 ----
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype("float32")          # pixels are 0..16
X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=200, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED)
print("split sizes:", len(X_tr), len(X_va), len(X_te))

T = torch.tensor
X_tr, X_va, X_te = T(X_tr), T(X_va), T(X_te)
y_tr, y_va, y_te = T(y_tr).long(), T(y_va).long(), T(y_te).long()
loss_fn = nn.CrossEntropyLoss()


# ---- Step 4: the three models ----
def small():
    return nn.Linear(64, 10)


def big():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 10))


def regularised():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(256, 10))


# ---- Step 3 and 5: one loop that records both losses; checkpoint + early stopping ----
def train(name, model, opt, epochs=300, patience=None):
    hist, best, wait = {"train": [], "val": []}, float("inf"), 0
    for epoch in range(epochs):
        model.train()
        loss = loss_fn(model(X_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()                                   # switches dropout off
        with torch.no_grad():
            hist["train"].append(loss_fn(model(X_tr), y_tr).item())
            hist["val"].append(loss_fn(model(X_va), y_va).item())
        if hist["val"][-1] < best:                     # keep the best checkpoint
            best, wait = hist["val"][-1], 0
            torch.save(model.state_dict(), f"best_{name}.pt")
        else:
            wait += 1
        if patience and wait >= patience:              # early stopping
            break
    return hist


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


# ---- Train the three runs ----
torch.manual_seed(SEED)
m_small = small()
h_small = train("small", m_small, torch.optim.SGD(m_small.parameters(), lr=0.01))

torch.manual_seed(SEED)
m_big = big()
h_big = train("big", m_big, torch.optim.Adam(m_big.parameters(), lr=0.001))

torch.manual_seed(SEED)
m_reg = regularised()
h_reg = train("reg", m_reg,
              torch.optim.AdamW(m_reg.parameters(), lr=0.001, weight_decay=0.05),
              patience=30)

# Test accuracy: small and big after all their epochs; regularised after reloading its best checkpoint
acc_small = accuracy(m_small, X_te, y_te)
acc_big_last = accuracy(m_big, X_te, y_te)
m_big_best = big()
m_big_best.load_state_dict(torch.load("best_big.pt"))
acc_big_best = accuracy(m_big_best, X_te, y_te)      # extra row for a like-for-like comparison
m_reg.load_state_dict(torch.load("best_reg.pt"))
acc_reg = accuracy(m_reg, X_te, y_te)

# ---- Step 6: three curve plots in one image ----
runs = [("Small model", h_small, (0, 2.5), "Underfitting / under-trained"),
        ("Big model", h_big, (0, 1.0), "Overfitting"),
        ("Big + dropout + decay", h_reg, (0, 1.0), "Early stopped")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, h, ylim, note) in zip(axes, runs):
    best_ep = int(np.argmin(h["val"]))
    ax.plot(h["train"], color="tab:orange", label="training loss")
    ax.plot(h["val"], color="tab:blue", linestyle="--", label="validation loss")
    ax.axvline(best_ep, color="gray", linestyle=":", label=f"best val: epoch {best_ep + 1}")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 300)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{title}\n{note} (ran {len(h['train'])} epochs)")
    ax.legend()
plt.tight_layout()
plt.savefig("curves.png", dpi=150)

# ---- Step 7: results table ----
def row(name, h, acc):
    b = int(np.argmin(h["val"]))
    return {"Model": name, "Final training loss": round(h["train"][-1], 3),
            "Best validation loss": f"{min(h['val']):.3f} at epoch {b + 1}",
            "Test accuracy": round(acc, 3)}

table = pd.DataFrame([
    row("Small", h_small, acc_small),
    row("Big (after 300 epochs)", h_big, acc_big_last),
    row("Big (best checkpoint, for comparison)", h_big, acc_big_best),
    row("Big + dropout + decay (best checkpoint)", h_reg, acc_reg),
])
print(table.to_string(index=False))
table.to_csv("results.csv", index=False)