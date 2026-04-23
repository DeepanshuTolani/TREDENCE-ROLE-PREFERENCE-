

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────
# PART 1 — Custom PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A linear layer augmented with per-weight learnable gates.

    For every weight w_ij, a gate score g_ij is also learned.
    The effective weight used in the forward pass is:
        pruned_weight_ij = w_ij * sigmoid(g_ij)

    When sigmoid(g_ij) → 0 the weight is "pruned" (suppressed to near zero).
    When sigmoid(g_ij) → 1 the weight is fully active.

    Gradients flow through both `weight` and `gate_scores` automatically
    because all operations (sigmoid, element-wise multiply, linear) are
    differentiable PyTorch operations.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias — same init as nn.Linear
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        # Gate scores — one scalar per weight, initialised near 0
        # sigmoid(0) = 0.5 → gates start at ~50 % open, giving the
        # optimiser room to move them in either direction.
        self.gate_scores = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

        # Kaiming uniform init for weights (standard practice)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Squash gate_scores into (0, 1) using sigmoid
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Step 2: Mask weights — zero gates suppress their weights
        pruned_weights = self.weight * gates             # element-wise

        # Step 3: Standard affine transform  y = x W^T + b
        return nn.functional.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached, on CPU)."""
        return torch.sigmoid(self.gate_scores).detach().cpu()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gate values for this layer (always positive)."""
        return torch.sigmoid(self.gate_scores).sum()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ─────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32×32×3 → 10 classes).
    All linear projections use PrunableLinear so every weight has a gate.
    BatchNorm + ReLU are placed after each prunable layer for training stability.
    """

    def __init__(self):
        super().__init__()

        # Simple convolutional feature extractor (not pruned — focus is on FC)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                      # 16×16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                      # 8×8

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                              # 4×4
        )
        # 256 × 4 × 4 = 4096 features fed into prunable FC layers

        self.prunable_layers = nn.ModuleList([
            PrunableLinear(4096, 1024),
            PrunableLinear(1024, 512),
            PrunableLinear(512,  256),
            PrunableLinear(256,  10),
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(1024),
            nn.LayerNorm(512),
            nn.LayerNorm(256),
        ])

        self.dropout = nn.Dropout(0.3)
        self.relu    = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)          # flatten → (B, 4096)

        for i, layer in enumerate(self.prunable_layers[:-1]):
            x = layer(x)
            x = self.norms[i](x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.prunable_layers[-1](x)     # logits (no activation)
        return x

    def sparsity_loss(self) -> torch.Tensor:
        """Sum L1 gate norms across ALL prunable layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers)

    def all_gates(self) -> torch.Tensor:
        """Concatenate every gate value in the network into one flat tensor."""
        return torch.cat([
            layer.get_gates().flatten()
            for layer in self.prunable_layers
        ])

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate < threshold (= effectively pruned)."""
        gates = self.all_gates()
        return (gates < threshold).float().mean().item()


# ─────────────────────────────────────────────
# PART 3 — Data, Training, Evaluation
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    """Download CIFAR-10 and return train / test DataLoaders."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, lam):
    model.train()
    total_cls_loss = total_sparse_loss = correct = total = 0
    ce_criterion = nn.CrossEntropyLoss()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)

        cls_loss    = ce_criterion(logits, labels)
        sparse_loss = model.sparsity_loss()
        loss        = cls_loss + lam * sparse_loss      # Total Loss formula

        loss.backward()
        optimizer.step()

        total_cls_loss    += cls_loss.item()
        total_sparse_loss += sparse_loss.item()
        preds              = logits.argmax(dim=1)
        correct           += (preds == labels).sum().item()
        total             += labels.size(0)

    n = len(loader)
    return total_cls_loss / n, total_sparse_loss / n, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        preds = model(inputs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def run_experiment(lam: float, epochs: int = 25, device_str: str = "auto"):
    """Train and evaluate for a given λ value. Returns (test_acc, sparsity)."""
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  λ = {lam}  |  Device: {device}")
    print(f"{'='*60}")

    train_loader, test_loader = get_cifar10_loaders()
    model = SelfPruningNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        cls_l, sp_l, tr_acc = train_one_epoch(model, train_loader, optimizer, device, lam)
        scheduler.step()
        sparsity = model.sparsity_level()

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>3}/{epochs} | "
                f"CE: {cls_l:.3f} | Sparse: {sp_l:.1f} | "
                f"Train Acc: {tr_acc:.3f} | Gate Sparsity: {sparsity:.2%}"
            )

    test_acc = evaluate(model, test_loader, device)
    sparsity = model.sparsity_level()
    print(f"\n  ✓ Test Accuracy: {test_acc:.4f}  |  Sparsity: {sparsity:.2%}")
    return model, test_acc, sparsity


def plot_gate_distribution(model, lam, save_path="gate_distribution.png"):
    """Plot histogram of gate values for the best model."""
    gates = model.all_gates().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Gate Value Distribution  (λ = {lam})\nSelf-Pruning Neural Network — Tredence Case Study",
        fontsize=13, fontweight="bold"
    )

    # Full distribution
    axes[0].hist(gates, bins=100, color="#4C72B0", edgecolor="white", linewidth=0.3)
    axes[0].set_title("All Gate Values")
    axes[0].set_xlabel("Gate Value (sigmoid output)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(0.01, color="red", linestyle="--", label="Prune threshold (0.01)")
    axes[0].legend()

    # Zoomed view near 0 and 1
    axes[1].hist(gates[gates < 0.1],  bins=50, color="#DD8452", alpha=0.8, label="Near 0  (pruned)")
    axes[1].hist(gates[gates > 0.9],  bins=50, color="#55A868", alpha=0.8, label="Near 1  (active)")
    axes[1].set_title("Zoomed: Pruned vs Active Gates")
    axes[1].set_xlabel("Gate Value")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    LAMBDAS = [1e-5, 1e-4, 1e-3]   # low, medium, high sparsity pressure
    EPOCHS  = 25                    # increase to 40–50 for best results

    results = {}
    best_model = None
    best_lam   = LAMBDAS[1]        # medium lambda for the gate plot

    for lam in LAMBDAS:
        model, acc, sparsity = run_experiment(lam, epochs=EPOCHS)
        results[lam] = {"accuracy": acc, "sparsity": sparsity}
        if lam == best_lam:
            best_model = model

    # Summary Table
    print("\n" + "="*55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>14} {'Sparsity (%)':>14}")
    print("-"*55)
    for lam, r in results.items():
        print(f"  {lam:<12.0e} {r['accuracy']:>13.2%} {r['sparsity']:>13.2%}")
    print("="*55)

    # Gate distribution plot for the best model
    if best_model:
        plot_gate_distribution(best_model, lam=best_lam)

    print("\n✓ Case study complete.")
