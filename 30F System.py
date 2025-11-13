import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import os

class GaussianRBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        prob = torch.sigmoid(F.linear(v, self.W.t(), self.b))
        return prob.bernoulli(), prob

    def sample_v(self, h):
        mu = F.linear(h, self.W, self.a)
        return mu + torch.randn_like(mu), mu

class BinaryRBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        prob = torch.sigmoid(F.linear(v, self.W.t(), self.b))
        return prob.bernoulli(), prob

    def sample_v(self, h):
        prob = torch.sigmoid(F.linear(h, self.W, self.a))
        return prob.bernoulli(), prob

class DBN(nn.Module):
    def __init__(self, n_visible, n_hidden1, n_hidden2):
        super().__init__()
        self.rbm1 = GaussianRBM(n_visible, n_hidden1)
        self.rbm2 = BinaryRBM(n_hidden1, n_hidden2)

    def forward(self, x):
        _, h1 = self.rbm1.sample_h(x)
        _, h2 = self.rbm2.sample_h(h1)
        return h2

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        return self.out(x)

def train_rbm_cd(rbm, loader, lr=1e-2, epochs=30):
    opt = torch.optim.Adam(rbm.parameters(), lr=lr)
    for _ in range(epochs):
        for v0, _ in loader:
            _, h0 = rbm.sample_h(v0)
            v1, _ = rbm.sample_v(h0)
            _, h1 = rbm.sample_h(v1)
            pos_grad = v0.t() @ h0 / v0.size(0)
            neg_grad = v1.t() @ h1 / v0.size(0)
            rbm.W.grad = -(pos_grad - neg_grad)
            rbm.a.grad = -torch.mean(v0 - v1, dim=0)
            rbm.b.grad = -torch.mean(h0 - h1, dim=0)
            opt.step()
            opt.zero_grad()

def visualize_reconstructions(rbm, data, num_samples=10, title_prefix="RBM Reconstruction"):
    rbm.eval()
    with torch.no_grad():
        samples = data[:num_samples]
        _, h = rbm.sample_h(samples)
        v_recon, _ = rbm.sample_v(h)

    fig, axs = plt.subplots(num_samples, 2, figsize=(8, 2 * num_samples))
    for i in range(num_samples):
        axs[i, 0].plot(samples[i].numpy(), label="Original")
        axs[i, 1].plot(v_recon[i].numpy(), label="Reconstructed", color="orange")
        axs[i, 0].set_title(f"Sample {i+1} - Original")
        axs[i, 1].set_title(f"Sample {i+1} - Reconstructed")
    plt.tight_layout()
    plt.suptitle(title_prefix, fontsize=16, y=1.02)
    plt.show()

def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = len(train_losses)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(range(epochs), train_losses, label="Train Loss")
    axs[0].plot(range(epochs), val_losses, label="Val Loss")
    axs[0].set_title("Loss Curve")
    axs[0].legend()

    axs[1].plot(range(epochs), train_accuracies, label="Train Acc")
    axs[1].plot(range(epochs), val_accuracies, label="Val Acc")
    axs[1].set_title("Accuracy Curve")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def evaluate_classifier(clf, X, y, class_names):
    clf.eval()
    with torch.no_grad():
        preds = clf(X).argmax(1)
        acc = accuracy_score(y, preds) * 100
        print(f"Accuracy: {acc:.2f}%")

        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        print("\nClassification Report:")
        print(classification_report(y, preds, target_names=class_names))

def plot_fdr_heatmaps(dbn, X, y, n_classes=30):
    with torch.no_grad():
        _, h1 = dbn.rbm1.sample_h(X)
        _, h2 = dbn.rbm2.sample_h(h1)

    h1_units = dbn.rbm1.W.shape[1]
    h2_units = dbn.rbm2.W.shape[1]
    fdr_tensor = torch.zeros((n_classes, h2_units, h1_units))

    for fault in range(n_classes):
        idx = (y == fault)
        if idx.sum() == 0:
            continue
        x_fault = X[idx]
        _, h1_f = dbn.rbm1.sample_h(x_fault)
        _, h2_f = dbn.rbm2.sample_h(h1_f)
        mean_act = torch.einsum('bi,bj->ij', h2_f, h1_f) / h1_f.size(0)
        fdr_tensor[fault] = mean_act.detach()

    all_logs = np.log10(fdr_tensor.cpu().numpy() + 1e-6)
    vmin = all_logs.min()
    vmax = all_logs.max()
    os.makedirs("fdr_heatmaps", exist_ok=True)
    for fault in range(n_classes):
        log_hm = all_logs[fault]
        plt.figure(figsize=(6, 4))
        plt.imshow(log_hm, aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)
        plt.title(f"FDR Activation Map - Fault {fault}")
        plt.xlabel("RBM1 Hidden Units")
        plt.ylabel("RBM2 Hidden Units")
        plt.colorbar(label="log10(FDR + 1e-6)")
        plt.tight_layout()
        plt.savefig(f"fdr_heatmaps/fault_{fault:02}.png", dpi=300)
        plt.close()

def generate_fault_data(n, fault_id):
    x = torch.randn(n, 50) * 0.05
    if fault_id == 0:
        return x
    start = (fault_id * 2) % 40
    mask = torch.zeros_like(x)
    mask[:, start:start+10] = 1
    x += mask * (1.0 + 0.2 * torch.randn_like(mask)) * (1 + 0.05 * fault_id)
    return x

if __name__ == "__main__":
    torch.manual_seed(42)
    n_classes = 30
    samples_per_class = 150

    X = torch.cat([generate_fault_data(samples_per_class, i) for i in range(n_classes)])
    y = torch.cat([torch.full((samples_per_class,), i) for i in range(n_classes)])

    perm = torch.randperm(X.size(0))
    X, y = X[perm], y[perm]
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    dbn = DBN(50, 128, 64)
    loader1 = DataLoader(TensorDataset(X_train, X_train), batch_size=64, shuffle=True)
    train_rbm_cd(dbn.rbm1, loader1, epochs=15)
    with torch.no_grad():
        _, h1_train = dbn.rbm1.sample_h(X_train)
    loader2 = DataLoader(TensorDataset(h1_train, h1_train), batch_size=64, shuffle=True)
    train_rbm_cd(dbn.rbm2, loader2, epochs=15)

    visualize_reconstructions(dbn.rbm1, X_val)

    with torch.no_grad():
        feat_train = dbn(X_train)
        feat_val = dbn(X_val)

    clf = MLPClassifier(input_dim=64, hidden_dim=256, n_classes=n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(60):
        clf.train()
        out = clf(feat_train)
        loss = crit(out, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            train_preds = clf(feat_train).argmax(1)
            val_preds = clf(feat_val).argmax(1)
            train_acc = (train_preds == y_train).float().mean().item() * 100
            val_acc = (val_preds == y_val).float().mean().item() * 100
            val_loss = crit(clf(feat_val), y_val).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

    plot_learning_curve(train_losses, val_losses, train_accs, val_accs)

    class_names = [f"F{i}" for i in range(n_classes)]
    evaluate_classifier(clf, feat_val, y_val, class_names)

    print("\nGenerating FDR Heatmaps...")
    plot_fdr_heatmaps(dbn, X_val, y_val, n_classes=n_classes)
