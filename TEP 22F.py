
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

class GaussianRBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(GaussianRBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        prob = torch.sigmoid(F.linear(v, self.W.t(), self.b))
        return prob, torch.bernoulli(prob)

    def forward(self, v):
        return self.sample_h(v)[0]


class BinaryRBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(BinaryRBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        prob = torch.sigmoid(F.linear(v, self.W.t(), self.b))
        return prob, torch.bernoulli(prob)

    def forward(self, v):
        return self.sample_h(v)[0]


class DBN_TE(nn.Module):
    def __init__(self):
        super(DBN_TE, self).__init__()
        self.rbm1 = GaussianRBM(n_visible=52, n_hidden=100)
        self.rbm2 = BinaryRBM(n_visible=100, n_hidden=50)
        self.classifier = nn.Linear(50, 22)

    def forward(self, x):
        x = self.rbm1(x)
        x = self.rbm2(x)
        return self.classifier(x)


def generate_dummy_tep_data(samples=1000, features=52, classes=22):
    X = np.random.randn(samples, features)
    y = np.random.randint(0, classes, samples)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train_model(model, dataloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def plot_fdr_heatmap(model, X_sample, title="FDR Heatmap"):
    model.eval()
    with torch.no_grad():
        h1 = model.rbm1(X_sample)
        h2 = model.rbm2(h1)
        h2_np = h2.numpy()

    plt.imshow(h2_np, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Hidden Neurons")
    plt.ylabel("Samples")
    plt.show()


if __name__ == "__main__":

    X, y = generate_dummy_tep_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = DBN_TE()
    train_model(model, dataloader)
es
    plot_fdr_heatmap(model, X[:100])

