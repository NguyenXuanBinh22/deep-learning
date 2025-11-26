import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CVAEDataset(Dataset):
    def __init__(self, x, y_onehot):
        assert x.shape[0] == y_onehot.shape[0]
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y_onehot, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ConditionalVAE(nn.Module):
    """
    Chuyển đổi từ cvae_generator TF:
    - Encoder: concat(X, Y) -> 2 hidden -> (mu, logvar)
    - Decoder: concat(z, Y) -> 2 hidden -> logits(X_hat)
    """
    def __init__(self,
                 n_features,
                 n_classes,
                 n_en_h1=500,
                 n_en_h2=250,
                 n_code=125,
                 n_de_h1=250,
                 n_de_h2=500):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_code = n_code

        # encoder
        self.fc_en1 = nn.Linear(n_features + n_classes, n_en_h1 + n_classes)
        self.bn_en1 = nn.BatchNorm1d(n_en_h1 + n_classes)
        self.fc_en2 = nn.Linear(n_en_h1 + n_classes, n_en_h2)
        self.bn_en2 = nn.BatchNorm1d(n_en_h2)
        self.fc_mu = nn.Linear(n_en_h2, n_code)
        self.fc_logvar = nn.Linear(n_en_h2, n_code)

        # decoder
        self.fc_de1 = nn.Linear(n_code + n_classes, n_de_h1)
        self.bn_de1 = nn.BatchNorm1d(n_de_h1)
        self.fc_de2 = nn.Linear(n_de_h1, n_de_h2)
        self.bn_de2 = nn.BatchNorm1d(n_de_h2)
        self.fc_out = nn.Linear(n_de_h2, n_features)

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def encode(self, x, y):
        h = torch.cat([x, y], dim=1)
        h = self.fc_en1(h)
        h = self.bn_en1(h)
        h = self.elu(h)
        h = self.fc_en2(h)
        h = self.bn_en2(h)
        h = self.tanh(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = self.fc_de1(h)
        h = self.bn_de1(h)
        h = self.tanh(h)
        h = self.fc_de2(h)
        h = self.bn_de2(h)
        h = self.elu(h)
        logits = self.fc_out(h)
        return logits

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, y)
        return logits, mu, logvar


def vae_loss(recon_logits, x, mu, logvar):
    # BCE with logits
    bce = nn.functional.binary_cross_entropy_with_logits(
        recon_logits, x, reduction='sum'
    )
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)


def train_cvae(model, train_loader, device,
               lr=1e-3,
               epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, mu, logvar = model(x, y)
            loss = vae_loss(logits, x, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        avg_loss = running_loss / n_samples
        print(f"Epoch {epoch:04d} | VAE loss: {avg_loss:.6f}")

    return model


def generate_samples(model,
                     y_sim,
                     cpg_list,
                     group_num,
                     out_dir="."):
    """
    y_sim: numpy array (N, n_classes)
    """
    device = next(model.parameters()).device
    model.eval()
    y = torch.tensor(y_sim, dtype=torch.float32, device=device)
    N = y.shape[0]
    z = torch.randn(N, model.n_code, device=device)
    with torch.no_grad():
        logits = model.decode(z, y)
        x_gen = torch.sigmoid(logits).cpu().numpy()

    df_gen = pd.DataFrame(x_gen, columns=cpg_list)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"generation_split_num_{group_num}.csv")
    df_gen.to_csv(out_path, index=False)
    print(f"Saved generated data to {out_path}")


def main():
    if len(sys.argv) != 5:
        print("Usage: python cvae_generator_torch.py "
              "train_X.csv train_Y.csv simulation_label.csv group_num")
        sys.exit(1)

    filename_x = sys.argv[1]
    filename_y = sys.argv[2]
    filename_simulation_label = sys.argv[3]
    group_num = sys.argv[4]

    # read data
    x_data_df = pd.read_csv(filename_x, dtype=np.float32)
    x_data = x_data_df.values
    cpg_list = x_data_df.columns.tolist()

    y_data = pd.read_csv(filename_y, dtype=np.float32).values
    y_sim = pd.read_csv(filename_simulation_label, dtype=np.float32).values

    n_features = x_data.shape[1]
    n_classes = y_data.shape[1]

    print("# feature:", n_features, "# train sample:", x_data.shape[0])

    train_ds = CVAEDataset(x_data, y_data)
    train_loader = DataLoader(train_ds,
                              batch_size=500,
                              shuffle=True,
                              drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE(
        n_features=n_features,
        n_classes=n_classes,
        n_en_h1=500,
        n_en_h2=250,
        n_code=125,
        n_de_h1=250,
        n_de_h2=500,
    )
    model = train_cvae(model, train_loader, device,
                       lr=1e-3,
                       epochs=100)

    generate_samples(model,
                     y_sim=y_sim,
                     cpg_list=cpg_list,
                     group_num=group_num,
                     out_dir=".")


if __name__ == "__main__":
    main()
