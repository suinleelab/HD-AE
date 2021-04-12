import pytorch_lightning as pl
import torch
import torch.nn as nn

from utils import one_hot, gram_matrix


class HD_AE(pl.LightningModule):
    def __init__(self, num_batches, hsic_penalty, lr, layer_sizes):
        super().__init__()

        encoder_layers = []
        decoder_layers = []
        self.num_batches = num_batches
        self.hsic_penalty = hsic_penalty
        self.lr = lr

        for i in range(len(layer_sizes)-1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]

            encoder_layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))

            if i+1 < len(layer_sizes)-1:
                encoder_layers.append(nn.ReLU())

            if i+1 < len(layer_sizes)-1:
                decoder_layers.append(nn.Linear(in_features=out_dim, out_features=in_dim))
                decoder_layers.append(nn.ReLU())
            else:
                decoder_layers.append(nn.Linear(in_features=out_dim+self.num_batches, out_features=in_dim))

        decoder_layers.reverse()

        self.encoder = nn.Sequential(*encoder_layers).to(self.device)
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)

        self.criterion = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, z = batch

        embedding = self.encoder(x)
        z = z.reshape(len(z), 1)

        z_one_hot = one_hot(z, self.num_batches).to(self.device)
        embedding_and_batch = torch.cat([embedding, z_one_hot], dim=1)
        reconstruction = self.decoder(embedding_and_batch)

        reconstruction_loss = self.criterion(x, reconstruction)

        hsic_loss = self.compute_hsic(embedding, z) if len(torch.unique(z)) > 1 else 0
        loss = reconstruction_loss + self.lambda_param * hsic_loss

        self.log("reconstruction_loss", value=reconstruction_loss, prog_bar=True)
        self.log("HSIC_loss", value=hsic_loss, prog_bar=True)

        return loss

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.encoder(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr
        )
        return opt

    def compute_hsic(self, x, y, sigma=1):
        m = x.shape[0]
        K = gram_matrix(x, sigma=sigma)
        y = y.float().reshape(-1, 1)
        L = gram_matrix(y, sigma=sigma)
        H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
        H = H.float().to(self.device)
        HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
        return HSIC
