import pytorch_lightning as pl
import torch
import torch.nn as nn
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import os
import numpy as np
import pickle

from .data import DatasetWithConfounder, SimpleDataset
from .utils import one_hot, gram_matrix


class HD_AE_model(pl.LightningModule):
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
        loss = reconstruction_loss + self.hsic_penalty * hsic_loss

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


class HD_AE:
    def __init__(self,
                 adata,
                 batch_key,
                 hidden_layer_sizes,
                 embedding_dimension,
                 learning_rate=1e-3,
                 hsic_penalty=1,
                 num_batches=None):
        self.adata = adata
        self.var_names = self.adata.var_names.astype(str)

        self.batch_key = batch_key
        self.hidden_layer_sizes = hidden_layer_sizes
        self.embedding_dimension = embedding_dimension
        self.learning_rate = learning_rate
        self.hsic_penalty = hsic_penalty

        input_size = adata.shape[1]
        layer_sizes = [input_size] + hidden_layer_sizes + [embedding_dimension]

        self.model = HD_AE_model(
            layer_sizes=layer_sizes,
            lr=learning_rate,
            num_batches=len(adata.obs[batch_key].unique()) if not num_batches else num_batches,
            hsic_penalty=hsic_penalty
        )

    def train(self, num_epochs):
        train_set = DatasetWithConfounder(
            X=self.adata.X.copy(),
            Z=LabelEncoder().fit_transform(self.adata.obs[self.batch_key])
        )

        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, max_epochs=num_epochs)
        trainer.fit(self.model, train_loader)
        self.model.eval()

    def embed_data(self, eval_adata):
        eval_set = SimpleDataset(X=eval_adata.X.copy())
        eval_loader = DataLoader(eval_set, batch_size=128)

        output_list = []
        with torch.no_grad():
            for X_batch in eval_loader:
                embedding = self.model(X_batch)
                output_list.append(embedding)
            output_list = torch.cat(output_list, dim=0).detach().cpu().numpy()

        embedding_adata = AnnData(X=output_list, obs=eval_adata.obs)
        return embedding_adata

    def save(self, dir_path, overwrite=False, save_anndata=False):
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )

        model_save_path = os.path.join(dir_path, "model_params.pt")
        varnames_save_path = os.path.join(dir_path, "var_names.csv")
        params_save_path = os.path.join(dir_path, "params.pkl")

        params = {
            "batch_key": self.batch_key,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "embedding_dimension": self.embedding_dimension,
            "learning_rate": self.learning_rate,
            "hsic_penalty": self.hsic_penalty,
            "num_batches": self.model.num_batches
        }


        np.savetxt(varnames_save_path, self.var_names.to_numpy(), fmt="%s")
        torch.save(self.model.state_dict(), model_save_path)
        pickle.dump(params, open(params_save_path, "wb"))

        if save_anndata:
            self.adata.write(os.path.join(dir_path, "adata.h5ad"))

    @classmethod
    def load(cls, dir_path):
        model_save_path = os.path.join(dir_path, "model_params.pt")
        varnames_save_path = os.path.join(dir_path, "var_names.csv")
        params_save_path = os.path.join(dir_path, "params.pkl")

        params = pickle.load(open(params_save_path, "rb"))
        var_names = np.genfromtxt(varnames_save_path, delimiter=",", dtype=str)

        adata = AnnData(X=np.zeros((1, len(var_names))))
        adata.var_names = var_names

        obj = HD_AE(
            adata=adata,
            **params
        )

        obj.model.load_state_dict(torch.load(model_save_path))

        return obj
