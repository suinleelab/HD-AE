import torch.nn as nn
import logging
import torch
from typing import Tuple
from utils import hilbert_schmidt, one_hot
import torch.nn.functional as F
import pdb

r"""Base class for all Autoencoder modules.

Any new AE based model should inherit from this class.  All AE's should have a forward pass method, used for
training the network, as well as methods for returning encodings and reconstructions individually.
"""


class Autoencoder(nn.Module):
    def __init__(self, configuration: dict):
        """
        Basic setup common to all Autoencoder models. For now, just determines which device the model should be run on.

        :param configuration: Dictionary of model parameters
        """
        super(Autoencoder, self).__init__()
        self.configuration = configuration
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        torch.backends.cudnn.benchmark = True
        self.current_epoch: int = 0

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def embedding(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def reconstruction(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def optimize_parameters(self):
        raise NotImplementedError

    def post_processing(self) -> None:
        return


r"""The most basic of autoencoders.  Consists of a single Encoder --> Bottleneck --> Decoder architecture."""


class SimpleAutoencoder(Autoencoder):
    def __init__(self, configuration: dict):
        r"""
        Defines encoder/decoder modules.

        TODO: Add more stuff to configuration
        :param configuration: Only thing specific to AE for now is "input_size".
        """
        super(SimpleAutoencoder, self).__init__(configuration)

        encoder_layers = []
        decoder_layers = []

        for in_dim, out_dim in zip(configuration["layer_sizes"], configuration["layer_sizes"][1:]):
            encoder_layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            encoder_layers.append(nn.ReLU())

            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(in_features=out_dim, out_features=in_dim))

        decoder_layers.reverse()

        self.encoder = nn.Sequential(*encoder_layers).to(self.device)
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=configuration['lr']
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)

        self.loss = self.criterion(x, reconstruction)

        return embedding, reconstruction

    def embedding(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x.to(self.device)
        return self.encoder(x)

    def reconstruction(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x.to(self.device)
        return self.decoder(self.encoder(x))

    def optimize_parameters(self) -> None:
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


class BatchDecoder(Autoencoder):
    def __init__(self, configuration):
        super().__init__(configuration)

        encoder_layers = []
        decoder_layers = []
        self.num_batches = configuration["num_batches"]

        for i in range(len(configuration["layer_sizes"])-1):
            in_dim = configuration["layer_sizes"][i]
            out_dim = configuration["layer_sizes"][i+1]

            encoder_layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))

            if i+1 < len(configuration["layer_sizes"])-1:
                encoder_layers.append(nn.ReLU())

            if i+1 < len(configuration["layer_sizes"])-1:
                decoder_layers.append(nn.Linear(in_features=out_dim, out_features=in_dim))
                decoder_layers.append(nn.ReLU())
            else:
                decoder_layers.append(nn.Linear(in_features=out_dim+self.num_batches, out_features=in_dim))

        decoder_layers.reverse()

        self.encoder = nn.Sequential(*encoder_layers).to(self.device)
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=configuration['lr']
        )


    def forward(self, x: torch.Tensor, z: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        embedding = self.encoder(x)

        z = z.reshape(len(z), 1)
        z = one_hot(z, self.num_batches).to(self.device)
        embedding_and_batch = torch.cat([embedding, z], dim=1)
        reconstruction = self.decoder(embedding_and_batch)

        self.loss = self.criterion(x, reconstruction)

        return embedding, reconstruction


    def optimize_parameters(self) -> None:
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


    def embedding(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x.to(self.device)
        return self.encoder(x)


class HilbertSchmidtAE(Autoencoder):
    def __init__(self, autoencoder: Autoencoder, configuration: dict):
        super(HilbertSchmidtAE, self).__init__(configuration)
        self.autoencoder: Autoencoder = autoencoder
        self.lambda_param = configuration["lambda"]

    def forward(self, x: torch.Tensor, z: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x, z = x.to(self.device), z.to(self.device)
        embedding, reconstruction = self.autoencoder(x=x, z=z)

        self.hsic = hilbert_schmidt(embedding, z) if len(torch.unique(z)) > 1 else 0
        self.reconstruction_loss = self.autoencoder.loss
        self.loss = self.reconstruction_loss + self.lambda_param * self.hsic

        return embedding, reconstruction

    def embedding(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x.to(self.device)
        return self.autoencoder.embedding(x=x)

    def reconstruction(self, x: torch.Tensor, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = x.to(self.device)
        return self.autoencoder.reconstruction(x=x, z=z)

    def optimize_parameters(self) -> None:
        self.autoencoder.optimizer.zero_grad()
        self.loss.backward()
        self.autoencoder.optimizer.step()

    def log_progress(self) -> None:
        logging.info("Reconstruction loss, hsic: {:3f}, {:3f}".format(
            self.reconstruction_loss,
            self.hsic
        ))
