import argparse
import time
import pdb

from anndata import AnnData
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import scvi

from pathlib import Path
import shutil

from desc_ethan import desc
from data import DatasetWithConfounder
from models import BatchDecoder
from models import HilbertSchmidtAE
from utils import configure_logging
from utils import embed_data, train
from utils import load_data
from utils import set_seeds

configure_logging()
set_seeds(12345)

parser = argparse.ArgumentParser(description="Experiment parameters")
parser.add_argument("model", type=str, choices=["hd-md", "desc", "scvi"], help="Which model to run")
parser.add_argument("dataset", type=str, choices=["pbmc", "pancreas", "pancreas_generalization", "pbmc_generalization"], help="Which dataset to choose")

args = parser.parse_args()
dataset = args.dataset
model_type = args.model

adata = load_data(dataset, normalized=False if model_type == "scvi" else True)

adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
adata.obs["study"] = adata.obs["study"].astype("category")

source_adata = adata

if model_type == "hd-md":
    train_set = DatasetWithConfounder(
        X=source_adata.X.copy(),
        Z=LabelEncoder().fit_transform(source_adata.obs.study)
    )
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    decoder_class = BatchDecoder

    model = HilbertSchmidtAE(
        decoder_class(
            configuration={
                "layer_sizes": [adata.X.shape[1], 500, 250, 50],
                "lr": 1e-3,
                "num_batches": len(adata.obs['study'].unique()),
            }
        ),
        configuration={
            "lambda": 1,
        }
    )

    t0 = time.time()

    train(model, train_loader, 100, logging_interval=5)
    print('{} seconds'.format(time.time() - t0))

    eval_set = DatasetWithConfounder(X=source_adata.X.copy(), Z=LabelEncoder().fit_transform(source_adata.obs.study))
    eval_loader = DataLoader(eval_set, batch_size=128)
    results = embed_data(model, eval_loader)

    eval_embeddings = AnnData(X=results, obs={
        "study":source_adata.obs["study"].values,
        "cell_type":source_adata.obs["cell_type"].values
    })

    eval_embeddings.write("results/{}/hd-md.h5ad".format(args.dataset))

elif model_type == "desc":
    source_adata = desc.scale_bygroup(source_adata, groupby="study", max_value=6)

    model_cache_dir = Path("result_tmp")
    if model_cache_dir.exists() and model_cache_dir.is_dir():
        print("Removing cached result")
        shutil.rmtree(model_cache_dir)

    _, desc_model = desc.train(
        source_adata,
        save_dir=str(model_cache_dir),
        louvain_resolution=[0.6] if dataset == "pbmc" or dataset == "pbmc_generalization" else [0.2],
        verbose=False
    )

    desc_results = AnnData(X=desc_model.extract_features(source_adata.X), obs={
        "study": source_adata.obs["study"].values,
        "cell_type": source_adata.obs["cell_type"].values
    })
    desc_results.write("results/{}/desc.h5ad".format(dataset))

elif model_type == "scvi":
    early_stopping_kwargs = {
        "early_stopping_metric": "elbo",
        "save_best_state_metric": "elbo",
        "patience": 10,
        "threshold": 0,
        "reduce_lr_on_plateau": True,
        "lr_patience": 8,
        "lr_factor": 0.1,
    }

    scvi.data.setup_anndata(adata, batch_key="study")
    model = scvi.model.SCVI(adata)
    model.train(n_epochs=1000, frequency=1, early_stopping_kwargs=early_stopping_kwargs)
    latent = model.get_latent_representation()
    eval_embeddings = AnnData(X=latent, obs={
        "study": adata.obs["study"].values,
        "cell_type": adata.obs["cell_type"].values
    })
    eval_embeddings.write("results/{}/scvi.h5ad".format(args.dataset))

