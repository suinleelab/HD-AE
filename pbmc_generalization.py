import argparse
import time
import pdb

from anndata import AnnData
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from data import DatasetWithConfounder
from models import BatchDecoder
from models import HilbertSchmidtAE
from utils import configure_logging
from utils import embed_data, train
from utils import load_data
from utils import set_seeds

from desc_ethan import desc
import scvi
import shutil
from pathlib import Path

from tqdm import tqdm

configure_logging()
set_seeds(12345)

parser = argparse.ArgumentParser(description="Experiment parameters")
parser.add_argument("model", type=str, choices=["hd-md", "hd-md_generalization", "desc", "scvi"], help="Which model to run")

args = parser.parse_args()
model_type = args.model

dataset = "pbmc_generalization"
excluded_cells = []

target_batches = ["Drop-seq", "10x Chromium (v3)"]

adata = load_data(dataset, normalized=False if model_type == 'scvi' else True)
adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
adata.obs["study"] = adata.obs["study"].astype("category")


source_adata = adata[~adata.obs["study"].isin(target_batches)]
target_adata = adata[adata.obs["study"].isin(target_batches)]

num_runs = 5

for i in tqdm(range(num_runs)):

    if model_type == 'hd-md':
        train_set = DatasetWithConfounder(X=adata.X.copy(), Z=LabelEncoder().fit_transform(adata.obs.study))
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

        eval_adata = AnnData.concatenate(source_adata, target_adata)

        eval_set = DatasetWithConfounder(X=eval_adata.X.copy(), Z=LabelEncoder().fit_transform(eval_adata.obs.study))
        eval_loader = DataLoader(eval_set, batch_size=128)
        results = embed_data(model, eval_loader)

        hd_md_eval_embeddings = AnnData(X=results, obs={
            "study": eval_adata.obs["study"].values,
            "cell_type": eval_adata.obs["cell_type"].values
        })

        hd_md_eval_embeddings.obs["cell_type"] = hd_md_eval_embeddings.obs["cell_type"].astype("category")
        hd_md_eval_embeddings.obs["study"] = hd_md_eval_embeddings.obs["study"].astype("category")

        hd_md_eval_embeddings.write("results/{}/hd-md_{}.h5ad".format(dataset, str(i)))

    elif model_type == 'hd-md_generalization':

        train_set = DatasetWithConfounder(X=source_adata.X.copy(), Z=LabelEncoder().fit_transform(source_adata.obs.study))
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

        eval_adata = AnnData.concatenate(source_adata, target_adata)

        eval_set = DatasetWithConfounder(X=eval_adata.X.copy(), Z=LabelEncoder().fit_transform(eval_adata.obs.study))
        eval_loader = DataLoader(eval_set, batch_size=128)
        results = embed_data(model, eval_loader)

        hd_md_eval_embeddings = AnnData(X=results, obs={
            "study": eval_adata.obs["study"].values,
            "cell_type": eval_adata.obs["cell_type"].values
        })

        hd_md_eval_embeddings.obs["cell_type"] = hd_md_eval_embeddings.obs["cell_type"].astype("category")
        hd_md_eval_embeddings.obs["study"] = hd_md_eval_embeddings.obs["study"].astype("category")

        hd_md_eval_embeddings.write("results/{}/hd-md_generalization_{}.h5ad".format(dataset, str(i)))

    elif model_type == 'desc':
        source_adata = desc.scale_bygroup(adata, groupby="study", max_value=6)

        model_cache_dir = Path("test")
        if model_cache_dir.exists() and model_cache_dir.is_dir():
            print("Removing cached result")
            shutil.rmtree(model_cache_dir)

        desc_results, desc_model = desc.train(
            source_adata,
            save_dir=str(model_cache_dir),
            louvain_resolution=[0.6] if dataset == "pbmc_generalization" else [0.2],
            verbose=False
        )

        eval_adata = AnnData.concatenate(source_adata, target_adata)

        desc_embeddings = AnnData(X=desc_model.extract_features(eval_adata.X), obs={
            "study": eval_adata.obs["study"].values,
            "cell_type": eval_adata.obs["cell_type"].values
        })

        desc_embeddings.obs["cell_type"] = desc_embeddings.obs["cell_type"].astype("category")
        desc_embeddings.obs["study"] = desc_embeddings.obs["study"].astype("category")
        desc_embeddings.write("results/{}/desc_{}.h5ad".format(dataset, str(i)))

    elif model_type == 'scvi':

        early_stopping_kwargs = {
            "early_stopping_metric": "elbo",
            "save_best_state_metric": "elbo",
            "patience": 10,
            "threshold": 0,
            "reduce_lr_on_plateau": True,
            "lr_patience": 8,
            "lr_factor": 0.1,
        }

        scvi.data.setup_anndata(adata, batch_key='study')
        model = scvi.model.SCVI(adata)
        model.train(n_epochs=1000, frequency=1, early_stopping_kwargs=early_stopping_kwargs)

        eval_adata = AnnData.concatenate(source_adata, target_adata)

        latent = model.get_latent_representation(eval_adata)

        scvi_eval_embeddings = AnnData(X=latent, obs={
            "cell_type": eval_adata.obs["cell_type"],
            "study": eval_adata.obs["study"]
        })

        scvi_eval_embeddings.write("results/{}/scvi_{}.h5ad".format(dataset, str(i)))