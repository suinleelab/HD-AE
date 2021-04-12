import argparse
import time
import pdb

from anndata import AnnData
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import numpy as np

from data import DatasetWithConfounder
from models import BatchDecoder
from models import HilbertSchmidtAE
from utils import configure_logging
from utils import embed_data, train
from utils import load_data
from utils import set_seeds

import scanpy as sc

from tqdm import tqdm

configure_logging()
set_seeds(12345)

dataset = "pbmc_generalization"
excluded_cells = []

adata = load_data(dataset, normalized=True)
adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
adata.obs["study"] = adata.obs["study"].astype("category")

target_batches = ["Drop-seq", "10x Chromium (v3)"]
source_batches = adata[~adata.obs["study"].isin(target_batches)].obs['study'].unique()
np.random.shuffle(source_batches)

num_runs = 1

for i in tqdm(range(1, len(source_batches))):
    target_batches = ["Drop-seq", "10x Chromium (v3)"]
    source_batches_sample = source_batches[:i+1]
    source_adata = adata[adata.obs["study"].isin(source_batches_sample)]

    placeholder_adata = adata[~adata.obs["study"].isin(target_batches)]
    target_adata = adata[adata.obs["study"].isin(target_batches)]


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

    eval_adata = AnnData.concatenate(placeholder_adata, target_adata)

    eval_set = DatasetWithConfounder(X=eval_adata.X.copy(), Z=LabelEncoder().fit_transform(eval_adata.obs.study))
    eval_loader = DataLoader(eval_set, batch_size=128)
    results = embed_data(model, eval_loader)

    hd_md_eval_embeddings = AnnData(X=results, obs={
        "study": eval_adata.obs["study"].values,
        "cell_type": eval_adata.obs["cell_type"].values
    })

    hd_md_eval_embeddings.obs["cell_type"] = hd_md_eval_embeddings.obs["cell_type"].astype("category")
    hd_md_eval_embeddings.obs["study"] = hd_md_eval_embeddings.obs["study"].astype("category")

    hd_md_eval_embeddings.write("results/{}/subsampling/hd-md_{}.h5ad".format(dataset, str(i+1)))
