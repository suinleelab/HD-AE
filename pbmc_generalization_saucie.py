import SAUCIE
from utils import load_data
import tensorflow as tf
import numpy as np
from anndata import AnnData

dataset = 'pbmc_generalization'
adata = load_data(dataset, normalized=True)

target_batches = ["Drop-seq", "10x Chromium (v3)"]
source_adata = adata

for i in range(5):
    tf.reset_default_graph()
    saucie = SAUCIE.SAUCIE(source_adata.shape[1])
    loadtrain = SAUCIE.Loader(source_adata.X, shuffle=True)
    saucie.train(loadtrain, steps=1000)

    loadeval = SAUCIE.Loader(adata.X, shuffle=False)
    embedding = saucie.get_embedding(loadeval)

    eval_adata = AnnData.concatenate(source_adata)
    saucie_results = AnnData(
        X = embedding,
        obs={
            "cell_type": eval_adata.obs['cell_type'].values,
            "study": eval_adata.obs['study'].values
        }
    )
    saucie_results.write("results/{}/saucie_{}.h5ad".format(dataset, i))