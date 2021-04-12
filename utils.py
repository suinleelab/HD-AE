import logging
import os
import random
from os import listdir
from os.path import join, isfile
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from sklearn.decomposition import PCA
import h5py
from scipy.stats import entropy, itemfreq
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
# from models import Autoencoder --- Need to fix recursive dependency here
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


def configure_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def train(net, trainloader: DataLoader, NUM_EPOCHS: int, logging_interval: int) -> None:
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data, label in trainloader:
            net(x=data, z=label)
            net.optimize_parameters()
            running_loss += net.loss.item()
        net.post_processing()
        loss = running_loss / len(trainloader)
        if (epoch+1) % logging_interval == 0:
            logging.info('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, NUM_EPOCHS, loss))
            net.log_progress()


def reconstruct_data(model, dataloader: DataLoader) -> torch.Tensor:
    output_list = []
    for X_batch, Z_batch in dataloader:
        reconstruction = model.reconstruction(X_batch, Z_batch)
        output_list.append(reconstruction)
    output_list = torch.cat(output_list, dim=0).detach().cpu().numpy()
    return output_list


def embed_data(model, dataloader: DataLoader) -> torch.Tensor:
    output_list = []
    with torch.no_grad():
        for X_batch, Z_batch in dataloader:
            embedding = model.embedding(X_batch, Z_batch)
            output_list.append(embedding)
        output_list = torch.cat(output_list, dim=0).detach().cpu().numpy()
    return output_list


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def silhouette_score(adata, label_key):
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    labels = LabelEncoder().fit_transform(adata.obs[label_key].to_numpy())
    scores = silhouette_samples(adata.obsm['X_umap'], labels, n_jobs=-1)
    return scores





def gram_matrix(x, sigma=1):
    pairwise_distances = x.unsqueeze(1) - x
    return torch.exp(-pairwise_distances.norm(2, dim=2) / (2 * sigma * sigma))


def entropy_batch_mixing(latent, labels, n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    def entropy_from_indices(indices):
        return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)))

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=-1).fit(latent)
    indices = neighbors.kneighbors(latent, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: labels[i])(indices)

    entropies = np.apply_along_axis(entropy_from_indices, axis=1, arr=batch_indices)

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])

    return score


def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return Variable(targets)


def load_data(dataset, normalized):
    data_path = "data/{}/{}/data/".format("normalized" if normalized else "raw", dataset)
    cell_path = "data/{}/{}/cell_labels/".format("normalized" if normalized else "raw", dataset)

    data_files = sorted([join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))])
    cell_label_files = sorted([join(cell_path, f) for f in listdir(cell_path) if isfile(join(cell_path, f))])

    df_list = []
    batch_labels = []
    cell_labels = []

    for i, (data_file, label_file) in enumerate(zip(data_files, cell_label_files)):
        df = pd.read_csv(data_file, index_col=0).astype("float32")
        df = df.T
        df_list.append(df)
        batch_labels.append(np.repeat(data_file.split("/")[-1].split(".")[0], df.shape[0]))
        cell_labels.append(pd.read_csv(label_file, index_col=0).x.values)

    X = pd.concat(df_list, axis=0)
    Z = np.concatenate(batch_labels)
    cell_labels = np.concatenate(cell_labels)

    adata = AnnData(X=X.reset_index(drop=True), obs={
        "study": Z,
        "cell_type": cell_labels
    })
    return adata



def load_results(method, dataset, idx=None):
    if 'hd-md' in method or method == 'scvi' or method == 'scvi_generalization' or method == 'desc' or method == 'saucie' or method == 'saucie_generalization':
        if idx is not None:
            results_adata = sc.read("results/{}/{}_{}.h5ad".format(dataset, method, idx))
        else:
            results_adata = sc.read("results/{}/{}.h5ad".format(dataset, method))

    elif method == 'conos':
        source_adata = load_data(dataset, normalized=True)
        f = h5py.File("results/{}/conos.h5".format(dataset), 'r')
        x = np.asarray(f['pseudopca']['pseudopca.df'])
        x = x.view(np.float64).reshape(x.shape + (-1,))

        batch_labels = pd.read_csv("results/{}/batch_labels.csv".format(dataset), index_col=0).x

        results_adata = AnnData(X=x, obs={
            "study": batch_labels.values,
            "cell_type": source_adata.obs["cell_type"].values
        })
    else:
        source_adata = load_data(dataset, normalized=True)
        results = pd.read_csv("results/{}/{}.csv".format(dataset, method))

        results = results.transpose()
        results.columns = results.iloc[0]
        results = results[1:]
        results = results.reset_index(drop=True)

        if method == "mnnCorrect" or "seurat":
            if method == 'seurat':
                results = StandardScaler().fit_transform(results)
            results = PCA(n_components=50).fit_transform(results)

        batch_labels = pd.read_csv("results/{}/batch_labels.csv".format(dataset), index_col=0).x

        results_adata = AnnData(X=results, obs={
            "study": batch_labels.values,
            "cell_type": source_adata.obs["cell_type"].values
        })

    return results_adata


def silhouette(method, dataset):
    results_adata = load_results(method, dataset)
    silhouette = silhouette_score(results_adata, "cell_type")
    return silhouette


def plot_results(method, dataset, keys, frameon=False, title=None, legend_loc='right margin'):
    set_seeds()
    sc.set_figure_params(dpi=200)
    results_adata = load_results(method, dataset)
    sc.pp.neighbors(results_adata)
    sc.tl.umap(results_adata, random_state=42)

    study_cmap_map = study_cmaps[dataset]
    cell_cmap_map = cell_type_cmaps[dataset]
    results_adata.uns["study_colors"] = [study_cmap_map[study] for study in results_adata.obs["study"].cat.categories]
    results_adata.uns["cell_type_colors"] = [cell_cmap_map[cell_type] for cell_type in
                                             results_adata.obs["cell_type"].cat.categories]

    for key in keys:
        sc.pl.umap(results_adata, color=key, frameon=frameon, legend_loc=legend_loc, title=title)


def JaccardIndex(x1, x2):
    intersection = np.sum(x1 * x2)
    union = np.sum((x1 + x2) > 0)
    return intersection / union

def ebm(method, dataset, excluded_cells=[]):
    results_adata = load_results(method, dataset)
    results_adata = results_adata[(~results_adata.obs["cell_type"].isin(excluded_cells))]
    ebm = entropy_batch_mixing(
        results_adata.X,
        results_adata.obs.study,
        n_neighbors=50,
        n_pools=50,
        n_samples_per_pool=100
    )
    return ebm


def knn_purity(results, dataset, label_key, method, k=50):
    purities = []
    full_data = load_data(dataset, normalized=False if method == 'scvi' or method == 'scvi_generalization' else True)

    for batch in results.obs[label_key].unique():

        combined_batch = results[results.obs[label_key] == batch]

        knn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
        original = full_data[full_data.obs["study"] == batch].X

        nbrs = knn.fit(original)
        nbrs = nbrs.kneighbors_graph(original).toarray()
        np.fill_diagonal(nbrs, 0)

        nbrs_ = knn.fit(combined_batch.X)
        nbrs_ = nbrs_.kneighbors_graph(combined_batch.X).toarray()

        np.fill_diagonal(nbrs_, 0)
        JI_avg = np.array([JaccardIndex(x1, x2) for x1, x2 in zip(nbrs, nbrs_)])
        purities.extend(JI_avg)

    return np.mean(np.array(purities))

def JaccardIndex(x1, x2):
    intersection = np.sum(x1 * x2)
    union = np.sum((x1 + x2) > 0)
    return intersection / union