import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import sem
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import entropy_batch_mixing
from utils import load_results
from utils import knn_purity

methods = ['mnnCorrect', 'seurat', 'harmony', 'conos', 'saucie', 'desc', 'scAlign', 'scvi', 'hd-md', 'hd-md_generalization']
proper_names = ['MNN', 'Seurat', 'Harmony', 'Conos', 'SAUCIE', 'DESC', 'scAlign', 'scVI', 'HD-AE', 'HD-AE*']
deep_methods = ['saucie', 'desc', 'scvi', 'hd-md', 'hd-md_generalization']

### ARI CALCULATIONS ###

resolutions = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

scores = np.zeros((len(resolutions), len(methods)))
errors = np.zeros((len(resolutions), len(methods)))
method_list = []
num_iterations = 5
for i, method in tqdm(enumerate(methods)):

    scores_method = np.zeros((len(resolutions), num_iterations))

    if method in deep_methods:
        for idx in range(num_iterations):
            results = load_results(method, 'pbmc_generalization', idx)
            sc.pp.neighbors(results)

            for j, resolution in enumerate(resolutions):
                sc.tl.leiden(results, resolution=resolution, random_state=42)
                score = adjusted_rand_score(results.obs['leiden'],
                                            LabelEncoder().fit_transform(results.obs['cell_type']))
                scores_method[j, idx] = score

        for j, resolution in enumerate(resolutions):
            scores[j, i] = scores_method[j, :].mean()
            errors[j, i] = sem(scores_method[j, :])
    else:
        results = load_results(method, 'pbmc_generalization')
        sc.pp.neighbors(results)

        for idx in range(num_iterations):

            for j, resolution in enumerate(resolutions):
                sc.tl.leiden(results, resolution=resolution, random_state=42)
                score = adjusted_rand_score(results.obs['leiden'],
                                            LabelEncoder().fit_transform(results.obs['cell_type']))
                scores_method[j, idx] = score

        for j, resolution in enumerate(resolutions):
            scores[j, i] = scores_method[j, :].mean()
            errors[j, i] = sem(scores_method[j, :])
    method_list.extend([method] * len(resolutions))

scores = pd.DataFrame(scores)
scores.columns = methods

scores.to_pickle("scores")
np.save("errors", errors)

### SILHOUETTE CALCULATIONS ###

silhouette_scores = []
for method in tqdm(methods):
    if method in deep_methods:
        results = load_results(method, 'pbmc_generalization', 0)
    else:
        results = load_results(method, 'pbmc_generalization')

    sc.pp.neighbors(results)
    sc.tl.umap(results)
    labels = LabelEncoder().fit_transform(results.obs['cell_type'].to_numpy())
    silhouette_scores.append(silhouette_samples(results.obsm['X_umap'], labels, n_jobs=-1))

silhouette_df = pd.DataFrame({
    "Score": np.concatenate(silhouette_scores),
    "Method": np.concatenate([np.repeat(proper_names[i], len(silhouette_scores[i])) for i in range(len(proper_names))]),
    "Dataset": np.repeat("PBMC", len(np.concatenate(silhouette_scores)))
})

silhouette_df.to_pickle("silhouette_df")

### EBM CALCULATIONS ###

print("Calculating EBM")

method_list = []
num_iterations = 5
num_neighbors = [15, 25, 50, 100, 200, 300]
for k in num_neighbors:
    print("k=", k)
    ebm_scores = np.zeros((len(methods), num_iterations))
    for i, method in tqdm(enumerate(methods)):

        if method in deep_methods:

            for idx in range(num_iterations):
                results = load_results(method, 'pbmc_generalization', idx)
                ebm = entropy_batch_mixing(
                    results.X,
                    results.obs.study,
                    n_neighbors=k,
                    n_pools=50,
                    n_samples_per_pool=100
                )
                ebm_scores[i, idx] = ebm
        else:
            results = load_results(method, 'pbmc_generalization')
            ebm = entropy_batch_mixing(
                results.X,
                results.obs.study,
                n_neighbors=k,
                n_pools=50,
                n_samples_per_pool=100
            )
            ebm_scores[i, :] = ebm

    np.save("ebm_scores_{}".format(k), ebm_scores)

### knn purity calculations ###

print("Calculating kNN purity")

method_list = []
num_iterations = 5

num_neighbors = [15, 25, 50, 100, 200, 300]
for k in num_neighbors:
    print("k=", k)
    purity_scores = np.zeros((len(methods), num_iterations))
    for i, method in tqdm(enumerate(methods)):
        if method in deep_methods:
            for idx in range(num_iterations):
                results = load_results(method, 'pbmc_generalization', idx)
                purity = knn_purity(
                    results,
                    'pbmc_generalization',
                    'study',
                    method,
                    k=k
                )
                purity_scores[i, idx] = purity
        else:
            results = load_results(method, 'pbmc_generalization')
            purity = knn_purity(
                results,
                'pbmc_generalization',
                'study',
                method,
                k=k
            )
            purity_scores[i, :] = purity

    np.save("purity_scores_{}".format(k), purity_scores)