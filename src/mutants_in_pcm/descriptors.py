# -*- coding: utf-8 -*-


"""Protein and molecular descriptor calculation."""

import os
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy.spatial.distance as sp
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from prodec import ProteinDescriptors, Transform, TransformType

import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text


def plot_descriptor_pca(outfile: str, format: str = 'svg', only_first_pcs: Optional[int] = 2,
                        descriptors: Optional[List[str]] = (
                        'ProtFP PCA', 'VHSE', 'Tscale', 'FASGAI', 'VARIMAX', 'MS-WHIM', 'STscale',
                        'Zscale Sandberg')) -> None:
    """Create a PCA projection plot of the
    different descriptors available in ProDEC.

    see DOI: 10.1023%2FA%3A1010933404324

    :param outfile: Path to the figure to be saved
    :param format: format of the output figure file
    :param only_first_pcs: number of first principal components to be used in the computation.
    Must be a positive integer (default = None, i.e. use all dimensions).
    :param descriptors: list of IDs of descriptors to be considered (default = None, i.e. use all available)
    """
    # Instantiate a descriptor factory
    desc_factory = ProteinDescriptors()
    descriptor_names = desc_factory.available_descriptors
    # Obtain descriptor distances between amino acid pairs
    data = {}
    pbar = tqdm(descriptor_names)
    for descriptor_name in pbar:
        pbar.set_description(f'Processing {descriptor_name}:')
        desc = desc_factory.get_descriptor(descriptor_name)
        # Skip distance and hash descriptors
        if desc.Type == 'Distance' or desc.ID in ['ProtFP hash']:
            continue
        # Skip descriptors not set in arguments
        if descriptors is not None and desc.ID not in descriptors:
            continue
        desc_values = pd.DataFrame(np.array(desc.get('ACDEFGHIKLMNPQRSTVWY', flatten=False)), index=list('ACDEFGHIKLMNPQRSTVWY'), columns=range(1, desc.Size + 1))
        # Scale values
        desc_values.iloc[:, :] = (desc_values.iloc[:, :] - desc_values.iloc[:, :].min().min()) / (desc_values.iloc[:, :].max().max() - desc_values.iloc[:, :].min().min())
        # Keep first PCs
        if desc_values.squeeze().ndim > 1:
            if only_first_pcs is not None and isinstance(only_first_pcs, int) and only_first_pcs > 0:
                desc_values = desc_values.iloc[:, :only_first_pcs]
            elif only_first_pcs is not None:
                raise ValueError('only_first_pcs must be a positive integer')
        # Minmax-scale matrix
        desc_values.iloc[:, :] = (desc_values - np.mean(desc_values.values)) / np.std(desc_values.values)
        # Create similarity matrix with euclidean distance
        sim_matrix = sp.squareform(sp.pdist(desc_values.values))
        data[descriptor_name] = sim_matrix
    # Calculate difference between pairs of descriptors
    desc_dist_avg = np.zeros((len(data), len(data)))
    desc_dist_std = np.zeros((len(data), len(data)))
    desc_names = list(data.keys())
    for desc_pair in combinations(desc_names, r=2):
        desc1, desc2 = desc_pair[0], desc_pair[1]
        desc1_ix, desc2_ix = desc_names.index(desc1), desc_names.index(desc2)
        diff = np.linalg.norm(data[desc1] - data[desc2])
        desc_dist_avg[desc1_ix, desc2_ix] = desc_dist_avg[desc2_ix, desc1_ix] = diff.mean()
        desc_dist_std[desc1_ix, desc2_ix] = desc_dist_std[desc2_ix, desc1_ix] = diff.std()
    # Save avg and std values to files
    np.savetxt(fname=f"{os.path.join(os.path.dirname(outfile), 'average_descriptor_pair_distances.npy')}",
               X=desc_dist_avg,
               delimiter='\t',
               encoding='utf-8')
    np.savetxt(fname=f"{os.path.join(os.path.dirname(outfile), 'stdev_descriptor_pair_distances.npy')}",
               X=desc_dist_std,
               delimiter='\t',
               encoding='utf-8')
    # Carry out PCA
    # scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=1234)
    desc_dist_avg = (desc_dist_avg - np.mean(desc_dist_avg)) / np.std(desc_dist_avg)
    desc_dist_avg = pd.DataFrame(desc_dist_avg, columns=desc_names, index=desc_names)
    pca.fit(desc_dist_avg[desc_dist_avg.index.isin((
                        'ProtFP PCA', 'VHSE', 'Tscale', 'FASGAI', 'VARIMAX', 'MS-WHIM', 'STscale',
                        'Zscale Sandberg')) & desc_dist_avg.columns.isin((
                        'ProtFP PCA', 'VHSE', 'Tscale', 'FASGAI', 'VARIMAX', 'MS-WHIM', 'STscale',
                        'Zscale Sandberg'))])
    # pca_values = pca.fit_transform(desc_dist_avg)
    pca_values = pca.transform(desc_dist_avg)

    # Plot distances
    sns.set_style("whitegrid")
    sns.set(font_scale=1.5, rc={'text.usetex': True})
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    plot = sns.scatterplot(x=-pca_values[:, 0], y=pca_values[:, 1], ax=ax)
    x_min, x_max = plot.get_xlim()
    y_min, y_max = plot.get_ylim()
    plot.set(xlabel=rf'$PC1$ $({pca.explained_variance_ratio_[0] * 100:.2f}\%)$',
             ylabel=rf'$PC2$ $({pca.explained_variance_ratio_[1] * 100:.2f}\%)$',
             xlim=(x_min * 2, x_max * 2),
             ylim=(y_min * 2, y_max * 2)
             )
    texts = [plt.text(-x, y, f'${name}$'.replace('ö', r'\ddot{o}')) for name, x, y in zip(desc_names, pca_values[:, 0], pca_values[:, 1])]
    adjust_text(texts, x=pca_values[:, 0], y=pca_values[:, 1], arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
                # expand_points=(100., 100.),
                # expand_text=(10., 10.),
                force_text=1,
                lim=10_000)
    plot.get_figure().savefig(outfile, format=format)
