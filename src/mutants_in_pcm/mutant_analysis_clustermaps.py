# -*- coding: utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mycolorpy import colorlist as mcp
import numpy as np
import pylab as pl
from matplotlib.patches import Patch
from matplotlib import colors
from math import floor
from matplotlib.cm import ScalarMappable

from .mutant_analysis_protein import calculate_average_residue_distance_to_ligand
from .mutant_analysis_type import extract_residue_number_list

"""Mutant statistics analysis. Part X"""
"""Analyzing bioactivity values for (strictly) common subsets per accession using clustermaps"""


def pivot_bioactivity_data(data: pd.DataFrame, strictly_common: bool, threshold_update: int,
                           variant_coverage_update: float, save: bool = False, output_dir: str = None):
    """
    Make bioactivity dataset suitable for heatmap/clustermap plotting. It is possible to redefine the pre-computed
    common subset with more strict thresholds and coverage thresholds.
    :param data: dataframe with bioactivity data containing at least columns target_id, connectivity,
    and pchembl_value_Mean
    :param strictly_common: whether to keep only compounds that have been tested in all variants, which is preferred
    for clustermaps to avoid NANs (the number of variants can be reduced by increasing variant_coverage_update)
    :param threshold_update: Stricter threshold (Minimum number of variants in which a compound has been tested in order
    to be included in the common subset)
    :param variant_coverage_update: Stricter variant coverage threshold (Minimum ratio of the common subset of compounds
    that have been tested on a variant in order to include that variant in the output)
    :param save: whether to save the pivoted data
    :param output_dir: path to directory to save pivoted data
    :return: pivoted bioactivity dataframe with variants as index and compound connectivity as columns
    """
    if strictly_common:
        file_tag = 'clustermap'
    else:
        file_tag = 'heatmap'

    if output_dir is None:
        heatmap_df_file = ''
    else:
        heatmap_df_file = os.path.join(output_dir, f'{file_tag}_{threshold_update}_{variant_coverage_update}.csv')

    if not os.path.isfile(heatmap_df_file):
        # Pivot data to plot heatmap
        heatmap_df = data.pivot(index='target_id', columns='connectivity', values='pchembl_value_Mean')

        # Make threshold more strict
        if threshold_update is not None:
            # Drop compounds not tested in at least threshold number of variants
            heatmap_df = heatmap_df.dropna(axis='columns', thresh=threshold_update)

        # Make variant coverage more strict
        if variant_coverage_update is not None:
            # Drop variants not tested for more than variant_coverage*100 % of common subset of compounds
            compound_threshold = variant_coverage_update * heatmap_df.shape[1]
            heatmap_df = heatmap_df.dropna(axis='index', thresh=compound_threshold)

        # Keep only strictly common subset (all compounds tested on all targets)
        if strictly_common:
            heatmap_df = heatmap_df[heatmap_df.columns[~heatmap_df.isnull().any()]]

        if save:
            heatmap_df_to_save = heatmap_df.reset_index()
            heatmap_df_to_save.to_csv(heatmap_df_file, sep='\t', index=False)
    else:
        heatmap_df = pd.read_csv(heatmap_df_file, sep='\t', index_col='target_id')

    return heatmap_df


def extract_unique_connectivity(data: pd.DataFrame, pivoted: bool):
    """
    Extract all unique compound connectivities from a bioactivity dataframe
    :param data: bioactivity dataframe
    :param pivoted: whether the bioactivity dataframe was pivoted
    :return: list of compound connectivities
    """
    if not pivoted:
        # Connectivity is a column
        unique_connectivity = data['connectivity'].unique().tolist()
    else:
        # Connectivities are the columns of the pivoted dataframe
        unique_connectivity = data.columns.tolist()

    return unique_connectivity

def extract_oldest_year(data: pd.DataFrame, accession: str, subset: list):
    """
    Get the oldest year each compound from a list was tested on any variant of a specific accession
    :param data: dataframe with bioactivity data. Must contain column 'Year'
    :param accession: Uniprot accession code of the target of interest
    :param subset: list of compounds' connectivities
    :return: dictionary mapping connectivities to years
    """
    # Make sure the bioactivity dataset only contains data for the target of interest
    data = data[data['accession'] == accession]

    # Subset the list of compounds of interest
    data = data[data['connectivity'].isin(subset)]

    # Remove instances where Year is not specified
    data = data.dropna(axis=0, subset=['Year'])

    # Keep oldest year per compound
    data_oldest = data.groupby('connectivity', group_keys=False).apply(lambda x: x.loc[x.Year.idxmin()])

    # Make dictionary mapping compounds and years
    year_dict = dict(zip(data_oldest['connectivity'],data_oldest['Year']))

    return year_dict

def plot_bioactivity_heatmap(accession: str, pivoted_data: pd.DataFrame, output_dir: str):
    """
    Plot heatmap of bioactivity data.
    :param accession: Uniprot accession code of the target of interest
    :param pivoted_data: pivoted bioactivity dataframe with variants as index and compound connectivity as columns
    :param output_dir: path to directory to save figures
    :return: figure
    """
    # Heatmap with full common subset (contains NAs)
    fig, ax = plt.subplots(1, 1, figsize=(24, 5))
    sns.heatmap(pivoted_data, cmap='mako_r', linewidth=0.1, linecolor='w', square=True,
                cbar_kws={'label': 'pChEMBL value (Mean)', 'aspect': 0.2})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, accession, f'heatmap_{accession}.svg'))

def plot_bioactivity_clustermap(accession: str, pivoted_data: pd.DataFrame, compound_annotation: str,
                                variant_annotation: str, output_dir: str, **kwargs):
    """
    Plot clustermap of accession's strictly common subset, optionally with molecule or variant annotations
    :param accession: Uniprot accession code of the target of interest
    :param pivoted_data: pivoted bioactivity dataframe with variants as index and compound connectivity as columns
    :param compound_annotation: property to annotate the compounds on. Options are 'butina_clusters' (Butina cluster
    id, requires 'connectivity_cluster_dict' and 'butina_cutoff' in kwargs), and 'year' (Year of first testing of the
    compound-accession
    date, requires 'connectivity_year_dict' in kwargs)
    :param variant_annotation: proeprty to annotate the variants on. Options are 'ligand_distance' (Distance from
    mutated residue to ligand COG in available crystal structures, requires 'dist_dir' in kwargs),
    and 'aa_change_epstein' (Epstein aa distance coefficient, requires 'epstein_dict' in kwargs)
    :param output_dir: path to output
    :param kwargs: dictionary of additional keyword arguments
    :return: figure
    """
    # Make sure the pivoted data includes only strictly common subset (zero NAs)
    pivoted_data = pivoted_data[pivoted_data.columns[~pivoted_data.isnull().any()]]

    if (compound_annotation == None and variant_annotation == None):
        # Clustermap with exclusively common subset (no NAs)
        sns.clustermap(pivoted_data, cmap='mako_r',
                       linewidth=0.1, linecolor='w', cbar_kws={'label': 'pChEMBL value (Mean)'})

        # save plot
        plt.savefig(os.path.join(output_dir, accession, f'clustermap_{accession}.svg'))

    if compound_annotation == 'butina_clusters':
        connectivity_cluster_dict = kwargs['connectivity_cluster_dict']
        butina_cutoff = kwargs['butina_cutoff']
        # Map strictly common subset to its butina cluster
        strict_subset_cluster = [connectivity_cluster_dict[connectivity] for connectivity in pivoted_data.columns]

        # Assign a color to each possible cluster
        clusters = sorted(list(set(connectivity_cluster_dict.values())))
        cluster_colors = mcp.gen_color(cmap='flare', n=len(clusters))
        cluster_color_dict = dict(zip(clusters, cluster_colors))

        # Map each molecule in the strictly common subset to its color based on the Butina cluster
        strict_subset_color = [cluster_color_dict[cluster] for cluster in strict_subset_cluster]

        # Clustermap with exclusively common subset (no NAs)
        sns.clustermap(pivoted_data, cmap='mako_r',
                       linewidth=0.1, linecolor='w', cbar_kws={'label': 'pChEMBL value (Mean)'},
                       col_colors=strict_subset_color)

        # Add legend of year-based color of molecules
        strict_subset_cluster_color_dict = dict(sorted(zip(strict_subset_cluster, strict_subset_color)))
        handles = [Patch(facecolor=color) for color in strict_subset_cluster_color_dict.values()]
        labels = strict_subset_cluster_color_dict.keys()

        plt.legend(handles, labels, title='Butina cluster',
                   bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')

        # save figure
        plt.savefig(os.path.join(output_dir, accession, f'clustermap_{accession}_ButinaCluster{butina_cutoff}_groups.svg'))

    elif compound_annotation == 'year':
        connectivity_year_dict = kwargs['connectivity_year_dict']
        # Map strictly common subset to its year
        strict_subset_year = [int(connectivity_year_dict[connectivity]) if (connectivity in connectivity_year_dict.keys())
                          else 0 for connectivity in pivoted_data.columns]

        # Assign a color to each possible year
        year_range = list(range(min(strict_subset_year), max(strict_subset_year) + 1, 1))
        year_colors = mcp.gen_color(cmap='flare', n=len(year_range))
        year_color_dict = dict(zip(year_range, year_colors))
        year_color_dict[0] = '#FFFFFF'  # Missing year is white

        # Map each molecule in the strictly common subset to its color based on the publication year
        strict_subset_color = [year_color_dict[int(year)] for year in strict_subset_year]

        # Clustermap with exclusively common subset (no NAs)
        sns.clustermap(pivoted_data, cmap='mako_r',
                       linewidth=0.1, linecolor='w', cbar_kws={'label': 'pChEMBL value (Mean)'},
                       col_colors=strict_subset_color)

        # Add legend of year-based color of molecules
        strict_subset_year_color_dict = dict(sorted(zip(strict_subset_year, strict_subset_color)))
        handles = [Patch(facecolor=color) for color in strict_subset_year_color_dict.values()]
        labels = strict_subset_year_color_dict.keys()

        plt.legend(handles, labels, title='Year',
                   bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')

        # save figure
        plt.savefig(os.path.join(output_dir, accession,f'clustermap_{accession}_year_groups.svg'))

    if variant_annotation == 'ligand_distance':
        dist_dir = kwargs['dist_dir']

        # Calculate distance to ligand from mutated residues
        target_id_list = pivoted_data.index.tolist()
        mutants_resn = extract_residue_number_list(target_id_list)

        distances_dict = calculate_average_residue_distance_to_ligand(accession=accession,
                                                                      resn=mutants_resn,
                                                                      common=False,
                                                                      pdb_dir=os.path.join(dist_dir, 'PDB'),
                                                                      output_dir=dist_dir)
        # Map distances to mutants
        mutants_dist = []
        for res in mutants_resn:
            if (res == 'WT') or (res == 'MUTANT'):
                mutants_dist.append(0)
            else:
                try:
                    mutants_dist.append(distances_dict[str(res)])
                except KeyError:
                    mutants_dist.append(0)

        # Create color map based on distances
        COLORS = sns.light_palette("darkred", reverse=True, as_cmap=False)

        # Colormap
        cmap = colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)

        # Normalizer
        def round_half_up(n, decimals=0):
            multiplier = 10 ** decimals

            return floor(n * multiplier + 0.5) / multiplier

        norm = colors.Normalize(vmin=floor(min(mutants_dist)), vmax=round_half_up(max(mutants_dist), 1))

        # Normalized colors. Each number of tracks is mapped to a color in the
        # color scale 'cmap'
        COLORS = cmap(norm(mutants_dist))

        # Clustermap with exclusively common subset (no NAs)
        sns.clustermap(pivoted_data, cmap='mako_r',
                       linewidth=0.1, linecolor='w', cbar_kws={'label': 'pChEMBL value (Mean)'},
                       row_colors=COLORS)
        # save figure
        plt.savefig(os.path.join(output_dir, accession,f'clustermap_{accession}_distance_groups.svg'))

        # Create the colorbar
        pl.figure(figsize=(4, 0.5))
        cax = pl.axes([0.1, 0.2, 0.8, 0.6])

        cb = pl.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            orientation="horizontal",
            cax=cax
        )

        # Remove the outline of the colorbar
        cb.outline.set_visible(False)

        # Set legend label and move it to the top (instead of default bottom)
        cb.set_label("Average distance of mutated residue\nCOG to ligand COG ($\\AA$)", size=10, labelpad=10)

        # save figure
        plt.savefig(os.path.join(output_dir, accession,f'clustermap_{accession}_distance_groups_legend.svg'))

    elif variant_annotation == 'aa_change_epstein':
        epstein_dict = kwargs['epstein_dict']

        # Map amino acid change to its Epstein coefficient
        mutants_epstein = [epstein_dict[f"{target_id.split('_')[1][0]}{target_id.split('_')[1][-1]}"] if
                           ((target_id.split('_')[1] != 'WT') and (target_id.split('_')[1] != 'MUTANT')) else 0 for
                           target_id in pivoted_data.index.tolist()]

        # Create color map based on distances
        COLORS = sns.light_palette("darkred", reverse=False, as_cmap=False)

        # Colormap
        cmap = colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)

        # Normalizer
        def round_half_up(n, decimals=0):
            multiplier = 10 ** decimals

            return floor(n * multiplier + 0.5) / multiplier

        norm = colors.Normalize(vmin=floor(min(mutants_epstein)), vmax=round_half_up(max(mutants_epstein), 1))

        # Normalized colors. Each number of tracks is mapped to a color in the
        # color scale 'cmap'
        COLORS = cmap(norm(mutants_epstein))

        # Clustermap with exclusively common subset (no NAs)
        sns.clustermap(pivoted_data, cmap='mako_r',
                       linewidth=0.1, linecolor='w', cbar_kws={'label': 'pChEMBL value (Mean)'},
                       row_colors=COLORS)
        # save figure
        plt.savefig(os.path.join(output_dir, accession,f'clustermap_{accession}_epstein_groups.svg'))

        # Create the colorbar
        pl.figure(figsize=(4, 0.5))
        cax = pl.axes([0.1, 0.2, 0.8, 0.6])

        cb = pl.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            orientation="horizontal",
            cax=cax
        )

        # Remove the outline of the colorbar
        cb.outline.set_visible(False)

        # Set legend label and move it to the top (instead of default bottom)
        cb.set_label("Epstein coefficient of difference", size=10, labelpad=10)

        # save figure
        plt.savefig(os.path.join(output_dir, accession, f'clustermap_{accession}_epstein_groups_legend.svg'))





