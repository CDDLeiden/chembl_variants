# -*- coding: utf-8 -*-


"""Mutant statistics analysis. Part 2"""
"""Analyzing mutant data across/per accession"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import papyrus_scripts
import numpy as np
import re
import copy

from .preprocessing import merge_chembl_papyrus_mutants
from .mutant_analysis_family import obtain_chembl_family,group_families
# from .mutant_analysis_clustermaps import read_common_subset,extract_unique_connectivity

# For reference, just Papyrus data
def filter_explore_activity_data(papyrus_version, target_accession_list):
    """
    Filter Papyrus dataset for targets of interest and explore the statistics of the resulting dataset

    Parameters
    ----------
    papyrus_version : str
        Version of the Papyrus dataset to read
    targets : dict
        Dictionary with target labels as keys and UniProt accession codes as values

    Returns
    -------
    pandas.DataFrame
        Filtered bioactivity dataset for input targets
    """
    # Read downloaded Papyrus dataset in chunks, as it does not fit in memory
    CHUNKSIZE = 100000
    data = papyrus_scripts.read_papyrus(
        version=papyrus_version, chunksize=CHUNKSIZE)

    # Create filter for targets of interest
    filter = papyrus_scripts.keep_accession(data, target_accession_list)

    # Iterate through chunks and apply the filter defined
    filtered_data = papyrus_scripts.preprocess.consume_chunks(
        filter,
        total=-(
            -papyrus_scripts.utils.IO.get_num_rows_in_file("bioactivities", False) // CHUNKSIZE
        ),
    )

    # Plot distribution of activity values (pchembl_value_Mean) per target
    g = sns.displot(
        filtered_data,
        x="pchembl_value_Mean",
        hue="target_id",
        element="step"
    )

    return filtered_data

def filter_accession_data(data: pd.DataFrame, accession: str):
    """
    Filter bioactivity data for an accession of interest.
    :param accession: Uniprot accession protein code
    :return: dataframe for protein of interest
    """
    accession_data = data[data['accession'] == accession]

    return accession_data

def double_density_pchembl_year(accession_data, subset, output_dir, accession, subset_name):
    """
    Plot accession bioactivity scatter plot with Year vs pChEMBL value, with density marginals to visualzie the
    distribution of mutant bioactivity points over time and how the bioactivity evolved.
    :param accession_data:
    :param subset:
    :param output_dir:
    :param accession:
    :param subset_name:
    :return:
    """
    # filter common subset
    subset_df = accession_data[accession_data['connectivity'].isin(subset)]

    # order mutants by first mutated residue (WT first) and assign this to color order
    mutant_list = subset_df['target_id'].unique().tolist()
    mutant_list.remove(f'{accession}_WT')
    sorted_mutant_list = sorted(mutant_list, key=lambda x: int(re.search(r'\d+', x.split('_')[1]).group()))
    hue_order = [f'{accession}_WT'] + sorted_mutant_list

    # Plot scatter plot
    subset_df = subset_df.sort_values('target_id',key=np.vectorize(hue_order.index))
    g = sns.JointGrid(data=subset_df, x="Year", y="pchembl_value_Mean", hue="target_id",
                      marginal_ticks=False, palette="turbo")

    g.plot_joint(
        sns.scatterplot)
    # Add marginals
    g.plot_marginals(sns.kdeplot)

    # Make plot prettier
    sns.move_legend(g.ax_joint, "upper left", title='Variants', frameon=False, bbox_to_anchor=(1.25, 1))

    g.ax_joint.set_ylabel('pChEMBL value (Mean)')

    # Save plot
    plt.savefig(os.path.join(output_dir, f'scatterplot_year_pchembl_{accession}_{subset_name}.svg'))

def count_proteins_in_dataset(chembl_version: str, papyrus_version: str, papyrus_flavor: str, chunksize:int,
                              annotation_round:int):
    """
    Count number of unique protein accessions in the dataset.
    :param chembl_version: version of ChEMBL
    :param papyrus_version: version of Papyrus
    :param papyrus_flavor: type of Papyrus data to get
    :param chunksize: chunk size for processing papyrus data
    :param annotation_round: Round of annotation
    :return: number of unique accession codes
    """
    # Read mutant annotated ChEMBL + Papyrus data
    data = merge_chembl_papyrus_mutants(chembl_version, papyrus_version, papyrus_flavor, chunksize, annotation_round)

    return len(data['accession'].unique())

def get_statistics_across_accessions(chembl_version: str, papyrus_version: str, papyrus_flavor: str, chunksize:int,
                                     annotation_round:int, output_dir: str, save: bool = False):
    """
    Compute statistics for all accessions in the dataset.
    :param chembl_version: version of ChEMBL
    :param papyrus_version: version of Papyrus
    :param papyrus_flavor: type of Papyrus data to get
    :param chunksize: chunk size for processing papyrus data
    :param annotation_round: Round of annotation
    :param output_dir: directory to save results to
    :param save: whether to save results to file
    :return: pd.DataFrame with statistics. Each row is one protein (accession)
    """
    output_file = os.path.join(output_dir, f'stats_per_accession_round{annotation_round}.txt')

    if not os.path.exists(output_file):
        # Read mutant annotated ChEMBL + Papyrus data
        data = merge_chembl_papyrus_mutants(chembl_version, papyrus_version, papyrus_flavor, chunksize, annotation_round)

        # Extract Organism and gene names
        data_organism = data.drop_duplicates(subset=['accession'])[['accession', 'Organism', 'HGNC_symbol']]

        # Compute number of WT activity datapoints per accession
        stats_activity_wt = (data[data['target_id'].str.contains('_WT')].groupby(['accession'])['connectivity'].nunique().
                             reset_index().rename(columns={'connectivity': 'n_data_wt'}))

        # Compute total number of activity datapoints per accession
        stats_activity = (data.groupby(['accession'])['connectivity'].count().
                          reset_index().rename(columns={'connectivity': 'n_data'}))
        # Compute number of unique compounds per accession
        stats_compounds = (data.groupby(['accession'])['connectivity'].nunique().
                           reset_index().rename(columns={'connectivity': 'n_compounds'}))
        # Compute number of unique mutants per accession
        stats_variants = (data.groupby(['accession'])['target_id'].nunique().
                          reset_index().rename(columns={'target_id': 'n_variants'}))
        # Merge all stats making sure that accession without WT activity are included
        stats = pd.merge(pd.merge(pd.merge(stats_activity, stats_activity_wt, how='left', on='accession').fillna(0),
                          stats_variants,on='accession'),stats_compounds,on='accession')
        # Calculate percentage of mutant activity datapoints (1-WT/total)*100
        stats['data_mutant_percentage'] = (1 - stats['n_data_wt'] / stats['n_data']) * 100

        # Merge with ChEMBL family classifications (L1-L5)
        chembl_families = group_families(obtain_chembl_family(chembl_version=chembl_version))
        stats_family = pd.merge(stats, chembl_families, on='accession')
        # Merge with organism and gene names
        stats_family_tax = pd.merge(stats_family, data_organism, on='accession')

        # Make sure type is correct
        stats_family_tax['n_data_wt'] = stats_family_tax['n_data_wt'].astype(int)

        # Save results
        if save:
            stats_family_tax.to_csv(output_file, sep='\t',index=False)
    else:
        stats_family_tax = pd.read_csv(output_file, sep='\t')

    return stats_family_tax

def get_statistics_across_variants(chembl_version: str, papyrus_version: str, papyrus_flavor: str, chunksize:int,
                                   annotation_round:int, output_dir: str, save: bool = False):
    """
    Compute statistics for all variants in the dataset.
    :param chembl_version: version of ChEMBL
    :param papyrus_version: version of Papyrus
    :param papyrus_flavor: type of Papyrus data to get
    :param chunksize: chunk size for processing papyrus data
    :param output_dir: directory to save results to
    :param save: whether to save results to file
    :return: pd.DataFrame with statistics. Each row is one variant (target_id)
    """
    # Read mutant annotated ChEMBL + Papyrus data
    data = merge_chembl_papyrus_mutants(chembl_version, papyrus_version, papyrus_flavor, chunksize, annotation_round)

    # Calculate statistics per accession
    stats = get_statistics_across_accessions(chembl_version, papyrus_version, papyrus_flavor, chunksize,
                                             annotation_round, output_dir)

    # Compute number of activity datapoints per variant
    stats_variant_activity = data.groupby(['accession', 'target_id'])['connectivity'].count().reset_index().rename \
        (columns={'connectivity': 'n_data_target_id'})

    # Merge with accession statistics
    stats_variant = pd.merge(stats_variant_activity, stats, on='accession')

    # Calculate percentage of variant activity datapoints (variant/total)*100
    stats_variant['data_variant_coverage'] = (stats_variant['n_data_target_id'] /
                                                        stats_variant['n_data'])
    # Define variant order from most to least populated
    stats_variant['variant_order'] = stats_variant.sort_values(by='n_data_target_id', ascending=False).groupby \
                                         (['accession']).cumcount() + 1
    # Calculate fold difference between the most populated variant and the rest per accession
    # (i.e. most populated variant has X times more data than the second/third/etc most populated variant
    stats_variant['data_variant_fold'] = stats_variant.groupby(['accession'])['n_data_target_id'] \
        .transform(lambda x: x.max() / x)

    # Save results
    if save:
        stats_variant.to_csv(os.path.join(output_dir, f'stats_per_variant_round{annotation_round}.txt'), sep='\t',
                             index=False)

    return stats_variant

def filter_statistics(stats: pd.DataFrame, min_data: int, max_data: int,
                      min_percentage: float, max_percentage: float,
                      min_variants: int, max_variants: int,
                      sort_output_by:str):
    """
    Filter statistics dataframe to include only accessions with a minimum number of mutants and minimum percentage of
    mutants.
    :param stats: dataframe with statistics
    :param min_data: minimum number of bioactivity data points
    :param max_data: maximum number of bioactivity data points
    :param min_percentage: minimum percentage of mutants
    :param max_percentage: maximum percentage of mutants
    :param min_variants: minimum number of variants
    :param max_variants: maximum number of variants
    :return: filtered dataframe
    """
    # Select maximum values if no maximum is provided
    if max_data is None:
        max_data = stats['n_data'].max()
    if max_variants is None:
        max_variants = stats['n_variants'].max()
    if max_percentage is None:
        max_perccentage = stats['data_mutant_percentage'].max()

    # Filter data
    filtered_stats = stats[(stats['n_data'] >= min_data) &
                             (stats['n_data'] <= max_data) &
                             (stats['n_variants'] >= min_variants) &
                            (stats['n_variants'] <= max_variants) &
                           (stats['data_mutant_percentage'] >=  min_percentage)
                             & (stats['data_mutant_percentage'] <= max_perccentage)
                           & (stats['data_mutant_percentage'] != 100.0)]

    # Sort output
    filtered_stats.sort_values(by=sort_output_by, ascending=False, inplace=True)

    # Print accession codes for filtered accessions
    filtered_ids = filtered_stats.drop_duplicates(subset=['accession'])[['accession','HGNC_symbol']]
    filtered_accession_list = filtered_ids['accession'].tolist()
    filtered_gene_list = filtered_ids['HGNC_symbol'].fillna('N/A').tolist()
    print(f'Accession codes for filtered accessions ({len(filtered_accession_list)}):')
    print(', '.join(filtered_accession_list))
    # Print gene names for filtered accessions (if available)
    print('Gene names for filtered accessions:')
    print(', '.join(filtered_gene_list))

    return filtered_stats

def plot_stats_bubble(stats: pd.DataFrame, filter_tag: str, hue_property: str, hue_title: str,
                      label_condition: str, xy_lims: list,
                      output_dir:str, save: bool = False):
    """
    Plot bubble plot for the statistics dataframe. Number of variants (x) vs. number of bioactivity data points (y)
    with size of the bubble representing the percentage of mutants and color representing the ChEMBL family.
    :param stats: dataframe with statistics
    :param filter_tag: tag to identify filtered data on in output file name. Strats wit '_' if not empty
    :param hue_property: property to use for coloring the bubbles
    :param hue_title: title to use for the color legend
    :param label_condition: condition to use for labeling the bubbles.
        Options are 'n_data' and 'data_mutant_percentage'
    :param xy_lims: list with x and y limits (tuples) for the plot
    :param output_dir: directory to save results to
    :param save: whether to save results to file
    :return:
    """
    # Define plot style
    plt.figure(figsize=(7, 7))
    sns.set_style("white")
    sns.set_context("talk")

    # Create plot
    g = sns.scatterplot(data=stats,
                        x='n_variants',
                        y='n_data',
                        size='data_mutant_percentage',
                        hue=hue_property,
                        palette="viridis",
                        alpha=0.6,
                        edgecolors="black",
                        sizes=(10, 500))

    # Modify legend titles and move to the side
    h, l = g.get_legend_handles_labels()
    l_modified = copy.deepcopy(l)
    l_modified[0] = hue_title
    l_size_index = l.index('data_mutant_percentage')
    l_modified[l_size_index] = 'Variant bioactivity %'
    g.legend(h, l_modified, loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)

    # Zoom in
    if xy_lims is not None:
        plt.xlim(xy_lims[0])
        plt.ylim(xy_lims[1])

    # Add grid lines
    plt.grid(linestyle=':')

    # Modify axes labels
    plt.ylabel('Number of bioactivity datapoints')
    plt.xlabel('Number of variants')

    # Add text labels
    if label_condition == 'n_data':
        # Option 1: for accessions with high number of mutants and bioactivity data points
        [plt.text(x=row['n_variants'] + 0.25, y=row['n_data'], s=row['accession'], size=11) for k, row in stats
            .iterrows() if (row['n_data'] > 10000 or row['n_variants'] > 20)]
    elif label_condition == 'data_mutant_percentage':
        # Option 2: for accessions with high percentage of mutants (applied when zooming in, so no labels outside of
        # axis limits)
        if xy_lims is None:
            [plt.text(x=row['n_variants'] + 0.25, y=row['n_data'], s=row['accession'], size=11) for k, row in
             stats.iterrows() if (row['data_mutant_percentage'] > 10) and 500 < row['n_data']]
        else:
            [plt.text(x=row['n_variants'] + 0.25, y=row['n_data'], s=row['accession'], size=11) for k, row in
             stats.iterrows() if (row['data_mutant_percentage'] > 10 and 500 < row['n_data'] < xy_lims[1][1] and
                                  row['n_variants'] < xy_lims[0][1])]
    # Save plot
    if save:
        if xy_lims is None:
            plt.savefig(os.path.join(output_dir, F'accession_distribution_{hue_property}{filter_tag}.svg'))
        else:
            plt.savefig(os.path.join(output_dir, F'accession_distribution_{hue_property}{filter_tag}_zoom.svg'))

    else:
        plt.show()

def plot_stats_histograms(stats: pd.DataFrame, output_dir:str, save: bool = False):
    """
    Plot histograms for the statistics dataframe.
    :param stats: dataframe with statistics
    :param output_dir: directory to save plots to
    :param save: whether to save plots to file
    :return: Six figures with histograms
    """
    # Define plot style
    sns.set_style("white")
    sns.set_context("paper", font_scale=2)

    # Plot 1
    sns.histplot(stats, x='n_data',
             bins=10,log_scale=True,color='#365c8d')
    plt.xlabel('Number of bioactivity datapoints (log)')
    if save:
        plt.savefig(os.path.join(output_dir, 'histogram_n_data_log.svg'))
        plt.clf()
    else:
        plt.show()

    # Plot 2
    sns.histplot(stats, x='data_mutant_percentage',
                 bins=10,log_scale=False,color='#277f8e')
    plt.xlabel('Variant bioactivity %')
    if save:
        plt.savefig(os.path.join(output_dir, 'histogram_data_mutant_percentage.svg'))
        plt.clf()
    else:
        plt.show()

    # Plot 3
    sns.histplot(stats, x='n_variants',
                 bins=10,log_scale=False,color='#4ac16d')
    plt.xlabel('Number of variants')
    if save:
        plt.savefig(os.path.join(output_dir, 'histogram_n_variants.svg'))
        plt.clf()
    else:
        plt.show()

    # Plot 4
    sns.histplot(stats, x='n_data', y='data_mutant_percentage',
                 bins=10, discrete=(False, False), log_scale=(True, False), color='#1fa187',
        cbar=True, cbar_kws=dict(shrink=.75))
    plt.xlabel('Number of bioactivity datapoints (log)')
    plt.ylabel('Variant bioactivity %')
    if save:
        plt.savefig(os.path.join(output_dir, 'histogram_n_data_mutant_percentage.svg'))
        plt.clf()
    else:
        plt.show()

    # plot 5
    sns.histplot(stats, x='n_variants', y='data_mutant_percentage',
                 bins=10, discrete=(False, False), log_scale=(False, False), color='#a0da39',
        cbar=True, cbar_kws=dict(shrink=.75))
    plt.xlabel('Number of variants')
    plt.ylabel('Variant bioactivity %')
    if save:
        plt.savefig(os.path.join(output_dir, 'histogram_n_variants_mutant_percentage.svg'))
        plt.clf()
    else:
        plt.show()

    # Plot 6
    sns.histplot(stats, x='n_data', y='n_variants',
                 bins=10, discrete=(False, False), log_scale=(True, False), color='#46327e',
        cbar=True, cbar_kws=dict(shrink=.75))
    plt.xlabel('Number of bioactivity datapoints (log)')
    plt.ylabel('Number of variants')
    if save:
        plt.savefig(os.path.join(output_dir, 'histogram_n_data_n_variants.svg'))
        plt.clf()
    else:
        plt.show()

def plot_variant_stats_lineplot(stats: pd.DataFrame, filter_tag:str, y_column:str, y_label:str,
                                xy_lims: list, output_dir:str, save: bool = False):
    """
    Plot a lineplot for the statistics dataframe.
    :param stats: dataframe with statistics
    :param filter_tag: tag to tag to identify filtered data on in output file name. Strats wit '_' if not empty
    :param y_column: column to plot on y-axis
    :param y_label: label for y-axis
    :param xy_lims: list with x and y limits for zooming in
    :param output_dir: directory to save plots to
    :param save: whether to save plots to file
    :return: Six figures with histograms
    """
    # Define plot style
    plt.figure(figsize=(10, 7))
    sns.set_style("white")
    sns.set_context("talk")

    # Create a column with accession and percentage for hue
    stats['accession_percentage'] = stats.\
        apply(lambda x: f'{x.accession} ({round(x["data_mutant_percentage"], 2)}%)', axis=1)

    # Pivot data using variable of interest
    stats_plot = stats.pivot(index='variant_order', columns='accession_percentage',
                                                    values=y_column)
    # Plot lineplot
    sns.lineplot(data=stats_plot, palette='rocket', markers=True, dashes=False)

    # Zoom in
    if xy_lims is not None:
        # Add all ticks on x axis
        plt.xticks(np.arange(min(stats_plot.index), max(stats_plot.index) + 1, 1.0))
        plt.xlim(xy_lims[0])
        plt.ylim(xy_lims[1])

    # Add dashed line at bioacivity = 500 datapoints for reference
    plt.axhline(y=500, linestyle=':', color='grey')

    # Change legend title
    plt.legend(title='Accession (variant %)')
    # Change axes titles
    plt.ylabel(y_label)
    plt.xlabel('Variant number')

    # Save plot
    if save:
        if xy_lims is None:
            plt.savefig(os.path.join(output_dir, f'{y_column}_per_variant{filter_tag}.svg'))
        else:
            plt.savefig(os.path.join(output_dir, f'{y_column}_per_variant_zoom{filter_tag}.svg'))
    else:
        plt.show()

def plot_variant_fold_change_stats(stats:pd.DataFrame, filter_tag:str, variant_i:int, x_column:str, x_label:str,
                                   size_column:str,size_label:str, color:str, output_dir:str, save: bool = False):
    """
    Plot a bubbleplot for the variant statistics dataframe.
    :param stats: dataframe with statistics
    :param filter_tag: tag to identify filtered data on in output file name. Strats wit '_' if not empty
    :param variant_i: variant number to plot fold change of most populated variant respect to
    :param x_column: column to plot on x-axis
    :param x_label: label for x-axis
    :param size_column: column to base size of bubbles on
    :param size_label: label for size legend
    :param color: hex color code for bubbles
    :param output_dir: directory to save plots to
    :param save: whether to save plots to file
    :return:
    """
    # Define plot style
    plt.figure(figsize=(4, 4))
    sns.set_style("white")
    sns.set_context("talk")

    # Select to plot data for the variant of interest
    stats = stats[stats['variant_order'] == variant_i]

    # Plot bubbleplot
    sns.scatterplot(data=stats, x=x_column, y='data_variant_fold', size=size_column,
                    color=color, alpha=0.5, sizes=(10, 300))

    # Add axes labels
    ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])
    plt.ylabel(f'Relative amount of data\nin most populated variant\ncompared to {ordinal(variant_i)}')
    plt.xlabel(x_label)

    # Change legend title
    plt.legend(title=size_label)

    # Save plot
    if save:
        plt.savefig(os.path.join(output_dir, f'variant_1_to_{variant_i}_ratio_{size_column}{filter_tag}.svg'))
