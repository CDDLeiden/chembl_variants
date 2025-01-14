# -*- coding: utf-8 -*-


"""Mutant statistics analysis. Part 1"""
"""Computing and analyzing bioacivity distributions for common subsets per accession"""
import json
import math
import statistics
import os

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from matplotlib.markers import MarkerStyle
import re

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

from .preprocessing import merge_chembl_papyrus_mutants
from .mutant_analysis_accession import count_proteins_in_dataset
from .data_path import get_data_path

def compute_pairwise_similarity(data: pd.DataFrame):
    """
    Compute pairwise Tanimoto similarity between unique compounds in a dataset and write to an output file
    :param data: DataFrame with activity data
    :return: dataframe with similarity values for all pairs of compounds
    """
    data_dir = get_data_path()
    out_file = os.path.join(data_dir,'similarity_matrix.csv')
    if not os.path.exists(out_file):
        unique_compounds = data.drop_duplicates('connectivity', keep='first')[['connectivity','SMILES']]

        ids = unique_compounds['connectivity'].tolist()
        c_smiles = unique_compounds['SMILES'].tolist()
        mols = [Chem.MolFromSmiles(x) for x in c_smiles]
        fps = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in mols]

        # the list for the dataframe
        qu, ta, sim = [], [], []

        # compare all fp pairwise without duplicates
        for n in range(len(fps) - 1):  # -1 so the last fp will not be used
            s = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n + 1:])  # +1 compare with the next to the last fp
            # collect the SMILES and values
            for m in range(len(s)):
                qu.append(ids[n])
                ta.append(ids[n + 1:][m])
                sim.append(s[m])
        # build the dataframe and sort it
        d = {'query': qu, 'target': ta, 'Similarity': sim}
        similarity_df = pd.DataFrame(data=d)
        similarity_df = similarity_df.sort_values('Similarity', ascending=False)

        similarity_df.to_csv(out_file, sep='\t', index=False)
    else:
        similarity_df = pd.read_csv(out_file, sep='\t')

    return similarity_df


def get_filename_tag(common: bool, sim: bool, sim_thres: int, threshold: int, variant_coverage: float):
    """
    Get a nametag for output files based on options for computing variant activity distribution
    :param common: Whether to use common subset for variants
    :param sim: Whether to include similar compounds in the definition of the common subset
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold: Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :return: name tag for output files
    """
    if common:
        if threshold is not None:
            if variant_coverage is not None:
                if sim:
                    options_filename_tag = f'Thr{threshold}_Cov{int(variant_coverage * 100)}_Sim{int(sim_thres * 100)}'
                else:
                    options_filename_tag = f'Thr{threshold}_Cov{int(variant_coverage * 100)}'
            else:
                options_filename_tag = f'Thr{threshold}_NoCov'
        else:
            options_filename_tag = f'NoThr'
    else:
        options_filename_tag = f'All'

    return options_filename_tag

def get_variant_similar_subset(data:pd.DataFrame, accession:str, sim_thres:int, threshold:int, variant_coverage: float,
                               save_dataset: bool, output_dir: str):
    """
    Get the common subset of compounds tested on all (or most) variants, including similar compounds in the definition.
    I.e. if a similar compound has been tested on a different variant, it is considered for filtering purposes that the
    original compound was tested on the original variant when keeping compounds tested on > threshold variants.
    :param data: DataFrame with mutation activity data
    :param accession: Uniprot accession code
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold: Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :param save_dataset: Whether to save the (subset) modelling dataset used for computing activity distribution
    :param output_dir: Location for the output files
    :return: Dataframe with bioactivity data for common subset of compounds and variants
            Dictionary with variant coverage of the subset compounds for each variant
    """
    # Customize filename tags based on function options
    options_filename_tag = get_filename_tag(common=True, sim=True, sim_thres=sim_thres, threshold=threshold,
                                            variant_coverage=variant_coverage)

    # Check if output dataset file exists
    dataset_file = os.path.join(output_dir, options_filename_tag,'modelling_datasets',
                                f'modelling_dataset_{accession}_{options_filename_tag}.csv')
    coverage_file = os.path.join(output_dir, options_filename_tag,'modelling_datasets',
                                 f'modelling_dataset_coverage_{accession}_{options_filename_tag}.json')

    if not os.path.exists(dataset_file):
        # Check if output directory exists, else create
        if not os.path.exists(os.path.join(output_dir, options_filename_tag,'modelling_datasets')):
            os.makedirs(os.path.join(output_dir, options_filename_tag,'modelling_datasets'))

        data_accession_agg = data[data['accession'] == accession]

        # Compute or read similarity
        unique_compounds = data_accession_agg.drop_duplicates('connectivity', keep='first')[['connectivity', 'SMILES']]
        ids = unique_compounds['connectivity'].tolist()

        similarity_df = compute_pairwise_similarity(data) # Similarity read for all possible accessions

        # Get similarity groups per compound
        similarity_groups = {}
        for connectivity in ids:
            similar_group_df = similarity_df[(
                        ((similarity_df['query'] == connectivity) | (similarity_df['target'] == connectivity)) & (
                            similarity_df['Similarity'] >sim_thres))]
            similar_group = list(
                set(similar_group_df['query'].unique().tolist() + similar_group_df['target'].unique().tolist()))
            if not similar_group:
                similar_group = [connectivity]
            similarity_groups[connectivity] = similar_group

        # Pivot data and apply filters to get common subset based on similarity
        data_pivot = pd.pivot(data_accession_agg,index='target_id', columns='connectivity', values='pchembl_value_Mean')

        # Create boolean mask defining whether the compound or a similar one has been tested on each target
        def check_sim(x, connectivity,columns):
            if not math.isnan(x[connectivity]):
                mask_val = True
            else:
                similar_val = []
                for similar_connectivity in similarity_groups[connectivity]:
                    if similar_connectivity in columns:
                        if not math.isnan(x[similar_connectivity]):
                            similar_val.append(True)
                if similar_val:
                    mask_val = True
                else:
                    mask_val = False
            return mask_val

        data_pivot_sim = data_pivot.copy(deep=True)
        for compound in [col for col in data_pivot.columns if col != 'connectivity']:
            data_pivot_sim[compound] = data_pivot.apply(check_sim, connectivity=compound, columns=data_pivot.columns, axis=1)

        # Drop compounds not tested in at least threshold number of variants
        common_sim_subset = data_pivot.drop(columns=data_pivot_sim.columns[data_pivot_sim.sum() < threshold])
        common_sim_subset_mask = data_pivot_sim.drop(columns=data_pivot_sim.columns[data_pivot_sim.sum() < threshold])

        # Drop variants not tested for more than variant_coverage*100 % of common subset of compounds
        compound_threshold = variant_coverage * common_sim_subset.shape[1]
        common_sim_subset = common_sim_subset.drop(
            index=common_sim_subset_mask.index[common_sim_subset_mask.sum(axis=1) < compound_threshold])
        common_sim_subset_mask = common_sim_subset_mask.drop(
            index=common_sim_subset_mask.index[common_sim_subset_mask.sum(axis=1) < compound_threshold])

        # Calculate coverage per variant (default and with similar compounds)
        coverage_dict = {}
        for variant in common_sim_subset.index:
            try:
                variant_coverage = len([x for x in common_sim_subset.loc[variant, :].values.tolist() if not math.isnan(x)]) / \
                                   common_sim_subset.shape[1]
                variant_coverage_sim = len([x for x in common_sim_subset_mask.loc[variant, :].values.tolist() if x == True]) / \
                                       common_sim_subset.shape[1]
            except ZeroDivisionError:
                variant_coverage = 0.0
                variant_coverage_sim = 0.0
            coverage_dict[variant] = [variant_coverage,variant_coverage_sim]

        # Melt
        common_sim_subset.reset_index(inplace=True)

        common_subset_melt = pd.melt(common_sim_subset, id_vars=['target_id'],
                                     value_vars=[x for x in common_sim_subset.columns if
                                                 x not in ['connectivity', 'target_id']],
                                     value_name='pchembl_value_Mean')

        # Write dataset and coverage to file
        if save_dataset:
            # First remove NAs
            common_subset_melt = common_subset_melt[common_subset_melt['pchembl_value_Mean'].notna()]
            common_subset_melt.to_csv(dataset_file, sep='\t', index=False)
            with open(coverage_file, 'w') as file:
                json.dump(coverage_dict, file)

    else:
        # Read dataset and coverage dictionary if they exist
        common_subset_melt = pd.read_csv(dataset_file, sep='\t')
        with open(coverage_file, 'r') as file:
            coverage_dict = json.load(file)

    return common_subset_melt,coverage_dict


def get_variant_common_subset(data: pd.DataFrame, accession:str, common:bool, threshold:int, variant_coverage: float,
                              save_dataset: bool, output_dir: str):
    """
    Get the common subset of compounds tested on all (or most) variants
    :param data: DataFrame with mutation activity data
    :param accession: Uniprot accession code
    :param common: Whether to use common subset for variants
    :param threshold: Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :param save_dataset: Whether to save the (subset) modelling dataset used for computing activity distribution
    :param output_dir: Location for the output files
    :return: Dataframe with bioactivity data for common subset of compounds and variants
            Dictionary with variant coverage of the subset compounds for each variant
    """
    # Customize filename tags based on function options
    options_filename_tag = get_filename_tag(common=common, sim=False, sim_thres=None, threshold=threshold,
                                            variant_coverage=variant_coverage)

    # Check if output dataset file exists
    dataset_file = os.path.join(output_dir, options_filename_tag,'modelling_datasets',
                                f'modelling_dataset_{accession}_{options_filename_tag}.csv')
    coverage_file = os.path.join(output_dir, options_filename_tag,'modelling_datasets',
                                 f'modelling_dataset_coverage_{accession}_{options_filename_tag}.json')

    if not os.path.exists(dataset_file):
        data_accession_agg = data[data['accession'] == accession]

        data_pivoted = pd.pivot(data_accession_agg,index='target_id',columns='connectivity',values='pchembl_value_Mean')

        def calculate_coverage(pivoted_data):
            coverage_dict = {}
            for variant in pivoted_data.index:
                try:
                    variant_coverage = len(
                        [x for x in pivoted_data.loc[variant, :].values.tolist() if not math.isnan(x)]) / \
                                       pivoted_data.shape[1]
                except ZeroDivisionError:
                    variant_coverage = 0.0
                coverage_dict[variant] = [variant_coverage]
            return coverage_dict

        if common:
            if threshold is None:
                # Drop compounds not tested in all variants
                common_subset = data_pivoted.dropna(axis='columns', how='any')
            else:
                # Drop compounds not tested in at least threshold number of variants
                common_subset = data_pivoted.dropna(axis='columns', thresh=threshold)

                if variant_coverage is not None:
                    # Drop variants not tested for more than variant_coverage*100 % of common subset of compounds
                    compound_threshold = variant_coverage * common_subset.shape[1]
                    common_subset = common_subset.dropna(axis='index', thresh=compound_threshold)

            # Calculate coverage per variant (default)
            coverage_dict = calculate_coverage(common_subset)

            common_subset.reset_index(inplace=True)

            common_subset_melt = pd.melt(common_subset, id_vars=['target_id'],
                                         value_vars=[x for x in common_subset.columns if x not in ['connectivity','target_id']],
                                         value_name='pchembl_value_Mean')
        else:
            coverage_dict = calculate_coverage(data_pivoted)
            common_subset_melt = data_accession_agg.copy(deep=True)

        if save_dataset:
            # Check if output directory exists, else create
            if not os.path.exists(os.path.join(output_dir, options_filename_tag, 'modelling_datasets')):
                os.makedirs(os.path.join(output_dir, options_filename_tag, 'modelling_datasets'))

            common_subset_melt.to_csv(dataset_file, sep='\t', index=False)
            with open(coverage_file, 'w') as file:
                json.dump(coverage_dict, file)

    else:
        # Read dataset and coverage dictionary if they exist
        common_subset_melt = pd.read_csv(dataset_file, sep='\t')
        with open(coverage_file, 'r') as file:
            coverage_dict = json.load(file)

    return common_subset_melt,coverage_dict

def read_common_subset(accession: str, common: bool, sim: bool, sim_thres: int,
                                           threshold: int, variant_coverage: float, output_dir: str):
    """
    Read the pre-calculated common subset of compounds tested on all (or most) variants, possibly including similar
    compounds in the definition.
    :param accession:Uniprot accession code
    :param common: Whether to use common subset for variants
    :param sim: Whether similar compounds are included in the definition
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold:Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :param output_dir: Location for the pre-calculated files
    :return: pd.DataFrame with pchembl_value, connectivity, and target_id columns for the common subset of interest
    """
    # Customize filename tags based on function options for subdirectories
    options_filename_tag = get_filename_tag(common, sim, sim_thres, threshold,variant_coverage)

    # Read bioactivity data for common subset precalculated
    try:
        data_common = pd.read_csv(os.path.join(output_dir, options_filename_tag,'modelling_datasets',
                                             f'modelling_dataset_{accession}_{options_filename_tag}.csv'), sep='\t')
    except FileNotFoundError:
        raise FileNotFoundError(f'No common subset dataset found for {options_filename_tag}. '
                                f'Please run compute_variant_activity_distribution first with '
                                f'save_dataset=True.')
    return data_common


def calculate_variant_stats(data_accession: pd.DataFrame, accession: str, diff: bool, one_on_one: bool):
    """
    Calculate mean and StD of activity values for each variant of one target (accession)
    :param data_accession: DataFrame with mutation activity data for accession of interest
    :param accession: Uniprot accession code
    :param diff: Whether to calculate the difference in Mean and StD between variants and WT
    :param one_on_one: Whether to calculate the mean and StD for the difference in bioactivity from WT to all accession
                variants in a set. This allows to compare not just the difference between variant means but the mean
                between one-on-one differences in a common subset of compounds
    :return: two dataframes with mean and standard deviation values for each variant
    """
    data_pivoted = pd.pivot(data_accession, index='connectivity', columns='target_id', values='pchembl_value_Mean')

    if diff and one_on_one:
        for target_id in [col for col in data_pivoted.columns if col != 'target_id']:
            try:
                data_pivoted[f'{target_id}_WTdif'] = data_pivoted.apply(lambda x: x[target_id] - x[f'{accession}_WT'], axis=1)
            except KeyError: # WT might not be included in the subset due to lack of data
                data_pivoted[f'{target_id}_WTdif'] = np.NaN

    data_mean = data_pivoted.mean(axis=0).reset_index(name='pchembl_value_Mean_Mean')
    data_std = data_pivoted.std(axis=0).apply(lambda x: x if not math.isnan(x) else 0).reset_index(name='pchembl_value_Mean_Std')

    if diff and not one_on_one:
        try:
            mean_WT = data_mean[data_mean['target_id'] == f'{accession}_WT']['pchembl_value_Mean_Mean'].values[0]
            def calculate_mean_diff(row):
                if row['target_id'] != f'{accession}_WT':
                    mean_diff = row['pchembl_value_Mean_Mean'] - mean_WT
                else:
                    mean_diff = 0
                return mean_diff

            data_mean['pchembl_value_Mean_Mean'] = data_mean.apply(calculate_mean_diff, axis=1)
            data_std['pchembl_value_Mean_Std'] = data_std['pchembl_value_Mean_Std'].apply(lambda x: x if not math.isnan(x) else 0)
        except IndexError:  # WT might not be included in the subset due to lack of data
            data_mean['pchembl_value_Mean_Mean'] = np.NaN
            data_std['pchembl_value_Mean_Std'] = np.NaN

    if diff and one_on_one:
        data_mean = data_mean[data_mean['target_id'].str.contains('WTdif')]
        data_mean['target_id'] = data_mean['target_id'].apply(lambda x: x.replace('_WTdif',''))
        data_std = data_std[data_std['target_id'].str.contains('WTdif')]
        data_std['target_id'] = data_std['target_id'].apply(lambda x: x.replace('_WTdif', ''))

    return data_mean,data_std

def order_variants(variant_list: list):
    """
    Order variants by position in the protein sequence (ascending) for the first, second, third [...] mutations
    respectively. If there are several mutations in the same position, order them by alphabetical order of the amino
    acid substitution (ascending). If WT is in the set, it is always the first in the list.
    :param variant_list: list of variants to be ordered
    :return: list of sorted variants
    """
    # Get the maximum amount of mutations in one variant
    max_variant_length = max([len(x.split('_')) for x in variant_list])

    # Split variants into mutations and separate them by position and substituted amino acid
    variants_dict = {}
    for variant in variant_list:
        mutations = []
        for mutation_n in range(1, max_variant_length):
            try:
                mutations.append(variant.split('_')[mutation_n][1:-1])
                mutations.append(variant.split('_')[mutation_n][-1])
            except IndexError:
                mutations.append('')
                mutations.append('')
        variants_dict[variant] = mutations

    # Make df with two columns for each mutation (position and substituted amino acid) and sort by them starting by
    # the first mutation
    column_names = [str(n) for n in range(1, max_variant_length * 2 - 1)]
    variants_df_sorted = pd.DataFrame.from_dict(variants_dict, orient='index', columns=column_names).sort_values \
        (by=column_names)

    # Extract sorted mutations
    variants_sorted = variants_df_sorted.index.tolist()
    return variants_sorted

def define_consistent_palette(data: pd.DataFrame, accession: str):
    """
    Define palette for plotting that is consistent for the same data and accession
    :param data: DataFrame with mutation activity data
    :param accession: Uniprot accession code
    :return: seaborn color palette
    """
    palette = {f'{accession}_WT': '#808080', f'{accession}_MUTANT':'#333333'} # Initialize with WT and undefined
    # mutant as grey
    list_variants = [target_id for target_id in data[data['accession'] == accession]['target_id'].unique().tolist()
                     if ((not 'WT' in target_id) and (not 'MUTANT' in target_id))]
    if len(list_variants) > 1:
        list_colors = sns.color_palette("Spectral", n_colors=len(list_variants)).as_hex()
    else:
        list_colors =['#f28e13'] # Orange to improve visibility
    try:
        list_variants_sorted = order_variants(list_variants)
    except ValueError:
        list_variants_sorted = list_variants
    list_colors_accession = list_colors[0:len(list_variants_sorted)+1]

    # Populate palette dictionary
    for variant,color in zip(list_variants_sorted,list_colors_accession):
        palette[variant] = color
    return palette


def compute_variant_activity_distribution(data: pd.DataFrame, accession: str, common: bool, sim: bool, sim_thres: int,
                                       threshold: int, variant_coverage: float, plot: bool, hist: bool, plot_mean: bool,
                                       color_palette: dict, save_dataset: bool, output_dir: str,
                                        replot: bool = False):
    """
    Generate (sub)set of bioactivity data for target and options of interest and compute variant activity distribution
    with the option to plot it and report the dataset and statistics
    :param data: DataFrame with mutation activity data
    :param accession: Uniprot accession code
    :param common: Whether to use common subset for variants
    :param sim: Whether to include similar compounds in the definition of the common subset
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold: Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :param plot: Whether to plot the activity distribution.
                This will save the figure file (.png) and append statistics (.txt) in output_dir
    :param hist: Whether to plot activity distribution as histogram instead of smooth kde distribution curve
    :param plot_mean: Whether to plot variant's mean as a vertical line
    :param color_palette: (Optional) Dictionary with colors for plotting. If not provided, it will be generated
                        automatically
    :param save_dataset: Whether to save the (subset) modelling dataset used for computing activity distribution
    :param output_dir: Location for the output files
    :param replot: whether to replot existing bioactivity distribution plotting results
    :return: None
    """
    # Customize filename tags based on function options
    options_filename_tag = get_filename_tag(common, sim, sim_thres, threshold, variant_coverage)

    # Check in output stats or dataset file if this accession was already analyzed. If so, skip analysis.
    stat_file = os.path.join(output_dir, options_filename_tag, f'stats_distribution_{options_filename_tag}.txt')
    dataset_file = os.path.join(output_dir, options_filename_tag,'modelling_datasets',
                                f'modelling_dataset_{accession}_{options_filename_tag}.csv')

    if plot and (os.path.exists(stat_file)) and (accession in pd.read_csv(stat_file, sep='\t')[
        'accession'].unique().tolist()) and not replot:
        print(f'{accession} already plotted and statistics analyzed. Skipping...')
    elif save_dataset and (os.path.exists(dataset_file)) and (accession in pd.read_csv(stat_file, sep='\t')[
        'accession'].unique().tolist()) and not replot:
        print(f'{accession} dataset already saved. Skipping...')

    else:
        # Define color palette and plotting order to ensure consistent colors over different subsets
        if not color_palette == None:
            palette = color_palette
        else:
            palette = define_consistent_palette(data,accession)

        # Calculate common subset for plotting (with or without similarity match amplification) and save if specified
        if not sim:
            data_accession,coverage_dict = get_variant_common_subset(data, accession, common, threshold,
                                                                     variant_coverage, save_dataset, output_dir)
        else:
            data_accession,coverage_dict = get_variant_similar_subset(data, accession, sim_thres, threshold,
                                                                      variant_coverage, save_dataset, output_dir)

        # Plot only if data has enough variance to be plotted in a distribution plot
        if plot:
            if not data_accession.groupby(['target_id'])['pchembl_value_Mean'].var().isnull().all():
                # Plotting options
                sns.set_style('ticks')
                # Plot distribution
                hue_order = [target_id for target_id in palette.keys() if target_id in data_accession['target_id'].unique().tolist()]
                if not hist: # Plot distribution curve
                    g = sns.displot(data_accession, x='pchembl_value_Mean', hue='target_id', kind='kde', fill=True,
                                    palette=palette, hue_order=hue_order, height=3.5, aspect=1.1, warn_singular=False,
                                    facet_kws={'despine':False})
                else: # Plot histograms
                    g = sns.displot(data_accession, x='pchembl_value_Mean', hue='target_id', kde=True,
                                    palette=palette, hue_order=hue_order, height=3.5, aspect=1.1, warn_singular=False,
                                    facet_kws={'despine':False})

                # Calculate subset variant statistics for reporting (and plotting)
                data_mean, data_std = calculate_variant_stats(data_accession, accession, diff=False, one_on_one=False)
                data_mean_error, data_std_error = calculate_variant_stats(data_accession, accession, diff=True, one_on_one=False)
                data_mean_error_strict, data_std_error_strict = calculate_variant_stats(data_accession, accession, diff=True, one_on_one=True)

                # Plot Mean pchembl_value per variant
                if plot_mean:
                    for i, variant in enumerate(data_accession['target_id'].unique().tolist()):
                        variant_mean = data_mean.loc[data_mean['target_id'] == variant, 'pchembl_value_Mean_Mean'].item()
                        variant_std = data_std.loc[data_std['target_id'] == variant, 'pchembl_value_Mean_Std'].item()
                        plt.axvline(variant_mean, color=palette[variant], linestyle=':', alpha=0.8, linewidth=1.0)

                # Add to variant legend: 1) Mean +/- Std pchembl_value 2) number of compounds 3) coverage of compounds in (sub)set
                if not sim:
                    new_legend_labels = [
                        f'{target_id} ({round(data_mean.loc[data_mean["target_id"] == target_id, "pchembl_value_Mean_Mean"].item(), 2)} +/-' \
                        f'{round(data_std.loc[data_std["target_id"] == target_id, "pchembl_value_Mean_Std"].item(), 2)}, ' \
                        f'n={sum(data_accession[data_accession["target_id"] == target_id]["pchembl_value_Mean"].notna().tolist())}, ' \
                        f'{int(coverage_dict[target_id][0]*100)} %)'
                        for target_id in hue_order]
                else:
                    new_legend_labels = [
                        f'{target_id} ({round(data_mean.loc[data_mean["target_id"] == target_id, "pchembl_value_Mean_Mean"].item(), 2)} +/-' \
                        f'{round(data_std.loc[data_std["target_id"] == target_id, "pchembl_value_Mean_Std"].item(), 2)}, ' \
                        f'n={sum(data_accession[data_accession["target_id"] == target_id]["pchembl_value_Mean"].notna().tolist())}, ' \
                        f'{int(coverage_dict[target_id][0] * 100)} / {int(coverage_dict[target_id][1] * 100)} %)'
                        for target_id in hue_order]

                for t, l in zip(g._legend.texts, new_legend_labels):
                    t.set_text(l)

                # Customize axis and legend labels based on function options
                if common:
                    if variant_coverage is not None:
                        if sim:
                            options_legend_tag = (f'Common subset unique (+ similars) n'
                                                  f'={len(data_accession["connectivity"].unique().tolist())}\n(> {variant_coverage * 100} % coverage)')
                        else:
                            options_legend_tag = (f'Common subset unique n'
                                                  f'={len(data_accession["connectivity"].unique().tolist())}\n(> {variant_coverage*100} % coverage)')
                    else:
                        options_legend_tag = (f'Common subset unique n'
                                              f'={len(data_accession["connectivity"].unique().tolist())}\n(Not defined coverage)')
                else:
                    options_legend_tag = f'Full set unique n={len(data_accession["connectivity"].unique().tolist())}'

                g._legend.set_title(f'{accession} variants\n{options_legend_tag}')
                sns.move_legend(g, bbox_to_anchor=(0.75, 0.5), loc='center left')
                plt.xlabel('pChEMBL value (Mean)')
                plt.xlim(2, 12)

                # Write figure
                if not os.path.exists(os.path.join(output_dir, options_filename_tag, 'distribution_plots')):
                    os.makedirs(os.path.join(output_dir, options_filename_tag, 'distribution_plots'))
                plt.savefig(os.path.join(output_dir,options_filename_tag,'distribution_plots',
                                         f'variant_activity_distribution_{accession}_{options_filename_tag}.png'),
                                         bbox_inches='tight', dpi=300)
                plt.savefig(os.path.join(output_dir,options_filename_tag,'distribution_plots',
                                         f'variant_activity_distribution_{accession}_{options_filename_tag}.svg'))

                # Write stats output file
                target_id_list = [target_id for target_id in hue_order]
                accession_list = [accession for x in target_id_list]
                mean_list = [data_mean.loc[data_mean['target_id'] == target_id, 'pchembl_value_Mean_Mean'].item() for target_id in target_id_list]
                std_list = [data_std.loc[data_std['target_id'] == target_id, 'pchembl_value_Mean_Std'].item() for target_id in target_id_list]
                mean_error_list = [data_mean_error.loc[data_mean_error['target_id'] == target_id, 'pchembl_value_Mean_Mean'].item() for target_id in target_id_list]
                mean_error_strict_list = [data_mean_error_strict.loc[data_mean_error_strict['target_id'] == target_id, 'pchembl_value_Mean_Mean'].item() for target_id in target_id_list]
                std_error_strict_list = [data_std_error_strict.loc[data_std_error_strict['target_id'] == target_id, 'pchembl_value_Mean_Std'].item() for target_id in target_id_list]
                n_accession_list = [len(data_accession['connectivity'].unique().tolist()) for x in target_id_list]
                n_target_id_list = [sum(data_accession[data_accession["target_id"] == target_id]["pchembl_value_Mean"].notna().tolist()) for target_id in target_id_list]
                coverage_list = [coverage_dict[target_id][0] for target_id in target_id_list]
                stat_dict = {'accession':accession_list,
                             'target_id':target_id_list,
                             'mean_pchembl':mean_list,
                             'std_pchembl':std_list,
                             'mean_error':mean_error_list,
                             'mean_error_strict':mean_error_strict_list,
                             'std_error_strict':std_error_strict_list,
                             'n_compounds_accession':n_accession_list,
                             'n_compounds_target_id':n_target_id_list,
                             'compound_variant_coverage':coverage_list}

                stat_df = pd.DataFrame(stat_dict)

                if not os.path.exists(stat_file):
                    stat_df.to_csv(stat_file, sep='\t', index=False)
                else:
                    if not accession in pd.read_csv(stat_file, sep='\t')['accession'].unique().tolist():
                        stat_df.to_csv(stat_file, mode='a', sep='\t', index=False, header=False)
                print(f'{accession} done.')

            # Write output file with skipped accession codes (not enough data for plotting)
            else:
                print(f'Skipping accession {accession}: not enough data for plotting.')
                if not os.path.exists(os.path.join(output_dir, options_filename_tag,
                                                   f'skipped_accession_{options_filename_tag}.txt')):
                    if not os.path.exists(os.path.join(output_dir, options_filename_tag)):
                        os.makedirs(os.path.join(output_dir, options_filename_tag))
                    with open(os.path.join(output_dir, options_filename_tag,
                                           f'skipped_accession_{options_filename_tag}.txt'),'w') as file:
                        file.write(f'Skipping accession {accession}\n')
                else:
                    with open(os.path.join(output_dir, options_filename_tag,
                                           f'skipped_accession_{options_filename_tag}.txt'),'r+') as file:
                            for line in file:
                                if accession in line:
                                    break
                            else:
                                file.write(f'Skipping accession {accession}\n')

def read_bioactivity_distribution_stats_file(file_dir: str, common: bool, sim: bool, sim_thres: int, threshold: int,
                               variant_coverage: float):
    """
    Read the bioactivity distribution stats file produced while plotting
    :param file_dir: Location of the stats file
    :param common: Whether to use common subset for variants
    :param sim: Whether to include similar compounds in the definition of the common subset
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold: Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :return: pd.DataFrame with the stats
    """
    filename_tag = get_filename_tag(common, sim, sim_thres, threshold, variant_coverage)
    if not os.path.exists(os.path.join(file_dir, filename_tag, f'stats_distribution_{filename_tag}.txt')):
        raise FileNotFoundError(f'No stats file found for {filename_tag}. '
                                f'Please run main.py first.')
    else:
        stat_df = pd.read_csv(os.path.join(file_dir, filename_tag, f'stats_distribution_{filename_tag}.txt'), sep='\t')
        stat_df.drop_duplicates(['accession', 'target_id'], inplace=True)

        return stat_df

def check_common_subset_status(chembl_version: str, papyrus_version: str, papyrus_flavor: str,
                               annotation_round: int, file_dir: str, common: bool,
                               sim: bool, sim_thres: int, threshold: int, variant_coverage: float):
    """
    Check whether the common subset has been calculated for the given parameters for all proteins
    :param chembl_version: ChEMBL version
    :param papyrus_version: Papyrus version
    :param papyrus_flavor: Papyrus flavor
    :param annotation_round: Annotation round
    :param file_dir: Directory where the common subset stats file is located
    :param common: Whether to use common subset for variants
    :param sim: Whether to include similar compounds in the definition of the common subset
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold: Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :return: True if the common subset has been calculated, False otherwise
    """
    print('Checking common subset analysis status...')
    # Get the total number of proteins in the dataset
    n_proteins = count_proteins_in_dataset(chembl_version, papyrus_version, papyrus_flavor, 1_000_000,
                                           annotation_round)

    # Check how many proteins have been successfully processed and recorded on the stats file
    common_stat = read_bioactivity_distribution_stats_file(file_dir, common, sim, sim_thres, threshold,
                                                           variant_coverage)
    n_proteins_processed = len(common_stat['accession'].unique().tolist())
    print(f'{n_proteins_processed} out of {n_proteins} proteins have been successfully processed.')

    # Check how many proteins have been skipped due to lack of data
    options_filename_tag = get_filename_tag(common, sim, sim_thres, threshold, variant_coverage)
    proteins_skipped = []
    with open(os.path.join(file_dir, options_filename_tag,
                      f'skipped_accession_{options_filename_tag}.txt'), 'r+') as file:
        for line in file:
            if 'Skipping accession' in line:
                protein_skipped = line.split(' ')[2].strip()
                if protein_skipped not in proteins_skipped:
                    proteins_skipped.append(protein_skipped)
    n_proteins_skipped = len(proteins_skipped)
    print(f'{n_proteins_skipped} out of {n_proteins} proteins have been skipped due to lack of data.')

    # Check if all proteins in the set have been covered (processed or skipped)
    n_proteins_missing = n_proteins - (n_proteins_processed + n_proteins_skipped)
    if n_proteins_missing == 0:
        print('Common subset analysis is complete.')
        return True
    else:
        print(f'Common subset analysis is not complete ({n_proteins_missing} proteins left to process).')
        print('Please, run main.py to compute the common subset analysis for all proteins.')
        return False

def calculate_accession_common_dataset_stats(accession: str, common: bool, sim: bool, sim_thres: int, threshold: int,
                                             variant_coverage: float, output_dir: str):
    """
    Calculate statistics of the common subset for modelling for each accession. Note that this function calculates
    statistics for the modelling dataset and focuses on the number of datapoints, while other statistics calculated
    for the bioactivity distribution focus report the number of unique compounds.
    :param accession: Uniprot accession code
    :param common: Whether to use common subset for variants
    :param sim: Whether to include similar compounds in the definition of the common subset
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold: Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :param output_dir: Location for the modelling dataset files
    """
    # Read modelling common dataset
    common_subset = read_common_subset(accession, common, sim, sim_thres, threshold, variant_coverage, output_dir)

    # Initialize dictionary for statistics
    subset_stats = {}

    # Compute statistics
    variant_n = len(common_subset['target_id'].unique())
    dataset_size = len(common_subset[common_subset['pchembl_value_Mean'].notna()])
    wt_dataset_size = len(common_subset[(common_subset['pchembl_value_Mean'].notna()) &
                                        (common_subset['target_id'] == f'{accession}_WT')])
    unique_compounds = common_subset['connectivity'].nunique()
    full_matrix_size = variant_n * unique_compounds
    variant_coverage = {}
    for variant in common_subset['target_id'].unique():
        variant_dataset_size = len(common_subset[(common_subset['pchembl_value_Mean'].notna()) &
                                                    (common_subset['target_id'] == variant)])
        variant_coverage[variant] = variant_dataset_size / unique_compounds
    if 'MUTANT' in [var.split('_')[1] for var in common_subset['target_id'].unique().tolist()]:
        subset_stats['undefined_mutants'] = True
    else:
        subset_stats['undefined_mutants'] = False

    subset_stats['n_variants'] = variant_n
    subset_stats['n_data'] = dataset_size
    subset_stats['full_matrix_size'] = full_matrix_size
    subset_stats['n_compounds'] = unique_compounds
    subset_stats['data_mutant_ratio'] = 1 - (wt_dataset_size / dataset_size)
    subset_stats['data_variant_coverage'] = variant_coverage
    subset_stats['sparsity'] = 1 - (dataset_size / full_matrix_size)
    subset_stats['balance_score'] = statistics.mean(list(variant_coverage.values()))

    return subset_stats

def calculate_accession_common_dataset_stats_all(common: bool, sim: bool, sim_thres: int, threshold: int,
                                                variant_coverage: float, output_dir: str):
    """
    Calculate statistics of the common subset for modelling for all accessions
    :param common: Whether to use common subset for variants
    :param sim: Whether to include similar compounds in the definition of the common subset
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold: Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :param output_dir: Location for the modelling dataset files
    """
    filename_tag = get_filename_tag(common, sim, sim_thres, threshold, variant_coverage)
    output_fle = os.path.join(output_dir, filename_tag, f'stats_modelling_{filename_tag}.txt')

    if not os.path.exists(output_fle):
        # Read stats file and extract accession codes
        accession_list = read_bioactivity_distribution_stats_file(output_dir, common, sim, sim_thres, threshold,
                                                                  variant_coverage)[
            'accession'].unique().tolist()

        # Initialize dictionary for statistics
        subset_stats = {}

        # Compute statistics for each accession
        for accession in accession_list:
            subset_stats[accession] = calculate_accession_common_dataset_stats(accession, common, sim, sim_thres,
                                                                               threshold, variant_coverage, output_dir)
        # Make a dataframe
        stats_df = pd.DataFrame(subset_stats).T
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={'index': 'accession'}, inplace=True)

        # Write output file
        stats_df.to_csv(output_fle, sep='\t', index=False)

    else:
        stats_df = pd.read_csv(output_fle, sep='\t')

    return stats_df

def enrich_accession_common_dataset_stats(common_dataset_stats_df: pd.DataFrame, general_accession_stats_df:
pd.DataFrame):
    """
    Merge statistics of the common dataset with general statistics per variant and calculate statistics per accession 
    :param common_dataset_stats_df: dataframe with statistics per variant of the common subset
    :param general_accession_stats_df: dataframe with general statistics per variant
    :return: enriched dataframe with statistics per accession of the modelling dataset
    """
    # Keep from general accession stats only the columns of interest
    general_accession_stats_df = general_accession_stats_df[['accession', 'n_variants', 'l1', 'l2', 'l3', 'l4',
                                                             'Organism','HGNC_symbol', 'n_data',
                                                             'data_mutant_percentage']]
    general_accession_stats_df.rename(columns={'n_variants':'n_variants_full',
                                               'n_data': 'n_data_full',
                                               'data_mutant_percentage': 'data_mutant_percentage_full'},
                                      inplace=True)

    # Enrich common subset stats with variant stats from the full set
    stats_enriched = pd.merge(common_dataset_stats_df, general_accession_stats_df,
                              how='left',on=['accession'])

    # Calculate the ratio of datapoints in the common subset compared to the full set
    stats_enriched['common_subset_data_percentage'] = stats_enriched['n_data'] / stats_enriched['n_data_full'] *100

    # Calculate the ratio of variants in the common subset compared to the full set
    stats_enriched['common_subset_variant_percentage'] = stats_enriched['n_variants'] / stats_enriched[
        'n_variants_full'] *100

    # Make sure the type of columns in the dataframe is correct
    stats_enriched['n_variants'] = stats_enriched['n_variants'].astype(int)
    stats_enriched['n_data'] = stats_enriched['n_data'].astype(int)
    stats_enriched['full_matrix_size'] = stats_enriched['full_matrix_size'].astype(int)
    stats_enriched['n_compounds'] = stats_enriched['n_compounds'].astype(int)
    stats_enriched['n_variants_full'] = stats_enriched['n_variants_full'].astype(int)
    stats_enriched['n_data_full'] = stats_enriched['n_data_full'].astype(int)
    stats_enriched['data_mutant_ratio'] = stats_enriched['data_mutant_ratio'].astype(float)
    stats_enriched['data_mutant_percentage_full'] = stats_enriched['data_mutant_percentage_full'].astype(float)
    stats_enriched['common_subset_data_percentage'] = stats_enriched['common_subset_data_percentage'].astype(float)
    stats_enriched['common_subset_variant_percentage'] = stats_enriched['common_subset_variant_percentage'].astype(float)
    stats_enriched['data_variant_coverage'] = stats_enriched['data_variant_coverage'].astype(str)
    stats_enriched['sparsity'] = stats_enriched['sparsity'].astype(float)
    stats_enriched['balance_score'] = stats_enriched['balance_score'].astype(float)

    return stats_enriched

def extract_relevant_targets(file_dir: str, common: bool, sim: bool, sim_thres: int, threshold: int, variant_coverage: float,
                             min_subset_n: int = 50, thres_error_mean: float = 0.5, error_mean_limit: str = 'min'):
    """
    Explore the file with bioactivity distribution statistics and extract the most interesting targets from it
    :param file_dir: Location of the stats file
    :param common: Whether to use common subset for variants
    :param sim: Whether to include similar compounds in the definition of the common subset
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold: Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :param min_subset_n: Minimum number of unique compounds in the target (sub)set
    :param thres_error_mean: Threshold to satisfy error condition
    :param error_mean_limit: Error conditions: 'min': select targets with at least 1 variant with a mean pchembl value
                                                        difference to WT bigger than the minimum threshold
                                               'max': select targets with at least 1 variant with a mean pchembl value
                                                        difference to WT smaller than the maximum threshold
                                               'var_min': select targets with a standard deviation between its variants
                                                        difference to WT bigger than the minimum threshold
                                               'var_max': select targets with a standard deviation between its variants
                                                        difference to WT smaller than the maximum threshold

    :return: pd.DataFrame statistics for the accession codes that satisfy the input conditions
    """
    filename_tag = get_filename_tag(common, sim, sim_thres, threshold, variant_coverage)
    stat_df =  pd.read_csv(os.path.join(file_dir, filename_tag,
                                        f'stats_distribution_{filename_tag}.txt'),
                           sep='\t')
    stat_df.drop_duplicates(['accession','target_id'], inplace=True)

    # Extract accession codes with  more unique compounds than min_subset
    accession_max_subset = stat_df[stat_df['n_compounds_accession'] > min_subset_n]['accession'].unique().tolist()

    # Extract accession codes that satisfy the defined error to WT conditions (A-D):
    mean_error_list = stat_df.groupby(['accession'])['mean_error'].apply(list).reset_index()
    mean_error_var = stat_df.groupby(['accession'])['mean_error'].std().reset_index()
    # A) at least one variant has a mean pchembl value difference to WT bigger than thres_error_mean
    if error_mean_limit == 'min':
        accession_limit_error = \
        mean_error_list[mean_error_list['mean_error'].apply(lambda x: any(np.abs(n) > thres_error_mean for n in x))][
            'accession'].tolist()
    # B) at least one variant has a mean pchembl value difference to WT smaller than thres_error_mean
    elif error_mean_limit == 'max':
        accession_limit_error = \
        mean_error_list[mean_error_list['mean_error'].apply(lambda x: any((np.abs(n) < thres_error_mean and n != 0)
                                                                          for n in x))]['accession'].tolist()
    # C) the variability (StD) between variants' mean pchembl value difference to WT is bigger than thres_error_mean
    elif error_mean_limit == 'var_min':
        accession_limit_error = mean_error_var[mean_error_var['mean_error'] > thres_error_mean]['accession'].tolist()
    # D) the variability (StD) between variants' mean pchembl value difference to WT is smaller than thres_error_mean
    elif error_mean_limit == 'var_max':
        accession_limit_error = mean_error_var[mean_error_var['mean_error'] < thres_error_mean]['accession'].tolist()

    # Keep accession codes that satisfy both conditions
    accession_keep = list(set(accession_max_subset).intersection(accession_limit_error))
    stat_df_keep = stat_df[stat_df['accession'].isin(accession_keep)]

    print(f'{len(accession_keep)} targets satisfy the specified conditions:')
    if error_mean_limit == 'min':
        print(stat_df_keep.groupby(['accession'])[['n_compounds_accession', 'mean_error']].apply(lambda x: x.abs(

        ).max()))
    elif error_mean_limit == 'max':
        print(stat_df_keep[stat_df_keep['mean_error'] != 0].groupby(['accession'])
              ['n_compounds_accession', 'mean_error'].apply(lambda x: x.abs().min()))
    else:
        print(stat_df_keep.groupby(['accession'])
              ['n_compounds_accession','mean_error'].agg({'n_compounds_accession': 'max',
                                                          'mean_error': 'std'}))

    return stat_df_keep

def read_multiple_bioactivity_distribution_stats(stats_dir: str, subset_type: str, accession: str):
    """Read all available stats files for the bioactivity distribution of a given target

    :param stats_dir: Directory where the stats files are located
    :param subset_type: Type of subset used to generate the different stats files. Options: 'butina_clusters',
                        'common_subsets'
    :param accession: Accession code of the target
    :return: pd.DataFrame with the stats for the bioactivity distribution of the target across the different subsets
    """
    # Read stats for all possible subtypes
    df_list = []
    if subset_type == 'butina_clusters_dual':
        dirs = [dir for dir in glob.glob(f'{stats_dir}/*/full_dual_tested_set/*/*/stats_distribution*.txt')]
        for dir in dirs:
            df = pd.read_csv(dir, sep='\t')
            subset_tag = dir.split(sep='\\')[-3]
            df['subset_tag'] = subset_tag
            df_list.append(df)
    elif subset_type == 'butina_clusters_full':
        dirs = [dir for dir in glob.glob(f'{stats_dir}/*/full_set/*/*/stats_distribution*.txt')]
        for dir in dirs:
            df = pd.read_csv(dir, sep='\t')
            subset_tag = dir.split(sep='\\')[-3]
            df['subset_tag'] = subset_tag
            df_list.append(df)
    elif subset_type == 'common_subsets':
        dirs = [dir for dir in glob.glob(f'{stats_dir}/*/stats_distribution*.txt')]
        for dir in dirs:
            df = pd.read_csv(dir, sep='\t')
            subset_tag = '_'.join(os.path.basename(dir).replace('.txt','').split('_')[2:])
            df['subset_tag'] = subset_tag
            df_list.append(df)

    df_full = pd.concat(df_list)

    # Filter for accession of interest
    df_accession_with_gaps = df_full[df_full['accession'] == accession]

    # Complete dataframe for all subtype and variants available for accession of interest
    mind = pd.MultiIndex.from_product(
        [df_accession_with_gaps.target_id.unique().tolist(),df_accession_with_gaps.subset_tag.unique().tolist(),df_accession_with_gaps.accession.unique().tolist()], names=['target_id','subset_tag','accession'])
    # Fill gaps with zeroes
    df_accession = df_accession_with_gaps.set_index(['target_id','subset_tag','accession']).reindex(mind, fill_value=0)\
        .reset_index()

    return df_accession

def scale_bioactivity_distibution_stats(df: pd.DataFrame):
    """Scale the bioactivity distribution stats to be used in a bubble plot

    :param df: pd.DataFrame with the stats for the bioactivity distribution of the target across the different subsets
    :return: pd.DataFrame with the scaled stats for the bioactivity distribution of the target across the different subsets
    """
    # Differentiate between positive (>WT) and negative (<WT) errors
    # Use square root to amplify small differences for bubble plot
    def scale_error(x, error_key):
        if x[error_key] > 0:
            sqrt_pos = np.sqrt(x[error_key])
            sqrt_neg = 0
        else:
            sqrt_pos = 0
            sqrt_neg = np.sqrt(np.abs(x[error_key]))
        # Multiply by number to get a bubble size that is visible
        scale = 500
        return sqrt_pos*scale, sqrt_neg*scale

    df[['mean_error_pos_upscaled','mean_error_neg_upscaled']] = df.apply(scale_error, args=('mean_error',), axis=1,
                                                                                     result_type='expand')
    try:
        df[['mean_error_strict_pos_upscaled','mean_error_strict_neg_upscaled']] = df.apply(scale_error, args=
        ('mean_error_strict',),axis=1,result_type='expand')
    except KeyError:
        print('mean error strict not calculated')

    # Multiply coverage to get a bubble size that is visible
    df['coverage_upscaled'] = df['compound_variant_coverage']*1000

    return df

def prepare_for_plotting_bioactivity_distribution_stats(df: pd.DataFrame, subset_type: str, accession: str):
    """Prepare the stats for the bioactivity distribution of the target across the different subsets for plotting

    :param df: pd.DataFrame with the stats for the bioactivity distribution of the target across the different subsets
    :param subset_type: Type of subset used to generate the different stats files. Options: 'butina_clusters',
                        'common_subsets'
    :param accession: Accession code of the target
    :return: pd.DataFrame with the stats for the bioactivity distribution of the target across the different subsets
    """
    # Define a consistent palette for all variants
    palette_dict = define_consistent_palette(df, accession)
    # Map variants to their color and order according to target_id
    df['color'] = df['target_id'].map(palette_dict)
    df['target_id_cat'] = pd.Categorical(
        df['target_id'],
        categories=palette_dict.keys(),
        ordered=True
    )

    # Order subset types consistently for plotting
    if 'butina_clusters' in subset_type:
        # Order clusters 1 > 10 based on the number of cluster
        def atoi(text):
           return int(text) if text.isdigit() else text
        def natural_keys(text):
            return [ atoi(c) for c in re.split(r'(\d+)',text) ]
        subset_order =df['subset_tag'].unique().tolist()
        subset_order.sort(key=natural_keys)
    elif subset_type == 'common_subsets':
        subset_order = ['All', 'Thr2_NoCov', 'Thr2_Cov1', 'Thr2_Cov20', 'Thr2_Cov20_Sim80'] # Otherwise 'NoCov goes
        # afterwards'

    df['subset_tag_cat'] = pd.Categorical(
        df['subset_tag'],
        categories = subset_order,
        ordered=True,
    )

    # Make X axis tag pretty
    df['subset_tag_pretty'] = df['subset_tag'].apply(lambda x: x.replace('_',' ').capitalize())

    # Sort values for consistent plotting
    df.sort_values(['target_id_cat','subset_tag_cat'], inplace=True, ascending=[False,True])

    return df, subset_order

def plot_bubble_bioactivity_distribution_stats(stats_dir: str, subset_type: str, accession: str, error_key: str,
                                               output_dir: str, manuscript_quality: bool = False):
    """Plot the stats for the bioactivity distribution of the target across the different subsets

    :param stats_dir: Directory with the stats for the bioactivity distribution of the target across the different subsets
    :param subset_type: Type of subset used to generate the different stats files. Options: 'butina_clusters_dual',
                        'butina_clusters_full', 'common_subsets'
    :param accession: Accession of the target
    :param error_key: Key of the error to plot. Options: 'mean_error', 'mean_error_strict'
    :param output_dir: Directory to save the plot
    """
    # Read stats
    df_accession = read_multiple_bioactivity_distribution_stats(stats_dir, subset_type, accession)
    # Scale stats for bubble plotting
    df_accession = scale_bioactivity_distibution_stats(df_accession)
    # Prepare dataframe for bubble plotting
    df_accession, subset_order = prepare_for_plotting_bioactivity_distribution_stats(df_accession, subset_type, accession)

    # Figure options
    if 'butina_clusters' in subset_type:
        if not manuscript_quality:
            fig, ax = plt.subplots(figsize=(17, 7))
        else:
            fig, ax = plt.subplots(figsize=(11, 5))
    elif subset_type == 'common_subsets':
        fig, ax = plt.subplots(figsize=(7, 10))
    plt.grid(ls="--")
    ax.set_xlim(-0.5, df_accession.subset_tag.unique().size+1.0)

    # Main bubble plot
    # Left bubbles: Cummulative error compared to WT (positive and negative separately)
    plt.scatter(data=df_accession, x='subset_tag_pretty', y='target_id', s=f'{error_key}_pos_upscaled', c='color',
                edgecolor="black", marker=MarkerStyle("o", fillstyle="left"))
    plt.scatter(data=df_accession, x='subset_tag_pretty', y='target_id', s=f'{error_key}_neg_upscaled', c='color',
                edgecolor="white", marker=MarkerStyle("o", fillstyle="left"), alpha=0.5)
    # Right bubbles: Variant coverage
    plt.scatter(data=df_accession, x='subset_tag_pretty', y='target_id', s='coverage_upscaled', c='color',
                edgecolor="black", marker=MarkerStyle("o", fillstyle="right"))

    # Add WT mean_pchembl_value annotations
    for i,sub in enumerate(subset_order):
        pchembl_value_WT = round(df_accession[(df_accession['subset_tag'] == sub) & (df_accession['target_id'].str.contains
        ('WT'))].iloc[0]['mean_pchembl'],2)
        ax.annotate(pchembl_value_WT, xy=(i-0.1,df_accession.target_id.unique().size-1), ha='right', color='#474747')

    if not manuscript_quality:
        # Color variant labels
        # for ticklabel,tickcolor in zip(plt.gca().get_yticklabels(),list(palette_dict.values())[::-1]): #TODO: Fix,
        #  currently expects all variants (which is no always the case for variants). Ideally we would map the ticklabel with
        #  te dict
        for ticklabel,tickcolor in zip(plt.gca().get_yticklabels(),df_accession.color.unique().tolist()):
            ticklabel.set_color(tickcolor) # Color each label
            # ticklabel.set_backgroundcolor(tickcolor) # Color each label background
    else:
        # Remove accession from variant labels and make them black
        plt.draw()
        ls = [label.get_text() for label in ax.get_yticklabels()]
        ax.set_yticklabels([' '.join(label.split('_')[1:]) for label in ls])

    # Add legend
    if not manuscript_quality:
        legend_x_pos = df_accession.subset_tag.unique().size+0.2
    else:
        legend_x_pos = df_accession.subset_tag.unique().size+0.3
    if 'butina_clusters' in subset_type:
        if not manuscript_quality:
            legend_y_pos = df_accession.target_id.unique().size-0.5
            legend_y_pos_step = 0.1*legend_y_pos
            legend_1_x_pos_steps = [0.5,0.0]
        else:
            legend_y_pos = df_accession.target_id.unique().size - 0.2
            legend_y_pos_step = 0.1 * legend_y_pos
            legend_1_x_pos_steps = [0.6, 0.0]
    elif subset_type == 'common_subsets':
        legend_y_pos = df_accession.target_id.unique().size
        legend_y_pos_step = 0.07*legend_y_pos
        legend_1_x_pos_steps = [0.5,0.2]

    # Legend 1: type of error respect to WT (positive or negative)
    # Sublegend title
    legend_1_title = 'Difference in mean\npchembl_value (vs. WT)'
    legend_1_left_title = 'Variant\n> WT'
    legend_1_right_title = 'Variant\n< WT'
    legend_2_title = 'Absolute\ndifference'

    ax.annotate(legend_1_title, xy=(legend_x_pos-0.2, legend_y_pos-legend_y_pos_step*1.7),
                    ha="center", va="center",
                    fontsize="small", fontweight="bold", color="black")
    # Left legend bubble and annotation
    ax.scatter(legend_x_pos-legend_1_x_pos_steps[0],
               legend_y_pos-legend_y_pos_step*3,
               s=1000,
               c='grey',
                edgecolor="black", marker=MarkerStyle("o", fillstyle="left"))
    ax.annotate(legend_1_left_title, xy=(legend_x_pos-legend_1_x_pos_steps[0], legend_y_pos-legend_y_pos_step*2.3),
                    ha="center", va="center",
                    fontsize="small", fontweight="regular", color="black")
    # Right legend bubble and annotation
    ax.scatter(legend_x_pos+legend_1_x_pos_steps[1],
               legend_y_pos-legend_y_pos_step*3,
               s=1000,
               c='grey',
                edgecolor="white", alpha=0.5, marker=MarkerStyle("o", fillstyle="left"))
    ax.annotate(legend_1_right_title, xy=(legend_x_pos+legend_1_x_pos_steps[1], legend_y_pos-legend_y_pos_step*2.3),
                    ha="center", va="center",
                    fontsize="small", fontweight="regular", color="black")

    # Legend 2: size of bubbles
    bubbles_n=10
    # Legend 2.1: size of left bubbles (error wrt WT)
    # Min, max, and step size
    bubbles_error_min = df_accession[error_key].abs().min()
    bubbles_error_min_scaled = min(df_accession[f'{error_key}_pos_upscaled'].abs().min(),
                                   df_accession[f'{error_key}_neg_upscaled'].abs().min())
    bubbles_error_max = df_accession[error_key].abs().max()
    bubbles_error_max_scaled = max(df_accession[f'{error_key}_pos_upscaled'].abs().max(),
                                   df_accession[f'{error_key}_neg_upscaled'].abs().max())
    bubbles_error_step = (bubbles_error_max-bubbles_error_min)/(bubbles_n-1)
    bubbles_error_step_scaled = (bubbles_error_max_scaled-bubbles_error_min_scaled)//(bubbles_n-1)

    # Sublegend title
    ax.annotate(legend_2_title, xy=(legend_x_pos-0.4, legend_y_pos-legend_y_pos_step*4.2),
                    ha="right", va="center",
                    fontsize="small", fontweight="bold", color="black")
    # Legend bubbles
    for i, bubbles_y in enumerate(np.linspace(legend_y_pos-legend_y_pos_step*bubbles_n, legend_y_pos-legend_y_pos_step*5, bubbles_n)):
        #plot each legend bubble to indicate different marker sizes
        ax.scatter(legend_x_pos-0.4,
                   bubbles_y,
                   s=(bubbles_error_min_scaled + i*bubbles_error_step_scaled),
                   c='grey', alpha=0.5, edgecolor="grey", marker=MarkerStyle("o", fillstyle="left"))
        #and label it with a value
        if i > 0:
            ax.annotate(round((bubbles_error_min+i*bubbles_error_step),2), xy=(legend_x_pos-0.6, bubbles_y),
                        ha="right", va="center",
                        fontsize="small", fontweight="regular", color="black")


    # Legend 2.2: size of right bubbles (variant coverage %)
    # Min, max, and step size
    bubbles_coverage_min = df_accession.coverage_upscaled.min()
    bubbles_coverage_max = df_accession.coverage_upscaled.max()
    bubbles_coverage_step = (bubbles_coverage_max - bubbles_coverage_min)//(bubbles_n-1)

    # Sublegend title
    ax.annotate('Variant\ncoverage', xy=(legend_x_pos-0.15, legend_y_pos-legend_y_pos_step*4.2),
                    ha="left", va="center",
                    fontsize="small", fontweight="bold", color="black")
    # Legend bubbles
    for i, bubbles_y in enumerate(np.linspace(legend_y_pos-legend_y_pos_step*bubbles_n, legend_y_pos-legend_y_pos_step*5, bubbles_n)):
        #plot each legend bubble to indicate different marker sizes
        ax.scatter(legend_x_pos-0.15,
                   bubbles_y,
                   s=(bubbles_coverage_min + i*bubbles_coverage_step),
                   c='grey', alpha=0.5, edgecolor="grey", marker=MarkerStyle("o", fillstyle="right"))
        #and label it with a value
        if i > 0:
            ax.annotate(('{:.0%}'.format((bubbles_coverage_min+i*bubbles_coverage_step)/1000)), xy=(legend_x_pos+0.1,
                                                                                                    bubbles_y),
                        ha="left", va="center",
                        fontsize="small", fontweight="regular", color="black")

    # Save plot
    if not manuscript_quality:
        plt.savefig(os.path.join(output_dir, f'bubbleplot_bioactivity_distribution_stats_{accession}_{subset_type}_'
                                           f'{error_key}.png'),dpi=300)
        plt.savefig(os.path.join(output_dir, f'bubbleplot_bioactivity_distribution_stats_{accession}_{subset_type}_'
                                           f'{error_key}.svg'))
    else:
        plt.savefig(os.path.join(output_dir, f'bubbleplot_bioactivity_distribution_stats_{accession}_{subset_type}_'
                                             f'{error_key}_MS.svg'))
