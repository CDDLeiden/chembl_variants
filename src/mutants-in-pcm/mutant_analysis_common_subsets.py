# -*- coding: utf-8 -*-


"""Mutant statistics analysis. Part 1"""
"""Computing and analyzing bioacivity distributions for common subsets per accession"""
import json
import math
import os

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from preprocessing import merge_chembl_papyrus_mutants

def compute_stats_per_accession(data: pd.DataFrame):
    """
    Compute statistics on mutant and bioactivity data counts for each target (accession)
    :param data: pd.DataFrame with annotated mutants, including at least the following columns:
                ['CID', 'target_id', 'accession', 'pchembl_value_Mean', 'source']
    :return: DataFrame with statistics
    """
    # Calculate stats per variant (target_id)
    def agg_functions_target_id(x):
        d = {}
        d['connectivity_count'] = len(list(x['connectivity']))
        d['pchembl_value_Mean_Mean'] = x['pchembl_value_Mean'].mean()
        d['pchembl_value_Mean_StD'] = x['pchembl_value_Mean'].std()
        d['Activity_class_consensus'] = pd.Series.mode(x['Activity_class_consensus']) # Check if this does what it's
        # supposed
        d['source'] = list(set(list(x['source'])))
        return pd.Series(d, index=['connectivity_count', 'pchembl_value_Mean_Mean', 'pchembl_value_Mean_StD',
                                   'Activity_class_consensus', 'source'])

    stats_target_id = data.groupby(['accession', 'target_id'], as_index=False).apply(agg_functions_target_id)

    # Calculate stats per target (accession)
    def agg_functions_accession(x):
        d = {}
        d['n_variants'] = len(list(x['target_id']))
        d['target_id'] = list(x['target_id'])
        d['connectivity_count'] = list(x['connectivity_count'])
        d['connectivity_count_sum'] = x['connectivity_count'].sum()
        d['pchembl_value_Mean_Mean'] = list(x['pchembl_value_Mean_Mean'])
        d['pchembl_value_Mean_StD'] = list(x['pchembl_value_Mean_StD'])
        d['Activity_class_consensus'] = list(x['Activity_class_consensus'])
        d['source'] = list(set([j for i in x['source'] for j in i]))
        return pd.Series(d, index=['n_variants','target_id','connectivity_count','connectivity_count_sum','pchembl_value_Mean_Mean',
                                   'pchembl_value_Mean_StD','Activity_class_consensus','source'])

    stats_accession = stats_target_id.groupby(['accession'], as_index=False).apply(agg_functions_accession)

    # Sort stats
    stats_accession.sort_values(by='n_variants', ascending=False, inplace=True)
    # stats_accession.sort_values(by='connectivity_count_sum', ascending=False, inplace=True)

    return stats_accession

def compute_pairwise_similarity(data: pd.DataFrame):
    """
    Compute pairwise Tanimoto similarity between unique compounds in a dataset and write to an output file
    :param data: DataFrame with activity data
    :return: dataframe with similarity values for all pairs of compounds
    """
    out_file = '../../data/similarity_matrix.csv'
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
    dataset_file = os.path.join(output_dir, f'modelling_dataset_{accession}_{options_filename_tag}.csv')
    coverage_file = os.path.join(output_dir, f'modelling_dataset_coverage_{accession}_{options_filename_tag}.json')

    if not os.path.exists(dataset_file):
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
    dataset_file = os.path.join(output_dir, f'modelling_dataset_{accession}_{options_filename_tag}.csv')
    coverage_file = os.path.join(output_dir, f'modelling_dataset_coverage_{accession}_{options_filename_tag}.json')

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
    # Read bioactivity data for common subset precalculated
    if not common:
        data_common = pd.read_csv(os.path.join(output_dir, f'modelling_dataset_{accession}_All.csv'), sep='\t')
    else:
        data_common = pd.read_csv(os.path.join(output_dir, f'modelling_dataset_{accession}_Thr{threshold}_Cov'
                                                          f'{int(variant_coverage*100)}_Sim'
                                                          f'{int(sim_thres*100)}.csv'), sep='\t')

    return data_common


def calculate_variant_stats(data_accession: pd.DataFrame, accession: str, diff: bool):
    """
    Calculate mean and StD of activity values for each variant of one target (accession)
    :param data_accession: DataFrame with mutation activity data for accession of interest
    :param accession: Uniprot accession code
    :param diff: Whether to calculate the mean and StD for the difference in bioactivity from WT to all accession
                variants in a set. This allows to compare not just the difference between variant means but the mean
                between one-on-one differences in a common subset of compounds
    :return: two dataframes with mean and standard deviation values for each variant
    """
    data_pivoted = pd.pivot(data_accession, index='connectivity', columns='target_id', values='pchembl_value_Mean')

    if diff:
        for target_id in [col for col in data_pivoted.columns if col != 'target_id']:
            try:
                data_pivoted[f'{target_id}_WTdif'] = data_pivoted.apply(lambda x: x[target_id] - x[f'{accession}_WT'], axis=1)
            except KeyError: # WT might not be included in the subset due to lack of data
                data_pivoted[f'{target_id}_WTdif'] = np.NaN

    data_mean = data_pivoted.mean(axis=0).reset_index(name='pchembl_value_Mean_Mean')
    data_std = data_pivoted.std(axis=0).apply(lambda x: x if not math.isnan(x) else 0).reset_index(name='pchembl_value_Mean_Std')

    if diff:
        data_mean = data_mean[data_mean['target_id'].str.contains('WTdif')]
        data_mean['target_id'] = data_mean['target_id'].apply(lambda x: x.replace('_WTdif',''))
        data_std = data_std[data_std['target_id'].str.contains('WTdif')]
        data_std['target_id'] = data_std['target_id'].apply(lambda x: x.replace('_WTdif', ''))

    return data_mean,data_std

def define_consistent_palette(data: pd.DataFrame, accession: str):
    """
    Define palette for plotting that is consistent for the same data and accession
    :param data: DataFrame with mutation activity data
    :param accession: Uniprot accession code
    :return: seaborn color palette
    """
    palette = {f'{accession}_WT': '#808080'} # Initialize with WT as grey
    list_variants = [target_id for target_id in data[data['accession'] == accession]['target_id'].unique().tolist() if not 'WT' in target_id]
    list_colors = sns.color_palette("Spectral", n_colors=len(list_variants)).as_hex()
    try:
        list_variants_sorted = sorted(list_variants, key=lambda x: int(x.split('_')[1][1:-1]))
    except ValueError:
        list_variants_sorted = list_variants
    list_colors_accession = list_colors[0:len(list_variants_sorted)+1]

    # Populate palette dictionary
    for variant,color in zip(list_variants_sorted,list_colors_accession):
        palette[variant] = color
    return palette


def compute_variant_activity_distribution(data: pd.DataFrame, accession: str, common: bool, sim: bool, sim_thres: int,
                                       threshold: int, variant_coverage: float, plot: bool, hist: bool, plot_mean: bool,
                                       save_dataset: bool, output_dir: str):
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
    :param save_dataset: Whether to save the (subset) modelling dataset used for computing activity distribution
    :param output_dir: Location for the output files
    :return: None
    """
    # Customize filename tags based on function options
    options_filename_tag = get_filename_tag(common, sim, sim_thres, threshold, variant_coverage)

    # Check in output stats or dataset file if this accession was already analyzed. If so, skip analysis.
    stat_file = os.path.join(output_dir, f'stats_file_{options_filename_tag}.txt')
    dataset_file = os.path.join(output_dir, f'modelling_dataset_{accession}_{options_filename_tag}.csv')

    if plot and (os.path.exists(stat_file)) and (accession in pd.read_csv(stat_file, sep='\t')['accession'].unique().tolist()):
        print(f'{accession} already plotted and statistics analyzed. Skipping...')
    elif save_dataset and (os.path.exists(dataset_file)) and (accession in pd.read_csv(stat_file, sep='\t')['accession'].unique().tolist()):
        print(f'{accession} dataset already saved. Skipping...')

    else:
        # Define color palette and plotting order to ensure consistent colors over different subsets
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
                data_mean, data_std = calculate_variant_stats(data_accession, accession, diff=False)
                data_mean_error, data_std_error = calculate_variant_stats(data_accession, accession, diff=True)

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
                            options_legend_tag = f'Common subset (+ similars) n={len(data_accession["connectivity"].unique().tolist())}\n(> {variant_coverage * 100} % coverage)'
                        else:
                            options_legend_tag = f'Common subset n={len(data_accession["connectivity"].unique().tolist())}\n(> {variant_coverage*100} % coverage)'
                    else:
                        options_legend_tag = f'Common subset n={len(data_accession["connectivity"].unique().tolist())}\n(Not defined coverage)'
                else:
                    options_legend_tag = f'Full set n={len(data_accession["connectivity"].unique().tolist())}'

                g._legend.set_title(f'{accession} variants\n{options_legend_tag}')
                sns.move_legend(g, bbox_to_anchor=(0.75, 0.5), loc='center left')
                plt.xlabel('pChEMBL value (Mean)')
                plt.xlim(2, 12)

                # Write figure
                plt.savefig(os.path.join(output_dir,
                                         f'variant_activity_distribution_{accession}_{options_filename_tag}.png'),
                                         bbox_inches='tight', dpi=300)
                plt.savefig(os.path.join(output_dir,
                                         f'variant_activity_distribution_{accession}_{options_filename_tag}.svg'))

                # Write stats output file
                target_id_list = [target_id for target_id in hue_order]
                accession_list = [accession for x in target_id_list]
                mean_list = [data_mean.loc[data_mean['target_id'] == target_id, 'pchembl_value_Mean_Mean'].item() for target_id in target_id_list]
                std_list = [data_std.loc[data_std['target_id'] == target_id, 'pchembl_value_Mean_Std'].item() for target_id in target_id_list]
                mean_error_list = [data_mean_error.loc[data_mean_error['target_id'] == target_id, 'pchembl_value_Mean_Mean'].item() for target_id in target_id_list]
                std_error_list = [data_std_error.loc[data_std_error['target_id'] == target_id, 'pchembl_value_Mean_Std'].item() for target_id in target_id_list]
                n_accession_list = [len(data_accession['connectivity'].unique().tolist()) for x in target_id_list]
                n_target_id_list = [sum(data_accession[data_accession["target_id"] == target_id]["pchembl_value_Mean"].notna().tolist()) for target_id in target_id_list]
                coverage_list = [coverage_dict[target_id][0] for target_id in target_id_list]
                stat_dict = {'accession':accession_list,
                             'target_id':target_id_list,
                             'mean_pchembl':mean_list,
                             'std_pchembl':std_list,
                             'mean_error':mean_error_list,
                             'std_error':std_error_list,
                             'n_accession':n_accession_list,
                             'n_target_id':n_target_id_list,
                             'coverage':coverage_list}

                stat_df = pd.DataFrame(stat_dict)

                if not os.path.exists(stat_file):
                    stat_df.to_csv(stat_file, sep='\t', index=False)
                else:
                    stat_df.to_csv(stat_file, mode='a', sep='\t', index=False, header=False)
                print(f'{accession} done.')

            # Write output file with skipped accession codes (not enough data for plotting)
            else:
                print(f'Skipping accession {accession}: not enough data for plotting.')
                if not os.path.exists(os.path.join(output_dir, f'skipped_accession_{options_filename_tag}.txt')):
                    with open(os.path.join(output_dir, f'skipped_accession_{options_filename_tag}.txt'),'w') as file:
                        file.write(f'Skipping accession {accession}\n')
                else:
                    with open(os.path.join(output_dir, f'skipped_accession_{options_filename_tag}.txt'),'r+') as file:
                        for line in file:
                            if accession in line:
                                break
                        else:
                            file.write(f'Skipping accession {accession}\n')

def extract_relevant_targets(file_dir: str, common: bool, sim: bool, sim_thres: int, threshold: int, variant_coverage: float,
                             min_subset_n: int = 50, thres_error_mean: float = 0.5, error_mean_limit: str = 'min'):
    """
    Explore the stats file produced while plotting and extract the most interesting targets from it
    :param file_dir: Location of the stats file
    :param common: Whether to use common subset for variants
    :param sim: Whether to include similar compounds in the definition of the common subset
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold: Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :param min_subset_n: Minimum number of compounds in the target (sub)set
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
    stat_df = pd.read_csv(os.path.join(file_dir, f'stats_file_{filename_tag}.txt'), sep='\t')
    stat_df.drop_duplicates(['accession','target_id'], inplace=True)

    # Extract accession code with subset size bigger than min_subset
    accession_max_subset = stat_df[stat_df['n_accession'] > min_subset_n]['accession'].unique().tolist()

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
        print(stat_df_keep.groupby(['accession'])['n_accession', 'mean_error'].apply(lambda x: x.abs().max()))
    elif error_mean_limit == 'max':
        print(stat_df_keep[stat_df_keep['mean_error'] != 0].groupby(['accession'])['n_accession', 'mean_error'].
              apply(lambda x: x.abs().min()))
    else:
        print(stat_df_keep.groupby(['accession'])['n_accession','mean_error'].agg({'n_accession': 'max',
                                                                                   'mean_error': 'std'}))

    return stat_df_keep



if __name__ == '__main__':
    pd.options.display.width = 0
    # Define output directory for mutant statistical analysis
    output_dir = 'C:\\Users\gorostiolam\Documents\Gorostiola Gonzalez, ' \
                 'Marina\PROJECTS\\6_Mutants_PCM\DATA\\2_Analysis\\0_mutant_statistics\\1_common_subset'

    # Get data with mutants
    data_with_mutants = merge_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000)

    # Compute quick statistics from data with mutants to check which targets might be of interest
    stats = compute_stats_per_accession(data_with_mutants)

    # Check if Adenosine A2A is in the set
    stats_a2a = stats[stats['accession'] == 'P29274']

    # Compute variant activity distribution and plot results for full set and common subsets for all targets
    for accession in stats['accession'].tolist():
        # Full dataset
        compute_variant_activity_distribution(data_with_mutants, accession, common=False, sim=False, sim_thres=None,
                                              threshold=None, variant_coverage=None, plot=True, hist=False, plot_mean=True,
                                              save_dataset=False,output_dir=os.path.join(output_dir, 'all'))
        # Strict common subset with > 20% coverage
        compute_variant_activity_distribution(data_with_mutants, accession, common=True, sim=False, sim_thres=None,
                                           threshold=2,variant_coverage=0.2, plot=True, hist=False, plot_mean=True,
                                           save_dataset=False, output_dir=os.path.join(output_dir,'common_subset_20'))
        # Common subset with > 20% coverage including similar compounds (>80% Tanimoto) tested in other variants
        compute_variant_activity_distribution(data_with_mutants, accession, common=True, sim=True, sim_thres=0.8,
                                           threshold=2,variant_coverage=0.2, plot=True, hist=False, plot_mean=True,
                                           save_dataset=False, output_dir=os.path.join(output_dir,'common_subset_20_sim_80'))

    # Extract relevant targets for reporting and modelling (using common subset with similarity)
    # A) Check which targets have the biggest common subsets
    extract_relevant_targets(os.path.join(output_dir,'common_subset_20_sim_80'),
                             common=True, sim=True, sim_thres=0.8, threshold=2,variant_coverage=0.2,
                             min_subset_n= 50, thres_error_mean=0, error_mean_limit='min')
    # B) Check which targets have the biggest variance between mutant activity distributions
    extract_relevant_targets(os.path.join(output_dir, 'common_subset_20_sim_80'),
                             common=True, sim=True, sim_thres=0.8, threshold=2, variant_coverage=0.2,
                             min_subset_n= 5, thres_error_mean=1,error_mean_limit='var_min')
    # C) Check which targets have the smallest variance between mutant activity distributions
    extract_relevant_targets(os.path.join(output_dir, 'common_subset_20_sim_80'),
                             common=True, sim=True, sim_thres=0.8, threshold=2, variant_coverage=0.2,
                             min_subset_n= 5, thres_error_mean=0.1,error_mean_limit='var_max')

    # Write datasets for modelling for the relevant targets of interest
    accession_large_subsets = extract_relevant_targets(os.path.join(output_dir,'common_subset_20_sim_80'),
                             common=True, sim=True, sim_thres=0.8, threshold=2,variant_coverage=0.2,
                             min_subset_n= 90, thres_error_mean=0, error_mean_limit='min')['accession'].unique().tolist()

    for accession in accession_large_subsets:
        # Full dataset
        compute_variant_activity_distribution(data_with_mutants, accession, common=False, sim=False, sim_thres=None,
                                              threshold=None, variant_coverage=None, plot=False, hist=False, plot_mean=True,
                                              save_dataset=True,output_dir=os.path.join(output_dir, 'all'))
        # Strict common subset with > 20% coverage
        compute_variant_activity_distribution(data_with_mutants, accession, common=True, sim=False, sim_thres=None,
                                           threshold=2,variant_coverage=0.2, plot=False, hist=False, plot_mean=True,
                                           save_dataset=True, output_dir=os.path.join(output_dir,'common_subset_20'))
        # Common subset with > 20% coverage including similar compounds (>80% Tanimoto) tested in other variants
        compute_variant_activity_distribution(data_with_mutants, accession, common=True, sim=True, sim_thres=0.8,
                                           threshold=2,variant_coverage=0.2, plot=False, hist=False, plot_mean=True,
                                           save_dataset=True, output_dir=os.path.join(output_dir,'common_subset_20_sim_80'))





