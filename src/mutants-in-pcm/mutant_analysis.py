# -*- coding: utf-8 -*-


"""Mutant statistics analysis."""
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
from preprocessing import combine_chembl_papyrus_mutants

def calculate_mean_activity_chembl_papyrus(data: pd.DataFrame):
    """
    From a dataset with concatenated ChEMBL and Papyrus entries, compute mean pchembl_value for the same target_id-connectivity pair
    :param data:
    :return:
    """
    def agg_functions_variant_connectivity(x):
        d ={}
        d['pchembl_value_Mean'] = x['pchembl_value_Mean'].mean()
        d['Activity_class_consensus'] = pd.Series.mode(x['Activity_class'])
        d['source'] = list(x['source'])
        d['SMILES'] = list(x['SMILES'])[0]
        return pd.Series(d, index=['pchembl_value_Mean', 'Activity_class_consensus', 'source', 'SMILES'])

    agg_activity_data = data.groupby(['target_id','connectivity'], as_index=False).apply(agg_functions_variant_connectivity)

    return agg_activity_data


def compute_stats_per_accession(data: pd.DataFrame):
    """
    Compute statistics on mutant and bioactivity data counts for each target (accession)
    :param data: pd.DataFrame with annotated mutants, including at least the following columns:
                ['CID', 'target_id', 'accession', 'pchembl_value_Mean', 'source']
    :return: pd.DataFrame
    """
    # Calculate stats per variant (target_id)
    def agg_functions_target_id(x):
        d = {}
        d['connectivity_count'] = len(list(x['connectivity']))
        d['pchembl_value_Mean_Mean'] = x['pchembl_value_Mean'].mean()
        d['pchembl_value_Mean_StD'] = x['pchembl_value_Mean'].std()
        d['Activity_class_consensus'] = pd.Series.mode(x['Activity_class']) # Check if this does what it's supposed
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

def compute_similarity_matrix(data: pd.DataFrame):
    """
    Compute Tanimoto similarity matrix between unique compounds in a dataset and write to a output file
    :param data:
    :return:
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

def get_variant_similar_subset(data:pd.DataFrame, accession:str, sim_thres:int, threshold:int, variant_coverage: float):
    """

    :param data:
    :param accession:
    :param sim_thres:
    :param threshold:
    :param variant_coverage:
    :return:
    """
    data_accession = data[data['accession'] == accession]

    # Combine ChEMBL and Papyrus bioactivity data for plotting
    data_accession_agg = calculate_mean_activity_chembl_papyrus(data_accession)

    # Compute or read similarity
    unique_compounds = data_accession_agg.drop_duplicates('connectivity', keep='first')[['connectivity', 'SMILES']]
    ids = unique_compounds['connectivity'].tolist()

    similarity_df = compute_similarity_matrix(data) # Similarity read for all possible accessions

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
    return common_subset_melt,coverage_dict



def get_variant_common_subset(data: pd.DataFrame, accession:str, common:bool, threshold:int, variant_coverage: float):
    """
    Get the common subset of compounds tested on all (or most) variants
    :param data:
    :param accession:
    :return:
    """
    data_accession = data[data['accession'] == accession]

    # Combine ChEMBL and Papyrus bioactivity data for plotting
    data_accession_agg = calculate_mean_activity_chembl_papyrus(data_accession)

    if common:
        data_pivoted = pd.pivot(data_accession_agg,index='target_id',columns='connectivity',values='pchembl_value_Mean')

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

        # Calculate coverage per variant (default and with similar compounds)
        coverage_dict = {}
        for variant in common_subset.index:
            try:
                variant_coverage = len(
                    [x for x in common_subset.loc[variant, :].values.tolist() if not math.isnan(x)]) / \
                                   common_subset.shape[1]
            except ZeroDivisionError:
                variant_coverage = 0.0
            coverage_dict[variant] = [variant_coverage]


        common_subset.reset_index(inplace=True)

        common_subset_melt = pd.melt(common_subset, id_vars=['target_id'],
                                     value_vars=[x for x in common_subset.columns if x not in ['connectivity','target_id']],
                                     value_name='pchembl_value_Mean')
        return common_subset_melt,coverage_dict
    else:
        return data_accession_agg,{}

def calculate_variant_stats(data_accession: pd.DataFrame, accession:str, diff:bool):
    """
    Calculate the difference in bioactivity from WT to all accession variants in a set
    :param data_accession:
    :param accession:
    :param diff:
    :return:
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

def define_consistent_palette(data, accession):
    """
    Define palette for plotting that is consistent for the same data and accession
    :param data:
    :param accession:
    :return:
    """
    palette = {f'{accession}_WT': '#808080'} # Initialize with WT as grey
    list_colors = sns.color_palette("pastel6",n_colors=71).as_hex()
    list_variants = [target_id for target_id in data[data['accession'] == accession]['target_id'].unique().tolist() if not 'WT' in target_id]
    try:
        list_variants_sorted = sorted(list_variants, key=lambda x: int(x.split('_')[1][1:-1]))
    except ValueError:
        list_variants_sorted = list_variants
    list_colors_accession = list_colors[0:len(list_variants_sorted)+1]

    # Populate palette dictionary
    for variant,color in zip(list_variants_sorted,list_colors_accession):
        palette[variant] = color
    return palette

def plot_variant_activity_distribution(data: pd.DataFrame, accession: str, common:bool, sim:bool, sim_thres:int,
                                       threshold:int, variant_coverage:float, hist:bool, plot_mean:bool,
                                       output_dir: str):
    """
    Plot the distribution of bioactivity data for all variants of one target (accession)
    :param data: DataFrame with bioactivity including mutant annotation (target_id)
    :param accession:
    :return:
    """
    # Check in output stats file if this accession was already analyzed. If so, skip analysis.
    stat_file = os.path.join(output_dir, 'stats_file.txt')
    if (os.path.exists(stat_file)) and (accession in pd.read_csv(stat_file, sep='\t')['accession'].unique().tolist()):
        print(f'{accession} already analyzed. Skipping...')
    else:
        # Define color palette and plotting order to ensure consistent colors over different subsets
        palette = define_consistent_palette(data,accession)

        # Calculate common subset for plotting (with or without similarity match amplification)
        if not sim:
            data_accession,coverage_dict = get_variant_common_subset(data, accession, common, threshold, variant_coverage)
        else:
            data_accession,coverage_dict = get_variant_similar_subset(data, accession, sim_thres, threshold, variant_coverage)

        # Plot only if data has enough variance to be plotted in a distribution plot
        if not data_accession.groupby(['target_id'])['pchembl_value_Mean'].var().isnull().all():
            # Plotting options
            sns.set_style('ticks')
            # Plot distribution
            hue_order = [target_id for target_id in palette.keys() if target_id in data_accession['target_id'].unique().tolist()]
            if not hist: # Plot distribution curve
                g = sns.displot(data_accession, x='pchembl_value_Mean', hue='target_id', kind='kde', fill=True,
                                palette=palette, hue_order=hue_order, height=5, aspect=1.5, warn_singular=False)
            else: # Plot histograms
                g = sns.displot(data_accession, x='pchembl_value_Mean', hue='target_id', kde=True,
                                palette=palette, hue_order=hue_order, height=5, aspect=1.5, warn_singular=False)

            # Calculate subset variant statistics for reporting (and plotting)
            data_mean, data_std = calculate_variant_stats(data_accession, accession, diff=False)
            data_mean_error, data_std_error = calculate_variant_stats(data_accession, accession, diff=True)

            # Plot Mean pchembl_value per variant
            if plot_mean:
                for i, variant in enumerate(data_accession['target_id'].unique().tolist()):
                    variant_mean = data_mean.loc[data_mean['target_id'] == variant, 'pchembl_value_Mean_Mean'].item()
                    variant_std = data_std.loc[data_std['target_id'] == variant, 'pchembl_value_Mean_Std'].item()
                    plt.axvline(variant_mean, color=palette[variant])

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

            # Customize axis, legend labels, and output file name based on function options
            if common:
                if variant_coverage is not None:
                    if sim:
                        options_legend_tag = f'Common subset (+ similars) n={len(data_accession["connectivity"].unique().tolist())}\n(> {variant_coverage * 100} % coverage)'
                        options_filename_tag = f'{int(variant_coverage * 100)}_sim_{int(sim_thres * 100)}'
                    else:
                        options_legend_tag = f'Common subset n={len(data_accession["connectivity"].unique().tolist())}\n(> {variant_coverage*100} % coverage)'
                        options_filename_tag = f'{int(variant_coverage * 100)}'
                else:
                    options_legend_tag = f'Common subset n={len(data_accession["connectivity"].unique().tolist())}\n(Not defined coverage)'
                    options_filename_tag = f'NoCov'
            else:
                options_legend_tag = f'Full set n={len(data_accession["connectivity"].unique().tolist())}'
                options_filename_tag = f'all'

            g._legend.set_title(f'{accession} variants\n{options_legend_tag}')
            plt.xlabel('pChEMBL value (Mean)')

            # Write figure
            plt.savefig(os.path.join(output_dir,
                                     f'variant_activity_distribution_{accession}_{options_filename_tag}.png'), dpi=300)

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
            if not os.path.exists(os.path.join(output_dir, 'skipped_accession.txt')):
                with open(os.path.join(output_dir, 'skipped_accession.txt'),'w') as file:
                    file.write(f'Skipping accession {accession}\n')
            else:
                with open(os.path.join(output_dir, 'skipped_accession.txt'),'r+') as file:
                    for line in file:
                        if accession in line:
                            break
                    else:
                        file.write(f'Skipping accession {accession}\n')

def extract_relevant_targets(file_dir: str, min_subset_n: int = 50, min_error_mean: float = 0.5):
    """
    Explore the stats file produced while plotting and extract the most interesting targets from it
    :param file_dir:
    :param min_subset_n:
    :param min_error_mean:
    :return:
    """
    stat_df = pd.read_csv(os.path.join(file_dir, 'stats_file.txt'), sep='\t')
    stat_df.drop_duplicates(['accession','target_id'], inplace=True)

    # Extract accession code with subset size bigger than min_subset
    accession_max_subset = stat_df[stat_df['n_accession'] > min_subset_n]['accession'].unique().tolist()

    # Extract accession codes where at least one variant has a mean pchembl value difference wo WT bigger than min_error_mean
    mean_error_list = stat_df.groupby(['accession'])['mean_error'].apply(list).reset_index()
    accession_max_error = mean_error_list[mean_error_list['mean_error'].apply(lambda x: any(np.abs(n) > min_error_mean for n in x))]['accession'].tolist()

    # Keep accession codes that satisfy both conditions
    accession_keep = list(set(accession_max_subset).intersection(accession_max_error))
    stat_df_keep = stat_df[stat_df['accession'].isin(accession_keep)]

    print(f'{len(accession_keep)} targets satisfy the specified conditions:')
    print(stat_df_keep.groupby(['accession'])['n_accession', 'mean_error'].apply(lambda x: x.abs().max()))

    return stat_df_keep



if __name__ == '__main__':
    pd.options.display.width = 0
    data_with_mutants = combine_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000)
    stats = compute_stats_per_accession(data_with_mutants)

    # # Check if Adenosine A2A is in the set
    # stats_a2a = stats[stats['accession'] == 'P29274']
    # print(stats_a2a)

    output_dir = 'C:\\Users\gorostiolam\Documents\Gorostiola Gonzalez, Marina\PROJECTS\\6_Mutants_PCM\DATA\\2_Analysis\\0_mutant_statistics'
    # for accession in stats['accession'].tolist():
    #     # plot_variant_activity_distribution(data_with_mutants, accession, common=True, sim=False, sim_thres=None,
    #     #                                    threshold=2,variant_coverage=0.8, hist=False, plot_mean=True,
    #     #                                    output_dir=os.path.join(output_dir, 'common_subset_80'))
    #     plot_variant_activity_distribution(data_with_mutants, accession, common=True, sim=False, sim_thres=None,
    #                                        threshold=2,variant_coverage=0.2, hist=False, plot_mean=True,
    #                                        output_dir=os.path.join(output_dir,'common_subset_20'))
    #     # plot_variant_activity_distribution(data_with_mutants, accession,common=True, sim=False, sim_thres=None,
    #     #                                    threshold=2,variant_coverage=None, hist=False, plot_mean=True, plot_mean_diff=False,output_dir=os.path.join(output_dir,'common_subset'))
    #     # plot_variant_activity_distribution(data_with_mutants, accession, common=False, sim=False, sim_thres=None,
    #     #                                    threshold=None,variant_coverage=None, hist=False, plot_mean=True, plot_mean_diff=False, output_dir=os.path.join(output_dir,'all'))
    #     plot_variant_activity_distribution(data_with_mutants, accession, common=True, sim=True, sim_thres=0.8,
    #                                        threshold=2,variant_coverage=0.2, hist=False, plot_mean=True,
    #                                        output_dir=os.path.join(output_dir,'common_subset_20_sim_80'))

    extract_relevant_targets(os.path.join(output_dir, 'common_subset_20'))
    extract_relevant_targets(os.path.join(output_dir,'common_subset_20_sim_80'))





