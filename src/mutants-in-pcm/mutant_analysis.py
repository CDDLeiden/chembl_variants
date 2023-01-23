# -*- coding: utf-8 -*-


"""Mutant statistics analysis."""
import json
import os

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

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
        return pd.Series(d, index=['pchembl_value_Mean', 'Activity_class_consensus', 'source'])

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
    # stats_accession.sort_values(by='n_variants', ascending=False, inplace=True)
    stats_accession.sort_values(by='connectivity_count_sum', ascending=False, inplace=True)

    return stats_accession

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

        common_subset.reset_index(inplace=True)

        common_subset_melt = pd.melt(common_subset, id_vars=['target_id'],
                                     value_vars=[x for x in common_subset.columns if x not in ['connectivity','target_id']],
                                     value_name='pchembl_value_Mean')
        return common_subset_melt
    else:
        return data_accession_agg

def plot_variant_activity_distribution(data: pd.DataFrame, accession: str, wt:bool, common:bool, threshold:int,
                                       variant_coverage:float, hist:bool, output_dir: str):
    """
    Plot the distribution of bioactivity data for all variants of one target (accession)
    :param data: DataFrame with bioactivity including mutant annotation (target_id)
    :param accession:
    :return:
    """
    if not wt:
        data = data[~data['target_id'].str.contains('WT')]

    data_accession = get_variant_common_subset(data, accession, common, threshold, variant_coverage)

    # Check if data has enough variance to be plotted in a distribution plot
    if not data_accession.groupby(['target_id'])['pchembl_value_Mean'].var().isnull().all():
        if not hist:
            g = sns.displot(data_accession, x='pchembl_value_Mean', hue='target_id', kind='kde', fill=True)
        else:
            g = sns.displot(data_accession, x='pchembl_value_Mean', hue='target_id', kde=True)

        plt.xlabel('pChEMBL value (Mean)')
        if variant_coverage is not None:
            g._legend.set_title(f'{accession} variants\nCommon subset \n({variant_coverage*100} % coverage)')
        else:
            g._legend.set_title(f'{accession} variants\nCommon subset \n(Not defined coverage)')

        plt.savefig(os.path.join(output_dir, f'variant_activity_distribution_{accession}.png'), dpi=300)

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


if __name__ == '__main__':
    pd.options.display.width = 0
    data_with_mutants = combine_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000)
    stats = compute_stats_per_accession(data_with_mutants)

    # # Check if Adenosine A2A is in the set
    # stats_a2a = stats[stats['accession'] == 'P29274']
    # print(stats_a2a)

    output_dir = 'C:\\Users\gorostiolam\Documents\Gorostiola Gonzalez, Marina\PROJECTS\\6_Mutants_PCM\DATA\\2_Analysis\\0_mutant_statistics'
    for accession in stats['accession'].tolist():
        plot_variant_activity_distribution(data_with_mutants, accession, wt=True, common=True, threshold=2,
                                           variant_coverage=0.2, hist=False, output_dir=os.path.join(output_dir,'common_subset_20'))
        plot_variant_activity_distribution(data_with_mutants, accession, wt=True, common=True, threshold=2,
                                           variant_coverage=None, hist=False, output_dir=os.path.join(output_dir,'common_subset'))
        plot_variant_activity_distribution(data_with_mutants, accession, wt=True, common=False, threshold=None,
                                           variant_coverage=None, hist=False, output_dir=os.path.join(output_dir,'all'))

