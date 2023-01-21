# -*- coding: utf-8 -*-


"""Mutant statistics analysis."""
import json
import os

import pandas as pd
import numpy as np
import re

from preprocessing import combine_chembl_papyrus_mutants

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
        d['CID_count'] = len(list(x['CID']))
        d['pchembl_value_Mean_Mean'] = x['pchembl_value_Mean'].mean()
        d['pchembl_value_Mean_StD'] = x['pchembl_value_Mean'].std()
        d['Activity_class_consensus'] = pd.Series.mode(x['Activity_class']) # Check if this does what it's supposed
        d['source'] = list(set(list(x['source'])))
        return pd.Series(d, index=['CID_count', 'pchembl_value_Mean_Mean', 'pchembl_value_Mean_StD',
                                   'Activity_class_consensus', 'source'])

    stats_target_id = data.groupby(['accession', 'target_id'], as_index=False).apply(agg_functions_target_id)

    # Calculate stats per target (accession)
    def agg_functions_accession(x):
        d = {}
        d['n_variants'] = len(list(x['target_id']))
        d['target_id'] = list(x['target_id'])
        d['CID_count'] = list(x['CID_count'])
        d['CID_count_sum'] = x['CID_count'].sum()
        d['pchembl_value_Mean_Mean'] = list(x['pchembl_value_Mean_Mean'])
        d['pchembl_value_Mean_StD'] = list(x['pchembl_value_Mean_StD'])
        d['Activity_class_consensus'] = list(x['Activity_class_consensus'])
        d['source'] = list(set([j for i in x['source'] for j in i]))
        return pd.Series(d, index=['n_variants','target_id','CID_count','CID_count_sum','pchembl_value_Mean_Mean',
                                   'pchembl_value_Mean_StD','Activity_class_consensus','source'])

    stats_accession = stats_target_id.groupby(['accession'], as_index=False).apply(agg_functions_accession)

    # Sort stats
    # stats_accession.sort_values(by='n_variants', ascending=False, inplace=True)
    stats_accession.sort_values(by='CID_count_sum', ascending=False, inplace=True)

    return stats_accession

if __name__ == '__main__':
    pd.options.display.width = 0
    data_with_mutants = combine_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000)
    stats = compute_stats_per_accession(data_with_mutants)
    print(stats)

    # Check if Adenosine A2A is in the set
    stats_a2a = stats[stats['accession'] == 'P29274']
    print(stats_a2a)