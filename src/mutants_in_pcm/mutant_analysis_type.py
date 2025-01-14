# -*- coding: utf-8 -*-


"""Mutant statistics analysis. Part X"""
"""Analyzing mutant data according to the type of mutation"""

import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as plticker
import seaborn as sns
from Bio.SeqUtils import seq1
from UniProtMapper import UniProtIDMapper
mapper = UniProtIDMapper()

from .preprocessing import merge_chembl_papyrus_mutants
from .mutant_analysis_protein import calculate_average_residue_distance_to_ligand
from .data_path import get_data_path


def map_mutation_distance_BLOSUM62(data: pd.DataFrame):
    """
    Annotate amino acid changes with their probability according to the BLOSUM62 matrix. BLOSUM scores range from -4
    to 11. More positive mean conservative aa change, and negative non-conservative.
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :return: dataframe with bioactivity data and mutation BLOSUM score annotated
    """


def read_mutation_distance_Epstein():
    """
    Read Epstein's coefficient of difference distance matrix of amino acid differences. Epstein's coefficient of
    difference is based on the differences in polarity and size between replaced pairs of amino acids.
    The lowest the value, the most similar two amino acids.
    The distances are directional (i.e. F > M is not the same as M > F).
    :return: dictionary with a value per directional aa change
    """
    data_dir = get_data_path()
    matrix = pd.read_csv(os.path.join(data_dir,'Epstein_matrix.csv'), sep=';', index_col=0)

    epstein_dict ={}

    for aa_1 in matrix.index.tolist():
        for aa_2 in matrix.columns:
            if not aa_1 == aa_2: # Skip silent mutations
                epstein_dict[f'{seq1(aa_1)}{seq1(aa_2)}'] = matrix.loc[aa_1, aa_2]


    return epstein_dict

def map_mutation_distance_Epstein(data: pd.DataFrame):
    """
    Annotate amino acid changes with their distance according to the properties defined by Epstein (1967) :
    polarity, and size.
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :return: dataframe with bioactivity data and mutation distance annotated
    """

def read_mutation_distance_Grantham():
    """
    Read Grantham's distance matrix of amino acid differences. Grantham's distance depends on three properties:
    composition, polarity and molecular volume. The lowest the value, the most similar two amino acids.
    The distances are not directional.
    :return: dictionary with a value per directional aa change
    """
    data_dir = get_data_path()
    matrix = pd.read_csv(os.path.join(data_dir,'Grantham_matrix.csv'), sep=';', index_col=0)

    grantham_dict ={}

    for aa_1 in matrix.index.tolist():
        for aa_2 in matrix.columns:
            if not aa_1 == aa_2: # Skip silent mutations
                # Matrix is not directional, so aa_1>aa_2 and aa_2>aa_1 is the same
                aa_change = f'{"".join(sorted([seq1(aa_1),seq1(aa_2)]))}'
                if not aa_change in grantham_dict.keys():
                    grantham_dict[aa_change] = matrix.loc[aa_1, aa_2]

    return grantham_dict

def read_blosum():
    """
    Read BLOSUM62 matrix
    :return:
    """
    data_dir = get_data_path()
    matrix = pd.read_csv(os.path.join(data_dir,'BLOSUM62.txt'), sep=';', index_col=0, skiprows=6)

    blosum_dict ={}

    for aa_1 in matrix.index.tolist():
        for aa_2 in matrix.columns:
            if not aa_1 == aa_2: # Skip silent mutations
                blosum_dict[f'{aa_1}{aa_2}'] = matrix.loc[aa_1, aa_2]

    return blosum_dict

def map_mutation_distance_Grantham(data: pd.DataFrame):
    """
    Annotate amino acid changes with their distance according to the properties defined by Grantham (1974) :
    composition, polarity, and molecular volume.
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :return: dataframe with bioactivity data and mutation distance annotated
    """

def extract_substitutions(data: pd.DataFrame):
    """
    Annotate unique amino acid substitutions for each target_id a dataframe
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :return: annotated dataframe
    """
    data_annotated = data.copy(deep=True)
    data_annotated['mutations'] = data_annotated['target_id'].apply(lambda x: x.split('_')[1:])
    data_annotated['aa_change'] = data_annotated['mutations'].apply(lambda x: '_'.join(x))

    def extract_target_id_mutation_n(x):
        if 'WT' in x:
            return 0
        elif 'MUTANT' in x:
            return -1
        else:
            return len(x)
    data_annotated['target_id_mutation_n'] = data_annotated['mutations'].apply(extract_target_id_mutation_n)

    return data_annotated

def keep_single_mutants(data: pd.DataFrame):
    """
    Keep only single mutants in the dataset
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :return: dataframe with bioactivity data and only single mutants
    """
    data_annotated = extract_substitutions(data)
    data_annotated = data_annotated[data_annotated['target_id_mutation_n'] == 1]

    return data_annotated

def expand_single_mutations(data: pd.DataFrame, keep_only_single: bool = False):
    """
    Expand dataframe with single mutants to include all possible single mutants for each target_id
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :param keep_only_single: whether to keep only single mutants in the dataset. If keep_only_single is
    True, only single mutants are kept. If keep_only_single is False and multiple mutants are present,
    they are expanded to all possible single mutants
    :return: dataframe with bioactivity data and all possible single mutants
    """
    if keep_only_single:
        data_annotated = keep_single_mutants(data)
    else:
        data_annotated = extract_substitutions(data)

    data_columns = data.columns.tolist()

    # Expand DataFrame based on the number of mutations
    expanded_data = []
    for index, row in data_annotated.iterrows():
        for mutation in row['mutations']:
            expanded_data.append({
                'target_id': row['target_id'],
                'connectivity': row['connectivity'],
                'aa_change': mutation,
                'single_substitution': mutation,
                'target_id_mutation_n': row['target_id_mutation_n']
            })
            # add rest of original columns
            for col in data_columns:
                if col not in ['target_id','mutations',
                               'aa_change','target_id_mutation_n',
                               'pchembl_value_Mean',
                               'Activity_class_consensus'
                               ]:
                    expanded_data[-1][col] = row[col]

    # Create expanded DataFrame
    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df

def unique_single_mutations(data: pd.DataFrame, accession: str):
    """
    Extract unique single mutations for a protein from a dataframe
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :param accession: Uniprot accession code
    :return: dataframe with bioactivity data and only single mutants
    """
    data = data[data['target_id'].str.contains(accession)]

    mutation_list = []
    for target_id in data['target_id']:
        for mutation in target_id.split('_')[1:]:
            if mutation not in mutation_list:
                mutation_list.append(mutation)

    return mutation_list

def map_aa_change(data: pd.DataFrame, direction: bool = False, keep_only_single: bool = False):
    """
    Annotate aminoacid change from wt to mutated aa
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :param direction: whether to take into account the direction of the mutation or just the change itself
    :param keep_only_single: whether to keep only single mutants in the dataset. If keep_only_single is
    True, only single mutants are kept. If keep_only_single is False and multiple mutants are present,
    they are expanded to all possible single mutants
    :return: dataframe with bioactivity data and amino acid change annotated expanded to all possible single mutants
    """
    # first decide what to do with multiple mutants
    data_annotated = expand_single_mutations(data, keep_only_single)

    # def extract_aa_change(x):
    def extract_aa_change(mutation):
        # mutation = x['target_id'].split('_')[1]
        if mutation == 'WT':
            aa_change = '-'
        elif mutation == 'MUTANT':
            aa_change = '-' # Undefined mutation (low confidence)
        else:
            wt_aa = mutation[0]
            mut_aa = mutation[-1]
            # Discard non-common amino acids
            aa_list = ['A','G','I','L','P','V','F','W','Y','D','E','R','H','K','S','T','C','M','N','Q']
            if wt_aa not in aa_list or mut_aa not in aa_list:
                aa_change = '-'
            else:
                aa_change_list = [wt_aa,mut_aa]
                # Do not consider the directionality of the mutation, just the type of aa change
                if not direction:
                    aa_change_list = sorted(aa_change_list)
                aa_change = ''.join(aa_change_list)

        return aa_change

    # data['aa_change'] = data.apply(extract_aa_change, axis=1)
    data_annotated['aa_change'] = data_annotated['aa_change'].apply(extract_aa_change)

    # return data
    return data_annotated

def map_mutation_type(data: pd.DataFrame):
    """
    Annotate mutation type (conservative, polar, size, polar_size) according to the differences in side chain and
    polarity.
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :return: dataframe with bioactivity data and mutation type annotated
    """
    # Define polarity of amino acid chains (reference: Virtual ChemBook Elmhurst College, 2003)
    non_polar_aas = ['A', 'G', 'I', 'L', 'P', 'V', 'M', 'F']
    polar_neutral_aas = ['N', 'Q', 'S', 'T', 'Y', 'C', 'W'] # C and W slightly polar
    polar_acidic_aas = ['E', 'D']
    polar_basic_aas = ['R', 'H', 'K']

    # Define relative size of amino acid chains  (reference: Epstein, 1967)
    bulky_aas = ['W', 'Y', 'R', 'F'] # Relative size 0.5-0.35
    intermediate_size_aas = ['H', 'E', 'Q', 'K', 'M', 'D', 'N', 'L', 'I', 'P'] # Relative size 0.30-0.20 (+ P)
    small_aas = ['C', 'T', 'V', 'A', 'G'] # Relative size 0.15-0 (- P)

    # Define the type of mutation based on whether the amino acid changes its polarity or size group
    def define_mutation_type(wt_aa,mut_aa):
        # Define type of polarity change
        for polarity_group in [non_polar_aas, polar_neutral_aas, polar_acidic_aas, polar_basic_aas]:
            if wt_aa in polarity_group and mut_aa in polarity_group:
                polarity_type = 'conservative'
                break
        else:
            polarity_type = 'polar'
        # Give its own label to a change of polarity that involves a flip of charge
        if (wt_aa in polar_acidic_aas and mut_aa in polar_basic_aas) or (wt_aa in polar_basic_aas and mut_aa in
                                                                         polar_acidic_aas):
            polarity_type = 'charge'

        # Define type of size change
        for size_group in [bulky_aas, intermediate_size_aas, small_aas]:
            if wt_aa in size_group and mut_aa in size_group:
                size_type = 'conservative'
                break
        else:
            size_type = 'size'

        # Make the final mutation type tag based both on polarity and size characteristics
        mutation_type = '_'.join(sorted(list(set([polarity_type, size_type]))))
        mutation_type = mutation_type.replace('_conservative', '').replace('conservative_', '')

        return mutation_type

    # Map mutation type to amino acid change in target_id
    def map_mutation_type_target_id(mutation):
        if mutation == '-': # WT or undefined mutation
            mutation_type = 'NA'
        else:
            wt_aa = mutation[0]
            mut_aa = mutation[-1]
            mutation_type = define_mutation_type(wt_aa, mut_aa)
        return mutation_type

    data['mutation_type'] = data['aa_change'].apply(map_mutation_type_target_id)

    return data

def plot_heatmap_aa_change(data: pd.DataFrame, output_dir: str, counts: str = 'activity',subset_col: str = None,
                           subset_value: str = None, keep_only_single: bool = False):
    """
    Plot in heatmap all unique amino acid change occurrences (number of activity datapoints or number of variants).

    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :param output_dir: directory to save the plot
    :param counts: what property to count in the heatmap. Options are 'activity' (number of activity datapoints),
        'variant' (number of variants), 'blosum' (BLOSUM62 matrix), or 'Epstein' (Epstein coefficient of difference).
    :param subset_col: column in data to subset by
    :param subset_value: value in subset_col to subset by
    :param keep_only_single: whether to keep only single mutants in the dataset. If keep_only_single is
    True, only single mutants are kept. If keep_only_single is False and multiple mutants are present,
    they are expanded to all possible single mutants. In that case, the counted property of choice is counted
    for each single mutant in the multiple mutant.
    """
    # Annotate bioactivity data with aa change and mutation type
    data_aa_change = map_mutation_type(map_aa_change(data, direction=True, keep_only_single=keep_only_single))

    # Make a subset
    if subset_col is not None:
        data_aa_change = data_aa_change[data_aa_change[subset_col] == subset_value]
        subset_flag = f'_{subset_col}-{subset_value}'
    else:
        subset_flag = ''

    if keep_only_single:
        subset_flag += '_single_mutants'
    else:
        subset_flag += '_all_mutants'

    # Keep only one datapoint per variant if plotting the number of variants instead of the number of datapoints
    if counts == 'variant':
        data_aa_change = data_aa_change.drop_duplicates(subset='target_id', keep='first')

    # Drop WT or undefined mutation instances
    data_aa_change = data_aa_change[data_aa_change['aa_change'] != '-']
    # Filter out silent mutations
    data_aa_change = data_aa_change[data_aa_change['aa_change'].apply(lambda x: x[0] != x[1])]

    # Read Epstein and BLOSUM matrix
    distance_dict = read_mutation_distance_Epstein()
    blosum_dict = read_blosum()

    # Calculate statistics to plot in heatmap
    stats = data_aa_change.groupby(['aa_change', 'mutation_type'])['connectivity'].count().reset_index()
    stats['distance_matrix'] = stats['aa_change'].apply(lambda x: distance_dict[x])
    stats['BLOSUM'] = stats['aa_change'].apply(lambda x: blosum_dict[x])

    # Define WT and mut amino acids for plotting
    stats['wt_aa'] = stats['aa_change'].apply(lambda x: x[0])
    stats['mut_aa'] = stats['aa_change'].apply(lambda x: x[1])

    # Pivot statistics to plot in heatmap
    if (counts == 'activity') or (counts == 'variant'):
        stats_heatmap = stats.pivot(columns='mut_aa',index='wt_aa',values='connectivity')
        if counts == 'activity':
            cbar_label = 'Number of bioactivity datapoints'
            # Report which are the most represented mutations
            top10_aa_change = stats.sort_values('connectivity', ascending=False).head(10)['aa_change'].tolist()
            top10_aa_change_mutations = data_aa_change[data_aa_change['aa_change'].isin(top10_aa_change)].groupby([
                'target_id', 'aa_change'])['connectivity'].count().reset_index()
            top10_aa_change_mutations['aa_change_total'] = top10_aa_change_mutations['aa_change'].map(
                stats.set_index('aa_change')['connectivity'])
            top10_aa_change_mutations['aa_change_fraction'] = top10_aa_change_mutations['connectivity'] / \
                                                                top10_aa_change_mutations['aa_change_total']
            top10_aa_change_mutations = top10_aa_change_mutations.sort_values(['aa_change_total','aa_change_fraction'],
                                                                              axis=0, ascending=False)

            top10_aa_change_mutations.to_csv(os.path.join(output_dir, f'top10_aa_change_mutations_ac'
                                                                      f'tivity{subset_flag}.csv'),sep='\t',index=False)
        elif counts == 'variant':
            cbar_label = 'Number of variants'
    elif counts == 'blosum':
        stats_heatmap = stats.pivot(columns='mut_aa',index='wt_aa',values='BLOSUM')
        cbar_label = 'BLOSUM62 score'
    elif counts == 'Epstein':
        stats_heatmap = stats.pivot(columns='mut_aa',index='wt_aa',values='distance_matrix')
        cbar_label = 'Epstein coefficient of difference'

    # Write dataframe with heatmap values to file
    stats_heatmap.to_csv(os.path.join(output_dir, f'heatmap_{counts}{subset_flag}.csv'),sep='\t')

    # Plot heatmap
    sns.set_style('white')
    sns.set_context('paper', font_scale=1.3)
    plt.figure(1,figsize=(5.5,5))
    ax = sns.heatmap(data=stats_heatmap, annot=False, cmap='rocket_r', cbar_kws={'label': cbar_label})
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    plt.yticks(rotation=0)

    # Save heatmap
    plt.savefig(os.path.join(output_dir,f'heatmap_{counts}{subset_flag}.png'),dpi=300)
    plt.close()

def stats_stacked_bars_mutation_type(data: pd.DataFrame, output_dir: str, direction: bool = True, counts: str ='activity',
                                    color: str = 'mutation_type', subset_col: str = None, subset_value: str = None,
                                    aa_change_labels: bool = True, keep_only_single: bool = False):
    """
    Plot in stacked bars all unique amino acid change occurrences (number of activity datapoints or number of
    variants) in each mutation type category.
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :param output_dir: directory to save the plot
    :param direction: whether to take into account the direction of the mutation or just the change itself
    :param counts: what property to count in the x axis. Options are 'activity' (number of activity datapoints) and
    'variant' (number of variants).
    :param color: what property to use to color. Options are 'mutation_type' (all aa changes with the same mutation
    type have the same color in different shades) and 'distance_matrix' (each aa change has a color depending on the
    distance matrix coefficient, either Epstein if direction=True ot Grantham is direction=False)
    :param subset_col: column to make a subset for
    :param subset_value: value of the column in subset_col to filter on
    :param aa_change_labels: whether to include aa change labels on top of each stacked bar layer.
    :param keep_only_single: whether to keep only single mutants in the dataset. If keep_only_single is
    True, only single mutants are kept. If keep_only_single is False and multiple mutants are present,
    they are expanded to all possible single mutants. In that case, the counted property of choice is counted
    for each single mutant in the multiple mutant.
    :return: figure
    """
    # Annotate bioactivity data with aa change and mutation type
    data_aa_change = map_mutation_type(map_aa_change(data, direction, keep_only_single=keep_only_single))

    # Make a subset
    if subset_col is not None:
        data_aa_change = data_aa_change[data_aa_change[subset_col] == subset_value]
        subset_flag = f'_{subset_col}-{subset_value}'
    else:
        subset_flag = ''

    if keep_only_single:
        subset_flag += '_single_mutants'
    else:
        subset_flag += '_all_mutants'

    # Keep only one datapoint per variant if plotting the number of variants instead of the number of datapoints
    if counts == 'variant':
        data_aa_change = data_aa_change.drop_duplicates(subset='target_id', keep='first')

    # Calculate the number of counts (activity datapoints/variants) for each type of aa change
    stats = data_aa_change.groupby(['aa_change', 'mutation_type']).count()
    plot_df = stats.loc[:, "connectivity"].reset_index()
    # Remove WT and undefined mutation instances (aa change == '-')
    plot_df = plot_df[plot_df['aa_change'] != '-']
    # Filter out silent mutations
    plot_df = plot_df[plot_df['aa_change'].apply(lambda x: x[0] != x[1])]

    # Order mutation types based on their total number of datapoints
    stats_mutation_type = data_aa_change.groupby(['mutation_type']).count()['connectivity'].\
        reset_index().sort_values(by='connectivity', axis=0, ascending=False)
    mutation_types = ['conservative', 'polar', 'size', 'charge', 'polar_size', 'charge_size']
    mutation_types_order = [x for x in stats_mutation_type['mutation_type'].tolist() if x != 'NA']
    mutation_types_order.extend([x for x in mutation_types if x not in mutation_types_order])
    mutation_types_order.reverse()

    # Read distance matrices to order amino acid changes in stacked bars by more to less disruptive
    if direction:
        # Use Epstein matrix, as it is directional
        distance_dict = read_mutation_distance_Epstein()
    else:
        # Use Grantham matrix, which is not directional
        distance_dict = read_mutation_distance_Grantham()

    plot_df['distance'] = plot_df['aa_change'].map(distance_dict)

    # Count number of datapoints with distance >0.4
    plot_df['distance_0.4'] = plot_df['distance'].apply(lambda x: 1 if x >= 0.4 else 0)
    plot_df['distance_0.4'] = plot_df['distance_0.4'] * plot_df['connectivity']


    return plot_df

def plot_stacked_bars_mutation_type(data: pd.DataFrame, output_dir: str, direction: bool = True, counts: str = 'activity',
                                    color: str = 'mutation_type', subset_col: str = None, subset_value: str = None,
                                    aa_change_labels: bool = True, keep_only_single: bool = False):
    """
    Plot in stacked bars all unique amino acid change occurrences (number of activity datapoints or number of
    variants) in each mutation type category.
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :param output_dir: directory to save the plot
    :param direction: whether to take into account the direction of the mutation or just the change itself
    :param counts: what property to count in the x axis. Options are 'activity' (number of activity datapoints) and
    'variant' (number of variants).
    :param color: what property to use to color. Options are 'mutation_type' (all aa changes with the same mutation
    type have the same color in different shades) and 'distance_matrix' (each aa change has a color depending on the
    distance matrix coefficient, either Epstein if direction=True ot Grantham is direction=False)
    :param subset_col: column to make a subset for
    :param subset_value: value of the column in subset_col to filter on
    :param aa_change_labels: whether to include aa change labels on top of each stacked bar layer.
    :param keep_only_single: whether to keep only single mutants in the dataset. If keep_only_single is
    True, only single mutants are kept. If keep_only_single is False and multiple mutants are present,
    they are expanded to all possible single mutants. In that case, the counted property of choice is counted
    for each single mutant in the multiple mutant.
    :return: figure
    """
    # Annotate bioactivity data with aa change and mutation type
    data_aa_change = map_mutation_type(map_aa_change(data, direction, keep_only_single=keep_only_single))

    # Make a subset
    if subset_col is not None:
        data_aa_change = data_aa_change[data_aa_change[subset_col] == subset_value]
        subset_flag = f'_{subset_col}-{subset_value}'
    else:
        subset_flag = ''

    if keep_only_single:
        subset_flag += '_single_mutants'
    else:
        subset_flag += '_all_mutants'

    # Keep only one datapoint per variant if plotting the number of variants instead of the number of datapoints
    if counts == 'variant':
        data_aa_change = data_aa_change.drop_duplicates(subset='target_id', keep='first')

    # Calculate the number of counts (activity datapoints/variants) for each type of aa change
    stats = data_aa_change.groupby(['aa_change', 'mutation_type']).count()
    plot_df = stats.loc[:, "connectivity"].reset_index()
    # Remove WT and undefined mutation instances (aa change == '-')
    plot_df = plot_df[plot_df['aa_change'] != '-']
    # Filter out silent mutations
    plot_df = plot_df[plot_df['aa_change'].apply(lambda x: x[0] != x[1])]

    # Order mutation types based on their total number of datapoints
    stats_mutation_type = data_aa_change.groupby(['mutation_type']).count()['connectivity'].\
        reset_index().sort_values(by='connectivity', axis=0, ascending=False)
    mutation_types = ['conservative', 'polar', 'size', 'charge', 'polar_size', 'charge_size']
    mutation_types_order = [x for x in stats_mutation_type['mutation_type'].tolist() if x != 'NA']
    mutation_types_order.extend([x for x in mutation_types if x not in mutation_types_order])
    mutation_types_order.reverse()

    # Read distance matrices to order amino acid changes in stacked bars by more to less disruptive
    if direction:
        # Use Epstein matrix, as it is directional
        distance_dict = read_mutation_distance_Epstein()
    else:
        # Use Grantham matrix, which is not directional
        distance_dict = read_mutation_distance_Grantham()

    plot_df['distance'] = plot_df['aa_change'].map(distance_dict)

    # Make dictionaries with number of datapoints and types of aa change for each mutation type
    bar_values = {}
    bar_labels = {}
    for mutation_type in mutation_types_order:
        # Sort mutation stacked bars by aa_change (alphabetical order)
        # mutation_type_df = plot_df[plot_df['mutation_type'] == mutation_type].sort_values(by='aa_change', axis=0)
        # Sort mutation types by how disruptive the aa_change is (distance matrix)
        mutation_type_df = plot_df[plot_df['mutation_type'] == mutation_type].sort_values(by='distance',
                                                                                           ascending=False,axis=0)
        aa_changes = mutation_type_df['aa_change'].tolist()
        aa_change_counts = mutation_type_df['connectivity'].tolist()
        bar_values[mutation_type] = aa_change_counts
        bar_labels[mutation_type] = aa_changes

    # Define colors
    if color == 'mutation_type':
        # Define colors for mutation types, and set in the right order according to how disruptive the type is
        palette_dict = {k: v for k, v in
                        zip(mutation_types, sns.color_palette("rocket_r", n_colors=len(mutation_types)))}
        colors_order = [palette_dict[k] for k in mutation_types_order]

    elif color == 'distance_matrix':
        # Define colors for each aa change in order of minimum to maximum distance
        distance_color_dict = {k: c for (k, v), c in zip(sorted(distance_dict.items(), key=lambda item: item[1]),
                                                         sns.color_palette("rocket_r", n_colors=len(
                                                             distance_dict.items())))}

    # Plot stacked bars from dictionaries defined above
    sns.set_style('white')
    sns.set_context('talk')
    fig, ax = plt.subplots()

    # Each layer of stacked bars is plotted at a time
    for i in range(0, len(max(list(bar_values.values()), key=len)), 1):
        # Number of datapoints of each amino acid change (bar width)
        bar_i_list = [v[i] if (len(v) > 0 and len(v) > i) else 0 for v in bar_values.values()]
        # Aggregated number of datapoints of the previous layer of stacked bars (bar left limit)
        bar_i_left = [sum(v[:i]) for v in bar_values.values()]
        # Amino acid change label positions (middle of the stacked bar layer)
        bar_i_label_pos = [(sum(v[:i+1]) + sum(v[:i]))/2 for v in bar_values.values()]
        # Amino acid change label
        bar_i_labels = [v[i] if (len(v) > 0 and len(v) > i) else '' for v in bar_labels.values()]


        if color == 'mutation_type':
            # Each mutation type has the same color, the alpha changes to differentiate between aa changes
            plt.barh([x.replace('_',' ').capitalize() for x in mutation_types_order], bar_i_list, left=bar_i_left,
            alpha=1/(1+i*0.1),color=colors_order, edgecolor='grey')

        elif color == 'distance_matrix':
            # Color of each aa change
            bar_i_colors = [distance_color_dict[aa_change] if aa_change != '' else (0.98137749, 0.92061729, 0.86536915)
            for aa_change in bar_i_labels]
            # Each aa change has a different color given by the distance matrix
            plt.barh([x.replace('_', ' ').capitalize() for x in mutation_types_order], bar_i_list, left=bar_i_left,
                     alpha=1,color=bar_i_colors, edgecolor='grey')

        # Add aa change label on top of bar segment
        if aa_change_labels:
            for j,mut_type in enumerate(mutation_types_order):
                plt.text(bar_i_label_pos[j],j, bar_i_labels[j], ha='center', va='center', size=8, color='white', weight='bold')


    # Add color legend
    if color == 'mutation_type':
        legend = [mpatches.Patch(color=v, label=k.replace('_',' ').capitalize()) for k,v in palette_dict.items()]
        plt.legend(handles=legend, title='Substitution type (severity)', loc='center left', bbox_to_anchor=(1, 0.5))


    # Make plot prettier
    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Switch off ticks
    plt.tick_params(left=False,bottom=False)

    # Draw vertical axis lines
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Set x-axis label
    if counts =='activity':
        x_label = 'Number of bioactivity datapoints'
    elif counts == 'variant':
        x_label = 'Number of variants'
    ax.set_xlabel(x_label, labelpad=10, weight='bold', size=14)

    # Set y-axis label
    ax.set_ylabel("Substitution type", labelpad=10, weight='bold', size=14)

    # Set title
    plt.suptitle(f'Substitution types present in ChEMBL + Papyrus bioactivity datasets', fontsize=16)
    if subset_col is not None:
        if subset_col == 'accession':
            # map accession to gene name
            result, failed = mapper(
                ids=[subset_value], from_db="UniProtKB_AC-ID", to_db='Gene_Name')
            plt.title(f'Subset {subset_col}:{subset_value} ({result["to"].tolist()[0]})', fontsize=12)
        else:
            plt.title(f'Subset {subset_col}:{subset_value}', fontsize=12)

    # Save plot
    if direction:
        direction_flag = 'Dir'
    else:
        direction_flag = 'NoDir'

    out_file = f'mutation_type_stacked{subset_flag}_{direction_flag}_Counts{counts}_Color{color}_Labels{aa_change_labels}.svg'
    plt.savefig(os.path.join(output_dir, out_file))

    # Create the colorbar (this is an independent plot)
    if color == 'distance_matrix':
        # Create new figure for the colormap
        fig, ax = plt.subplots(figsize=(8, 1))
        fig.subplots_adjust(bottom=0.5)
        # Make colorbar
        cmap = sns.color_palette('rocket_r', as_cmap=True)
        norm = mpl.colors.Normalize(vmin=min(list(distance_dict.values())), vmax=max(list(distance_dict.values())))
        # Plot colorbar
        cb = mpl.colorbar.ColorbarBase(norm=norm, cmap=cmap, orientation="horizontal", ax=ax)
        # Remove the outline of the colorbar
        cb.outline.set_visible(False)
        # Set legend label and move it to the top (instead of default bottom)
        if direction:
            cb.set_label("Epstein coefficient of difference", size=10, labelpad=10)
        else:
            cb.set_label("Grantham's distance", size=10, labelpad=10)

        # Save plot
        out_file = f'mutation_type_stacked{subset_flag}_{direction_flag}_Counts{counts}_Color{color}_Labels' \
                   f'{aa_change_labels}_ColorMap.svg'
        plt.savefig(os.path.join(output_dir, out_file))

def extract_residue_number_list(target_id_list: list):
    """
    From a list of unique target_id, extract the residue number of the mutation for each target_id
    Results from the extraction are stored in a list of lists, where each sublist contains the residue numbers of the
    mutations for each target_id. WT and undefined mutations are stored as 'WT' and 'MUTANT' respectively.
    :param target_id_list: list of unique target_id
    :return: list of lists with residue numbers of mutations for each target_id
    """
    mutants_resn = []
    for target_id in target_id_list:
        if 'WT' in target_id:
            mutants_resn.append('WT')
        elif 'MUTANT' in target_id:  # Undefined mutant
            mutants_resn.append('MUTANT')
        else:
            substitutions = target_id.split('_')[1:]
            mutants_resn.append([int(substitution[1:-1]) for substitution in substitutions])

    return mutants_resn


def plot_bubble_aachange_distance(data: pd.DataFrame, accession_list: list, subset_alias: str, dist_dir: str,
                                  output_dir: str, direction: bool = True, ignore_no_structure: bool = False,
                                  keep_only_single: bool = True, sum_multiple_mutants: bool = False):
    """
    Plot bubble plot for all mutants of targets of interest (accession), showing aa distance matrix value (X axis) vs.
    distance from mutated residue to ligand centroid (Y axis). Color represents mutation type and bubble size number of
    bioactivity.
    :param data: dataframe with bioactivity data and mutations reflected in target_id
    :param accession_list: List of Uniprot accession codes to targets of interest
    :param subset_alias: Alias of the subset of accession codes for naming the output file
    :param dist_dir: Path to directory containing the distance dictionaries
    :param output_dir: path to directory to store output
    :param direction: whether to take into account the direction of the mutation or just the change itself
    :param ignore_no_structure: whether to ignore accession codes for which the distance to ligand centroid cannot be
    calculated due to lack of structures
    :param keep_only_single: whether to keep only single mutants in the dataset. If keep_only_single is
    True, only single mutants are kept. If keep_only_single is False and multiple mutants are present,
    they are expanded to all possible single mutants. In that case, the amount of data is counted
    for each single mutant in the multiple mutant.
    :param sum_multiple_mutants: whether to sum the number of bioactivity datapoints for each single substitution
    :return: figure
    """
    # Subset accession codes
    data = data[data['accession'].isin(accession_list)]

    if keep_only_single:
        subset_alias += '_single_mutants'
    else:
        subset_alias += '_all_mutants'
        if sum_multiple_mutants:
            subset_alias += '_sum'


    # Extract gene names for later mapping
    gene_dict = dict(zip(data['accession'],data['HGNC_symbol'].fillna('NaN')))
    # Fill gene ID from Uniprot API if not available in dataset
    accession_list_no_gene = [k for k,v in gene_dict.items() if v == 'NaN']
    if len(accession_list_no_gene) > 0:
        result, failed = mapper(ids=accession_list_no_gene, from_db="UniProtKB_AC-ID", to_db='Gene_Name')
        gene_dict_fix = dict(zip(result['from'],result['to']))
        for acc in accession_list_no_gene:
            if acc not in gene_dict_fix.keys():
                gene_dict_fix[acc] = 'NaN'

        gene_dict = {k:v1 if v1 != 'NaN' else gene_dict_fix[k] for k,v1 in gene_dict.items()}


    # Annotate bioactivity data with aa change and mutation type
    data_aa_change = map_mutation_type(map_aa_change(data, direction, keep_only_single=keep_only_single))

    # Drop WT and undefined mutation instances
    data_aa_change = data_aa_change[data_aa_change['aa_change'] != '-']
    # Filter out silent mutations
    data_aa_change = data_aa_change[data_aa_change['aa_change'].apply(lambda x: x[0] != x[1])]
    # Calculate the number of counts (activity datapoints/variants) for each type of aa change
    stats = data_aa_change.groupby(['target_id', 'aa_change', 'single_substitution', 'mutation_type']).count()
    plot_df = stats.loc[:, "connectivity"].reset_index()

    # Annotate amino acid change distance matrix
    if direction:
        # Use Epstein matrix, as it is directional
        distance_dict = read_mutation_distance_Epstein()
    else:
        # Use Grantham matrix, which is not directional
        distance_dict = read_mutation_distance_Grantham()
    plot_df['distance_matrix'] = plot_df['aa_change'].apply(lambda x: distance_dict[x])

    # Calculate distance to ligand from mutated residues
    distances_dict = {}
    accession_list_clean = []
    for accession in accession_list:
        try:
            # Extract all possible single mutations for the target
            # target_id_list = [target_id for target_id in plot_df['target_id'].tolist() if accession in target_id]
            # mutants_resn_old = extract_residue_number_list(target_id_list)
            unique_mutations = unique_single_mutations(plot_df, accession)
            mutants_resn = [int(mutation[1:-1]) for mutation in unique_mutations]
        except ValueError:
            mutants_resn = []
        distances_dict_accession = calculate_average_residue_distance_to_ligand(accession=accession,
                                                                      resn=mutants_resn,
                                                                      common=False,
                                                                      pdb_dir=os.path.join(dist_dir, 'PDB'),
                                                                      output_dir=dist_dir)

        # Drop targets with no distance dict
        if ignore_no_structure and len(distances_dict_accession.values()) == 0:
            plot_df = plot_df[~plot_df['target_id'].str.contains(accession)]
        else:
            distances_dict[accession] = distances_dict_accession
            accession_list_clean.append(accession)

    # Map distances to mutants
    def map_distance_to_mutant(row):
        target_id = row['target_id']
        if 'WT' in target_id:
            return 0
        elif 'MUTANT' in target_id: # Undefined mutant
            return 0
        else:
            try:
                single_mutation = row['single_substitution']
                single_mutation_residue = single_mutation[1:-1]
                return distances_dict[target_id.split('_')[0]][single_mutation_residue]
            except KeyError:
                return 0

    plot_df['mutant_dist'] = plot_df.apply(map_distance_to_mutant, axis=1)
    if keep_only_single or (not sum_multiple_mutants):
        print(plot_df.sort_values(by='connectivity', ascending=False))
    else:
        if sum_multiple_mutants:
            # sum connectivity for each single substitution. Keep list of different target_ids
            plot_df = plot_df.groupby(['single_substitution','aa_change', 'mutation_type', 'distance_matrix',
                                       'mutant_dist']).agg({'connectivity':'sum', 'target_id': lambda x: list(x)}).reset_index()
            print(plot_df.sort_values(by='connectivity', ascending=False))

    # Define colors for mutation types, map to color property
    mutation_types = ['conservative', 'polar', 'size', 'charge', 'polar_size', 'charge_size']
    palette_dict = {k: v for k, v in
                    zip(mutation_types, sns.color_palette("rocket_r", n_colors=len(mutation_types)))}
    plot_df['mutation_type_color'] = plot_df['mutation_type'].apply(lambda x: palette_dict[x])

    # Plot bubble plot
    sns.set_style('white')
    sns.set_context('talk', font_scale=1)
    # fig, ax = plt.subplots(figsize=(6.4, 5))
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)



    scatter = plt.scatter(
        x=plot_df['distance_matrix'],
        y=plot_df['mutant_dist'],
        s=plot_df['connectivity'],
        c=plot_df['mutation_type_color'],
        label=plot_df['mutation_type'],
        cmap="Accent",
        alpha=0.6,
        edgecolors="white",
        linewidth=2)

    # Add X ticks at specific locations
    loc = plticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)

    # Add titles (main and on axis)
    if direction:
        x_label = "Epstein coefficient of difference"
    else:
        x_label = "Grantham's distance"
    plt.xlabel(x_label, labelpad=10, weight='bold', size=14)

    plt.ylabel("Average distance of substituted\nresidue to ligand centroid ($\AA$)",
               labelpad=10, weight='bold', size=14)
    # map accession list to gene names
    if len(accession_list_clean) < 10:
        plt.title(f"{', '.join(accession_list_clean)}\n"
                  f"({', '.join([gene_dict[accession] for accession in accession_list_clean])})")
    else:
        plt.title(subset_alias)

    # Add legends for color and size
    handles = [mpl.lines.Line2D([0], [0], marker='o', alpha=0.6, linewidth=0, color=v,markeredgecolor='white',
                                label=k.replace('_',' ').capitalize(),markersize=7) for k,v in palette_dict.items()]
    legend1 = ax.legend(handles=handles,
                        title='Substitution type', loc='lower left',
                        bbox_to_anchor=(1,0.45),fontsize="12")
    ax.add_artist(legend1)

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=6, color='silver', markeredgewidth=0.0)
    legend2 = ax.legend(handles, labels, loc="upper left", title="Data", bbox_to_anchor=(1.1, 0.45), fontsize="12")

    # Add limits
    plt.ylim(0, 30)
    if direction:
        plt.xlim(0, 1.1)
    else:
        plt.xlim(0, 216)

    plt.tight_layout()

    # Save plot
    if direction:
        direction_flag = 'Dir'
    else:
        direction_flag = 'NoDir'

    out_file = f'mutation_type_bubble_{subset_alias}_{direction_flag}.svg'
    plt.savefig(os.path.join(output_dir, out_file))