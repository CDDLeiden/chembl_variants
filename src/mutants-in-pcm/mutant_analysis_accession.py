# -*- coding: utf-8 -*-


"""Mutant statistics analysis. Part 2"""
"""Analyzing mutant data per accession"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import papyrus_scripts
import numpy as np
import re

from preprocessing import merge_chembl_papyrus_mutants
# from mutant_analysis_clustermaps import read_common_subset,extract_unique_connectivity

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
    sns.move_legend(g.ax_joint, "upper left", title='Mutants', frameon=False, bbox_to_anchor=(1.25, 1))

    g.ax_joint.set_ylabel('pChEMBL value (Mean)')

    # Save plot
    plt.savefig(os.path.join(output_dir, f'scatterplot_year_pchembl_{accession}_{subset_name}.svg'))

## Add here code for following plots:
# distance vs. type of mutant
# type of mutant vs. bioactivity
# distance vs. bioactivity
# year vs. bioactivity

if __name__ == '__main__':
    output_dir = 'C:\\Users\\gorostiolam\\Documents\\Gorostiola Gonzalez, ' \
                        'Marina\\PROJECTS\\6_Mutants_PCM\\DATA\\2_Analysis\\0_mutant_statistics\\2_target_stats'

    # Define an accession to analyze
    accession = 'P00533' # (EGFR)

    # Read annotated bioactivity data for the accession of interest
    accession_data = filter_accession_data(merge_chembl_papyrus_mutants('31', '05.5', 'nostereo', 1_000_000), accession)

    # Read Papyrus data only, for comparison
    papyrus_accession_data = filter_explore_activity_data('05.5', [accession])

    # Define subsets for analysis
    # All molecules that have been tested in the target of interest
    accession_subset = accession_data['connectivity'].unique().tolist()
    papyrus_accession_subset = papyrus_accession_data['connectivity'].unique().tolist()


    # Plot scatterplots of bioactivity data respect to the year of testing
    double_density_pchembl_year(accession_data, accession_subset, output_dir, accession, 'annotated_data')

    double_density_pchembl_year(papyrus_accession_data, papyrus_accession_subset, output_dir, accession,
                                'papyrus_data')

