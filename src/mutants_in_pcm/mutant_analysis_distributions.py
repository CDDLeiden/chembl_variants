import os
import json
import pandas as pd
from scipy.stats import wasserstein_distance
from .mutant_analysis_common_subsets import read_common_subset,get_filename_tag,read_bioactivity_distribution_stats_file,calculate_accession_common_dataset_stats

def compute_variant_wasserstein_distance(accession: str, common: bool, sim: bool, sim_thres: int,
                                threshold: int, variant_coverage: float, output_dir: str):
    """Compute the Wasserstein distance between the bioactivity distributions of the
    wildtype and all variants in the dataset.

    :param accession:Uniprot accession code
    :param common: Whether to use common subset for variants
    :param sim: Whether similar compounds are included in the definition
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold:Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :param output_dir: Location for the pre-calculated files
    """
    # Customize filename tags based on function options for subdirectories
    options_filename_tag = get_filename_tag(common, sim, sim_thres, threshold, variant_coverage)
    output_file = os.path.join(output_dir, options_filename_tag, 'wasserstein_distances',
                               f'wasserstein_distances_{accession}_{options_filename_tag}.json')

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    if not os.path.isfile(output_file):
        # Read common subset
        common_subset = read_common_subset(accession, common, sim, sim_thres, threshold, variant_coverage, output_dir)

        # Initialize dictionary for Wasserstein distances
        w_distances = {}

        # Get wiltype bioactivity distribution
        wt_bioactivity = common_subset[common_subset['target_id'] == f'{accession}_WT']['pchembl_value_Mean'].tolist()
        wt_bioactivity.sort()

        if len(wt_bioactivity) == 0:
            print(f'No wildtype bioactivity data for {accession}.')

        else:
            # Compute Wasserstein distance for each variant
            for variant in common_subset['target_id'].unique():
                if 'WT' not in variant:
                    # Get variant bioactivity distributions
                    var_bioactivity = common_subset[common_subset['target_id'] == variant]['pchembl_value_Mean'].tolist()
                    var_bioactivity.sort()
                    # Compute Wasserstein distance
                    w_distances[variant] = wasserstein_distance(wt_bioactivity, var_bioactivity)

                # Save dictionary
                with open(output_file, 'w') as f:
                    json.dump(w_distances, f)
    else:
        print(f'Wasserstein distances for {accession} already computed.')
        # read dictionary
        with open(output_file, 'r') as f:
            w_distances = json.load(f)

    return w_distances

def compute_subset_wasserstein_distance(accession: str, subset_1_dict: dict,
                                        subset_2_dict: dict, output_dir: str):
    """Compute the Wasserstein distance between bioactivity distributions of each variant
    of two subsets of compounds for a given protein.

    :param accession: Uniprot accession code
    :param subset_1_dict: Dictionary with arguments to describe the first subset.
                            Keys: common, sim, sim_thres, threshold, variant_coverage.
    :param subset_2_dict: Dictionary with arguments to describe the second subset.
                            Keys: common, sim, sim_thres, threshold, variant_coverage.
    :param output_dir: Location for the pre-calculated files
    """
    # Customize filename tags based on function options for subdirectories
    options_filename_tag_1 = get_filename_tag(**subset_1_dict)
    options_filename_tag_2 = get_filename_tag(**subset_2_dict)

    # Order subsets tags alphabetically to avoid double computation
    if options_filename_tag_1 > options_filename_tag_2:
        options_filename_tag_1, options_filename_tag_2 = options_filename_tag_2, options_filename_tag_1

    output_file = os.path.join(output_dir,'wasserstein_distances',f'wasserstein_distances_{options_filename_tag_1}_'
                                          f'vs_{options_filename_tag_2}_{accession}.json')

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    if not os.path.isfile(output_file):
        # Read common subset
        common_subset_1 = read_common_subset(accession, **subset_1_dict, output_dir=output_dir)
        common_subset_2 = read_common_subset(accession, **subset_2_dict, output_dir=output_dir)

        n_data_difference = abs(len(common_subset_1) - len(common_subset_2))

        # Initialize dictionary for Wasserstein distances
        w_distances = {}

        n_variants_difference = []

        # Compute Wasserstein distance for each variant
        for variant in common_subset_1['target_id'].unique():
            # Get variant bioactivity distributions
            var_bioactivity_1 = common_subset_1[(common_subset_1['target_id'] == variant) & (common_subset_1[
                'pchembl_value_Mean'].notnull())]['pchembl_value_Mean'].tolist()
            var_bioactivity_1.sort()
            var_bioactivity_2 = common_subset_2[(common_subset_2['target_id'] == variant) & (common_subset_2[
                                                                                                 'pchembl_value_Mean'].notnull())]['pchembl_value_Mean'].tolist()
            var_bioactivity_2.sort()

            # Compute Wasserstein distance
            if len(var_bioactivity_1) == 0 or len(var_bioactivity_2) == 0:
                print(f'{accession} variant {variant} is not in both sets.')
                n_variants_difference.append(variant)
            else:
                w_distances[variant] = wasserstein_distance(var_bioactivity_1, var_bioactivity_2)

        # Calculate additional statistics to report differences between datasets
        subset_1_stats = calculate_accession_common_dataset_stats(accession, **subset_1_dict, output_dir=output_dir)
        subset_2_stats = calculate_accession_common_dataset_stats(accession, **subset_2_dict, output_dir=output_dir)
        # Add additional statistics to dictionary
        stats_to_add = ['n_variants','n_data','n_compounds','data_mutant_ratio','sparsity','balance_score']
        for key, value in subset_1_stats.items():
            if key in stats_to_add:
                w_distances[f'{key}_set1'] = value
        for key, value in subset_2_stats.items():
            if key in stats_to_add:
                w_distances[f'{key}_set2'] = value

        # Save dictionary
        with open(output_file, 'w') as f:
            json.dump(w_distances, f)

    else:
        print(f'Wasserstein distances for {accession} already computed between common subset {options_filename_tag_1}'
              f' and common subset {options_filename_tag_2}.')
        # read dictionary
        with open(output_file, 'r') as f:
            w_distances = json.load(f)

    return w_distances

def aggregate_variant_wasserstein_distances(common: bool, sim: bool, sim_thres: int,
                                threshold: int, variant_coverage: float, output_dir: str):
    """Aggregate the Wasserstein distances between the bioactivity distributions of the
    wildtype and all variants in the dataset for all proteins.

    :param common: Whether to use common subset for variants
    :param sim: Whether similar compounds are included in the definition
    :param sim_thres: Similarity threshold (Tanimoto) if similarity is used for common subset
    :param threshold:Minimum number of variants in which a compound has been tested in order to be included in the
                    common subset
    :param variant_coverage: Minimum ratio of the common subset of compounds that have been tested on a variant in order
                            to include that variant in the output
    :param output_dir: Location for the pre-calculated files
    """
    # Customize filename tags based on function options for subdirectories
    options_filename_tag = get_filename_tag(common, sim, sim_thres, threshold, variant_coverage)

    output_file = os.path.join(output_dir, options_filename_tag,
                               f'wasserstein_distances_{options_filename_tag}_all.json')

    if not os.path.isfile(output_file):
        # Initialize dictionary for Wasserstein distances
        w_distances_all = {}

        # Read statistics file of the common subset to get list of accessions
        common_stat = read_bioactivity_distribution_stats_file(output_dir, common, sim, sim_thres, threshold,
                                                               variant_coverage)
        accession_list = common_stat['accession'].unique().tolist()

        # Read or compute all wasserstein distances for each protein
        for accession in accession_list:
            print(f'Computing Wasserstein distances for {accession}...')
            w_distances = compute_variant_wasserstein_distance(accession, common, sim, sim_thres, threshold,
                                                               variant_coverage, output_dir)
            w_distances_all[accession] = w_distances

        # Save dictionary
        with open(output_file, 'w') as f:
            json.dump(w_distances_all, f)

    else:
        print(f'Wasserstein distances for all proteins already computed with common subset {options_filename_tag}.')
        # read dictionary
        with open(output_file, 'r') as f:
            w_distances_all = json.load(f)

    return w_distances_all

def aggregate_subset_wasserstein_distances(subset_1_dict: dict,
                                subset_2_dict: dict, output_dir: str):
    """Compute the Wasserstein distance between bioactivity distributions of each variant
    of two subsets of compounds for all proteins.

    :param subset_1_dict: Dictionary with arguments to describe the first subset.
                            Keys: common, sim, sim_thres, threshold, variant_coverage.
    :param subset_2_dict: Dictionary with arguments to describe the second subset.
                            Keys: common, sim, sim_thres, threshold, variant_coverage.
    :param output_dir: Location for the pre-calculated files
    """
    # Customize filename tags based on function options for subdirectories
    options_filename_tag_1 = get_filename_tag(**subset_1_dict)
    options_filename_tag_2 = get_filename_tag(**subset_2_dict)

    # Order subsets tags alphabetically to avoid double computation
    if options_filename_tag_1 > options_filename_tag_2:
        options_filename_tag_1, options_filename_tag_2 = options_filename_tag_2, options_filename_tag_1

    output_file = os.path.join(output_dir,f'wasserstein_distances_{options_filename_tag_1}_'
                                          f'vs_{options_filename_tag_2}_all.json')

    if not os.path.isfile(output_file):
        # Initialize dictionary for Wasserstein distances
        w_distances_all = {}

        # Read statistics file of the common subset to get list of accessions
        accession_list_1 = read_bioactivity_distribution_stats_file(output_dir, **subset_1_dict)['accession'].unique().tolist()
        accession_list_2 = read_bioactivity_distribution_stats_file(output_dir, **subset_2_dict)['accession'].unique().tolist()
        # Keep shared accessions
        accession_list = list(set(accession_list_1).intersection(accession_list_2))

        # Read or compute Wasserstein distance for each variant
        for accession in accession_list:
            print(f'Computing Wasserstein distances for {accession}...')

            w_distances = compute_subset_wasserstein_distance(accession, subset_1_dict, subset_2_dict, output_dir)
            w_distances_all[accession] = w_distances

        # Save dictionary
        with open(output_file, 'w') as f:
            json.dump(w_distances_all, f)

    else:
        print(f'Wasserstein distances for all proteins already computed between common subset {options_filename_tag_1}'
              f' and common subset {options_filename_tag_2}.')
        # read dictionary
        with open(output_file, 'r') as f:
            w_distances_all = json.load(f)

    return w_distances_all

def calculate_wasserstein_variant_average(w_distances: dict):
    """Calculate the average Wasserstein distance between the bioactivity distributions of the
    wildtype and all variants in the dataset for a given protein.

    :param w_distances: Dictionary with the Wasserstein distances for each variant
    """
    # Initialize list of distances
    distances = []

    # Compute average distance
    for variant, distance in w_distances.items():
        if ('WT' not in variant) and (variant not in ['n_variants_set1','n_data_set1',
                                                      'n_compounds_set1','data_mutant_ratio_set1',
                                                      'sparsity_set1','balance_score_set1','n_variants_set2',
                                                      'n_data_set2','n_compounds_set2',
                                                      'data_mutant_ratio_set2','sparsity_set2',
                                                      'balance_score_set2']):
            distances.append(distance)
    try:
        average = sum(distances)/len(distances)
    except ZeroDivisionError:
        average = None

    return average

def wasserstein_distance_statistics(common: bool, sim: bool, sim_thres: int,
                                threshold: int, variant_coverage: float, output_dir: str):
    """Calculate the average Wasserstein distance between the bioactivity distributions of the
    wildtype and all variants in the dataset for all proteins.

    :param w_distances_all: Dictionary with the Wasserstein distances for each variant for all proteins
    """
    # Customize filename tags based on function options for subdirectories
    options_filename_tag = get_filename_tag(common, sim, sim_thres, threshold, variant_coverage)

    output_file = os.path.join(output_dir, options_filename_tag,
                               f'wasserstein_distances_{options_filename_tag}_all.csv')

    # Read or compute all wasserstein distances for all proteins
    w_distances_all = aggregate_variant_wasserstein_distances(common, sim, sim_thres, threshold,
                                                               variant_coverage, output_dir)

    # Initialize dictionary for statistics
    w_distances_stats = {}

    # Compute average distance
    for accession, w_distances in w_distances_all.items():
        w_distances_stats[accession] = calculate_wasserstein_variant_average(w_distances)

    # Make dataframe from dictionary
    w_distances_stats_df = pd.DataFrame.from_dict(w_distances_stats, orient='index', columns=[
        'wasserstein_distance_avg'])

    # Save dataframe
    w_distances_stats_df.to_csv(output_file, sep='\t', index=True)

    return w_distances_stats_df



def wasserstein_distance_dual_statistics(subset_1_dict: dict,
                                subset_2_dict: dict, output_dir: str):
    """Report the average Wasserstein distance between bioactivity distributions of each variant
    of two subsets of compounds for all proteins. Report WT separately. Report number of variants missing in one of the
    subsets.
    """
    # Customize filename tags based on function options for subdirectories
    options_filename_tag_1 = get_filename_tag(**subset_1_dict)
    options_filename_tag_2 = get_filename_tag(**subset_2_dict)

    # Order subsets tags alphabetically to avoid double computation
    if options_filename_tag_1 > options_filename_tag_2:
        options_filename_tag_1, options_filename_tag_2 = options_filename_tag_2, options_filename_tag_1

    # output_file = os.path.join(output_dir, f'wasserstein_distances_{options_filename_tag_1}_'
    #                                        f'vs_{options_filename_tag_2}_all.csv')
    output_file = os.path.join(output_dir, f'wasserstein_distances_{options_filename_tag_1}_'
                                           f'vs_{options_filename_tag_2}_all.xlsx')

    if not os.path.isfile(output_file):
        # Read or compute all wasserstein distances for all proteins
        w_distances_all = aggregate_subset_wasserstein_distances(subset_1_dict, subset_2_dict, output_dir)

        # Initialize dictionary for statistics
        w_distances_stats = {}

        # Compute average distance
        for accession, w_distances in w_distances_all.items():
            w_distances_stats[accession] = {}
            # Report WT separately
            try:
                w_distances_stats[accession]['WT'] = w_distances[f'{accession}_WT']
            except KeyError:
                w_distances_stats[accession]['WT'] = None
            w_distances_stats[accession]['mutants_avg'] = calculate_wasserstein_variant_average(w_distances)
            # Report the rest of the statistics
            order_keys = ['n_variants_set1','n_variants_set2',
                           'n_data_set1','n_data_set2',
                            'n_compounds_set1','n_compounds_set2',
                           'data_mutant_ratio_set1','data_mutant_ratio_set2',
                           'sparsity_set1','sparsity_set2',
                           'balance_score_set1','balance_score_set2']
            for key in order_keys:
                w_distances_stats[accession][key] = w_distances[key]

        # Make dataframe from dictionary
        w_distances_stats_df = pd.DataFrame.from_dict(w_distances_stats, orient='index').sort_values(
            by='n_data_set2', ascending=False)
        print(w_distances_stats_df)

        # Save dataframe
        # w_distances_stats_df.to_csv(output_file, sep='\t', index=True)
        w_distances_stats_df.to_excel(output_file, index=True)

    else:
        print(f'Wasserstein distances for all proteins already computed between common subset {options_filename_tag_1}'
              f' and common subset {options_filename_tag_2}.')
        # read dataframe
        # w_distances_stats_df = pd.read_csv(output_file, sep='\t', index_col=0)
        w_distances_stats_df = pd.read_excel(output_file, index_col=0)

    return w_distances_stats_df




