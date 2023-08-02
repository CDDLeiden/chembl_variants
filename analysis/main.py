from utils import *

# Define ChEMBL and Papyrus versions
chembl_version = '31'
papyrus_version = '05.5'
papyrus_flavor = 'nostereo'

# Define annotation round
annotation_round = 1

# Define directories for this annotation round
directories_file = './directories.json'
annotation_dir = get_annotation_analysis_path(directories_file, annotation_round)
family_analysis_dir = get_mutant_analysis_path(directories_file, 'family', annotation_round)
accession_analysis_dir = get_mutant_analysis_path(directories_file, 'accession', annotation_round)
type_analysis_dir = get_mutant_analysis_path(directories_file, 'type', annotation_round)
common_analysis_dir = get_mutant_analysis_path(directories_file, 'common', annotation_round)
compound_analysis_dir = get_mutant_analysis_path(directories_file, 'compound', annotation_round)
bioactivity_analysis_dir = get_mutant_analysis_path(directories_file, 'bioactivity', annotation_round)
distance_dir = get_distance_path(directories_file)

# Import project path to allow for relative imports
import project_path

########################################################################################################################
# Step 1: ChEMBL annotation and combination with non-ChEMBL Papyrus mutants
from preprocessing import merge_chembl_papyrus_mutants
print('Annotating and merging ChEMBL and Papyrus mutants...')
annotated_data = merge_chembl_papyrus_mutants(chembl_version, papyrus_version, papyrus_flavor, 1_000_000,
                                              annotation_round, predefined_variants=False)
print('Done.')
########################################################################################################################
# Step 2: Compute statistics per protein (accession) and variant (target_id)
from mutant_analysis_accession import get_statistics_across_accessions,get_statistics_across_variants
print('Computing statistics per protein and variant...')
stats_protein = get_statistics_across_accessions(chembl_version, papyrus_version, papyrus_flavor, 1_000_000,
                                           annotation_round,
                                         accession_analysis_dir, save=True)
stats_variant = get_statistics_across_variants(chembl_version, papyrus_version, papyrus_flavor, 1_000_000,
                                               annotation_round, accession_analysis_dir,save=True)
print('Done.')
########################################################################################################################
# Step 3: Compute bioactivity distributions for common subsets for all proteins
from mutant_analysis_common_subsets import compute_variant_activity_distribution, extract_relevant_targets
print('Computing bioactivity distributions for common subsets...')
for accession in stats_protein['accession'].tolist():
    # Full dataset
    compute_variant_activity_distribution(annotated_data, accession, common=False, sim=False, sim_thres=None,
                                          threshold=None, variant_coverage=None, plot=True, hist=False, plot_mean=True,
                                          color_palette=None, save_dataset=False,
                                          output_dir=common_analysis_dir)
    # Strict common subset with > 20% coverage
    compute_variant_activity_distribution(annotated_data, accession, common=True, sim=False, sim_thres=None,
                                          threshold=2, variant_coverage=0.2, plot=True, hist=False, plot_mean=True,
                                          color_palette=None, save_dataset=False,
                                          output_dir=common_analysis_dir)
    # Common subset with > 20% coverage including similar compounds (>80% Tanimoto) tested in other variants
    compute_variant_activity_distribution(annotated_data, accession, common=True, sim=True, sim_thres=0.8,
                                          threshold=2, variant_coverage=0.2, plot=True, hist=False, plot_mean=True,
                                          color_palette=None, save_dataset=False,
                                          output_dir=common_analysis_dir)
    # Strictly common subset (threshold=None)
    compute_variant_activity_distribution(annotated_data, accession, common=True, sim=False, sim_thres=None,
                                          threshold=None, variant_coverage=None, plot=True, hist=False, plot_mean=True,
                                          color_palette=None, save_dataset=False,
                                          output_dir=common_analysis_dir)
print('Done.')
# Write datasets for modelling for the relevant targets of interest (biggest common subsets)
print('Extracting targets with big common subsets...')
accession_large_common_subsets = extract_relevant_targets(common_analysis_dir,
                         common=True, sim=True, sim_thres=0.8, threshold=2,variant_coverage=0.2,
                         min_subset_n= 90, thres_error_mean=0, error_mean_limit='min')['accession'].unique().tolist()
print('Done.')
print('Writing datasets for modelling...')
for accession in accession_large_common_subsets:
    # Full dataset
    compute_variant_activity_distribution(annotated_data, accession, common=False, sim=False, sim_thres=None,
                                          threshold=None, variant_coverage=None, plot=False, hist=False, plot_mean=True,
                                          color_palette=None,save_dataset=True,
                                          output_dir=common_analysis_dir)
    # Common subset with > 20% coverage including similar compounds (>80% Tanimoto) tested in other variants
    compute_variant_activity_distribution(annotated_data, accession, common=True, sim=True, sim_thres=0.8,
                                       threshold=2,variant_coverage=0.2, plot=False, hist=False, plot_mean=True,
                                       color_palette=None,save_dataset=True,
                                          output_dir=common_analysis_dir)
print('Done.')
########################################################################################################################
# Step 4: Compute bioactivity distributions for compound clusters for proteins with the biggest common subsets
from mutant_analysis_compounds import plot_bioactivity_distribution_cluster_subset
from mutant_analysis_common_subsets import plot_bubble_bioactivity_distribution_stats
print('Computing bioactivity distributions for compound clusters...')
for accession in accession_large_common_subsets:
    # Compute clusters and plot bioactivity distributions for each cluster
    plot_bioactivity_distribution_cluster_subset(accession, annotation_round, compound_analysis_dir)
    # Plot summary bubbleplot for top 10 clusters
    plot_bubble_bioactivity_distribution_stats(compound_analysis_dir, 'butina_clusters', accession, 'mean_error',
                                               bioactivity_analysis_dir, True)
print('Done.')
########################################################################################################################



