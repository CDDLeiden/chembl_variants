from utils import *
from mutants_in_pcm import data_path

# Define ChEMBL and Papyrus versions
chembl_version = '31'
papyrus_version = '05.5'
papyrus_flavor = 'nostereo'

# Define data path
data_path.data_dir = '../data'

###### ROUND 1 (Fully automatic annotation) #####
# Define annotation round
annotation_round = 1

# Define directories for this annotation round
annotation_dir = get_annotation_analysis_path('0_annotation_analysis', annotation_round)

# ChEMBL automatic annotation and combination with non-ChEMBL Papyrus mutants
from mutants_in_pcm.annotation import chembl_annotation
print('Annotating ChEMBL mutants...')
chembl_annotation(chembl_version, annotation_round)
print('Done.')

# Extract discrepancies between automatic and ChEMBL annotations
from mutants_in_pcm.annotation_check import export_positive_annotations,classify_negative_annotations
print('Extracting discrepancies between automatic and ChEMBL annotations...')
export_positive_annotations(chembl_version, annotation_round, annotation_dir)
classify_negative_annotations(chembl_version, annotation_round)
print('Done.')

# Manually annotate false positives from discrepancies (false negatives are annotated automatically)
from mutants_in_pcm.annotation_check import print_manual_annotation_instructions,check_manual_positive_annotations,read_manual_positive_annotations
if read_manual_positive_annotations(chembl_version, annotation_round):
    pass
else:
    print_manual_annotation_instructions(chembl_version, annotation_round, annotation_dir)
    print("Press any key to continue when the annotation is complete...")
    input()

check_manual_positive_annotations(chembl_version, annotation_round)

###### ROUND 2 (Semi-automatic annotation) #####
# Define annotation round
annotation_round = 2

# Define directories for this annotation round
annotation_dir = get_annotation_analysis_path('0_annotation_analysis', annotation_round)
family_analysis_dir = get_mutant_analysis_path('1_mutant_statistics', 'family', annotation_round)
accession_analysis_dir = get_mutant_analysis_path('1_mutant_statistics', 'accession', annotation_round)
type_analysis_dir = get_mutant_analysis_path('1_mutant_statistics', 'type', annotation_round)
common_analysis_dir = get_mutant_analysis_path('1_mutant_statistics', 'common', annotation_round)
compound_analysis_dir = get_mutant_analysis_path('1_mutant_statistics', 'compound', annotation_round)
bioactivity_analysis_dir = get_mutant_analysis_path('1_mutant_statistics', 'bioactivity', annotation_round)
distance_dir = get_distance_path('1_mutant_statistics')

# ChEMBL reannotation and combination with non-ChEMBL Papyrus mutants
from mutants_in_pcm.annotation import manual_reannotation
from mutants_in_pcm.preprocessing import merge_chembl_papyrus_mutants

print('Reannotating ChEMBL mutants...')
manual_reannotation(chembl_version=chembl_version,
                    annotation_round=annotation_round,
                    correct_false_positives=True,
                    correct_false_negatives=True)
print('Done.')

print('Annotating and merging ChEMBL and Papyrus mutants...')
annotated_data = merge_chembl_papyrus_mutants(chembl_version, papyrus_version, papyrus_flavor, 1_000_000,
                                              annotation_round, predefined_variants=False)
print('Done.')

# Compute statistics per protein (accession) and variant (target_id)
from mutants_in_pcm.mutant_analysis_accession import get_statistics_across_accessions,get_statistics_across_variants
print('Computing statistics per protein and variant...')
stats_protein = get_statistics_across_accessions(chembl_version, papyrus_version, papyrus_flavor, 1_000_000,
                                           annotation_round,
                                         accession_analysis_dir, save=True)
stats_variant = get_statistics_across_variants(chembl_version, papyrus_version, papyrus_flavor, 1_000_000,
                                               annotation_round, accession_analysis_dir,save=True)
print('Done.')

# Compute bioactivity distributions for common subsets for all proteins
from mutants_in_pcm.mutant_analysis_common_subsets import compute_variant_activity_distribution, extract_relevant_targets
print('Computing bioactivity distributions for common subsets...')
for accession in stats_protein['accession'].tolist():
    # Full dataset
    compute_variant_activity_distribution(annotated_data, accession, common=False, sim=False, sim_thres=None,
                                          threshold=None, variant_coverage=None, plot=True, hist=False, plot_mean=True,
                                          color_palette=None, save_dataset=True,
                                          output_dir=common_analysis_dir)
    # Strict common subset with > 20% coverage
    compute_variant_activity_distribution(annotated_data, accession, common=True, sim=False, sim_thres=None,
                                          threshold=2, variant_coverage=0.2, plot=True, hist=False, plot_mean=True,
                                          color_palette=None, save_dataset=True,
                                          output_dir=common_analysis_dir)
    # Common subset with > 20% coverage including similar compounds (>80% Tanimoto) tested in other variants
    compute_variant_activity_distribution(annotated_data, accession, common=True, sim=True, sim_thres=0.8,
                                          threshold=2, variant_coverage=0.2, plot=True, hist=False, plot_mean=True,
                                          color_palette=None, save_dataset=True,
                                          output_dir=common_analysis_dir)
    # Strictly common subset (threshold=None)
    compute_variant_activity_distribution(annotated_data, accession, common=True, sim=False, sim_thres=None,
                                          threshold=None, variant_coverage=None, plot=True, hist=False, plot_mean=True,
                                          color_palette=None, save_dataset=True,
                                          output_dir=common_analysis_dir)
print('Done.')

# Compute bioactivity distributions for compound clusters for proteins with the biggest common subsets
from mutants_in_pcm.mutant_analysis_compounds import plot_bioactivity_distribution_cluster_subset
from mutants_in_pcm.mutant_analysis_common_subsets import plot_bubble_bioactivity_distribution_stats
# Extract relevant targets of interest (biggest common subsets)
print('Extracting targets with big common subsets...')
accession_large_common_subsets = extract_relevant_targets(common_analysis_dir,
                         common=True, sim=True, sim_thres=0.8, threshold=2,variant_coverage=0.2,
                         min_subset_n= 90, thres_error_mean=0, error_mean_limit='min')['accession'].unique().tolist()
print('Done.')
print('Computing bioactivity distributions for compound clusters...')
for accession in accession_large_common_subsets:
    # Compute clusters and plot bioactivity distributions for each cluster
    plot_bioactivity_distribution_cluster_subset(accession, annotation_round, compound_analysis_dir)
    # Plot summary bubbleplot for top 10 clusters
    plot_bubble_bioactivity_distribution_stats(compound_analysis_dir, 'butina_clusters_dual', accession, 'mean_error',
                                               bioactivity_analysis_dir, True)
print('Done.')

# Compute bioactivity modeling for all proteins



